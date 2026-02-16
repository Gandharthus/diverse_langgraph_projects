"""
Elasticsearch Ingest Pipeline Generation & Validation Wizard
=============================================================

A LangGraph-based agent that helps users create, validate, and test
Elasticsearch ingest pipelines.

Module layout
-------------
- **config.yaml**     – tunable parameters (forbidden processors, ECS catalog, retries)
- **prompts.py**      – all LLM prompt templates
- **validators.py**   – deterministic guardrail & ECS validation logic
- **pipeline_wizard.py** (this file) – state, nodes, graph, orchestrator
- **graph.py**        – thin entry-point for ``langgraph dev``

Usage::

    from pipeline_wizard import PipelineWizard

    wizard = PipelineWizard(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
    result = await wizard.run(log_samples=["<raw log line>", ...])
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from prompts import (
    build_generation_system_prompt,
    build_guardrail_fix_prompt,
    build_simulation_fix_prompt,
    format_errors_for_prompt,
)
from validators import check_guardrails, GuardrailResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load ``config.yaml`` and return it as a plain dict.

    Resolution order for the config file:
    1. Explicit ``config_path`` argument.
    2. ``PIPELINE_WIZARD_CONFIG`` environment variable.
    3. ``config.yaml`` next to *this* Python file.
    """
    if config_path is None:
        config_path = os.getenv(
            "PIPELINE_WIZARD_CONFIG",
            Path(__file__).parent / "config.yaml",
        )
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Create one from the template or set PIPELINE_WIZARD_CONFIG."
        )

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    logger.info("Loaded config from %s", config_path)
    return cfg


# Module-level config – loaded once at import time.
_CFG = _load_config()

MAX_GUARDRAIL_RETRIES: int = _CFG.get("max_guardrail_retries", 3)
MAX_SIMULATION_RETRIES: int = _CFG.get("max_simulation_retries", 3)
FORBIDDEN_PROCESSORS: set[str] = set(_CFG.get("forbidden_processors", []))
PROCESSOR_PHASES: dict[str, list[str]] = _CFG.get("processor_phases", {})
ECS_NAMESPACES: set[str] = set(_CFG.get("ecs_namespaces", []))
ECS_FIELD_CATALOG: dict[str, dict[str, Any]] = _CFG.get("ecs_field_catalog", {})

# Pre-build the generation system prompt (it only depends on forbidden list)
GENERATION_SYSTEM_PROMPT: str = build_generation_system_prompt(FORBIDDEN_PROCESSORS)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    """Tracks the current logical stage of the workflow."""
    INIT = "init"
    SAMPLE_VALIDATED = "sample_validated"
    GENERATION_CONTEXT_BUILT = "generation_context_built"
    PIPELINE_GENERATED = "pipeline_generated"
    GUARDRAILS_CHECKED = "guardrails_checked"
    GUARDRAIL_FIX_CONTEXT_BUILT = "guardrail_fix_context_built"
    SIMULATION_CONTEXT_BUILT = "simulation_context_built"
    SIMULATED = "simulated"
    SIMULATION_FIX_CONTEXT_BUILT = "simulation_fix_context_built"
    PRESENTED = "presented"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class WizardState(TypedDict, total=False):
    """
    Minimal state passed between LangGraph nodes.

    Each node reads only what it needs and writes only what it produces –
    no conversation history dragging.
    """
    # Input
    raw_log_samples: list[str]

    # Sample validation
    validated_samples: list[dict[str, str]]
    sample_validation_error: Optional[str]

    # Generation
    generation_system_prompt: str
    _generation_user_message: str
    pipeline_json: Optional[dict]
    pipeline_raw_text: str

    # Guardrails
    guardrail_errors: list[str]
    guardrail_fix_prompt: str

    # Simulation
    simulation_result: Optional[dict]
    simulation_errors: list[str]
    simulation_fix_prompt: str

    # Retry bookkeeping
    guardrail_retry_count: int
    simulation_retry_count: int

    # User decision
    user_decision: str

    # Control flow
    stage: PipelineStage
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_json_from_text(text: str) -> dict | None:
    """
    Robustly extract a JSON object from LLM output that may contain markdown
    fences, preamble text, or trailing commentary.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First top-level brace block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def wrap_samples_for_simulate(raw_samples: list[str]) -> list[dict]:
    """Wrap raw log lines in the ``_simulate`` API envelope."""
    return [{"_source": {"message": line}} for line in raw_samples]


# ──────────────────────────────────────────────────────────────────────────────
# Node functions
# ──────────────────────────────────────────────────────────────────────────────

async def validate_sample_node(state: WizardState) -> WizardState:
    """Validate that user-provided log samples are non-empty and usable."""
    logger.info("Node: validate_sample")
    raw = state.get("raw_log_samples", [])
    errors: list[str] = []

    if not raw:
        errors.append("No log samples provided.")

    cleaned: list[str] = []
    for idx, line in enumerate(raw):
        line = line.strip()
        if not line:
            errors.append(f"Sample {idx + 1} is empty or whitespace-only.")
            continue
        if len(line) < 5:
            errors.append(
                f"Sample {idx + 1} is suspiciously short ({len(line)} chars)."
            )
            continue
        cleaned.append(line)

    if errors:
        return {
            **state,
            "validated_samples": [],
            "sample_validation_error": "; ".join(errors),
            "stage": PipelineStage.FAILED,
            "error": "; ".join(errors),
        }

    return {
        **state,
        "validated_samples": wrap_samples_for_simulate(cleaned),
        "sample_validation_error": None,
        "stage": PipelineStage.SAMPLE_VALIDATED,
    }


async def build_generation_context_node(state: WizardState) -> WizardState:
    """Assemble the system prompt + user message for generation."""
    logger.info("Node: build_generation_context")
    samples_text = "\n".join(
        doc["_source"]["message"] for doc in state["validated_samples"]
    )
    user_content = (
        f"Generate an Elasticsearch ingest pipeline for the following log samples.\n"
        f"REMEMBER: Your pipeline must handle these specific samples ONLY. "
        f"Do not generalize beyond what is shown here.\n\n{samples_text}"
    )
    return {
        **state,
        "generation_system_prompt": GENERATION_SYSTEM_PROMPT,
        "_generation_user_message": user_content,
        "stage": PipelineStage.GENERATION_CONTEXT_BUILT,
    }


async def generate_pipeline_node(
    state: WizardState, chatmodel: ChatOpenAI
) -> WizardState:
    """Call the LLM to produce the initial pipeline JSON."""
    logger.info("Node: generate_pipeline")
    messages = [
        SystemMessage(content=state["generation_system_prompt"]),
        HumanMessage(content=state.get("_generation_user_message", "")),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    raw_text = response.content

    pipeline = extract_json_from_text(raw_text)
    if pipeline is None:
        return {
            **state,
            "pipeline_json": None,
            "pipeline_raw_text": raw_text,
            "stage": PipelineStage.FAILED,
            "error": "LLM output could not be parsed as valid JSON.",
        }

    if "processors" not in pipeline:
        if isinstance(pipeline, list):
            pipeline = {"processors": pipeline}
        else:
            return {
                **state,
                "pipeline_json": pipeline,
                "pipeline_raw_text": raw_text,
                "stage": PipelineStage.FAILED,
                "error": "LLM JSON missing top-level 'processors' key.",
            }

    return {
        **state,
        "pipeline_json": pipeline,
        "pipeline_raw_text": raw_text,
        "guardrail_retry_count": 0,
        "simulation_retry_count": 0,
        "stage": PipelineStage.PIPELINE_GENERATED,
    }


async def check_guardrails_node(state: WizardState) -> WizardState:
    """Run deterministic guardrail checks (delegates to validators.py)."""
    logger.info("Node: check_guardrails")
    result: GuardrailResult = check_guardrails(
        state.get("pipeline_json"),
        forbidden_processors=FORBIDDEN_PROCESSORS,
        processor_phases=PROCESSOR_PHASES,
        ecs_namespaces=ECS_NAMESPACES,
        ecs_field_catalog=ECS_FIELD_CATALOG,
    )
    return {
        **state,
        "guardrail_errors": result.errors,
        "stage": PipelineStage.GUARDRAILS_CHECKED,
    }


async def build_guardrail_fix_context_node(state: WizardState) -> WizardState:
    """Assemble the LLM prompt for fixing guardrail violations."""
    logger.info("Node: build_guardrail_fix_context")
    prompt = build_guardrail_fix_prompt(
        pipeline_json_str=json.dumps(state.get("pipeline_json", {}), indent=2),
        errors=state.get("guardrail_errors", []),
        forbidden_processors=FORBIDDEN_PROCESSORS,
    )
    return {
        **state,
        "guardrail_fix_prompt": prompt,
        "stage": PipelineStage.GUARDRAIL_FIX_CONTEXT_BUILT,
    }


async def fix_pipeline_guardrails_node(
    state: WizardState, chatmodel: ChatOpenAI
) -> WizardState:
    """Ask the LLM to fix guardrail violations."""
    logger.info(
        "Node: fix_pipeline_guardrails (attempt %d)",
        state.get("guardrail_retry_count", 0) + 1,
    )
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=state["guardrail_fix_prompt"]),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    raw_text = response.content

    pipeline = extract_json_from_text(raw_text)
    if pipeline is None:
        return {
            **state,
            "pipeline_raw_text": raw_text,
            "guardrail_retry_count": state.get("guardrail_retry_count", 0) + 1,
            "stage": PipelineStage.PIPELINE_GENERATED,
            "guardrail_errors": ["LLM fix output not parseable as JSON."],
        }

    if "processors" not in pipeline and isinstance(pipeline, list):
        pipeline = {"processors": pipeline}

    return {
        **state,
        "pipeline_json": pipeline,
        "pipeline_raw_text": raw_text,
        "guardrail_retry_count": state.get("guardrail_retry_count", 0) + 1,
        "stage": PipelineStage.PIPELINE_GENERATED,
    }


async def simulate_pipeline_node(
    state: WizardState, mcp_client: Any
) -> WizardState:
    """Call the Elasticsearch ``_simulate`` API via the MCP tool."""
    logger.info("Node: simulate_pipeline")
    pipeline = state.get("pipeline_json")
    samples = state.get("validated_samples", [])

    if pipeline is None:
        return {
            **state,
            "simulation_result": None,
            "simulation_errors": ["No pipeline to simulate."],
            "stage": PipelineStage.SIMULATED,
        }

    try:
        tools = await mcp_client.get_tools()
        simulate_tool = next(
            (t for t in tools if t.name == "simulate_pipeline"), None
        )
        if simulate_tool is None:
            return {
                **state,
                "simulation_result": None,
                "simulation_errors": ["MCP tool 'simulate_pipeline' not found."],
                "stage": PipelineStage.SIMULATED,
            }

        result = await simulate_tool.ainvoke({"pipeline": pipeline, "docs": samples})
        
        # Handle MCP content list structure (e.g. [{"type": "text", "text": "{...}"}])
        if isinstance(result, list) and len(result) > 0:
             first = result[0]
             # Check if it's a dict-like object with text content
             if isinstance(first, dict) and first.get("type") == "text" and "text" in first:
                 try:
                     result = json.loads(first["text"])
                 except (json.JSONDecodeError, TypeError) as e:
                     with open("log.txt", "a") as f:
                         f.write(f"Failed to parse MCP text block: {e}\n")
                     pass
             # Also handle if it's an object with attributes (LangChain/Pydantic models)
             elif hasattr(first, "type") and getattr(first, "type") == "text" and hasattr(first, "text"):
                 try:
                     result = json.loads(getattr(first, "text"))
                 except (json.JSONDecodeError, TypeError) as e:
                     with open("log.txt", "a") as f:
                         f.write(f"Failed to parse MCP object block: {e}\n")
                     pass

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass
        
        # DEBUG: Log full simulation response
        with open("log.txt", "a") as f:
            f.write("\n\n--- Simulation Result ---\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n-------------------------\n")

        sim_errors = _extract_simulation_errors(result)
        
        # Fallback: if we failed to parse a dict but see critical error keywords in the raw result
        if not isinstance(result, dict):
            raw_str = str(result)
            if "parse_exception" in raw_str:
                sim_errors.append("Critical: 'parse_exception' found in raw output (JSON parsing failed).")
            elif "error" in raw_str.lower() and "root_cause" in raw_str:
                sim_errors.append("Critical: 'root_cause' error found in raw output (JSON parsing failed).")
        
        # DEBUG: Log extracted errors
        with open("log.txt", "a") as f:
            f.write(f"Extracted Errors: {sim_errors}\n")

        return {
            **state,
            "simulation_result": result,
            "simulation_errors": sim_errors,
            "stage": PipelineStage.SIMULATED,
        }

    except Exception as exc:
        logger.exception("Simulation failed")
        return {
            **state,
            "simulation_result": None,
            "simulation_errors": [f"Simulation exception: {exc}"],
            "stage": PipelineStage.SIMULATED,
        }


def _find_recursive_errors(data: Any, target: str) -> list[str]:
    """Recursively search for error objects containing specific type string."""
    found: list[str] = []
    if isinstance(data, dict):
        # Check if this dict itself is the error object (e.g. {"type": "parse_exception", ...})
        if "type" in data and isinstance(data["type"], str) and target in data["type"]:
            reason = data.get("reason", "No reason provided")
            found.append(f"Found {data['type']}: {reason}")
        
        # Recurse into values
        for value in data.values():
            found.extend(_find_recursive_errors(value, target))
            
    elif isinstance(data, list):
        for item in data:
            found.extend(_find_recursive_errors(item, target))
            
    return found


def _extract_simulation_errors(result: Any) -> list[str]:
    """Parse the simulate-pipeline response for per-doc errors."""
    errors: list[str] = []
    if not isinstance(result, dict):
        return errors
    
    # Check for top-level error (e.g. invalid pipeline definition, 400 Bad Request)
    if "error" in result:
        err = result["error"]
        if isinstance(err, dict):
            reason = err.get("reason", str(err))
            root_cause = err.get("root_cause", [])
            if root_cause and isinstance(root_cause, list):
                # Append root causes if available for more detail
                details_list = [rc.get("reason", str(rc)) for rc in root_cause if isinstance(rc, dict)]
                details = "; ".join(details_list)
                if details:
                    errors.append(f"Pipeline Error: {reason} (Details: {details})")
                else:
                    errors.append(f"Pipeline Error: {reason}")
            else:
                errors.append(f"Pipeline Error: {reason}")
        else:
            errors.append(f"Pipeline Error: {str(err)}")
        # If there's a top-level error, we usually don't have docs to check, but we can continue if needed.
    
    for idx, doc_result in enumerate(result.get("docs", [])):
        if "error" in doc_result:
            err = doc_result["error"]
            reason = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
            errors.append(f"Doc {idx}: {reason}")
        tags = doc_result.get("doc", {}).get("_source", {}).get("tags", [])
        if isinstance(tags, list):
            failures = [t for t in tags if "_failure" in str(t)]
            if failures:
                errors.append(f"Doc {idx}: failure tags: {failures}")
    
    # Deep search for parse_exception as requested
    deep_errors = _find_recursive_errors(result, "parse_exception")
    # Add unique deep errors
    for de in deep_errors:
        # Avoid exact duplicates if possible (simple check)
        if not any(de in e for e in errors):
             errors.append(de)

    return errors


async def build_simulation_fix_context_node(state: WizardState) -> WizardState:
    """Assemble the LLM prompt for fixing simulation failures."""
    logger.info("Node: build_simulation_fix_context")
    prompt = build_simulation_fix_prompt(
        pipeline_json_str=json.dumps(state.get("pipeline_json", {}), indent=2),
        errors=state.get("simulation_errors", []),
        samples_json_str=json.dumps(state.get("validated_samples", []), indent=2),
        forbidden_processors=FORBIDDEN_PROCESSORS,
    )
    return {
        **state,
        "simulation_fix_prompt": prompt,
        "stage": PipelineStage.SIMULATION_FIX_CONTEXT_BUILT,
    }


async def fix_pipeline_simulation_node(
    state: WizardState, chatmodel: ChatOpenAI
) -> WizardState:
    """Ask the LLM to fix the pipeline based on simulation errors."""
    logger.info(
        "Node: fix_pipeline_simulation (attempt %d)",
        state.get("simulation_retry_count", 0) + 1,
    )
    messages = [
        SystemMessage(content=GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=state["simulation_fix_prompt"]),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    raw_text = response.content

    pipeline = extract_json_from_text(raw_text)
    if pipeline is None:
        return {
            **state,
            "pipeline_raw_text": raw_text,
            "simulation_retry_count": state.get("simulation_retry_count", 0) + 1,
            "stage": PipelineStage.PIPELINE_GENERATED,
            "simulation_errors": ["LLM simulation-fix output not parseable as JSON."],
        }

    if "processors" not in pipeline and isinstance(pipeline, list):
        pipeline = {"processors": pipeline}

    return {
        **state,
        "pipeline_json": pipeline,
        "pipeline_raw_text": raw_text,
        "simulation_retry_count": state.get("simulation_retry_count", 0) + 1,
        "stage": PipelineStage.PIPELINE_GENERATED,
    }


async def present_results_node(state: WizardState) -> WizardState:
    """Mark pipeline as ready for user review."""
    logger.info("Node: present_results")
    logger.info(
        "Pipeline ready for review:\n%s",
        json.dumps(state.get("pipeline_json", {}), indent=2)[:2000],
    )
    return {**state, "stage": PipelineStage.PRESENTED}


async def accept_pipeline_node(state: WizardState) -> WizardState:
    """Terminal node – pipeline accepted, ready for deployment."""
    logger.info("Node: accept_pipeline – pipeline accepted.")
    return {**state, "stage": PipelineStage.ACCEPTED}


# ──────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ──────────────────────────────────────────────────────────────────────────────

def route_after_sample_validation(state: WizardState) -> str:
    if state.get("sample_validation_error"):
        return "fail"
    return "continue"


def route_after_guardrails(state: WizardState) -> str:
    errors = state.get("guardrail_errors", [])
    if not errors:
        return "pass"
    if state.get("guardrail_retry_count", 0) < MAX_GUARDRAIL_RETRIES:
        return "fix"
    return "fail"


def route_after_simulation(state: WizardState) -> str:
    """
    Route based on simulation errors.
    Simple logic: Errors exist? -> Fix it (if under retry limit).
    """
    errors = state.get("simulation_errors") or []
    retry_count = state.get("simulation_retry_count", 0)
    
    # DEBUG: Log routing decision
    with open("log.txt", "a") as f:
        f.write(f"\n--- Routing Check ---\n")
        f.write(f"Errors found: {len(errors)}\n")
        f.write(f"Retry count: {retry_count} / {MAX_SIMULATION_RETRIES}\n")

    if len(errors) > 0:
        if retry_count < MAX_SIMULATION_RETRIES:
            with open("log.txt", "a") as f:
                f.write("Decision: fix (errors > 0 and retries available)\n")
            return "fix"
        else:
            with open("log.txt", "a") as f:
                f.write("Decision: pass_with_warnings (errors > 0 but max retries exceeded)\n")
            return "pass_with_warnings"
            
    with open("log.txt", "a") as f:
        f.write("Decision: pass (no errors)\n")
    return "pass"


def route_after_user_decision(state: WizardState) -> str:
    if state.get("user_decision", "reject") == "accept":
        return "accept"
    return "reject"


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline_wizard_graph(
    chatmodel: ChatOpenAI,
    mcp_client: Any,
) -> Any:
    """Construct and compile the full LangGraph StateGraph."""

    # Closures so every node has the ``async def(state) -> state`` signature
    async def _generate(s: WizardState) -> WizardState:
        return await generate_pipeline_node(s, chatmodel)

    async def _fix_guard(s: WizardState) -> WizardState:
        return await fix_pipeline_guardrails_node(s, chatmodel)

    async def _simulate(s: WizardState) -> WizardState:
        return await simulate_pipeline_node(s, mcp_client)

    async def _fix_sim(s: WizardState) -> WizardState:
        return await fix_pipeline_simulation_node(s, chatmodel)

    graph = StateGraph(WizardState)

    # Nodes
    graph.add_node("validate_sample", validate_sample_node)
    graph.add_node("build_generation_context", build_generation_context_node)
    graph.add_node("generate_pipeline", _generate)
    graph.add_node("check_guardrails", check_guardrails_node)
    graph.add_node("build_guardrail_fix_context", build_guardrail_fix_context_node)
    graph.add_node("fix_pipeline_guardrails", _fix_guard)
    graph.add_node("simulate_pipeline", _simulate)
    graph.add_node("build_simulation_fix_context", build_simulation_fix_context_node)
    graph.add_node("fix_pipeline_simulation", _fix_sim)
    graph.add_node("present_results", present_results_node)
    graph.add_node("accept_pipeline", accept_pipeline_node)

    # Entry
    graph.set_entry_point("validate_sample")

    # Edges
    graph.add_conditional_edges(
        "validate_sample",
        route_after_sample_validation,
        {"continue": "build_generation_context", "fail": END},
    )
    graph.add_edge("build_generation_context", "generate_pipeline")
    graph.add_edge("generate_pipeline", "check_guardrails")

    graph.add_conditional_edges(
        "check_guardrails",
        route_after_guardrails,
        {"pass": "simulate_pipeline", "fix": "build_guardrail_fix_context", "fail": END},
    )
    graph.add_edge("build_guardrail_fix_context", "fix_pipeline_guardrails")
    graph.add_edge("fix_pipeline_guardrails", "check_guardrails")

    graph.add_conditional_edges(
        "simulate_pipeline",
        route_after_simulation,
        {"pass": "present_results", "fix": "build_simulation_fix_context", "pass_with_warnings": "present_results"},
    )
    graph.add_edge("build_simulation_fix_context", "fix_pipeline_simulation")
    graph.add_edge("fix_pipeline_simulation", "check_guardrails")

    graph.add_conditional_edges(
        "present_results",
        route_after_user_decision,
        {"accept": "accept_pipeline", "reject": END},
    )
    graph.add_edge("accept_pipeline", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class PipelineWizard:
    """
    High-level orchestrator.

    Usage::

        wizard = PipelineWizard(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        result = await wizard.run(log_samples=[...])
        print(result.summary())
    """

    def __init__(
        self,
        mcp_client: Any,
        chatmodel: ChatOpenAI,
        *,
        max_guardrail_retries: int | None = None,
        max_simulation_retries: int | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.chatmodel = chatmodel

        # Allow per-instance overrides (mutate module globals so routing
        # functions see the new limits).
        global MAX_GUARDRAIL_RETRIES, MAX_SIMULATION_RETRIES
        if max_guardrail_retries is not None:
            MAX_GUARDRAIL_RETRIES = max_guardrail_retries
        if max_simulation_retries is not None:
            MAX_SIMULATION_RETRIES = max_simulation_retries

        self.graph = build_pipeline_wizard_graph(chatmodel, mcp_client)

    async def run(
        self,
        log_samples: list[str],
        *,
        user_decision: str = "accept",
    ) -> WizardResult:
        initial: WizardState = {
            "raw_log_samples": log_samples,
            "_generation_user_message": "",
            "validated_samples": [],
            "sample_validation_error": None,
            "generation_system_prompt": "",
            "pipeline_json": None,
            "pipeline_raw_text": "",
            "guardrail_errors": [],
            "guardrail_fix_prompt": "",
            "simulation_result": None,
            "simulation_errors": [],
            "simulation_fix_prompt": "",
            "guardrail_retry_count": 0,
            "simulation_retry_count": 0,
            "user_decision": user_decision,
            "stage": PipelineStage.INIT,
            "error": None,
        }
        logger.info("Starting Pipeline Wizard with %d sample(s)", len(log_samples))
        final = await self.graph.ainvoke(initial)
        return WizardResult.from_state(final)


@dataclass
class WizardResult:
    """Structured output from a Pipeline Wizard run."""

    pipeline: dict | None
    accepted: bool
    stage: PipelineStage
    simulation_result: dict | None
    errors: list[str]
    warnings: list[str]
    guardrail_retries: int
    simulation_retries: int

    @classmethod
    def from_state(cls, state: WizardState) -> WizardResult:
        all_errors: list[str] = []
        if state.get("error"):
            all_errors.append(state["error"])
        all_errors.extend(state.get("guardrail_errors", []))
        all_errors.extend(state.get("simulation_errors", []))

        warnings = [e for e in all_errors if e.startswith("[ECS warning]")]
        hard = [e for e in all_errors if not e.startswith("[ECS warning]")]

        return cls(
            pipeline=state.get("pipeline_json"),
            accepted=state.get("stage") == PipelineStage.ACCEPTED,
            stage=state.get("stage", PipelineStage.FAILED),
            simulation_result=state.get("simulation_result"),
            errors=hard,
            warnings=warnings,
            guardrail_retries=state.get("guardrail_retry_count", 0),
            simulation_retries=state.get("simulation_retry_count", 0),
        )

    def summary(self) -> str:
        lines = [
            "Pipeline Wizard Result",
            f"  Stage:              {self.stage.value}",
            f"  Accepted:           {self.accepted}",
            f"  Guardrail retries:  {self.guardrail_retries}",
            f"  Simulation retries: {self.simulation_retries}",
        ]
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"    - {w}")
        if self.pipeline:
            p = json.dumps(self.pipeline, indent=2)
            if len(p) > 2000:
                p = p[:2000] + "\n    ... (truncated)"
            lines.append(f"  Pipeline:\n{p}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE_LOG_SAMPLES = [
    '2024-01-15T10:23:45.123Z INFO  [web-server] 192.168.1.100 GET /api/users 200 45ms "Mozilla/5.0"',
    '2024-01-15T10:23:46.456Z ERROR [db-handler] 10.0.0.55 POST /api/orders 500 1200ms "curl/7.88.1"',
    '2024-01-15T10:23:47.789Z WARN  [auth-svc] 172.16.0.10 PUT /api/users/123 403 12ms "Python-urllib/3.11"',
]


async def main() -> None:
    """Quick demo – uses a mock MCP client."""
    chatmodel = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
    )

    class MockMCPClient:
        async def get_tools(self):
            from types import SimpleNamespace

            async def _sim(params):
                return {
                    "docs": [
                        {"doc": {"_source": {"message": "ok", "@timestamp": "2024-01-15T10:23:45.123Z"}}}
                        for _ in params.get("docs", [])
                    ]
                }
            return [SimpleNamespace(name="simulate_pipeline", ainvoke=_sim)]

    wizard = PipelineWizard(mcp_client=MockMCPClient(), chatmodel=chatmodel)
    result = await wizard.run(log_samples=EXAMPLE_LOG_SAMPLES)
    print(result.summary())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
