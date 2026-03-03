"""
Elasticsearch Log Search Agent
================================

A LangGraph-based agent that translates natural-language search requests
into validated Elasticsearch DSL queries, executes them via MCP tools,
and returns results – or falls back to a Kibana Dev Tools payload when
the agent lacks index access.

Module layout
-------------
- **config.yaml**          – shared config (search guardrails section)
- **search_prompts.py**    – all LLM prompt templates
- **search_validators.py** – deterministic DSL guardrail checks
- **search_agent.py** (this file) – state, nodes, graph, orchestrator
- **search_graph.py**      – thin entry-point for ``langgraph dev``

Usage::

    from search_agent import SearchAgent

    agent = SearchAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
    result = await agent.run("show me 403 errors from the firewall logs in the last hour")
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

from ingest_pipes.agent.prompts.search_prompts import (
    INTENT_SYSTEM_PROMPT,
    DSL_GENERATION_SYSTEM_PROMPT,
    build_dsl_user_message,
    build_query_fix_prompt,
    format_kibana_payload,
    format_mapping_fields,
)
from ingest_pipes.agent.others.search_validators import validate_search_query, QueryValidationResult

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml – same resolution logic as pipeline_wizard."""
    if config_path is None:
        config_path = os.getenv(
            "PIPELINE_WIZARD_CONFIG",
            Path(__file__).parent / "config.yaml",
        )
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


_CFG = _load_config()
_SEARCH_CFG = _CFG.get("search_agent", {})

MAX_QUERY_FIX_RETRIES: int = _SEARCH_CFG.get("max_query_fix_retries", 3)
MAX_AGG_DEPTH: int = _SEARCH_CFG.get("max_agg_depth", 3)
MAX_BUCKETS: int = _SEARCH_CFG.get("max_buckets", 1000)
TOP_K_INDICES: int = _SEARCH_CFG.get("top_k_indices", 5)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

class SearchStage(str, Enum):
    """Tracks the current logical stage of the search workflow."""
    INIT = "init"
    INTENT_PARSED = "intent_parsed"
    INDEX_RESOLVED = "index_resolved"
    MAPPING_FETCHED = "mapping_fetched"
    QUERY_GENERATED = "query_generated"
    QUERY_VALIDATED = "query_validated"
    QUERY_FIX_CONTEXT_BUILT = "query_fix_context_built"
    QUERY_ES_VALIDATED = "query_es_validated"
    EXECUTED = "executed"
    KIBANA_FALLBACK = "kibana_fallback"
    FAILED = "failed"


class SearchState(TypedDict, total=False):
    """
    State passed between LangGraph nodes.

    Each node reads only what it needs, writes only what it produces.
    """
    # Input
    user_request: str

    # Intent parsing
    search_plan: Optional[dict]
    intent_error: Optional[str]

    # Index resolution
    resolved_indices: list[dict]
    selected_index: Optional[str]
    index_accessible: bool

    # Mapping
    index_mapping: Optional[dict]
    mapping_summary: str

    # Query generation
    query_json: Optional[dict]
    query_raw_text: str

    # Validation
    validation_errors: list[str]
    validation_warnings: list[str]
    query_fix_prompt: str

    # ES-side validation (_validate/query)
    es_validation_passed: Optional[bool]
    es_validation_error: Optional[str]

    # Execution
    search_result: Optional[dict]
    execution_error: Optional[str]

    # Fallback
    kibana_payload: Optional[str]

    # Retry bookkeeping
    query_fix_retry_count: int

    # Control flow
    stage: SearchStage
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_json_from_text(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
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


def _parse_mcp_result(result: Any) -> Any:
    """Normalize MCP tool response (list of content blocks → parsed data)."""
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict) and first.get("type") == "text" and "text" in first:
            try:
                return json.loads(first["text"])
            except (json.JSONDecodeError, TypeError):
                pass
        elif hasattr(first, "type") and getattr(first, "type") == "text" and hasattr(first, "text"):
            try:
                return json.loads(getattr(first, "text"))
            except (json.JSONDecodeError, TypeError):
                pass
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Node: Parse user intent
# ──────────────────────────────────────────────────────────────────────────────

async def parse_intent_node(
    state: SearchState, chatmodel: ChatOpenAI
) -> SearchState:
    """Use LLM to extract structured search intent from natural language."""
    logger.info("Node: parse_intent")
    user_request = state.get("user_request", "")

    if not user_request.strip():
        return {
            **state,
            "search_plan": None,
            "intent_error": "Empty search request.",
            "stage": SearchStage.FAILED,
            "error": "Empty search request.",
        }

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_request),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    plan = extract_json_from_text(response.content)

    if plan is None:
        return {
            **state,
            "search_plan": None,
            "intent_error": "Could not parse search intent from LLM output.",
            "stage": SearchStage.FAILED,
            "error": "Intent parsing failed.",
        }

    return {
        **state,
        "search_plan": plan,
        "intent_error": None,
        "stage": SearchStage.INTENT_PARSED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Resolve indices via MCP tool
# ──────────────────────────────────────────────────────────────────────────────

async def resolve_index_node(
    state: SearchState, mcp_client: Any
) -> SearchState:
    """Call the resolve_index MCP tool to find relevant indices."""
    logger.info("Node: resolve_index")
    plan = state.get("search_plan", {})
    hints = plan.get("index_hints", [])

    if not hints:
        hints = ["*"]

    try:
        tools = await mcp_client.get_tools()
        resolve_tool = next(
            (t for t in tools if t.name == "resolve_index"), None
        )
        if resolve_tool is None:
            return {
                **state,
                "resolved_indices": [],
                "selected_index": None,
                "index_accessible": False,
                "stage": SearchStage.FAILED,
                "error": "MCP tool 'resolve_index' not found.",
            }

        # Call resolve_index for each hint pattern, collect results
        all_indices: list[dict] = []
        for pattern in hints:
            result = await resolve_tool.ainvoke({
                "pattern": pattern,
                "top_k": TOP_K_INDICES,
            })
            parsed = _parse_mcp_result(result)
            if isinstance(parsed, list):
                all_indices.extend(parsed)
            elif isinstance(parsed, dict) and "indices" in parsed:
                all_indices.extend(parsed["indices"])
            elif isinstance(parsed, dict) and "error" in parsed:
                logger.warning("resolve_index error for '%s': %s", pattern, parsed["error"])

        # Deduplicate by index name
        seen = set()
        unique: list[dict] = []
        for idx in all_indices:
            name = idx.get("name", idx.get("index", ""))
            if name and name not in seen:
                seen.add(name)
                unique.append(idx)

        if not unique:
            return {
                **state,
                "resolved_indices": [],
                "selected_index": None,
                "index_accessible": False,
                "stage": SearchStage.INDEX_RESOLVED,
            }

        # Pick the first (most relevant) index
        selected = unique[0].get("name", unique[0].get("index", ""))
        return {
            **state,
            "resolved_indices": unique,
            "selected_index": selected,
            "index_accessible": True,
            "stage": SearchStage.INDEX_RESOLVED,
        }

    except Exception as exc:
        logger.exception("Index resolution failed")
        return {
            **state,
            "resolved_indices": [],
            "selected_index": None,
            "index_accessible": False,
            "stage": SearchStage.INDEX_RESOLVED,
            "error": f"Index resolution exception: {exc}",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Fetch mapping via MCP tool
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_mapping_node(
    state: SearchState, mcp_client: Any
) -> SearchState:
    """Fetch the mapping for the selected index."""
    logger.info("Node: fetch_mapping")
    index_name = state.get("selected_index")

    if not index_name:
        return {
            **state,
            "index_mapping": None,
            "mapping_summary": "(no index selected)",
            "stage": SearchStage.MAPPING_FETCHED,
        }

    try:
        tools = await mcp_client.get_tools()
        search_tool = next(
            (t for t in tools if t.name == "search"), None
        )
        # Use the search tool to call GET /<index>/_mapping
        # Since we have a 1:1 mapping with _search, we use a dedicated
        # get_mapping tool if available, otherwise fall back to search
        mapping_tool = next(
            (t for t in tools if t.name == "get_mapping"), None
        )

        if mapping_tool:
            result = await mapping_tool.ainvoke({"index": index_name})
        else:
            # Fallback: no dedicated mapping tool, we'll work with
            # what the resolve_index tool gave us or skip
            logger.warning("No 'get_mapping' MCP tool found, proceeding with limited mapping info")
            return {
                **state,
                "index_mapping": None,
                "mapping_summary": "(mapping not available – no get_mapping tool)",
                "stage": SearchStage.MAPPING_FETCHED,
            }

        parsed = _parse_mcp_result(result)

        # ES returns {index_name: {mappings: {properties: {...}}}}
        mapping = parsed
        if isinstance(parsed, dict):
            if index_name in parsed:
                mapping = parsed[index_name]
            # Some responses have the index as the top key
            for key in parsed:
                if isinstance(parsed[key], dict) and "mappings" in parsed[key]:
                    mapping = parsed[key]
                    break

        summary = format_mapping_fields(mapping)

        return {
            **state,
            "index_mapping": mapping,
            "mapping_summary": summary,
            "stage": SearchStage.MAPPING_FETCHED,
        }

    except Exception as exc:
        logger.exception("Mapping fetch failed")
        return {
            **state,
            "index_mapping": None,
            "mapping_summary": f"(mapping fetch failed: {exc})",
            "stage": SearchStage.MAPPING_FETCHED,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Generate DSL query
# ──────────────────────────────────────────────────────────────────────────────

async def generate_query_node(
    state: SearchState, chatmodel: ChatOpenAI
) -> SearchState:
    """Call the LLM to produce the DSL query."""
    logger.info("Node: generate_query")

    user_msg = build_dsl_user_message(
        user_request=state.get("user_request", ""),
        search_plan=state.get("search_plan", {}),
        index_name=state.get("selected_index", "unknown"),
        mapping_summary=state.get("mapping_summary", "(not available)"),
    )

    messages = [
        SystemMessage(content=DSL_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    raw_text = response.content

    query = extract_json_from_text(raw_text)
    if query is None:
        return {
            **state,
            "query_json": None,
            "query_raw_text": raw_text,
            "stage": SearchStage.FAILED,
            "error": "LLM output could not be parsed as valid JSON.",
        }

    return {
        **state,
        "query_json": query,
        "query_raw_text": raw_text,
        "query_fix_retry_count": 0,
        "stage": SearchStage.QUERY_GENERATED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Validate query (deterministic guardrails)
# ──────────────────────────────────────────────────────────────────────────────

async def validate_query_node(state: SearchState) -> SearchState:
    """Run deterministic guardrail checks on the DSL query."""
    logger.info("Node: validate_query")
    result: QueryValidationResult = validate_search_query(
        state.get("query_json"),
        max_agg_depth=MAX_AGG_DEPTH,
        max_buckets=MAX_BUCKETS,
    )
    return {
        **state,
        "validation_errors": result.errors,
        "validation_warnings": result.warnings,
        "stage": SearchStage.QUERY_VALIDATED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Build fix context
# ──────────────────────────────────────────────────────────────────────────────

async def build_query_fix_context_node(state: SearchState) -> SearchState:
    """Assemble the LLM prompt for fixing query validation errors."""
    logger.info("Node: build_query_fix_context")
    prompt = build_query_fix_prompt(
        query_json_str=json.dumps(state.get("query_json", {}), indent=2),
        validation_errors=state.get("validation_errors", []),
        original_request=state.get("user_request", ""),
    )
    return {
        **state,
        "query_fix_prompt": prompt,
        "stage": SearchStage.QUERY_FIX_CONTEXT_BUILT,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Fix query via LLM
# ──────────────────────────────────────────────────────────────────────────────

async def fix_query_node(
    state: SearchState, chatmodel: ChatOpenAI
) -> SearchState:
    """Ask the LLM to fix query validation errors."""
    logger.info(
        "Node: fix_query (attempt %d)",
        state.get("query_fix_retry_count", 0) + 1,
    )
    messages = [
        SystemMessage(content=DSL_GENERATION_SYSTEM_PROMPT),
        HumanMessage(content=state["query_fix_prompt"]),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    raw_text = response.content

    query = extract_json_from_text(raw_text)
    if query is None:
        return {
            **state,
            "query_raw_text": raw_text,
            "query_fix_retry_count": state.get("query_fix_retry_count", 0) + 1,
            "validation_errors": ["LLM fix output not parseable as JSON."],
            "stage": SearchStage.QUERY_GENERATED,
        }

    return {
        **state,
        "query_json": query,
        "query_raw_text": raw_text,
        "query_fix_retry_count": state.get("query_fix_retry_count", 0) + 1,
        "stage": SearchStage.QUERY_GENERATED,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node: ES-side query validation (_validate/query)
# ──────────────────────────────────────────────────────────────────────────────

async def es_validate_query_node(
    state: SearchState, mcp_client: Any
) -> SearchState:
    """Use the ES _validate/query API to check the query server-side."""
    logger.info("Node: es_validate_query")
    query = state.get("query_json")
    index = state.get("selected_index")

    if not query or not index:
        return {
            **state,
            "es_validation_passed": False,
            "es_validation_error": "No query or index to validate.",
            "stage": SearchStage.QUERY_ES_VALIDATED,
        }

    try:
        tools = await mcp_client.get_tools()
        validate_tool = next(
            (t for t in tools if t.name == "validate_query"), None
        )

        if validate_tool is None:
            # If no validate tool, skip ES validation and proceed
            logger.warning("No 'validate_query' MCP tool – skipping ES validation")
            return {
                **state,
                "es_validation_passed": True,
                "es_validation_error": None,
                "stage": SearchStage.QUERY_ES_VALIDATED,
            }

        # Extract just the query part for _validate/query
        query_body = {"query": query.get("query", {"match_all": {}})}

        result = await validate_tool.ainvoke({
            "index": index,
            "body": query_body,
        })
        parsed = _parse_mcp_result(result)

        if isinstance(parsed, dict):
            valid = parsed.get("valid", False)
            error = parsed.get("error", None)
            if not valid and error:
                error_str = error if isinstance(error, str) else json.dumps(error)
                return {
                    **state,
                    "es_validation_passed": False,
                    "es_validation_error": error_str,
                    "stage": SearchStage.QUERY_ES_VALIDATED,
                }
            return {
                **state,
                "es_validation_passed": valid,
                "es_validation_error": None,
                "stage": SearchStage.QUERY_ES_VALIDATED,
            }

        return {
            **state,
            "es_validation_passed": True,
            "es_validation_error": None,
            "stage": SearchStage.QUERY_ES_VALIDATED,
        }

    except Exception as exc:
        logger.exception("ES query validation failed")
        # Don't block execution on validation failure – treat as warning
        return {
            **state,
            "es_validation_passed": True,
            "es_validation_error": f"Validation exception (non-blocking): {exc}",
            "stage": SearchStage.QUERY_ES_VALIDATED,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Execute search via MCP
# ──────────────────────────────────────────────────────────────────────────────

async def execute_search_node(
    state: SearchState, mcp_client: Any
) -> SearchState:
    """Execute the validated DSL query via the search MCP tool."""
    logger.info("Node: execute_search")
    query = state.get("query_json")
    index = state.get("selected_index")

    if not query or not index:
        return {
            **state,
            "search_result": None,
            "execution_error": "No query or index to execute.",
            "stage": SearchStage.FAILED,
            "error": "No query or index to execute.",
        }

    try:
        tools = await mcp_client.get_tools()
        search_tool = next(
            (t for t in tools if t.name == "search"), None
        )
        if search_tool is None:
            return {
                **state,
                "search_result": None,
                "execution_error": "MCP tool 'search' not found.",
                "stage": SearchStage.FAILED,
                "error": "MCP tool 'search' not found.",
            }

        # Strip _comment before sending to ES
        query_clean = {k: v for k, v in query.items() if k != "_comment"}

        result = await search_tool.ainvoke({
            "index": index,
            "body": query_clean,
        })
        parsed = _parse_mcp_result(result)

        # Check for ES-level errors in response
        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            error_str = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
            return {
                **state,
                "search_result": parsed,
                "execution_error": f"Elasticsearch error: {error_str}",
                "stage": SearchStage.EXECUTED,
            }

        return {
            **state,
            "search_result": parsed,
            "execution_error": None,
            "stage": SearchStage.EXECUTED,
        }

    except Exception as exc:
        logger.exception("Search execution failed")
        return {
            **state,
            "search_result": None,
            "execution_error": f"Search exception: {exc}",
            "stage": SearchStage.EXECUTED,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Node: Kibana fallback (no index access)
# ──────────────────────────────────────────────────────────────────────────────

async def kibana_fallback_node(state: SearchState) -> SearchState:
    """Generate a Kibana Dev Tools payload when agent lacks index access."""
    logger.info("Node: kibana_fallback")
    query = state.get("query_json", {})
    index = state.get("selected_index") or state.get("search_plan", {}).get("index_hints", ["<index>"])[0]

    # Strip _comment for the payload
    query_clean = {k: v for k, v in query.items() if k != "_comment"} if query else {}

    payload = format_kibana_payload(index, query_clean)
    return {
        **state,
        "kibana_payload": payload,
        "stage": SearchStage.KIBANA_FALLBACK,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ──────────────────────────────────────────────────────────────────────────────

def route_after_index_resolution(state: SearchState) -> str:
    """Route based on whether we found accessible indices."""
    if state.get("error"):
        return "fail"
    if not state.get("resolved_indices"):
        # No indices found at all – can still help with DSL
        return "no_access"
    if not state.get("index_accessible", False):
        return "no_access"
    return "continue"


def route_after_validation(state: SearchState) -> str:
    """Route based on query validation results."""
    errors = state.get("validation_errors", [])
    if not errors:
        return "pass"
    if state.get("query_fix_retry_count", 0) < MAX_QUERY_FIX_RETRIES:
        return "fix"
    return "fail"


def route_after_es_validation(state: SearchState) -> str:
    """Route based on ES-side validation."""
    if state.get("es_validation_passed", True):
        return "execute"
    # ES validation failed – treat as a fix-needed scenario
    retry_count = state.get("query_fix_retry_count", 0)
    if retry_count < MAX_QUERY_FIX_RETRIES:
        return "fix"
    # Out of retries – fall back to Kibana
    return "kibana"


def route_after_execution(state: SearchState) -> str:
    """Route based on execution results."""
    if state.get("execution_error"):
        return "error"
    return "done"


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────────────────────

def build_search_agent_graph(
    chatmodel: ChatOpenAI,
    mcp_client: Any,
) -> Any:
    """Construct and compile the search agent StateGraph."""

    # Closures for node signatures
    async def _parse_intent(s: SearchState) -> SearchState:
        return await parse_intent_node(s, chatmodel)

    async def _resolve_index(s: SearchState) -> SearchState:
        return await resolve_index_node(s, mcp_client)

    async def _fetch_mapping(s: SearchState) -> SearchState:
        return await fetch_mapping_node(s, mcp_client)

    async def _generate_query(s: SearchState) -> SearchState:
        return await generate_query_node(s, chatmodel)

    async def _fix_query(s: SearchState) -> SearchState:
        return await fix_query_node(s, chatmodel)

    async def _es_validate(s: SearchState) -> SearchState:
        return await es_validate_query_node(s, mcp_client)

    async def _execute(s: SearchState) -> SearchState:
        return await execute_search_node(s, mcp_client)

    graph = StateGraph(SearchState)

    # ── Nodes ──
    graph.add_node("parse_intent", _parse_intent)
    graph.add_node("resolve_index", _resolve_index)
    graph.add_node("fetch_mapping", _fetch_mapping)
    graph.add_node("generate_query", _generate_query)
    graph.add_node("validate_query", validate_query_node)
    graph.add_node("build_query_fix_context", build_query_fix_context_node)
    graph.add_node("fix_query", _fix_query)
    graph.add_node("es_validate_query", _es_validate)
    graph.add_node("execute_search", _execute)
    graph.add_node("kibana_fallback", kibana_fallback_node)

    # ── Entry ──
    graph.set_entry_point("parse_intent")

    # ── Edges ──
    graph.add_edge("parse_intent", "resolve_index")

    graph.add_conditional_edges(
        "resolve_index",
        route_after_index_resolution,
        {
            "continue": "fetch_mapping",
            "no_access": "generate_query",  # Skip mapping, still generate DSL
            "fail": END,
        },
    )

    graph.add_edge("fetch_mapping", "generate_query")
    graph.add_edge("generate_query", "validate_query")

    graph.add_conditional_edges(
        "validate_query",
        route_after_validation,
        {
            "pass": "es_validate_query",
            "fix": "build_query_fix_context",
            "fail": "kibana_fallback",  # Out of retries → give user the payload
        },
    )

    graph.add_edge("build_query_fix_context", "fix_query")
    graph.add_edge("fix_query", "validate_query")

    graph.add_conditional_edges(
        "es_validate_query",
        route_after_es_validation,
        {
            "execute": "execute_search",
            "fix": "build_query_fix_context",
            "kibana": "kibana_fallback",
        },
    )

    graph.add_conditional_edges(
        "execute_search",
        route_after_execution,
        {
            "done": END,
            "error": "kibana_fallback",  # Execution error → give user Kibana payload
        },
    )

    graph.add_edge("kibana_fallback", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class SearchAgent:
    """
    High-level orchestrator for the search agent.

    Usage::

        agent = SearchAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        result = await agent.run("show me 403 errors from firewall in the last hour")
        print(result.summary())
    """

    def __init__(
        self,
        mcp_client: Any,
        chatmodel: ChatOpenAI,
    ) -> None:
        self.mcp_client = mcp_client
        self.chatmodel = chatmodel
        self.graph = build_search_agent_graph(chatmodel, mcp_client)

    async def run(self, request: str) -> SearchResult:
        initial: SearchState = {
            "user_request": request,
            "search_plan": None,
            "intent_error": None,
            "resolved_indices": [],
            "selected_index": None,
            "index_accessible": False,
            "index_mapping": None,
            "mapping_summary": "",
            "query_json": None,
            "query_raw_text": "",
            "validation_errors": [],
            "validation_warnings": [],
            "query_fix_prompt": "",
            "es_validation_passed": None,
            "es_validation_error": None,
            "search_result": None,
            "execution_error": None,
            "kibana_payload": None,
            "query_fix_retry_count": 0,
            "stage": SearchStage.INIT,
            "error": None,
        }
        logger.info("Starting Search Agent for: %s", request[:100])
        final = await self.graph.ainvoke(initial)
        return SearchResult.from_state(final)


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Structured output from a Search Agent run."""

    query: dict | None
    index: str | None
    search_result: dict | None
    kibana_payload: str | None
    stage: SearchStage
    errors: list[str]
    warnings: list[str]
    query_fix_retries: int
    mode: Literal["executed", "kibana_fallback", "failed"]

    @classmethod
    def from_state(cls, state: SearchState) -> SearchResult:
        all_errors: list[str] = []
        if state.get("error"):
            all_errors.append(state["error"])
        if state.get("execution_error"):
            all_errors.append(state["execution_error"])
        all_errors.extend(state.get("validation_errors", []))

        stage = state.get("stage", SearchStage.FAILED)
        if stage == SearchStage.EXECUTED and not state.get("execution_error"):
            mode = "executed"
        elif stage == SearchStage.KIBANA_FALLBACK:
            mode = "kibana_fallback"
        else:
            mode = "failed"

        return cls(
            query=state.get("query_json"),
            index=state.get("selected_index"),
            search_result=state.get("search_result"),
            kibana_payload=state.get("kibana_payload"),
            stage=stage,
            errors=all_errors,
            warnings=state.get("validation_warnings", []),
            query_fix_retries=state.get("query_fix_retry_count", 0),
            mode=mode,
        )

    def summary(self) -> str:
        lines = [
            "Search Agent Result",
            f"  Mode:              {self.mode}",
            f"  Stage:             {self.stage.value}",
            f"  Index:             {self.index or '(none)'}",
            f"  Query fix retries: {self.query_fix_retries}",
        ]

        if self.mode == "executed" and self.search_result:
            hits = self.search_result.get("hits", {})
            total = hits.get("total", {})
            if isinstance(total, dict):
                total_val = total.get("value", 0)
            else:
                total_val = total
            lines.append(f"  Total hits:        {total_val}")

            hit_list = hits.get("hits", [])
            if hit_list:
                lines.append(f"  Returned docs:     {len(hit_list)}")

            aggs = self.search_result.get("aggregations", {})
            if aggs:
                lines.append(f"  Aggregations:      {list(aggs.keys())}")

        if self.mode == "kibana_fallback" and self.kibana_payload:
            lines.append(f"\n  Kibana Dev Tools payload:\n  ────────────────────────")
            for line in self.kibana_payload.split("\n"):
                lines.append(f"  {line}")

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"    - {w}")

        if self.query:
            q = json.dumps(self.query, indent=2)
            if len(q) > 2000:
                q = q[:2000] + "\n    ... (truncated)"
            lines.append(f"  Generated query:\n{q}")

        return "\n".join(lines)
