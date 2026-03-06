"""
Illumio Service Consumers Agent
================================

A LangGraph-based agent that answers:

  "Quelles sont toutes les applications qui consomment mon service ?"

The agent:
  1. Parses the user's natural-language request to extract the application
     code (Code AP, e.g. AP12345).
  2. Deterministically builds the Elasticsearch DSL query from a template:
       - Filters on the destination app label (prefix "A_<app_code>-")
       - Filters on policy_decision = "Allowed"
       - Aggregates source app labels to identify all consumers
  3. Executes the query via the MCP ``search`` tool.
  4. Formats the result into a human-readable answer (in French).
  5. Falls back to a Kibana Dev Tools payload when direct index access is
     unavailable or the MCP tool is missing.

Module layout
-------------
- **config.yaml**                 – ``illumio_consumers_agent`` section
- **illumio_consumers_prompts.py** – LLM prompt templates
- **illumio_consumers_agent.py**  – state, nodes, graph, orchestrator  ← this file
- **illumio_consumers_graph.py**  – thin entry-point for ``langgraph dev``
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from illumio_consumers_prompts import (
    ILLUMIO_CONSUMERS_INTENT_SYSTEM_PROMPT,
    format_kibana_consumers_payload,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(config_path: str | Path | None = None) -> dict[str, Any]:
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
_ILLUMIO_CONSUMERS_CFG: dict[str, Any] = _CFG.get("illumio_consumers_agent", {})


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IllumioConsumersStage(str, Enum):
    """Tracks the current logical stage of the consumers workflow."""
    INIT            = "init"
    INTENT_PARSED   = "intent_parsed"
    QUERY_BUILT     = "query_built"
    EXECUTED        = "executed"
    ANSWERED        = "answered"
    KIBANA_FALLBACK = "kibana_fallback"
    FAILED          = "failed"


class IllumioConsumersState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Input
    user_request: str

    # Intent
    app_code:     Optional[str]   # e.g. "AP12345"
    date_range:   Optional[str]   # ES relative date-math, e.g. "now-1h", "now-7d" (None = no filter)
    intent_error: Optional[str]

    # Query
    query_json:    Optional[dict]
    index_pattern: Optional[str]

    # Execution
    search_result:   Optional[dict]
    execution_error: Optional[str]

    # Output
    answer:         Optional[str]
    kibana_payload: Optional[str]

    # Control flow
    stage: IllumioConsumersStage
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
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


def _build_consumers_query(
    app_code: str, cfg: dict, date_range: str | None = None
) -> dict:
    """
    Deterministically build the Illumio service-consumers DSL query.

    Filters on:
      - destination app label prefix matching the user's application
      - policy_decision = Allowed

    Aggregates source app labels to surface all consuming applications.

    When date_range is provided (e.g. "now-1h", "now-30d") a range filter on
    @timestamp is added to restrict results to that time window.
    """
    dst_app_field   = cfg.get("dest_app_field",        "illumio.destination.labels.app")
    src_app_field   = cfg.get("source_app_field",      "illumio.source.labels.app")
    policy_field    = cfg.get("policy_decision_field", "policy_decision")
    allowed_value   = cfg.get("allowed_value",         "Allowed")
    agg_size        = cfg.get("agg_size",              20)
    timestamp_field = cfg.get("timestamp_field",       "@timestamp")
    prefix_tpl      = cfg.get("app_prefix_format",     "A_{app_code}-")
    app_prefix      = prefix_tpl.format(app_code=app_code)

    filters: list[dict] = [
        {"prefix": {dst_app_field: app_prefix}},
        {"term":   {policy_field:  allowed_value}},
    ]
    if date_range:
        filters.append({"range": {timestamp_field: {"gte": date_range, "lte": "now"}}})

    return {
        "size": 0,
        "query": {"bool": {"filter": filters}},
        "aggs": {
            "top_consumers": {
                "terms": {
                    "field": src_app_field,
                    "size":  agg_size,
                }
            }
        },
    }


def _format_consumers_answer(
    search_result: dict, app_code: str, date_range: str | None = None
) -> str:
    """Produce the human-readable French answer from an Elasticsearch response."""
    hits      = search_result.get("hits", {})
    total     = hits.get("total", {})
    total_val = total.get("value", 0) if isinstance(total, dict) else int(total or 0)

    period_label = f" (période : {date_range} → now)" if date_range else ""

    if total_val == 0:
        return (
            f"Aucun flux autorisé à destination du service {app_code} "
            f"n'a été détecté{period_label}. Aucune application consommatrice identifiée."
        )

    aggs    = search_result.get("aggregations", {})
    buckets = aggs.get("top_consumers", {}).get("buckets", [])

    lines = [
        f"Les applications suivantes consomment le service {app_code}{period_label} :",
        "",
        f"Nombre total de flux autorisés détectés : {total_val}",
        "",
        f"Applications consommatrices identifiées ({len(buckets)}) :",
    ]
    for b in buckets:
        key   = b.get("key", "?")
        count = b.get("doc_count", 0)
        lines.append(f"  • {key}  ({count} flux)")

    if not buckets:
        lines.append("  (aucune application source identifiée dans les agrégations)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Node: Parse user intent
# ─────────────────────────────────────────────────────────────────────────────

async def parse_intent_node(
    state: IllumioConsumersState, chatmodel: ChatOpenAI
) -> IllumioConsumersState:
    """Use the LLM to extract app_code from the user request."""
    logger.info("Node: parse_intent (consumers)")
    user_request = state.get("user_request", "")

    if not user_request.strip():
        return {
            **state,
            "app_code":     None,
            "intent_error": "Requête vide.",
            "stage": IllumioConsumersStage.FAILED,
            "error": "Empty user request.",
        }

    messages = [
        SystemMessage(content=ILLUMIO_CONSUMERS_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_request),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    intent = _extract_json(response.content)

    if intent is None:
        return {
            **state,
            "app_code":     None,
            "date_range":   None,
            "intent_error": "Impossible d'analyser l'intention depuis la réponse du modèle.",
            "stage": IllumioConsumersStage.FAILED,
            "error": "Intent parsing failed – LLM returned non-JSON output.",
        }

    app_code   = intent.get("app_code")
    date_range = intent.get("date_range")  # None means no time filter

    if not app_code:
        msg = (
            "Aucun code application (Code AP) n'a été identifié dans votre demande. "
            "Veuillez préciser le code AP de votre service (ex. : AP12345)."
        )
        return {
            **state,
            "app_code":     None,
            "date_range":   date_range,
            "intent_error": msg,
            "stage": IllumioConsumersStage.FAILED,
            "error": None,
        }

    return {
        **state,
        "app_code":     app_code,
        "date_range":   date_range,
        "intent_error": None,
        "stage": IllumioConsumersStage.INTENT_PARSED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Build DSL query (deterministic – no LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def build_query_node(
    state: IllumioConsumersState, cfg: dict
) -> IllumioConsumersState:
    """Deterministically build the service-consumers DSL query from a template."""
    logger.info("Node: build_query (consumers)")
    app_code      = state.get("app_code", "")
    date_range    = state.get("date_range")
    index_pattern = cfg.get("illumio_index_pattern", "your-index-*")

    query = _build_consumers_query(app_code, cfg, date_range)

    return {
        **state,
        "query_json":    query,
        "index_pattern": index_pattern,
        "stage": IllumioConsumersStage.QUERY_BUILT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Execute search via MCP
# ─────────────────────────────────────────────────────────────────────────────

async def execute_search_node(
    state: IllumioConsumersState, mcp_client: Any
) -> IllumioConsumersState:
    """Execute the consumers query via the MCP ``search`` tool."""
    logger.info("Node: execute_search (consumers)")
    query         = state.get("query_json")
    index_pattern = state.get("index_pattern", "your-index-*")

    if not query:
        return {
            **state,
            "search_result":   None,
            "execution_error": "No query to execute.",
            "stage": IllumioConsumersStage.FAILED,
            "error": "No query to execute.",
        }

    try:
        tools = await mcp_client.get_tools()
        search_tool = next((t for t in tools if t.name == "search"), None)

        if search_tool is None:
            logger.warning("MCP tool 'search' not found – will fall back to Kibana payload")
            return {
                **state,
                "search_result":   None,
                "execution_error": "MCP tool 'search' not found.",
                "stage": IllumioConsumersStage.EXECUTED,
            }

        result = await search_tool.ainvoke({
            "index": index_pattern,
            "body":  query,
        })
        parsed = _parse_mcp_result(result)

        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            error_str = err.get("reason", str(err)) if isinstance(err, dict) else str(err)
            return {
                **state,
                "search_result":   parsed,
                "execution_error": f"Elasticsearch error: {error_str}",
                "stage": IllumioConsumersStage.EXECUTED,
            }

        return {
            **state,
            "search_result":   parsed,
            "execution_error": None,
            "stage": IllumioConsumersStage.EXECUTED,
        }

    except Exception as exc:
        logger.exception("Search execution failed")
        return {
            **state,
            "search_result":   None,
            "execution_error": f"Search exception: {exc}",
            "stage": IllumioConsumersStage.EXECUTED,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Format answer (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

async def format_answer_node(state: IllumioConsumersState) -> IllumioConsumersState:
    """Build the human-readable French answer from the Elasticsearch response."""
    logger.info("Node: format_answer (consumers)")
    answer = _format_consumers_answer(
        search_result=state.get("search_result", {}),
        app_code=state.get("app_code", ""),
        date_range=state.get("date_range"),
    )
    return {
        **state,
        "answer": answer,
        "stage":  IllumioConsumersStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

async def kibana_fallback_node(state: IllumioConsumersState) -> IllumioConsumersState:
    """Generate a Kibana Dev Tools payload when direct ES access is unavailable."""
    logger.info("Node: kibana_fallback (consumers)")
    query         = state.get("query_json", {})
    index_pattern = state.get("index_pattern", "your-index-*")
    payload       = format_kibana_consumers_payload(index_pattern, query or {})
    return {
        **state,
        "kibana_payload": payload,
        "stage": IllumioConsumersStage.KIBANA_FALLBACK,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_intent(state: IllumioConsumersState) -> str:
    if state.get("stage") == IllumioConsumersStage.FAILED:
        return "fail"
    return "build_query"


def _route_after_execution(state: IllumioConsumersState) -> str:
    if state.get("execution_error") or state.get("search_result") is None:
        return "kibana"
    return "answer"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_consumers_graph(
    chatmodel: ChatOpenAI,
    mcp_client: Any,
    cfg: dict | None = None,
) -> Any:
    """Construct and compile the Illumio service-consumers StateGraph."""
    if cfg is None:
        cfg = _ILLUMIO_CONSUMERS_CFG

    async def _parse_intent(s: IllumioConsumersState) -> IllumioConsumersState:
        return await parse_intent_node(s, chatmodel)

    async def _build_query(s: IllumioConsumersState) -> IllumioConsumersState:
        return await build_query_node(s, cfg)

    async def _execute_search(s: IllumioConsumersState) -> IllumioConsumersState:
        return await execute_search_node(s, mcp_client)

    graph = StateGraph(IllumioConsumersState)

    # ── Nodes ──
    graph.add_node("parse_intent",    _parse_intent)
    graph.add_node("build_query",     _build_query)
    graph.add_node("execute_search",  _execute_search)
    graph.add_node("format_answer",   format_answer_node)
    graph.add_node("kibana_fallback", kibana_fallback_node)

    # ── Entry ──
    graph.set_entry_point("parse_intent")

    # ── Edges ──
    graph.add_conditional_edges(
        "parse_intent",
        _route_after_intent,
        {
            "build_query": "build_query",
            "fail":        END,
        },
    )

    graph.add_edge("build_query", "execute_search")

    graph.add_conditional_edges(
        "execute_search",
        _route_after_execution,
        {
            "answer": "format_answer",
            "kibana": "kibana_fallback",
        },
    )

    graph.add_edge("format_answer",   END)
    graph.add_edge("kibana_fallback", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IllumioConsumersResult:
    """Structured output from an Illumio service-consumers agent run."""

    app_code:       str | None
    date_range:     str | None
    query:          dict | None
    index:          str | None
    search_result:  dict | None
    answer:         str | None
    kibana_payload: str | None
    stage:          IllumioConsumersStage
    mode:           Literal["answered", "kibana_fallback", "failed"]
    errors:         list[str]

    @classmethod
    def from_state(cls, state: IllumioConsumersState) -> "IllumioConsumersResult":
        all_errors: list[str] = []
        for key in ("error", "intent_error", "execution_error"):
            val = state.get(key)
            if val:
                all_errors.append(val)

        stage = state.get("stage", IllumioConsumersStage.FAILED)
        if stage == IllumioConsumersStage.ANSWERED:
            mode = "answered"
        elif stage == IllumioConsumersStage.KIBANA_FALLBACK:
            mode = "kibana_fallback"
        else:
            mode = "failed"

        return cls(
            app_code=state.get("app_code"),
            date_range=state.get("date_range"),
            query=state.get("query_json"),
            index=state.get("index_pattern"),
            search_result=state.get("search_result"),
            answer=state.get("answer"),
            kibana_payload=state.get("kibana_payload"),
            stage=stage,
            mode=mode,
            errors=all_errors,
        )

    def summary(self) -> str:
        lines = [
            "Illumio Service Consumers Agent Result",
            f"  Mode:       {self.mode}",
            f"  Stage:      {self.stage.value}",
            f"  App code:   {self.app_code or '(none)'}",
            f"  Date range: {self.date_range or '(all time)'}",
            f"  Index:      {self.index or '(none)'}",
        ]

        if self.mode == "answered" and self.answer:
            lines.append("")
            lines.append("  Answer:")
            for line in self.answer.split("\n"):
                lines.append(f"    {line}")

        if self.mode == "kibana_fallback" and self.kibana_payload:
            lines.append("")
            lines.append("  Kibana Dev Tools payload:")
            lines.append("  " + "─" * 40)
            for line in self.kibana_payload.split("\n"):
                lines.append(f"  {line}")

        if self.errors:
            lines.append(f"\n  Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IllumioConsumersAgent:
    """
    High-level orchestrator for the Illumio service-consumers agent.

    Answers: "Quelles sont toutes les applications qui consomment mon service ?"

    Usage::

        agent = IllumioConsumersAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        result = await agent.run(
            "Quelles applications consomment le service AP12345 ?"
        )
        print(result.summary())
    """

    def __init__(
        self,
        mcp_client: Any,
        chatmodel: ChatOpenAI,
        cfg: dict | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.chatmodel  = chatmodel
        self.cfg        = cfg if cfg is not None else _ILLUMIO_CONSUMERS_CFG
        self.graph      = build_illumio_consumers_graph(chatmodel, mcp_client, self.cfg)

    async def run(self, request: str) -> IllumioConsumersResult:
        initial: IllumioConsumersState = {
            "user_request":    request,
            "app_code":        None,
            "date_range":      None,
            "intent_error":    None,
            "query_json":      None,
            "index_pattern":   None,
            "search_result":   None,
            "execution_error": None,
            "answer":          None,
            "kibana_payload":  None,
            "stage": IllumioConsumersStage.INIT,
            "error": None,
        }
        logger.info("Starting Illumio Service Consumers Agent for: %s", request[:100])
        final = await self.graph.ainvoke(initial)
        return IllumioConsumersResult.from_state(final)
