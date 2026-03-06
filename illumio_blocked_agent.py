"""
Illumio Blocked Flows Agent
============================

A LangGraph-based agent that answers:

  "Est-ce qu'il y a eu des flux bloqués vers ou depuis mon application/serveur ?"

The agent:
  1. Parses the user's natural-language request to extract:
       - target      – hostname (e.g. "web-prod-01") or app code (e.g. "AP12345")
       - target_type – "hostname" | "app"
       - direction   – "inbound" | "outbound" | "both"
  2. Deterministically builds one or two Elasticsearch DSL queries (inbound /
     outbound) filtered on ``policy_decision = denied``.
  3. Executes the query/queries via the MCP ``search`` tool.
  4. Formats the results into a human-readable French answer.
  5. Falls back to Kibana Dev Tools payload(s) when direct ES access is
     unavailable or the MCP tool is missing.

Module layout
-------------
- **config.yaml**                – ``illumio_blocked_agent`` section
- **illumio_blocked_prompts.py** – LLM prompt templates
- **illumio_blocked_agent.py**   – state, nodes, graph, orchestrator  ← this file
- **illumio_blocked_graph.py**   – thin entry-point for ``langgraph dev``
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

from illumio_blocked_prompts import (
    ILLUMIO_BLOCKED_INTENT_SYSTEM_PROMPT,
    format_kibana_blocked_payload,
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
_ILLUMIO_BLOCKED_CFG: dict[str, Any] = _CFG.get("illumio_blocked_agent", {})


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IllumioBlockedStage(str, Enum):
    """Tracks the current logical stage of the blocked-flows workflow."""
    INIT            = "init"
    INTENT_PARSED   = "intent_parsed"
    QUERY_BUILT     = "query_built"
    EXECUTED        = "executed"
    ANSWERED        = "answered"
    KIBANA_FALLBACK = "kibana_fallback"
    FAILED          = "failed"


class IllumioBlockedState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Input
    user_request: str

    # Intent
    target:       Optional[str]   # hostname or app code
    target_type:  Optional[str]   # "hostname" | "app"
    direction:    Optional[str]   # "inbound" | "outbound" | "both"
    date_range:   Optional[str]   # ES relative date-math, e.g. "now-1h", "now-7d" (None = no filter)
    intent_error: Optional[str]

    # Queries (one or both depending on direction)
    inbound_query:  Optional[dict]
    outbound_query: Optional[dict]
    index_pattern:  Optional[str]

    # Execution results
    inbound_result:  Optional[dict]
    outbound_result: Optional[dict]
    execution_error: Optional[str]

    # Output
    answer:         Optional[str]
    kibana_payload: Optional[str]

    # Control flow
    stage: IllumioBlockedStage
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


def _unwrap_exception(exc: BaseException) -> str:
    """Recursively unwrap ExceptionGroup / anyio TaskGroup exceptions.

    Python 3.11+ and anyio wrap task failures in an ExceptionGroup whose
    ``exceptions`` attribute holds the actual sub-exceptions.  Stringifying
    the group directly only yields the opaque
    "unhandled errors in a TaskGroup (N sub-exceptions)" message, hiding the
    real cause.  This helper drills down to expose every leaf error so the
    full details land in ``execution_error`` (visible in LangSmith) without
    surfacing anything extra to the end-user.
    """
    if hasattr(exc, "exceptions") and exc.exceptions:
        parts = "; ".join(_unwrap_exception(e) for e in exc.exceptions)
        return f"{type(exc).__name__}({parts})"
    return f"{type(exc).__name__}: {exc}"


def _build_blocked_queries(
    target: str,
    target_type: str,
    direction: str,
    cfg: dict,
    date_range: str | None = None,
) -> tuple[dict | None, dict | None]:
    """
    Build inbound and/or outbound blocked-flow DSL queries.

    Returns (inbound_query, outbound_query) – either may be None depending on
    the requested direction.

    For ``target_type == "hostname"``:
      - inbound:  policy_decision=denied AND destination.hostname=<target>
      - outbound: policy_decision=denied AND source.hostname=<target>

    For ``target_type == "app"``:
      - inbound:  policy_decision=denied AND destination.labels.app prefix <app_prefix>
      - outbound: policy_decision=denied AND source.labels.app prefix <app_prefix>

    When date_range is provided (e.g. "now-1h") a range filter on @timestamp
    is added to both queries to restrict results to that time window.
    """
    policy_field    = cfg.get("policy_decision_field",  "illumio.policy_decision")
    denied_value    = cfg.get("denied_value",            "denied")
    agg_size        = cfg.get("agg_size",                20)
    timestamp_field = cfg.get("timestamp_field",         "@timestamp")

    src_host_field = cfg.get("source_hostname_field",  "illumio.source.hostname")
    dst_host_field = cfg.get("dest_hostname_field",    "illumio.destination.hostname")
    src_app_field  = cfg.get("source_app_field",       "illumio.source.labels.app")
    dst_app_field  = cfg.get("dest_app_field",         "illumio.destination.labels.app")

    dest_port_field = cfg.get("dest_port_field",  "destination.port")
    protocol_field  = cfg.get("protocol_field",   "network.protocol")

    # Build the target-specific filter clause for each direction
    if target_type == "app":
        prefix_tpl = cfg.get("app_prefix_format", "A_{app_code}-")
        app_prefix = prefix_tpl.format(app_code=target)
        inbound_target_filter  = {"prefix": {dst_app_field: app_prefix}}
        outbound_target_filter = {"prefix": {src_app_field: app_prefix}}
    else:  # hostname
        inbound_target_filter  = {"term": {dst_host_field: target}}
        outbound_target_filter = {"term": {src_host_field: target}}

    policy_filter = {"term": {policy_field: denied_value}}
    range_filter  = (
        [{"range": {timestamp_field: {"gte": date_range, "lte": "now"}}}]
        if date_range else []
    )

    def _inbound() -> dict:
        return {
            "size": 0,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "filter": [policy_filter, inbound_target_filter] + range_filter,
                }
            },
            "aggs": {
                "top_sources": {
                    "terms": {"field": src_host_field, "size": agg_size},
                },
                "top_dest_ports": {
                    "terms": {"field": dest_port_field, "size": agg_size},
                },
                "top_protocols": {
                    "terms": {"field": protocol_field, "size": 10},
                },
            },
        }

    def _outbound() -> dict:
        return {
            "size": 0,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "filter": [policy_filter, outbound_target_filter] + range_filter,
                }
            },
            "aggs": {
                "top_destinations": {
                    "terms": {"field": dst_host_field, "size": agg_size},
                },
                "top_dest_ports": {
                    "terms": {"field": dest_port_field, "size": agg_size},
                },
                "top_protocols": {
                    "terms": {"field": protocol_field, "size": 10},
                },
            },
        }

    inbound  = _inbound()  if direction in ("inbound",  "both") else None
    outbound = _outbound() if direction in ("outbound", "both") else None
    return inbound, outbound


def _total_hits(result: dict | None) -> int:
    """Extract total hit count from an Elasticsearch response."""
    if not result:
        return 0
    hits  = result.get("hits", {})
    total = hits.get("total", {})
    return total.get("value", 0) if isinstance(total, dict) else int(total or 0)


def _bucket_lines(buckets: list[dict], label: str, indent: str = "  ") -> list[str]:
    lines = [f"{label} ({len(buckets)}) :"]
    for b in buckets:
        lines.append(f"{indent}• {b.get('key', '?')}  ({b.get('doc_count', 0)} flux)")
    if not buckets:
        lines.append(f"{indent}(aucun résultat)")
    return lines


def _format_blocked_answer(
    inbound_result:  dict | None,
    outbound_result: dict | None,
    target:      str,
    target_type: str,
    direction:   str,
    date_range:  str | None = None,
) -> str:
    """Produce the human-readable French answer from Elasticsearch responses."""
    target_label  = (
        f"l'application {target}" if target_type == "app" else f"le serveur {target}"
    )
    period_label = f" (période : {date_range} → now)" if date_range else ""

    inbound_total  = _total_hits(inbound_result)
    outbound_total = _total_hits(outbound_result)
    total = inbound_total + outbound_total

    if total == 0:
        if direction == "inbound":
            return (
                f"Non, aucun flux bloqué à destination de {target_label} "
                f"n'a été détecté{period_label}."
            )
        if direction == "outbound":
            return (
                f"Non, aucun flux bloqué en provenance de {target_label} "
                f"n'a été détecté{period_label}."
            )
        return (
            f"Non, aucun flux bloqué vers ou depuis {target_label} "
            f"n'a été détecté{period_label}."
        )

    lines: list[str] = []
    if period_label:
        lines += [f"Période analysée : {period_label.strip()}", ""]

    # ── Inbound section ──
    if inbound_result is not None:
        aggs = inbound_result.get("aggregations", {})
        if inbound_total > 0:
            lines += [
                f"[BLOQUES ENTRANTS] Flux bloqués vers {target_label} : "
                f"{inbound_total} flux détectés.",
                "",
            ]
            lines += _bucket_lines(
                aggs.get("top_sources", {}).get("buckets", []),
                "Sources les plus fréquentes",
            )
            lines.append("")
            lines += _bucket_lines(
                aggs.get("top_dest_ports", {}).get("buckets", []),
                "Ports de destination les plus ciblés",
            )
            lines.append("")
            lines += _bucket_lines(
                aggs.get("top_protocols", {}).get("buckets", []),
                "Protocoles",
            )
        else:
            lines.append(f"Aucun flux bloqué vers {target_label}.")
        lines.append("")

    # ── Outbound section ──
    if outbound_result is not None:
        aggs = outbound_result.get("aggregations", {})
        if outbound_total > 0:
            lines += [
                f"[BLOQUES SORTANTS] Flux bloqués depuis {target_label} : "
                f"{outbound_total} flux détectés.",
                "",
            ]
            lines += _bucket_lines(
                aggs.get("top_destinations", {}).get("buckets", []),
                "Destinations les plus fréquentes",
            )
            lines.append("")
            lines += _bucket_lines(
                aggs.get("top_dest_ports", {}).get("buckets", []),
                "Ports de destination les plus bloqués",
            )
            lines.append("")
            lines += _bucket_lines(
                aggs.get("top_protocols", {}).get("buckets", []),
                "Protocoles",
            )
        else:
            lines.append(f"Aucun flux bloqué depuis {target_label}.")

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Node: Parse user intent
# ─────────────────────────────────────────────────────────────────────────────

async def parse_intent_node(
    state: IllumioBlockedState, chatmodel: ChatOpenAI
) -> IllumioBlockedState:
    """Use the LLM to extract target, target_type, and direction from the user request."""
    logger.info("Node: parse_intent (blocked)")
    user_request = state.get("user_request", "")

    if not user_request.strip():
        return {
            **state,
            "target":       None,
            "target_type":  None,
            "direction":    None,
            "intent_error": "Requête vide.",
            "stage": IllumioBlockedStage.FAILED,
            "error": "Empty user request.",
        }

    messages = [
        SystemMessage(content=ILLUMIO_BLOCKED_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_request),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    intent = _extract_json(response.content)

    if intent is None:
        return {
            **state,
            "target":       None,
            "target_type":  None,
            "direction":    None,
            "date_range":   None,
            "intent_error": "Impossible d'analyser l'intention depuis la réponse du modèle.",
            "stage": IllumioBlockedStage.FAILED,
            "error": "Intent parsing failed – LLM returned non-JSON output.",
        }

    target      = intent.get("target")
    target_type = intent.get("target_type", "hostname")
    direction   = intent.get("direction", "both")
    date_range  = intent.get("date_range")  # None means no time filter

    if target_type not in ("hostname", "app"):
        target_type = "hostname"
    if direction not in ("inbound", "outbound", "both"):
        direction = "both"

    if not target:
        msg = (
            "Aucun serveur ou code application n'a été identifié dans votre demande. "
            "Veuillez préciser le nom du serveur ou le code AP de l'application "
            "(ex. : web-prod-01 ou AP12345)."
        )
        return {
            **state,
            "target":       None,
            "target_type":  target_type,
            "direction":    direction,
            "date_range":   date_range,
            "intent_error": msg,
            "stage": IllumioBlockedStage.FAILED,
            "error": None,
        }

    return {
        **state,
        "target":       target,
        "target_type":  target_type,
        "direction":    direction,
        "date_range":   date_range,
        "intent_error": None,
        "stage": IllumioBlockedStage.INTENT_PARSED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Build DSL queries (deterministic – no LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def build_query_node(
    state: IllumioBlockedState, cfg: dict
) -> IllumioBlockedState:
    """Deterministically build the blocked-flow DSL queries from templates."""
    logger.info("Node: build_query (blocked)")
    target        = state.get("target", "")
    target_type   = state.get("target_type", "hostname")
    direction     = state.get("direction", "both")
    date_range    = state.get("date_range")
    index_pattern = cfg.get("illumio_index_pattern", "your-index-*")

    inbound_query, outbound_query = _build_blocked_queries(
        target, target_type, direction, cfg, date_range
    )

    return {
        **state,
        "inbound_query":  inbound_query,
        "outbound_query": outbound_query,
        "index_pattern":  index_pattern,
        "stage": IllumioBlockedStage.QUERY_BUILT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Execute search via MCP
# ─────────────────────────────────────────────────────────────────────────────

async def execute_search_node(
    state: IllumioBlockedState, mcp_client: Any
) -> IllumioBlockedState:
    """Execute one or both blocked-flow queries via the MCP ``search`` tool."""
    logger.info("Node: execute_search (blocked)")
    inbound_query  = state.get("inbound_query")
    outbound_query = state.get("outbound_query")
    index_pattern  = state.get("index_pattern", "your-index-*")

    if not inbound_query and not outbound_query:
        return {
            **state,
            "inbound_result":  None,
            "outbound_result": None,
            "execution_error": "No queries to execute.",
            "stage": IllumioBlockedStage.FAILED,
            "error": "No queries to execute.",
        }

    try:
        tools = await mcp_client.get_tools()
        search_tool = next((t for t in tools if t.name == "search"), None)

        if search_tool is None:
            logger.warning("MCP tool 'search' not found – will fall back to Kibana payload")
            return {
                **state,
                "inbound_result":  None,
                "outbound_result": None,
                "execution_error": "MCP tool 'search' not found.",
                "stage": IllumioBlockedStage.EXECUTED,
            }

        inbound_result  = None
        outbound_result = None
        errors: list[str] = []

        if inbound_query:
            try:
                raw    = await search_tool.ainvoke({"index": index_pattern, "body": inbound_query})
                parsed = _parse_mcp_result(raw)
                if isinstance(parsed, dict) and "error" in parsed:
                    err = parsed["error"]
                    errors.append(
                        f"Inbound ES error: "
                        f"{err.get('reason', str(err)) if isinstance(err, dict) else str(err)}"
                    )
                else:
                    inbound_result = parsed
            except Exception as exc:
                error_msg = _unwrap_exception(exc)
                logger.error("Inbound search failed: %s", error_msg, exc_info=True)
                if hasattr(exc, "exceptions"):
                    for i, sub in enumerate(exc.exceptions, 1):
                        logger.error("  Sub-exception %d: %s", i, _unwrap_exception(sub), exc_info=sub)
                errors.append(f"Inbound search failed: {error_msg}")

        if outbound_query:
            try:
                raw    = await search_tool.ainvoke({"index": index_pattern, "body": outbound_query})
                parsed = _parse_mcp_result(raw)
                if isinstance(parsed, dict) and "error" in parsed:
                    err = parsed["error"]
                    errors.append(
                        f"Outbound ES error: "
                        f"{err.get('reason', str(err)) if isinstance(err, dict) else str(err)}"
                    )
                else:
                    outbound_result = parsed
            except Exception as exc:
                error_msg = _unwrap_exception(exc)
                logger.error("Outbound search failed: %s", error_msg, exc_info=True)
                if hasattr(exc, "exceptions"):
                    for i, sub in enumerate(exc.exceptions, 1):
                        logger.error("  Sub-exception %d: %s", i, _unwrap_exception(sub), exc_info=sub)
                errors.append(f"Outbound search failed: {error_msg}")

        # Only propagate execution_error when *both* queries failed
        both_failed = (inbound_result is None and outbound_result is None)
        execution_error = "; ".join(errors) if (errors and both_failed) else None

        return {
            **state,
            "inbound_result":  inbound_result,
            "outbound_result": outbound_result,
            "execution_error": execution_error,
            "stage": IllumioBlockedStage.EXECUTED,
        }

    except Exception as exc:
        error_msg = _unwrap_exception(exc)
        logger.error("Search execution failed: %s", error_msg, exc_info=True)
        if hasattr(exc, "exceptions"):
            for i, sub in enumerate(exc.exceptions, 1):
                logger.error("  Sub-exception %d: %s", i, _unwrap_exception(sub), exc_info=sub)
        return {
            **state,
            "inbound_result":  None,
            "outbound_result": None,
            "execution_error": f"Search failed: {error_msg}",
            "stage": IllumioBlockedStage.EXECUTED,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Format answer (deterministic)
# ─────────────────────────────────────────────────────────────────────────────

async def format_answer_node(state: IllumioBlockedState) -> IllumioBlockedState:
    """Build the human-readable French answer from the Elasticsearch responses."""
    logger.info("Node: format_answer (blocked)")
    answer = _format_blocked_answer(
        inbound_result=state.get("inbound_result"),
        outbound_result=state.get("outbound_result"),
        target=state.get("target", ""),
        target_type=state.get("target_type", "hostname"),
        direction=state.get("direction", "both"),
        date_range=state.get("date_range"),
    )
    return {
        **state,
        "answer": answer,
        "stage":  IllumioBlockedStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

async def kibana_fallback_node(state: IllumioBlockedState) -> IllumioBlockedState:
    """Generate Kibana Dev Tools payload(s) when direct ES access is unavailable."""
    logger.info("Node: kibana_fallback (blocked)")
    payload = format_kibana_blocked_payload(
        index_pattern=state.get("index_pattern", "your-index-*"),
        inbound_query=state.get("inbound_query"),
        outbound_query=state.get("outbound_query"),
    )
    return {
        **state,
        "kibana_payload": payload,
        "stage": IllumioBlockedStage.KIBANA_FALLBACK,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_intent(state: IllumioBlockedState) -> str:
    if state.get("stage") == IllumioBlockedStage.FAILED:
        return "fail"
    return "build_query"


def _route_after_execution(state: IllumioBlockedState) -> str:
    # Kibana fallback only when *both* directions failed with no results at all
    if (
        state.get("execution_error")
        and state.get("inbound_result") is None
        and state.get("outbound_result") is None
    ):
        return "kibana"
    return "answer"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_blocked_graph(
    chatmodel: ChatOpenAI,
    mcp_client: Any,
    cfg: dict | None = None,
) -> Any:
    """Construct and compile the Illumio blocked-flows StateGraph."""
    if cfg is None:
        cfg = _ILLUMIO_BLOCKED_CFG

    async def _parse_intent(s: IllumioBlockedState) -> IllumioBlockedState:
        return await parse_intent_node(s, chatmodel)

    async def _build_query(s: IllumioBlockedState) -> IllumioBlockedState:
        return await build_query_node(s, cfg)

    async def _execute_search(s: IllumioBlockedState) -> IllumioBlockedState:
        return await execute_search_node(s, mcp_client)

    graph = StateGraph(IllumioBlockedState)

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
class IllumioBlockedResult:
    """Structured output from an Illumio blocked-flows agent run."""

    target:          str | None
    target_type:     str | None
    direction:       str | None
    date_range:      str | None
    inbound_query:   dict | None
    outbound_query:  dict | None
    index:           str | None
    inbound_result:  dict | None
    outbound_result: dict | None
    answer:          str | None
    kibana_payload:  str | None
    stage:           IllumioBlockedStage
    mode:            Literal["answered", "kibana_fallback", "failed"]
    errors:          list[str]

    @classmethod
    def from_state(cls, state: IllumioBlockedState) -> "IllumioBlockedResult":
        all_errors: list[str] = []
        for key in ("error", "intent_error", "execution_error"):
            val = state.get(key)
            if val:
                all_errors.append(val)

        stage = state.get("stage", IllumioBlockedStage.FAILED)
        if stage == IllumioBlockedStage.ANSWERED:
            mode = "answered"
        elif stage == IllumioBlockedStage.KIBANA_FALLBACK:
            mode = "kibana_fallback"
        else:
            mode = "failed"

        return cls(
            target=state.get("target"),
            target_type=state.get("target_type"),
            direction=state.get("direction"),
            date_range=state.get("date_range"),
            inbound_query=state.get("inbound_query"),
            outbound_query=state.get("outbound_query"),
            index=state.get("index_pattern"),
            inbound_result=state.get("inbound_result"),
            outbound_result=state.get("outbound_result"),
            answer=state.get("answer"),
            kibana_payload=state.get("kibana_payload"),
            stage=stage,
            mode=mode,
            errors=all_errors,
        )

    def summary(self) -> str:
        lines = [
            "Illumio Blocked Flows Agent Result",
            f"  Mode:        {self.mode}",
            f"  Stage:       {self.stage.value}",
            f"  Target:      {self.target or '(none)'}  [{self.target_type or '?'}]",
            f"  Direction:   {self.direction or '(none)'}",
            f"  Date range:  {self.date_range or '(all time)'}",
            f"  Index:       {self.index or '(none)'}",
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

class IllumioBlockedAgent:
    """
    High-level orchestrator for the Illumio blocked-flows agent.

    Usage::

        agent = IllumioBlockedAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        result = await agent.run(
            "Est-ce qu'il y a eu des flux bloqués vers ou depuis le serveur web-prod-01 ?"
        )
        print(result.summary())

    Also works for application codes::

        result = await agent.run(
            "Y a-t-il des flux bloqués vers l'application AP12345 ?"
        )
    """

    def __init__(
        self,
        mcp_client: Any,
        chatmodel: ChatOpenAI,
        cfg: dict | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.chatmodel  = chatmodel
        self.cfg        = cfg if cfg is not None else _ILLUMIO_BLOCKED_CFG
        self.graph      = build_illumio_blocked_graph(chatmodel, mcp_client, self.cfg)

    async def run(self, request: str) -> IllumioBlockedResult:
        initial: IllumioBlockedState = {
            "user_request":    request,
            "target":          None,
            "target_type":     None,
            "direction":       None,
            "date_range":      None,
            "intent_error":    None,
            "inbound_query":   None,
            "outbound_query":  None,
            "index_pattern":   None,
            "inbound_result":  None,
            "outbound_result": None,
            "execution_error": None,
            "answer":          None,
            "kibana_payload":  None,
            "stage": IllumioBlockedStage.INIT,
            "error": None,
        }
        logger.info("Starting Illumio Blocked Flows Agent for: %s", request[:100])
        final = await self.graph.ainvoke(initial)
        return IllumioBlockedResult.from_state(final)
