"""
Illumio Query Builder Sub-Agent
================================

A deterministic sub-agent that:
  1. Accepts structured query parameters (no NL parsing needed)
  2. Validates inputs against policy constraints
  3. Builds a valid Elasticsearch DSL query
  4. Executes it via the MCP ``search`` tool
  5. Returns the raw result (+ a Kibana Dev Tools fallback when MCP is unavailable)

Input interface
---------------
- ``time_range``      (optional) – dict with date-math keys for ``@timestamp``
- ``policy_decision`` (optional) – ``"Blocked"`` | ``"Allowed"``
- ``source_app``      (optional*) – prefix matched on ``illumio.source.labels.app``
- ``destination_app`` (optional*) – prefix matched on ``illumio.destination.labels.app``
  * at least one of ``source_app`` / ``destination_app`` is required
- ``env``             (optional) – ``"E_PROD"`` | ``"HPROD"``
- ``aggs``            (optional) – terms-only aggregations (max depth 2, max size 20)

DSL mapping rules
-----------------
- ``time_range``       → ``range`` filter on ``@timestamp``
- ``policy_decision``  → ``term``  filter on ``illumio.policy_decision``
- ``source_app``       → ``prefix`` filter on ``illumio.source.labels.app``
- ``destination_app``  → ``prefix`` filter on ``illumio.destination.labels.app``
- ``env = E_PROD``     → ``term`` filter on the env field of the selected app side
- ``env = HPROD``      → ``must_not term`` (env field = E_PROD) – HPROD is never literal

Aggregation constraints
-----------------------
- Only ``terms`` aggregations are allowed
- Maximum ``size`` per terms aggregation is 20
- Maximum nesting depth is 2
- Composite aggregations are forbidden

Module layout
-------------
- **config.yaml**                        – ``illumio_query_builder`` section
- **illumio_query_builder_agent.py**     – state, nodes, graph, orchestrator  ← this file
- **illumio_query_builder_graph.py**     – thin entry-point for ``langgraph dev``

This module also exposes ``QUERY_BUILDER_SKILLS`` — a descriptor list that the
Illumio expert agent uses to discover what this sub-agent can do.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import yaml
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Skills descriptor  (exposed to the Illumio expert agent)
# ─────────────────────────────────────────────────────────────────────────────

QUERY_BUILDER_SKILLS: list[dict[str, Any]] = [
    {
        "name": "build_and_execute_query",
        "description": (
            "Build a valid Elasticsearch DSL query from structured parameters "
            "and execute it against the Illumio flow-logs index via MCP. "
            "All filter translation is deterministic – no LLM is involved."
        ),
        "parameters": {
            "time_range": {
                "type": "object",
                "required": False,
                "description": (
                    "Time range filter on @timestamp. "
                    "Accepts standard Elasticsearch date-math keys: "
                    "gte, lte, gt, lt (e.g. {\"gte\": \"now-24h\", \"lte\": \"now\"})."
                ),
            },
            "policy_decision": {
                "type": "string",
                "required": False,
                "enum": ["Blocked", "Allowed"],
                "description": "Term filter on illumio.policy_decision.",
            },
            "source_app": {
                "type": "string",
                "required": "one of source_app / destination_app",
                "description": (
                    "Prefix value matched against illumio.source.labels.app. "
                    "At least one of source_app or destination_app is required."
                ),
            },
            "destination_app": {
                "type": "string",
                "required": "one of source_app / destination_app",
                "description": (
                    "Prefix value matched against illumio.destination.labels.app. "
                    "At least one of source_app or destination_app is required."
                ),
            },
            "env": {
                "type": "string",
                "required": False,
                "enum": ["E_PROD", "HPROD"],
                "description": (
                    "E_PROD → term filter on the env field of the selected app side "
                    "(illumio.source.labels.env or illumio.destination.labels.env). "
                    "HPROD → must_not term filter (env != E_PROD) on the same field. "
                    "HPROD never appears as a literal value in the query."
                ),
            },
            "aggs": {
                "type": "object",
                "required": False,
                "description": (
                    "Aggregation block injected into the DSL. "
                    "Constraints: only 'terms' aggregations; max size 20; "
                    "max nesting depth 2; composite aggregations are forbidden."
                ),
            },
        },
        "returns": {
            "query_json": "The Elasticsearch DSL query that was built.",
            "search_result": "The raw Elasticsearch response (None on hard failure).",
            "kibana_payload": "Kibana Dev Tools fallback payload (only when MCP is unavailable).",
            "stage": "Final workflow stage (executed | kibana_fallback | failed).",
        },
        "constraints": [
            "At least one of source_app or destination_app must be provided.",
            "policy_decision must be 'Blocked' or 'Allowed' if provided.",
            "env must be 'E_PROD' or 'HPROD' if provided.",
            "Only 'terms' aggregations are allowed.",
            "Aggregation size must not exceed 20.",
            "Aggregation nesting depth must not exceed 2.",
            "Composite aggregations are forbidden.",
            "Fields are never invented; only known Illumio fields are used.",
        ],
    }
]


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
_QB_CFG: dict[str, Any] = _CFG.get("illumio_query_builder", {})


# ─────────────────────────────────────────────────────────────────────────────
# Stage
# ─────────────────────────────────────────────────────────────────────────────

class QueryBuilderStage(str, Enum):
    """Tracks the current logical stage of the QueryBuilder workflow."""
    INIT            = "init"
    VALIDATED       = "validated"
    QUERY_BUILT     = "query_built"
    EXECUTED        = "executed"
    KIBANA_FALLBACK = "kibana_fallback"
    FAILED          = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class QueryBuilderState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # ── Input parameters ──────────────────────────────────────────────────────
    time_range:       Optional[dict]   # {"gte": "now-24h", "lte": "now"}
    policy_decision:  Optional[str]    # "Blocked" | "Allowed"
    source_app:       Optional[str]    # prefix for illumio.source.labels.app
    destination_app:  Optional[str]    # prefix for illumio.destination.labels.app
    env:              Optional[str]    # "E_PROD" | "HPROD"
    aggs:             Optional[dict]   # caller-supplied aggregation block

    # ── Built query ───────────────────────────────────────────────────────────
    query_json:    Optional[dict]
    index_pattern: Optional[str]

    # ── Execution ─────────────────────────────────────────────────────────────
    search_result:   Optional[dict]
    execution_error: Optional[str]

    # ── Output ────────────────────────────────────────────────────────────────
    kibana_payload: Optional[str]

    # ── Control flow ──────────────────────────────────────────────────────────
    stage:            QueryBuilderStage
    validation_error: Optional[str]
    error:            Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_aggs(aggs: dict, depth: int = 0) -> list[str]:
    """
    Recursively validate aggregations against policy constraints.

    Parameters
    ----------
    aggs:
        The aggregation dict to validate.
    depth:
        Current nesting level (0 = top-level aggregations).

    Returns
    -------
    list[str]
        Violation messages; an empty list means the aggregations are valid.
    """
    errors: list[str] = []

    if depth > 2:
        errors.append(
            f"Aggregation nesting depth exceeds the maximum of 2 "
            f"(reached depth {depth})."
        )
        return errors  # stop recursing to avoid cascading noise

    for agg_name, agg_body in aggs.items():
        if not isinstance(agg_body, dict):
            continue

        # ── Reject composite aggregations ────────────────────────────────────
        if "composite" in agg_body:
            errors.append(
                f"Aggregation '{agg_name}': composite aggregations are forbidden."
            )
            continue

        # ── Identify the aggregation type (any key that isn't sub-aggs) ──────
        agg_type: str | None = next(
            (k for k in agg_body if k not in ("aggs", "aggregations")),
            None,
        )

        if agg_type is not None and agg_type != "terms":
            errors.append(
                f"Aggregation '{agg_name}': only 'terms' aggregations are "
                f"allowed (got '{agg_type}')."
            )

        # ── Enforce max size for terms ────────────────────────────────────────
        if agg_type == "terms":
            terms_cfg = agg_body.get("terms", {})
            if isinstance(terms_cfg, dict):
                size = terms_cfg.get("size", 10)
                if isinstance(size, int) and size > 20:
                    errors.append(
                        f"Aggregation '{agg_name}': 'size' must not exceed 20 "
                        f"(got {size})."
                    )

        # ── Recurse into nested aggregations ─────────────────────────────────
        nested: dict = agg_body.get("aggs") or agg_body.get("aggregations") or {}
        if nested:
            errors.extend(_validate_aggs(nested, depth + 1))

    return errors


def _build_dsl_query(
    time_range: dict | None,
    policy_decision: str | None,
    source_app: str | None,
    destination_app: str | None,
    env: str | None,
    aggs: dict | None,
    cfg: dict,
) -> dict:
    """
    Deterministically build the Elasticsearch DSL query.

    All field names and environment values are read from *cfg* so they can be
    overridden via ``config.yaml`` without touching this function.

    Mapping rules
    ~~~~~~~~~~~~~
    - ``time_range``      → ``range``  filter on ``@timestamp``
    - ``policy_decision`` → ``term``   filter on ``illumio.policy_decision``
    - ``source_app``      → ``prefix`` filter on ``illumio.source.labels.app``
    - ``destination_app`` → ``prefix`` filter on ``illumio.destination.labels.app``
    - ``env = E_PROD``    → ``term`` filter on the env field of the selected side
    - ``env = HPROD``     → ``must_not term`` (env field = E_PROD) on the selected side
    """
    timestamp_field  = cfg.get("timestamp_field",        "@timestamp")
    policy_field     = cfg.get("policy_decision_field",  "illumio.policy_decision")
    src_app_field    = cfg.get("source_app_field",       "illumio.source.labels.app")
    dst_app_field    = cfg.get("dest_app_field",         "illumio.destination.labels.app")
    src_env_field    = cfg.get("source_env_field",       "illumio.source.labels.env")
    dst_env_field    = cfg.get("dest_env_field",         "illumio.destination.labels.env")
    prod_env_value   = cfg.get("prod_env_value",         "E_PROD")

    must_filters:     list[dict] = []
    must_not_filters: list[dict] = []

    # ── time_range → range filter on @timestamp ───────────────────────────────
    if time_range:
        must_filters.append({"range": {timestamp_field: time_range}})

    # ── policy_decision → term filter on illumio.policy_decision ─────────────
    if policy_decision:
        must_filters.append({"term": {policy_field: policy_decision}})

    # ── source_app → prefix filter on illumio.source.labels.app ──────────────
    if source_app:
        must_filters.append({"prefix": {src_app_field: source_app}})

    # ── destination_app → prefix filter on illumio.destination.labels.app ─────
    if destination_app:
        must_filters.append({"prefix": {dst_app_field: destination_app}})

    # ── env → term / must_not on the matching side's env field ───────────────
    if env:
        # Apply the env filter to whichever side(s) have an app selector
        env_fields: list[str] = []
        if source_app:
            env_fields.append(src_env_field)
        if destination_app:
            env_fields.append(dst_env_field)

        if env == "E_PROD":
            for ef in env_fields:
                must_filters.append({"term": {ef: prod_env_value}})
        elif env == "HPROD":
            # HPROD never appears as a literal value; it means NOT E_PROD
            for ef in env_fields:
                must_not_filters.append({"term": {ef: prod_env_value}})

    return {
        "size": 0,
        "query": {
            "bool": {
                "filter":   must_filters,
                "must_not": must_not_filters,
            }
        },
        "aggs": aggs or {},
    }


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


def _format_kibana_payload(index_pattern: str, query: dict) -> str:
    """Format a query as a Kibana Dev Tools console snippet."""
    return (
        f"GET {index_pattern}/_search\n"
        + json.dumps(query, indent=2, ensure_ascii=False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node: Validate input parameters
# ─────────────────────────────────────────────────────────────────────────────

async def validate_params_node(state: QueryBuilderState) -> QueryBuilderState:
    """Validate that all structured input parameters satisfy policy constraints."""
    logger.info("Node: validate_params (query_builder)")
    errors: list[str] = []

    source_app      = state.get("source_app")
    destination_app = state.get("destination_app")
    policy_decision = state.get("policy_decision")
    env             = state.get("env")
    time_range      = state.get("time_range")
    aggs            = state.get("aggs")

    # ── At least one app selector is required ────────────────────────────────
    if not source_app and not destination_app:
        errors.append(
            "At least one of 'source_app' or 'destination_app' must be provided."
        )

    # ── policy_decision must be a recognised value ────────────────────────────
    if policy_decision is not None and policy_decision not in ("Blocked", "Allowed"):
        errors.append(
            f"'policy_decision' must be 'Blocked' or 'Allowed' "
            f"(got '{policy_decision}')."
        )

    # ── env must be a recognised value ────────────────────────────────────────
    if env is not None and env not in ("E_PROD", "HPROD"):
        errors.append(f"'env' must be 'E_PROD' or 'HPROD' (got '{env}').")

    # ── time_range must be a dict ─────────────────────────────────────────────
    if time_range is not None and not isinstance(time_range, dict):
        errors.append(
            f"'time_range' must be a dict (got {type(time_range).__name__})."
        )

    # ── Validate aggregations ─────────────────────────────────────────────────
    if aggs is not None:
        if not isinstance(aggs, dict):
            errors.append(f"'aggs' must be a dict (got {type(aggs).__name__}).")
        else:
            errors.extend(_validate_aggs(aggs))

    if errors:
        validation_error = "; ".join(errors)
        logger.warning("Param validation failed: %s", validation_error)
        return {
            **state,
            "validation_error": validation_error,
            "stage": QueryBuilderStage.FAILED,
            "error": validation_error,
        }

    return {
        **state,
        "validation_error": None,
        "stage": QueryBuilderStage.VALIDATED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Build DSL query  (deterministic – no LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def build_query_node(
    state: QueryBuilderState, cfg: dict
) -> QueryBuilderState:
    """Deterministically build the Elasticsearch DSL query from the validated params."""
    logger.info("Node: build_query (query_builder)")
    index_pattern = cfg.get("illumio_index_pattern", "your-index-*")

    query = _build_dsl_query(
        time_range=state.get("time_range"),
        policy_decision=state.get("policy_decision"),
        source_app=state.get("source_app"),
        destination_app=state.get("destination_app"),
        env=state.get("env"),
        aggs=state.get("aggs"),
        cfg=cfg,
    )

    return {
        **state,
        "query_json":    query,
        "index_pattern": index_pattern,
        "stage": QueryBuilderStage.QUERY_BUILT,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Execute search via MCP
# ─────────────────────────────────────────────────────────────────────────────

async def execute_search_node(
    state: QueryBuilderState, mcp_client: Any
) -> QueryBuilderState:
    """Execute the built DSL query via the MCP ``search`` tool."""
    logger.info("Node: execute_search (query_builder)")
    query         = state.get("query_json")
    index_pattern = state.get("index_pattern", "your-index-*")

    if not query:
        return {
            **state,
            "search_result":   None,
            "execution_error": "No query to execute.",
            "stage": QueryBuilderStage.FAILED,
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
                "stage": QueryBuilderStage.EXECUTED,
            }

        result = await search_tool.ainvoke({
            "index": index_pattern,
            "body":  query,
        })
        parsed = _parse_mcp_result(result)

        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            error_str = (
                err.get("reason", str(err)) if isinstance(err, dict) else str(err)
            )
            return {
                **state,
                "search_result":   parsed,
                "execution_error": f"Elasticsearch error: {error_str}",
                "stage": QueryBuilderStage.EXECUTED,
            }

        return {
            **state,
            "search_result":   parsed,
            "execution_error": None,
            "stage": QueryBuilderStage.EXECUTED,
        }

    except Exception as exc:
        logger.exception("Search execution failed")
        return {
            **state,
            "search_result":   None,
            "execution_error": f"Search exception: {exc}",
            "stage": QueryBuilderStage.EXECUTED,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

async def kibana_fallback_node(state: QueryBuilderState) -> QueryBuilderState:
    """Generate a Kibana Dev Tools payload when direct ES access is unavailable."""
    logger.info("Node: kibana_fallback (query_builder)")
    query         = state.get("query_json", {})
    index_pattern = state.get("index_pattern", "your-index-*")
    payload       = _format_kibana_payload(index_pattern, query or {})
    return {
        **state,
        "kibana_payload": payload,
        "stage": QueryBuilderStage.KIBANA_FALLBACK,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_validation(state: QueryBuilderState) -> str:
    if state.get("stage") == QueryBuilderStage.FAILED:
        return "fail"
    return "build_query"


def _route_after_execution(state: QueryBuilderState) -> str:
    # Fall back to Kibana only when no search result was obtained at all
    # (tool missing, network exception, etc.)
    if state.get("search_result") is None:
        return "kibana"
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_query_builder_graph(
    mcp_client: Any,
    cfg: dict | None = None,
) -> Any:
    """
    Construct and compile the Illumio QueryBuilder StateGraph.

    Graph layout::

        validate_params
            │ valid
            ▼
        build_query
            │
            ▼
        execute_search
            │ search_result != None     │ search_result is None
            ▼                           ▼
           END                    kibana_fallback
                                        │
                                       END
    """
    if cfg is None:
        cfg = _QB_CFG

    async def _build_query(s: QueryBuilderState) -> QueryBuilderState:
        return await build_query_node(s, cfg)

    async def _execute_search(s: QueryBuilderState) -> QueryBuilderState:
        return await execute_search_node(s, mcp_client)

    graph = StateGraph(QueryBuilderState)

    # ── Nodes ──
    graph.add_node("validate_params", validate_params_node)
    graph.add_node("build_query",     _build_query)
    graph.add_node("execute_search",  _execute_search)
    graph.add_node("kibana_fallback", kibana_fallback_node)

    # ── Entry ──
    graph.set_entry_point("validate_params")

    # ── Edges ──
    graph.add_conditional_edges(
        "validate_params",
        _route_after_validation,
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
            "done":   END,
            "kibana": "kibana_fallback",
        },
    )

    graph.add_edge("kibana_fallback", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryBuilderResult:
    """Structured output from a QueryBuilder sub-agent run."""

    # ── Echo of inputs ────────────────────────────────────────────────────────
    time_range:       dict | None
    policy_decision:  str | None
    source_app:       str | None
    destination_app:  str | None
    env:              str | None
    aggs:             dict | None

    # ── Outputs ───────────────────────────────────────────────────────────────
    query:          dict | None
    index:          str | None
    search_result:  dict | None
    kibana_payload: str | None

    # ── Status ────────────────────────────────────────────────────────────────
    stage:  QueryBuilderStage
    mode:   Literal["executed", "kibana_fallback", "failed"]
    errors: list[str]

    @classmethod
    def from_state(cls, state: QueryBuilderState) -> "QueryBuilderResult":
        all_errors: list[str] = []
        for key in ("error", "validation_error", "execution_error"):
            val = state.get(key)
            if val:
                all_errors.append(val)

        stage = state.get("stage", QueryBuilderStage.FAILED)
        if stage == QueryBuilderStage.EXECUTED:
            mode = "executed"
        elif stage == QueryBuilderStage.KIBANA_FALLBACK:
            mode = "kibana_fallback"
        else:
            mode = "failed"

        return cls(
            time_range=state.get("time_range"),
            policy_decision=state.get("policy_decision"),
            source_app=state.get("source_app"),
            destination_app=state.get("destination_app"),
            env=state.get("env"),
            aggs=state.get("aggs"),
            query=state.get("query_json"),
            index=state.get("index_pattern"),
            search_result=state.get("search_result"),
            kibana_payload=state.get("kibana_payload"),
            stage=stage,
            mode=mode,
            errors=all_errors,
        )

    def summary(self) -> str:
        lines = [
            "Illumio QueryBuilder Sub-Agent Result",
            f"  Mode:             {self.mode}",
            f"  Stage:            {self.stage.value}",
            f"  source_app:       {self.source_app or '(none)'}",
            f"  destination_app:  {self.destination_app or '(none)'}",
            f"  policy_decision:  {self.policy_decision or '(none)'}",
            f"  env:              {self.env or '(none)'}",
            f"  Index:            {self.index or '(none)'}",
        ]

        if self.query:
            lines.append("")
            lines.append("  DSL query:")
            for line in json.dumps(self.query, indent=4, ensure_ascii=False).split("\n"):
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
# Orchestrator  (Sub-Agent public interface)
# ─────────────────────────────────────────────────────────────────────────────

class IllumioQueryBuilderSubAgent:
    """
    Deterministic sub-agent that builds and executes Illumio ES queries.

    This agent accepts **structured parameters** directly (no natural-language
    parsing) and is designed to be called programmatically by the Illumio
    expert agent or other orchestrators.

    No LLM is involved — all DSL construction is rule-based.

    Usage
    -----
    Build and execute a query::

        agent = IllumioQueryBuilderSubAgent(mcp_client=MCP_CLIENT)
        result = await agent.run(
            source_app="A_AP12345-",
            policy_decision="Blocked",
            time_range={"gte": "now-7d", "lte": "now"},
            aggs={
                "top_destinations": {
                    "terms": {"field": "illumio.destination.labels.app", "size": 10}
                }
            },
        )
        print(result.summary())

    Discover capabilities (for the expert agent)::

        skills = agent.describe_skills()
        # Returns QUERY_BUILDER_SKILLS — a list with one skill descriptor dict.
    """

    def __init__(
        self,
        mcp_client: Any,
        cfg: dict | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.cfg        = cfg if cfg is not None else _QB_CFG
        self.graph      = build_illumio_query_builder_graph(mcp_client, self.cfg)

    # ── Skills interface ───────────────────────────────────────────────────────

    def describe_skills(self) -> list[dict[str, Any]]:
        """
        Return the skills descriptor for this sub-agent.

        The Illumio expert agent calls this method to discover what the
        QueryBuilder can do and what parameters it accepts.

        Returns
        -------
        list[dict]
            Module-level ``QUERY_BUILDER_SKILLS`` — a single-element list
            containing a descriptor dict with keys:
            ``name``, ``description``, ``parameters``, ``returns``,
            ``constraints``.
        """
        return QUERY_BUILDER_SKILLS

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(
        self,
        *,
        source_app: str | None = None,
        destination_app: str | None = None,
        policy_decision: str | None = None,
        time_range: dict | None = None,
        env: str | None = None,
        aggs: dict | None = None,
    ) -> QueryBuilderResult:
        """
        Build and execute an Elasticsearch DSL query from structured parameters.

        Parameters
        ----------
        source_app:
            Prefix value matched against ``illumio.source.labels.app``.
            At least one of ``source_app`` / ``destination_app`` is required.
        destination_app:
            Prefix value matched against ``illumio.destination.labels.app``.
            At least one of ``source_app`` / ``destination_app`` is required.
        policy_decision:
            ``"Blocked"`` or ``"Allowed"`` — term filter on
            ``illumio.policy_decision``.
        time_range:
            Dict with date-math keys (``gte``, ``lte``, ``gt``, ``lt``) for
            the ``@timestamp`` range filter.
            Example: ``{"gte": "now-24h", "lte": "now"}``
        env:
            ``"E_PROD"``  → adds a ``term`` filter on the env field of the
            selected app side.
            ``"HPROD"``   → adds a ``must_not term`` filter (env != E_PROD)
            on the same field.  HPROD never appears as a literal value.
        aggs:
            Aggregation block.  Only ``terms`` aggs are allowed; max size 20;
            max nesting depth 2; composite aggregations are forbidden.

        Returns
        -------
        QueryBuilderResult
            Contains the built DSL query, raw ES search result (or Kibana
            fallback payload), and status/error information.
        """
        initial: QueryBuilderState = {
            "time_range":       time_range,
            "policy_decision":  policy_decision,
            "source_app":       source_app,
            "destination_app":  destination_app,
            "env":              env,
            "aggs":             aggs,
            "query_json":       None,
            "index_pattern":    None,
            "search_result":    None,
            "execution_error":  None,
            "kibana_payload":   None,
            "stage":            QueryBuilderStage.INIT,
            "validation_error": None,
            "error":            None,
        }

        logger.info(
            "QueryBuilder sub-agent | source_app=%s destination_app=%s "
            "policy_decision=%s env=%s",
            source_app,
            destination_app,
            policy_decision,
            env,
        )

        final = await self.graph.ainvoke(initial)
        return QueryBuilderResult.from_state(final)
