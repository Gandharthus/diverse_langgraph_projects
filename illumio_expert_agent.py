"""
Illumio Expert Agent
====================

A LangGraph-based orchestrator that routes every Illumio-related natural-
language question to the most appropriate specialist sub-agent:

  ┌─────────────────────────────────────────────────────────────────┐
  │              Illumio Expert Agent  (this file)                  │
  │                                                                 │
  │  parse_intent (LLM)                                             │
  │       │                                                         │
  │       ├─ traffic_analysis  → IllumioTrafficAgent                │
  │       ├─ blocked_flows     → IllumioBlockedAgent                │
  │       ├─ consumers         → IllumioConsumersAgent              │
  │       └─ query_builder     → IllumioQueryBuilderSubAgent ◄─────── NEW
  └─────────────────────────────────────────────────────────────────┘

The QueryBuilder integration uses the sub-agent's ``describe_skills()``
interface to dynamically populate the expert's intent-classification prompt,
keeping both agents in sync without manual maintenance.

Module layout
-------------
- **config.yaml**               – shared config (no dedicated section needed)
- **illumio_expert_prompts.py** – ``build_expert_intent_prompt(qb_skill)``
- **illumio_expert_agent.py**   – state, nodes, graph, orchestrator  ← this file
- **illumio_expert_graph.py**   – thin entry-point for ``langgraph dev``
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

from illumio_agent           import IllumioTrafficAgent
from illumio_blocked_agent   import IllumioBlockedAgent
from illumio_consumers_agent import IllumioConsumersAgent
from illumio_query_builder_agent import IllumioQueryBuilderSubAgent
from illumio_expert_prompts  import build_expert_intent_prompt, format_unknown_answer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (reuses the shared config.yaml)
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

# ─────────────────────────────────────────────────────────────────────────────
# Valid intents
# ─────────────────────────────────────────────────────────────────────────────

_VALID_INTENTS = frozenset(
    {"traffic_analysis", "blocked_flows", "consumers", "query_builder", "unknown"}
)

# ─────────────────────────────────────────────────────────────────────────────
# Stage
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertStage(str, Enum):
    """Tracks the current logical stage of the expert-agent workflow."""
    INIT          = "init"
    INTENT_PARSED = "intent_parsed"
    ANSWERED      = "answered"
    UNKNOWN       = "unknown"
    FAILED        = "failed"


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # ── Input ─────────────────────────────────────────────────────────────────
    user_request: str

    # ── Intent classification ─────────────────────────────────────────────────
    intent:        Optional[str]   # one of _VALID_INTENTS
    intent_error:  Optional[str]
    unknown_reason: Optional[str]

    # ── QueryBuilder extracted params (populated only for "query_builder") ────
    qb_source_app:      Optional[str]
    qb_destination_app: Optional[str]
    qb_policy_decision: Optional[str]
    qb_time_range:      Optional[dict]
    qb_env:             Optional[str]
    qb_aggs:            Optional[dict]

    # ── Delegated result ──────────────────────────────────────────────────────
    answer:         Optional[str]
    kibana_payload: Optional[str]

    # ── Control flow ──────────────────────────────────────────────────────────
    stage: IllumioExpertStage
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


# ─────────────────────────────────────────────────────────────────────────────
# Node: Parse intent  (LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def parse_intent_node(
    state: IllumioExpertState,
    chatmodel: ChatOpenAI,
    system_prompt: str,
) -> IllumioExpertState:
    """
    Use the LLM to classify the user's intent and extract QB params when needed.

    The *system_prompt* is built once at agent construction time from the
    QueryBuilder sub-agent's ``describe_skills()`` output, so it always
    reflects the sub-agent's actual parameter interface.
    """
    logger.info("Node: parse_intent (expert)")
    user_request = state.get("user_request", "")

    if not user_request.strip():
        return {
            **state,
            "intent":       "unknown",
            "intent_error": "Empty request.",
            "stage": IllumioExpertStage.FAILED,
            "error": "Empty user request.",
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_request),
    ]
    response: AIMessage = await chatmodel.ainvoke(messages)
    parsed = _extract_json(response.content)

    if parsed is None:
        return {
            **state,
            "intent":       None,
            "intent_error": "Could not parse intent from LLM output.",
            "stage": IllumioExpertStage.FAILED,
            "error": "Intent parsing failed – LLM returned non-JSON output.",
        }

    intent = parsed.get("intent", "unknown")
    if intent not in _VALID_INTENTS:
        intent = "unknown"

    base = {
        **state,
        "intent":        intent,
        "intent_error":  None,
        "unknown_reason": parsed.get("reason"),
        "stage": IllumioExpertStage.INTENT_PARSED,
    }

    # ── Extract QB params when routed to the query builder ───────────────────
    if intent == "query_builder":
        qb = parsed.get("qb_params", {}) or {}
        return {
            **base,
            "qb_source_app":      qb.get("source_app"),
            "qb_destination_app": qb.get("destination_app"),
            "qb_policy_decision": qb.get("policy_decision"),
            "qb_time_range":      qb.get("time_range"),
            "qb_env":             qb.get("env"),
            "qb_aggs":            qb.get("aggs"),
        }

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Node: Delegate → IllumioTrafficAgent
# ─────────────────────────────────────────────────────────────────────────────

async def answer_traffic_node(
    state: IllumioExpertState,
    traffic_agent: IllumioTrafficAgent,
) -> IllumioExpertState:
    """Delegate to the cross-environment traffic analysis specialist."""
    logger.info("Node: answer_traffic (expert)")
    result = await traffic_agent.run(state.get("user_request", ""))

    if result.mode == "failed":
        return {
            **state,
            "answer": None,
            "stage": IllumioExpertStage.FAILED,
            "error": "; ".join(result.errors) if result.errors else "Traffic agent failed.",
        }

    answer = result.answer or result.kibana_payload or result.summary()
    return {
        **state,
        "answer":         answer,
        "kibana_payload": result.kibana_payload,
        "stage": IllumioExpertStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Delegate → IllumioBlockedAgent
# ─────────────────────────────────────────────────────────────────────────────

async def answer_blocked_node(
    state: IllumioExpertState,
    blocked_agent: IllumioBlockedAgent,
) -> IllumioExpertState:
    """Delegate to the blocked-flows analysis specialist."""
    logger.info("Node: answer_blocked (expert)")
    result = await blocked_agent.run(state.get("user_request", ""))

    if result.mode == "failed":
        return {
            **state,
            "answer": None,
            "stage": IllumioExpertStage.FAILED,
            "error": "; ".join(result.errors) if result.errors else "Blocked-flows agent failed.",
        }

    answer = result.answer or result.kibana_payload or result.summary()
    return {
        **state,
        "answer":         answer,
        "kibana_payload": result.kibana_payload,
        "stage": IllumioExpertStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Delegate → IllumioConsumersAgent
# ─────────────────────────────────────────────────────────────────────────────

async def answer_consumers_node(
    state: IllumioExpertState,
    consumers_agent: IllumioConsumersAgent,
) -> IllumioExpertState:
    """Delegate to the service-consumers analysis specialist."""
    logger.info("Node: answer_consumers (expert)")
    result = await consumers_agent.run(state.get("user_request", ""))

    if result.mode == "failed":
        return {
            **state,
            "answer": None,
            "stage": IllumioExpertStage.FAILED,
            "error": "; ".join(result.errors) if result.errors else "Consumers agent failed.",
        }

    answer = result.answer or result.kibana_payload or result.summary()
    return {
        **state,
        "answer":         answer,
        "kibana_payload": result.kibana_payload,
        "stage": IllumioExpertStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Delegate → IllumioQueryBuilderSubAgent
# ─────────────────────────────────────────────────────────────────────────────

async def answer_query_builder_node(
    state: IllumioExpertState,
    qb_sub_agent: IllumioQueryBuilderSubAgent,
) -> IllumioExpertState:
    """
    Delegate to the QueryBuilder sub-agent using the structured params
    extracted by ``parse_intent_node``.

    The sub-agent's ``describe_skills()`` interface was already consulted
    at construction time to build the intent-classification prompt, ensuring
    the expert knows exactly what parameters to extract.
    """
    logger.info("Node: answer_query_builder (expert)")

    result = await qb_sub_agent.run(
        source_app=state.get("qb_source_app"),
        destination_app=state.get("qb_destination_app"),
        policy_decision=state.get("qb_policy_decision"),
        time_range=state.get("qb_time_range"),
        env=state.get("qb_env"),
        aggs=state.get("qb_aggs"),
    )

    if result.mode == "failed":
        return {
            **state,
            "answer": None,
            "stage": IllumioExpertStage.FAILED,
            "error": "; ".join(result.errors) if result.errors else "QueryBuilder sub-agent failed.",
        }

    return {
        **state,
        "answer":         result.summary(),
        "kibana_payload": result.kibana_payload,
        "stage": IllumioExpertStage.ANSWERED,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Unknown intent
# ─────────────────────────────────────────────────────────────────────────────

async def handle_unknown_node(state: IllumioExpertState) -> IllumioExpertState:
    """Return a helpful out-of-scope message."""
    logger.info("Node: handle_unknown (expert)")
    reason = state.get("unknown_reason") or "Intent not recognised."
    return {
        **state,
        "answer": format_unknown_answer(reason),
        "stage":  IllumioExpertStage.UNKNOWN,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_intent(state: IllumioExpertState) -> str:
    if state.get("stage") == IllumioExpertStage.FAILED:
        return "fail"
    intent = state.get("intent", "unknown")
    routes = {
        "traffic_analysis": "traffic",
        "blocked_flows":     "blocked",
        "consumers":         "consumers",
        "query_builder":     "query_builder",
    }
    return routes.get(intent, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_expert_graph(
    chatmodel:       ChatOpenAI,
    mcp_client:      Any,
    traffic_agent:   IllumioTrafficAgent   | None = None,
    blocked_agent:   IllumioBlockedAgent   | None = None,
    consumers_agent: IllumioConsumersAgent | None = None,
    qb_sub_agent:    IllumioQueryBuilderSubAgent | None = None,
) -> Any:
    """
    Construct and compile the Illumio Expert StateGraph.

    Specialist agents are instantiated here when not supplied, so the graph
    can be wired from the entry-point with just *chatmodel* and *mcp_client*.

    The intent-classification prompt is built dynamically from
    ``qb_sub_agent.describe_skills()``, so the expert's knowledge of the
    QueryBuilder's interface is always current.

    Graph layout::

        parse_intent
            │
            ├─ traffic_analysis  → answer_traffic       → END
            ├─ blocked_flows     → answer_blocked        → END
            ├─ consumers         → answer_consumers      → END
            ├─ query_builder     → answer_query_builder  → END
            ├─ unknown           → handle_unknown        → END
            └─ fail              → END
    """
    # ── Instantiate specialist agents if not provided ────────────────────────
    if traffic_agent is None:
        traffic_agent = IllumioTrafficAgent(
            mcp_client=mcp_client,
            chatmodel=chatmodel,
            cfg=_CFG.get("illumio_agent", {}),
        )
    if blocked_agent is None:
        blocked_agent = IllumioBlockedAgent(
            mcp_client=mcp_client,
            chatmodel=chatmodel,
            cfg=_CFG.get("illumio_blocked_agent", {}),
        )
    if consumers_agent is None:
        consumers_agent = IllumioConsumersAgent(
            mcp_client=mcp_client,
            chatmodel=chatmodel,
            cfg=_CFG.get("illumio_consumers_agent", {}),
        )
    if qb_sub_agent is None:
        qb_sub_agent = IllumioQueryBuilderSubAgent(
            mcp_client=mcp_client,
            cfg=_CFG.get("illumio_query_builder", {}),
        )

    # ── Build intent prompt from QB sub-agent's skills descriptor ────────────
    qb_skill      = qb_sub_agent.describe_skills()[0]
    system_prompt = build_expert_intent_prompt(qb_skill)

    logger.info(
        "Expert agent intent prompt built from QueryBuilder skill '%s'",
        qb_skill.get("name"),
    )

    # ── Node closures ─────────────────────────────────────────────────────────
    async def _parse_intent(s: IllumioExpertState) -> IllumioExpertState:
        return await parse_intent_node(s, chatmodel, system_prompt)

    async def _answer_traffic(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_traffic_node(s, traffic_agent)

    async def _answer_blocked(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_blocked_node(s, blocked_agent)

    async def _answer_consumers(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_consumers_node(s, consumers_agent)

    async def _answer_query_builder(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_query_builder_node(s, qb_sub_agent)

    # ── Graph wiring ──────────────────────────────────────────────────────────
    graph = StateGraph(IllumioExpertState)

    graph.add_node("parse_intent",          _parse_intent)
    graph.add_node("answer_traffic",        _answer_traffic)
    graph.add_node("answer_blocked",        _answer_blocked)
    graph.add_node("answer_consumers",      _answer_consumers)
    graph.add_node("answer_query_builder",  _answer_query_builder)
    graph.add_node("handle_unknown",        handle_unknown_node)

    graph.set_entry_point("parse_intent")

    graph.add_conditional_edges(
        "parse_intent",
        _route_after_intent,
        {
            "traffic":       "answer_traffic",
            "blocked":       "answer_blocked",
            "consumers":     "answer_consumers",
            "query_builder": "answer_query_builder",
            "unknown":       "handle_unknown",
            "fail":          END,
        },
    )

    graph.add_edge("answer_traffic",       END)
    graph.add_edge("answer_blocked",       END)
    graph.add_edge("answer_consumers",     END)
    graph.add_edge("answer_query_builder", END)
    graph.add_edge("handle_unknown",       END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IllumioExpertResult:
    """Structured output from an Illumio Expert agent run."""

    user_request:   str
    intent:         str | None
    answer:         str | None
    kibana_payload: str | None
    stage:          IllumioExpertStage
    mode:           Literal["answered", "unknown", "failed"]
    errors:         list[str]

    @classmethod
    def from_state(cls, state: IllumioExpertState) -> "IllumioExpertResult":
        all_errors: list[str] = []
        for key in ("error", "intent_error"):
            val = state.get(key)
            if val:
                all_errors.append(val)

        stage = state.get("stage", IllumioExpertStage.FAILED)
        if stage == IllumioExpertStage.ANSWERED:
            mode = "answered"
        elif stage == IllumioExpertStage.UNKNOWN:
            mode = "unknown"
        else:
            mode = "failed"

        return cls(
            user_request=state.get("user_request", ""),
            intent=state.get("intent"),
            answer=state.get("answer"),
            kibana_payload=state.get("kibana_payload"),
            stage=stage,
            mode=mode,
            errors=all_errors,
        )

    def summary(self) -> str:
        lines = [
            "Illumio Expert Agent Result",
            f"  Mode:    {self.mode}",
            f"  Stage:   {self.stage.value}",
            f"  Intent:  {self.intent or '(none)'}",
        ]

        if self.answer:
            lines.append("")
            lines.append("  Answer:")
            for line in self.answer.split("\n"):
                lines.append(f"    {line}")

        if self.errors:
            lines.append(f"\n  Errors ({len(self.errors)}):")
            for e in self.errors[:10]:
                lines.append(f"    - {e}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertAgent:
    """
    High-level orchestrator for the Illumio Expert Agent.

    Routes every Illumio-related natural-language question to the most
    appropriate specialist sub-agent:

    +-----------------------+-------------------------------+
    | Intent                | Specialist called             |
    +-----------------------+-------------------------------+
    | traffic_analysis      | IllumioTrafficAgent           |
    | blocked_flows         | IllumioBlockedAgent           |
    | consumers             | IllumioConsumersAgent         |
    | query_builder         | IllumioQueryBuilderSubAgent   |
    | unknown               | (out-of-scope reply)          |
    +-----------------------+-------------------------------+

    The QueryBuilder sub-agent's ``describe_skills()`` output is used at
    construction time to build the intent-classification prompt, ensuring the
    expert always knows the sub-agent's exact parameter interface.

    Usage::

        agent = IllumioExpertAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)

        # Natural-language question → automatic routing
        result = await agent.run(
            "Quels flux bloqués y a-t-il vers l'application AP12345 ?"
        )
        print(result.summary())

        # Structured query → routed to QueryBuilder sub-agent
        result = await agent.run(
            "Montre-moi les flux Blocked depuis A_AP12345- en HPROD "
            "sur les 7 derniers jours, agrégés par destination."
        )
        print(result.summary())
    """

    def __init__(
        self,
        mcp_client:      Any,
        chatmodel:       ChatOpenAI,
        traffic_agent:   IllumioTrafficAgent   | None = None,
        blocked_agent:   IllumioBlockedAgent   | None = None,
        consumers_agent: IllumioConsumersAgent | None = None,
        qb_sub_agent:    IllumioQueryBuilderSubAgent | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.chatmodel  = chatmodel
        self.graph = build_illumio_expert_graph(
            chatmodel=chatmodel,
            mcp_client=mcp_client,
            traffic_agent=traffic_agent,
            blocked_agent=blocked_agent,
            consumers_agent=consumers_agent,
            qb_sub_agent=qb_sub_agent,
        )

    async def run(self, request: str) -> IllumioExpertResult:
        """
        Route *request* to the appropriate specialist and return the result.

        Parameters
        ----------
        request:
            Natural-language question about Illumio network flows.

        Returns
        -------
        IllumioExpertResult
            Contains ``intent``, ``answer``, optional ``kibana_payload``,
            and status information.
        """
        initial: IllumioExpertState = {
            "user_request":       request,
            "intent":             None,
            "intent_error":       None,
            "unknown_reason":     None,
            "qb_source_app":      None,
            "qb_destination_app": None,
            "qb_policy_decision": None,
            "qb_time_range":      None,
            "qb_env":             None,
            "qb_aggs":            None,
            "answer":             None,
            "kibana_payload":     None,
            "stage": IllumioExpertStage.INIT,
            "error": None,
        }
        logger.info("Illumio Expert Agent: %s", request[:120])
        final = await self.graph.ainvoke(initial)
        return IllumioExpertResult.from_state(final)
