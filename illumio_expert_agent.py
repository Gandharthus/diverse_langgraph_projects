"""
Illumio Expert Domain Agent
============================

The single conversational entry point for all Illumio-related questions.

Architecture
------------
The expert owns a typed state that accumulates extracted entities across turns.
When a sub-agent is needed, the expert passes only the relevant state variables
(app_code, hostname, direction, target_type) — never raw messages or full state.
Sub-agents return only two vars back: answer and kibana_payload.

Inter-agent contract
--------------------
  Expert → sub-agent:  app_code, direction            (traffic / consumers)
                        target, target_type, direction  (blocked)

  Sub-agent → expert:  answer          (str | None)
                        kibana_payload  (str | None)

If a required entity is missing the expert asks the user, stores the answer,
then retries the sub-agent — no raw message history is forwarded.

For general Illumio knowledge questions the expert answers directly, using a
bounded window of recent messages to avoid context overflow.

Module layout
-------------
- illumio_expert_prompts.py  – LLM prompt templates
- illumio_expert_agent.py    – state, nodes, graph, orchestrator  ← this file
- illumio_expert_graph.py    – thin entry-point for ``langgraph dev``
"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from illumio_agent import IllumioTrafficAgent
from illumio_blocked_agent import IllumioBlockedAgent
from illumio_consumers_agent import IllumioConsumersAgent
from illumio_expert_prompts import (
    ILLUMIO_EXPERT_CLASSIFY_EXTRACT_PROMPT,
    ILLUMIO_EXPERT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Maximum number of recent messages sent to the LLM for general answers.
# Keeps context bounded even in long conversations.
_MAX_GENERAL_HISTORY = 10


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertState(TypedDict, total=False):
    """Typed state for the Illumio Expert conversational agent.

    ``messages`` accumulates the full conversation (human + AI turns) using
    the ``add_messages`` reducer so LangGraph appends rather than replaces.

    Entity fields persist across turns – once set they are available in all
    subsequent graph invocations within the same thread/checkpoint.

    Sub-agent output is written to ``answer`` / ``kibana_payload`` only.
    No sub-agent internal state ever leaks into this state.
    """

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Extracted entities (persist across turns) ─────────────────────────────
    app_code:    Optional[str]   # "AP12345"     – required for traffic + consumers + blocked(app)
    hostname:    Optional[str]   # "web-prod-01" – required for blocked(hostname)
    direction:   Optional[str]   # traffic: "dev_to_prod"|"prod_to_dev"
                                 # blocked:  "inbound"|"outbound"|"both"
    target_type: Optional[str]   # "app" | "hostname" — for blocked intent only

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: Optional[str]        # "traffic" | "blocked" | "consumers" | "general"

    # ── Sub-agent output (clean boundary – only these two vars cross back) ────
    answer:         Optional[str]   # formatted natural-language answer
    kibana_payload: Optional[str]   # Kibana Dev Tools snippet (ES fallback)

    # ── Misc ──────────────────────────────────────────────────────────────────
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
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


def _format_recent_messages(messages: list[BaseMessage], n: int) -> str:
    """Return the last ``n`` messages as a plain-text string for the LLM."""
    recent = messages[-n:] if len(messages) > n else messages
    lines: list[str] = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Node: Classify intent + extract entities (single LLM call)
# ─────────────────────────────────────────────────────────────────────────────

async def classify_and_extract_node(
    state: IllumioExpertState, chatmodel: ChatOpenAI
) -> IllumioExpertState:
    """Single LLM call: classify intent and extract/update entity state vars.

    Passes the recent conversation as context so the LLM can:
    - Resolve coreferences ("same app", "ce serveur")
    - Understand follow-up questions ("and the blocked flows?")
    - Inherit entities from previous turns without re-parsing raw history
    """
    logger.info("Node: classify_and_extract")
    messages = state.get("messages", [])

    if not _last_human_text(messages).strip():
        return {**state, "intent": "general", "error": None}

    # Send last 8 messages (bounded) as plain-text context
    conversation_context = _format_recent_messages(messages, n=8)

    # Also expose currently known entities so the LLM can confirm/override them
    known = {
        "app_code":    state.get("app_code"),
        "hostname":    state.get("hostname"),
        "direction":   state.get("direction"),
        "target_type": state.get("target_type"),
    }
    known_str = json.dumps({k: v for k, v in known.items() if v is not None}, ensure_ascii=False)

    extraction_input = (
        f"Currently known entities: {known_str or '{}'}\n\n"
        f"Conversation:\n{conversation_context}\n\n"
        "Classify intent and extract/update entities from the latest User message."
    )

    llm_messages = [
        SystemMessage(content=ILLUMIO_EXPERT_CLASSIFY_EXTRACT_PROMPT),
        HumanMessage(content=extraction_input),
    ]
    response: AIMessage = await chatmodel.ainvoke(llm_messages)
    result = _extract_json(response.content)

    if not result:
        logger.warning("classify_and_extract: LLM returned non-JSON, defaulting to general")
        return {**state, "intent": "general", "error": None}

    intent = result.get("intent", "general")
    if intent not in ("traffic", "blocked", "consumers", "general"):
        intent = "general"

    # Merge: only update a field when the LLM returned a non-null value
    # (preserves previously extracted entities on follow-up turns)
    updates: dict[str, Any] = {"intent": intent, "error": None}
    for field in ("app_code", "hostname", "direction", "target_type"):
        value = result.get(field)
        if value is not None:
            updates[field] = value

    logger.info(
        "Intent: %s | app_code: %s | hostname: %s | direction: %s | target_type: %s",
        intent,
        updates.get("app_code", state.get("app_code")),
        updates.get("hostname", state.get("hostname")),
        updates.get("direction", state.get("direction")),
        updates.get("target_type", state.get("target_type")),
    )
    return {**state, **updates}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Ask the user for a missing required entity
# ─────────────────────────────────────────────────────────────────────────────

async def ask_for_entity_node(state: IllumioExpertState) -> IllumioExpertState:
    """Append a clarification question and end the turn.

    The user's answer on the next turn will be processed by classify_and_extract,
    which will extract the entity and store it in state before retrying.
    """
    logger.info("Node: ask_for_entity")
    intent = state.get("intent")

    if intent in ("traffic", "consumers"):
        msg = (
            "Pour répondre à votre question, j'ai besoin du **code application** (Code AP). "
            "Veuillez l'indiquer au format AP suivi de chiffres, par exemple : `AP12345`."
        )
    elif intent == "blocked":
        msg = (
            "Pour analyser les flux bloqués, j'ai besoin de l'identifiant de la cible. "
            "Veuillez préciser soit :\n"
            "- le **code application** (ex. `AP12345`)\n"
            "- le **nom du serveur** (ex. `web-prod-01`)"
        )
    else:
        msg = "Pourriez-vous préciser le code application ou le nom du serveur concerné ?"

    return {
        **state,
        "messages":       [AIMessage(content=msg)],
        "answer":         None,
        "kibana_payload": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodes: Sub-agent invocations
# Expert → sub-agent: typed entity vars only (no messages, no full state)
# Sub-agent → expert: answer + kibana_payload only
# ─────────────────────────────────────────────────────────────────────────────

async def invoke_traffic_agent_node(
    state: IllumioExpertState,
    traffic_agent: IllumioTrafficAgent,
) -> IllumioExpertState:
    """Call IllumioTrafficAgent with typed entity vars."""
    logger.info("Node: invoke_traffic_agent")
    app_code  = state["app_code"]  # guaranteed non-None by routing guard
    direction = state.get("direction") or "dev_to_prod"
    # Normalise: only traffic-valid values are acceptable
    if direction not in ("dev_to_prod", "prod_to_dev"):
        direction = "dev_to_prod"

    result = await traffic_agent.run_structured(app_code=app_code, direction=direction)

    answer         = None
    kibana_payload = None
    if result.mode == "answered":
        answer = result.answer
    elif result.mode == "kibana_fallback":
        kibana_payload = result.kibana_payload
    else:
        answer = (
            f"Je n'ai pas pu analyser le trafic inter-environnements pour {app_code} : "
            + ("; ".join(result.errors) if result.errors else "erreur inconnue.")
        )

    return {**state, "answer": answer, "kibana_payload": kibana_payload}


async def invoke_blocked_agent_node(
    state: IllumioExpertState,
    blocked_agent: IllumioBlockedAgent,
) -> IllumioExpertState:
    """Call IllumioBlockedAgent with typed entity vars."""
    logger.info("Node: invoke_blocked_agent")
    target_type = state.get("target_type", "hostname")
    target      = state["app_code"] if target_type == "app" else state["hostname"]
    direction   = state.get("direction") or "both"
    # Normalise: only blocked-valid values are acceptable
    if direction not in ("inbound", "outbound", "both"):
        direction = "both"

    result = await blocked_agent.run_structured(
        target=target, target_type=target_type, direction=direction
    )

    answer         = None
    kibana_payload = None
    if result.mode == "answered":
        answer = result.answer
    elif result.mode == "kibana_fallback":
        kibana_payload = result.kibana_payload
    else:
        answer = (
            f"Je n'ai pas pu analyser les flux bloqués pour {target} : "
            + ("; ".join(result.errors) if result.errors else "erreur inconnue.")
        )

    return {**state, "answer": answer, "kibana_payload": kibana_payload}


async def invoke_consumers_agent_node(
    state: IllumioExpertState,
    consumers_agent: IllumioConsumersAgent,
) -> IllumioExpertState:
    """Call IllumioConsumersAgent with typed entity vars."""
    logger.info("Node: invoke_consumers_agent")
    app_code = state["app_code"]  # guaranteed non-None by routing guard

    result = await consumers_agent.run_structured(app_code=app_code)

    answer         = None
    kibana_payload = None
    if result.mode == "answered":
        answer = result.answer
    elif result.mode == "kibana_fallback":
        kibana_payload = result.kibana_payload
    else:
        answer = (
            f"Je n'ai pas pu identifier les consommateurs du service {app_code} : "
            + ("; ".join(result.errors) if result.errors else "erreur inconnue.")
        )

    return {**state, "answer": answer, "kibana_payload": kibana_payload}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Direct LLM answer (general Illumio knowledge)
# ─────────────────────────────────────────────────────────────────────────────

async def answer_directly_node(
    state: IllumioExpertState, chatmodel: ChatOpenAI
) -> IllumioExpertState:
    """Answer a general Illumio question from LLM domain knowledge.

    Uses a bounded window of recent messages to prevent context overflow
    regardless of how long the conversation has been running.
    """
    logger.info("Node: answer_directly")
    messages = state.get("messages", [])
    recent = (
        messages[-_MAX_GENERAL_HISTORY:]
        if len(messages) > _MAX_GENERAL_HISTORY
        else messages
    )

    llm_messages: list[BaseMessage] = [SystemMessage(content=ILLUMIO_EXPERT_SYSTEM_PROMPT)]
    llm_messages.extend(recent)

    response: AIMessage = await chatmodel.ainvoke(llm_messages)
    return {**state, "answer": response.content, "kibana_payload": None}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Compose final response from answer / kibana_payload
# ─────────────────────────────────────────────────────────────────────────────

async def compose_response_node(state: IllumioExpertState) -> IllumioExpertState:
    """Build the AIMessage from sub-agent output vars and append it to messages."""
    logger.info("Node: compose_response")
    answer         = state.get("answer")
    kibana_payload = state.get("kibana_payload")

    if answer:
        text = answer
    elif kibana_payload:
        text = (
            "Je n'ai pas pu accéder directement à Elasticsearch. "
            "Voici la requête Kibana Dev Tools :\n\n"
            f"```\n{kibana_payload}\n```"
        )
    else:
        text = "Je n'ai pas pu traiter votre demande."

    return {**state, "messages": [AIMessage(content=text)]}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_classify(state: IllumioExpertState) -> str:
    """Route based on intent AND entity availability."""
    intent      = state.get("intent", "general")
    app_code    = state.get("app_code")
    hostname    = state.get("hostname")

    if intent == "traffic":
        return "invoke_traffic"   if app_code            else "ask_for_entity"
    if intent == "blocked":
        return "invoke_blocked"   if (app_code or hostname) else "ask_for_entity"
    if intent == "consumers":
        return "invoke_consumers" if app_code            else "ask_for_entity"
    return "answer_directly"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_expert_graph(
    chatmodel:       ChatOpenAI,
    traffic_agent:   IllumioTrafficAgent,
    blocked_agent:   IllumioBlockedAgent,
    consumers_agent: IllumioConsumersAgent,
) -> Any:
    """Construct and compile the Illumio Expert conversational StateGraph."""

    # ── Close over dependencies ──────────────────────────────────────────────

    async def _classify_and_extract(s: IllumioExpertState) -> IllumioExpertState:
        return await classify_and_extract_node(s, chatmodel)

    async def _invoke_traffic(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_traffic_agent_node(s, traffic_agent)

    async def _invoke_blocked(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_blocked_agent_node(s, blocked_agent)

    async def _invoke_consumers(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_consumers_agent_node(s, consumers_agent)

    async def _answer_directly(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_directly_node(s, chatmodel)

    # ── Build graph ──────────────────────────────────────────────────────────

    graph = StateGraph(IllumioExpertState)

    graph.add_node("classify_and_extract", _classify_and_extract)
    graph.add_node("ask_for_entity",       ask_for_entity_node)
    graph.add_node("invoke_traffic",       _invoke_traffic)
    graph.add_node("invoke_blocked",       _invoke_blocked)
    graph.add_node("invoke_consumers",     _invoke_consumers)
    graph.add_node("answer_directly",      _answer_directly)
    graph.add_node("compose_response",     compose_response_node)

    graph.set_entry_point("classify_and_extract")

    graph.add_conditional_edges(
        "classify_and_extract",
        _route_after_classify,
        {
            "ask_for_entity":  "ask_for_entity",
            "invoke_traffic":  "invoke_traffic",
            "invoke_blocked":  "invoke_blocked",
            "invoke_consumers":"invoke_consumers",
            "answer_directly": "answer_directly",
        },
    )

    # ask_for_entity ends the turn (user must reply with the missing entity)
    graph.add_edge("ask_for_entity", END)

    # All other paths flow through compose_response before ending
    for node in ("invoke_traffic", "invoke_blocked", "invoke_consumers", "answer_directly"):
        graph.add_edge(node, "compose_response")

    graph.add_edge("compose_response", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator (high-level Python API)
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertAgent:
    """
    High-level orchestrator for the Illumio Expert conversational agent.

    Single-turn::

        agent = IllumioExpertAgent(mcp_client=..., chatmodel=...)
        answer = await agent.chat("Qu'est-ce que le mode Selective ?")

    Multi-turn (the caller manages the state dict across turns)::

        agent = IllumioExpertAgent(mcp_client=..., chatmodel=...)
        state: dict = {}

        state, answer = await agent.chat_turn("Y a-t-il du trafic dev→prod ?", state)
        # → expert asks for app_code

        state, answer = await agent.chat_turn("AP12345", state)
        # → expert calls traffic sub-agent and returns the answer
    """

    def __init__(self, mcp_client: Any, chatmodel: ChatOpenAI) -> None:
        self.traffic_agent   = IllumioTrafficAgent(mcp_client=mcp_client, chatmodel=chatmodel)
        self.blocked_agent   = IllumioBlockedAgent(mcp_client=mcp_client, chatmodel=chatmodel)
        self.consumers_agent = IllumioConsumersAgent(mcp_client=mcp_client, chatmodel=chatmodel)
        self.graph = build_illumio_expert_graph(
            chatmodel=chatmodel,
            traffic_agent=self.traffic_agent,
            blocked_agent=self.blocked_agent,
            consumers_agent=self.consumers_agent,
        )

    async def chat(self, user_message: str) -> str:
        """Single-turn helper."""
        initial: IllumioExpertState = {
            "messages":       [HumanMessage(content=user_message)],
            "app_code":       None,
            "hostname":       None,
            "direction":      None,
            "target_type":    None,
            "intent":         None,
            "answer":         None,
            "kibana_payload": None,
            "error":          None,
        }
        final = await self.graph.ainvoke(initial)
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content
        return "Je n'ai pas pu traiter votre demande."

    async def chat_turn(
        self,
        user_message: str,
        prev_state: dict,
    ) -> tuple[dict, str]:
        """Multi-turn: merges the new user message into the previous state.

        ``prev_state`` should be the dict returned by the previous call (or
        an empty dict for the first turn).  All entity vars (app_code, hostname,
        direction, target_type, intent) are preserved across turns.
        """
        current: IllumioExpertState = {
            # Carry over entities and intent from previous turn
            "app_code":       prev_state.get("app_code"),
            "hostname":       prev_state.get("hostname"),
            "direction":      prev_state.get("direction"),
            "target_type":    prev_state.get("target_type"),
            "intent":         prev_state.get("intent"),
            # Append new user message to existing history
            "messages":       [
                *prev_state.get("messages", []),
                HumanMessage(content=user_message),
            ],
            "answer":         None,
            "kibana_payload": None,
            "error":          None,
        }
        final = await self.graph.ainvoke(current)

        answer = "Je n'ai pas pu traiter votre demande."
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        return dict(final), answer
