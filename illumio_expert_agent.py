"""
Illumio Expert Domain Agent
============================

A conversational LangGraph agent that is the single entry point for all
Illumio-related questions at BNP Paribas.

It classifies the user's intent from the conversation and delegates to
specialised sub-agents when a live Elasticsearch query is required:

  - IllumioTrafficAgent    → cross-environment traffic (dev↔prod)
  - IllumioBlockedAgent    → blocked / denied flow analysis
  - IllumioConsumersAgent  → service consumer discovery

For general Illumio knowledge questions (concepts, labels, policies, …) it
answers directly from its built-in domain expertise without querying ES.

Conversation is multi-turn: the full message history is carried in the
``messages`` field using the LangGraph ``add_messages`` reducer, which means
the LangGraph API server can checkpoint and restore threads transparently.

Module layout
-------------
- **illumio_expert_prompts.py** – LLM prompt templates
- **illumio_expert_agent.py**   – state, nodes, graph, orchestrator  ← this file
- **illumio_expert_graph.py**   – thin entry-point for ``langgraph dev``
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
    ILLUMIO_EXPERT_INTENT_SYSTEM_PROMPT,
    ILLUMIO_EXPERT_LANGUAGE_ADAPT_PROMPT,
    ILLUMIO_EXPERT_NATURAL_RESPONSE_PROMPT,
    ILLUMIO_EXPERT_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class IllumioExpertState(TypedDict, total=False):
    """State for the conversational Illumio expert agent.

    ``messages`` uses the ``add_messages`` reducer so that LangGraph appends
    new messages rather than replacing the list on each graph invocation.
    This enables transparent multi-turn conversation via the LangGraph API.
    """

    # Conversation history (human + AI turns)
    messages: Annotated[list[BaseMessage], add_messages]

    # Intermediate classification result
    intent: Optional[str]  # "traffic" | "blocked" | "consumers" | "general"

    # Answer produced by the chosen sub-agent or direct LLM response
    subagent_answer: Optional[str]

    # Error from classification (non-fatal – falls back to "general")
    error: Optional[str]

    # Error from the most recent sub-agent call (None when successful).
    # Set to a human-readable description whenever a sub-agent falls back to
    # Kibana or fails entirely, e.g.:
    #   "Elasticsearch MCP unavailable: <reason>. Kibana fallback provided."
    call_error: Optional[str]

    # Persistent entities extracted from the conversation.
    # Once set, these are retained across turns so sub-agents always have
    # enough context even when the user omits them in follow-up messages.
    ap_code:    Optional[str]   # e.g. "AP12345"
    hostname:   Optional[str]   # e.g. "srv-prod-db01"
    date_range: Optional[str]   # ES relative date-math, e.g. "now-2h", "now-7d"


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


def _last_human_message(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
    return ""


def _format_conversation_context(messages: list[BaseMessage], max_turns: int = 6) -> str:
    """Format the recent conversation history as a plain-text string for the LLM."""
    recent = messages[-max_turns:] if len(messages) > max_turns else messages
    lines: list[str] = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _enrich_request(
    user_request: str,
    ap_code: str | None,
    hostname: str | None,
    date_range: str | None = None,
) -> str:
    """Prepend known context entities so sub-agents don't need to re-extract them.

    This lets the user say "and what about the blocked flows?" in a follow-up
    without repeating the AP code, hostname, or time range — the expert agent
    remembers them.
    """
    extras: list[str] = []
    if ap_code:
        extras.append(f"AP code: {ap_code}")
    if hostname:
        extras.append(f"Hostname: {hostname}")
    if date_range:
        extras.append(f"Date range: {date_range}")
    if extras:
        context_line = "[Known context — " + ", ".join(extras) + "]"
        return f"{context_line}\n\n{user_request}"
    return user_request


_INTENT_DESCRIPTIONS: dict[str, str] = {
    "traffic":   "cross-environment traffic analysis (dev ↔ prod flows)",
    "blocked":   "blocked / denied flow analysis",
    "consumers": "service consumer discovery (which apps connect to a service)",
}


async def _naturalize_fallback(
    chatmodel: ChatOpenAI,
    conversation_context: str,
    intent: str,
    mode: str,  # "kibana_fallback" | "failed"
    errors: list[str],
    kibana_payload: str | None,
) -> str:
    """Ask the LLM to produce a natural, language-aware response for fallback/error cases.

    For *kibana_fallback*: the LLM wraps the Kibana query in a friendly message.
    For *failed*: the LLM figures out what info is missing and asks for it naturally.
    """
    intent_desc = _INTENT_DESCRIPTIONS.get(intent, intent)

    if mode == "kibana_fallback" and kibana_payload:
        situation = "Direct Elasticsearch access is unavailable."
        extra = (
            "You have a Kibana Dev Tools query that the user can run manually to get "
            "the data they need. Present it as a useful next step:\n\n"
            f"```\n{kibana_payload}\n```"
        )
    else:
        error_str = "; ".join(errors) if errors else "Unknown error."
        situation = f"The query could not be completed. Technical reason: {error_str}"
        extra = (
            "If the failure is because the user did not provide a required identifier "
            "(such as an AP code, a hostname, or an application name), ask for it "
            "naturally and positively — do NOT apologise excessively. "
            "Otherwise, briefly explain the issue and suggest what the user can do next."
        )

    prompt = ILLUMIO_EXPERT_NATURAL_RESPONSE_PROMPT.format(
        intent_description=intent_desc,
        conversation_context=conversation_context,
        situation_description=situation,
        extra_context=extra,
    )

    response: AIMessage = await chatmodel.ainvoke([SystemMessage(content=prompt)])
    return response.content


async def _adapt_language(
    chatmodel: ChatOpenAI,
    conversation_context: str,
    subagent_answer: str,
) -> str:
    """Rewrite a sub-agent answer in the user's language (detected from conversation)."""
    prompt = ILLUMIO_EXPERT_LANGUAGE_ADAPT_PROMPT.format(
        subagent_answer=subagent_answer,
        conversation_context=conversation_context,
    )
    response: AIMessage = await chatmodel.ainvoke([SystemMessage(content=prompt)])
    return response.content


# ─────────────────────────────────────────────────────────────────────────────
# Node: Classify intent
# ─────────────────────────────────────────────────────────────────────────────

async def classify_intent_node(
    state: IllumioExpertState, chatmodel: ChatOpenAI
) -> IllumioExpertState:
    """Use the LLM to classify the user's intent from the latest message."""
    logger.info("Node: classify_intent")
    messages = state.get("messages", [])
    user_request = _last_human_message(messages)

    if not user_request.strip():
        return {**state, "intent": "general", "error": None}

    # Include recent conversation so the classifier handles follow-up questions
    context = _format_conversation_context(messages)
    classification_input = (
        f"Conversation so far:\n{context}\n\n"
        "Classify the intent of the latest User message."
    )

    llm_messages = [
        SystemMessage(content=ILLUMIO_EXPERT_INTENT_SYSTEM_PROMPT),
        HumanMessage(content=classification_input),
    ]
    try:
        response: AIMessage = await chatmodel.ainvoke(llm_messages)
        result = _extract_json(response.content)
    except Exception as exc:
        logger.exception("classify_intent LLM call failed")
        return {**state, "intent": "general", "error": f"Intent classification failed: {exc}"}

    intent = "general"  # safe default
    if result and isinstance(result.get("intent"), str):
        raw = result["intent"]
        if raw in ("traffic", "blocked", "consumers", "general"):
            intent = raw

    # Extract entities – only overwrite existing state values when the LLM
    # found something new in the current message (non-null, non-empty string).
    ap_code    = state.get("ap_code")
    hostname   = state.get("hostname")
    date_range = state.get("date_range")

    if result:
        new_ap    = result.get("ap_code")
        new_host  = result.get("hostname")
        new_range = result.get("date_range")
        if isinstance(new_ap, str) and new_ap.strip():
            ap_code = new_ap.strip()
        if isinstance(new_host, str) and new_host.strip():
            hostname = new_host.strip()
        if isinstance(new_range, str) and new_range.strip():
            date_range = new_range.strip()

    logger.info(
        "Classified intent: %s | ap_code=%s | hostname=%s | date_range=%s",
        intent, ap_code, hostname, date_range,
    )
    return {
        **state,
        "intent":     intent,
        "error":      None,
        "ap_code":    ap_code,
        "hostname":   hostname,
        "date_range": date_range,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodes: Sub-agent invocations
# ─────────────────────────────────────────────────────────────────────────────

async def invoke_traffic_agent_node(
    state: IllumioExpertState,
    traffic_agent: IllumioTrafficAgent,
    chatmodel: ChatOpenAI,
) -> IllumioExpertState:
    """Delegate to the IllumioTrafficAgent sub-agent."""
    logger.info("Node: invoke_traffic_agent")
    messages = state.get("messages", [])
    user_request = _last_human_message(messages)
    context = _format_conversation_context(messages)
    enriched = _enrich_request(user_request, state.get("ap_code"), state.get("hostname"), state.get("date_range"))

    try:
        result = await traffic_agent.run(enriched)
    except Exception as exc:
        logger.exception("Traffic agent run raised an unexpected exception")
        call_error = f"Traffic agent crashed unexpectedly: {exc}"
        return {
            **state,
            "subagent_answer": await _naturalize_fallback(
                chatmodel, context, "traffic", "failed", [call_error], None
            ),
            "call_error": call_error,
        }

    call_error: str | None = None
    if result.mode == "answered" and result.answer:
        answer = await _adapt_language(chatmodel, context, result.answer)
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        error_detail = (": " + "; ".join(result.errors)) if result.errors else ""
        call_error = (
            f"Elasticsearch MCP access failed{error_detail}. "
            "Kibana fallback query provided instead."
        )
        logger.warning("Traffic agent kibana_fallback – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "traffic", "kibana_fallback",
            result.errors, result.kibana_payload,
        )
    else:
        call_error = (
            "; ".join(result.errors)
            if result.errors
            else "Traffic agent failed with unknown error."
        )
        logger.error("Traffic agent failed – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "traffic", "failed",
            result.errors, None,
        )

    return {**state, "subagent_answer": answer, "call_error": call_error}


async def invoke_blocked_agent_node(
    state: IllumioExpertState,
    blocked_agent: IllumioBlockedAgent,
    chatmodel: ChatOpenAI,
) -> IllumioExpertState:
    """Delegate to the IllumioBlockedAgent sub-agent."""
    logger.info("Node: invoke_blocked_agent")
    messages = state.get("messages", [])
    user_request = _last_human_message(messages)
    context = _format_conversation_context(messages)
    enriched = _enrich_request(user_request, state.get("ap_code"), state.get("hostname"), state.get("date_range"))

    try:
        result = await blocked_agent.run(enriched)
    except Exception as exc:
        logger.exception("Blocked agent run raised an unexpected exception")
        call_error = f"Blocked-flows agent crashed unexpectedly: {exc}"
        return {
            **state,
            "subagent_answer": await _naturalize_fallback(
                chatmodel, context, "blocked", "failed", [call_error], None
            ),
            "call_error": call_error,
        }

    call_error: str | None = None
    if result.mode == "answered" and result.answer:
        answer = await _adapt_language(chatmodel, context, result.answer)
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        error_detail = (": " + "; ".join(result.errors)) if result.errors else ""
        call_error = (
            f"Elasticsearch MCP access failed{error_detail}. "
            "Kibana fallback query provided instead."
        )
        logger.warning("Blocked agent kibana_fallback – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "blocked", "kibana_fallback",
            result.errors, result.kibana_payload,
        )
    else:
        call_error = (
            "; ".join(result.errors)
            if result.errors
            else "Blocked-flows agent failed with unknown error."
        )
        logger.error("Blocked agent failed – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "blocked", "failed",
            result.errors, None,
        )

    return {**state, "subagent_answer": answer, "call_error": call_error}


async def invoke_consumers_agent_node(
    state: IllumioExpertState,
    consumers_agent: IllumioConsumersAgent,
    chatmodel: ChatOpenAI,
) -> IllumioExpertState:
    """Delegate to the IllumioConsumersAgent sub-agent."""
    logger.info("Node: invoke_consumers_agent")
    messages = state.get("messages", [])
    user_request = _last_human_message(messages)
    context = _format_conversation_context(messages)
    enriched = _enrich_request(user_request, state.get("ap_code"), state.get("hostname"), state.get("date_range"))

    try:
        result = await consumers_agent.run(enriched)
    except Exception as exc:
        logger.exception("Consumers agent run raised an unexpected exception")
        call_error = f"Service-consumers agent crashed unexpectedly: {exc}"
        return {
            **state,
            "subagent_answer": await _naturalize_fallback(
                chatmodel, context, "consumers", "failed", [call_error], None
            ),
            "call_error": call_error,
        }

    call_error: str | None = None
    if result.mode == "answered" and result.answer:
        answer = await _adapt_language(chatmodel, context, result.answer)
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        error_detail = (": " + "; ".join(result.errors)) if result.errors else ""
        call_error = (
            f"Elasticsearch MCP access failed{error_detail}. "
            "Kibana fallback query provided instead."
        )
        logger.warning("Consumers agent kibana_fallback – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "consumers", "kibana_fallback",
            result.errors, result.kibana_payload,
        )
    else:
        call_error = (
            "; ".join(result.errors)
            if result.errors
            else "Service-consumers agent failed with unknown error."
        )
        logger.error("Consumers agent failed – %s", call_error)
        answer = await _naturalize_fallback(
            chatmodel, context, "consumers", "failed",
            result.errors, None,
        )

    return {**state, "subagent_answer": answer, "call_error": call_error}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Direct LLM answer (general Illumio knowledge)
# ─────────────────────────────────────────────────────────────────────────────

async def answer_directly_node(
    state: IllumioExpertState, chatmodel: ChatOpenAI
) -> IllumioExpertState:
    """Answer a general Illumio question directly from the LLM's domain knowledge."""
    logger.info("Node: answer_directly")
    messages = state.get("messages", [])

    # Feed the full conversation to the LLM with the expert system prompt
    llm_messages: list[BaseMessage] = [SystemMessage(content=ILLUMIO_EXPERT_SYSTEM_PROMPT)]
    llm_messages.extend(messages)

    try:
        response: AIMessage = await chatmodel.ainvoke(llm_messages)
        return {**state, "subagent_answer": response.content, "call_error": None}
    except Exception as exc:
        logger.exception("Direct LLM answer failed")
        call_error = f"Direct answer LLM call failed: {exc}"
        return {
            **state,
            "subagent_answer": "Je suis désolé, je ne suis pas en mesure de traiter votre demande pour le moment.",
            "call_error": call_error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node: Compose final response
# ─────────────────────────────────────────────────────────────────────────────

async def compose_response_node(state: IllumioExpertState) -> IllumioExpertState:
    """Append the answer as an AIMessage to the conversation history."""
    logger.info("Node: compose_response")
    answer = state.get("subagent_answer") or "I was unable to process your request."
    # Returning {"messages": [...]} causes add_messages to *append* the new
    # AIMessage to the existing list rather than replacing it.
    return {**state, "messages": [AIMessage(content=answer)]}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_by_intent(state: IllumioExpertState) -> str:
    return state.get("intent", "general")


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_illumio_expert_graph(
    chatmodel: ChatOpenAI,
    traffic_agent: IllumioTrafficAgent,
    blocked_agent: IllumioBlockedAgent,
    consumers_agent: IllumioConsumersAgent,
) -> Any:
    """Construct and compile the Illumio Expert conversational StateGraph."""

    # ── Close over dependencies for each node ──

    async def _classify_intent(s: IllumioExpertState) -> IllumioExpertState:
        return await classify_intent_node(s, chatmodel)

    async def _invoke_traffic(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_traffic_agent_node(s, traffic_agent, chatmodel)

    async def _invoke_blocked(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_blocked_agent_node(s, blocked_agent, chatmodel)

    async def _invoke_consumers(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_consumers_agent_node(s, consumers_agent, chatmodel)

    async def _answer_directly(s: IllumioExpertState) -> IllumioExpertState:
        return await answer_directly_node(s, chatmodel)

    # ── Build the graph ──
    graph = StateGraph(IllumioExpertState)

    graph.add_node("classify_intent",  _classify_intent)
    graph.add_node("invoke_traffic",   _invoke_traffic)
    graph.add_node("invoke_blocked",   _invoke_blocked)
    graph.add_node("invoke_consumers", _invoke_consumers)
    graph.add_node("answer_directly",  _answer_directly)
    graph.add_node("compose_response", compose_response_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "traffic":   "invoke_traffic",
            "blocked":   "invoke_blocked",
            "consumers": "invoke_consumers",
            "general":   "answer_directly",
        },
    )

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

    This is the single entry point for all Illumio-related questions.
    It internally delegates to specialised sub-agents for data queries or
    answers directly from domain knowledge.

    Single-turn usage::

        agent = IllumioExpertAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        answer = await agent.chat("Qu'est-ce que le mode d'enforcement Selective ?")

    Multi-turn usage::

        agent = IllumioExpertAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
        history: list[BaseMessage] = []

        history, answer = await agent.chat_with_history(
            "Est-ce que j'ai du trafic dev→prod pour AP12345 ?", history
        )
        history, answer = await agent.chat_with_history(
            "Et les connexion bloqués pour la même application ?", history
        )
    """

    def __init__(self, mcp_client: Any, chatmodel: ChatOpenAI) -> None:
        self.chatmodel       = chatmodel
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
        """Single-turn: process one message and return the answer string."""
        initial: IllumioExpertState = {
            "messages":        [HumanMessage(content=user_message)],
            "intent":          None,
            "subagent_answer": None,
            "error":           None,
            "call_error":      None,
            "ap_code":         None,
            "hostname":        None,
            "date_range":      None,
        }
        final = await self.graph.ainvoke(initial)
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content
        return "I was unable to process your request."

    async def chat_with_history(
        self,
        user_message: str,
        history: list[BaseMessage],
        ap_code:    str | None = None,
        hostname:   str | None = None,
        date_range: str | None = None,
    ) -> tuple[list[BaseMessage], str, str | None, str | None, str | None, str | None]:
        """Multi-turn: process one message with existing history.

        Accepts and returns ``ap_code`` / ``hostname`` / ``date_range`` so
        callers can persist them across turns without re-parsing the message
        themselves.

        Returns ``(updated_messages, answer, ap_code, hostname, date_range, call_error)``.
        ``call_error`` is ``None`` on success, or a human-readable description
        of what went wrong when a sub-agent fell back to Kibana or failed.
        """
        initial: IllumioExpertState = {
            "messages":        [*history, HumanMessage(content=user_message)],
            "intent":          None,
            "subagent_answer": None,
            "error":           None,
            "call_error":      None,
            "ap_code":         ap_code,
            "hostname":        hostname,
            "date_range":      date_range,
        }
        final = await self.graph.ainvoke(initial)
        final_messages = final.get("messages", initial["messages"])

        answer = "I was unable to process your request."
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        return (
            final_messages,
            answer,
            final.get("ap_code"),
            final.get("hostname"),
            final.get("date_range"),
            final.get("call_error"),
        )
