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
    response: AIMessage = await chatmodel.ainvoke(llm_messages)
    result = _extract_json(response.content)

    intent = "general"  # safe default
    if result and isinstance(result.get("intent"), str):
        raw = result["intent"]
        if raw in ("traffic", "blocked", "consumers", "general"):
            intent = raw

    logger.info("Classified intent: %s", intent)
    return {**state, "intent": intent, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Nodes: Sub-agent invocations
# ─────────────────────────────────────────────────────────────────────────────

async def invoke_traffic_agent_node(
    state: IllumioExpertState,
    traffic_agent: IllumioTrafficAgent,
) -> IllumioExpertState:
    """Delegate to the IllumioTrafficAgent sub-agent."""
    logger.info("Node: invoke_traffic_agent")
    user_request = _last_human_message(state.get("messages", []))

    result = await traffic_agent.run(user_request)

    if result.mode == "answered" and result.answer:
        answer = result.answer
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        answer = (
            "Je n'ai pas pu accéder directement à Elasticsearch. "
            "Voici la requête Kibana Dev Tools pour analyser le trafic inter-environnements :\n\n"
            f"```\n{result.kibana_payload}\n```"
        )
    else:
        errors = "; ".join(result.errors) if result.errors else "Erreur inconnue."
        answer = (
            f"Je n'ai pas pu répondre à votre question sur le trafic "
            f"inter-environnements : {errors}"
        )

    return {**state, "subagent_answer": answer}


async def invoke_blocked_agent_node(
    state: IllumioExpertState,
    blocked_agent: IllumioBlockedAgent,
) -> IllumioExpertState:
    """Delegate to the IllumioBlockedAgent sub-agent."""
    logger.info("Node: invoke_blocked_agent")
    user_request = _last_human_message(state.get("messages", []))

    result = await blocked_agent.run(user_request)

    if result.mode == "answered" and result.answer:
        answer = result.answer
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        answer = (
            "Je n'ai pas pu accéder directement à Elasticsearch. "
            "Voici la requête Kibana Dev Tools pour les flux bloqués :\n\n"
            f"```\n{result.kibana_payload}\n```"
        )
    else:
        errors = "; ".join(result.errors) if result.errors else "Erreur inconnue."
        answer = f"Je n'ai pas pu répondre à votre question sur les flux bloqués : {errors}"

    return {**state, "subagent_answer": answer}


async def invoke_consumers_agent_node(
    state: IllumioExpertState,
    consumers_agent: IllumioConsumersAgent,
) -> IllumioExpertState:
    """Delegate to the IllumioConsumersAgent sub-agent."""
    logger.info("Node: invoke_consumers_agent")
    user_request = _last_human_message(state.get("messages", []))

    result = await consumers_agent.run(user_request)

    if result.mode == "answered" and result.answer:
        answer = result.answer
    elif result.mode == "kibana_fallback" and result.kibana_payload:
        answer = (
            "Je n'ai pas pu accéder directement à Elasticsearch. "
            "Voici la requête Kibana Dev Tools pour les consommateurs du service :\n\n"
            f"```\n{result.kibana_payload}\n```"
        )
    else:
        errors = "; ".join(result.errors) if result.errors else "Erreur inconnue."
        answer = (
            f"Je n'ai pas pu répondre à votre question sur les consommateurs "
            f"du service : {errors}"
        )

    return {**state, "subagent_answer": answer}


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

    response: AIMessage = await chatmodel.ainvoke(llm_messages)
    return {**state, "subagent_answer": response.content}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Compose final response
# ─────────────────────────────────────────────────────────────────────────────

async def compose_response_node(state: IllumioExpertState) -> IllumioExpertState:
    """Append the answer as an AIMessage to the conversation history."""
    logger.info("Node: compose_response")
    answer = state.get("subagent_answer") or "Je n'ai pas pu traiter votre demande."
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
        return await invoke_traffic_agent_node(s, traffic_agent)

    async def _invoke_blocked(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_blocked_agent_node(s, blocked_agent)

    async def _invoke_consumers(s: IllumioExpertState) -> IllumioExpertState:
        return await invoke_consumers_agent_node(s, consumers_agent)

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
            "Et les flux bloqués pour la même application ?", history
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
            "messages":       [HumanMessage(content=user_message)],
            "intent":         None,
            "subagent_answer": None,
            "error":          None,
        }
        final = await self.graph.ainvoke(initial)
        for msg in reversed(final.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content
        return "Je n'ai pas pu traiter votre demande."

    async def chat_with_history(
        self,
        user_message: str,
        history: list[BaseMessage],
    ) -> tuple[list[BaseMessage], str]:
        """Multi-turn: process one message with existing history.

        Returns the updated message list and the agent's answer string.
        """
        initial: IllumioExpertState = {
            "messages":       [*history, HumanMessage(content=user_message)],
            "intent":         None,
            "subagent_answer": None,
            "error":          None,
        }
        final = await self.graph.ainvoke(initial)
        final_messages = final.get("messages", initial["messages"])

        answer = "Je n'ai pas pu traiter votre demande."
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        return final_messages, answer
