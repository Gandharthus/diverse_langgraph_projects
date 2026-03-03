"""
LangGraph Dev Server Entry Point – Illumio Expert Domain Agent
==============================================================

This module exposes the compiled ``graph`` object that ``langgraph dev``
expects for the Illumio expert conversational agent.

The ``langgraph.json`` config references this file as:

    "graphs": { ..., "illumio_expert_agent": "./illumio_expert_graph.py:graph" }

The expert agent is the single entry point for all Illumio-related questions.
It classifies the user's intent and delegates to the appropriate specialised
sub-agent, or answers general Illumio knowledge questions directly.
"""

from __future__ import annotations

import logging
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from illumio_agent import IllumioTrafficAgent
from illumio_blocked_agent import IllumioBlockedAgent
from illumio_consumers_agent import IllumioConsumersAgent
from illumio_expert_agent import build_illumio_expert_graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM
# ─────────────────────────────────────────────────────────────────────────────

_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

chatmodel = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=_groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MCP client (shared by all sub-agents)
# ─────────────────────────────────────────────────────────────────────────────

MCP_CLIENT = MultiServerMCPClient(
    {
        "elasticsearch": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio",
        }
    }
)

logger.info("Illumio Expert agent: initialising sub-agents...")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sub-agents
# ─────────────────────────────────────────────────────────────────────────────

_traffic_agent   = IllumioTrafficAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
_blocked_agent   = IllumioBlockedAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)
_consumers_agent = IllumioConsumersAgent(mcp_client=MCP_CLIENT, chatmodel=chatmodel)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build and expose the compiled graph
# ─────────────────────────────────────────────────────────────────────────────

graph = build_illumio_expert_graph(
    chatmodel=chatmodel,
    traffic_agent=_traffic_agent,
    blocked_agent=_blocked_agent,
    consumers_agent=_consumers_agent,
)

logger.info("Illumio Expert agent graph ready.")
