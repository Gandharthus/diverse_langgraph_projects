"""
LangGraph Dev Server Entry Point – Search Agent
=================================================

This module exposes the compiled ``graph`` object that ``langgraph dev``
expects for the search agent.

The ``langgraph.json`` config references this file as:

    "graphs": { ..., "search_agent": "search_graph.py:graph" }
"""

from __future__ import annotations

import os
import logging

from langchain_openai import ChatOpenAI
from search_agent import build_search_agent_graph

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configure the LLM (same provider as pipeline wizard)
# ──────────────────────────────────────────────────────────────────────────────
_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

chatmodel = ChatOpenAI(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=_groq_api_key,
    base_url="https://api.groq.com/openai/v1",
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Configure the MCP client (search-specific tools)
# ──────────────────────────────────────────────────────────────────────────────
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_CLIENT = MultiServerMCPClient(
    {
        "elasticsearch": {
            "command": "python",
            "args": ["mcp_server.py"],
            "transport": "stdio",
        }
    }
)

logger.info("Search agent using MCP client via mcp_server.py")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Build and expose the compiled graph
# ──────────────────────────────────────────────────────────────────────────────
graph = build_search_agent_graph(chatmodel=chatmodel, mcp_client=MCP_CLIENT)
