"""
LangGraph Dev Server Entry Point – Illumio Query Builder Sub-Agent
==================================================================

This module exposes the compiled ``graph`` object that ``langgraph dev``
expects for the Illumio QueryBuilder sub-agent.

The ``langgraph.json`` config references this file as:

    "graphs": { ..., "illumio_query_builder": "./illumio_query_builder_graph.py:graph" }

Unlike the other Illumio agents, this sub-agent requires **no LLM** — it
accepts structured parameters directly and builds the DSL deterministically.

Input state (set by the caller or LangGraph Studio)
----------------------------------------------------
- ``time_range``      – optional time filter dict (gte/lte/gt/lt keys)
- ``policy_decision`` – optional ``"Blocked"`` | ``"Allowed"``
- ``source_app``      – optional prefix for ``illumio.source.labels.app``
- ``destination_app`` – optional prefix for ``illumio.destination.labels.app``
- ``env``             – optional ``"E_PROD"`` | ``"HPROD"``
- ``aggs``            – optional aggregation block (terms only, depth ≤ 2, size ≤ 20)
"""

from __future__ import annotations

import logging

from langchain_mcp_adapters.client import MultiServerMCPClient

from illumio_query_builder_agent import build_illumio_query_builder_graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MCP client
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

logger.info("Illumio QueryBuilder sub-agent using MCP client via mcp_server.py")


# ─────────────────────────────────────────────────────────────────────────────
# Build and expose the compiled graph
# ─────────────────────────────────────────────────────────────────────────────

graph = build_illumio_query_builder_graph(mcp_client=MCP_CLIENT)
