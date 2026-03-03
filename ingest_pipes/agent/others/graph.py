"""
LangGraph Dev Server Entry Point
=================================

This module exposes the compiled ``graph`` object that ``langgraph dev``
expects.  It wires up the chatmodel and MCP client from environment
variables so the graph can be constructed at import time.

The ``langgraph.json`` config references this file as:

    "graphs": { "pipeline_wizard": "graph.py:graph" }

When you run ``langgraph dev``, it imports this module, picks up the
``graph`` variable, and serves it with Studio.
"""

from __future__ import annotations

import os
import logging

from langchain_openai import ChatOpenAI
from ingest_pipes.agent.pipeline_wizard import build_pipeline_wizard_graph

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configure the LLM (Switched to Groq)
# ──────────────────────────────────────────────────────────────────────────────
_groq_api_key = os.environ.get("GROQ_API_KEY")
if not _groq_api_key:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

chatmodel = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=_groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Configure the MCP client
# ──────────────────────────────────────────────────────────────────────────────
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_CLIENT = MultiServerMCPClient(
    {
        "elasticsearch": {
            "command": "python",
            "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")],
            "transport": "stdio",
        }
    }
)

logger.info("Using real MCP client via mcp_server.py")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Build and expose the compiled graph
# ──────────────────────────────────────────────────────────────────────────────
# This is the variable that langgraph.json references via "graph.py:graph"
graph = build_pipeline_wizard_graph(chatmodel=chatmodel, mcp_client=MCP_CLIENT)
