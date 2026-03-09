"""
LangGraph Dev Server Entry Point – Illumio Expert Agent
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from illumio_expert_agent import build_illumio_expert_graph

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LLM  (intent classification only – specialist agents reuse the same model)
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
# 2. MCP client  (shared by all sub-agents)
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

logger.info("Illumio Expert Agent using shared MCP client via mcp_server.py")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build and expose the compiled graph
# ─────────────────────────────────────────────────────────────────────────────

graph = build_illumio_expert_graph(chatmodel=chatmodel, mcp_client=MCP_CLIENT)
