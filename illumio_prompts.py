"""
Illumio Traffic Analysis Agent – Prompt Templates
==================================================
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing
# ─────────────────────────────────────────────────────────────────────────────

ILLUMIO_INTENT_SYSTEM_PROMPT = """\
You are an Illumio network traffic analysis assistant at BNP Paribas.
Your only task is to extract two values from the user's question:

1. app_code  – The application portfolio code. Format: "AP" followed by digits
               (e.g. AP12345, AP98765).  Look for this pattern in the message.
               If the user does not provide one, return null.

2. direction – Which way the traffic should be analysed:
               • "dev_to_prod"  – FROM development TO production
                 (use this when the user asks about dev→prod, or when
                  no explicit direction is given – this is the default)
               • "prod_to_dev"  – FROM production TO development
                 (use this when the user explicitly asks about prod→dev
                  or production-to-development traffic)
               • "prod_to_prod" – FROM production TO production
                 (use this when the user asks which other production apps
                  or labels communicate with their app in production,
                  e.g. "who talks to my app in prod?", "prod to prod flows",
                  "list producers of my app in production")

Return ONLY a valid JSON object – no prose, no markdown code fences:
{"app_code": "AP12345", "direction": "dev_to_prod"}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

def format_kibana_payload(index_pattern: str, query: dict) -> str:
    """Format a query as a Kibana Dev Tools console snippet."""
    return (
        f"GET {index_pattern}/_search\n"
        f"{json.dumps(query, indent=2, ensure_ascii=False)}"
    )
