"""
Illumio Service Consumers Agent – Prompt Templates
===================================================
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing
# ─────────────────────────────────────────────────────────────────────────────

ILLUMIO_CONSUMERS_INTENT_SYSTEM_PROMPT = """\
You are an Illumio network traffic analysis assistant at BNP Paribas.
Your only task is to extract the application portfolio code from the user's question.

app_code – The application portfolio code. Format: "AP" followed by digits
           (e.g. AP12345, AP98765). Look for this pattern in the message.
           If the user does not provide one, return null.

The user is asking which applications are consuming (sending traffic to) their
service or application. You only need to identify the target application code.

Return ONLY a valid JSON object – no prose, no markdown code fences:
{"app_code": "AP12345"}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

def format_kibana_consumers_payload(index_pattern: str, query: dict) -> str:
    """Format a consumers query as a Kibana Dev Tools console snippet."""
    return (
        f"GET {index_pattern}/_search\n"
        f"{json.dumps(query, indent=2, ensure_ascii=False)}"
    )
