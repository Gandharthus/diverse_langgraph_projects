"""
Illumio Blocked Traffic Agent – Prompt Templates
=================================================
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
# Intent parsing
# ─────────────────────────────────────────────────────────────────────────────

ILLUMIO_BLOCKED_INTENT_SYSTEM_PROMPT = """\
You are an Illumio network security analyst at BNP Paribas.
Your task is to extract information from the user's question about BLOCKED network traffic flows.

Extract the following five values:

1. target – The exact hostname or application code the user is asking about.
   - Hostnames look like: "server01", "web-prod-01", "10.0.0.1"
   - Application codes look like: "AP12345", "AP98765" (format: "AP" followed by digits)
   - Return exactly as stated by the user.
   - If the user does not provide one, return null.

2. target_type – Whether the target is a hostname or an application code:
   - "hostname" – if it looks like a server name, FQDN, or IP address
   - "app"      – if it matches the application portfolio code pattern (AP + digits)
   - Default to "hostname" if unclear.

3. direction – Which direction of traffic to analyse:
   - "inbound"  – only traffic DESTINED TO the target ("vers mon serveur/application")
   - "outbound" – only traffic ORIGINATING FROM the target ("depuis mon serveur/application")
   - "both"     – both inbound AND outbound (use when the user says "vers ou depuis",
                   or does not specify a direction – this is the default)

4. time_from – Start of the time window as an Elasticsearch date-math string.
   Map natural-language expressions to the following values:
   - "maintenant"/"now"                               → null  (no filter)
   - "dernière heure" / "last hour" / "1h"            → "now-1h"
   - "deux dernières heures" / "2 heures" / "2h"      → "now-2h"
   - "6 dernières heures" / "6h"                      → "now-6h"
   - "12 dernières heures" / "12h"                    → "now-12h"
   - "dernières 24 heures" / "hier" / "1 jour" / "1d" → "now-24h"
   - "cette semaine" / "7 jours" / "7d"               → "now-7d"
   - "ce mois" / "30 jours" / "30d"                   → "now-30d"
   - If no time window is mentioned                    → null

5. time_to – End of the time window:
   - Set to "now" whenever time_from is not null.
   - Set to null when time_from is null.

Return ONLY a valid JSON object – no prose, no markdown code fences:
{"target": "web-prod-01", "target_type": "hostname", "direction": "both", "time_from": "now-2h", "time_to": "now"}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Kibana fallback
# ─────────────────────────────────────────────────────────────────────────────

def format_kibana_blocked_payload(
    index_pattern: str,
    inbound_query: dict | None,
    outbound_query: dict | None,
) -> str:
    """Format one or two blocked-flow queries as Kibana Dev Tools console snippets."""
    parts = []
    if inbound_query:
        parts.append(
            f"# --- Flux BLOQUES vers la cible (inbound) ---\n"
            f"GET {index_pattern}/_search\n"
            f"{json.dumps(inbound_query, indent=2, ensure_ascii=False)}"
        )
    if outbound_query:
        parts.append(
            f"# --- Flux BLOQUES depuis la cible (outbound) ---\n"
            f"GET {index_pattern}/_search\n"
            f"{json.dumps(outbound_query, indent=2, ensure_ascii=False)}"
        )
    return "\n\n".join(parts)
