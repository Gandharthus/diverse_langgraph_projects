"""
Illumio Expert Agent – Prompt Templates
========================================

The intent system prompt is built **dynamically** from the QueryBuilder
sub-agent's ``describe_skills()`` output, so the expert's knowledge of that
tool is always in sync with its actual interface.
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_params(params: dict[str, Any]) -> str:
    """Render a parameters dict as a readable bullet list."""
    lines: list[str] = []
    for name, info in params.items():
        req  = info.get("required", False)
        typ  = info.get("type", "any")
        desc = info.get("description", "")
        enum_vals: list[str] | None = info.get("enum")

        if "one of" in str(req):
            req_label = "one-of-required"
        elif req is True or req == "required":
            req_label = "required"
        else:
            req_label = "optional"

        enum_str = (
            f"  Allowed values: {', '.join(repr(v) for v in enum_vals)}\n"
            if enum_vals
            else ""
        )
        lines.append(
            f"  • {name}  ({typ}, {req_label})\n"
            f"    {desc}\n"
            f"{enum_str}"
        )
    return "\n".join(lines)


def _fmt_constraints(constraints: list[str]) -> str:
    return "\n".join(f"  • {c}" for c in constraints)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic intent system prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_expert_intent_prompt(qb_skill: dict[str, Any]) -> str:
    """
    Build the expert agent's intent-classification system prompt.

    The query-builder section is generated from *qb_skill*, the skill
    descriptor returned by ``IllumioQueryBuilderSubAgent.describe_skills()[0]``.
    This keeps the expert's knowledge of the sub-agent's interface in sync
    automatically.

    Parameters
    ----------
    qb_skill:
        A single skill descriptor dict (name, description, parameters,
        returns, constraints).

    Returns
    -------
    str
        The full system prompt string.
    """
    qb_params_doc      = _fmt_params(qb_skill.get("parameters", {}))
    qb_constraints_doc = _fmt_constraints(qb_skill.get("constraints", []))
    qb_description     = qb_skill.get("description", "")

    return f"""\
You are an Illumio network security expert at BNP Paribas.
You orchestrate a set of specialised analysis agents and route every user
question to the most appropriate one.

Your ONLY task is to analyse the user's question and return a single JSON
object that identifies the intent and, when needed, extracted parameters.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE INTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── 1. "traffic_analysis" ────────────────────────────────────────────────────
Use when the user asks about cross-environment traffic flows (dev ↔ prod).
Typical keywords: traffic between environments, dev to prod, prod to dev,
flux entre environnements.

Return:
{{"intent": "traffic_analysis"}}

── 2. "blocked_flows" ───────────────────────────────────────────────────────
Use when the user asks about blocked or denied traffic to/from a server or
application.
Typical keywords: blocked flows, flux bloqués, denied traffic, rejected.

Return:
{{"intent": "blocked_flows"}}

── 3. "consumers" ───────────────────────────────────────────────────────────
Use when the user asks which applications or services are consuming /
calling a given application.
Typical keywords: consumers, consommateurs, which apps call my service,
applications consuming, qui consomme mon service.

Return:
{{"intent": "consumers"}}

── 4. "query_builder" ───────────────────────────────────────────────────────
{qb_description}

Use this intent when the user explicitly provides structured filter criteria
such as: a specific app name (source or destination), policy decision
(Blocked / Allowed), time range, environment (E_PROD / HPROD), or asks for
custom aggregations.  This is the right choice when no single specialist
agent above covers the request.

You must extract the following parameters from the user's question.
Extract only what the user explicitly states; use null for everything else.

Parameters:
{qb_params_doc}
Constraints enforced by the sub-agent (do NOT invent values outside these):
{qb_constraints_doc}

Return:
{{
  "intent": "query_builder",
  "qb_params": {{
    "source_app":       <prefix string or null>,
    "destination_app":  <prefix string or null>,
    "policy_decision":  <"Blocked" | "Allowed" | null>,
    "time_range":       <{{"gte": "...", "lte": "..."}} or null>,
    "env":              <"E_PROD" | "HPROD" | null>,
    "aggs":             <terms-only aggregation dict or null>
  }}
}}

── 5. "unknown" ─────────────────────────────────────────────────────────────
Use when the question is unrelated to Illumio network analysis.

Return:
{{"intent": "unknown", "reason": "<brief explanation>"}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Return ONLY valid JSON — no prose, no markdown code fences.
• For "query_builder": extract values the user explicitly stated.
  Do NOT invent app names, time ranges, or aggregation fields.
• "traffic_analysis", "blocked_flows", "consumers" never need qb_params.
• When multiple intents could match, prefer the most specific specialist
  (e.g. "blocked_flows" over "query_builder" for a blocked-traffic question).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Unknown-intent answer
# ─────────────────────────────────────────────────────────────────────────────

def format_unknown_answer(reason: str) -> str:
    """Human-readable reply for an unrecognised or out-of-scope question."""
    return (
        "Je ne suis pas en mesure de répondre à cette question avec les "
        "agents Illumio disponibles.\n\n"
        f"Raison : {reason}\n\n"
        "Les agents disponibles couvrent :\n"
        "  • Analyse du trafic inter-environnements (dev ↔ prod)\n"
        "  • Détection des flux bloqués / refusés\n"
        "  • Identification des consommateurs de service\n"
        "  • Requêtes Elasticsearch structurées (QueryBuilder)"
    )
