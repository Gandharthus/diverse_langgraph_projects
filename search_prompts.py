"""
Prompt templates for the Elasticsearch Search Agent.

All LLM-facing text lives here so you can iterate on wording without
touching node logic.  Templates use ``str.format()`` placeholders –
the calling code fills them in at runtime.
"""

from __future__ import annotations

import textwrap


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_index_list(indices: list[dict]) -> str:
    """Format resolved indices into a readable markdown list."""
    if not indices:
        return "(no indices found)"
    lines = []
    for idx in indices:
        name = idx.get("name", idx.get("index", "unknown"))
        desc = idx.get("description", "")
        if desc:
            lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(f"- **{name}**")
    return "\n".join(lines)


def format_mapping_fields(mapping: dict, max_fields: int = 80) -> str:
    """Flatten an ES mapping into a readable field list with types."""
    fields: list[str] = []

    def _walk(props: dict, prefix: str = "") -> None:
        for name, conf in props.items():
            full = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            ftype = conf.get("type", "object")
            fields.append(f"  {full} ({ftype})")
            if "properties" in conf:
                _walk(conf["properties"], full)

    if "properties" in mapping:
        _walk(mapping["properties"])
    elif "mappings" in mapping:
        # Handle full index mapping response
        m = mapping["mappings"]
        if "properties" in m:
            _walk(m["properties"])

    if len(fields) > max_fields:
        return "\n".join(fields[:max_fields]) + f"\n  ... ({len(fields) - max_fields} more fields)"
    return "\n".join(fields) if fields else "(no fields found)"


# ──────────────────────────────────────────────────────────────────────────────
# System prompt – intent understanding
# ──────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Elasticsearch search analyst at a large enterprise.
    Your job is to understand natural-language search requests and translate
    them into a structured search plan.

    Given the user's request, extract:
    1. **index_hints** – keywords or patterns that suggest which index/indices
       to search (e.g. "firewall logs" → "firewall*", "nginx" → "nginx*").
       Output a list of glob patterns.
    2. **time_range** – any time constraints mentioned (e.g. "last 24 hours",
       "yesterday", "since January"). Output as {{gte, lte}} in ES date math
       format (e.g. "now-24h", "now-1d/d"). Use null if not mentioned.
    3. **search_terms** – the key entities, values, IPs, error codes, user names
       etc. the user is looking for.
    4. **intent** – one of: "find_logs", "count", "aggregate", "unique_values",
       "top_n", "time_series", "exists_check".
    5. **agg_field** – if intent is aggregate/top_n/unique_values, which field(s)
       to aggregate on (best guess, will be refined after mapping).
    6. **size** – how many documents to return. Default 20 for find_logs, 0 for
       aggregations.

    Respond ONLY with valid JSON, no markdown fences:
    {{
      "index_hints": ["pattern1*", "pattern2*"],
      "time_range": {{"gte": "now-24h", "lte": "now"}} or null,
      "search_terms": ["term1", "term2"],
      "intent": "find_logs",
      "agg_field": null,
      "size": 20
    }}
""")


# ──────────────────────────────────────────────────────────────────────────────
# System prompt – DSL query generation
# ──────────────────────────────────────────────────────────────────────────────

DSL_GENERATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Elasticsearch DSL query engineer at a large enterprise.
    You write precise, efficient queries that follow company guardrails.

    ## Company guardrails (MUST follow)

    1. **Exclusively use Query DSL** – never use SQL, EQL, or runtime scripts.
    2. **No `script` usage** – script queries, script fields, and scripted_metric
       aggregations are forbidden.
    3. **Aggregation nesting** – maximum 3 levels of nested aggregations.
       Flatter is better.
    4. **Bucket limits** – every terms/histogram aggregation MUST have
       `"size"` set to a value <= 1000. Default to 20 for terms aggs.
    5. **Performance** – prefer `filter` context over `query` context when
       you don't need scoring. Use `bool.filter` for exact matches, ranges,
       and terms.
    6. **Date filtering** – always use `range` on `@timestamp` when a time
       window is specified. Place it in `bool.filter`.

    ## Input you'll receive
    - The user's original request.
    - A structured search plan (intent, terms, time range).
    - The target index name.
    - The index mapping (field names and types).

    ## Output format
    Return ONLY valid JSON – the body you'd send to `POST /<index>/_search`.
    No markdown fences. No commentary outside the JSON.
    Include a top-level `"_comment"` field (string) with a 1-sentence
    explanation of what the query does.
""")


# ──────────────────────────────────────────────────────────────────────────────
# User message builders
# ──────────────────────────────────────────────────────────────────────────────

def build_dsl_user_message(
    user_request: str,
    search_plan: dict,
    index_name: str,
    mapping_summary: str,
) -> str:
    """Build the user message for DSL query generation."""
    return textwrap.dedent(f"""\
        ## User request
        {user_request}

        ## Search plan
        - Intent: {search_plan.get('intent', 'find_logs')}
        - Search terms: {search_plan.get('search_terms', [])}
        - Time range: {search_plan.get('time_range', 'none')}
        - Aggregation field: {search_plan.get('agg_field', 'none')}
        - Result size: {search_plan.get('size', 20)}

        ## Target index
        {index_name}

        ## Index mapping (available fields)
        {mapping_summary}
    """)


def build_query_fix_prompt(
    query_json_str: str,
    validation_errors: list[str],
    original_request: str,
) -> str:
    """Build the prompt for fixing a query that failed validation."""
    errors_text = "\n".join(f"{i+1}. {e}" for i, e in enumerate(validation_errors))
    return textwrap.dedent(f"""\
        The following Elasticsearch DSL query failed validation.
        Fix ONLY the listed issues. Return corrected JSON, no markdown fences.

        ## Original user request
        {original_request}

        ## Current query
        ```json
        {query_json_str}
        ```

        ## Validation errors
        {errors_text}

        ## Reminder
        - No scripts. No SQL. Pure DSL only.
        - Max 3 aggregation nesting levels.
        - Every terms/histogram agg needs "size" <= 1000.
        - Use bool.filter for non-scoring clauses.
    """)


# ──────────────────────────────────────────────────────────────────────────────
# Kibana Dev Tools fallback
# ──────────────────────────────────────────────────────────────────────────────

def format_kibana_payload(index: str, query: dict) -> str:
    """Format a query as a copy-pasteable Kibana Dev Tools snippet."""
    import json
    body = json.dumps(query, indent=2)
    return f"GET /{index}/_search\n{body}"
