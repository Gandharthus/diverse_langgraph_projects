"""
Prompt templates for the Pipeline Wizard.

All LLM-facing text lives here so you can iterate on wording without
touching node logic.  Templates use ``str.format()`` placeholders
(``{forbidden}``, ``{pipeline_json}``, etc.) – the calling code fills
them in at runtime.
"""

from __future__ import annotations

import textwrap


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_errors_for_prompt(errors: list[str]) -> str:
    """Format a list of error strings into a numbered markdown list."""
    if not errors:
        return "(none)"
    return "\n".join(f"{i + 1}. {e}" for i, e in enumerate(errors))


def build_generation_system_prompt(forbidden_processors: set[str]) -> str:
    """
    Build the full system prompt for the *initial* pipeline generation call.

    The ``{forbidden}`` placeholder is filled from the live config so the
    prompt always reflects the current policy.
    """
    return _GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        forbidden=", ".join(f"`{p}`" for p in sorted(forbidden_processors)),
    )


def build_guardrail_fix_prompt(
    pipeline_json_str: str,
    errors: list[str],
    forbidden_processors: set[str],
) -> str:
    """Assemble the prompt sent to the LLM to fix guardrail violations."""
    return _GUARDRAIL_FIX_TEMPLATE.format(
        pipeline_json=pipeline_json_str,
        errors=format_errors_for_prompt(errors),
        forbidden=", ".join(f"`{p}`" for p in sorted(forbidden_processors)),
    )


def build_simulation_fix_prompt(
    pipeline_json_str: str,
    errors: list[str],
    samples_json_str: str,
    forbidden_processors: set[str],
) -> str:
    """Assemble the prompt sent to the LLM to fix simulation failures."""
    return _SIMULATION_FIX_TEMPLATE.format(
        pipeline_json=pipeline_json_str,
        errors=format_errors_for_prompt(errors),
        samples=samples_json_str,
        forbidden=", ".join(f"`{p}`" for p in sorted(forbidden_processors)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Raw templates (private – use the builder functions above)
# ──────────────────────────────────────────────────────────────────────────────

_GENERATION_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert Elasticsearch ingest pipeline engineer. Your job is to
    produce a complete, production-ready ingest pipeline in JSON that will
    parse, transform, and enrich raw log lines.

    ## Scope and Limitation
    **Strictly focus on the provided log samples.**
    Do NOT attempt to infer, guess, or handle other log formats or edge cases not present in the samples.
    The pipeline must be tailored *specifically* and *exclusively* to ingest the samples provided.
    Do not add logic for hypothetical variations.

    ## Company best practices (MUST follow)

    1. **Prefer `dissect` over `grok`.**
       `dissect` is deterministic and faster. Only fall back to `grok` when
       the log format genuinely requires regex (variable-length optional
       fields, alternation, etc.). Justify every use of `grok` in a comment
       inside the processor's `description` field.
       **CRITICAL**: When using `dissect`, the pattern MUST match the entire log line structure.
       - Every delimiter (spaces, brackets, quotes) in the log must be present in the pattern.
       - Do not skip fields structure. Capture all parts of the message.

    2. **Flexible Processor Ordering.**
       While it is generally good practice to parse, then transform, then enrich,
       you MAY order processors as needed to achieve the desired outcome.
       For example, you might use `gsub` to clean a message before `dissect`,
       or `rename`/`remove` fields at the end of the pipeline.

    3. **ECS Compliance.**
       Output field names should conform to the Elastic Common Schema (ECS) where possible.
       Use `labels.*` for custom fields that do not fit into standard ECS fields.
       (e.g., `labels.duration_ms`).
       **CRITICAL**: Map ALL available information.
       - If the log contains a log level, map it to `log.level`.
       - If the log contains a user agent, map it to `user_agent.original`.
       - If the log contains a service name, map it to `service.name` (or `labels.service`).

    4. **Forbidden processors.**
       Do NOT use any of: {forbidden}.

    5. **Error handling.**
       Processors MAY include an `on_failure` clause if necessary, but it is not mandatory.

    6. **`@timestamp`.**
       Always produce a `@timestamp` field using the `date` processor.

    ## Output format

    Return **only** a valid JSON object with the top-level key
    `"processors"` containing an array of processor objects. Do NOT wrap
    the JSON in markdown fences. Do NOT include commentary outside the JSON.

    Example skeleton:

    {{
      "processors": [
        {{
          "dissect": {{
            "field": "message",
            "pattern": "...",
            "description": "Parse the syslog header"
          }}
        }},
        {{
          "date": {{
            "field": "timestamp_str",
            "formats": ["ISO8601"],
            "target_field": "@timestamp",
            "description": "Set @timestamp from parsed timestamp"
          }}
        }}
      ]
    }}
""")


_GUARDRAIL_FIX_TEMPLATE = textwrap.dedent("""\
    The following Elasticsearch ingest pipeline failed code-based guardrail
    validation. Fix **only** the listed issues while preserving the rest
    of the pipeline logic. Return the corrected JSON (processors array only,
    no markdown fences).

    ## Current pipeline
    ```json
    {pipeline_json}
    ```

    ## Guardrail errors
    {errors}

    ## Reminder
    - Prefer `dissect` over `grok`.
    - Processor order: parse → transform → calculate.
    - All fields must be ECS-compliant.
    - Forbidden processors: {forbidden}.
""")


_SIMULATION_FIX_TEMPLATE = textwrap.dedent("""\
    The following Elasticsearch ingest pipeline was simulated against real
    log samples and produced errors. Fix the pipeline so simulation succeeds.
    Return the corrected JSON (processors array only, no markdown fences).

    ## Current pipeline
    ```json
    {pipeline_json}
    ```

    ## Simulation errors
    {errors}

    ## Sample documents used
    ```json
    {samples}
    ```

    ## Reminder
    - Prefer `dissect` over `grok`.
    - Processor order: parse → transform → calculate.
    - All fields must be ECS-compliant.
    - Forbidden processors: {forbidden}.
""")
