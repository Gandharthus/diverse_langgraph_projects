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

    ## Production rules (MUST follow)

    1. **Preserve the raw log line.**
       Keep the original raw log line in `event.original` (preferred) or keep `message` intact.
       Never delete `message` unless the raw line has been copied to `event.original`.
       Never overwrite `message` with a parsed application message.

    2. **Single primary parse pass.**
       Choose ONE primary parsing processor per format (grok OR dissect OR json).
       Avoid sequential parsing of the same source field (no multiple grok/dissect passes).
       If the format has variants, handle them within a single parsing strategy.

    3. **Parser choice rubric.**
       - If the sample line is JSON, use the `json` processor.
       - If the format is strict delimiter-based with no quoted strings containing spaces, use `dissect`.
       - If fields are optional, reordered, or contain quoted strings with spaces, use `grok`.
       - If quoted strings contain spaces, do NOT use naive `dissect` patterns; use `grok` or quote-safe `dissect` with quoted field delimiters.

    4. **Types.**
       Convert numeric fields used for aggregations (status, duration, retry, bytes, size, count, etc.) to numeric types.
       Use grok type hints (e.g., `:int`, `:long`) or `convert` processors.

    5. **Minimalism.**
       Avoid scripts unless absolutely necessary.
       Avoid many alternative grok patterns unless the format truly varies.

    6. **Naming hygiene.**
       Do not reuse `message` as a parsed “app message”.
       If you extract an application message, write it to `log.message` or `labels.msg`.

    7. **Consistency.**
       Keep field names stable and avoid writing the same semantic field to multiple paths.

    8. **ECS Compliance.**
       Output field names should conform to the Elastic Common Schema (ECS) where possible.
       Use `labels.*` for custom fields that do not fit into standard ECS fields.
       **CRITICAL**: Map ALL available information.
       - If the log contains a log level, map it to `log.level`.
       - If the log contains a user agent, map it to `user_agent.original`.
       - If the log contains a service name, map it to `service.name` (or `labels.service`).

    9. **Forbidden processors.**
       Do NOT use any of: {forbidden}.

    10. **Error handling.**
        Processors MAY include an `on_failure` clause if necessary, but it is not mandatory.

    11. **`@timestamp`.**
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
    - Preserve raw log line in `event.original` or keep `message` intact.
    - Use a single primary parse step and avoid sequential parsing of the same field.
    - Choose json vs dissect vs grok using the rubric from the system prompt.
    - Convert numeric fields to numeric types.
    - Keep parsed app message in `log.message` or `labels.msg`, not `message`.
    - Avoid scripts and forbidden processors: {forbidden}.
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
    - Preserve raw log line in `event.original` or keep `message` intact.
    - Use a single primary parse step and avoid sequential parsing of the same field.
    - Choose json vs dissect vs grok using the rubric from the system prompt.
    - Convert numeric fields to numeric types.
    - Keep parsed app message in `log.message` or `labels.msg`, not `message`.
    - Avoid scripts and forbidden processors: {forbidden}.
""")
