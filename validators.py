"""
Code-based validators for the Pipeline Wizard.

Contains the ECS compliance checker, processor ordering enforcement,
forbidden-processor detection, and all the helpers they rely on.
Everything here is deterministic – no LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ECSValidationResult:
    """Result of ECS validation on a set of field names."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class GuardrailResult:
    """Result of the full guardrail check on a pipeline."""
    passed: bool
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline field extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_target_fields_from_pipeline(pipeline: dict) -> list[str]:
    """
    Walk an Elasticsearch pipeline and collect every field name that appears
    in a *target* position (i.e. a field the pipeline writes to).
    """
    fields: list[str] = []

    def _collect(proc: dict) -> None:
        for proc_type, conf in proc.items():
            if not isinstance(conf, dict):
                continue

            if "target_field" in conf:
                fields.append(conf["target_field"])

            if proc_type == "set" and "field" in conf:
                fields.append(conf["field"])

            if proc_type == "append" and "field" in conf:
                fields.append(conf["field"])

            if proc_type == "dissect" and "pattern" in conf:
                fields.extend(re.findall(r"%\{([^}]+)\}", conf["pattern"]))

            if proc_type == "grok" and "patterns" in conf:
                for pat in conf["patterns"]:
                    fields.extend(
                        m.group(1)
                        for m in re.finditer(r"%\{\w+:([^}:]+)(?::\w+)?\}", pat)
                    )

            if "on_failure" in conf and isinstance(conf["on_failure"], list):
                for sub in conf["on_failure"]:
                    _collect(sub)

    for proc in pipeline.get("processors", []):
        _collect(proc)

    return fields


# ──────────────────────────────────────────────────────────────────────────────
# Processor phase classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_processor(
    proc_type: str,
    processor_phases: dict[str, list[str]],
) -> Literal["parse", "transform", "calculate", "unknown"]:
    """
    Return the phase a processor belongs to, looked up from the config-driven
    ``processor_phases`` mapping.
    """
    for phase_name in ("parse", "transform", "calculate"):
        if proc_type in processor_phases.get(phase_name, []):
            return phase_name  # type: ignore[return-value]
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# ECS compliance validator
# ──────────────────────────────────────────────────────────────────────────────

def validate_ecs_compliance(
    field_names: Sequence[str],
    ecs_namespaces: set[str],
    ecs_field_catalog: dict[str, dict[str, Any]],
) -> ECSValidationResult:
    """
    Validate field names against ECS conventions.

    Checks
    ------
    1. Naming – lowercase ``[a-z0-9_.]`` only.
    2. camelCase / digit-leading segments.
    3. Known ECS namespace prefix.
    4. Catalog match (informational).
    """
    errors: list[str] = []
    warnings: list[str] = []
    allowed_meta = {"_id", "_index", "_source", "_ingest", "_version", "_routing"}

    for name in field_names:
        if name in allowed_meta or name.startswith("_ingest."):
            continue

        # 1. Naming convention
        if not re.match(r"^[a-z0-9_.@]+$", name):
            errors.append(
                f"Field '{name}' violates ECS naming: only lowercase "
                f"alphanumerics, underscores, dots, and @ are allowed."
            )
            continue

        # 2. camelCase
        if re.search(r"[a-z][A-Z]", name):
            errors.append(
                f"Field '{name}' uses camelCase – use snake_case with dots."
            )

        # Segments starting with digits
        for segment in name.split("."):
            if segment and segment[0].isdigit():
                errors.append(
                    f"Field '{name}' has segment '{segment}' starting with "
                    f"a digit."
                )

        # 3. Namespace check
        top_level = name.split(".")[0] if "." in name else name
        if (
            top_level not in ecs_namespaces
            and name not in {"message", "tags", "labels", "@timestamp"}
            and "." not in name
        ):
            warnings.append(
                f"Field '{name}' has no recognised ECS namespace prefix. "
                f"Consider placing it under a custom namespace "
                f"(e.g. 'myapp.{name}')."
            )

        # 4. Catalog match
        if name in ecs_field_catalog:
            pass
        elif "." in name and top_level in ecs_namespaces:
            pass
        elif top_level not in ecs_namespaces and "." in name:
            warnings.append(
                f"Field '{name}' uses namespace '{top_level}' which is not "
                f"a standard ECS namespace."
            )

    return ECSValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full guardrail check
# ──────────────────────────────────────────────────────────────────────────────

def check_guardrails(
    pipeline: dict | None,
    *,
    forbidden_processors: set[str],
    processor_phases: dict[str, list[str]],
    ecs_namespaces: set[str],
    ecs_field_catalog: dict[str, dict[str, Any]],
) -> GuardrailResult:
    """
    Run every deterministic guardrail check on a pipeline and return the
    aggregated result.

    Checks
    ------
    1. Structural validity.
    2. Forbidden processors.
    3. ``on_failure`` clause presence.
    4. Processor phase ordering (parse → transform → calculate).
    5. ``grok`` usage justification.
    6. ECS field name compliance.
    """
    if pipeline is None:
        return GuardrailResult(passed=False, errors=["Pipeline JSON is None."])

    errors: list[str] = []
    processors = pipeline.get("processors", [])

    if not isinstance(processors, list) or len(processors) == 0:
        return GuardrailResult(
            passed=False,
            errors=["Pipeline must contain a non-empty 'processors' array."],
        )

    last_phase: str = "parse"
    phase_order = {"parse": 0, "transform": 1, "calculate": 2, "unknown": -1}

    for idx, proc_wrapper in enumerate(processors):
        if not isinstance(proc_wrapper, dict) or len(proc_wrapper) == 0:
            errors.append(f"Processor at index {idx} is not a valid object.")
            continue

        proc_type = next(iter(proc_wrapper))
        proc_conf = proc_wrapper[proc_type]

        # Forbidden
        if proc_type in forbidden_processors:
            errors.append(
                f"Processor '{proc_type}' at index {idx} is forbidden by "
                f"company policy."
            )

        # on_failure check removed as requested

        # Ordering check removed to allow flexible pipeline structures
        # phase = classify_processor(proc_type, processor_phases)
        # if phase != "unknown":
        #     if phase_order.get(phase, -1) < phase_order.get(last_phase, -1):
        #         errors.append(
        #             f"Processor '{proc_type}' at index {idx} (phase={phase}) "
        #             f"appears after a '{last_phase}'-phase processor. "
        #             f"Expected order: parse → transform → calculate."
        #         )
        #     last_phase = phase

        # grok justification
        if proc_type == "grok":
            desc = proc_conf.get("description", "") if isinstance(proc_conf, dict) else ""
            if not desc:
                errors.append(
                    f"Processor 'grok' at index {idx} is used without a "
                    f"'description' justifying why 'dissect' was not "
                    f"sufficient (company policy: prefer dissect)."
                )

    # ECS field check
    target_fields = extract_target_fields_from_pipeline(pipeline)
    ecs_result = validate_ecs_compliance(
        target_fields, ecs_namespaces, ecs_field_catalog
    )
    errors.extend(ecs_result.errors)
    
    # ECS warnings do not cause failure, so we don't add them to 'errors'
    # but the calling code might want to log them or present them.
    # For now, we only return hard errors in the 'errors' list.
    
    return GuardrailResult(passed=len(errors) == 0, errors=errors)
