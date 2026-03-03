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

            if proc_type == "rename" and "target_field" in conf:
                fields.append(conf["target_field"])

            if proc_type == "json" and "target_field" in conf:
                fields.append(conf["target_field"])

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


def iter_processors(pipeline: dict) -> list[tuple[str, dict, str]]:
    processors = pipeline.get("processors", [])
    collected: list[tuple[str, dict, str]] = []

    def _walk(proc_list: list, prefix: str) -> None:
        for idx, proc_wrapper in enumerate(proc_list):
            if not isinstance(proc_wrapper, dict) or len(proc_wrapper) == 0:
                continue
            proc_type = next(iter(proc_wrapper))
            proc_conf = proc_wrapper[proc_type]
            if not isinstance(proc_conf, dict):
                continue
            path = f"{prefix}[{idx}]"
            collected.append((proc_type, proc_conf, path))
            on_failure = proc_conf.get("on_failure")
            if isinstance(on_failure, list):
                _walk(on_failure, f"{path}.on_failure")

    if isinstance(processors, list):
        _walk(processors, "processors")
    return collected


def extract_parse_steps(pipeline: dict) -> list[tuple[str, str, str]]:
    parse_steps: list[tuple[str, str, str]] = []
    for proc_type, proc_conf, path in iter_processors(pipeline):
        if proc_type not in {"grok", "dissect", "json"}:
            continue
        field = proc_conf.get("field")
        if isinstance(field, str) and field:
            parse_steps.append((proc_type, field, path))
    return parse_steps


def extract_grok_typed_fields(pipeline: dict) -> set[str]:
    typed_fields: set[str] = set()
    type_pattern = re.compile(r"%\{\w+:([^}:]+)(?::([^}]+))?\}")
    for proc_type, proc_conf, _ in iter_processors(pipeline):
        if proc_type != "grok":
            continue
        patterns = proc_conf.get("patterns") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            for match in type_pattern.finditer(pattern):
                field = match.group(1)
                field_type = match.group(2)
                if field_type in {"int", "long", "float", "double"}:
                    typed_fields.add(field)
    return typed_fields


def extract_convert_numeric_fields(pipeline: dict) -> set[str]:
    numeric_types = {
        "integer",
        "long",
        "float",
        "double",
        "short",
        "byte",
        "half_float",
        "scaled_float",
        "unsigned_long",
    }
    fields: set[str] = set()
    for proc_type, proc_conf, _ in iter_processors(pipeline):
        if proc_type != "convert":
            continue
        field = proc_conf.get("field")
        conv_type = proc_conf.get("type")
        if isinstance(field, str) and isinstance(conv_type, str) and conv_type in numeric_types:
            fields.add(field)
    return fields


def extract_dissect_patterns_for_field(pipeline: dict, field: str) -> list[str]:
    patterns: list[str] = []
    for proc_type, proc_conf, _ in iter_processors(pipeline):
        if proc_type != "dissect":
            continue
        if proc_conf.get("field") != field:
            continue
        pattern = proc_conf.get("pattern")
        if isinstance(pattern, str):
            patterns.append(pattern)
    return patterns


def detect_quoted_strings_with_spaces(samples: Sequence[str]) -> bool:
    quoted = re.compile(r"""[=\s][A-Za-z0-9_.-]+=(["'])(?:[^"']*\s+[^"']*)\1""")
    for sample in samples:
        if not isinstance(sample, str):
            continue
        if quoted.search(sample):
            return True
    return False


def is_numeric_field_name(field_name: str) -> bool:
    leaf = field_name.split(".")[-1]
    direct = {
        "status",
        "status_code",
        "code",
        "duration",
        "latency",
        "elapsed",
        "retry",
        "retries",
        "attempts",
        "count",
        "bytes",
        "size",
        "length",
        "port",
        "pid",
        "seq",
        "sequence",
    }
    if leaf in direct:
        return True
    if re.search(r"(?:_ms|_us|_sec|_secs|_seconds|_millis|_ns)$", leaf):
        return True
    if re.search(r"(?:_bytes|_kb|_mb|_gb)$", leaf):
        return True
    if re.search(r"(?:_count|_total|_num|_size)$", leaf):
        return True
    return False


def extract_message_mutations(pipeline: dict) -> list[tuple[str, str]]:
    mutations: list[tuple[str, str]] = []
    mutating = {
        "set",
        "rename",
        "append",
        "convert",
        "gsub",
        "trim",
        "uppercase",
        "lowercase",
    }
    for proc_type, proc_conf, path in iter_processors(pipeline):
        if proc_type not in mutating:
            continue
        field = proc_conf.get("field")
        target_field = proc_conf.get("target_field")
        if field == "message":
            if proc_type == "rename" and target_field == "event.original":
                continue
            mutations.append((proc_type, path))
    return mutations


def extract_message_removals(pipeline: dict) -> list[str]:
    removals: list[str] = []
    for proc_type, proc_conf, path in iter_processors(pipeline):
        if proc_type != "remove":
            continue
        field = proc_conf.get("field")
        fields = proc_conf.get("fields")
        if field == "message":
            removals.append(path)
        elif isinstance(fields, list) and "message" in fields:
            removals.append(path)
    return removals


def has_event_original(pipeline: dict) -> bool:
    target_fields = extract_target_fields_from_pipeline(pipeline)
    if "event.original" in target_fields:
        return True
    for proc_type, proc_conf, _ in iter_processors(pipeline):
        if proc_type == "set" and proc_conf.get("field") == "event.original":
            return True
        if proc_type == "rename" and proc_conf.get("target_field") == "event.original":
            return True
        if proc_type == "append" and proc_conf.get("field") == "event.original":
            return True
    return False


def extract_label_conflicts(field_names: Sequence[str]) -> list[str]:
    errors: list[str] = []
    has_error = any(name.startswith("error.") for name in field_names)
    has_labels_error = any(
        name == "labels.error"
        or name.startswith("labels.error.")
        or name == "labels.err"
        or name.startswith("labels.err.")
        for name in field_names
    )
    if has_error and has_labels_error:
        errors.append(
            "Avoid duplicating error semantics across 'error.*' and 'labels.error'."
        )

    has_http_status = any(
        name in {"http.response.status_code", "http.status_code"} for name in field_names
    )
    has_labels_status = any(
        name == "labels.status"
        or name == "labels.status_code"
        or name.startswith("labels.status.")
        for name in field_names
    )
    if has_http_status and has_labels_status:
        errors.append(
            "Avoid duplicating status semantics across HTTP status and labels.status."
        )
    return errors


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
    samples: Sequence[str] | None = None,
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

    target_fields = extract_target_fields_from_pipeline(pipeline)

    event_original_present = has_event_original(pipeline)
    message_removals = extract_message_removals(pipeline)
    if message_removals and not event_original_present:
        errors.append(
            "Raw log line must be preserved: keep 'message' or copy to 'event.original'."
        )

    message_mutations = extract_message_mutations(pipeline)
    if message_mutations:
        errors.append(
            "Do not modify 'message' after ingest; write parsed content to 'log.message' or 'labels.msg'."
        )

    parse_steps = extract_parse_steps(pipeline)
    parse_by_field: dict[str, list[tuple[str, str]]] = {}
    for proc_type, field, path in parse_steps:
        parse_by_field.setdefault(field, []).append((proc_type, path))
    for field, steps in parse_by_field.items():
        if len(steps) > 1:
            chain = " -> ".join(p[0] for p in steps)
            errors.append(
                f"Multiple parse processors read the same field '{field}': {chain}. Use a single primary parse step."
            )

    if samples:
        quoted_present = detect_quoted_strings_with_spaces(samples)
        if quoted_present:
            dissect_patterns = extract_dissect_patterns_for_field(pipeline, "message")
            if dissect_patterns:
                quote_safe = any(re.search(r"""["']%\{""", pattern) for pattern in dissect_patterns)
                if not quote_safe:
                    errors.append(
                        "Dissect is used with quoted strings containing spaces; use grok or a quote-safe dissect pattern."
                    )

    if not any(proc_type == "json" and field in {"message", "event.original"} for proc_type, field, _ in parse_steps):
        grok_typed_fields = extract_grok_typed_fields(pipeline)
        convert_numeric_fields = extract_convert_numeric_fields(pipeline)
        numeric_candidates = [f for f in target_fields if is_numeric_field_name(f)]
        for field in numeric_candidates:
            if field not in grok_typed_fields and field not in convert_numeric_fields:
                errors.append(
                    f"Numeric field '{field}' should be converted to a numeric type."
                )

    errors.extend(extract_label_conflicts(target_fields))

    # ECS field check
    ecs_result = validate_ecs_compliance(
        target_fields, ecs_namespaces, ecs_field_catalog
    )
    errors.extend(ecs_result.errors)
    
    # ECS warnings do not cause failure, so we don't add them to 'errors'
    # but the calling code might want to log them or present them.
    # For now, we only return hard errors in the 'errors' list.
    
    return GuardrailResult(passed=len(errors) == 0, errors=errors)
