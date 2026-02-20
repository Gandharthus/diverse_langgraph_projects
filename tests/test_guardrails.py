from __future__ import annotations

from pathlib import Path

import yaml

from pipeline_wizard import _load_config  # type: ignore[import-not-found]
from validators import check_guardrails


def run_guardrails(pipeline: dict, samples: list[str] | None = None) -> list[str]:
    cfg = _load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    result = check_guardrails(
        pipeline,
        forbidden_processors=set(cfg.get("forbidden_processors", [])),
        processor_phases=cfg.get("processor_phases", {}),
        ecs_namespaces=set(cfg.get("ecs_namespaces", [])),
        ecs_field_catalog=cfg.get("ecs_field_catalog", {}),
        samples=samples,
    )
    return result.errors


def test_preserve_raw_message_required() -> None:
    pipeline = {"processors": [{"remove": {"field": "message"}}]}
    errors = run_guardrails(pipeline)
    assert any("Raw log line must be preserved" in e for e in errors)


def test_single_parse_on_same_field() -> None:
    pipeline = {
        "processors": [
            {
                "grok": {
                    "field": "message",
                    "patterns": ["%{WORD:log.level}"],
                    "description": "Regex required for variable tokens",
                }
            },
            {"dissect": {"field": "message", "pattern": "%{log.level}"}},
        ]
    }
    errors = run_guardrails(pipeline)
    assert any("Multiple parse processors read the same field" in e for e in errors)


def test_quoted_strings_require_safe_parser() -> None:
    pipeline = {
        "processors": [
            {"set": {"field": "event.original", "value": "{{message}}"}},
            {
                "dissect": {
                    "field": "message",
                    "pattern": "level=%{labels.level} msg=%{labels.msg} user=%{labels.user}",
                }
            },
        ]
    }
    samples = ['level=INFO msg="hello world" user=bob']
    errors = run_guardrails(pipeline, samples=samples)
    assert any("quoted strings" in e for e in errors)


def test_numeric_fields_require_conversion() -> None:
    pipeline = {
        "processors": [
            {"set": {"field": "event.original", "value": "{{message}}"}},
            {
                "dissect": {
                    "field": "message",
                    "pattern": "status=%{http.response.status_code} duration_ms=%{event.duration}",
                }
            },
        ]
    }
    errors = run_guardrails(pipeline)
    assert any("Numeric field 'http.response.status_code'" in e for e in errors)
    assert any("Numeric field 'event.duration'" in e for e in errors)


def test_valid_pipeline_passes_guardrails() -> None:
    pipeline = {
        "processors": [
            {"set": {"field": "event.original", "value": "{{message}}"}},
            {
                "grok": {
                    "field": "message",
                    "patterns": [
                        "status=%{NUMBER:http.response.status_code:int} duration_ms=%{NUMBER:event.duration:long}"
                    ],
                    "description": "Regex required for optional separators",
                }
            },
        ]
    }
    errors = run_guardrails(pipeline)
    assert errors == []
