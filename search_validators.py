"""
Deterministic validators for the Elasticsearch Search Agent.

Enforces company guardrails on DSL queries before they hit the cluster.
No LLM calls – pure structural checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryValidationResult:
    """Result of validating a DSL query against company guardrails."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation depth / bucket checks
# ──────────────────────────────────────────────────────────────────────────────

def _count_agg_depth(aggs: dict, current_depth: int = 1) -> int:
    """Recursively find the maximum nesting depth of aggregations."""
    max_depth = current_depth
    for agg_name, agg_body in aggs.items():
        if not isinstance(agg_body, dict):
            continue
        # Look for nested "aggs" or "aggregations" inside this agg
        for sub_key in ("aggs", "aggregations"):
            if sub_key in agg_body:
                nested = agg_body[sub_key]
                if isinstance(nested, dict):
                    d = _count_agg_depth(nested, current_depth + 1)
                    max_depth = max(max_depth, d)
    return max_depth


def _check_bucket_sizes(
    aggs: dict,
    max_buckets: int,
    path: str = "aggs",
) -> list[str]:
    """Check that every bucket aggregation has a size <= max_buckets."""
    errors: list[str] = []
    bucket_agg_types = {
        "terms", "significant_terms", "rare_terms",
        "histogram", "date_histogram",
        "composite",
    }

    for agg_name, agg_body in aggs.items():
        if not isinstance(agg_body, dict):
            continue
        for agg_type in bucket_agg_types:
            if agg_type in agg_body:
                conf = agg_body[agg_type]
                if isinstance(conf, dict):
                    size = conf.get("size")
                    if size is None and agg_type in ("terms", "significant_terms"):
                        errors.append(
                            f"{path}.{agg_name}: '{agg_type}' aggregation missing "
                            f"explicit 'size' (ES defaults to 10, but policy requires "
                            f"an explicit value <= {max_buckets})."
                        )
                    elif size is not None and isinstance(size, (int, float)) and size > max_buckets:
                        errors.append(
                            f"{path}.{agg_name}: '{agg_type}' aggregation has "
                            f"size={size} which exceeds the limit of {max_buckets}."
                        )

        # Recurse into nested aggs
        for sub_key in ("aggs", "aggregations"):
            if sub_key in agg_body and isinstance(agg_body[sub_key], dict):
                errors.extend(
                    _check_bucket_sizes(
                        agg_body[sub_key],
                        max_buckets,
                        path=f"{path}.{agg_name}.{sub_key}",
                    )
                )

    return errors


# ──────────────────────────────────────────────────────────────────────────────
# Script detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_scripts(data: Any, path: str = "root") -> list[str]:
    """Recursively search for any 'script' keys in the query."""
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "script":
                found.append(
                    f"Forbidden 'script' usage found at {path}.script"
                )
            elif key == "scripted_metric":
                found.append(
                    f"Forbidden 'scripted_metric' aggregation at {path}.scripted_metric"
                )
            elif key == "script_score":
                found.append(
                    f"Forbidden 'script_score' query at {path}.script_score"
                )
            else:
                found.extend(_find_scripts(value, f"{path}.{key}"))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            found.extend(_find_scripts(item, f"{path}[{i}]"))
    return found


# ──────────────────────────────────────────────────────────────────────────────
# Main validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_search_query(
    query: dict | None,
    *,
    max_agg_depth: int = 3,
    max_buckets: int = 1000,
) -> QueryValidationResult:
    """
    Run all deterministic guardrail checks on an Elasticsearch DSL query.

    Checks
    ------
    1. Structural validity (must be a dict).
    2. No script usage anywhere.
    3. Aggregation nesting depth <= max_agg_depth.
    4. Bucket aggregation sizes <= max_buckets.
    5. Size sanity (query size <= 10000).
    """
    if query is None:
        return QueryValidationResult(passed=False, errors=["Query is None."])

    if not isinstance(query, dict):
        return QueryValidationResult(
            passed=False,
            errors=["Query must be a JSON object (dict)."],
        )

    errors: list[str] = []
    warnings: list[str] = []

    # Strip _comment if present (we add it for readability)
    query_clean = {k: v for k, v in query.items() if k != "_comment"}

    # 1. Script detection
    script_errors = _find_scripts(query_clean)
    errors.extend(script_errors)

    # 2. Aggregation depth
    for agg_key in ("aggs", "aggregations"):
        if agg_key in query_clean and isinstance(query_clean[agg_key], dict):
            depth = _count_agg_depth(query_clean[agg_key])
            if depth > max_agg_depth:
                errors.append(
                    f"Aggregation nesting depth is {depth}, "
                    f"exceeds maximum of {max_agg_depth}."
                )

    # 3. Bucket sizes
    for agg_key in ("aggs", "aggregations"):
        if agg_key in query_clean and isinstance(query_clean[agg_key], dict):
            errors.extend(
                _check_bucket_sizes(query_clean[agg_key], max_buckets)
            )

    # 4. Size sanity
    size = query_clean.get("size")
    if size is not None and isinstance(size, (int, float)) and size > 10000:
        errors.append(
            f"Query size={size} exceeds safe maximum of 10000."
        )

    # 5. Warn if no time filter present (common mistake)
    query_body = query_clean.get("query", {})
    has_time_filter = _has_timestamp_filter(query_body)
    if not has_time_filter:
        warnings.append(
            "No @timestamp range filter detected. Consider adding one "
            "to avoid scanning the entire index."
        )

    return QueryValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _has_timestamp_filter(query_body: Any) -> bool:
    """Check if the query contains a range filter on @timestamp."""
    if not isinstance(query_body, dict):
        return False
    for key, value in query_body.items():
        if key == "range" and isinstance(value, dict) and "@timestamp" in value:
            return True
        if isinstance(value, dict):
            if _has_timestamp_filter(value):
                return True
        if isinstance(value, list):
            for item in value:
                if _has_timestamp_filter(item):
                    return True
    return False
