"""
illumio_query.py – Generic Elasticsearch DSL builder for Illumio flow queries.

All existing Illumio agent queries (traffic, blocked flows, service consumers)
are expressible as combinations of term filters, prefix filters, and terms aggs.

Example – traffic (dev→prod):
    build_illumio_query(
        term_filters=[
            {"field": "illumio.source.labels.env",      "value": "E_DEV"},
            {"field": "illumio.destination.labels.env", "value": "E_PROD"},
        ],
        prefix_filters=[
            {"field": "illumio.destination.labels.app", "value": "A_AP12345-"},
        ],
        aggs=[
            {"name": "apps", "field": "illumio.source.labels.app", "size": 50},
        ],
        date_range="now-7d",
    )

Example – blocked inbound flows:
    build_illumio_query(
        term_filters=[
            {"field": "illumio.policy_decision", "value": "denied"},
        ],
        prefix_filters=[
            {"field": "illumio.destination.labels.app", "value": "A_AP12345-"},
        ],
        aggs=[
            {"name": "top_sources",    "field": "illumio.source.hostname", "size": 20},
            {"name": "top_dest_ports", "field": "destination.port",        "size": 20},
            {"name": "top_protocols",  "field": "network.protocol",        "size": 10},
        ],
        track_total_hits=True,
    )

Example – service consumers:
    build_illumio_query(
        term_filters=[
            {"field": "policy_decision",             "value": "Allowed"},
            {"field": "illumio.source.labels.env",   "value": "E_PROD"},
        ],
        prefix_filters=[
            {"field": "illumio.destination.labels.app", "value": "A_AP12345-"},
        ],
        aggs=[
            {"name": "top_consumers", "field": "illumio.source.labels.app", "size": 20},
        ],
    )
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Type aliases (plain dicts kept intentionally simple for JSON-serialisability)
# ---------------------------------------------------------------------------

TermFilter   = dict  # {"field": str, "value": str}
PrefixFilter = dict  # {"field": str, "value": str}
Agg          = dict  # {"name": str, "field": str, "size": int}


def build_illumio_query(
    term_filters:   list[TermFilter]   | None = None,
    prefix_filters: list[PrefixFilter] | None = None,
    aggs:           list[Agg]          | None = None,
    date_range:     str | None = None,
    timestamp_field: str = "@timestamp",
    track_total_hits: bool = False,
) -> dict:
    """
    Build an Elasticsearch DSL query for Illumio flow data.

    Parameters
    ----------
    term_filters:
        Exact-match filters.  Each dict must have ``"field"`` and ``"value"``.
        Produces ``{"term": {field: value}}`` clauses.
    prefix_filters:
        Prefix-match filters.  Each dict must have ``"field"`` and ``"value"``.
        Produces ``{"prefix": {field: value}}`` clauses.
    aggs:
        Terms aggregations to include.  Each dict must have ``"name"``,
        ``"field"``, and ``"size"``.
    date_range:
        Elasticsearch date-math expression for the lower bound of ``@timestamp``
        (e.g. ``"now-1h"``, ``"now-7d"``).  When provided a range filter of
        ``{gte: date_range, lte: "now"}`` is appended automatically.
    timestamp_field:
        The timestamp field name (default: ``"@timestamp"``).
    track_total_hits:
        When ``True`` adds ``"track_total_hits": True`` to the query root,
        useful for blocked-flow queries where total hit count matters.

    Returns
    -------
    dict
        A ready-to-use Elasticsearch DSL query body.
    """
    # Build filter list
    bool_filter: list[dict] = []

    for f in term_filters or []:
        bool_filter.append({"term": {f["field"]: f["value"]}})

    for f in prefix_filters or []:
        bool_filter.append({"prefix": {f["field"]: f["value"]}})

    if date_range:
        bool_filter.append(
            {"range": {timestamp_field: {"gte": date_range, "lte": "now"}}}
        )

    # Build aggregations map
    aggs_body: dict = {}
    for agg in aggs or []:
        aggs_body[agg["name"]] = {
            "terms": {
                "field": agg["field"],
                "size":  agg["size"],
            }
        }

    # Assemble query
    query: dict = {
        "size": 0,
        "query": {
            "bool": {
                "filter": bool_filter,
            }
        },
    }

    if track_total_hits:
        query["track_total_hits"] = True

    if aggs_body:
        query["aggs"] = aggs_body

    return query
