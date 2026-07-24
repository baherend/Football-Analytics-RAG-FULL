"""
query_schema.py — Phase 3: StructuredQuery and StructuredResult Schemas

Defines the canonical query and result types for the structured analytics path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass
class Filter:
    """A single filter condition."""
    dimension: str  # "player_name" | "team_name" | "stage" | "match_id" | etc.
    operator: str   # "eq" | "neq" | "gt" | "lt" | "gte" | "lte" | "in"
    value: Any

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# StructuredQuery
# ---------------------------------------------------------------------------


@dataclass
class StructuredQuery:
    """
    A parsed structured query against match_facts.json.

    Examples:
        "How many goals did Messi score?"
        → intent="numeric", entity="player", entity_name="Messi",
          metric="goals", aggregation="sum"

        "Who scored the most goals?"
        → intent="superlative", entity="player", metric="goals",
          aggregation="max", limit=1

        "Which team had the highest xG?"
        → intent="aggregation", entity="team", metric="xg",
          aggregation="max"

        "Koundé's xG vs Morocco"
        → intent="slice", entity="player", entity_name="Koundé",
          metric="xg", filters=[Filter("opponent", "eq", "Morocco")]
    """
    intent: str  # "numeric" | "superlative" | "slice" | "aggregation"
    entity: str  # "player" | "team" | "match"
    metric: str  # canonical metric name from vocab
    aggregation: str = "sum"  # "sum" | "avg" | "max" | "min" | "count"
    entity_name: str | None = None  # specific entity name
    entity_id: int | None = None  # specific entity ID
    filters: list[Filter] = field(default_factory=list)
    limit: int | None = None  # for superlatives
    sort_order: str = "desc"  # "asc" | "desc"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filters"] = [f.to_dict() for f in self.filters]
        return d


# ---------------------------------------------------------------------------
# StructuredResult
# ---------------------------------------------------------------------------


@dataclass
class StructuredResult:
    """
    Result of executing a StructuredQuery.

    Status values:
        "resolved"  — all filters honored, result is complete
        "partial"   — some filters dropped (listed in dropped_filters)
        "empty"     — no matching records found
    """
    status: str  # "resolved" | "partial" | "empty"
    query: StructuredQuery
    data: list[dict] = field(default_factory=list)  # matching records
    aggregated_value: float | int | None = None  # result of aggregation
    dropped_filters: list[str] = field(default_factory=list)  # filters that couldn't be honored
    explanation: str = ""  # human-readable explanation

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "query": self.query.to_dict(),
            "data": self.data,
            "aggregated_value": self.aggregated_value,
            "dropped_filters": self.dropped_filters,
            "explanation": self.explanation,
        }
        return d


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def numeric_query(player_name: str, metric: str, aggregation: str = "sum") -> StructuredQuery:
    """Create a numeric query for a specific player."""
    return StructuredQuery(
        intent="numeric",
        entity="player",
        metric=metric,
        aggregation=aggregation,
        entity_name=player_name,
    )


def superlative_query(entity: str, metric: str, aggregation: str = "max",
                      limit: int = 1, filters: list[Filter] | None = None) -> StructuredQuery:
    """Create a superlative query (top/bottom N)."""
    return StructuredQuery(
        intent="superlative",
        entity=entity,
        metric=metric,
        aggregation=aggregation,
        filters=filters or [],
        limit=limit,
    )


def slice_query(player_name: str, metric: str, filters: list[Filter]) -> StructuredQuery:
    """Create a slice query for a specific player with filters."""
    return StructuredQuery(
        intent="slice",
        entity="player",
        metric=metric,
        aggregation="sum",
        entity_name=player_name,
        filters=filters,
    )


def aggregation_query(entity: str, metric: str, aggregation: str = "sum",
                      filters: list[Filter] | None = None) -> StructuredQuery:
    """Create an aggregation query across a group."""
    return StructuredQuery(
        intent="aggregation",
        entity=entity,
        metric=metric,
        aggregation=aggregation,
        filters=filters or [],
    )
