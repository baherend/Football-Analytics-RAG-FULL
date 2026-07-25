"""
test_structured.py — Phase 3: Structured Query Unit Tests

Tests run directly against match_facts.json — no vector store needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.query.query_schema import StructuredQuery, StructuredResult, Filter
from src.query.resolver import resolve, _load_data
from src.query.vocab import resolve_metric, resolve_aggregation


@pytest.fixture(scope="module")
def data():
    """Load match_facts.json once for all tests in this module."""
    return _load_data()


# ---------------------------------------------------------------------------
# Tests: Vocabulary
# ---------------------------------------------------------------------------


def test_metric_resolution():
    """Metric synonyms should resolve correctly."""
    assert resolve_metric("goals") == "goals"
    assert resolve_metric("goal") == "goals"
    assert resolve_metric("scored") == "goals"
    assert resolve_metric("xg") == "xg"
    assert resolve_metric("expected goals") == "xg"
    assert resolve_metric("passes") == "passes_attempted"
    assert resolve_metric("minutes") == "minutes"
    assert resolve_metric("nonexistent") is None


def test_aggregation_resolution():
    """Aggregation synonyms should resolve correctly."""
    assert resolve_aggregation("sum") == "sum"
    assert resolve_aggregation("total") == "sum"
    assert resolve_aggregation("most") == "max"
    assert resolve_aggregation("highest") == "max"
    assert resolve_aggregation("average") == "avg"
    assert resolve_aggregation("nonexistent") is None


# ---------------------------------------------------------------------------
# Tests: Numeric queries
# ---------------------------------------------------------------------------


def test_messi_goals(data):
    """How many goals did Messi score? → 7"""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
    )
    result = resolve(q, data)
    assert result.status == "resolved", f"Status={result.status}"
    assert result.aggregated_value == 7, f"Goals={result.aggregated_value}"


def test_messi_xg(data):
    """What is Messi's xG? → ~6.03"""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="xg",
        aggregation="sum",
        entity_name="Messi",
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert abs(result.aggregated_value - 6.03) < 0.1, f"xG={result.aggregated_value}"


def test_messi_minutes(data):
    """How many minutes did Messi play? → ~733.9"""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="minutes",
        aggregation="sum",
        entity_name="Messi",
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert abs(result.aggregated_value - 733.9) < 1.0, f"Minutes={result.aggregated_value}"


def test_mbappe_goals(data):
    """How many goals did Mbappé score? → 8"""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Mbappé",
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert result.aggregated_value == 8, f"Goals={result.aggregated_value}"


# ---------------------------------------------------------------------------
# Tests: Superlative queries
# ---------------------------------------------------------------------------


def test_top_scorer(data):
    """Who scored the most goals? → Mbappé (8)"""
    q = StructuredQuery(
        intent="superlative",
        entity="player",
        metric="goals",
        aggregation="sum",
        limit=1,
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert len(result.data) == 1
    top = result.data[0]
    # Mbappé scored 8 goals in the tournament
    assert "Mbapp" in top["player_name"], f"Top scorer: {top['player_name']}"
    assert result.aggregated_value == 8, f"Goals={result.aggregated_value}"


def test_top_xg_player(data):
    """Who had the highest xG? → Mbappé"""
    q = StructuredQuery(
        intent="superlative",
        entity="player",
        metric="xg",
        aggregation="sum",
        limit=1,
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert len(result.data) == 1


def test_top_3_scorers(data):
    """Top 3 scorers."""
    q = StructuredQuery(
        intent="superlative",
        entity="player",
        metric="goals",
        aggregation="sum",
        limit=3,
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert len(result.data) == 3
    goals = [r.get("goals", 0) for r in result.data]
    # Should be sorted descending
    assert goals == sorted(goals, reverse=True)


# ---------------------------------------------------------------------------
# Tests: Aggregation queries
# ---------------------------------------------------------------------------


def test_team_total_goals(data):
    """Which team scored the most goals? → Argentina (15)

    TeamMatchFacts does not store per-team goal counts directly. Goals are
    player-level metrics. This test verifies the resolver returns a sensible
    (partial) result when querying a metric absent from team records.
    """
    q = StructuredQuery(
        intent="aggregation",
        entity="team",
        metric="goals",
        aggregation="sum",
        limit=1,
    )
    result = resolve(q, data)
    # goals is not a field on TeamMatchFacts — resolver returns 0 or partial
    assert result.status in ("resolved", "partial")


# ---------------------------------------------------------------------------
# Tests: Slice queries
# ---------------------------------------------------------------------------


def test_messi_knockout_goals(data):
    """Messi's goals in knockout matches."""
    q = StructuredQuery(
        intent="slice",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
        filters=[Filter("is_knockout", "eq", True)],
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    # Messi scored 5 goals in knockout matches (R16: 1, QF: 1, SF: 1, Final: 2)
    assert result.aggregated_value == 5, f"Knockout goals={result.aggregated_value}"


def test_messi_group_goals(data):
    """Messi's goals in group stage."""
    q = StructuredQuery(
        intent="slice",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
        filters=[Filter("stage", "eq", "Group Stage")],
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    # Messi scored 2 group-stage goals: 1 vs Saudi Arabia, 1 vs Mexico
    assert result.aggregated_value == 2, f"Group goals={result.aggregated_value}"


def test_messi_vs_france(data):
    """Messi's goals vs France."""
    q = StructuredQuery(
        intent="slice",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
        filters=[Filter("opponent", "eq", "France")],
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert result.aggregated_value == 2, f"Goals vs France={result.aggregated_value}"


# ---------------------------------------------------------------------------
# Tests: Filter isolation (regression: cache key must include filters)
# ---------------------------------------------------------------------------


def test_filtered_queries_do_not_collide(data):
    """
    Regression guard: queries that differ only by their filters must not share
    a cache entry. Previously the cache key omitted filters, so Messi's group
    goals returned his knockout total.
    """
    def messi_goals(filters):
        return resolve(StructuredQuery(
            intent="slice", entity="player", metric="goals", aggregation="sum",
            entity_name="Messi", filters=filters), data).aggregated_value

    knockout = messi_goals([Filter("is_knockout", "eq", True)])
    group = messi_goals([Filter("stage", "eq", "Group Stage")])
    overall = resolve(StructuredQuery(
        intent="numeric", entity="player", metric="goals", aggregation="sum",
        entity_name="Messi"), data).aggregated_value

    assert knockout == 5, f"knockout={knockout}"
    assert group == 2, f"group={group}"
    assert overall == 7, f"overall={overall}"
    assert group != knockout, "filtered queries collided in the cache"


# ---------------------------------------------------------------------------
# Tests: Period slicing (regression: by_period must be honored)
# ---------------------------------------------------------------------------


def test_period_slicing_reads_by_period_blocks():
    """
    Unit test of the period-aware read logic, independent of the persisted
    artifact: when a record carries per-period blocks, slicing must return the
    per-period value, and combining periods must sum them.
    """
    from src.query.resolver import _read_metric_from_record, _aggregate_period_aware

    rec = {
        "passes_attempted": 10,
        "by_period": {"1": {"passes_attempted": 6}, "2": {"passes_attempted": 4}},
    }
    assert _read_metric_from_record(rec, "passes_attempted", None)[0] == 10   # total
    assert _read_metric_from_record(rec, "passes_attempted", (1,))[0] == 6     # 1st half
    assert _read_metric_from_record(rec, "passes_attempted", (2,))[0] == 4     # 2nd half
    assert _read_metric_from_record(rec, "passes_attempted", (1, 2))[0] == 10  # both
    assert _aggregate_period_aware([rec, rec], "passes_attempted", "sum", (1,)) == 12


def test_period_filter_without_data_is_honest_partial(data):
    """
    The shipped artifact has no per-period breakdown. A period slice must then
    be reported as a dropped filter (honest partial) and fall back to the match
    total — never a silent wrong slice.
    """
    total = resolve(StructuredQuery(
        intent="numeric", entity="player", metric="passes_attempted",
        aggregation="sum", entity_name="Messi"), data).aggregated_value

    sliced = resolve(StructuredQuery(
        intent="numeric", entity="player", metric="passes_attempted",
        aggregation="sum", entity_name="Messi",
        filters=[Filter("period", "in", [1])]), data)

    assert sliced.status == "partial", f"status={sliced.status}"
    assert "period" in sliced.dropped_filters, f"dropped={sliced.dropped_filters}"
    assert sliced.aggregated_value == total, "expected honest fallback to the total"


def test_period_filter_on_minutes_is_partial(data):
    """
    minutes is a match-grain metric with no per-period breakdown. A period
    filter on it must be reported as a dropped filter (honest partial), not
    silently return an empty or wrong result.
    """
    q = StructuredQuery(
        intent="numeric", entity="player", metric="minutes", aggregation="sum",
        entity_name="Messi", filters=[Filter("period", "in", [1])])
    result = resolve(q, data)
    assert result.status == "partial", f"status={result.status}"
    assert "period" in result.dropped_filters, f"dropped={result.dropped_filters}"
    assert result.aggregated_value and result.aggregated_value > 0


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


def test_nonexistent_player(data):
    """Query for a player that doesn't exist."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Zidane",  # Not in WC 2022 data
    )
    result = resolve(q, data)
    assert result.status == "empty"
    assert "No player found" in result.explanation


def test_invalid_metric(data):
    """Query with an invalid metric."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="nonexistent_metric",
        aggregation="sum",
        entity_name="Messi",
    )
    result = resolve(q, data)
    assert result.status == "empty"
    assert "Unknown metric" in result.explanation


def test_empty_result(data):
    """Query that returns no results."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
        filters=[Filter("stage", "eq", "Nonexistent Stage")],
    )
    result = resolve(q, data)
    assert result.status == "empty"


def test_partial_filter(data):
    """Query with a filter that gets dropped."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Messi",
        filters=[Filter("nonexistent_dimension", "eq", "value")],
    )
    result = resolve(q, data)
    # The filter should be dropped, but the query should still resolve
    assert result.status in ("resolved", "partial")
    if result.status == "partial":
        assert len(result.dropped_filters) > 0


# ---------------------------------------------------------------------------
# Tests: Entity resolution
# ---------------------------------------------------------------------------


def test_partial_name_resolution(data):
    """Partial player names should resolve via accent normalization."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Mbappe",  # Missing accent — should still resolve
    )
    result = resolve(q, data)
    assert result.status == "resolved", (
        f"Accent normalization failed for 'Mbappe': status={result.status}"
    )
    assert result.aggregated_value == 8


def test_case_insensitive(data):
    """Player names should be case-insensitive."""
    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="messi",
    )
    result = resolve(q, data)
    assert result.status == "resolved"
    assert result.aggregated_value == 7


# ---------------------------------------------------------------------------
# Tests: Data integrity
# ---------------------------------------------------------------------------


def test_hennessey_was_dismissed(data):
    """Hennessey received a red card — was_dismissed should be True."""
    hennessey = [
        p for p in data["player_match_facts"]
        if p["player_name"] == "Wayne Hennessey"
    ]
    assert len(hennessey) >= 1, f"Expected at least 1 Hennessey record, got {len(hennessey)}"
    dismissed = [p for p in hennessey if p["was_dismissed"] is True]
    assert len(dismissed) == 1, (
        f"Expected exactly 1 dismissed Hennessey record, got {len(dismissed)}"
    )


def test_period_filter_returns_honest_partial(data):
    """When by_period data is absent, period filters must return partial.

    The current match_facts.json does not carry per-period breakdowns.
    The resolver detects this and honestly reports the period filter as
    dropped rather than silently returning an unsliced total.
    """
    q = StructuredQuery(
        intent="numeric", entity="player", metric="goals",
        aggregation="sum", entity_name="Messi",
        filters=[Filter("period", "in", [1])])
    result = resolve(q, data)
    assert result.status == "partial", (
        f"Expected partial when by_period is absent, got {result.status}"
    )
    assert "period" in result.dropped_filters


# ---------------------------------------------------------------------------
# Tests: Team-level queries
# ---------------------------------------------------------------------------


def test_team_possession(data):
    """Argentina's possession in a match should be between 0 and 100."""
    # Query team_match_facts directly for Argentina
    argentina_teams = [
        t for t in data["team_match_facts"]
        if "Argentina" in t["team_name"]
    ]
    assert len(argentina_teams) > 0, "No Argentina team records found"
    for record in argentina_teams:
        if record.get("possession_share") is not None:
            assert 0 <= record["possession_share"] <= 100, (
                f"Argentina possession_share={record['possession_share']} out of range"
            )


def test_team_knockout_crosses(data):
    """Teams should have crosses in knockout matches.

    TeamMatchFacts has 'crosses' as a stored metric — use it to verify
    knockout team queries work correctly.
    """
    q = StructuredQuery(
        intent="aggregation",
        entity="team",
        metric="crosses",
        aggregation="sum",
        filters=[Filter("is_knockout", "eq", True)],
        limit=1,
    )
    result = resolve(q, data)
    assert result.status in ("resolved", "partial")
    assert result.aggregated_value is not None and result.aggregated_value > 0, (
        f"Knockout team crosses={result.aggregated_value}"
    )


def test_team_first_shot_minute(data):
    """Team first_shot_minute should resolve for at least one match."""
    q = StructuredQuery(
        intent="aggregation",
        entity="team",
        metric="first_shot_minute",
        aggregation="min",
        limit=1,
    )
    result = resolve(q, data)
    assert result.status in ("resolved", "partial")
    assert result.aggregated_value is not None and result.aggregated_value >= 0, (
        f"Team first_shot_minute={result.aggregated_value}"
    )
