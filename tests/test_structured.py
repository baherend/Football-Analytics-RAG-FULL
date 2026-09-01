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
    assert resolve_metric("هدفًا") == "goals"
    assert resolve_metric("الأهداف") == "goals"
    assert resolve_metric("nonexistent") is None


def test_aggregation_resolution():
    """Aggregation synonyms should resolve correctly."""
    assert resolve_aggregation("sum") == "sum"
    assert resolve_aggregation("total") == "sum"
    assert resolve_aggregation("most") == "max"
    assert resolve_aggregation("highest") == "max"
    assert resolve_aggregation("average") == "avg"
    assert resolve_aggregation("الأكثر") == "max"
    assert resolve_aggregation("أكبر عدد من") == "max"
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
    """Sum of "goals" across every team_match_facts record (no entity_name
    filter) = total goals scored across the whole tournament, since each
    team's record now carries its derived goals value (see
    test_team_goals_are_derived_from_authoritative_match_scores below) and
    every match contributes exactly one home-side and one away-side record.
    Not a per-team superlative (that would use intent="superlative").
    """
    q = StructuredQuery(
        intent="aggregation",
        entity="team",
        metric="goals",
        aggregation="sum",
        limit=1,
    )
    result = resolve(q, data)
    assert result.status == "resolved", f"status={result.status}, explanation={result.explanation!r}"
    # Cross-check against match_facts's own authoritative home_score/away_score
    # totals, independent of the team-record derivation path.
    expected_total = sum(
        (m.get("home_score") or 0) + (m.get("away_score") or 0)
        for m in data["match_facts"]
    )
    assert result.aggregated_value == expected_total, (
        f"aggregated_value={result.aggregated_value!r} != "
        f"sum(home_score + away_score) across match_facts={expected_total}"
    )


def test_team_goals_are_derived_from_authoritative_match_scores():
    """
    Team Comparison goals correctness blocker: team_match_facts records do
    not store a "goals" field directly (see test_team_total_goals above),
    so entity="team", metric="goals" previously silently returned 0 via
    _read_metric_from_record()'s generic STORED-metric default
    (`.get("goals", 0)`) -- not because the team genuinely scored 0. Team
    goals must be derived from match_facts's authoritative home_score/
    away_score fields, matched by match_id and team identity -- the same
    final-score fields StatsBomb itself records -- never guessed from
    player aggregates, semantic text, or an LLM.
    """
    synthetic_data = {
        "player_match_facts": [],
        "match_facts": [
            {"match_id": 1, "home_team": "Alpha FC", "away_team": "Beta FC", "home_score": 2, "away_score": 0},
            {"match_id": 2, "home_team": "Gamma FC", "away_team": "Alpha FC", "home_score": 0, "away_score": 1},
        ],
        "team_match_facts": [
            {"team_id": 1, "team_name": "Alpha FC", "match_id": 1, "crosses": 5},
            {"team_id": 1, "team_name": "Alpha FC", "match_id": 2, "crosses": 3},
            {"team_id": 2, "team_name": "Beta FC", "match_id": 1, "crosses": 2},
            {"team_id": 3, "team_name": "Gamma FC", "match_id": 2, "crosses": 4},
        ],
    }

    q = StructuredQuery(
        intent="numeric", entity="team", metric="goals",
        aggregation="sum", entity_name="Alpha FC",
    )
    result = resolve(q, synthetic_data)

    assert result.status == "resolved", f"status={result.status}, explanation={result.explanation!r}"
    assert result.aggregated_value == 3, (
        "Alpha FC scored 2 (match 1, home_score) + 1 (match 2, away_score) = 3 "
        f"goals total, got aggregated_value={result.aggregated_value!r}"
    )


def test_unsupported_team_metric_is_rejected_not_defaulted_to_zero(data):
    """
    Missing-vs-zero safety net: "assists" is a real registered metric (valid
    for players) but has no team-level source. Before entity-scoped
    validation, entity="team", metric="assists" would silently resolve via
    _read_metric_from_record()'s generic STORED default (`.get("assists", 0)`)
    and look like an authoritative 0 for every team. It must be rejected at
    validation time instead, the same way a genuinely unregistered metric
    name is -- distinguishing "unsupported for this entity" from "actual
    value is 0".
    """
    q = StructuredQuery(
        intent="numeric", entity="team", metric="assists",
        aggregation="sum", entity_name="Argentina",
    )
    result = resolve(q, data)
    assert result.status == "empty", f"status={result.status}, explanation={result.explanation!r}"
    assert "Unknown metric" in result.explanation


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



def test_period_filter_without_data_is_honest_partial():
    """Missing by_period data must produce an honest partial fallback."""
    no_period_data = {
        "player_match_facts": [
            {
                "player_id": 1,
                "player_name": "Test Player",
                "team_name": "Test FC",
                "match_id": 10,
                "passes_attempted": 10,
            }
        ],
        "match_facts": [],
        "team_match_facts": [],
    }

    total = resolve(
        StructuredQuery(
            intent="numeric",
            entity="player",
            metric="passes_attempted",
            aggregation="sum",
            entity_name="Test Player",
        ),
        no_period_data,
    ).aggregated_value

    sliced = resolve(
        StructuredQuery(
            intent="numeric",
            entity="player",
            metric="passes_attempted",
            aggregation="sum",
            entity_name="Test Player",
            filters=[Filter("period", "in", [1])],
        ),
        no_period_data,
    )

    assert sliced.status == "partial", f"status={sliced.status}"
    assert "period" in sliced.dropped_filters
    assert sliced.aggregated_value == total

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



def test_period_filter_returns_honest_partial():
    """Period slicing without by_period data must not pretend to be resolved."""
    no_period_data = {
        "player_match_facts": [
            {
                "player_id": 1,
                "player_name": "Test Player",
                "team_name": "Test FC",
                "match_id": 10,
                "goals": 3,
            }
        ],
        "match_facts": [],
        "team_match_facts": [],
    }

    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Test Player",
        filters=[Filter("period", "in", [1])],
    )

    result = resolve(q, no_period_data)

    assert result.status == "partial"
    assert "period" in result.dropped_filters
    assert result.aggregated_value == 3

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


def test_period_filter_accepts_numeric_string_and_slices_by_period():
    """Legacy Filter period values from parsing may arrive as numeric strings."""
    from src.stage_taxonomy import StageTaxonomy

    data = {
        "player_match_facts": [
            {
                "player_id": 1,
                "player_name": "Test Player",
                "team_name": "Test FC",
                "match_id": 10,
                "stage": "Regular Season",
                "goals": 3,
                "by_period": {
                    "1": {"goals": 1},
                    "2": {"goals": 2},
                },
            }
        ],
        "match_facts": [],
        "team_match_facts": [],
    }

    taxonomy = StageTaxonomy.discover(
        stages=["Regular Season"],
        knockout_stages=[],
        group_stages=[],
    )

    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="goals",
        aggregation="sum",
        entity_name="Test Player",
        filters=[
            Filter("stage", "eq", "Regular Season"),
            Filter("period", "eq", "1"),
        ],
    )

    result = resolve(q, data=data, stage_taxonomy=taxonomy)

    assert result.status == "resolved"
    assert result.aggregated_value == 1
    assert "period" not in result.dropped_filters


def test_derived_ratio_aggregates_components_before_division():
    """Season-level derived ratios must use summed numerator/denominator."""
    data = {
        "player_match_facts": [
            {
                "player_id": 1,
                "player_name": "Test Player",
                "team_name": "Test FC",
                "match_id": 10,
                "passes_completed": 1,
                "passes_attempted": 1,
            },
            {
                "player_id": 1,
                "player_name": "Test Player",
                "team_name": "Test FC",
                "match_id": 11,
                "passes_completed": 0,
                "passes_attempted": 9,
            },
        ],
        "match_facts": [],
        "team_match_facts": [],
    }

    q = StructuredQuery(
        intent="numeric",
        entity="player",
        metric="pass_completion_pct",
        aggregation="avg",
        entity_name="Test Player",
    )

    result = resolve(q, data)

    assert result.status == "resolved"
    assert result.aggregated_value == 10.0


def test_superlative_derived_ratio_ranks_by_combined_components():
    """Superlative ratio ranking must use combined numerator/denominator."""
    data = {
        "player_match_facts": [
            {
                "player_id": 1,
                "player_name": "Player A",
                "team_name": "Test FC",
                "match_id": 1,
                "passes_completed": 1,
                "passes_attempted": 1,
            },
            {
                "player_id": 1,
                "player_name": "Player A",
                "team_name": "Test FC",
                "match_id": 2,
                "passes_completed": 0,
                "passes_attempted": 9,
            },
            {
                "player_id": 2,
                "player_name": "Player B",
                "team_name": "Test FC",
                "match_id": 3,
                "passes_completed": 6,
                "passes_attempted": 10,
            },
        ],
        "match_facts": [],
        "team_match_facts": [],
    }

    q = StructuredQuery(
        intent="superlative",
        entity="player",
        metric="pass_completion_pct",
        aggregation="max",
        limit=1,
        sort_order="desc",
    )

    result = resolve(q, data)

    assert result.status == "resolved"
    assert result.data[0]["player_name"] == "Player B"
    assert result.aggregated_value == 60.0


def test_player_name_resolution_requires_all_supplied_name_words():
    """A multi-word partial name must not resolve from one shared first name."""
    from src.query.resolver import _resolve_player_name

    data = {
        "player_match_facts": [
            {"player_name": "Sergio Germ\u00e1n Romero"},
            {"player_name": "Sergio Leonel Ag\u00fcero del Castillo"},
            {"player_name": "Arouna Kon\u00e9"},
        ]
    }

    assert _resolve_player_name("Sergio Aguero", data) == "Sergio Leonel Ag\u00fcero del Castillo"
    assert _resolve_player_name("Arouna Kone", data) == "Arouna Kon\u00e9"
