"""
test_stage_taxonomy.py — Competition Stage Taxonomy + Knockout Classification

Proves that knockout and group-stage status come from an explicit
StageTaxonomy rather than being guessed from stage-name comparisons (e.g.
`stage != "Group Stage"`) or derived from each other (`not is_knockout` is
NOT the same thing as "group stage" — a league match is neither), and that
a league competition — arbitrary stage names, no knockout phase, no group
phase — works through extraction, structured validation, and rendering
without crashing, being rejected, or being mislabeled.
"""

from __future__ import annotations

import pytest

from src.stage_taxonomy import StageTaxonomy, WC2022_STAGE_TAXONOMY
from src.extraction.match_facts import (
    _extract_match_facts, DATA_ROOT, COMPETITION_ID, SEASON_ID, load_json,
)
from src.query.query_schema import StructuredQuery, Filter
from src.query.vocab import validate_query
from src.rendering.render import render_team, render_level4, render_all


# ---------------------------------------------------------------------------
# 1. League-style stages are accepted
# ---------------------------------------------------------------------------


def test_validate_query_accepts_league_stage_with_custom_taxonomy():
    league_taxonomy = StageTaxonomy.discover(
        stages=["Regular Season", "Matchday 1", "Matchday 2"],
        knockout_stages=[],
    )
    q = StructuredQuery(
        intent="numeric", entity="player", metric="goals",
        aggregation="sum", entity_name="Some Player",
        filters=[Filter("stage", "eq", "Matchday 1")],
    )
    result = validate_query(q, stage_taxonomy=league_taxonomy)
    assert result.ok, f"league stage rejected: {result.issues}"


def test_validate_query_still_rejects_league_stage_under_wc2022_default():
    """The fix is additive: the WC2022 default vocabulary is untouched."""
    q = StructuredQuery(
        intent="numeric", entity="player", metric="goals",
        aggregation="sum", entity_name="Some Player",
        filters=[Filter("stage", "eq", "Matchday 1")],
    )
    result = validate_query(q)  # no taxonomy => WC2022 default
    assert not result.ok
    assert any(i.field == "stage" for i in result.issues)


# ---------------------------------------------------------------------------
# 2. A competition with no knockout stages produces is_knockout=False
# ---------------------------------------------------------------------------


def _fake_match(stage_name: str) -> dict:
    return {
        "match_id": 999001,
        "match_date": "2026-01-01",
        "competition_stage": {"name": stage_name},
        "stadium": {"name": "Test Stadium"},
        "home_team": {"home_team_name": "Home FC", "home_team_id": 1},
        "away_team": {"away_team_name": "Away FC", "away_team_id": 2},
        "home_score": 1,
        "away_score": 0,
    }


def test_league_with_no_knockout_stages_is_never_knockout():
    league_taxonomy = StageTaxonomy.discover(
        stages=["Matchday 1", "Matchday 2", "Final"],  # "Final" reused as a plain league label
        knockout_stages=[],  # explicitly no knockout stages
    )
    for stage_name in ("Matchday 1", "Matchday 2", "Final"):
        match = _fake_match(stage_name)
        mf = _extract_match_facts(match, events=[], lineups=[], stage_taxonomy=league_taxonomy)
        assert mf.is_knockout is False, f"stage={stage_name!r} incorrectly flagged knockout"


# ---------------------------------------------------------------------------
# 3. WC2022 default knockout behavior remains unchanged
# ---------------------------------------------------------------------------

MATCHES = load_json(DATA_ROOT / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json")
MATCH_INDEX = {m["match_id"]: m for m in MATCHES}


def _load_real_match(match_id: int):
    events = load_json(DATA_ROOT / "events" / f"{match_id}.json")
    lineups = load_json(DATA_ROOT / "lineups" / f"{match_id}.json")
    return MATCH_INDEX[match_id], events, lineups


def test_wc2022_final_is_knockout_by_default():
    """Final (3869685) — no stage_taxonomy argument passed, default applies."""
    match, events, lineups = _load_real_match(3869685)
    mf = _extract_match_facts(match, events, lineups)
    assert mf.stage == "Final"
    assert mf.is_knockout is True


def test_wc2022_group_stage_match_is_not_knockout_by_default():
    """Qatar v Ecuador (3857286) — no stage_taxonomy argument passed, default applies."""
    match, events, lineups = _load_real_match(3857286)
    mf = _extract_match_facts(match, events, lineups)
    assert mf.stage == "Group Stage"
    assert mf.is_knockout is False


# ---------------------------------------------------------------------------
# 4. Rendering does not crash on a non-WC2022 stage
# ---------------------------------------------------------------------------


def _fake_team_fact(stage: str, match_id: int, match_date: str) -> dict:
    return {
        "team_name": "Testville FC",
        "team_id": 42,
        "match_id": match_id,
        "match_date": match_date,
        "stage": stage,
        "opponent": "Rival FC",
        "is_knockout": False,
        "possession_share": 55.0,
        "possession_is_proxy": True,
        "play_patterns": {"Regular Play": 10},
        "formations": {"4-4-2": 1},
        "pass_types": {"Open Play": 20},
        "crosses": 3,
        "first_shot_minute": 5,
        "first_goal_minute": 10,
    }


def test_render_team_does_not_crash_on_league_stages():
    league_taxonomy = StageTaxonomy.discover(
        stages=["Matchday 1", "Matchday 2", "Matchday 3"], knockout_stages=[],
    )
    team_facts = [
        _fake_team_fact("Matchday 1", 1, "2026-01-01"),
        _fake_team_fact("Matchday 2", 2, "2026-01-08"),
        _fake_team_fact("Matchday 3", 3, "2026-01-15"),
    ]
    match_index = {tf["match_id"]: {"match_date": tf["match_date"]} for tf in team_facts}

    doc = render_team("Testville FC", team_facts, match_index, stage_taxonomy=league_taxonomy)

    # No fixed stage_order for the league -> falls back to the chronologically
    # last match rather than crashing on list.index().
    assert doc.metadata["furthest_stage"] == "Matchday 3"


def test_render_team_default_taxonomy_matches_previous_wc2022_ranking():
    """WC2022's real knockout-progression ranking is preserved unchanged."""
    team_facts = [
        _fake_team_fact("Group Stage", 1, "2022-11-22"),
        _fake_team_fact("Round of 16", 2, "2022-12-03"),
        _fake_team_fact("Quarter-finals", 3, "2022-12-09"),
    ]
    match_index = {tf["match_id"]: {"match_date": tf["match_date"]} for tf in team_facts}

    doc = render_team("Testville FC", team_facts, match_index)  # default: WC2022 taxonomy

    assert doc.metadata["furthest_stage"] == "Quarter-finals"


def _fake_player_fact(stage: str, is_knockout: bool, match_id: int) -> dict:
    return {
        "player_id": 1, "player_name": "Test Player", "team_name": "Testville FC",
        "match_id": match_id, "match_date": "2026-01-01", "stage": stage,
        "opponent": "Rival FC", "is_knockout": is_knockout,
        "minutes": 90.0, "shots": 1, "xg": 0.1, "goals": 0, "assists": 0,
        "passes_attempted": 20, "passes_completed": 15, "pass_completion_pct": 75.0,
        "passes_under_pressure": 2, "pass_completion_under_pressure_pct": 50.0,
        "shots_inside_box": 1, "shots_outside_box": 0,
        "successful_tackles": 1, "successful_interceptions": 1,
        "clearances": 0, "ball_losses": 2, "carries": 3, "carry_distance": 20.0,
        "pressures": 4, "final_third_passes": 2,
    }


def test_render_level4_does_not_crash_and_has_no_knockout_matches_for_league():
    """
    Regression for the reviewer-found bug: a league match is neither
    knockout NOR group stage. Asserting group_matches == 0 in addition to
    knockout_matches == 0 proves group is not derived as `not is_knockout`.
    """
    player_facts = [
        _fake_player_fact("Matchday 1", is_knockout=False, match_id=1),
        _fake_player_fact("Matchday 2", is_knockout=False, match_id=2),
    ]
    match_index = {
        1: {"match_date": "2026-01-01"},
        2: {"match_date": "2026-01-08"},
    }
    league_taxonomy = StageTaxonomy.discover(stages=["Matchday 1", "Matchday 2"], knockout_stages=[])
    doc = render_level4(player_id=1, player_facts=player_facts, match_index=match_index, stage_taxonomy=league_taxonomy)

    assert doc.metadata["knockout_matches"] == 0
    assert doc.metadata["group_matches"] == 0

    text_lower = doc.text.lower()
    assert "group stage" not in text_lower, f"false group-stage claim in text: {doc.text!r}"
    assert "knockout" not in text_lower, f"false knockout claim in text: {doc.text!r}"


def test_render_all_threads_taxonomy_into_level4_for_league(monkeypatch):
    """render_all() must pass the active taxonomy to render_level4(), not just render_team()."""
    import src.rendering.render as render_module

    league_taxonomy = StageTaxonomy.discover(stages=["Matchday 1", "Matchday 2"], knockout_stages=[])
    facts = {
        "match_facts": [],
        "player_match_facts": [
            _fake_player_fact("Matchday 1", is_knockout=False, match_id=1),
        ],
        "team_match_facts": [],
    }

    received = {}
    real_render_level4 = render_module.render_level4

    def spy_render_level4(player_id, player_facts, match_index, stage_taxonomy=WC2022_STAGE_TAXONOMY):
        received["stage_taxonomy"] = stage_taxonomy
        # match_index isn't populated for this player's match_id in this
        # minimal fixture (facts["match_facts"] is deliberately empty to
        # avoid needing full Level 1/2 match fixtures) — only the
        # taxonomy-threading argument matters for this test.
        return real_render_level4(player_id, player_facts, {1: {"match_date": "2026-01-01"}}, stage_taxonomy=stage_taxonomy)

    monkeypatch.setattr(render_module, "render_level4", spy_render_level4)

    documents = render_all(facts, stage_taxonomy=league_taxonomy)

    assert received["stage_taxonomy"] is league_taxonomy
    level4_docs = [d for d in documents if d.level == "4"]
    assert len(level4_docs) == 1
    assert level4_docs[0].metadata["group_matches"] == 0
    assert level4_docs[0].metadata["knockout_matches"] == 0


# ---------------------------------------------------------------------------
# Taxonomy unit behavior
# ---------------------------------------------------------------------------


def test_league_taxonomy_supports_empty_knockout_stages():
    league_taxonomy = StageTaxonomy.discover(stages=["Regular Season"], knockout_stages=[])
    assert league_taxonomy.knockout_stages == frozenset()
    assert league_taxonomy.is_knockout("Regular Season") is False
    assert league_taxonomy.is_valid_stage("Regular Season") is True
    assert league_taxonomy.is_valid_stage("Unknown Stage") is False


def test_league_taxonomy_group_and_knockout_both_false_for_any_stage():
    """A league has neither a group phase nor a knockout phase."""
    league_taxonomy = StageTaxonomy.discover(
        stages=["Regular Season", "Matchday 1"], knockout_stages=[], group_stages=[],
    )
    for stage in ("Regular Season", "Matchday 1", "Unknown Stage"):
        assert league_taxonomy.is_group_stage(stage) is False
        assert league_taxonomy.is_knockout(stage) is False


def test_wc2022_taxonomy_knockout_stages_explicit():
    assert WC2022_STAGE_TAXONOMY.is_knockout("Group Stage") is False
    for stage in ("Round of 16", "Quarter-finals", "Semi-finals", "3rd Place Final", "Final"):
        assert WC2022_STAGE_TAXONOMY.is_knockout(stage) is True


def test_wc2022_taxonomy_group_stage_explicit():
    assert WC2022_STAGE_TAXONOMY.is_group_stage("Group Stage") is True
    assert WC2022_STAGE_TAXONOMY.is_group_stage("Final") is False
    for stage in ("Round of 16", "Quarter-finals", "Semi-finals", "3rd Place Final"):
        assert WC2022_STAGE_TAXONOMY.is_group_stage(stage) is False


def test_stage_taxonomy_rejects_overlapping_group_and_knockout_stages():
    """A stage must never be silently both group and knockout."""
    with pytest.raises(ValueError):
        StageTaxonomy.discover(
            stages=["Ambiguous Stage"],
            knockout_stages=["Ambiguous Stage"],
            group_stages=["Ambiguous Stage"],
        )
