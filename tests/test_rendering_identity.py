"""
test_rendering_identity.py — Competition Identity -> Rendering (Batch 3)

Proves that rendered documents (metadata AND prose text) use the active
dataset's competition/season identity — derived from the StatsBomb match
records at extraction time and persisted in match_facts.json's metadata
block — rather than the module-level WC2022 constants render.py used to
carry independently of src/extraction/match_facts.py.
"""

from __future__ import annotations

import pytest

from src.extraction.match_facts import DatasetIdentity, WC2022_DATASET_IDENTITY
from src.rendering.render import (
    render_level1, render_level2, render_level3, render_level4, render_team, render_all,
)

CUSTOM_IDENTITY = DatasetIdentity(
    competition_id=2, competition_name="Test League",
    season_id=27, season_name="2025/2026",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_match_facts(stage="Group Stage", is_knockout=False, match_id=1):
    return {
        "match_id": match_id,
        "match_date": "2026-01-01",
        "stage": stage,
        "is_knockout": is_knockout,
        "stadium": "Test Stadium",
        "home_team": "Home FC",
        "away_team": "Away FC",
        "home_score": 2,
        "away_score": 1,
        "went_to_extra_time": False,
        "shootout_goals": {},
        "goals": [],
        "own_goals": [],
        "possession": {},
        "cards": [],
        "card_parity_ok": True,
    }


def _fake_player_match_facts(match_id=1):
    return {
        "match_id": match_id, "player_name": "Test Player", "team_name": "Test FC",
        "stage": "Group Stage", "opponent": "Rival FC", "roles": ["Right Back"],
        "was_dismissed": False, "match_date": "2026-01-01",
        "is_knockout": False, "minutes": 90.0, "player_id": 1,
    }


def _fake_player_tournament_facts(match_id=1):
    return {
        "player_id": 1, "player_name": "Test Player", "team_name": "Test FC",
        "match_id": match_id, "match_date": "2026-01-01", "stage": "Group Stage",
        "opponent": "Rival FC", "is_knockout": False,
        "minutes": 90.0, "shots": 1, "xg": 0.1, "goals": 0, "assists": 0,
        "passes_attempted": 20, "passes_completed": 15, "pass_completion_pct": 75.0,
        "passes_under_pressure": 2, "pass_completion_under_pressure_pct": 50.0,
        "shots_inside_box": 1, "shots_outside_box": 0,
        "successful_tackles": 1, "successful_interceptions": 1,
        "clearances": 0, "ball_losses": 2, "carries": 3, "carry_distance": 20.0,
        "pressures": 4, "final_third_passes": 2,
    }


def _fake_team_facts(match_id=1):
    return {
        "team_name": "Test FC", "team_id": 42, "match_id": match_id,
        "match_date": "2026-01-01", "stage": "Group Stage", "opponent": "Rival FC",
        "is_knockout": False, "possession_share": 55.0, "possession_is_proxy": True,
        "play_patterns": {"Regular Play": 10}, "formations": {"4-4-2": 1},
        "pass_types": {"Open Play": 20}, "crosses": 3,
        "first_shot_minute": 5, "first_goal_minute": 10,
    }


# ---------------------------------------------------------------------------
# render_level1
# ---------------------------------------------------------------------------


def test_render_level1_default_identity_is_wc2022():
    doc = render_level1(_fake_match_facts())

    assert doc.metadata["competition_id"] == 43
    assert doc.metadata["competition_name"] == "FIFA World Cup"
    assert doc.metadata["season_id"] == 106
    assert doc.metadata["season_name"] == "2022"
    assert "FIFA World Cup 2022" in doc.text


def test_render_level1_custom_identity():
    doc = render_level1(_fake_match_facts(), dataset_identity=CUSTOM_IDENTITY)

    assert doc.metadata["competition_id"] == 2
    assert doc.metadata["competition_name"] == "Test League"
    assert doc.metadata["season_id"] == 27
    assert doc.metadata["season_name"] == "2025/2026"
    assert "Test League 2025/2026" in doc.text
    assert "FIFA World Cup 2022" not in doc.text


# ---------------------------------------------------------------------------
# render_level2
# ---------------------------------------------------------------------------


def test_render_level2_custom_identity():
    doc = render_level2(_fake_match_facts(), dataset_identity=CUSTOM_IDENTITY)

    assert doc.metadata["competition_id"] == 2
    assert doc.metadata["competition_name"] == "Test League"
    assert doc.metadata["season_id"] == 27
    assert doc.metadata["season_name"] == "2025/2026"
    assert "Test League 2025/2026" in doc.text
    assert "FIFA World Cup 2022" not in doc.text


# ---------------------------------------------------------------------------
# render_level3 (metadata only — no competition prose at this level)
# ---------------------------------------------------------------------------


def test_render_level3_custom_identity():
    doc = render_level3(_fake_player_match_facts(), dataset_identity=CUSTOM_IDENTITY)

    assert doc.metadata["competition_id"] == 2
    assert doc.metadata["competition_name"] == "Test League"
    assert doc.metadata["season_id"] == 27
    assert doc.metadata["season_name"] == "2025/2026"
    assert "FIFA World Cup 2022" not in doc.text


# ---------------------------------------------------------------------------
# render_level4
# ---------------------------------------------------------------------------


def test_render_level4_custom_identity():
    player_facts = [_fake_player_tournament_facts()]
    match_index = {1: {"match_date": "2026-01-01"}}

    doc = render_level4(1, player_facts, match_index, dataset_identity=CUSTOM_IDENTITY)

    assert doc.metadata["competition_id"] == 2
    assert doc.metadata["competition_name"] == "Test League"
    assert doc.metadata["season_id"] == 27
    assert doc.metadata["season_name"] == "2025/2026"
    assert "Test League 2025/2026" in doc.text
    assert "FIFA World Cup 2022" not in doc.text
    assert "Across the competition" in doc.text
    assert "Across the tournament" not in doc.text


# ---------------------------------------------------------------------------
# render_team
# ---------------------------------------------------------------------------


def test_render_team_custom_identity():
    team_facts = [_fake_team_facts()]
    match_index = {1: {"match_date": "2026-01-01"}}

    doc = render_team("Test FC", team_facts, match_index, dataset_identity=CUSTOM_IDENTITY)

    assert doc.metadata["competition_id"] == 2
    assert doc.metadata["competition_name"] == "Test League"
    assert doc.metadata["season_id"] == 27
    assert doc.metadata["season_name"] == "2025/2026"
    assert "Test League 2025/2026" in doc.text
    assert "FIFA World Cup 2022" not in doc.text


# ---------------------------------------------------------------------------
# render_all — identity sourced from facts["metadata"], as persist() writes it
# ---------------------------------------------------------------------------


def _full_facts(metadata=None):
    facts = {
        "match_facts": [_fake_match_facts()],
        "player_match_facts": [_fake_player_tournament_facts()],
        "team_match_facts": [_fake_team_facts()],
    }
    if metadata is not None:
        facts["metadata"] = metadata
    return facts


def test_render_all_derives_identity_from_facts_metadata():
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
        "stage_taxonomy": {
            "stages": ["Group Stage"],
            "knockout_stages": [],
            "group_stages": [],
            "stage_order": [],
        },
    })

    documents = render_all(facts)

    for doc in documents:
        assert doc.metadata["competition_id"] == 2
        assert doc.metadata["competition_name"] == "Test League"
        assert doc.metadata["season_id"] == 27
        assert doc.metadata["season_name"] == "2025/2026"
        assert "FIFA World Cup 2022" not in doc.text

    level1_doc = next(d for d in documents if d.level == "1")
    assert "Test League 2025/2026" in level1_doc.text


def test_render_all_falls_back_to_wc2022_when_metadata_absent():
    """Backward compatibility: facts dicts without a metadata block (as
    every pre-Batch-3 caller and hand-built test fixture produces) still
    render correctly under the WC2022 default identity."""
    facts = _full_facts(metadata=None)

    documents = render_all(facts)

    for doc in documents:
        assert doc.metadata["competition_id"] == 43
        assert doc.metadata["competition_name"] == "FIFA World Cup"
        assert doc.metadata["season_id"] == 106
        assert doc.metadata["season_name"] == "2022"

    level1_doc = next(d for d in documents if d.level == "1")
    assert "FIFA World Cup 2022" in level1_doc.text


def test_render_all_resolves_legacy_wc2022_metadata_with_missing_names():
    """
    The real, already-shipped output/match_facts.json artifact has exactly
    this shape: competition_id=43, season_id=106, no competition_name/
    season_name (it was persisted before Batch 3 added those fields). The
    IDs alone prove it's WC2022, so this is a safe, non-guessing fallback.
    """
    facts = _full_facts(metadata={
        "competition_id": 43, "competition_name": None,
        "season_id": 106, "season_name": None,
    })

    documents = render_all(facts)

    level1_doc = next(d for d in documents if d.level == "1")
    assert level1_doc.metadata["competition_name"] == "FIFA World Cup"
    assert level1_doc.metadata["season_name"] == "2022"
    assert "FIFA World Cup 2022" in level1_doc.text


def test_render_all_raises_for_custom_ids_with_missing_names():
    """
    A metadata block with non-WC2022 IDs and missing names is NOT safe to
    silently resolve to WC2022 (that would mislabel the corpus) and there
    is no way to know the real names — must raise rather than guess.
    """
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": None,
        "season_id": 27, "season_name": None,
    })

    with pytest.raises(ValueError):
        render_all(facts)


def test_render_all_accepts_explicit_dataset_identity_matching_facts_metadata():
    """An explicit identity whose IDs agree with facts["metadata"] is allowed
    — its names may fill in what the persisted metadata was missing."""
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": None,
        "season_id": 27, "season_name": None,
        "stage_taxonomy": {
            "stages": ["Group Stage"],
            "knockout_stages": [],
            "group_stages": [],
            "stage_order": [],
        },
    })
    matching_identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )

    documents = render_all(facts, dataset_identity=matching_identity)

    level1_doc = next(d for d in documents if d.level == "1")
    assert level1_doc.metadata["competition_name"] == "Test League"
    assert "Test League 2025/2026" in level1_doc.text


def test_render_all_explicit_dataset_identity_ignores_absent_metadata():
    """With no metadata block at all, there is nothing to conflict with, so
    an explicit dataset_identity is used freely."""
    facts = _full_facts(metadata=None)
    other_identity = DatasetIdentity(
        competition_id=999, competition_name="Other League",
        season_id=1, season_name="1999",
    )

    documents = render_all(facts, dataset_identity=other_identity)

    level1_doc = next(d for d in documents if d.level == "1")
    assert level1_doc.metadata["competition_name"] == "Other League"
    assert "Other League 1999" in level1_doc.text


def test_render_all_raises_when_explicit_dataset_identity_conflicts_with_facts_metadata():
    """An explicit override can never silently contradict what the data
    itself says — this is the core Problem 2 fix."""
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
    })
    conflicting_identity = DatasetIdentity(
        competition_id=999, competition_name="Other League",
        season_id=1, season_name="1999",
    )

    with pytest.raises(ValueError):
        render_all(facts, dataset_identity=conflicting_identity)


# ---------------------------------------------------------------------------
# Strict identity completeness (final Batch 3 review fix)
#
# _dataset_identity_from_facts() must never accept a structurally partial
# identity (an ID without its pair, or a name without both IDs), and
# _resolve_render_dataset_identity() must require an explicit override to
# match a COMPLETE facts["metadata"] on all four fields, not just the IDs.
# ---------------------------------------------------------------------------


def test_render_all_raises_when_metadata_has_names_but_missing_an_id():
    """competition_id=None with names present must NOT create a DatasetIdentity."""
    facts = _full_facts(metadata={
        "competition_id": None, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
    })

    with pytest.raises(ValueError):
        render_all(facts)


def test_render_all_raises_when_metadata_has_only_one_id():
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": None,
        "season_id": None, "season_name": None,
    })

    with pytest.raises(ValueError):
        render_all(facts)


def test_render_all_raises_when_metadata_has_one_name_missing():
    """Both IDs present, competition_name present, season_name missing —
    structurally partial, must not be treated as the safe 'ids_only' shape."""
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": None,
    })

    with pytest.raises(ValueError):
        render_all(facts)


def test_render_all_accepts_explicit_dataset_identity_matching_complete_metadata():
    """Complete metadata + an explicit identity agreeing on all four fields
    is allowed (this is a legitimate no-op override, not a conflict)."""
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
        "stage_taxonomy": {
            "stages": ["Group Stage"],
            "knockout_stages": [],
            "group_stages": [],
            "stage_order": [],
        },
    })
    same_identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )

    documents = render_all(facts, dataset_identity=same_identity)

    level1_doc = next(d for d in documents if d.level == "1")
    assert level1_doc.metadata["competition_name"] == "Test League"
    assert "Test League 2025/2026" in level1_doc.text


def test_render_all_raises_when_ids_match_but_competition_name_conflicts():
    """
    Same competition_id/season_id, but the explicit identity's
    competition_name disagrees with the complete metadata already
    persisted — comparing IDs alone (the old behavior) would have let this
    through. Names are not disposable once persisted as part of the
    dataset identity.
    """
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
    })
    wrong_name_identity = DatasetIdentity(
        competition_id=2, competition_name="Wrong League",
        season_id=27, season_name="2025/2026",
    )

    with pytest.raises(ValueError):
        render_all(facts, dataset_identity=wrong_name_identity)


def test_render_all_raises_when_ids_match_but_season_name_conflicts():
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": "Test League",
        "season_id": 27, "season_name": "2025/2026",
    })
    wrong_season_name_identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="Wrong Season",
    )

    with pytest.raises(ValueError):
        render_all(facts, dataset_identity=wrong_season_name_identity)


def test_render_all_raises_when_ids_only_metadata_conflicts_with_explicit_ids():
    """Non-WC IDs with missing names, but the explicit identity's IDs
    disagree with facts["metadata"]'s IDs — must raise, not silently
    prefer the explicit identity."""
    facts = _full_facts(metadata={
        "competition_id": 2, "competition_name": None,
        "season_id": 27, "season_name": None,
    })
    mismatched_identity = DatasetIdentity(
        competition_id=3, competition_name="Other League",
        season_id=27, season_name="2025/2026",
    )

    with pytest.raises(ValueError):
        render_all(facts, dataset_identity=mismatched_identity)

def test_render_level4_default_wc2022_keeps_tournament_wording():
    player_facts = [_fake_player_tournament_facts()]
    match_index = {1: {"match_date": "2026-01-01"}}

    doc = render_level4(1, player_facts, match_index)

    assert doc.metadata["competition_id"] == 43
    assert doc.metadata["season_id"] == 106
    assert "Across the tournament" in doc.text
    assert "Across the competition" not in doc.text
