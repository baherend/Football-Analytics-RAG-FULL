"""
test_extraction.py — Unit tests for Phase 1 structured extraction.

Tests run on small samples (specific matches/players), not the full dataset.
Uses the same regression cases as 01_documents.py §8.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.extraction.match_facts import (
    extract_all, persist, load_json,
    _extract_match_facts, _extract_player_match_facts, _extract_team_match_facts,
    DATA_ROOT, COMPETITION_ID, SEASON_ID,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

MATCHES = load_json(DATA_ROOT / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json")
MATCH_INDEX = {m["match_id"]: m for m in MATCHES}


def _load_match(match_id: int):
    events = load_json(DATA_ROOT / "events" / f"{match_id}.json")
    lineups = load_json(DATA_ROOT / "lineups" / f"{match_id}.json")
    match = MATCH_INDEX[match_id]
    return match, events, lineups


# ---------------------------------------------------------------------------
# Tests: Minutes computation (§3)
# ---------------------------------------------------------------------------


def test_hennessey_red_card():
    """Hennessey (3857273) red card truncates minutes to ~83.7."""
    match, events, lineups = _load_match(3857273)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    hennessey = next((p for p in pf if "Hennessey" in p.player_name), None)
    assert hennessey is not None, "Hennessey not found"
    assert abs(hennessey.minutes - 83.7) < 0.5, f"Hennessey minutes={hennessey.minutes}, expected ~83.7"
    assert hennessey.was_dismissed, "Hennessey should be marked as dismissed"


def test_aboubakar_second_yellow():
    """Aboubakar (3857280) second yellow truncates minutes to ~92.9."""
    match, events, lineups = _load_match(3857280)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    aboubakar = next((p for p in pf if "Aboubakar" in p.player_name), None)
    assert aboubakar is not None, "Aboubakar not found"
    assert abs(aboubakar.minutes - 92.9) < 0.5, f"Aboubakar minutes={aboubakar.minutes}, expected ~92.9"
    assert aboubakar.was_dismissed, "Aboubakar should be marked as dismissed"


def test_final_extra_time():
    """Final (3869685) team totals ~1365 minutes each (11 players * 124.1)."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)

    arg_mins = sum(p.minutes for p in pf if p.team_name == "Argentina")
    fra_mins = sum(p.minutes for p in pf if p.team_name == "France")
    assert abs(arg_mins - 1365.0) < 5.0, f"Argentina total={arg_mins:.1f}, expected ~1365"
    assert abs(fra_mins - 1365.0) < 5.0, f"France total={fra_mins:.1f}, expected ~1365"


def test_final_kounde_varane():
    """Koundé ~120.6, Varane ~112.5 in the Final (duplicate segments)."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)

    kounde = next((p for p in pf if "Kound" in p.player_name), None)
    varane = next((p for p in pf if "Varane" in p.player_name), None)
    assert kounde is not None, "Koundé not found"
    assert varane is not None, "Varane not found"
    assert abs(kounde.minutes - 120.6) < 0.5, f"Koundé minutes={kounde.minutes}, expected ~120.6"
    assert abs(varane.minutes - 112.5) < 0.5, f"Varane minutes={varane.minutes}, expected ~112.5"


# ---------------------------------------------------------------------------
# Tests: Shots and goals (§2/§4)
# ---------------------------------------------------------------------------


def test_opening_match_goals():
    """Qatar v Ecuador (3857286): Ecuador scored 2 goals."""
    match, events, lineups = _load_match(3857286)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert len(mf.goals) == 2, f"Expected 2 goals, got {len(mf.goals)}"
    assert all(g.team == "Ecuador" for g in mf.goals), "All goals should be Ecuador"
    assert all(g.player == "Enner Remberto Valencia Lastra" for g in mf.goals), "Scorer should be Valencia"


def test_final_score_excludes_shootout():
    """Final (3869685): score is 3-3, shootout is 4-2."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert mf.home_score == 3, f"Home score={mf.home_score}, expected 3"
    assert mf.away_score == 3, f"Away score={mf.away_score}, expected 3"
    assert mf.shootout_goals.get("Argentina") == 4, f"Argentina shootout={mf.shootout_goals.get('Argentina')}"
    assert mf.shootout_goals.get("France") == 2, f"France shootout={mf.shootout_goals.get('France')}"


def test_shootout_excluded_from_xg():
    """Final: no player should carry shootout xG. Max legitimate xG < 2.0."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    max_xg = max((p.xg for p in pf if p.team_name in ("Argentina", "France")), default=0.0)
    assert max_xg < 2.0, f"Max player xG={max_xg:.2f}, shootout xG may be leaking"


# ---------------------------------------------------------------------------
# Tests: Cards (§5)
# ---------------------------------------------------------------------------


def test_final_card_timing():
    """Paredes card at 113:27, not the truncated lineups value."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    paredes = next((c for c in mf.cards if "Paredes" in c.player), None)
    assert paredes is not None, "Paredes card not found"
    assert paredes.minute == 113, f"Paredes minute={paredes.minute}, expected 113"
    assert paredes.second == 27, f"Paredes second={paredes.second}, expected 27"


def test_card_parity():
    """Card counts from events should match lineups for most matches."""
    match, events, lineups = _load_match(3869685)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert mf.card_parity_ok, f"Card parity failed: events={len(mf.cards)}, lineups={mf.lineup_card_count}"


# ---------------------------------------------------------------------------
# Tests: Possession proxy (§7.4)
# ---------------------------------------------------------------------------


def test_possession_proxy():
    """Possession should sum to ~100% across both teams."""
    match, events, lineups = _load_match(3857286)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    total = sum(v for v in mf.possession.values() if v is not None)
    assert abs(total - 100.0) < 0.1, f"Possession total={total:.1f}%, expected ~100%"


# ---------------------------------------------------------------------------
# Tests: Player ID consistency
# ---------------------------------------------------------------------------


def test_player_id_from_lineups():
    """Player IDs should come from lineups, not events."""
    match, events, lineups = _load_match(3857286)
    from match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    # All players should have a player_id
    null_ids = [p for p in pf if p.player_id is None]
    assert len(null_ids) == 0, f"{len(null_ids)} players with player_id=None"


def test_stage_coverage():
    """All 6 stages should be represented across 64 matches."""
    from collections import Counter
    stages = Counter(m["competition_stage"]["name"] for m in MATCHES)
    expected = {"Group Stage": 48, "Round of 16": 8, "Quarter-finals": 4,
                "Semi-finals": 2, "3rd Place Final": 1, "Final": 1}
    assert dict(stages) == expected, f"Stage counts={dict(stages)}, expected={expected}"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_hennessey_red_card,
        test_aboubakar_second_yellow,
        test_final_extra_time,
        test_final_kounde_varane,
        test_opening_match_goals,
        test_final_score_excludes_shootout,
        test_shootout_excluded_from_xg,
        test_final_card_timing,
        test_card_parity,
        test_possession_proxy,
        test_player_id_from_lineups,
        test_stage_coverage,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    print("Running extraction unit tests...\n")
    failures = run_all_tests()
    raise SystemExit(1 if failures else 0)
