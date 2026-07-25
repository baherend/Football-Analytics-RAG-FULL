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
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    hennessey = next((p for p in pf if "Hennessey" in p.player_name), None)
    assert hennessey is not None, "Hennessey not found"
    assert abs(hennessey.minutes - 83.7) < 0.5, f"Hennessey minutes={hennessey.minutes}, expected ~83.7"
    assert hennessey.was_dismissed, "Hennessey should be marked as dismissed"


def test_aboubakar_second_yellow():
    """Aboubakar (3857280) second yellow truncates minutes to ~92.9."""
    match, events, lineups = _load_match(3857280)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)
    aboubakar = next((p for p in pf if "Aboubakar" in p.player_name), None)
    assert aboubakar is not None, "Aboubakar not found"
    assert abs(aboubakar.minutes - 92.9) < 0.5, f"Aboubakar minutes={aboubakar.minutes}, expected ~92.9"
    assert aboubakar.was_dismissed, "Aboubakar should be marked as dismissed"


def test_final_extra_time():
    """Final (3869685) team totals ~1365 minutes each (11 players * 124.1)."""
    match, events, lineups = _load_match(3869685)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    pf = _extract_player_match_facts(match, events, lineups, mf)

    arg_mins = sum(p.minutes for p in pf if p.team_name == "Argentina")
    fra_mins = sum(p.minutes for p in pf if p.team_name == "France")
    assert abs(arg_mins - 1365.0) < 5.0, f"Argentina total={arg_mins:.1f}, expected ~1365"
    assert abs(fra_mins - 1365.0) < 5.0, f"France total={fra_mins:.1f}, expected ~1365"


def test_final_kounde_varane():
    """Koundé ~120.6, Varane ~112.5 in the Final (duplicate segments)."""
    match, events, lineups = _load_match(3869685)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
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
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert len(mf.goals) == 2, f"Expected 2 goals, got {len(mf.goals)}"
    assert all(g.team == "Ecuador" for g in mf.goals), "All goals should be Ecuador"
    assert all(g.player == "Enner Remberto Valencia Lastra" for g in mf.goals), "Scorer should be Valencia"


def test_final_score_excludes_shootout():
    """Final (3869685): score is 3-3, shootout is 4-2."""
    match, events, lineups = _load_match(3869685)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert mf.home_score == 3, f"Home score={mf.home_score}, expected 3"
    assert mf.away_score == 3, f"Away score={mf.away_score}, expected 3"
    assert mf.shootout_goals.get("Argentina") == 4, f"Argentina shootout={mf.shootout_goals.get('Argentina')}"
    assert mf.shootout_goals.get("France") == 2, f"France shootout={mf.shootout_goals.get('France')}"


def test_shootout_excluded_from_xg():
    """Final: no player should carry shootout xG. Max legitimate xG < 2.0."""
    match, events, lineups = _load_match(3869685)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
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
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    paredes = next((c for c in mf.cards if "Paredes" in c.player), None)
    assert paredes is not None, "Paredes card not found"
    assert paredes.minute == 113, f"Paredes minute={paredes.minute}, expected 113"
    assert paredes.second == 27, f"Paredes second={paredes.second}, expected 27"


def test_card_parity():
    """Card counts from events should match lineups for most matches."""
    match, events, lineups = _load_match(3869685)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    assert mf.card_count_matches, f"Card parity failed: events={len(mf.cards)}, lineups={mf.lineup_card_count}"


# ---------------------------------------------------------------------------
# Tests: Possession proxy (§7.4)
# ---------------------------------------------------------------------------


def test_possession_proxy():
    """Possession should sum to ~100% across both teams."""
    match, events, lineups = _load_match(3857286)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
    mf = extract_mf(match, events, lineups)
    total = sum(v for v in mf.possession.values() if v is not None)
    assert abs(total - 100.0) < 0.1, f"Possession total={total:.1f}%, expected ~100%"


# ---------------------------------------------------------------------------
# Tests: Player ID consistency
# ---------------------------------------------------------------------------


def test_player_id_from_lineups():
    """Player IDs should come from lineups, not events."""
    match, events, lineups = _load_match(3857286)
    from src.extraction.match_facts import _extract_match_facts as extract_mf
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
# Tests: Full-pipeline document structure (§8 structural checks)
# ---------------------------------------------------------------------------


def test_document_counts():
    """Pipeline should produce 64 L1, 64 L2, 32 team documents."""
    docs_path = Path("output/documents.json")
    if not docs_path.exists():
        import pytest; pytest.skip("documents.json not found — run generate_documents.py first")
    import json
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    from collections import Counter
    counts = Counter(d["level"] for d in docs)
    assert counts.get("1", 0) == 64, f"Level 1: {counts.get('1', 0)}, expected 64"
    assert counts.get("2", 0) == 64, f"Level 2: {counts.get('2', 0)}, expected 64"
    assert counts.get("team", 0) == 32, f"Team: {counts.get('team', 0)}, expected 32"
    assert len(docs) == 2835, f"Total: {len(docs)}, expected 2835"


def test_document_ids_unique():
    """All document_id values must be unique."""
    docs_path = Path("output/documents.json")
    if not docs_path.exists():
        import pytest; pytest.skip("documents.json not found")
    import json
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    ids = [d["document_id"] for d in docs]
    assert len(ids) == len(set(ids)), f"Duplicate document_ids found: {len(ids)} total, {len(set(ids))} unique"


def test_document_texts_nonempty():
    """Every document must have non-empty text."""
    docs_path = Path("output/documents.json")
    if not docs_path.exists():
        import pytest; pytest.skip("documents.json not found")
    import json
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    empty = [d["document_id"] for d in docs if not d.get("text", "").strip()]
    assert len(empty) == 0, f"{len(empty)} documents with empty text: {empty[:5]}"


def test_final_l1_text_contains_scores():
    """Final L1 document text must contain '3-3' and '4-2' separately."""
    docs_path = Path("output/documents.json")
    if not docs_path.exists():
        import pytest; pytest.skip("documents.json not found")
    import json
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    final_l1 = next((d for d in docs if d["document_id"] == "L1-match-3869685"), None)
    assert final_l1 is not None, "Final L1 document not found"
    text = final_l1["text"]
    assert "3-3" in text, f"Final L1 text missing '3-3': {text[:200]}"
    assert "4-2" in text, f"Final L1 text missing '4-2': {text[:200]}"


def test_unused_subs_no_level3_doc():
    """Players with empty positions (unused subs) produce no Level 3 document.
    Checks that all L3 documents have minutes > 0."""
    docs_path = Path("output/documents.json")
    if not docs_path.exists():
        import pytest; pytest.skip("documents.json not found")
    import json
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    l3_docs = [d for d in docs if d["level"] == "3"]
    zero_mins = [d for d in l3_docs if d.get("metadata", {}).get("minutes", 1) <= 0]
    assert len(zero_mins) == 0, f"{len(zero_mins)} L3 documents with minutes <= 0"


def test_empty_position_rows_count():
    """Full extraction should find exactly 1249 empty-position rows (unused subs)."""
    from src.extraction.minutes_played import build_match_context, minutes_played
    matches = load_json(DATA_ROOT / "matches" / str(COMPETITION_ID) / f"{SEASON_ID}.json")
    empty_count = 0
    for match in matches:
        match_id = match["match_id"]
        events = load_json(DATA_ROOT / "events" / f"{match_id}.json")
        lineups = load_json(DATA_ROOT / "lineups" / f"{match_id}.json")
        for team_data in lineups:
            for player in team_data["lineup"]:
                if not (player.get("positions") or []):
                    empty_count += 1
    assert empty_count == 1249, f"Empty position rows: {empty_count}, expected 1249"


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
        test_document_counts,
        test_document_ids_unique,
        test_document_texts_nonempty,
        test_final_l1_text_contains_scores,
        test_unused_subs_no_level3_doc,
        test_empty_position_rows_count,
    ]

    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print(f"  PASS {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    print("Running extraction unit tests...\n")
    failures = run_all_tests()
    raise SystemExit(1 if failures else 0)
