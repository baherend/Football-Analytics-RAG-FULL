"""
test_semantic_ground_truth.py — Tests for the Semantic Ground Truth Foundation

Tests verify:
- Dataset size and case-group distribution (24 cases, 4 per group)
- Original six pilot cases are preserved (canonical SHA-256 check)
- Foundation twelve-case preservation (canonical SHA-256 check)
- Metadata snapshot integrity
- Document existence in chunks.json
- Evidence snippet verbatim presence
- Level representation
- Identity regressions (Final match, L3 pilot, L4 summary, Team possession)
- Expanded case identities (England vs Iran, Griezmann, Mbappé, Morocco, etc.)
- Advanced case identities (shootout matches, Morocco vs Portugal QF, Mbappé vs Poland, etc.)
- Extended case identities (England vs France QF, Enzo Fernández, Germany, cross-team comparison)
- Correction-specific regression checks (Quarter-finals evidence, 'off t' truncation)
- Full validation pass with zero errors
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.ground_truth.semantic import (
    SEMANTIC_GROUND_TRUTH,
    SEMANTIC_GROUND_TRUTH_METADATA,
    EXPECTED_CASE_IDS,
    FOUNDATION_TWELVE_CASES_SHA256,
    ORIGINAL_PILOT_CASE_IDS,
    ORIGINAL_PILOT_CASES_SHA256,
    compute_canonical_case_hash,
    index_chunks_by_document_id,
    load_chunks,
    validate_semantic_ground_truth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chunks():
    """Load chunks.json once per module."""
    return load_chunks(SEMANTIC_GROUND_TRUTH_METADATA["chunks_path"])


@pytest.fixture(scope="module")
def chunks_by_doc(chunks):
    """Index chunks by document_id once per module."""
    return index_chunks_by_document_id(chunks)


@pytest.fixture(scope="module")
def cases_by_id():
    """Index ground-truth cases by ID."""
    return {c["id"]: c for c in SEMANTIC_GROUND_TRUTH}


# ---------------------------------------------------------------------------
# 1. Dataset Size and Distribution
# ---------------------------------------------------------------------------

def test_semantic_ground_truth_has_exactly_twenty_four_cases():
    """Verify exactly 24 total cases with all expected IDs and 4 per group."""
    assert len(SEMANTIC_GROUND_TRUTH) == 24, (
        f"Expected 24 cases, found {len(SEMANTIC_GROUND_TRUTH)}"
    )
    actual_ids = {c["id"] for c in SEMANTIC_GROUND_TRUTH}
    expected_ids = set(EXPECTED_CASE_IDS)
    assert actual_ids == expected_ids, (
        f"ID mismatch.\nMissing: {expected_ids - actual_ids}\n"
        f"Extra: {actual_ids - expected_ids}"
    )
    groups: dict[str, int] = {}
    for c in SEMANTIC_GROUND_TRUTH:
        g = c["case_group"]
        groups[g] = groups.get(g, 0) + 1
    for group_name in ("l1", "l2", "l3", "l4", "team", "multi"):
        assert groups.get(group_name) == 4, (
            f"Group '{group_name}' has {groups.get(group_name)} cases, expected 4"
        )


# ---------------------------------------------------------------------------
# 2. Original Pilot Preservation
# ---------------------------------------------------------------------------

def test_original_six_pilot_cases_are_unchanged():
    """Verify the original six pilot cases have the same canonical SHA-256."""
    pilot_cases = [
        c for c in SEMANTIC_GROUND_TRUTH if c["id"] in ORIGINAL_PILOT_CASE_IDS
    ]
    assert len(pilot_cases) == 6, (
        f"Expected 6 pilot cases, found {len(pilot_cases)}"
    )
    actual_hash = compute_canonical_case_hash(pilot_cases)
    assert actual_hash == ORIGINAL_PILOT_CASES_SHA256, (
        f"Original pilot cases hash changed.\n"
        f"Expected: {ORIGINAL_PILOT_CASES_SHA256}\n"
        f"Actual:   {actual_hash}\n"
        f"One or more original pilot cases were modified."
    )


# ---------------------------------------------------------------------------
# 2b. Foundation Twelve-Case Preservation
# ---------------------------------------------------------------------------

def test_verified_twelve_case_foundation_is_unchanged():
    """Verify the foundation 12 cases have the same canonical SHA-256."""
    foundation_ids = set(EXPECTED_CASE_IDS[:12])
    foundation_cases = [
        c for c in SEMANTIC_GROUND_TRUTH if c["id"] in foundation_ids
    ]
    assert len(foundation_cases) == 12, (
        f"Expected 12 foundation cases, found {len(foundation_cases)}"
    )
    actual_hash = compute_canonical_case_hash(foundation_cases)
    assert actual_hash == FOUNDATION_TWELVE_CASES_SHA256, (
        f"Foundation twelve-case hash changed.\n"
        f"Expected: {FOUNDATION_TWELVE_CASES_SHA256}\n"
        f"Actual:   {actual_hash}\n"
        f"One or more foundation cases were modified."
    )


# ---------------------------------------------------------------------------
# 3. Metadata Snapshot
# ---------------------------------------------------------------------------

def test_metadata_snapshot_matches_chunks(chunks):
    """Verify the stored SHA-256 matches the actual chunks.json."""
    import hashlib
    from pathlib import Path

    path = Path(SEMANTIC_GROUND_TRUTH_METADATA["chunks_path"])
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_sha == SEMANTIC_GROUND_TRUTH_METADATA["chunks_sha256"], (
        f"chunks SHA-256 mismatch.\n"
        f"Stored: {SEMANTIC_GROUND_TRUTH_METADATA['chunks_sha256']}\n"
        f"Actual: {actual_sha}"
    )


# ---------------------------------------------------------------------------
# 4. Document IDs Exist
# ---------------------------------------------------------------------------

def test_all_document_ids_exist_in_chunks(chunks_by_doc, cases_by_id):
    """Verify every relevant and optional document ID exists in chunks."""
    for case_id, case in cases_by_id.items():
        for doc_id in case.get("relevant_document_ids", []):
            assert doc_id in chunks_by_doc, (
                f"[{case_id}] relevant document '{doc_id}' not in chunks"
            )
        for doc_id in case.get("optional_relevant_document_ids", []):
            assert doc_id in chunks_by_doc, (
                f"[{case_id}] optional document '{doc_id}' not in chunks"
            )


# ---------------------------------------------------------------------------
# 5. Evidence Snippets
# ---------------------------------------------------------------------------

def test_all_evidence_snippets_exist_verbatim(chunks_by_doc, cases_by_id):
    """Verify every evidence snippet is an exact substring of its source doc."""
    for case_id, case in cases_by_id.items():
        for fact in case.get("required_facts", []):
            fid = fact["fact_id"]
            src_ids = fact.get("source_document_ids", [])
            for snippet in fact.get("evidence_snippets", []):
                found = False
                for src_id in src_ids:
                    for chunk in chunks_by_doc.get(src_id, []):
                        if snippet in chunk.get("text", ""):
                            found = True
                            break
                    if found:
                        break
                assert found, (
                    f"[{case_id}][{fid}] evidence snippet not found: "
                    f"'{snippet[:100]}...'"
                )


# ---------------------------------------------------------------------------
# 6. Level Representation
# ---------------------------------------------------------------------------

def test_primary_level_represented_by_relevant_documents(chunks_by_doc, cases_by_id):
    """Verify each case's primary_level appears in its relevant documents."""
    for case_id, case in cases_by_id.items():
        primary = case["primary_level"]
        rel_docs = case.get("relevant_document_ids", [])
        doc_levels = set()
        for doc_id in rel_docs:
            for chunk in chunks_by_doc.get(doc_id, []):
                doc_levels.add(chunk.get("level"))
        assert primary in doc_levels, (
            f"[{case_id}] primary_level '{primary}' not in document levels {doc_levels}"
        )


# ---------------------------------------------------------------------------
# 7. Final Identity Regression
# ---------------------------------------------------------------------------

def test_final_identity_is_3869685(cases_by_id, chunks_by_doc):
    """Regression: 3869685 is the Argentina vs France Final, NOT 3857270."""
    # Check L2 Final document
    l2_case = cases_by_id["gt-pilot-l2-01"]
    l2_doc_id = l2_case["relevant_document_ids"][0]
    assert "3869685" in l2_doc_id, f"L2 Final doc should reference 3869685, got {l2_doc_id}"

    l2_chunks = chunks_by_doc[l2_doc_id]
    full_text = " ".join(c["text"] for c in l2_chunks)
    assert "Argentina" in full_text
    assert "France" in full_text

    # Ensure no Final case references 3857270
    for case_id, case in cases_by_id.items():
        for doc_id in case.get("relevant_document_ids", []):
            if "Final" in case.get("notes", "") or "Final" in case.get("query", ""):
                assert "3857270" not in doc_id, (
                    f"[{case_id}] Final case references 3857270 (Portugal vs Uruguay)"
                )


# ---------------------------------------------------------------------------
# 8. L3 Pilot Identity
# ---------------------------------------------------------------------------

def test_l3_pilot_is_messi_vs_croatia(chunks_by_doc, cases_by_id):
    """Verify gt-pilot-l3-01 is Messi vs Croatia Semi-finals."""
    case = cases_by_id["gt-pilot-l3-01"]
    doc_id = case["relevant_document_ids"][0]
    chunks = chunks_by_doc[doc_id]
    text = chunks[0]["text"]

    assert "Messi" in text or "Lionel" in text, f"Expected Messi in text: {text[:200]}"
    assert "Croatia" in text, f"Expected Croatia in text: {text[:200]}"
    assert "Semi" in text, f"Expected Semi-finals in text: {text[:200]}"


# ---------------------------------------------------------------------------
# 9. L4 Tournament Summary
# ---------------------------------------------------------------------------

def test_l4_pilot_is_tournament_summary(chunks_by_doc, cases_by_id):
    """Verify gt-pilot-l4-01 is a tournament summary, not a match doc."""
    case = cases_by_id["gt-pilot-l4-01"]
    doc_id = case["relevant_document_ids"][0]

    assert doc_id.startswith("L4-"), f"Expected L4 prefix, got {doc_id}"
    chunks = chunks_by_doc[doc_id]
    text = chunks[0]["text"]
    assert "tournament" in text.lower() or "matches at" in text.lower(), (
        f"Expected tournament summary language: {text[:200]}"
    )


# ---------------------------------------------------------------------------
# 10. Team Possession Proxy
# ---------------------------------------------------------------------------

def test_team_possession_proxy_limitation_preserved(cases_by_id):
    """Verify possession is described as event-share proxy, not broadcast."""
    for case_id in ("gt-pilot-team-01", "gt-team-02", "gt-team-03", "gt-team-04"):
        case = cases_by_id[case_id]
        for fact in case.get("required_facts", []):
            claim = fact.get("claim", "")
            if "possession" in claim.lower():
                assert "event-share proxy" in claim.lower() or "proxy" in claim.lower(), (
                    f"[{case_id}][{fact['fact_id']}] possession claim does not "
                    f"mention proxy limitation: {claim}"
                )


# ---------------------------------------------------------------------------
# 11. New L1 Case: England vs Iran
# ---------------------------------------------------------------------------

def test_new_l1_case_identifies_england_vs_iran(chunks_by_doc, cases_by_id):
    """Verify gt-l1-02 is the England vs Iran Group Stage match."""
    case = cases_by_id["gt-l1-02"]
    assert case["case_group"] == "l1"
    assert case["primary_level"] == "1"

    doc_id = case["relevant_document_ids"][0]
    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})

    assert chunks[0].get("level") == "1"
    assert meta.get("home_team") == "England" or meta.get("away_team") == "England"
    assert meta.get("home_team") == "Iran" or meta.get("away_team") == "Iran"

    # Verify evidence snippets
    full_text = " ".join(c["text"] for c in chunks)
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-l1-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 12. New L2 Case: Argentina vs Croatia Substitutions
# ---------------------------------------------------------------------------

def test_new_l2_case_identifies_argentina_croatia_substitutions(chunks_by_doc, cases_by_id):
    """Verify gt-l2-02 is Argentina vs Croatia Semi-finals with substitution evidence."""
    case = cases_by_id["gt-l2-02"]
    assert case["case_group"] == "l2"
    assert case["primary_level"] == "2"

    doc_id = case["relevant_document_ids"][0]
    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})
    full_text = " ".join(c["text"] for c in chunks)

    assert chunks[0].get("level") == "2"
    assert meta.get("stage") == "Semi-finals"
    home = meta.get("home_team", "")
    away = meta.get("away_team", "")
    assert ("Argentina" in home or "Argentina" in away)
    assert ("Croatia" in home or "Croatia" in away)

    # Verify substitution evidence exists
    assert "Substitution" in full_text

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-l2-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 13. New L3 Case: Griezmann in Final
# ---------------------------------------------------------------------------

def test_new_l3_case_identifies_griezmann_in_final(chunks_by_doc, cases_by_id):
    """Verify gt-l3-02 is Griezmann's passing stats in the Final."""
    case = cases_by_id["gt-l3-02"]
    assert case["case_group"] == "l3"
    assert case["primary_level"] == "3"

    doc_id = case["relevant_document_ids"][0]
    assert "3869685" in doc_id, f"Expected Final match ID 3869685 in {doc_id}"
    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})
    full_text = " ".join(c["text"] for c in chunks)

    assert chunks[0].get("level") == "3"
    assert meta.get("stage") == "Final"
    assert meta.get("opponent") == "Argentina"
    assert "Griezmann" in full_text
    assert "pass" in full_text.lower()

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-l3-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 14. New L4 Case: Mbappé Tournament Summary
# ---------------------------------------------------------------------------

def test_new_l4_case_identifies_mbappe_tournament_summary(chunks_by_doc, cases_by_id):
    """Verify gt-l4-02 is Mbappé's tournament summary."""
    case = cases_by_id["gt-l4-02"]
    assert case["case_group"] == "l4"
    assert case["primary_level"] == "4"

    doc_id = case["relevant_document_ids"][0]
    assert doc_id.startswith("L4-"), f"Expected L4 prefix, got {doc_id}"
    chunks = chunks_by_doc[doc_id]
    full_text = " ".join(c["text"] for c in chunks)

    assert chunks[0].get("level") == "4"
    assert "Mbapp" in full_text or "Kylian" in full_text
    assert "7 matches" in full_text
    assert "8 goals" in full_text

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-l4-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 15. New Team Case: Morocco
# ---------------------------------------------------------------------------

def test_new_team_case_identifies_morocco(chunks_by_doc, cases_by_id):
    """Verify gt-team-02 is Morocco's Team tournament document."""
    case = cases_by_id["gt-team-02"]
    assert case["case_group"] == "team"
    assert case["primary_level"] == "team"

    doc_id = case["relevant_document_ids"][0]
    assert doc_id.startswith("TEAM-"), f"Expected TEAM prefix, got {doc_id}"
    chunks = chunks_by_doc[doc_id]
    full_text = " ".join(c["text"] for c in chunks)

    assert chunks[0].get("level") == "team"
    assert "Morocco" in full_text
    assert "433" in full_text or "343" in full_text  # formation evidence

    # Verify possession limitation
    for fact in case["required_facts"]:
        if "possession" in fact["claim"].lower():
            assert "event-share proxy" in fact["claim"].lower()

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-team-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 16. New Multi Case: Argentina vs Croatia Semi-final
# ---------------------------------------------------------------------------

def test_new_multi_case_uses_one_semifinal_match(chunks_by_doc, cases_by_id):
    """Verify gt-multi-02 uses L1+L2+L3 for Argentina vs Croatia Semi-finals."""
    case = cases_by_id["gt-multi-02"]
    assert case["case_group"] == "multi"
    assert case["primary_level"] == "2"
    assert set(case["acceptable_levels"]) == {"1", "2", "3"}

    rel_docs = case["relevant_document_ids"]
    assert len(rel_docs) == 3

    # All docs must share match 3869519
    for doc_id in rel_docs:
        assert "3869519" in doc_id, f"Expected match 3869519 in {doc_id}"

    # Verify L1
    l1_doc = [d for d in rel_docs if d.startswith("L1-")][0]
    l1_chunks = chunks_by_doc[l1_doc]
    l1_meta = l1_chunks[0].get("metadata", {})
    assert l1_meta.get("stage") == "Semi-finals"
    home = l1_meta.get("home_team", "")
    away = l1_meta.get("away_team", "")
    assert ("Argentina" in home or "Argentina" in away)
    assert ("Croatia" in home or "Croatia" in away)

    # Verify L2
    l2_doc = [d for d in rel_docs if d.startswith("L2-")][0]
    assert l2_doc in chunks_by_doc

    # Verify L3 Messi
    l3_doc = [d for d in rel_docs if d.startswith("L3-")][0]
    l3_chunks = chunks_by_doc[l3_doc]
    l3_meta = l3_chunks[0].get("metadata", {})
    l3_text = l3_chunks[0].get("text", "")
    assert l3_meta.get("opponent") == "Croatia"
    assert l3_meta.get("stage") == "Semi-finals"
    assert "Messi" in l3_text

    # No Final documents
    for doc_id in rel_docs:
        assert "3869685" not in doc_id, f"Final doc {doc_id} should not be in multi-02"

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            found = False
            for src_id in fact.get("source_document_ids", []):
                for chunk in chunks_by_doc.get(src_id, []):
                    if snippet in chunk.get("text", ""):
                        found = True
                        break
                if found:
                    break
            assert found, (
                f"[gt-multi-02][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 17. New L1 Advanced Case: All Knockout Shootout Matches
# ---------------------------------------------------------------------------

def test_new_l1_case_covers_all_shootout_knockout_matches(chunks_by_doc, cases_by_id):
    """Verify gt-l1-03 covers every L1 knockout match decided by shootout."""
    import re

    case = cases_by_id["gt-l1-03"]
    assert case["case_group"] == "l1"
    assert case["primary_level"] == "1"

    # Independently discover all L1 knockout shootout documents
    discovered_shootout_docs = set()
    for doc_id, doc_chunks in chunks_by_doc.items():
        if not doc_id.startswith("L1-"):
            continue
        meta = doc_chunks[0].get("metadata", {})
        stage = meta.get("stage", "")
        if stage not in ("Round of 16", "Quarter-finals", "Semi-finals", "Final"):
            continue
        full_text = " ".join(c.get("text", "") for c in doc_chunks)
        # Look for actual shootout result, not just the disclaimer
        if re.search(r"won the penalty shootout \d+-\d+", full_text):
            discovered_shootout_docs.add(doc_id)

    case_docs = set(case["relevant_document_ids"])

    # Must match exactly
    assert case_docs == discovered_shootout_docs, (
        f"Shootout document set mismatch.\n"
        f"Case has: {sorted(case_docs)}\n"
        f"Discovered: {sorted(discovered_shootout_docs)}\n"
        f"Missing from case: {sorted(discovered_shootout_docs - case_docs)}\n"
        f"Extra in case: {sorted(case_docs - discovered_shootout_docs)}"
    )

    # At least 2 documents
    assert len(case_docs) >= 2, (
        f"Expected at least 2 shootout documents, found {len(case_docs)}"
    )

    # Every document must be level 1 and knockout stage
    for doc_id in case_docs:
        doc_chunks = chunks_by_doc[doc_id]
        assert doc_chunks[0].get("level") == "1", f"{doc_id} is not level 1"
        meta = doc_chunks[0].get("metadata", {})
        stage = meta.get("stage", "")
        assert stage in ("Round of 16", "Quarter-finals", "Semi-finals", "Final"), (
            f"{doc_id} stage '{stage}' is not knockout"
        )
        full_text = " ".join(c.get("text", "") for c in doc_chunks)
        assert "won the penalty shootout" in full_text, (
            f"{doc_id} does not contain explicit shootout result"
        )


# ---------------------------------------------------------------------------
# 18. New L2 Advanced Case: Morocco vs Portugal Second Half
# ---------------------------------------------------------------------------

def test_new_l2_case_is_morocco_portugal_second_half(chunks_by_doc, cases_by_id):
    """Verify gt-l2-03 is Morocco vs Portugal QF with second-half evidence."""
    case = cases_by_id["gt-l2-03"]
    assert case["case_group"] == "l2"
    assert case["primary_level"] == "2"

    doc_id = case["relevant_document_ids"][0]
    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})
    full_text = " ".join(c["text"] for c in chunks)

    # Verify level and match identity
    assert chunks[0].get("level") == "2"
    assert meta.get("stage") == "Quarter-finals"
    home = meta.get("home_team", "")
    away = meta.get("away_team", "")
    assert "Morocco" in (home, away), f"Expected Morocco, got {home} vs {away}"
    assert "Portugal" in (home, away), f"Expected Portugal, got {home} vs {away}"

    # Verify no required second-half fact is sourced SOLELY from an unambiguous
    # first-half event. A fact that references first-half events as context
    # (e.g. "no goals scored in the second half; the only goal was in the first half")
    # is acceptable, as long as it is not claiming a first-half event IS a second-half event.
    for fact in case["required_facts"]:
        claim = fact.get("claim", "")
        # Only flag facts that assert something happened IN the second half
        # but whose only evidence is a first-half event
        if "second half" in claim.lower() and "first half" not in claim.lower():
            for snippet in fact["evidence_snippets"]:
                if "period 1" in snippet and "41st" in snippet:
                    assert False, (
                        f"[gt-l2-03][{fact['fact_id']}] second-half claim sourced "
                        f"solely from first-half event: {snippet[:80]}"
                    )

    # Verify at least 3 second-half events exist in the document
    # (substitutions after 45th minute and/or chance events after 45th minute)
    import re
    second_half_events = 0
    for line in full_text.split("."):
        m = re.search(r"(\d+)\w*\s+minute", line)
        if m:
            minute = int(m.group(1))
            if minute > 45:
                second_half_events += 1
    assert second_half_events >= 3, (
        f"Expected at least 3 second-half events, found {second_half_events}"
    )

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-l2-03][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 19. New L3 Advanced Case: Mbappé vs Poland
# ---------------------------------------------------------------------------

def test_new_l3_case_identifies_mbappe_against_poland(chunks_by_doc, cases_by_id):
    """Verify gt-l3-03 is Mbappé vs Poland Round of 16 player-match doc."""
    case = cases_by_id["gt-l3-03"]
    assert case["case_group"] == "l3"
    assert case["primary_level"] == "3"

    doc_id = case["relevant_document_ids"][0]
    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})
    full_text = chunks[0].get("text", "")

    # Level
    assert chunks[0].get("level") == "3"

    # Player identity
    assert "Mbapp" in full_text, f"Expected Mbappé in text: {full_text[:200]}"
    assert meta.get("player_id") == 3009, f"Expected player_id 3009, got {meta.get('player_id')}"

    # Team is France
    assert "France" in full_text, f"Expected France in text: {full_text[:200]}"

    # Opponent is Poland
    assert meta.get("opponent") == "Poland", f"Expected opponent Poland, got {meta.get('opponent')}"

    # Stage is Round of 16
    assert meta.get("stage") == "Round of 16", f"Expected Round of 16, got {meta.get('stage')}"

    # Unique player-match document
    matching_docs = []
    for d_id, d_chunks in chunks_by_doc.items():
        if not d_id.startswith("L3-"):
            continue
        d_meta = d_chunks[0].get("metadata", {})
        if (d_meta.get("player_id") == 3009
                and d_meta.get("opponent") == "Poland"
                and d_meta.get("stage") == "Round of 16"):
            matching_docs.append(d_id)
    assert len(matching_docs) == 1, (
        f"Expected 1 unique Mbappé vs Poland R16 doc, found {len(matching_docs)}: {matching_docs}"
    )
    assert matching_docs[0] == doc_id

    # Verify evidence snippets
    full_text_all = " ".join(c["text"] for c in chunks)
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text_all, (
                f"[gt-l3-03][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 20. New L4 Advanced Case: Griezmann Tournament Summary
# ---------------------------------------------------------------------------

def test_new_l4_case_identifies_griezmann_tournament_summary(chunks_by_doc, cases_by_id):
    """Verify gt-l4-03 is Griezmann's L4 tournament summary."""
    case = cases_by_id["gt-l4-03"]
    assert case["case_group"] == "l4"
    assert case["primary_level"] == "4"

    doc_id = case["relevant_document_ids"][0]
    assert doc_id.startswith("L4-"), f"Expected L4 prefix, got {doc_id}"

    chunks = chunks_by_doc[doc_id]
    meta = chunks[0].get("metadata", {})
    full_text = chunks[0].get("text", "")

    # Level
    assert chunks[0].get("level") == "4"

    # Player identity
    assert "Griezmann" in full_text, f"Expected Griezmann in text: {full_text[:200]}"
    assert meta.get("player_id") == 5487, f"Expected player_id 5487, got {meta.get('player_id')}"

    # Team is France
    assert "France" in full_text, f"Expected France in text: {full_text[:200]}"

    # Tournament summary grain (not L3 single-match)
    assert not doc_id.startswith("L3-"), "Should not be an L3 document"
    assert "7 matches" in full_text, f"Expected '7 matches' in tournament summary"

    # Verify evidence snippets
    full_text_all = " ".join(c["text"] for c in chunks)
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text_all, (
                f"[gt-l4-03][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 21. New Team Advanced Case: France Style Document
# ---------------------------------------------------------------------------

def test_new_team_case_identifies_france_style_document(chunks_by_doc, cases_by_id):
    """Verify gt-team-03 is France's Team tournament document."""
    case = cases_by_id["gt-team-03"]
    assert case["case_group"] == "team"
    assert case["primary_level"] == "team"

    doc_id = case["relevant_document_ids"][0]
    assert doc_id.startswith("TEAM-"), f"Expected TEAM prefix, got {doc_id}"

    chunks = chunks_by_doc[doc_id]
    full_text = " ".join(c["text"] for c in chunks)

    # Level
    assert chunks[0].get("level") == "team"

    # France identity
    assert "France" in full_text, f"Expected France in text: {full_text[:200]}"

    # Formation evidence
    assert "4231" in full_text or "433" in full_text or "442" in full_text, (
        f"Expected formation evidence in text: {full_text[:200]}"
    )

    # Passing/play-pattern evidence
    assert "pass" in full_text.lower() or "play pattern" in full_text.lower(), (
        f"Expected passing/play-pattern evidence in text: {full_text[:200]}"
    )

    # Proxy limitation preserved
    for fact in case["required_facts"]:
        if "possession" in fact["claim"].lower():
            assert "event-share proxy" in fact["claim"].lower(), (
                f"[gt-team-03][{fact['fact_id']}] possession claim does not "
                f"mention proxy limitation: {fact['claim']}"
            )

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            assert snippet in full_text, (
                f"[gt-team-03][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 22. New Multi Advanced Case: Morocco Route to Semi-finals
# ---------------------------------------------------------------------------

def test_new_multi_case_covers_morocco_route_to_semifinals(chunks_by_doc, cases_by_id):
    """Verify gt-multi-03 covers Morocco's route through R16 and QF with team style."""
    case = cases_by_id["gt-multi-03"]
    assert case["case_group"] == "multi"
    assert case["primary_level"] == "1"
    assert set(case["acceptable_levels"]) == {"1", "2", "team"}

    rel_docs = case["relevant_document_ids"]

    # Must include L1, L2, and Team levels
    doc_levels = set()
    for doc_id in rel_docs:
        for chunk in chunks_by_doc.get(doc_id, []):
            doc_levels.add(chunk.get("level"))
    assert "1" in doc_levels, "Missing level 1 documents"
    assert "2" in doc_levels, "Missing level 2 documents"
    assert "team" in doc_levels, "Missing team level document"

    # Identify match documents
    l1_docs = [d for d in rel_docs if d.startswith("L1-")]
    l2_docs = [d for d in rel_docs if d.startswith("L2-")]
    team_docs = [d for d in rel_docs if d.startswith("TEAM-")]

    assert len(l1_docs) >= 2, f"Expected at least 2 L1 docs, found {len(l1_docs)}"
    assert len(l2_docs) >= 2, f"Expected at least 2 L2 docs, found {len(l2_docs)}"
    assert len(team_docs) >= 1, f"Expected at least 1 Team doc, found {len(team_docs)}"

    # R16 and QF match IDs must be different (extract from document_id)
    import re
    def extract_match_id_from_doc_id(doc_id: str) -> str:
        """Extract numeric match ID from document ID like L1-match-3869220."""
        m = re.search(r"match-(\d+)", doc_id)
        return m.group(1) if m else ""

    r16_match_ids = set()
    qf_match_ids = set()
    for doc_id in l1_docs + l2_docs:
        meta = chunks_by_doc[doc_id][0].get("metadata", {})
        stage = meta.get("stage", "")
        match_id = extract_match_id_from_doc_id(doc_id)
        if stage == "Round of 16":
            r16_match_ids.add(match_id)
        elif stage == "Quarter-finals":
            qf_match_ids.add(match_id)

    assert len(r16_match_ids) >= 1, "No Round of 16 match found"
    assert len(qf_match_ids) >= 1, "No Quarter-final match found"
    assert r16_match_ids != qf_match_ids, "R16 and QF match IDs must differ"

    # L1 and L2 docs for the same match must share a match ID
    for stage_name in ("Round of 16", "Quarter-finals"):
        stage_l1 = [d for d in l1_docs if chunks_by_doc[d][0].get("metadata", {}).get("stage") == stage_name]
        stage_l2 = [d for d in l2_docs if chunks_by_doc[d][0].get("metadata", {}).get("stage") == stage_name]
        if stage_l1 and stage_l2:
            l1_match_id = extract_match_id_from_doc_id(stage_l1[0])
            l2_match_id = extract_match_id_from_doc_id(stage_l2[0])
            assert l1_match_id == l2_match_id, (
                f"{stage_name} L1 match_id {l1_match_id} != L2 match_id {l2_match_id}"
            )

    # Morocco must be present in both matches
    for doc_id in l1_docs:
        meta = chunks_by_doc[doc_id][0].get("metadata", {})
        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        assert "Morocco" in (home, away), (
            f"{doc_id} does not feature Morocco: {home} vs {away}"
        )

    # Team Morocco document must be present
    team_doc_present = any(
        "Morocco" in " ".join(c.get("text", "") for c in chunks_by_doc.get(d, []))
        for d in team_docs
    )
    assert team_doc_present, "No Morocco Team document found"

    # No France Semi-final document used as required progression evidence
    for doc_id in rel_docs:
        meta = chunks_by_doc[doc_id][0].get("metadata", {})
        stage = meta.get("stage", "")
        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        assert not (stage == "Semi-finals" and "France" in (home, away)), (
            f"France Semi-final doc {doc_id} should not be required for Morocco progression"
        )

    # Shootout and normal-score facts are not conflated
    for fact in case["required_facts"]:
        claim = fact.get("claim", "")
        if "shootout" in claim.lower():
            # Shootout facts should not mention normal match goals as shootout goals
            for snippet in fact["evidence_snippets"]:
                assert "beat" not in snippet.lower() or "shootout" in snippet.lower(), (
                    f"[gt-multi-03][{fact['fact_id']}] shootout claim has beat-result snippet"
                )

    # Verify evidence snippets
    for fact in case["required_facts"]:
        for snippet in fact["evidence_snippets"]:
            found = False
            for src_id in fact.get("source_document_ids", []):
                for chunk in chunks_by_doc.get(src_id, []):
                    if snippet in chunk.get("text", ""):
                        found = True
                        break
                if found:
                    break
            assert found, (
                f"[gt-multi-03][{fact['fact_id']}] snippet not found: {snippet[:80]}"
            )


# ---------------------------------------------------------------------------
# 23. Full Validation Pass
# ---------------------------------------------------------------------------

def test_expanded_twenty_four_case_ground_truth_full_validation_passes():
    """Run the full validator and assert zero errors."""
    errors = validate_semantic_ground_truth(
        SEMANTIC_GROUND_TRUTH_METADATA,
        SEMANTIC_GROUND_TRUTH,
        SEMANTIC_GROUND_TRUTH_METADATA["chunks_path"],
    )
    assert errors == [], (
        f"Validation produced {len(errors)} error(s):\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# 24. Extended Case Identities
# ---------------------------------------------------------------------------


def test_extended_l1_case_is_england_france_quarter_final(cases_by_id):
    """Verify gt-l1-04 is England vs France Quarter-final decided in normal time."""
    case = cases_by_id["gt-l1-04"]
    assert case["case_group"] == "l1"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "1"
    assert case["acceptable_levels"] == ["1"]
    assert case["relevant_document_ids"] == ["L1-match-3869354"]

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert "f1" in fact_ids
    assert "f5" in fact_ids  # yellow cards

    # f1 must mention Quarter-finals
    f1 = next(f for f in case["required_facts"] if f["fact_id"] == "f1")
    assert "Quarter-finals" in f1["claim"]
    # Evidence must contain explicit Quarter-finals sentence
    assert any("Quarter-finals" in s for s in f1["evidence_snippets"]), (
        "f1 evidence must contain 'Quarter-finals'"
    )


def test_extended_l2_case_is_england_france_key_events(cases_by_id):
    """Verify gt-l2-04 is England vs France QF key turning points."""
    case = cases_by_id["gt-l2-04"]
    assert case["case_group"] == "l2"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "2"
    assert case["acceptable_levels"] == ["2"]
    assert case["relevant_document_ids"] == ["L2-match-3869354"]

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert fact_ids == {"f1", "f2", "f3", "f4", "f5", "f6"}

    # f4 must preserve corpus truncation 'off t'
    f4 = next(f for f in case["required_facts"] if f["fact_id"] == "f4")
    assert any("off t" in s for s in f4["evidence_snippets"]), (
        "f4 evidence must preserve corpus truncation 'off t'"
    )
    # Must NOT silently correct to 'off target'
    assert not any("off target" in s for s in f4["evidence_snippets"]), (
        "f4 evidence must not silently correct 'off t' to 'off target'"
    )

    # f6 must be injury substitution
    f6 = next(f for f in case["required_facts"] if f["fact_id"] == "f6")
    assert "injury" in f6["claim"].lower()


def test_extended_l3_case_is_enzo_fernandez_defensive(cases_by_id):
    """Verify gt-l3-04 is Enzo Fernández defensive performance in the Final."""
    case = cases_by_id["gt-l3-04"]
    assert case["case_group"] == "l3"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "3"
    assert case["acceptable_levels"] == ["3"]
    assert case["relevant_document_ids"] == ["L3-match-3869685-player-38718"]

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert fact_ids == {"f1", "f2", "f3", "f4"}

    # f4 must mention defensive stats
    f4 = next(f for f in case["required_facts"] if f["fact_id"] == "f4")
    assert "tackle" in f4["claim"].lower()
    assert "interception" in f4["claim"].lower()
    assert "clearance" in f4["claim"].lower()


def test_extended_l4_case_is_enzo_fernandez_tournament(cases_by_id):
    """Verify gt-l4-04 is Enzo Fernández tournament summary."""
    case = cases_by_id["gt-l4-04"]
    assert case["case_group"] == "l4"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "4"
    assert case["acceptable_levels"] == ["4"]
    assert case["relevant_document_ids"] == ["L4-player-38718"]

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert fact_ids == {"f1", "f2", "f3", "f4", "f5"}

    # f1 must mention 7 matches
    f1 = next(f for f in case["required_facts"] if f["fact_id"] == "f1")
    assert "7 matches" in f1["claim"]

    # f4 must mention defensive workload
    f4 = next(f for f in case["required_facts"] if f["fact_id"] == "f4")
    assert "defensive workload" in f4["claim"].lower()


def test_extended_team_case_is_group_stage_elimination(cases_by_id):
    """Verify gt-team-04 is Germany group-stage eliminated team document."""
    case = cases_by_id["gt-team-04"]
    assert case["case_group"] == "team"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "team"
    assert case["acceptable_levels"] == ["team"]
    assert case["relevant_document_ids"] == ["TEAM-770"]

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert fact_ids == {"f1", "f2", "f3", "f4", "f5"}

    # f1 must mention 3 matches (group-stage elimination)
    f1 = next(f for f in case["required_facts"] if f["fact_id"] == "f1")
    assert "3 matches" in f1["claim"]

    # f2 must mention possession proxy
    f2 = next(f for f in case["required_facts"] if f["fact_id"] == "f2")
    assert "event-share proxy" in f2["claim"].lower()


def test_extended_multi_case_is_cross_team_comparison(cases_by_id, chunks_by_doc):
    """Verify gt-multi-04 requires both TEAM-779 and TEAM-771 for comparison."""
    case = cases_by_id["gt-multi-04"]
    assert case["case_group"] == "multi"
    assert case["expected_route"] == "semantic"
    assert case["primary_level"] == "team"
    assert case["acceptable_levels"] == ["team"]

    # Must require BOTH Argentina and France TEAM documents
    rel_docs = set(case["relevant_document_ids"])
    assert "TEAM-779" in rel_docs, "Must include Argentina TEAM document"
    assert "TEAM-771" in rel_docs, "Must include France TEAM document"
    assert len(rel_docs) == 2, f"Expected exactly 2 relevant docs, got {len(rel_docs)}"

    # Both documents must exist in corpus
    assert "TEAM-779" in chunks_by_doc
    assert "TEAM-771" in chunks_by_doc

    # Key fact identities
    fact_ids = {f["fact_id"] for f in case["required_facts"]}
    assert fact_ids == {"f1", "f2", "f3", "f4"}

    # All facts must source from both documents
    for fact in case["required_facts"]:
        src_ids = set(fact["source_document_ids"])
        assert "TEAM-779" in src_ids, (
            f"[{fact['fact_id']}] must source from Argentina TEAM-779"
        )
        assert "TEAM-771" in src_ids, (
            f"[{fact['fact_id']}] must source from France TEAM-771"
        )

    # f1 must compare possession
    f1 = next(f for f in case["required_facts"] if f["fact_id"] == "f1")
    assert "possession" in f1["claim"].lower()
    assert "event-share proxy" in f1["claim"].lower()


# ---------------------------------------------------------------------------
# 25. Correction-Specific Regression Checks
# ---------------------------------------------------------------------------


def test_gt_l1_04_f1_evidence_includes_quarter_finals(cases_by_id):
    """Regression: gt-l1-04/f1 evidence must contain 'Quarter-finals' sentence."""
    case = cases_by_id["gt-l1-04"]
    f1 = next(f for f in case["required_facts"] if f["fact_id"] == "f1")
    quarter_final_snippets = [
        s for s in f1["evidence_snippets"] if "Quarter-finals" in s
    ]
    assert len(quarter_final_snippets) >= 1, (
        "gt-l1-04/f1 must have at least one evidence snippet containing 'Quarter-finals'"
    )
    # The snippet must be the full sentence with date and venue
    assert any("Al Bayt Stadium" in s for s in quarter_final_snippets), (
        "Quarter-finals snippet must include venue 'Al Bayt Stadium'"
    )
    assert any("2022-12-10" in s for s in quarter_final_snippets), (
        "Quarter-finals snippet must include date '2022-12-10'"
    )


def test_gt_l2_04_f4_evidence_preserves_corpus_truncation(cases_by_id):
    """Regression: gt-l2-04/f4 evidence must preserve 'off t' truncation."""
    case = cases_by_id["gt-l2-04"]
    f4 = next(f for f in case["required_facts"] if f["fact_id"] == "f4")
    # Must contain the exact truncated text
    assert any("shot ended off t" in s for s in f4["evidence_snippets"]), (
        "gt-l2-04/f4 evidence must contain 'shot ended off t' (corpus truncation)"
    )
    # Must NOT silently correct to 'off target'
    assert not any("off target" in s for s in f4["evidence_snippets"]), (
        "gt-l2-04/f4 must not silently correct 'off t' to 'off target'"
    )
    # Must contain the xG threshold context
    assert any("0.3 xG" in s for s in f4["evidence_snippets"]), (
        "gt-l2-04/f4 evidence must contain '0.3 xG' threshold"
    )
