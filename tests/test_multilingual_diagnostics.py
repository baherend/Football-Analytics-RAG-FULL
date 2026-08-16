"""
test_multilingual_diagnostics.py -- structural tests for the diagnostic
instrumentation used in the multilingual root-cause investigation. Not a
retrieval-quality test suite -- proves the diagnostic helpers correctly
compose existing production functions, using small synthetic fixtures.
"""

from __future__ import annotations

from tests.multilingual_diagnostics import (
    ENTITY_TRANSLITERATIONS,
    build_case_lookup,
    make_translated_case,
)
from tests.semantic_ground_truth import SEMANTIC_GROUND_TRUTH


def test_build_case_lookup_covers_all_semantic_cases():
    lookup = build_case_lookup()
    assert len(lookup) == len(SEMANTIC_GROUND_TRUTH)
    for case in SEMANTIC_GROUND_TRUTH:
        assert lookup[case["id"]] is case or lookup[case["id"]] == case


def test_make_translated_case_swaps_only_query():
    source = build_case_lookup()["gt-pilot-l3-01"]
    new_case = make_translated_case("gt-pilot-l3-01", "custom diagnostic query text")

    assert new_case["query"] == "custom diagnostic query text"
    for field in ("relevant_document_ids", "case_group", "primary_level", "acceptable_levels"):
        assert new_case[field] == source[field]


def test_entity_transliterations_are_non_empty_distinct_pairs():
    for latin, arabic in ENTITY_TRANSLITERATIONS.items():
        assert latin.strip()
        assert arabic.strip()
        assert latin != arabic
