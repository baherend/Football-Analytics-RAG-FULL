"""
test_multilingual_diagnostics.py -- structural tests for the diagnostic
instrumentation used in the multilingual root-cause investigation. Not a
retrieval-quality test suite -- proves the diagnostic helpers correctly
compose existing production functions, using small synthetic fixtures.
"""

from __future__ import annotations

from src.evaluation.diagnostics import (
    ENTITY_TRANSLITERATIONS,
    build_case_lookup,
    evaluate_model_hybrid_method,
    make_translated_case,
)
from src.evaluation.ground_truth.semantic import SEMANTIC_GROUND_TRUTH


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


def test_model_hybrid_uses_candidate_dense_and_restores_production_dense(monkeypatch):
    from types import SimpleNamespace
    from src.retrieval import search

    production_dense = search.dense_search

    def candidate_dense(*args, **kwargs):
        return []

    def fake_hybrid(query, **kwargs):
        assert search.dense_search is candidate_dense
        return [{
            "chunk_id": "doc-1-chunk-0",
            "text": "evidence",
            "metadata": {"document_id": "doc-1", "level": "L1"},
            "score": 1.0,
        }]

    monkeypatch.setattr(search, "hybrid_search", fake_hybrid)
    model_index = SimpleNamespace(dense_search=candidate_dense)
    case = {
        "id": "case-1",
        "query": "query",
        "relevant_document_ids": ["doc-1"],
        "case_group": "l1",
    }

    result = evaluate_model_hybrid_method(
        model_index, [case], {"doc-1": "L1"}, k_values=(1,), candidate_chunk_depth=2,
    )

    assert result[0]["runs_by_k"]["1"]["metrics"]["hit_at_k"] == 1.0
    assert search.dense_search is production_dense
