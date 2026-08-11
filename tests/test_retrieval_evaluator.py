"""
Unit tests for the document-level retrieval evaluator.

Uses synthetic cases and synthetic retrieval results for metric tests.
Does NOT call real Dense retriever, load MiniLM, open real Chroma,
or run the full 18-case baseline.
"""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from tests.retrieval_evaluator import (
    DEFAULT_CANDIDATE_CHUNK_DEPTH,
    DEFAULT_K_VALUES,
    DOCUMENT_RANKING_POLICY,
    RELEVANCE_UNIT,
    RETRIEVAL_EVALUATOR_SCHEMA_VERSION,
    SUPPORTED_METHODS,
    RetrievalEvaluationError,
    acceptable_precision_at_k,
    aggregate_case_results,
    aggregate_by_group,
    all_required_at_k,
    build_document_ranking,
    evaluate_case,
    evaluate_retrieval_method,
    first_relevant_document_rank,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevant_level_coverage_at_k,
    validate_chunk_result,
    validate_ground_truth_and_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def _make_chunk_result(
    chunk_id: str = "c1",
    document_id: str = "doc1",
    level: str = "1",
    text: str = "some text",
    score: float = 0.9,
    rank: Optional[int] = None,
    source: str = "bm25",
    rrf_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a synthetic retrieval result."""
    result = {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "document_id": document_id,
            "level": level,
        },
        "score": score,
        "source": source,
    }
    if rank is not None:
        result["rank"] = rank
    if rrf_score is not None:
        result["rrf_score"] = rrf_score
    return result


def _make_case(
    case_id: str = "test-01",
    case_group: str = "l1",
    query: str = "test query",
    required_docs: List[str] = None,
    optional_docs: List[str] = None,
    primary_level: str = "1",
    acceptable_levels: List[str] = None,
) -> Dict[str, Any]:
    """Create a synthetic Ground Truth case."""
    return {
        "id": case_id,
        "dataset_id": "test-dataset",
        "case_group": case_group,
        "query": query,
        "expected_route": "semantic",
        "answerability": "answerable",
        "primary_level": primary_level,
        "acceptable_levels": acceptable_levels or [primary_level],
        "relevant_document_ids": required_docs or [],
        "optional_relevant_document_ids": optional_docs or [],
        "required_facts": [],
        "forbidden_claims": [],
        "notes": "",
    }


# ---------------------------------------------------------------------------
# 1. test_document_ranking_deduplicates_by_document_id
# ---------------------------------------------------------------------------


def test_document_ranking_deduplicates_by_document_id():
    """Repeated chunks from one document produce one ranked document."""
    results = [
        _make_chunk_result(chunk_id="c1", document_id="doc1", rank=1),
        _make_chunk_result(chunk_id="c2", document_id="doc2", rank=2),
        _make_chunk_result(chunk_id="c3", document_id="doc1", rank=3),
        _make_chunk_result(chunk_id="c4", document_id="doc2", rank=4),
        _make_chunk_result(chunk_id="c5", document_id="doc3", rank=5),
    ]

    ranked, diag = build_document_ranking(results, "bm25", "test-01")

    assert len(ranked) == 3
    assert diag["unique_document_count"] == 3
    assert diag["duplicate_document_hit_count"] == 2
    assert ranked[0]["document_id"] == "doc1"
    assert ranked[1]["document_id"] == "doc2"
    assert ranked[2]["document_id"] == "doc3"
    assert ranked[0]["duplicate_chunk_count"] == 1
    assert ranked[1]["duplicate_chunk_count"] == 1
    assert ranked[2]["duplicate_chunk_count"] == 0


# ---------------------------------------------------------------------------
# 2. test_document_ranking_preserves_first_occurrence_order
# ---------------------------------------------------------------------------


def test_document_ranking_preserves_first_occurrence_order():
    """Scores out of order must not affect document ranking."""
    results = [
        _make_chunk_result(chunk_id="c1", document_id="docA", score=0.5, rank=1),
        _make_chunk_result(chunk_id="c2", document_id="docB", score=0.99, rank=2),
        _make_chunk_result(chunk_id="c3", document_id="docC", score=0.1, rank=3),
    ]

    ranked, _ = build_document_ranking(results, "bm25", "test-01")

    assert ranked[0]["document_id"] == "docA"
    assert ranked[1]["document_id"] == "docB"
    assert ranked[2]["document_id"] == "docC"
    # Even though docB has score 0.99 > docA 0.5, order preserved
    assert ranked[0]["score"] == 0.5
    assert ranked[1]["score"] == 0.99


# ---------------------------------------------------------------------------
# 3. test_document_ranking_rejects_missing_metadata
# ---------------------------------------------------------------------------


def test_document_ranking_rejects_missing_metadata():
    """Missing metadata raises RetrievalEvaluationError."""
    results = [{"chunk_id": "c1", "text": "x", "score": 0.5}]

    with pytest.raises(RetrievalEvaluationError, match="metadata is not a dict"):
        build_document_ranking(results, "bm25", "test-01")


# ---------------------------------------------------------------------------
# 4. test_document_ranking_rejects_missing_document_id
# ---------------------------------------------------------------------------


def test_document_ranking_rejects_missing_document_id():
    """Missing document_id raises RetrievalEvaluationError."""
    results = [
        {"chunk_id": "c1", "text": "x", "metadata": {"level": "1"}, "score": 0.5}
    ]

    with pytest.raises(RetrievalEvaluationError, match="missing or blank document_id"):
        build_document_ranking(results, "bm25", "test-01")


# ---------------------------------------------------------------------------
# 5. test_document_ranking_rejects_chunk_id_as_document_id
# ---------------------------------------------------------------------------


def test_document_ranking_rejects_chunk_id_as_document_id():
    """Document ID containing '-chunk-' is rejected."""
    results = [
        _make_chunk_result(
            chunk_id="c1", document_id="L3-match-123-chunk-5", rank=1
        ),
    ]

    with pytest.raises(RetrievalEvaluationError, match="contains '-chunk-'"):
        build_document_ranking(results, "bm25", "test-01")


# ---------------------------------------------------------------------------
# 6. test_single_relevant_document_metrics
# ---------------------------------------------------------------------------


def test_single_relevant_document_metrics():
    """Exact metric values for a single required document."""
    ranked_ids = ["doc1", "doc2", "doc3"]
    required = ["doc2"]

    assert hit_at_k(ranked_ids, required, 1) == 0.0
    assert hit_at_k(ranked_ids, required, 2) == 1.0
    assert recall_at_k(ranked_ids, required, 1) == 0.0
    assert recall_at_k(ranked_ids, required, 2) == 1.0
    assert precision_at_k(ranked_ids, required, 1) == 0.0
    assert precision_at_k(ranked_ids, required, 2) == 0.5
    assert all_required_at_k(ranked_ids, required, 1) == 0.0
    assert all_required_at_k(ranked_ids, required, 2) == 1.0
    assert reciprocal_rank(ranked_ids, required) == 0.5  # 1/2
    assert first_relevant_document_rank(ranked_ids, required) == 2

    # nDCG@2: doc1=0, doc2=1 => DCG = 0/log2(2) + 1/log2(3) = 0 + 0.6309
    # IDCG: 1/log2(2) = 1.0
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(ranked_ids, required, 2) == pytest.approx(expected_ndcg)


# ---------------------------------------------------------------------------
# 7. test_multi_document_recall_and_exact_success
# ---------------------------------------------------------------------------


def test_multi_document_recall_and_exact_success():
    """Partial recall, all_required transitions from 0 to 1."""
    ranked_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    required = ["doc1", "doc3", "doc5"]

    assert recall_at_k(ranked_ids, required, 1) == pytest.approx(1 / 3)
    assert recall_at_k(ranked_ids, required, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranked_ids, required, 5) == pytest.approx(3 / 3)

    assert all_required_at_k(ranked_ids, required, 1) == 0.0
    assert all_required_at_k(ranked_ids, required, 3) == 0.0
    assert all_required_at_k(ranked_ids, required, 5) == 1.0


# ---------------------------------------------------------------------------
# 8. test_optional_documents_affect_only_acceptable_precision
# ---------------------------------------------------------------------------


def test_optional_documents_affect_only_acceptable_precision():
    """Optional docs increase acceptable precision but not strict metrics."""
    ranked_ids = ["opt1", "doc2", "doc3"]
    required = ["doc2"]
    optional = ["opt1"]

    # Strict precision: only doc2 is required
    assert precision_at_k(ranked_ids, required, 1) == 0.0
    assert precision_at_k(ranked_ids, required, 2) == 0.5

    # Acceptable precision: opt1 counts
    assert acceptable_precision_at_k(ranked_ids, required, optional, 1) == 1.0
    assert acceptable_precision_at_k(ranked_ids, required, optional, 2) == 1.0

    # hit: 0 at k=1 (opt1 is not required)
    assert hit_at_k(ranked_ids, required, 1) == 0.0
    # hit: 1 at k=2 (doc2 is required)
    assert hit_at_k(ranked_ids, required, 2) == 1.0

    # MRR: first required is doc2 at rank 2
    assert reciprocal_rank(ranked_ids, required) == 0.5

    # nDCG@1: opt1 is not required => 0
    assert ndcg_at_k(ranked_ids, required, 1) == 0.0

    # all_required@1: doc2 not in top 1
    assert all_required_at_k(ranked_ids, required, 1) == 0.0


# ---------------------------------------------------------------------------
# 9. test_precision_denominator_remains_k
# ---------------------------------------------------------------------------


def test_precision_denominator_remains_k():
    """Fewer than K unique docs: denominator remains K."""
    ranked_ids = ["doc1"]  # only 1 doc
    required = ["doc1"]

    assert precision_at_k(ranked_ids, required, 5) == 0.2  # 1/5, not 1/1


# ---------------------------------------------------------------------------
# 10. test_reciprocal_rank_uses_first_required_document
# ---------------------------------------------------------------------------


def test_reciprocal_rank_uses_first_required_document():
    """RR = 1/rank of first required document."""
    ranked_ids = ["a", "b", "c", "d"]
    required = ["c", "d"]

    assert reciprocal_rank(ranked_ids, required) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 11. test_ndcg_binary_relevance_formula
# ---------------------------------------------------------------------------


def test_ndcg_binary_relevance_formula():
    """Hand-calculated nDCG check."""
    ranked_ids = ["d1", "d2", "d3", "d4"]
    required = ["d1", "d3"]

    # DCG@4 = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5)
    dcg = 1.0 / math.log2(2) + 0 + 1.0 / math.log2(4) + 0
    # IDCG@4 = 1/log2(2) + 1/log2(3)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg

    assert ndcg_at_k(ranked_ids, required, 4) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 12. test_relevant_level_coverage_uses_required_document_levels
# ---------------------------------------------------------------------------


def test_relevant_level_coverage_uses_required_document_levels():
    """Level coverage derives from required doc levels, not acceptable_levels."""
    ranked_ids = ["doc1", "doc2", "doc3"]
    required = ["doc1", "doc2"]
    doc_levels = {"doc1": "1", "doc2": "2", "doc3": "3"}

    # doc1 (level 1) and doc2 (level 2) in top-3 => 2/2 required levels
    assert relevant_level_coverage_at_k(ranked_ids, required, doc_levels, 3) == 1.0

    # Only doc1 in top-1 => level 1 covered => 1/2
    assert relevant_level_coverage_at_k(ranked_ids, required, doc_levels, 1) == 0.5


# ---------------------------------------------------------------------------
# 13. test_method_runner_uses_exact_query
# ---------------------------------------------------------------------------


def test_method_runner_uses_exact_query():
    """Retrieval function receives the exact query string."""
    calls = []

    def fake_bm25(query, k=20):
        calls.append(("bm25", query, k))
        return []

    case = _make_case(
        case_id="test-01",
        query="What happened in the opening match?",
        required_docs=["doc1"],
    )
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
    )

    assert len(calls) == 1
    assert calls[0][1] == "What happened in the opening match?"


# ---------------------------------------------------------------------------
# 14. test_bm25_method_uses_candidate_depth
# ---------------------------------------------------------------------------


def test_bm25_method_uses_candidate_depth():
    """BM25 receives the correct candidate depth as k."""
    calls = []

    def fake_bm25(query, k=20):
        calls.append(k)
        return []

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
        candidate_chunk_depth=100,
    )
    assert calls[-1] == 100

    evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
        candidate_chunk_depth=50,
    )
    assert calls[-1] == 50


# ---------------------------------------------------------------------------
# 15. test_dense_method_uses_no_level_filter
# ---------------------------------------------------------------------------


def test_dense_method_uses_no_level_filter():
    """Dense receives level_filter=None."""
    calls = []

    def fake_dense(query, k=20, level_filter=None):
        calls.append(level_filter)
        return []

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_dense,
        method="dense",
        document_levels=doc_levels,
    )

    assert calls[-1] is None


# ---------------------------------------------------------------------------
# 16. test_hybrid_method_called_separately_for_every_k
# ---------------------------------------------------------------------------


def test_hybrid_method_called_separately_for_every_k():
    """Hybrid receives one call per evaluation K with k=<K>, bm25_k=depth, dense_k=depth."""
    calls = []

    def fake_hybrid(query, k=5, bm25_k=20, dense_k=20, level_filter=None):
        calls.append(
            {"k": k, "bm25_k": bm25_k, "dense_k": dense_k, "level_filter": level_filter}
        )
        return []

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_hybrid,
        method="hybrid",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    assert len(calls) == 4
    assert calls[0] == {"k": 1, "bm25_k": 100, "dense_k": 100, "level_filter": None}
    assert calls[1] == {"k": 3, "bm25_k": 100, "dense_k": 100, "level_filter": None}
    assert calls[2] == {"k": 5, "bm25_k": 100, "dense_k": 100, "level_filter": None}
    assert calls[3] == {"k": 10, "bm25_k": 100, "dense_k": 100, "level_filter": None}


# ---------------------------------------------------------------------------
# 17. test_unknown_method_is_rejected
# ---------------------------------------------------------------------------


def test_unknown_method_is_rejected():
    """Unknown method raises RetrievalEvaluationError."""
    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    with pytest.raises(RetrievalEvaluationError, match="Unsupported method"):
        evaluate_retrieval_method(
            method="unknown",
            cases=[case],
            retrieval_module=MagicMock(),
            document_levels=doc_levels,
        )


# ---------------------------------------------------------------------------
# 18. test_aggregate_metrics_are_macro_averages
# ---------------------------------------------------------------------------


def test_aggregate_metrics_are_macro_averages():
    """Aggregation uses macro (arithmetic mean) averaging with K-scoped rank metrics."""
    case_results = [
        {
            "runs_by_k": {
                "1": {
                    "reciprocal_rank_at_k": 1.0,
                    "ranked_unique_document_count": 5,
                    "duplicate_document_hit_count": 0,
                    "first_relevant_document_rank_at_k": 1,
                    "metrics": {
                        "hit_at_k": 1.0,
                        "recall_at_k": 1.0,
                        "precision_at_k": 1.0,
                        "acceptable_precision_at_k": 1.0,
                        "all_required_at_k": 1.0,
                        "ndcg_at_k": 1.0,
                        "relevant_level_coverage_at_k": 1.0,
                    },
                },
            },
        },
        {
            "runs_by_k": {
                "1": {
                    "reciprocal_rank_at_k": 0.0,
                    "ranked_unique_document_count": 3,
                    "duplicate_document_hit_count": 1,
                    "first_relevant_document_rank_at_k": None,
                    "metrics": {
                        "hit_at_k": 0.0,
                        "recall_at_k": 0.0,
                        "precision_at_k": 0.0,
                        "acceptable_precision_at_k": 0.0,
                        "all_required_at_k": 0.0,
                        "ndcg_at_k": 0.0,
                        "relevant_level_coverage_at_k": 0.0,
                    },
                },
            },
        },
    ]

    agg = aggregate_case_results(case_results, k_values=(1,))

    assert agg["cases_evaluated"] == 2
    assert agg["metrics_by_k"]["1"]["mean_reciprocal_rank_at_k"] == pytest.approx(
        0.5, abs=1e-6
    )
    assert agg["metrics_by_k"]["1"]["mean_first_relevant_document_rank_at_k"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert agg["metrics_by_k"]["1"]["hit_at_k"] == pytest.approx(0.5, abs=1e-6)
    assert agg["metrics_by_k"]["1"]["recall_at_k"] == pytest.approx(0.5, abs=1e-6)
    assert agg["metrics_by_k"]["1"]["exact_required_success_count"] == 1
    assert agg["metrics_by_k"]["1"]["exact_required_success_rate"] == pytest.approx(
        0.5, abs=1e-6
    )


# ---------------------------------------------------------------------------
# 19. test_aggregate_metrics_group_by_case_group
# ---------------------------------------------------------------------------


def test_aggregate_metrics_group_by_case_group():
    """Aggregation groups cases correctly with K-scoped rank metrics."""
    case_results = [
        {
            "case_group": "l1",
            "runs_by_k": {
                "1": {
                    "reciprocal_rank_at_k": 1.0,
                    "ranked_unique_document_count": 5,
                    "duplicate_document_hit_count": 0,
                    "first_relevant_document_rank_at_k": 1,
                    "metrics": {
                        "hit_at_k": 1.0,
                        "recall_at_k": 1.0,
                        "precision_at_k": 1.0,
                        "acceptable_precision_at_k": 1.0,
                        "all_required_at_k": 1.0,
                        "ndcg_at_k": 1.0,
                        "relevant_level_coverage_at_k": 1.0,
                    },
                },
            },
        },
        {
            "case_group": "l2",
            "runs_by_k": {
                "1": {
                    "reciprocal_rank_at_k": 0.5,
                    "ranked_unique_document_count": 3,
                    "duplicate_document_hit_count": 1,
                    "first_relevant_document_rank_at_k": 2,
                    "metrics": {
                        "hit_at_k": 0.0,
                        "recall_at_k": 0.0,
                        "precision_at_k": 0.0,
                        "acceptable_precision_at_k": 0.0,
                        "all_required_at_k": 0.0,
                        "ndcg_at_k": 0.0,
                        "relevant_level_coverage_at_k": 0.0,
                    },
                },
            },
        },
    ]

    by_group = aggregate_by_group(case_results, k_values=(1,))

    assert "l1" in by_group
    assert "l2" in by_group
    assert by_group["l1"]["cases_evaluated"] == 1
    assert by_group["l2"]["cases_evaluated"] == 1
    assert by_group["l1"]["metrics_by_k"]["1"]["mean_reciprocal_rank_at_k"] == pytest.approx(
        1.0
    )
    assert by_group["l2"]["metrics_by_k"]["1"]["mean_reciprocal_rank_at_k"] == pytest.approx(
        0.5
    )


# ---------------------------------------------------------------------------
# 20. test_output_is_json_serializable
# ---------------------------------------------------------------------------


def test_output_is_json_serializable():
    """Complete synthetic Schema 2.0 result can be serialized to JSON."""
    case_result = {
        "case_id": "test-01",
        "case_group": "l1",
        "query": "test query",
        "expected_route": "semantic",
        "primary_level": "1",
        "acceptable_levels": ["1"],
        "required_relevant_document_ids": ["doc1"],
        "optional_relevant_document_ids": [],
        "required_relevant_levels": ["1"],
        "method": "bm25",
        "candidate_chunk_depth": 100,
        "runs_by_k": {
            "1": {
                "final_k": 1,
                "candidate_chunk_depth": 100,
                "raw_chunk_result_count": 5,
                "ranked_unique_document_count": 1,
                "duplicate_document_hit_count": 0,
                "ranked_document_ids": ["doc1"],
                "ranked_documents": [
                    {
                        "document_id": "doc1",
                        "document_rank": 1,
                        "first_chunk_rank": 1,
                        "first_chunk_id": "c1",
                        "level": "1",
                        "source": "bm25",
                        "score": 0.9,
                        "rrf_score": None,
                        "duplicate_chunk_count": 0,
                    }
                ],
                "reciprocal_rank_at_k": 1.0,
                "first_relevant_document_rank_at_k": 1,
                "metrics": {
                    "hit_at_k": 1.0,
                    "recall_at_k": 1.0,
                    "precision_at_k": 1.0,
                    "acceptable_precision_at_k": 1.0,
                    "all_required_at_k": 1.0,
                    "ndcg_at_k": 1.0,
                    "relevant_level_coverage_at_k": 1.0,
                    "required_documents_retrieved": ["doc1"],
                    "required_documents_missing": [],
                    "optional_documents_retrieved": [],
                },
            }
        },
    }

    result = {
        "evaluator_schema_version": "2.0",
        "run_metadata": {
            "dataset_id": "test",
            "ground_truth_schema_version": "1.0",
            "chunks_sha256": "abc123",
            "case_count": 1,
            "methods": ["bm25"],
            "k_values": [1],
            "candidate_chunk_depth": 100,
            "relevance_unit": "document_id",
            "document_ranking_policy": "best_chunk_first_occurrence",
            "optional_document_policy": "acceptable_precision_only",
            "offline_embedding_required": True,
            "execution_policy": {
                "bm25": "single_candidate_depth_run_prefix_evaluation",
                "dense": "single_candidate_depth_run_prefix_evaluation",
                "hybrid": "independent_final_k_execution",
                "hybrid_candidate_depth": 100,
            },
        },
        "methods": {
            "bm25": {
                "overall": aggregate_case_results([case_result], (1,)),
                "by_case_group": {},
                "cases": [case_result],
            }
        },
        "integrity": {
            "ground_truth_validation_errors": [],
            "original_chroma_sha256_before": "abc",
            "original_chroma_sha256_after": "abc",
            "original_chroma_unchanged": True,
            "temporary_chroma_used": False,
            "temporary_chroma_deleted": True,
        },
    }

    # Must not raise
    json_str = json.dumps(result, indent=2)
    parsed = json.loads(json_str)
    assert parsed["evaluator_schema_version"] == "2.0"
    assert "execution_policy" in parsed["run_metadata"]
    assert parsed["run_metadata"]["execution_policy"]["hybrid"] == "independent_final_k_execution"


# ---------------------------------------------------------------------------
# 21. test_current_ground_truth_snapshot_is_valid
# ---------------------------------------------------------------------------


def test_current_ground_truth_snapshot_is_valid():
    """Validate the real current Ground Truth and chunks (read-only)."""
    from tests.semantic_ground_truth import (
        SEMANTIC_GROUND_TRUTH,
        SEMANTIC_GROUND_TRUTH_METADATA,
        SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION,
        validate_semantic_ground_truth,
    )

    assert SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION == "1.0"
    assert len(SEMANTIC_GROUND_TRUTH) == 24
    assert SEMANTIC_GROUND_TRUTH_METADATA["expected_case_count"] == 24
    assert SEMANTIC_GROUND_TRUTH_METADATA["dataset_id"] == "statsbomb-fifa-world-cup-2022"

    from collections import Counter

    groups = Counter(c["case_group"] for c in SEMANTIC_GROUND_TRUTH)
    for g in ("l1", "l2", "l3", "l4", "team", "multi"):
        assert groups[g] == 4, f"Expected 4 cases in group {g}, got {groups[g]}"

    errors = validate_semantic_ground_truth(
        SEMANTIC_GROUND_TRUTH_METADATA,
        SEMANTIC_GROUND_TRUTH,
        Path("output/chunks.json"),
    )
    assert errors == []

    # Chunks hash
    import hashlib

    with open("output/chunks.json", "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_hash == SEMANTIC_GROUND_TRUTH_METADATA["chunks_sha256"]

    # All document IDs exist and no '-chunk-'
    import json

    with open("output/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_doc_ids = set()
    for c in chunks:
        doc_id = c.get("document_id", c.get("metadata", {}).get("document_id", ""))
        if doc_id:
            chunk_doc_ids.add(doc_id)

    for case in SEMANTIC_GROUND_TRUTH:
        for doc_id in case["relevant_document_ids"]:
            assert doc_id in chunk_doc_ids, f"Required doc {doc_id} not in chunks"
            assert "-chunk-" not in doc_id
        for doc_id in case.get("optional_relevant_document_ids", []):
            assert doc_id in chunk_doc_ids, f"Optional doc {doc_id} not in chunks"
            assert "-chunk-" not in doc_id


# ---------------------------------------------------------------------------
# 22. test_temporary_chroma_context_uses_copy_and_cleans_up
# ---------------------------------------------------------------------------


def test_temporary_chroma_context_uses_copy_and_cleans_up(tmp_path):
    """Temporary Chroma context uses a copy, patches path, and cleans up."""
    from tests.retrieval_evaluator import temporary_chroma_copy

    # Create a fake original Chroma directory
    original_dir = str(tmp_path / "original_chroma")
    os.makedirs(original_dir, exist_ok=True)
    sqlite_path = os.path.join(original_dir, "chroma.sqlite3")
    with open(sqlite_path, "wb") as f:
        f.write(b"fake sqlite content for testing")

    # Hash original
    import hashlib

    with open(sqlite_path, "rb") as f:
        original_hash = hashlib.sha256(f.read()).hexdigest()

    # Mock retrieval module
    mock_module = MagicMock()
    mock_module.CHROMA_DIR = original_dir

    with temporary_chroma_copy(original_dir, mock_module) as tmp_chroma:
        # Patched path should point to copy
        assert mock_module.CHROMA_DIR == tmp_chroma
        assert tmp_chroma != original_dir
        assert os.path.exists(tmp_chroma)

        # Copy SQLite should match original
        copy_sqlite = os.path.join(tmp_chroma, "chroma.sqlite3")
        with open(copy_sqlite, "rb") as f:
            copy_hash = hashlib.sha256(f.read()).hexdigest()
        assert copy_hash == original_hash

    # After exit: original path restored
    assert mock_module.CHROMA_DIR == original_dir

    # Temporary directory deleted
    assert not os.path.exists(tmp_chroma)

    # Original unchanged
    with open(sqlite_path, "rb") as f:
        post_hash = hashlib.sha256(f.read()).hexdigest()
    assert post_hash == original_hash


# ---------------------------------------------------------------------------
# 23. test_bm25_only_run_does_not_request_chroma_copy
# ---------------------------------------------------------------------------


def test_bm25_only_run_does_not_request_chroma_copy():
    """BM25-only evaluation does not create a Chroma copy."""
    from tests.retrieval_evaluator import run_retrieval_baseline

    mock_module = MagicMock()
    mock_module.bm25_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    fake_metadata = {
        "dataset_id": "test",
        "schema_version": "1.0",
        "chunks_sha256": "abc",
        "expected_case_count": 1,
    }
    fake_cases = [
        {
            "id": "test-01",
            "dataset_id": "test",
            "case_group": "l1",
            "query": "test query",
            "expected_route": "semantic",
            "answerability": "answerable",
            "primary_level": "1",
            "acceptable_levels": ["1"],
            "relevant_document_ids": ["doc1"],
            "optional_relevant_document_ids": [],
            "required_facts": [],
            "forbidden_claims": [],
            "notes": "",
        }
    ]
    fake_doc_levels = {"doc1": "1"}

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
    ) as mock_chroma:
        result = run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
        )

        # temporary_chroma_copy should NOT have been called
        mock_chroma.assert_not_called()
        assert "bm25" in result["methods"]
        assert result["integrity"]["temporary_chroma_used"] is False


# ---------------------------------------------------------------------------
# 24. test_dense_or_hybrid_run_requests_one_temporary_copy
# ---------------------------------------------------------------------------


def test_dense_or_hybrid_run_requests_one_temporary_copy():
    """Dense+Hybrid combined uses one temporary copy, not one per case."""
    from contextlib import contextmanager as _contextmanager
    from tests.retrieval_evaluator import run_retrieval_baseline

    mock_module = MagicMock()
    mock_module.dense_search = MagicMock(return_value=[])
    mock_module.hybrid_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    fake_metadata = {
        "dataset_id": "test",
        "schema_version": "1.0",
        "chunks_sha256": "abc",
        "expected_case_count": 1,
    }
    fake_cases = [
        {
            "id": "test-01",
            "dataset_id": "test",
            "case_group": "l1",
            "query": "test query",
            "expected_route": "semantic",
            "answerability": "answerable",
            "primary_level": "1",
            "acceptable_levels": ["1"],
            "relevant_document_ids": ["doc1"],
            "optional_relevant_document_ids": [],
            "required_facts": [],
            "forbidden_claims": [],
            "notes": "",
        }
    ]
    fake_doc_levels = {"doc1": "1"}

    call_count = 0

    @_contextmanager
    def counting_chroma_copy(original_dir, mod):
        nonlocal call_count
        call_count += 1
        mod.CHROMA_DIR = "/tmp/fake"
        yield "/tmp/fake"
        mod.CHROMA_DIR = original_dir

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
        side_effect=counting_chroma_copy,
    ):
        run_retrieval_baseline(
            methods=("dense", "hybrid"),
            k_values=(1,),
            candidate_chunk_depth=100,
        )

        # Exactly one temporary copy for combined dense+hybrid
        assert call_count == 1


# ---------------------------------------------------------------------------
# 25. test_cli_argument_validation_rejects_non_positive_k
# ---------------------------------------------------------------------------


def test_cli_argument_validation_rejects_non_positive_k():
    """CLI rejects K <= 0."""
    from tests.retrieval_evaluator import main

    assert main(["--k-values", "0"]) == 1
    assert main(["--k-values", "-1"]) == 1


# ---------------------------------------------------------------------------
# 26. test_cli_argument_validation_rejects_candidate_depth_below_max_k
# ---------------------------------------------------------------------------


def test_cli_argument_validation_rejects_candidate_depth_below_max_k():
    """CLI rejects candidate depth < max(k_values)."""
    from tests.retrieval_evaluator import main

    exit_code = main(["--k-values", "10", "--candidate-chunk-depth", "5"])
    assert exit_code == 1


# ---------------------------------------------------------------------------
# 27. test_per_case_output_does_not_include_chunk_text
# ---------------------------------------------------------------------------


def test_per_case_output_does_not_include_chunk_text():
    """Per-case output must not contain full chunk text."""
    results = [
        _make_chunk_result(chunk_id="c1", document_id="doc1", text="full chunk text here"),
        _make_chunk_result(chunk_id="c2", document_id="doc2", text="another chunk text"),
    ]

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1", "doc2": "2"}

    def fake_bm25(query, k=20):
        return results

    output = evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
    )

    # Schema 2.0: ranked_documents lives inside runs_by_k
    for k_str, run in output["runs_by_k"].items():
        for doc in run["ranked_documents"]:
            assert "text" not in doc

    # The case output itself should not have chunk text embedded
    output_str = json.dumps(output)
    assert "full chunk text here" not in output_str
    assert "another chunk text" not in output_str


# ---------------------------------------------------------------------------
# 28. test_metric_values_stay_between_zero_and_one
# ---------------------------------------------------------------------------


def test_metric_values_stay_between_zero_and_one():
    """All normalized metric fields are in [0, 1]."""
    ranked_ids = ["d1", "d2", "d3", "d4", "d5"]
    required = ["d1", "d3", "d5"]
    optional = ["d2"]
    doc_levels = {"d1": "1", "d2": "2", "d3": "3", "d4": "4", "d5": "1"}

    for k in (1, 2, 3, 4, 5):
        h = hit_at_k(ranked_ids, required, k)
        r = recall_at_k(ranked_ids, required, k)
        p = precision_at_k(ranked_ids, required, k)
        ap = acceptable_precision_at_k(ranked_ids, required, optional, k)
        a = all_required_at_k(ranked_ids, required, k)
        n = ndcg_at_k(ranked_ids, required, k)
        lc = relevant_level_coverage_at_k(ranked_ids, required, doc_levels, k)

        for val, name in [
            (h, "hit"),
            (r, "recall"),
            (p, "precision"),
            (ap, "acceptable_precision"),
            (a, "all_required"),
            (n, "ndcg"),
            (lc, "level_coverage"),
        ]:
            assert 0.0 <= val <= 1.0, f"{name}@{k} = {val} out of range"

    rr = reciprocal_rank(ranked_ids, required)
    assert 0.0 <= rr <= 1.0


# ---------------------------------------------------------------------------
# 29. test_evaluator_schema_version_is_2_0
# ---------------------------------------------------------------------------


def test_evaluator_schema_version_is_2_0():
    """Evaluator schema version is 2.0."""
    assert RETRIEVAL_EVALUATOR_SCHEMA_VERSION == "2.0"


# ---------------------------------------------------------------------------
# 30. test_bm25_called_once_at_candidate_depth
# ---------------------------------------------------------------------------


def test_bm25_called_once_at_candidate_depth():
    """BM25 is called exactly once at candidate_chunk_depth, regardless of K count."""
    calls = []

    def fake_bm25(query, k=20):
        calls.append(k)
        return []

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    assert len(calls) == 1
    assert calls[0] == 100


# ---------------------------------------------------------------------------
# 31. test_dense_called_once_at_candidate_depth
# ---------------------------------------------------------------------------


def test_dense_called_once_at_candidate_depth():
    """Dense is called exactly once at candidate_chunk_depth, regardless of K count."""
    calls = []

    def fake_dense(query, k=20, level_filter=None):
        calls.append({"k": k, "level_filter": level_filter})
        return []

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1"}

    evaluate_case(
        case=case,
        retrieval_fn=fake_dense,
        method="dense",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    assert len(calls) == 1
    assert calls[0] == {"k": 100, "level_filter": None}


# ---------------------------------------------------------------------------
# 32. test_hybrid_k_dependent_retriever_produces_k_scoped_metrics
# ---------------------------------------------------------------------------


def test_hybrid_k_dependent_retriever_produces_k_scoped_metrics():
    """Metrics come from independent K Hybrid runs, not a sliced k=100 ranking."""
    def k_dependent_hybrid(query, k=5, bm25_k=20, dense_k=20, level_filter=None):
        if k == 1:
            # doc_b only, no required doc
            return [_make_chunk_result(chunk_id="c_b", document_id="doc_b", rank=1)]
        elif k == 3:
            # doc_a at rank 1
            return [
                _make_chunk_result(chunk_id="c_a", document_id="doc_a", rank=1),
                _make_chunk_result(chunk_id="c_b", document_id="doc_b", rank=2),
                _make_chunk_result(chunk_id="c_c", document_id="doc_c", rank=3),
            ]
        elif k == 5:
            # doc_a at rank 5
            return [
                _make_chunk_result(chunk_id="c_b", document_id="doc_b", rank=1),
                _make_chunk_result(chunk_id="c_c", document_id="doc_c", rank=2),
                _make_chunk_result(chunk_id="c_d", document_id="doc_d", rank=3),
                _make_chunk_result(chunk_id="c_e", document_id="doc_e", rank=4),
                _make_chunk_result(chunk_id="c_a", document_id="doc_a", rank=5),
            ]
        else:  # k=10
            # doc_a at rank 1
            results = [_make_chunk_result(chunk_id="c_a", document_id="doc_a", rank=1)]
            for i in range(2, 11):
                results.append(
                    _make_chunk_result(chunk_id=f"c_{i}", document_id=f"doc_{i}", rank=i)
                )
            return results

    case = _make_case(required_docs=["doc_a"])
    doc_levels = {"doc_a": "1", "doc_b": "2", "doc_c": "3", "doc_d": "4", "doc_e": "5"}
    for i in range(2, 11):
        doc_levels[f"doc_{i}"] = str(i)

    result = evaluate_case(
        case=case,
        retrieval_fn=k_dependent_hybrid,
        method="hybrid",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    runs = result["runs_by_k"]

    # k=1: doc_b only, no required doc
    assert runs["1"]["reciprocal_rank_at_k"] == 0.0
    assert runs["1"]["first_relevant_document_rank_at_k"] is None
    assert runs["1"]["metrics"]["hit_at_k"] == 0.0
    assert runs["1"]["raw_chunk_result_count"] == 1

    # k=3: doc_a at rank 1
    assert runs["3"]["reciprocal_rank_at_k"] == 1.0
    assert runs["3"]["first_relevant_document_rank_at_k"] == 1
    assert runs["3"]["metrics"]["hit_at_k"] == 1.0
    assert runs["3"]["raw_chunk_result_count"] == 3

    # k=5: doc_a at rank 5
    assert runs["5"]["reciprocal_rank_at_k"] == pytest.approx(0.2)
    assert runs["5"]["first_relevant_document_rank_at_k"] == 5
    assert runs["5"]["metrics"]["hit_at_k"] == 1.0  # doc_a in top-5
    assert runs["5"]["raw_chunk_result_count"] == 5

    # k=10: doc_a at rank 1
    assert runs["10"]["reciprocal_rank_at_k"] == 1.0
    assert runs["10"]["first_relevant_document_rank_at_k"] == 1
    assert runs["10"]["raw_chunk_result_count"] == 10

    # The key proof: k=1 has rr=0.0, k=3 has rr=1.0, k=5 has rr=0.2
    # If we had sliced a single k=100 run, all K values would share one ranking.
    # The differing rr values prove K-independence.
    assert runs["1"]["reciprocal_rank_at_k"] != runs["3"]["reciprocal_rank_at_k"]
    assert runs["3"]["reciprocal_rank_at_k"] != runs["5"]["reciprocal_rank_at_k"]


# ---------------------------------------------------------------------------
# 33. test_rank_metrics_are_k_scoped_with_correct_defaults
# ---------------------------------------------------------------------------


def test_rank_metrics_are_k_scoped_with_correct_defaults():
    """reciprocal_rank_at_k=0.0 and first_relevant_document_rank_at_k=None when relevant docs outside K."""
    def no_required_hybrid(query, k=5, bm25_k=20, dense_k=20, level_filter=None):
        return [
            _make_chunk_result(chunk_id=f"c_{i}", document_id=f"doc_{i}", rank=i)
            for i in range(1, k + 1)
        ]

    def no_required_bm25(query, k=20):
        return [
            _make_chunk_result(chunk_id=f"c_{i}", document_id=f"doc_{i}", rank=i)
            for i in range(1, k + 1)
        ]

    case = _make_case(required_docs=["required_doc"])
    doc_levels = {"required_doc": "1"}
    for i in range(1, 11):
        doc_levels[f"doc_{i}"] = str(i)

    # Hybrid
    result_hybrid = evaluate_case(
        case=case,
        retrieval_fn=no_required_hybrid,
        method="hybrid",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    for k_str in ("1", "3", "5", "10"):
        assert result_hybrid["runs_by_k"][k_str]["reciprocal_rank_at_k"] == 0.0
        assert result_hybrid["runs_by_k"][k_str]["first_relevant_document_rank_at_k"] is None
        assert result_hybrid["runs_by_k"][k_str]["metrics"]["hit_at_k"] == 0.0
        assert result_hybrid["runs_by_k"][k_str]["metrics"]["recall_at_k"] == 0.0

    # BM25
    result_bm25 = evaluate_case(
        case=case,
        retrieval_fn=no_required_bm25,
        method="bm25",
        document_levels=doc_levels,
        k_values=(1, 3, 5, 10),
        candidate_chunk_depth=100,
    )

    for k_str in ("1", "3", "5", "10"):
        assert result_bm25["runs_by_k"][k_str]["reciprocal_rank_at_k"] == 0.0
        assert result_bm25["runs_by_k"][k_str]["first_relevant_document_rank_at_k"] is None


# ---------------------------------------------------------------------------
# 34. test_schema_2_0_output_contains_runs_by_k
# ---------------------------------------------------------------------------


def test_schema_2_0_output_contains_runs_by_k():
    """Schema 2.0 output contains runs_by_k with required fields per K."""
    results = [
        _make_chunk_result(chunk_id="c1", document_id="doc1", rank=1),
        _make_chunk_result(chunk_id="c2", document_id="doc2", rank=2),
    ]

    case = _make_case(required_docs=["doc1"])
    doc_levels = {"doc1": "1", "doc2": "2"}

    def fake_bm25(query, k=20):
        return results

    output = evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels=doc_levels,
        k_values=(1, 3),
        candidate_chunk_depth=100,
    )

    assert "runs_by_k" in output
    assert "1" in output["runs_by_k"]
    assert "3" in output["runs_by_k"]

    required_run_keys = {
        "final_k",
        "candidate_chunk_depth",
        "raw_chunk_result_count",
        "ranked_unique_document_count",
        "duplicate_document_hit_count",
        "ranked_document_ids",
        "ranked_documents",
        "reciprocal_rank_at_k",
        "first_relevant_document_rank_at_k",
        "metrics",
    }

    for k_str in ("1", "3"):
        run = output["runs_by_k"][k_str]
        assert required_run_keys.issubset(run.keys()), (
            f"Missing keys in runs_by_k[{k_str}]: {required_run_keys - run.keys()}"
        )

    # K-specific values
    assert output["runs_by_k"]["1"]["final_k"] == 1
    assert output["runs_by_k"]["3"]["final_k"] == 3
    assert output["runs_by_k"]["1"]["candidate_chunk_depth"] == 100
    assert output["runs_by_k"]["1"]["ranked_unique_document_count"] == 1
    assert output["runs_by_k"]["3"]["ranked_unique_document_count"] == 2


# ---------------------------------------------------------------------------
# 35. test_semantic_ground_truth_unchanged_at_1_0
# ---------------------------------------------------------------------------


def test_semantic_ground_truth_unchanged_at_1_0():
    """Semantic Ground Truth schema version remains 1.0 (not affected by evaluator 2.0)."""
    from tests.semantic_ground_truth import (
        SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION,
    )

    assert SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# 36. Competition portability -- ArtifactPaths propagation
# ---------------------------------------------------------------------------


def test_evaluate_case_threads_namespaced_artifact_paths():
    from src.artifacts import ArtifactPaths

    selected = ArtifactPaths(2, 27)
    captured = {}

    def fake_bm25(query, k=20, artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return []

    case = _make_case(required_docs=["doc1"])

    evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels={"doc1": "1"},
        artifact_paths=selected,
    )

    assert captured["artifact_paths"] is selected


def test_evaluate_case_threads_namespaced_artifact_paths_for_dense():
    """Dense retrieval also receives artifact_paths when one is selected."""
    from src.artifacts import ArtifactPaths

    selected = ArtifactPaths(2, 27)
    captured = {}

    def fake_dense(query, k=20, level_filter=None, artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return []

    case = _make_case(required_docs=["doc1"])

    evaluate_case(
        case=case,
        retrieval_fn=fake_dense,
        method="dense",
        document_levels={"doc1": "1"},
        artifact_paths=selected,
    )

    assert captured["artifact_paths"] is selected


def test_evaluate_case_threads_namespaced_artifact_paths_for_hybrid():
    """Hybrid retrieval receives artifact_paths on every independent K execution."""
    from src.artifacts import ArtifactPaths

    selected = ArtifactPaths(2, 27)
    captured = []

    def fake_hybrid(
        query, k=20, bm25_k=100, dense_k=100, level_filter=None, artifact_paths=None
    ):
        captured.append(artifact_paths)
        return []

    case = _make_case(required_docs=["doc1"])

    evaluate_case(
        case=case,
        retrieval_fn=fake_hybrid,
        method="hybrid",
        document_levels={"doc1": "1"},
        k_values=(1, 3),
        artifact_paths=selected,
    )

    assert captured == [selected, selected]


def test_evaluate_case_omits_artifact_paths_kwarg_for_legacy_bm25_fake():
    """A legacy BM25 fake with no artifact_paths parameter still works when None."""

    def fake_bm25(query, k=20):
        return []

    case = _make_case(required_docs=["doc1"])

    result = evaluate_case(
        case=case,
        retrieval_fn=fake_bm25,
        method="bm25",
        document_levels={"doc1": "1"},
    )

    assert result["method"] == "bm25"


def test_evaluate_case_omits_artifact_paths_kwarg_for_legacy_dense_fake():
    """A legacy Dense fake with no artifact_paths parameter still works when None."""

    def fake_dense(query, k=20, level_filter=None):
        return []

    case = _make_case(required_docs=["doc1"])

    result = evaluate_case(
        case=case,
        retrieval_fn=fake_dense,
        method="dense",
        document_levels={"doc1": "1"},
    )

    assert result["method"] == "dense"


def test_evaluate_case_omits_artifact_paths_kwarg_for_legacy_hybrid_fake():
    """A legacy Hybrid fake with no artifact_paths parameter still works when None."""

    def fake_hybrid(query, k=20, bm25_k=100, dense_k=100, level_filter=None):
        return []

    case = _make_case(required_docs=["doc1"])

    result = evaluate_case(
        case=case,
        retrieval_fn=fake_hybrid,
        method="hybrid",
        document_levels={"doc1": "1"},
    )

    assert result["method"] == "hybrid"


def test_evaluate_retrieval_method_forwards_artifact_paths_to_every_case():
    """evaluate_retrieval_method threads artifact_paths through to each case."""
    from src.artifacts import ArtifactPaths

    selected = ArtifactPaths(2, 27)
    captured = []

    class FakeModule:
        @staticmethod
        def bm25_search(query, k=20, artifact_paths=None):
            captured.append(artifact_paths)
            return []

    cases = [
        _make_case(case_id="c1", required_docs=["doc1"]),
        _make_case(case_id="c2", required_docs=["doc1"]),
    ]

    evaluate_retrieval_method(
        method="bm25",
        cases=cases,
        retrieval_module=FakeModule,
        document_levels={"doc1": "1"},
        artifact_paths=selected,
    )

    assert captured == [selected, selected]


# ---------------------------------------------------------------------------
# 37. Competition portability -- Ground Truth injection (GroundTruthBundle)
# ---------------------------------------------------------------------------


def test_validate_ground_truth_and_chunks_defaults_to_wc2022_bundle():
    """Omitting `ground_truth` keeps validating against the real WC2022 benchmark."""
    from tests.retrieval_evaluator import validate_ground_truth_and_chunks
    from tests.semantic_ground_truth import SEMANTIC_GROUND_TRUTH_METADATA

    metadata, cases, document_levels = validate_ground_truth_and_chunks()

    assert metadata is SEMANTIC_GROUND_TRUTH_METADATA
    assert len(cases) == 24


def test_validate_ground_truth_and_chunks_accepts_custom_ground_truth_bundle(tmp_path):
    """A caller-supplied GroundTruthBundle is used instead of the WC2022 default."""
    import hashlib

    from tests.retrieval_evaluator import (
        GroundTruthBundle,
        validate_ground_truth_and_chunks,
    )

    chunks_path = tmp_path / "chunks.json"
    chunks = [{"document_id": "epl-doc-1", "level": "1", "chunk_id": "c1", "text": "x"}]
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    actual_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    metadata = {
        "dataset_id": "epl-2015-2016",
        "schema_version": "1.0",
        "chunks_sha256": actual_hash,
        "expected_case_count": 1,
    }
    cases = [_make_case(case_id="epl-01", required_docs=["epl-doc-1"])]

    validate_calls = []

    def fake_validate_fn(meta, c, path):
        validate_calls.append((meta, c, path))
        return []

    bundle = GroundTruthBundle(metadata=metadata, cases=cases, validate_fn=fake_validate_fn)

    resolved_metadata, resolved_cases, document_levels = validate_ground_truth_and_chunks(
        chunks_path=str(chunks_path),
        ground_truth=bundle,
    )

    assert resolved_metadata is metadata
    assert resolved_cases is cases
    assert document_levels == {"epl-doc-1": "1"}
    assert len(validate_calls) == 1
    assert validate_calls[0][0] is metadata
    assert validate_calls[0][1] is cases


def test_validate_ground_truth_and_chunks_rejects_hash_mismatch_for_custom_bundle(tmp_path):
    """Chunks-hash integrity checking is not weakened for a custom benchmark."""
    from tests.retrieval_evaluator import (
        GroundTruthBundle,
        RetrievalEvaluationError,
        validate_ground_truth_and_chunks,
    )

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps([{"document_id": "d1", "level": "1"}]), encoding="utf-8")

    metadata = {
        "dataset_id": "epl-2015-2016",
        "schema_version": "1.0",
        "chunks_sha256": "0" * 64,  # deliberately wrong
        "expected_case_count": 0,
    }
    bundle = GroundTruthBundle(metadata=metadata, cases=[], validate_fn=lambda m, c, p: [])

    with pytest.raises(RetrievalEvaluationError, match="hash mismatch"):
        validate_ground_truth_and_chunks(chunks_path=str(chunks_path), ground_truth=bundle)


def test_validate_ground_truth_and_chunks_propagates_custom_validator_errors(tmp_path):
    """A custom benchmark's own structural validator errors still fail evaluation."""
    import hashlib

    from tests.retrieval_evaluator import (
        GroundTruthBundle,
        RetrievalEvaluationError,
        validate_ground_truth_and_chunks,
    )

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text("[]", encoding="utf-8")
    actual_hash = hashlib.sha256(chunks_path.read_bytes()).hexdigest()

    metadata = {"dataset_id": "x", "chunks_sha256": actual_hash}
    bundle = GroundTruthBundle(
        metadata=metadata, cases=[], validate_fn=lambda m, c, p: ["synthetic error"]
    )

    with pytest.raises(RetrievalEvaluationError, match="synthetic error"):
        validate_ground_truth_and_chunks(chunks_path=str(chunks_path), ground_truth=bundle)


# ---------------------------------------------------------------------------
# 38. Competition portability -- run_retrieval_baseline artifact resolution
# ---------------------------------------------------------------------------


def _fake_gt_fixture():
    fake_metadata = {
        "dataset_id": "test",
        "schema_version": "1.0",
        "chunks_sha256": "abc",
        "expected_case_count": 1,
    }
    fake_cases = [_make_case(case_id="test-01", required_docs=["doc1"])]
    fake_doc_levels = {"doc1": "1"}
    return fake_metadata, fake_cases, fake_doc_levels


def test_run_retrieval_baseline_resolves_namespaced_chunks_path(tmp_path):
    """run_retrieval_baseline validates against the selected dataset's chunks.json."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: [])

    mock_module = MagicMock()
    mock_module.bm25_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    captured_chunks_paths = []

    def fake_validate(chunks_path, ground_truth=None):
        captured_chunks_paths.append(chunks_path)
        return fake_metadata, fake_cases, fake_doc_levels

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        side_effect=fake_validate,
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ):
        run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert captured_chunks_paths == [str(selected.chunks)]


def test_run_retrieval_baseline_resolves_namespaced_chroma_dir(tmp_path):
    """run_retrieval_baseline copies the selected dataset's Chroma dir, not the legacy one."""
    from contextlib import contextmanager as _contextmanager

    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: [])

    mock_module = MagicMock()
    mock_module.dense_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    captured_chroma_dirs = []

    @_contextmanager
    def fake_chroma_copy(original_dir, mod):
        captured_chroma_dirs.append(original_dir)
        yield original_dir

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
        side_effect=fake_chroma_copy,
    ):
        run_retrieval_baseline(
            methods=("dense",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert captured_chroma_dirs == [str(selected.chroma_dir)]


# ---------------------------------------------------------------------------
# 40. Competition portability -- Dense/Hybrid must never bypass the
#     temporary Chroma copy for a namespaced dataset (TemporaryChromaArtifactPaths)
# ---------------------------------------------------------------------------


def test_namespaced_dense_evaluation_receives_temporary_chroma_view(tmp_path):
    """Dense evaluation for a namespaced dataset must receive an
    artifact-path view whose chroma_dir points at the temporary Chroma
    copy, while chunks/bm25_index/chroma_collection_name still resolve to
    the real, selected dataset -- proving Dense never opens the source
    Chroma directory directly during evaluation."""
    from contextlib import contextmanager as _contextmanager

    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    tmp_chroma_copy = str(tmp_path / "tmp_chroma_copy")
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(
        metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: []
    )

    captured = []

    def fake_dense(query, k=20, level_filter=None, artifact_paths=None):
        captured.append(artifact_paths)
        return []

    mock_module = MagicMock()
    mock_module.dense_search = fake_dense
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    @_contextmanager
    def fake_chroma_copy(original_dir, mod):
        yield tmp_chroma_copy

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
        side_effect=fake_chroma_copy,
    ):
        run_retrieval_baseline(
            methods=("dense",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert len(captured) == 1
    received = captured[0]
    assert received is not selected
    assert received.chroma_dir == Path(tmp_chroma_copy)
    assert received.chunks == selected.chunks
    assert received.bm25_index == selected.bm25_index
    assert received.chroma_collection_name == selected.chroma_collection_name
    assert received.competition_id == selected.competition_id
    assert received.season_id == selected.season_id


def test_namespaced_hybrid_evaluation_receives_same_temporary_chroma_view_for_every_k(tmp_path):
    """Hybrid retrieval executes independently per K (Schema 2.0) -- every
    one of those executions must receive the identical temporary-Chroma
    artifact-path view (not the real artifact_paths, and not a fresh proxy
    reconstructed per K)."""
    from contextlib import contextmanager as _contextmanager

    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    tmp_chroma_copy = str(tmp_path / "tmp_chroma_copy")
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(
        metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: []
    )

    captured = []

    def fake_hybrid(
        query, k=20, bm25_k=100, dense_k=100, level_filter=None, artifact_paths=None
    ):
        captured.append(artifact_paths)
        return []

    mock_module = MagicMock()
    mock_module.hybrid_search = fake_hybrid
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    @_contextmanager
    def fake_chroma_copy(original_dir, mod):
        yield tmp_chroma_copy

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
        side_effect=fake_chroma_copy,
    ):
        run_retrieval_baseline(
            methods=("hybrid",),
            k_values=(1, 3, 5),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert len(captured) == 3  # one independent execution per K
    assert all(c.chroma_dir == Path(tmp_chroma_copy) for c in captured)
    assert all(c is captured[0] for c in captured), "expected the same view object every K"


def test_temporary_chroma_view_does_not_mutate_original_artifact_paths(tmp_path):
    """Constructing a TemporaryChromaArtifactPaths view must never mutate
    the original, selected ArtifactPaths object it wraps."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import TemporaryChromaArtifactPaths

    original = ArtifactPaths(2, 27, output_root=tmp_path)
    original_chroma_dir = original.chroma_dir
    original_chunks = original.chunks
    original_collection_name = original.chroma_collection_name

    view = TemporaryChromaArtifactPaths(
        base=original, chroma_dir_override=tmp_path / "tmp_chroma_copy"
    )

    # The view overrides chroma_dir...
    assert view.chroma_dir == tmp_path / "tmp_chroma_copy"
    # ...but delegates everything else, and the original is untouched.
    assert original.chroma_dir == original_chroma_dir
    assert original.chunks == original_chunks
    assert original.chroma_collection_name == original_collection_name
    assert view.chunks == original_chunks
    assert view.bm25_index == original.bm25_index
    assert view.chroma_collection_name == original_collection_name
    assert view.base is original


def test_namespaced_bm25_only_evaluation_receives_original_artifact_paths(tmp_path):
    """BM25-only namespaced evaluation must receive the real, selected
    artifact_paths object directly (by identity, not a proxy) and must
    never trigger a temporary Chroma copy at all."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(
        metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: []
    )

    captured = []

    def fake_bm25(query, k=20, artifact_paths=None):
        captured.append(artifact_paths)
        return []

    mock_module = MagicMock()
    mock_module.bm25_search = fake_bm25
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
    ) as mock_chroma_copy:
        run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert len(captured) == 1
    assert captured[0] is selected
    mock_chroma_copy.assert_not_called()


def test_legacy_chroma_evaluation_receives_temporary_flat_layout_view():
    """Legacy Dense evaluation must explicitly use the temporary Chroma copy.

    The view must keep legacy flat-layout chunks/BM25 paths while overriding
    only Chroma, avoiding reliance on module-level CHROMA_DIR patching.
    """
    from contextlib import contextmanager as _contextmanager
    from pathlib import Path

    from tests.retrieval_evaluator import run_retrieval_baseline

    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    captured = []

    def fake_dense(query, k=20, level_filter=None, artifact_paths=None):
        captured.append(artifact_paths)
        return []

    mock_module = MagicMock()
    mock_module.dense_search = fake_dense
    mock_module._bm25_cache = {}
    mock_module._chunks_cache = {}

    @_contextmanager
    def fake_chroma_copy(original_dir, mod):
        yield "/tmp/fake-legacy-chroma"

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ), patch(
        "tests.retrieval_evaluator.temporary_chroma_copy",
        side_effect=fake_chroma_copy,
    ):
        run_retrieval_baseline(
            methods=("dense",),
            k_values=(1,),
            candidate_chunk_depth=100,
        )

    assert len(captured) == 1
    view = captured[0]
    assert view is not None
    assert Path(view.chroma_dir) == Path("/tmp/fake-legacy-chroma")
    assert Path(view.chunks) == Path("output/chunks.json")
    assert Path(view.bm25_index) == Path("output/indices/bm25.pkl")
    assert view.chroma_collection_name == "wc2022_documents"


def test_run_retrieval_baseline_integrity_checks_remain_active_for_namespaced_dataset(tmp_path):
    """The pre/post Chroma SHA-256 integrity check in run_retrieval_baseline's
    result must still run for namespaced datasets, using the REAL
    temporary_chroma_copy (not mocked) -- proving the switch to
    TemporaryChromaArtifactPaths did not disable or skip integrity checks,
    and that Dense actually went through the copy, not the real directory."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    selected.chroma_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = selected.chroma_dir / "chroma.sqlite3"
    sqlite_path.write_bytes(b"fake sqlite content for integrity test")

    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()
    bundle = GroundTruthBundle(
        metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: []
    )

    captured_dense_artifact_paths = []

    def fake_dense(query, k=20, level_filter=None, artifact_paths=None):
        captured_dense_artifact_paths.append(artifact_paths)
        return []

    mock_module = MagicMock()
    mock_module.dense_search = fake_dense
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ):
        result = run_retrieval_baseline(
            methods=("dense",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    # Integrity block is populated and reports the source unchanged.
    assert result["integrity"]["original_chroma_sha256_before"] is not None
    assert result["integrity"]["original_chroma_unchanged"] is True
    assert result["integrity"]["temporary_chroma_used"] is True

    # The retrieval call actually went through a copy, not the real
    # selected.chroma_dir.
    assert len(captured_dense_artifact_paths) == 1
    assert captured_dense_artifact_paths[0].chroma_dir != selected.chroma_dir

    # Source file on disk is untouched.
    assert sqlite_path.read_bytes() == b"fake sqlite content for integrity test"


def test_run_retrieval_baseline_uses_legacy_paths_when_artifact_paths_is_none():
    """Legacy no-selection behavior keeps validating output/chunks.json."""
    from tests.retrieval_evaluator import run_retrieval_baseline

    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()

    mock_module = MagicMock()
    mock_module.bm25_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    captured_chunks_paths = []

    def fake_validate(chunks_path, ground_truth=None):
        captured_chunks_paths.append(chunks_path)
        return fake_metadata, fake_cases, fake_doc_levels

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        side_effect=fake_validate,
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ):
        run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
        )

    assert captured_chunks_paths == [os.path.join(".", "output", "chunks.json")]


def test_run_retrieval_baseline_refuses_non_wc2022_namespaced_dataset_without_ground_truth(tmp_path):
    """A non-WC2022 namespaced dataset with no explicit benchmark is refused,
    not silently paired with the default WC2022 Semantic Ground Truth."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import RetrievalEvaluationError, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)

    with pytest.raises(RetrievalEvaluationError, match="Ground Truth"):
        run_retrieval_baseline(artifact_paths=selected)


def test_run_retrieval_baseline_does_not_reject_namespaced_wc2022_identity(tmp_path):
    """Regression test: a namespaced artifact_paths whose (competition_id,
    season_id) matches the default WC2022 benchmark's own declared identity
    must NOT be rejected by the 'no explicit benchmark' guard -- it falls
    through to the default WC2022 GroundTruthBundle, and real chunks
    SHA-256 validation is what determines whether the namespaced snapshot
    actually matches."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import run_retrieval_baseline
    from tests.semantic_ground_truth import SEMANTIC_GROUND_TRUTH_METADATA

    selected = ArtifactPaths(43, 106, output_root=tmp_path)
    fake_metadata, fake_cases, fake_doc_levels = _fake_gt_fixture()

    mock_module = MagicMock()
    mock_module.bm25_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    captured_ground_truths = []

    def fake_validate(chunks_path, ground_truth=None):
        captured_ground_truths.append(ground_truth)
        return fake_metadata, fake_cases, fake_doc_levels

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        side_effect=fake_validate,
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ):
        # Must not raise -- this is the exact bug ChatGPT reported.
        run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
        )

    assert len(captured_ground_truths) == 1
    assert captured_ground_truths[0].metadata is SEMANTIC_GROUND_TRUTH_METADATA


def test_run_retrieval_baseline_accepts_explicit_ground_truth_matching_identity(tmp_path):
    """An explicit GroundTruthBundle whose declared identity matches
    artifact_paths is accepted through the identity guard."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import GroundTruthBundle, run_retrieval_baseline

    selected = ArtifactPaths(2, 27, output_root=tmp_path)
    fake_metadata = {
        "dataset_id": "epl-2015-2016",
        "schema_version": "1.0",
        "chunks_sha256": "abc",
        "expected_case_count": 1,
        "competition_id": 2,
        "season_id": 27,
    }
    fake_cases = [_make_case(case_id="epl-01", required_docs=["doc1"])]
    fake_doc_levels = {"doc1": "1"}
    bundle = GroundTruthBundle(
        metadata=fake_metadata, cases=fake_cases, validate_fn=lambda m, c, p: []
    )

    mock_module = MagicMock()
    mock_module.bm25_search = MagicMock(return_value=[])
    mock_module._bm25_cache = None
    mock_module._chunks_cache = None

    with patch(
        "tests.retrieval_evaluator.validate_ground_truth_and_chunks",
        return_value=(fake_metadata, fake_cases, fake_doc_levels),
    ), patch(
        "tests.retrieval_evaluator.load_retrieval_module",
        return_value=mock_module,
    ):
        result = run_retrieval_baseline(
            methods=("bm25",),
            k_values=(1,),
            candidate_chunk_depth=100,
            artifact_paths=selected,
            ground_truth=bundle,
        )

    assert "bm25" in result["methods"]


def test_run_retrieval_baseline_rejects_explicit_ground_truth_mismatched_identity(tmp_path):
    """An explicit GroundTruthBundle whose declared competition_id/season_id
    does not match artifact_paths is rejected before any retrieval."""
    from src.artifacts import ArtifactPaths
    from tests.retrieval_evaluator import (
        GroundTruthBundle,
        RetrievalEvaluationError,
        run_retrieval_baseline,
    )

    selected = ArtifactPaths(2, 27, output_root=tmp_path)

    def _bundle(competition_id, season_id):
        return GroundTruthBundle(
            metadata={
                "dataset_id": "wrong-dataset",
                "chunks_sha256": "abc",
                "competition_id": competition_id,
                "season_id": season_id,
            },
            cases=[],
            validate_fn=lambda m, c, p: [],
        )

    with patch("tests.retrieval_evaluator.load_retrieval_module") as mock_load:
        with pytest.raises(RetrievalEvaluationError, match="identity mismatch"):
            run_retrieval_baseline(artifact_paths=selected, ground_truth=_bundle(999, 27))
        with pytest.raises(RetrievalEvaluationError, match="identity mismatch"):
            run_retrieval_baseline(artifact_paths=selected, ground_truth=_bundle(2, 999))
        mock_load.assert_not_called()


def test_cli_does_not_falsely_reject_namespaced_wc2022_identity(capsys):
    """CLI-level regression test for the exact reported bug: selecting the
    namespaced WC2022 identity (competition_id=43, season_id=106) must not
    be rejected by the 'no explicit Ground Truth benchmark' guard. The
    namespaced WC2022 snapshot doesn't exist on disk in this repo, so this
    now fails on real chunks validation instead -- not a false rejection."""
    from tests.retrieval_evaluator import main

    exit_code = main(
        [
            "--competition-id", "43",
            "--season-id", "106",
            "--namespaced",
            "--methods", "bm25",
            "--k-values", "1",
            "--candidate-chunk-depth", "100",
        ]
    )

    captured = capsys.readouterr()
    assert "was supplied without an explicit Ground Truth benchmark" not in captured.err
    assert "identity does not match the default benchmark" not in captured.err
    assert exit_code == 1


# ---------------------------------------------------------------------------
# 39. Competition portability -- CLI dataset selection
# ---------------------------------------------------------------------------


def test_parse_args_defaults_have_no_dataset_selection():
    """No-argument CLI invocation requests no competition/season selection."""
    from tests.retrieval_evaluator import parse_args

    args = parse_args([])

    assert args.competition_id is None
    assert args.season_id is None
    assert args.namespaced is False


def test_cli_requires_competition_and_season_id_together():
    """--competition-id and --season-id must be given together."""
    from tests.retrieval_evaluator import main

    assert main(["--competition-id", "2"]) == 1
    assert main(["--season-id", "27"]) == 1


def test_cli_refuses_namespaced_dataset_without_ground_truth_benchmark(capsys):
    """Selecting a namespaced competition/season on the CLI fails clearly, not silently."""
    from tests.retrieval_evaluator import main

    exit_code = main(["--competition-id", "2", "--season-id", "27"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Ground Truth" in captured.err


def test_reset_retrieval_caches_preserves_dict_cache_contract():
    """Evaluator cache reset must leave retrieval path-keyed caches usable."""
    import types
    from tests.retrieval_evaluator import reset_retrieval_caches

    fake = types.SimpleNamespace(
        _bm25_cache={"x": object()},
        _chunks_cache={"y": []},
    )

    reset_retrieval_caches(fake)

    assert fake._bm25_cache == {}
    assert fake._chunks_cache == {}
