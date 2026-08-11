"""
Document-level retrieval evaluator for the Football-Analytics-RAG-FULL project.

Evaluates BM25, Dense, and Hybrid retrieval methods against the verified
18-case Semantic Ground Truth using a best-chunk-first-occurrence document
ranking policy.

Schema version: 2.0
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.artifacts import ArtifactPaths, resolve_runtime_artifact_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRIEVAL_EVALUATOR_SCHEMA_VERSION = "2.0"

DEFAULT_K_VALUES = (1, 3, 5, 10)

DEFAULT_CANDIDATE_CHUNK_DEPTH = 100

SUPPORTED_METHODS = ("bm25", "dense", "hybrid")

DOCUMENT_RANKING_POLICY = "best_chunk_first_occurrence"

RELEVANCE_UNIT = "document_id"

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RetrievalEvaluationError(Exception):
    """Raised when evaluation cannot proceed due to invalid data or state."""


# ---------------------------------------------------------------------------
# Ground Truth / Chunks Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthBundle:
    """
    Explicit Ground Truth benchmark selection for the evaluator.

    The evaluator is infrastructure, not a WC2022-only tool: it needs to
    validate/score whichever benchmark's cases match the chunks under
    evaluation. `validate_ground_truth_and_chunks` and
    `run_retrieval_baseline` default to `None`, which resolves to the
    existing, unmodified WC2022 Semantic Ground Truth
    (tests.semantic_ground_truth) -- unchanged legacy behavior. Passing a
    bundle lets a caller supply a different competition's metadata/cases/
    validator instead, without coupling this module to one fixed source.

    `validate_fn` has the same shape as
    `tests.semantic_ground_truth.validate_semantic_ground_truth`:
    (metadata, cases, chunks_path) -> list[str] of error strings.
    """

    metadata: dict
    cases: list
    validate_fn: Callable[[dict, list, Any], List[str]]


def _default_ground_truth_bundle() -> "GroundTruthBundle":
    from tests.semantic_ground_truth import (
        SEMANTIC_GROUND_TRUTH,
        SEMANTIC_GROUND_TRUTH_METADATA,
        validate_semantic_ground_truth,
    )

    return GroundTruthBundle(
        metadata=SEMANTIC_GROUND_TRUTH_METADATA,
        cases=SEMANTIC_GROUND_TRUTH,
        validate_fn=validate_semantic_ground_truth,
    )


def _ensure_ground_truth_matches_artifact_paths(
    metadata: dict,
    artifact_paths: ArtifactPaths,
    ground_truth_was_explicit: bool,
) -> None:
    """Reject a Ground Truth benchmark whose declared identity doesn't match
    the selected artifact_paths, before any retrieval happens.

    Only checks when the benchmark metadata declares *both*
    `competition_id` and `season_id` -- e.g. the WC2022 Semantic Ground
    Truth metadata does (see tests/semantic_ground_truth.py). A benchmark
    that doesn't declare an identity isn't constrained here; the chunks
    SHA-256 check in `validate_ground_truth_and_chunks` still catches a
    snapshot mismatch regardless.

    This is what lets a namespaced WC2022 selection (competition_id=43,
    season_id=106 -- the identity the default bundle's own metadata
    declares) fall through to the default WC2022 GroundTruthBundle instead
    of being rejected outright: the check compares declared identities, not
    "is artifact_paths None".
    """
    if "competition_id" not in metadata or "season_id" not in metadata:
        return

    declared = (metadata["competition_id"], metadata["season_id"])
    selected = (artifact_paths.competition_id, artifact_paths.season_id)
    if declared == selected:
        return

    if ground_truth_was_explicit:
        raise RetrievalEvaluationError(
            "Ground Truth benchmark identity mismatch: benchmark declares "
            f"competition_id={declared[0]}, season_id={declared[1]}, but the "
            f"selected artifact_paths is competition_id={selected[0]}, "
            f"season_id={selected[1]}. Refusing to evaluate mismatched "
            "chunks against this benchmark's relevance judgments."
        )

    raise RetrievalEvaluationError(
        "No Ground Truth benchmark was supplied for artifact_paths "
        f"(competition_id={selected[0]}, season_id={selected[1]}), and its "
        "identity does not match the default benchmark "
        f"(competition_id={declared[0]}, season_id={declared[1]}). Refusing "
        "to pair its chunks with the default benchmark -- that would "
        "silently produce meaningless metrics. Supply a GroundTruthBundle "
        "for this dataset."
    )


def validate_ground_truth_and_chunks(
    chunks_path: str = "output/chunks.json",
    ground_truth: Optional[GroundTruthBundle] = None,
) -> Tuple[dict, list, dict]:
    """Validate a Ground Truth benchmark against a chunks snapshot.

    `ground_truth` defaults to the existing WC2022 Semantic Ground Truth
    (tests.semantic_ground_truth) when not supplied -- unchanged legacy
    behavior. Passing a `GroundTruthBundle` validates/scores a different
    benchmark instead.

    Returns (metadata, cases, document_levels) where document_levels maps
    document_id -> level for every chunk.

    Raises RetrievalEvaluationError on any validation failure.
    """
    if ground_truth is None:
        ground_truth = _default_ground_truth_bundle()

    metadata = ground_truth.metadata
    all_cases = ground_truth.cases

    # Validate Ground Truth structure
    errors = ground_truth.validate_fn(metadata, all_cases, Path(chunks_path))
    if errors:
        raise RetrievalEvaluationError(
            f"Ground Truth validation failed with {len(errors)} errors: {errors[:3]}"
        )

    # Validate chunks hash
    with open(chunks_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    stored_hash = metadata["chunks_sha256"]
    if actual_hash != stored_hash:
        raise RetrievalEvaluationError(
            f"Chunks hash mismatch: stored={stored_hash}, actual={actual_hash}"
        )

    # Load chunks and build document_id -> level
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    document_levels: Dict[str, str] = {}
    for chunk in chunks:
        doc_id = chunk.get("document_id", "")
        if not doc_id:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("document_id", "")
        level = chunk.get("level", "")
        if not level:
            meta = chunk.get("metadata", {})
            level = meta.get("level", "")
        if doc_id:
            document_levels[doc_id] = level

    # Validate all required and optional document IDs exist
    for case in all_cases:
        case_id = case["id"]
        for doc_id in case.get("relevant_document_ids", []):
            if "-chunk-" in doc_id:
                raise RetrievalEvaluationError(
                    f"Case {case_id}: required doc_id contains '-chunk-': {doc_id}"
                )
            if doc_id not in document_levels:
                raise RetrievalEvaluationError(
                    f"Case {case_id}: required doc_id not found in chunks: {doc_id}"
                )
        for doc_id in case.get("optional_relevant_document_ids", []):
            if "-chunk-" in doc_id:
                raise RetrievalEvaluationError(
                    f"Case {case_id}: optional doc_id contains '-chunk-': {doc_id}"
                )
            if doc_id not in document_levels:
                raise RetrievalEvaluationError(
                    f"Case {case_id}: optional doc_id not found in chunks: {doc_id}"
                )
        if not case.get("relevant_document_ids"):
            raise RetrievalEvaluationError(
                f"Case {case_id}: no required relevant documents"
            )

    return metadata, all_cases, document_levels


# ---------------------------------------------------------------------------
# Result Validation
# ---------------------------------------------------------------------------


def validate_chunk_result(
    result: Any,
    method: str,
    case_id: str,
    position: int,
) -> None:
    """Validate a single retrieval result dict.

    Raises RetrievalEvaluationError on any structural problem.
    """
    if not isinstance(result, dict):
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: "
            f"result is not a dict, got {type(result).__name__}"
        )

    chunk_id = result.get("chunk_id", "")
    if not chunk_id or not str(chunk_id).strip():
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: missing or blank chunk_id"
        )

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: "
            f"metadata is not a dict, got {type(metadata).__name__}"
        )

    doc_id = metadata.get("document_id", "")
    if not doc_id or not str(doc_id).strip():
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: "
            f"missing or blank document_id"
        )

    if "-chunk-" in str(doc_id):
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: "
            f"document_id contains '-chunk-': {doc_id}"
        )

    level = metadata.get("level", "")
    if not level or not str(level).strip():
        raise RetrievalEvaluationError(
            f"[{method}] case={case_id} position={position}: "
            f"missing or blank level"
        )


# ---------------------------------------------------------------------------
# Document Ranking
# ---------------------------------------------------------------------------


def build_document_ranking(
    chunk_results: List[Dict[str, Any]],
    method: str,
    case_id: str,
    max_documents: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert ranked chunk results into a ranked list of unique documents.

    Uses best-chunk-first-occurrence policy: the first chunk belonging to a
    document determines that document's rank. Later chunks from the same
    document are counted as duplicates but do not affect ranking.

    Returns (ranked_documents, diagnostics).
    """
    seen_documents: Dict[str, int] = {}  # doc_id -> index in ranked_documents
    ranked_documents: List[Dict[str, Any]] = []
    invalid_count = 0
    duplicate_hits = 0

    for raw_pos, result in enumerate(chunk_results):
        # Validate — raises RetrievalEvaluationError on any structural problem
        validate_chunk_result(result, method, case_id, raw_pos)

        metadata = result.get("metadata", {})
        doc_id = metadata["document_id"]

        if doc_id in seen_documents:
            duplicate_hits += 1
            # Count duplicate chunk for the existing document entry
            idx = seen_documents[doc_id]
            ranked_documents[idx]["duplicate_chunk_count"] += 1
            continue

        if max_documents is not None and len(ranked_documents) >= max_documents:
            break

        doc_rank = len(ranked_documents) + 1
        chunk_rank = result.get("rank", raw_pos + 1)

        entry = {
            "document_id": doc_id,
            "document_rank": doc_rank,
            "first_chunk_rank": chunk_rank,
            "first_chunk_id": result.get("chunk_id", ""),
            "level": metadata.get("level", ""),
            "source": result.get("source", ""),
            "score": result.get("score"),
            "rrf_score": result.get("rrf_score"),
            "duplicate_chunk_count": 0,
        }
        seen_documents[doc_id] = len(ranked_documents)
        ranked_documents.append(entry)

    diagnostics = {
        "raw_chunk_result_count": len(chunk_results),
        "unique_document_count": len(ranked_documents),
        "duplicate_document_hit_count": duplicate_hits,
        "invalid_result_count": invalid_count,
    }

    return ranked_documents, diagnostics


# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------


def hit_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    k: int,
) -> float:
    """1.0 if at least one required doc in Top-K, else 0.0."""
    top_k = set(ranked_doc_ids[:k])
    return 1.0 if any(d in top_k for d in required_docs) else 0.0


def recall_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    k: int,
) -> float:
    """Required docs retrieved in Top-K / total required docs."""
    if not required_docs:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    retrieved = sum(1 for d in required_docs if d in top_k)
    return retrieved / len(required_docs)


def precision_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    k: int,
) -> float:
    """Strict precision: required docs in Top-K / K."""
    top_k = set(ranked_doc_ids[:k])
    retrieved = sum(1 for d in required_docs if d in top_k)
    return retrieved / k


def acceptable_precision_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    optional_docs: List[str],
    k: int,
) -> float:
    """(required + optional) docs in Top-K / K."""
    top_k = set(ranked_doc_ids[:k])
    acceptable = set(required_docs) | set(optional_docs)
    retrieved = sum(1 for d in acceptable if d in top_k)
    return retrieved / k


def all_required_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    k: int,
) -> float:
    """1.0 only if every required doc is in Top-K."""
    if not required_docs:
        return 0.0
    top_k = set(ranked_doc_ids[:k])
    return 1.0 if all(d in top_k for d in required_docs) else 0.0


def reciprocal_rank(
    ranked_doc_ids: List[str],
    required_docs: List[str],
) -> float:
    """1/rank of first required document, or 0.0 if none found."""
    required_set = set(required_docs)
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in required_set:
            return 1.0 / (i + 1)
    return 0.0


def first_relevant_document_rank(
    ranked_doc_ids: List[str],
    required_docs: List[str],
) -> Optional[int]:
    """Rank (1-indexed) of first required document, or None."""
    required_set = set(required_docs)
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in required_set:
            return i + 1
    return None


def ndcg_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    k: int,
) -> float:
    """Binary-relevance nDCG@K using required docs only.

    DCG@K = sum(rel_i / log2(i+1)) for i=1..K
    IDCG uses min(|required|, K) ideal positions.
    """
    if not required_docs:
        return 0.0

    required_set = set(required_docs)

    # DCG
    dcg = 0.0
    for i in range(min(k, len(ranked_doc_ids))):
        rel = 1.0 if ranked_doc_ids[i] in required_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1

    # IDCG
    ideal_count = min(len(required_docs), k)
    idcg = 0.0
    for i in range(ideal_count):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def relevant_level_coverage_at_k(
    ranked_doc_ids: List[str],
    required_docs: List[str],
    document_levels: Dict[str, str],
    k: int,
) -> float:
    """Required levels represented by retrieved required docs in Top-K / total required levels."""
    if not required_docs:
        return 0.0

    required_set = set(required_docs)
    required_levels = set()
    for d in required_docs:
        level = document_levels.get(d, "")
        if level:
            required_levels.add(level)

    if not required_levels:
        return 0.0

    top_k = set(ranked_doc_ids[:k])
    retrieved_required = top_k & required_set
    retrieved_levels = set()
    for d in retrieved_required:
        level = document_levels.get(d, "")
        if level:
            retrieved_levels.add(level)

    return len(retrieved_levels) / len(required_levels)


# ---------------------------------------------------------------------------
# Per-Case Evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: Dict[str, Any],
    retrieval_fn: Callable[..., List[Dict[str, Any]]],
    method: str,
    document_levels: Dict[str, str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_chunk_depth: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
    artifact_paths: Optional[ArtifactPaths] = None,
) -> Dict[str, Any]:
    """Evaluate a single case using the given retrieval function.

    Schema 2.0: For BM25 and Dense, retrieval is executed once at
    candidate_chunk_depth and K values are evaluated from prefixes.
    For Hybrid, hybrid_search is executed independently for every K.

    `artifact_paths` selects a namespaced dataset's chunks/BM25/Chroma (see
    src/artifacts.py). When given, it is forwarded to `retrieval_fn` as an
    `artifact_paths` keyword argument. When `None` (the default), the
    keyword is omitted entirely so existing test doubles and legacy
    retrieval function signatures with no `artifact_paths` parameter keep
    working unchanged.

    Returns a dict with runs_by_k containing K-specific evaluations.
    """
    case_id = case["id"]
    query = case["query"]
    required_docs = list(case.get("relevant_document_ids", []))
    optional_docs = list(case.get("optional_relevant_document_ids", []))

    required_levels = sorted(
        set(
            document_levels.get(d, "")
            for d in required_docs
            if document_levels.get(d)
        )
    )

    runs_by_k: Dict[str, Dict[str, Any]] = {}

    if method in ("bm25", "dense"):
        # BM25 / Dense: single retrieval at candidate depth
        if method == "bm25":
            if artifact_paths is not None:
                raw_results = retrieval_fn(
                    query, k=candidate_chunk_depth, artifact_paths=artifact_paths
                )
            else:
                raw_results = retrieval_fn(query, k=candidate_chunk_depth)
        else:
            if artifact_paths is not None:
                raw_results = retrieval_fn(
                    query,
                    k=candidate_chunk_depth,
                    level_filter=None,
                    artifact_paths=artifact_paths,
                )
            else:
                raw_results = retrieval_fn(
                    query, k=candidate_chunk_depth, level_filter=None
                )

        full_ranked_docs, full_diagnostics = build_document_ranking(
            raw_results, method, case_id
        )
        full_ranked_doc_ids = [d["document_id"] for d in full_ranked_docs]

        for k in k_values:
            k_str = str(k)
            k_docs = full_ranked_docs[:k]
            k_doc_ids = full_ranked_doc_ids[:k]
            k_duplicate_hits = sum(
                d.get("duplicate_chunk_count", 0) for d in k_docs
            )

            rr = reciprocal_rank(k_doc_ids, required_docs)
            first_rel = first_relevant_document_rank(k_doc_ids, required_docs)

            retrieved_req = [d for d in required_docs if d in set(k_doc_ids)]
            missing_req = [
                d for d in required_docs if d not in set(k_doc_ids)
            ]
            retrieved_opt = [
                d for d in optional_docs if d in set(k_doc_ids)
            ]

            metrics = {
                "hit_at_k": hit_at_k(k_doc_ids, required_docs, k),
                "recall_at_k": recall_at_k(k_doc_ids, required_docs, k),
                "precision_at_k": precision_at_k(k_doc_ids, required_docs, k),
                "acceptable_precision_at_k": acceptable_precision_at_k(
                    k_doc_ids, required_docs, optional_docs, k
                ),
                "all_required_at_k": all_required_at_k(
                    k_doc_ids, required_docs, k
                ),
                "ndcg_at_k": ndcg_at_k(k_doc_ids, required_docs, k),
                "relevant_level_coverage_at_k": relevant_level_coverage_at_k(
                    k_doc_ids, required_docs, document_levels, k
                ),
                "required_documents_retrieved": sorted(retrieved_req),
                "required_documents_missing": list(missing_req),
                "optional_documents_retrieved": sorted(retrieved_opt),
            }

            runs_by_k[k_str] = {
                "final_k": k,
                "candidate_chunk_depth": candidate_chunk_depth,
                "raw_chunk_result_count": full_diagnostics[
                    "raw_chunk_result_count"
                ],
                "ranked_unique_document_count": len(k_docs),
                "duplicate_document_hit_count": k_duplicate_hits,
                "ranked_document_ids": k_doc_ids,
                "ranked_documents": k_docs,
                "reciprocal_rank_at_k": rr,
                "first_relevant_document_rank_at_k": first_rel,
                "metrics": metrics,
            }

    elif method == "hybrid":
        # Hybrid: independent execution for each K
        for k in k_values:
            k_str = str(k)
            if artifact_paths is not None:
                raw_results = retrieval_fn(
                    query,
                    k=k,
                    bm25_k=candidate_chunk_depth,
                    dense_k=candidate_chunk_depth,
                    level_filter=None,
                    artifact_paths=artifact_paths,
                )
            else:
                raw_results = retrieval_fn(
                    query,
                    k=k,
                    bm25_k=candidate_chunk_depth,
                    dense_k=candidate_chunk_depth,
                    level_filter=None,
                )

            ranked_docs, diagnostics = build_document_ranking(
                raw_results, method, case_id
            )
            ranked_doc_ids = [d["document_id"] for d in ranked_docs]

            rr = reciprocal_rank(ranked_doc_ids, required_docs)
            first_rel = first_relevant_document_rank(
                ranked_doc_ids, required_docs
            )

            retrieved_req = [
                d for d in required_docs if d in set(ranked_doc_ids)
            ]
            missing_req = [
                d for d in required_docs if d not in set(ranked_doc_ids)
            ]
            retrieved_opt = [
                d for d in optional_docs if d in set(ranked_doc_ids)
            ]

            metrics = {
                "hit_at_k": hit_at_k(ranked_doc_ids, required_docs, k),
                "recall_at_k": recall_at_k(ranked_doc_ids, required_docs, k),
                "precision_at_k": precision_at_k(
                    ranked_doc_ids, required_docs, k
                ),
                "acceptable_precision_at_k": acceptable_precision_at_k(
                    ranked_doc_ids, required_docs, optional_docs, k
                ),
                "all_required_at_k": all_required_at_k(
                    ranked_doc_ids, required_docs, k
                ),
                "ndcg_at_k": ndcg_at_k(ranked_doc_ids, required_docs, k),
                "relevant_level_coverage_at_k": relevant_level_coverage_at_k(
                    ranked_doc_ids, required_docs, document_levels, k
                ),
                "required_documents_retrieved": sorted(retrieved_req),
                "required_documents_missing": list(missing_req),
                "optional_documents_retrieved": sorted(retrieved_opt),
            }

            runs_by_k[k_str] = {
                "final_k": k,
                "candidate_chunk_depth": candidate_chunk_depth,
                "raw_chunk_result_count": diagnostics["raw_chunk_result_count"],
                "ranked_unique_document_count": diagnostics[
                    "unique_document_count"
                ],
                "duplicate_document_hit_count": diagnostics[
                    "duplicate_document_hit_count"
                ],
                "ranked_document_ids": ranked_doc_ids,
                "ranked_documents": ranked_docs,
                "reciprocal_rank_at_k": rr,
                "first_relevant_document_rank_at_k": first_rel,
                "metrics": metrics,
            }
    else:
        raise RetrievalEvaluationError(f"Unknown method: {method}")

    return {
        "case_id": case_id,
        "case_group": case.get("case_group", ""),
        "query": query,
        "expected_route": case.get("expected_route", ""),
        "primary_level": case.get("primary_level", ""),
        "acceptable_levels": list(case.get("acceptable_levels", [])),
        "required_relevant_document_ids": required_docs,
        "optional_relevant_document_ids": optional_docs,
        "required_relevant_levels": required_levels,
        "method": method,
        "candidate_chunk_depth": candidate_chunk_depth,
        "runs_by_k": runs_by_k,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_case_results(
    case_results: List[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Macro-average metrics across cases.

    Schema 2.0: rank-based metrics (MRR, first-relevant-rank) are reported
    explicitly per K via mean_reciprocal_rank_at_k and
    mean_first_relevant_document_rank_at_k.
    """
    n = len(case_results)
    if n == 0:
        return {}

    metric_names = [
        "hit_at_k",
        "recall_at_k",
        "precision_at_k",
        "acceptable_precision_at_k",
        "all_required_at_k",
        "ndcg_at_k",
        "relevant_level_coverage_at_k",
    ]

    max_k_str = str(max(k_values))

    overall: Dict[str, Any] = {
        "cases_evaluated": n,
        "cases_with_any_required_document": sum(
            1
            for c in case_results
            if c["runs_by_k"][max_k_str]["reciprocal_rank_at_k"] > 0
        ),
        "mean_unique_documents_returned": round(
            _mean(
                [
                    c["runs_by_k"][max_k_str]["ranked_unique_document_count"]
                    for c in case_results
                ]
            ),
            6,
        ),
        "mean_duplicate_document_hits": round(
            _mean(
                [
                    c["runs_by_k"][max_k_str]["duplicate_document_hit_count"]
                    for c in case_results
                ]
            ),
            6,
        ),
        "metrics_by_k": {},
    }

    for k in k_values:
        k_str = str(k)
        k_metrics: Dict[str, Any] = {}
        for metric_name in metric_names:
            values = [
                c["runs_by_k"][k_str]["metrics"][metric_name]
                for c in case_results
            ]
            k_metrics[metric_name] = round(_mean(values), 6)

        # K-scoped rank metrics
        k_metrics["mean_reciprocal_rank_at_k"] = round(
            _mean(
                [
                    c["runs_by_k"][k_str]["reciprocal_rank_at_k"]
                    for c in case_results
                ]
            ),
            6,
        )
        first_rel_values = [
            c["runs_by_k"][k_str]["first_relevant_document_rank_at_k"]
            for c in case_results
        ]
        non_none_first_rel = [v for v in first_rel_values if v is not None]
        k_metrics["mean_first_relevant_document_rank_at_k"] = (
            round(_mean(non_none_first_rel), 6) if non_none_first_rel else None
        )

        # Exact success
        exact_success = sum(
            1
            for c in case_results
            if c["runs_by_k"][k_str]["metrics"]["all_required_at_k"] == 1.0
        )
        k_metrics["exact_required_success_count"] = exact_success
        k_metrics["exact_required_success_rate"] = (
            round(exact_success / n, 6) if n else 0.0
        )

        overall["metrics_by_k"][k_str] = k_metrics

    return overall


def aggregate_by_group(
    case_results: List[Dict[str, Any]],
    k_values: Sequence[int],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics by case_group."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for cr in case_results:
        group = cr.get("case_group", "unknown")
        groups.setdefault(group, []).append(cr)

    result = {}
    for group_name in sorted(groups.keys()):
        result[group_name] = aggregate_case_results(groups[group_name], k_values)
    return result


# ---------------------------------------------------------------------------
# Retrieval Module Loading
# ---------------------------------------------------------------------------


def load_retrieval_module(project_root: str = ".") -> Any:
    """Load 06_retrieve_context.py as a module without executing CLI."""
    module_path = os.path.join(project_root, "06_retrieve_context.py")
    if not os.path.exists(module_path):
        raise RetrievalEvaluationError(
            f"Retrieval module not found: {module_path}"
        )

    # Ensure project root is on sys.path
    abs_root = os.path.abspath(project_root)
    if abs_root not in sys.path:
        sys.path.insert(0, abs_root)

    import importlib

    # Force reload to get fresh module
    if "06_retrieve_context" in sys.modules:
        del sys.modules["06_retrieve_context"]

    mod = importlib.import_module("06_retrieve_context")
    return mod


def reset_retrieval_caches(retrieval_module: Any) -> None:
    """Reset retrieval module caches."""
    if hasattr(retrieval_module, "_bm25_cache"):
        retrieval_module._bm25_cache = None
    if hasattr(retrieval_module, "_chunks_cache"):
        retrieval_module._chunks_cache = None
    try:
        from src.cache import clear_all_caches

        clear_all_caches()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Temporary Chroma Context
# ---------------------------------------------------------------------------


@contextmanager
def temporary_chroma_copy(
    original_chroma_dir: str,
    retrieval_module: Any,
):
    """Context manager that copies Chroma to a temp dir and patches the module.

    Yields the temporary copy path. On exit, restores original, cleans up,
    and verifies original integrity.
    """
    original_sqlite = os.path.join(original_chroma_dir, "chroma.sqlite3")
    if not os.path.exists(original_sqlite):
        raise RetrievalEvaluationError(
            f"Original Chroma SQLite not found: {original_sqlite}"
        )

    # Hash original
    with open(original_sqlite, "rb") as f:
        original_hash = hashlib.sha256(f.read()).hexdigest()

    tmp_dir = tempfile.mkdtemp(prefix="retrieval_eval_chroma_")
    tmp_chroma = os.path.join(tmp_dir, "chroma_db")

    try:
        shutil.copytree(original_chroma_dir, tmp_chroma)

        # Verify copy
        copy_sqlite = os.path.join(tmp_chroma, "chroma.sqlite3")
        with open(copy_sqlite, "rb") as f:
            copy_hash = hashlib.sha256(f.read()).hexdigest()

        if copy_hash != original_hash:
            raise RetrievalEvaluationError(
                f"Chroma copy hash mismatch: original={original_hash}, copy={copy_hash}"
            )

        # Patch
        original_chroma_attr = getattr(retrieval_module, "CHROMA_DIR", None)
        retrieval_module.CHROMA_DIR = tmp_chroma

        # Clear caches so Chroma re-opens from new path
        reset_retrieval_caches(retrieval_module)

        yield tmp_chroma

    finally:
        # Restore
        if original_chroma_attr is not None:
            retrieval_module.CHROMA_DIR = original_chroma_attr
        reset_retrieval_caches(retrieval_module)

        # Delete temp
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(tmp_dir):
            import stat

            def _remove_readonly(func, path, exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(tmp_dir, onerror=_remove_readonly)

        # Verify original unchanged
        if os.path.exists(original_sqlite):
            with open(original_sqlite, "rb") as f:
                post_hash = hashlib.sha256(f.read()).hexdigest()
            if post_hash != original_hash:
                raise RetrievalEvaluationError(
                    "Original Chroma database was modified during evaluation"
                )


@dataclass(frozen=True)
class TemporaryChromaArtifactPaths:
    """
    Evaluator-only, ArtifactPaths-compatible view used for Dense/Hybrid
    retrieval while inside a `temporary_chroma_copy(...)` context.

    Production `dense_search`/`hybrid_search` (see 06_retrieve_context.py)
    take an `artifact_paths` argument and, when it is not None, read
    `artifact_paths.chroma_dir` / `artifact_paths.chroma_collection_name`
    directly -- ignoring the retrieval module's patched `CHROMA_DIR`
    module attribute entirely. That means forwarding the real, selected
    `ArtifactPaths` during namespaced evaluation would make Dense/Hybrid
    open the *source* Chroma database directly, bypassing the temporary
    copy `temporary_chroma_copy` exists to isolate reads to.

    This view exists to close that gap without touching production
    retrieval code or `src.artifacts.ArtifactPaths` globally: it delegates
    every attribute to `base` (the real, selected dataset) except
    `chroma_dir`, which is overridden to the temporary copy's path.
    `chroma_collection_name` is deliberately NOT overridden -- it is
    derived from competition_id/season_id only (see
    `ArtifactPaths.chroma_collection_name`), so it is identical between
    `base` and the temporary copy, and `temporary_chroma_copy` copies the
    *entire* Chroma directory (all collections, under their real names),
    so the copy still holds a collection under this exact name.

    `bm25_index`/`chunks`/`match_facts` (needed by Hybrid's BM25 half and
    by Hybrid's context-expansion helpers) are intentionally left pointing
    at the real, selected dataset -- `temporary_chroma_copy` only ever
    copies the Chroma directory, never BM25/chunks/match_facts, so those
    must keep resolving to the originals.
    """

    base: ArtifactPaths
    chroma_dir_override: Path

    @property
    def competition_id(self) -> int:
        return self.base.competition_id

    @property
    def season_id(self) -> int:
        return self.base.season_id

    @property
    def output_root(self) -> Path:
        return self.base.output_root

    @property
    def root(self) -> Path:
        return self.base.root

    @property
    def match_facts(self) -> Path:
        return self.base.match_facts

    @property
    def documents(self) -> Path:
        return self.base.documents

    @property
    def processed_documents(self) -> Path:
        return self.base.processed_documents

    @property
    def chunks(self) -> Path:
        return self.base.chunks

    @property
    def indices_dir(self) -> Path:
        return self.base.indices_dir

    @property
    def bm25_index(self) -> Path:
        return self.base.bm25_index

    @property
    def embeddings_dir(self) -> Path:
        return self.base.embeddings_dir

    @property
    def embeddings_file(self) -> Path:
        return self.base.embeddings_file

    @property
    def chroma_dir(self) -> Path:
        return self.chroma_dir_override

    @property
    def chroma_collection_name(self) -> str:
        return self.base.chroma_collection_name


# ---------------------------------------------------------------------------
# Method Evaluation
# ---------------------------------------------------------------------------


def evaluate_retrieval_method(
    method: str,
    cases: List[Dict[str, Any]],
    retrieval_module: Any,
    document_levels: Dict[str, str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_chunk_depth: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
    artifact_paths: Optional[ArtifactPaths] = None,
) -> List[Dict[str, Any]]:
    """Evaluate all cases for a single retrieval method.

    `artifact_paths` selects a namespaced dataset (see src/artifacts.py)
    and is forwarded to `evaluate_case` for every case. Defaults to `None`
    -- unchanged legacy WC2022 behavior.
    """
    if method not in SUPPORTED_METHODS:
        raise RetrievalEvaluationError(
            f"Unsupported method: {method}. Must be one of {SUPPORTED_METHODS}"
        )

    if method == "bm25":
        retrieval_fn = retrieval_module.bm25_search
    elif method == "dense":
        retrieval_fn = retrieval_module.dense_search
    elif method == "hybrid":
        retrieval_fn = retrieval_module.hybrid_search

    results = []
    for case in cases:
        case_result = evaluate_case(
            case=case,
            retrieval_fn=retrieval_fn,
            method=method,
            document_levels=document_levels,
            k_values=k_values,
            candidate_chunk_depth=candidate_chunk_depth,
            artifact_paths=artifact_paths,
        )
        results.append(case_result)

    return results


# ---------------------------------------------------------------------------
# Baseline Runner
# ---------------------------------------------------------------------------


def _validate_baseline_args(
    methods: Sequence[str],
    k_values: Sequence[int],
    candidate_chunk_depth: int,
    cases: List[Dict[str, Any]],
) -> None:
    """Validate baseline arguments."""
    if not methods:
        raise RetrievalEvaluationError("No methods specified")
    for m in methods:
        if m not in SUPPORTED_METHODS:
            raise RetrievalEvaluationError(
                f"Unsupported method: {m}. Must be one of {SUPPORTED_METHODS}"
            )
    if not k_values:
        raise RetrievalEvaluationError("No K values specified")
    for k in k_values:
        if not isinstance(k, int) or k <= 0:
            raise RetrievalEvaluationError(f"K must be a positive integer, got: {k}")
    if len(set(k_values)) != len(k_values):
        raise RetrievalEvaluationError(f"Duplicate K values: {k_values}")
    if candidate_chunk_depth <= 0:
        raise RetrievalEvaluationError(
            f"candidate_chunk_depth must be positive, got: {candidate_chunk_depth}"
        )
    if candidate_chunk_depth < max(k_values):
        raise RetrievalEvaluationError(
            f"candidate_chunk_depth ({candidate_chunk_depth}) must be >= max(k_values) ({max(k_values)})"
        )
    if not cases:
        raise RetrievalEvaluationError("No cases provided")
    for case in cases:
        if not case.get("relevant_document_ids"):
            raise RetrievalEvaluationError(
                f"Case {case.get('id', '?')} has no required relevant documents"
            )


def run_retrieval_baseline(
    methods: Sequence[str] = SUPPORTED_METHODS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_chunk_depth: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
    project_root: str = ".",
    artifact_paths: Optional[ArtifactPaths] = None,
    ground_truth: Optional[GroundTruthBundle] = None,
) -> Dict[str, Any]:
    """Run the full retrieval baseline evaluation.

    `artifact_paths` selects a namespaced dataset's chunks.json and Chroma
    directory (see src/artifacts.py) instead of the legacy
    `output/chunks.json` / `output/chroma_db` flat layout, and is forwarded
    to every retrieval call. Defaults to `None` -- unchanged legacy WC2022
    artifact locations.

    `ground_truth` selects the Ground Truth benchmark to validate/score
    against (see `GroundTruthBundle`). Defaults to `None`, which resolves to
    the default WC2022 Semantic Ground Truth -- unchanged legacy behavior.

    Dataset/benchmark identity is validated before any retrieval happens
    (see `_ensure_ground_truth_matches_artifact_paths`):

    - `artifact_paths=None` -- no identity check; legacy WC2022 evaluation
      is preserved exactly.
    - `artifact_paths` selects the same (competition_id, season_id) the
      *effective* benchmark's metadata declares (e.g. a namespaced WC2022
      selection, competition_id=43/season_id=106, against the default
      WC2022 bundle) -- allowed to proceed. The chunks SHA-256 check in
      `validate_ground_truth_and_chunks` still determines whether that
      namespaced snapshot actually matches the benchmark.
    - `artifact_paths` selects a different identity than the effective
      benchmark declares -- refused with a clear error before evaluation,
      whether the benchmark came from the WC2022 default (no
      `ground_truth` supplied) or from an explicitly mismatched
      `GroundTruthBundle`.

    Returns a JSON-serializable result dict.
    """
    # Sort K values
    k_values = tuple(sorted(k_values))
    methods = tuple(methods)

    if ground_truth is not None:
        resolved_ground_truth = ground_truth
    elif artifact_paths is None:
        resolved_ground_truth = _default_ground_truth_bundle()
    else:
        from tests.ground_truth_registry import resolve_ground_truth_bundle

        resolved_ground_truth = resolve_ground_truth_bundle(
            competition_id=artifact_paths.competition_id,
            season_id=artifact_paths.season_id,
        )

    if artifact_paths is not None:
        _ensure_ground_truth_matches_artifact_paths(
            resolved_ground_truth.metadata,
            artifact_paths,
            ground_truth_was_explicit=ground_truth is not None,
        )

    # Resolve chunks/Chroma locations for the selected dataset
    if artifact_paths is not None:
        chunks_path = str(artifact_paths.chunks)
        chroma_dir = str(artifact_paths.chroma_dir)
    else:
        chunks_path = os.path.join(project_root, "output", "chunks.json")
        chroma_dir = os.path.join(project_root, "output", "chroma_db")

    # Validate Ground Truth and chunks
    metadata, cases, document_levels = validate_ground_truth_and_chunks(
        chunks_path=chunks_path,
        ground_truth=resolved_ground_truth,
    )

    _validate_baseline_args(methods, k_values, candidate_chunk_depth, cases)

    # Load retrieval module
    retrieval_module = load_retrieval_module(project_root)

    # Determine which methods need Chroma
    chroma_methods = [m for m in methods if m in ("dense", "hybrid")]
    non_chroma_methods = [m for m in methods if m not in ("dense", "hybrid")]

    # Original Chroma hash
    chroma_sqlite = os.path.join(chroma_dir, "chroma.sqlite3")
    original_chroma_hash = None
    if os.path.exists(chroma_sqlite):
        with open(chroma_sqlite, "rb") as f:
            original_chroma_hash = hashlib.sha256(f.read()).hexdigest()

    all_method_results: Dict[str, Any] = {}

    def _run_methods(
        method_list: List[str],
        methods_artifact_paths: Optional[ArtifactPaths],
    ) -> Dict[str, Any]:
        results = {}
        for m in method_list:
            case_results = evaluate_retrieval_method(
                method=m,
                cases=cases,
                retrieval_module=retrieval_module,
                document_levels=document_levels,
                k_values=k_values,
                candidate_chunk_depth=candidate_chunk_depth,
                artifact_paths=methods_artifact_paths,
            )
            overall = aggregate_case_results(case_results, k_values)
            by_group = aggregate_by_group(case_results, k_values)
            results[m] = {
                "overall": overall,
                "by_case_group": by_group,
                "cases": case_results,
            }
        return results

    try:
        # BM25-only methods (no Chroma needed) -- the real, selected
        # artifact_paths (no temp copy involved; BM25 never touches Chroma).
        if non_chroma_methods:
            reset_retrieval_caches(retrieval_module)
            all_method_results.update(_run_methods(non_chroma_methods, artifact_paths))
            reset_retrieval_caches(retrieval_module)

        # Chroma methods (use temporary copy). Dense/Hybrid must never open
        # the source Chroma directly -- see TemporaryChromaArtifactPaths.
        if chroma_methods:
            with temporary_chroma_copy(chroma_dir, retrieval_module) as tmp_chroma:
                if artifact_paths is not None:
                    chroma_methods_artifact_paths = TemporaryChromaArtifactPaths(
                        base=artifact_paths,
                        chroma_dir_override=Path(tmp_chroma),
                    )
                else:
                    # Legacy: no artifact_paths object at all -- Dense/Hybrid
                    # rely on the module-level CHROMA_DIR patch that
                    # temporary_chroma_copy already applies.
                    chroma_methods_artifact_paths = None
                all_method_results.update(
                    _run_methods(chroma_methods, chroma_methods_artifact_paths)
                )
    finally:
        reset_retrieval_caches(retrieval_module)

    # Post-run Chroma hash
    post_chroma_hash = None
    if os.path.exists(chroma_sqlite):
        with open(chroma_sqlite, "rb") as f:
            post_chroma_hash = hashlib.sha256(f.read()).hexdigest()

    # Build result
    result = {
        "evaluator_schema_version": RETRIEVAL_EVALUATOR_SCHEMA_VERSION,
        "run_metadata": {
            "dataset_id": metadata.get("dataset_id", ""),
            "ground_truth_schema_version": metadata.get("schema_version", ""),
            "chunks_sha256": metadata.get("chunks_sha256", ""),
            "case_count": len(cases),
            "methods": list(methods),
            "k_values": list(k_values),
            "candidate_chunk_depth": candidate_chunk_depth,
            "relevance_unit": RELEVANCE_UNIT,
            "document_ranking_policy": DOCUMENT_RANKING_POLICY,
            "optional_document_policy": "acceptable_precision_only",
            "offline_embedding_required": True,
            "execution_policy": {
                "bm25": "single_candidate_depth_run_prefix_evaluation",
                "dense": "single_candidate_depth_run_prefix_evaluation",
                "hybrid": "independent_final_k_execution",
                "hybrid_candidate_depth": candidate_chunk_depth,
            },
        },
        "methods": all_method_results,
        "integrity": {
            "ground_truth_validation_errors": [],
            "original_chroma_sha256_before": original_chroma_hash,
            "original_chroma_sha256_after": post_chroma_hash,
            "original_chroma_unchanged": original_chroma_hash == post_chroma_hash,
            "temporary_chroma_used": len(chroma_methods) > 0,
            "temporary_chroma_deleted": True,
        },
    }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football-Analytics-RAG retrieval evaluator"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(SUPPORTED_METHODS),
        choices=SUPPORTED_METHODS,
        help="Retrieval methods to evaluate",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="K values for Top-K metrics",
    )
    parser.add_argument(
        "--candidate-chunk-depth",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK_DEPTH,
        help="Number of raw chunk candidates to retrieve",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print indented JSON",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        help="Competition ID of a namespaced dataset to evaluate "
        "(see src/artifacts.py). Must be given together with --season-id.",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        help="Season ID of a namespaced dataset to evaluate "
        "(see src/artifacts.py). Must be given together with --competition-id.",
    )
    parser.add_argument(
        "--namespaced",
        action="store_true",
        help="Use the namespaced artifact layout for the legacy default "
        "dataset instead of the flat output/ layout",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns exit code."""
    args = parse_args(argv)

    # Validate
    for k in args.k_values:
        if k <= 0:
            print(f"Error: K values must be positive, got {k}", file=sys.stderr)
            return 1
    if args.candidate_chunk_depth <= 0:
        print(
            f"Error: candidate-chunk-depth must be positive, got {args.candidate_chunk_depth}",
            file=sys.stderr,
        )
        return 1
    if args.candidate_chunk_depth < max(args.k_values):
        print(
            f"Error: candidate-chunk-depth ({args.candidate_chunk_depth}) must be >= max(k-values) ({max(args.k_values)})",
            file=sys.stderr,
        )
        return 1

    if (args.competition_id is None) != (args.season_id is None):
        print(
            "Error: --competition-id and --season-id must be provided together",
            file=sys.stderr,
        )
        return 1

    if args.competition_id is None:
        artifact_paths = resolve_runtime_artifact_paths(
            legacy_default=not args.namespaced
        )
    else:
        artifact_paths = resolve_runtime_artifact_paths(
            args.competition_id,
            args.season_id,
            legacy_default=not args.namespaced,
        )

    try:
        result = run_retrieval_baseline(
            methods=tuple(args.methods),
            k_values=tuple(sorted(args.k_values)),
            candidate_chunk_depth=args.candidate_chunk_depth,
            artifact_paths=artifact_paths,
        )
    except RetrievalEvaluationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
