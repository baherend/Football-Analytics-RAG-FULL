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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def validate_ground_truth_and_chunks(
    chunks_path: str = "output/chunks.json",
) -> Tuple[dict, list, dict]:
    """Validate Ground Truth and chunks snapshot.

    Returns (metadata, cases, document_levels) where document_levels maps
    document_id -> level for every chunk.

    Raises RetrievalEvaluationError on any validation failure.
    """
    from tests.semantic_ground_truth import (
        SEMANTIC_GROUND_TRUTH,
        SEMANTIC_GROUND_TRUTH_METADATA,
        SEMANTIC_GROUND_TRUTH_SCHEMA_VERSION,
        validate_semantic_ground_truth,
    )

    # Validate Ground Truth structure
    errors = validate_semantic_ground_truth(
        SEMANTIC_GROUND_TRUTH_METADATA, SEMANTIC_GROUND_TRUTH, Path(chunks_path)
    )
    if errors:
        raise RetrievalEvaluationError(
            f"Ground Truth validation failed with {len(errors)} errors: {errors[:3]}"
        )

    # Validate chunks hash
    with open(chunks_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    stored_hash = SEMANTIC_GROUND_TRUTH_METADATA["chunks_sha256"]
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
    all_cases = SEMANTIC_GROUND_TRUTH
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

    return SEMANTIC_GROUND_TRUTH_METADATA, all_cases, document_levels


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
) -> Dict[str, Any]:
    """Evaluate a single case using the given retrieval function.

    Schema 2.0: For BM25 and Dense, retrieval is executed once at
    candidate_chunk_depth and K values are evaluated from prefixes.
    For Hybrid, hybrid_search is executed independently for every K.

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
            raw_results = retrieval_fn(query, k=candidate_chunk_depth)
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
) -> List[Dict[str, Any]]:
    """Evaluate all cases for a single retrieval method."""
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
) -> Dict[str, Any]:
    """Run the full retrieval baseline evaluation.

    Returns a JSON-serializable result dict.
    """
    # Sort K values
    k_values = tuple(sorted(k_values))
    methods = tuple(methods)

    # Validate Ground Truth and chunks
    metadata, cases, document_levels = validate_ground_truth_and_chunks(
        chunks_path=os.path.join(project_root, "output", "chunks.json")
    )

    _validate_baseline_args(methods, k_values, candidate_chunk_depth, cases)

    # Load retrieval module
    retrieval_module = load_retrieval_module(project_root)

    # Determine which methods need Chroma
    chroma_methods = [m for m in methods if m in ("dense", "hybrid")]
    non_chroma_methods = [m for m in methods if m not in ("dense", "hybrid")]

    # Original Chroma hash
    chroma_dir = os.path.join(project_root, "output", "chroma_db")
    chroma_sqlite = os.path.join(chroma_dir, "chroma.sqlite3")
    original_chroma_hash = None
    if os.path.exists(chroma_sqlite):
        with open(chroma_sqlite, "rb") as f:
            original_chroma_hash = hashlib.sha256(f.read()).hexdigest()

    all_method_results: Dict[str, Any] = {}

    def _run_methods(
        method_list: List[str],
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
        # BM25-only methods (no Chroma needed)
        if non_chroma_methods:
            reset_retrieval_caches(retrieval_module)
            all_method_results.update(_run_methods(non_chroma_methods))
            reset_retrieval_caches(retrieval_module)

        # Chroma methods (use temporary copy)
        if chroma_methods:
            with temporary_chroma_copy(chroma_dir, retrieval_module):
                all_method_results.update(_run_methods(chroma_methods))
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

    try:
        result = run_retrieval_baseline(
            methods=tuple(args.methods),
            k_values=tuple(sorted(args.k_values)),
            candidate_chunk_depth=args.candidate_chunk_depth,
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
