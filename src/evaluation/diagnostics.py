"""
multilingual_diagnostics.py -- Root-cause diagnostic instrumentation for
Arabic retrieval degradation.

This module NEVER modifies production code (src/retrieval/search.py,
src/query/router.py, src/query/resolver.py, 07_prompting.py). It only
CALLS existing production functions in different combinations, or builds
fully separate temporary/isolated artifacts (a raw BM25+RRF composition
using the same building blocks hybrid_search() itself uses; temporary,
throwaway Chroma collections for candidate embedding models), to isolate
which retrieval stage is responsible for Arabic degradation.

Reused unmodified from src.evaluation.retrieval_evaluator: evaluate_case(),
GroundTruthBundle, all metric functions, temporary_chroma_copy() and its
ArtifactPaths views, reset_retrieval_caches(). Reused unmodified from
src.evaluation.ground_truth.multilingual: MultilingualQueryVariant,
build_translated_cases(), MULTILINGUAL_QUERY_VARIANTS,
ENTITY_SCRIPT_DIAGNOSTIC_VARIANTS.

Temporary artifacts this module can create (all cleaned up / never touch
production paths):
- A raw BM25+Dense+RRF composition -- pure in-memory function composition,
  no filesystem artifact.
- Temporary Chroma collections for candidate embedding models, always
  under a fresh tempfile.mkdtemp() directory, deleted afterward.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.evaluation.retrieval_evaluator import (
    DEFAULT_CANDIDATE_CHUNK_DEPTH,
    DEFAULT_K_VALUES,
    aggregate_by_group,
    aggregate_case_results,
    evaluate_case,
    reset_retrieval_caches,
)
from src.evaluation.ground_truth.semantic import SEMANTIC_GROUND_TRUTH


# ---------------------------------------------------------------------------
# Phase 3.C -- raw BM25 + Dense + RRF, before hybrid_search()'s
# post-retrieval safeguards (_ensure_comparison_entities,
# _ensure_team_style_doc, _ensure_match_summary,
# _expand_query_entity_siblings, select_relevant_chunks).
# ---------------------------------------------------------------------------


def raw_rrf_search(
    query: str,
    k: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
    level_filter: Optional[str] = None,
    artifact_paths: Any = None,
) -> List[Dict[str, Any]]:
    """
    BM25 + Dense + Reciprocal Rank Fusion + rerank() -- i.e. exactly
    hybrid_search()'s pipeline UP TO AND INCLUDING its "Step 4: Re-rank"
    (a no-op today), but WITHOUT any of the entity/team-style/match-summary
    safeguards or select_relevant_chunks() that follow in production
    hybrid_search(). Composed entirely from src.retrieval.search's own
    unmodified public functions -- this file does not reimplement BM25,
    Dense, or RRF.

    Signature matches src.retrieval.search.dense_search()'s calling
    convention (query, k, level_filter, artifact_paths) so it can be
    dropped directly into evaluate_case(..., method="dense") -- a single
    execution at candidate depth, evaluated via K-prefix slicing, exactly
    like the real bm25/dense methods. `level_filter` is accepted for
    signature compatibility but unused, matching dense_search().
    """
    from src.retrieval.search import bm25_search, dense_search, reciprocal_rank_fusion, rerank

    if artifact_paths is not None:
        bm25_results = bm25_search(query, k=k, artifact_paths=artifact_paths)
        dense_results = dense_search(query, k=k, level_filter=None, artifact_paths=artifact_paths)
    else:
        bm25_results = bm25_search(query, k=k)
        dense_results = dense_search(query, k=k, level_filter=None)

    merged = reciprocal_rank_fusion([bm25_results, dense_results])
    return rerank(merged, query)


def evaluate_raw_rrf_method(
    cases: List[Dict[str, Any]],
    document_levels: Dict[str, str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_chunk_depth: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
    artifact_paths: Any = None,
) -> List[Dict[str, Any]]:
    """
    Same shape as src.evaluation.retrieval_evaluator.evaluate_retrieval_method(),
    for the raw_rrf_search diagnostic -- not one of the evaluator's
    SUPPORTED_METHODS, so this thin wrapper (not a evaluator modification)
    is the smallest way to reuse evaluate_case() for it.
    """
    results = []
    for case in cases:
        case_result = evaluate_case(
            case=case,
            retrieval_fn=raw_rrf_search,
            method="dense",  # reuses dense's single-shot + K-prefix convention
            document_levels=document_levels,
            k_values=k_values,
            candidate_chunk_depth=candidate_chunk_depth,
            artifact_paths=artifact_paths,
        )
        case_result["method"] = "raw_rrf"
        results.append(case_result)
    return results


# ---------------------------------------------------------------------------
# Phase 4/5 -- entity-script and language-vs-entity minimal-pair builders.
# ---------------------------------------------------------------------------

# Canonical Latin <-> Arabic-transliterated forms used across Phase 4/5
# construction below. Diagnostic data only -- never used by production
# query handling.
ENTITY_TRANSLITERATIONS: Dict[str, str] = {
    "Messi": "ميسي",
    "Mbappé": "مبابي",
    "Mbappe": "مبابي",
    "Griezmann": "جريزمان",
    "Antoine Griezmann": "أنطوان جريزمان",
    "Enzo Fernández": "إنزو فرنانديز",
    "Argentina": "الأرجنتين",
    "France": "فرنسا",
    "Croatia": "كرواتيا",
    "Morocco": "المغرب",
    "Portugal": "البرتغال",
    "England": "إنجلترا",
    "Germany": "ألمانيا",
    "Poland": "بولندا",
    "Iran": "إيران",
}


def build_case_lookup() -> Dict[str, Dict[str, Any]]:
    return {c["id"]: c for c in SEMANTIC_GROUND_TRUTH}


def make_translated_case(source_case_id: str, query: str) -> Dict[str, Any]:
    """One-off translated case dict: English relevance truth + a substituted query."""
    source = build_case_lookup()[source_case_id]
    new_case = dict(source)
    new_case["query"] = query
    return new_case


# ---------------------------------------------------------------------------
# Phase 6 -- isolated temporary Chroma collection for a candidate embedding
# model. Always under tempfile.mkdtemp(); never touches output/chroma_db
# or output/competitions/.
# ---------------------------------------------------------------------------


class TempModelIndex:
    """
    Context manager: builds a throwaway Chroma collection embedding
    `chunks` with `model_name`, under a fresh temp directory. On exit,
    deletes the temp directory entirely. Mirrors
    05_create_chroma_store.py::create_vector_store()'s embedding/insert
    logic (same libraries, same batching), parameterized by model_name --
    that production function hardcodes MODEL_NAME as a module constant,
    so it cannot be reused directly without either mutating shared module
    state or this explicit, self-contained diagnostic copy.
    """

    def __init__(self, chunks: List[Dict[str, Any]], model_name: str, collection_name: str = "diagnostic"):
        self.chunks = chunks
        self.model_name = model_name
        self.collection_name = collection_name
        self._tmp_dir: Optional[str] = None
        self.model = None
        self.collection = None

    def __enter__(self) -> "TempModelIndex":
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._tmp_dir = tempfile.mkdtemp(prefix="ml_diag_chroma_")
        print(f"  Loading candidate model: {self.model_name}", flush=True)
        self.model = SentenceTransformer(self.model_name)

        client = chromadb.PersistentClient(path=self._tmp_dir)
        self.collection = client.create_collection(name=self.collection_name)

        texts = [c.get("search_text", c["text"]) for c in self.chunks]
        ids = [c["chunk_id"] for c in self.chunks]
        metadatas = []
        for c in self.chunks:
            meta = {"document_id": c["document_id"], "level": c.get("level", "unknown")}
            if c.get("match_id"):
                meta["match_id"] = c["match_id"]
            if c.get("player_name"):
                meta["player_name"] = c["player_name"]
            if c.get("team_name"):
                meta["team_name"] = c["team_name"]
            metadatas.append(meta)

        print(f"  Embedding {len(texts)} chunks with {self.model_name}...", flush=True)
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        return self

    def dense_search(self, query: str, k: int = 20, level_filter: Optional[str] = None,
                      artifact_paths: Any = None) -> List[Dict[str, Any]]:
        """Mirrors src.retrieval.search.dense_search()'s query-time logic exactly, against this temp collection/model."""
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        where = {"level": level_filter} if level_filter else None
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=k, where=where,
            include=["documents", "metadatas", "distances"],
        )
        formatted = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            score = max(0, 1 - distance)
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score,
                "rank": i + 1,
                "source": "dense",
            })
        return formatted

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del self.model
        del self.collection
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


def evaluate_model_dense_method(
    model_index: "TempModelIndex",
    cases: List[Dict[str, Any]],
    document_levels: Dict[str, str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    candidate_chunk_depth: int = DEFAULT_CANDIDATE_CHUNK_DEPTH,
) -> List[Dict[str, Any]]:
    """Evaluate all `cases` against a TempModelIndex's Dense-only search, reusing evaluate_case() unmodified."""
    results = []
    for case in cases:
        case_result = evaluate_case(
            case=case,
            retrieval_fn=model_index.dense_search,
            method="dense",
            document_levels=document_levels,
            k_values=k_values,
            candidate_chunk_depth=candidate_chunk_depth,
            artifact_paths=None,
        )
        results.append(case_result)
    return results
