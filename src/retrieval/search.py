"""
src/retrieval/search.py -- Retrieval compatibility layer + index loading +
orchestration + context building.

Migration Step 2 (Retrieval Split): this module used to contain BM25, dense
retrieval, RRF fusion, and safeguards directly, alongside orchestration and
context building. The mechanics are now split into focused sibling modules:

    src/retrieval/bm25.py        -- lexical (BM25) retrieval
    src/retrieval/dense.py       -- dense (vector) retrieval via ChromaDB
    src/retrieval/fusion.py      -- Reciprocal Rank Fusion + re-ranking
    src/retrieval/safeguards.py  -- Arabic/comparison/team-style/match safeguards

This module re-exports their public (and test-facing underscore-prefixed)
symbols so existing callers (src/query/router.py, tests/) keep working
unchanged -- see PROJECT_MEMORY.md's Architecture Decisions for why this
compatibility layer exists instead of updating every caller in the same
phase.

Three things intentionally stay defined HERE rather than moving out:

1. Index loading (_load_bm25_index/_load_chunks/_get_tokenizer and their
   caches). src/evaluation/retrieval_evaluator.py::reset_retrieval_caches() resets
   these caches by reassigning `_bm25_cache`/`_chunks_cache` directly on
   this module object between benchmark cases, without a fresh module
   reimport. Moving the caches to another module and re-exporting them by
   name here would silently desync the two copies (Python attribute
   reassignment doesn't propagate back into the module that actually owns
   the name) -- bm25.py and safeguards.py reach these via a lazy,
   documented import back into this module instead.
2. hybrid_search() orchestration. A first pass moved this into a separate
   service.py, calling bm25_search/dense_search/the safeguard functions via
   a lazy import back into this module (the same technique used for index
   loading) so that tests monkeypatching e.g. `search.dense_search` would
   still be observed. On review that was rejected: it made the *new*
   orchestration module permanently, structurally dependent on the facade
   it was meant to replace, for every call, not just during a transition --
   more reverse-coupling surface than the loaders alone, for a module that
   is supposed to be this migration's clean centerpiece. Keeping
   hybrid_search() here instead needs no lazy-import trick at all: Python
   functions resolve global names via their own module's namespace, so a
   function defined in *this* file naturally sees `search.<name>`
   monkeypatches on the names this file already imports for re-export --
   exactly how it worked before any of this module was split. See
   PROJECT_MEMORY.md's Architecture Decisions for the full comparison
   (dependency direction, monkeypatch compatibility, coupling debt). Moves
   to a real service.py once a future phase updates the test-harness
   contract to target the mechanics modules directly, removing the need for
   any module to stand in as a monkeypatch target.
3. Context building (build_context, retrieve_context). Per
   docs/architecture/overview.md, context engineering (dedup, coverage,
   budgeting, evidence-pack construction) is a distinct future
   responsibility (Migration Step 4) that must NOT be folded into
   retrieval just because it currently sits in this file.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from src.artifacts import ArtifactPaths

# ---------------------------------------------------------------------------
# Compatibility re-exports -- see module docstring. Existing callers
# (src/query/router.py, tests/) import these names from this module.
# ---------------------------------------------------------------------------

from src.retrieval.bm25 import bm25_search
from src.retrieval.dense import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    dense_search,
    semantic_search,
)
from src.retrieval.fusion import RRF_K, reciprocal_rank_fusion, rerank
from src.retrieval.safeguards import (
    _AR_BETTER_WORD,
    _AR_LATIN_ENTITY,
    _LATIN_ENTITY_SPAN,
    _MATCH_QUERY_PATTERNS,
    _STAGE_KEYWORDS,
    _STYLE_KEYWORDS,
    _STYLE_KEYWORDS_AR,
    _detect_comparison_entities,
    _detect_match_query,
    _detect_team_style_entities,
    _detect_team_style_query,
    _ensure_comparison_entities,
    _ensure_match_summary,
    _ensure_team_style_doc,
    _expand_query_entity_siblings,
    _extract_latin_entity_spans,
    _find_l4_document,
    _normalize_arabic_for_matching,
)
# Context Engineering (Migration Step 4): evidence selection and context
# rendering moved to src/context/. Two edges point that way from here:
#
#   * select_relevant_chunks -- called by hybrid_search()'s step 9. This is a
#     retrieval -> context inversion (context engineering should consume
#     retrieval output, not be called from inside it). Left in place
#     deliberately: hybrid_search() is the single most benchmarked function in
#     the system, and moving step 9 out to the orchestrator changes what it
#     returns. Recorded as OPEN debt in PROJECT_MEMORY.md, not hidden.
#   * build_context -- re-exported for existing callers and used by
#     retrieve_context() below.
from src.context.rendering import build_context
from src.context.selection import select_relevant_chunks

__all__ = [
    "bm25_search",
    "dense_search",
    "semantic_search",
    "reciprocal_rank_fusion",
    "rerank",
    "hybrid_search",
    "build_context",
    "retrieve_context",
]


# ---------------------------------------------------------------------------
# Index Loading (LAB 8 — Step 1)
# ---------------------------------------------------------------------------

INDICES_DIR = Path("output/indices")
CHUNKS_PATH = Path("output/chunks.json")

# Cache loaded indices to avoid reloading on every query, keyed by resolved
# path so different dataset namespaces (see src/artifacts.py) never share a
# cached index/chunk list within the same process.
_bm25_cache: dict[Path, object] = {}
_chunks_cache: dict[Path, list[dict]] = {}


def _load_bm25_index(path: Path | None = None):
    """Load BM25 index from disk (cached per resolved path)."""
    bm25_path = path if path is not None else (INDICES_DIR / "bm25.pkl")
    resolved = bm25_path.resolve()

    if resolved in _bm25_cache:
        return _bm25_cache[resolved]

    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. "
            "Run 04_vector_representation.py first."
        )

    with open(bm25_path, "rb") as f:
        _bm25_cache[resolved] = pickle.load(f)

    return _bm25_cache[resolved]


def _load_chunks(path: Path | None = None) -> list[dict]:
    """Load chunks from disk (cached per resolved path)."""
    chunks_path = path if path is not None else CHUNKS_PATH
    resolved = chunks_path.resolve()

    if resolved in _chunks_cache:
        return _chunks_cache[resolved]

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found at {chunks_path}. "
            "Run 03_chunking.py first."
        )

    with open(chunks_path, encoding="utf-8") as f:
        _chunks_cache[resolved] = json.load(f)

    return _chunks_cache[resolved]


def _get_tokenizer():
    """Get the BM25 tokenizer."""
    import re

    def simple_tokenize(text: str) -> list[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    return simple_tokenize


# ---------------------------------------------------------------------------
# Hybrid Search — Orchestrator (LAB 8 — Step 6)
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    k: int = 5,
    bm25_k: int = 20,
    dense_k: int = 20,
    level_filter: str | None = None,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Complete hybrid retrieval pipeline.

    1. BM25 retrieval (lexical)
    2. Dense retrieval (semantic)
    3. Merge via Reciprocal Rank Fusion
    4. Re-rank
    5. Ensure comparison entities' L4 docs are included
    6. Return Top-K

    Parameters:
        query: User query
        k: Final number of results
        bm25_k: Number of BM25 candidates to retrieve
        dense_k: Number of dense candidates to retrieve
        level_filter: Optional filter by document level
        artifact_paths: selects a namespaced dataset's BM25 index,
            chunks.json, and Chroma directory (see src/artifacts.py)
            instead of the legacy WC2022 defaults. Defaults to None --
            unchanged legacy behavior. Threaded through every helper below
            (including the sibling-expansion safeguard) so no step
            silently falls back to another dataset's artifacts.

    Coordinates bm25_search/dense_search/reciprocal_rank_fusion/rerank/the
    safeguard functions -- implements no retrieval algorithm itself, see
    bm25.py/dense.py/fusion.py/safeguards.py for the mechanics. Calls them
    as bare names (not via an imported alias) so that tests monkeypatching
    `search.<name>` directly -- see e.g.
    tests/test_artifact_paths.py::test_default_runtime_calls_remain_legacy_compatible
    and tests/test_chunk_selector.py -- are observed here exactly as they
    were before this module was split; see module docstring point 2.

    Returns:
        Top-K results with RRF scores.
    """
    # Step 1: BM25 retrieval
    bm25_results = bm25_search(query, k=bm25_k, artifact_paths=artifact_paths)

    # Step 2: Dense retrieval
    dense_results = dense_search(query, k=dense_k, level_filter=level_filter,
                                 artifact_paths=artifact_paths)

    # Step 3: Merge via RRF
    merged = reciprocal_rank_fusion([bm25_results, dense_results])

    # Step 4: Re-rank
    reranked = rerank(merged, query)

    # Step 5: Ensure comparison entities' L4 docs are included
    reranked = _ensure_comparison_entities(query, reranked, k, artifact_paths=artifact_paths)

    # Step 6: Ensure team-level doc for style queries
    reranked = _ensure_team_style_doc(query, reranked, k, artifact_paths=artifact_paths)

    # Step 7: Ensure Level-1 match summary for match-level queries
    reranked = _ensure_match_summary(query, reranked, k, artifact_paths=artifact_paths)

    # Step 8: Expand siblings for query-matched entity documents
    expanded = _expand_query_entity_siblings(query, reranked, artifact_paths=artifact_paths)

    # Step 9: Select chunks with answer-bearing query-facet coverage
    selected = select_relevant_chunks(query, expanded, max_chunks=k)

    # Step 10: Return selected Top-K
    return selected


# ---------------------------------------------------------------------------
# Convenience (LAB 8 — Step 7)
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    k: int = 5,
    max_length: int = 3000,
    level_filter: str | None = None,
    mode: str = "hybrid",
    artifact_paths: ArtifactPaths | None = None,
) -> dict:
    """
    Retrieve context for a query.

    Parameters:
        query: User query
        k: Number of results
        max_length: Max context length
        level_filter: Optional filter by document level
        mode: "hybrid" (default) or "semantic" (dense-only)
        artifact_paths: selects a namespaced dataset's BM25/chunks/Chroma
            artifacts (see src/artifacts.py) instead of the legacy WC2022
            defaults. Defaults to None -- unchanged legacy behavior.

    Returns:
        {query, context, chunks, num_chunks, mode}
    """
    if mode == "hybrid":
        chunks = hybrid_search(query, k=k, level_filter=level_filter, artifact_paths=artifact_paths)
    else:
        if artifact_paths is not None:
            persist_dir = artifact_paths.chroma_dir
            collection_name = artifact_paths.chroma_collection_name
        else:
            persist_dir = CHROMA_DIR
            collection_name = COLLECTION_NAME
        chunks = semantic_search(
            query,
            k=k,
            level_filter=level_filter,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )

    context = build_context(chunks, max_length)

    return {
        "query": query,
        "context": context,
        "chunks": chunks,
        "num_chunks": len(chunks),
        "mode": mode,
    }
