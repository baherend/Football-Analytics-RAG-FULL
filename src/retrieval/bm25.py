"""
src/retrieval/bm25.py -- Lexical (BM25) retrieval.

Migration Step 2 (Retrieval Split): mechanically extracted from
src/retrieval/search.py's Lexical Retrieval section -- no logic changes.
See src/retrieval/search.py for the compatibility re-exports existing
callers keep using, and for _load_bm25_index()/_load_chunks()/_get_tokenizer()
(index loading stays there -- see this module's lazy import below for why).

Does NOT do dense retrieval, fusion, safeguards, or orchestration -- see
src/retrieval/dense.py, fusion.py, safeguards.py, service.py for those.
"""

from __future__ import annotations

from src.artifacts import ArtifactPaths


def bm25_search(query: str, k: int = 20, artifact_paths: ArtifactPaths | None = None) -> list[dict]:
    """
    Lexical retrieval using BM25.

    `artifact_paths` selects a namespaced dataset's bm25.pkl/chunks.json
    (see src/artifacts.py) instead of the module-level legacy defaults.
    Defaults to None -- unchanged legacy WC2022 behavior.

    Returns list of {chunk_id, text, metadata, score, rank}.
    Retrieves more candidates (k=20) for fusion.
    """
    # Deliberately a lazy (call-time, not import-time) import: the
    # underlying index/chunk loaders and their caches must stay defined in
    # search.py -- src/evaluation/retrieval_evaluator.py::reset_retrieval_caches()
    # resets those caches by reassigning `_bm25_cache`/`_chunks_cache`
    # directly on the `src.retrieval.search` module object between
    # benchmark cases. Re-exporting them into this module by name would
    # silently desync from search.py's copy (attribute reassignment
    # doesn't propagate back into the module that actually owns them). A
    # top-level (import-time) import here would also create a circular
    # import with search.py, which imports bm25_search back for re-export.
    from src.retrieval import search as _search

    bm25_path = artifact_paths.bm25_index if artifact_paths is not None else None
    chunks_path = artifact_paths.chunks if artifact_paths is not None else None
    bm25 = _search._load_bm25_index(bm25_path)
    chunks = _search._load_chunks(chunks_path)
    tokenize = _search._get_tokenizer()

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top-k indices sorted by score descending
    top_indices = scores.argsort()[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            chunk = chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "document_id": chunk["document_id"],
                    "level": chunk["level"],
                    "match_id": chunk.get("match_id"),
                    "player_name": chunk.get("player_name"),
                    "team_name": chunk.get("team_name"),
                },
                "score": float(scores[idx]),
                "rank": rank + 1,  # 1-indexed rank
                "source": "bm25",
            })

    return results
