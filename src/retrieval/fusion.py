"""
src/retrieval/fusion.py -- Reciprocal Rank Fusion and re-ranking.

Migration Step 2 (Retrieval Split): mechanically extracted from
src/retrieval/search.py's Fusion and Re-ranking sections -- no logic
changes. See src/retrieval/search.py for the compatibility re-exports
existing callers keep using.

Does NOT do BM25, dense retrieval, safeguards, or orchestration -- see
src/retrieval/bm25.py, dense.py, safeguards.py, service.py for those.
"""

from __future__ import annotations

# RRF constant (standard from the original RRF paper)
RRF_K = 60


def reciprocal_rank_fusion(
    result_sets: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple result sets using Reciprocal Rank Fusion (RRF).

    RRF_score(d) = Σ 1/(k + rank_i(d))

    Parameters:
        result_sets: List of result lists, each with {chunk_id, rank, ...}
        k: RRF constant (default 60, standard from the paper)

    Returns:
        Merged results sorted by RRF score descending.
    """
    # Collect all unique chunk_ids and their RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for result_set in result_sets:
        for result in result_set:
            chunk_id = result["chunk_id"]
            rank = result["rank"]

            # Accumulate RRF score
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
            rrf_scores[chunk_id] += 1.0 / (k + rank)

            # Store chunk data (first occurrence wins)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

    # Build merged results
    merged = []
    for chunk_id, rrf_score in rrf_scores.items():
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = rrf_score
        result["appears_in"] = [
            rs["source"] for r_set in result_sets
            for rs in r_set if rs["chunk_id"] == chunk_id
        ]
        merged.append(result)

    # Sort by RRF score descending
    merged.sort(key=lambda x: x["rrf_score"], reverse=True)

    return merged


def rerank(results: list[dict], query: str) -> list[dict]:
    """
    Re-rank results after fusion.

    Currently a pass-through — the RRF score is the final ranking.
    Could be extended with a cross-encoder reranker in the future.
    """
    # For now, RRF score is the final ranking
    # Future: add cross-encoder reranking here
    return results
