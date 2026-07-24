"""
07_retrieve_context.py — Phase 4: Hybrid Retrieval Pipeline

Combines lexical (BM25) and semantic (dense) retrieval using
Reciprocal Rank Fusion (RRF).

Pipeline:
    User Query → BM25 Search → Dense Search → Merge → RRF → Top-K → Context

Input: user query
Output: retrieved context

Backward compatibility:
    - semantic_search() remains unchanged (dense-only)
    - hybrid_search() is the new default
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICES_DIR = Path("output/indices")
CHUNKS_PATH = Path("output/chunks.json")
CHROMA_DIR = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RRF constant (standard from the original RRF paper)
RRF_K = 60


# ---------------------------------------------------------------------------
# Index Loading (LAB 8 — Step 1)
# ---------------------------------------------------------------------------

# Cache loaded indices to avoid reloading on every query
_bm25_cache = None
_chunks_cache = None


def _load_bm25_index():
    """Load BM25 index from disk (cached)."""
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    bm25_path = INDICES_DIR / "bm25.pkl"
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. "
            "Run 04_representation.py first."
        )

    with open(bm25_path, "rb") as f:
        _bm25_cache = pickle.load(f)

    return _bm25_cache


def _load_chunks() -> list[dict]:
    """Load chunks from disk (cached)."""
    global _chunks_cache
    if _chunks_cache is not None:
        return _chunks_cache

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks not found at {CHUNKS_PATH}. "
            "Run 03_chunking.py first."
        )

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        _chunks_cache = json.load(f)

    return _chunks_cache


def _get_tokenizer():
    """Get the BM25 tokenizer (same as 04_representation.py)."""
    import re

    def simple_tokenize(text: str) -> list[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    return simple_tokenize


# ---------------------------------------------------------------------------
# Lexical Retrieval — BM25 (LAB 8 — Step 2)
# ---------------------------------------------------------------------------


def bm25_search(query: str, k: int = 20) -> list[dict]:
    """
    Lexical retrieval using BM25.

    Returns list of {chunk_id, text, metadata, score, rank}.
    Retrieves more candidates (k=20) for fusion.
    """
    bm25 = _load_bm25_index()
    chunks = _load_chunks()
    tokenize = _get_tokenizer()

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


# ---------------------------------------------------------------------------
# Dense Retrieval — ChromaDB (LAB 8 — Step 3)
# ---------------------------------------------------------------------------


def dense_search(query: str, k: int = 20,
                 level_filter: str | None = None) -> list[dict]:
    """
    Dense retrieval using ChromaDB embeddings.

    Returns list of {chunk_id, text, metadata, score, rank}.
    Retrieves more candidates (k=20) for fusion.
    """
    from chromadb import PersistentClient
    from src.cache import get_embedding_model

    # Use cached model (loaded once)
    model = get_embedding_model(EMBEDDING_MODEL)
    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    # Generate query embedding
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    # Build where filter
    where = None
    if level_filter:
        where = {"level": level_filter}

    # Search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        score = max(0, 1 - distance)  # Convert distance to similarity score

        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score,
            "rank": i + 1,  # 1-indexed rank
            "source": "dense",
        })

    return formatted


# ---------------------------------------------------------------------------
# Fusion — Reciprocal Rank Fusion (LAB 8 — Step 4)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Re-ranking (LAB 8 — Step 5)
# ---------------------------------------------------------------------------


def rerank(results: list[dict], query: str) -> list[dict]:
    """
    Re-rank results after fusion.

    Currently a pass-through — the RRF score is the final ranking.
    Could be extended with a cross-encoder reranker in the future.
    """
    # For now, RRF score is the final ranking
    # Future: add cross-encoder reranking here
    return results


# ---------------------------------------------------------------------------
# Comparison Entity Detection
# ---------------------------------------------------------------------------


def _detect_comparison_entities(query: str) -> list[str]:
    """
    Detect if a query is comparing two entities (players/teams).

    Returns list of entity names if comparison detected, empty list otherwise.
    """
    import re

    query_lower = query.lower().strip()

    # Pattern: "Compare X and Y"
    match = re.search(r"compare\s+(.+?)\s+and\s+(.+?)(?:\s|'s|$|\?)", query_lower)
    if match:
        return [match.group(1).strip().rstrip("'s"), match.group(2).strip().rstrip("'s")]

    # Pattern: "X vs Y"
    match = re.search(r"(\w+)\s+vs\.?\s+(\w+)", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Pattern: "Who performed better ... X or Y"
    match = re.search(r"who\s+(?:performed|played|did)\s+better.*?(\w+)\s+or\s+(\w+)", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    # Pattern: "X or Y" (simple)
    match = re.search(r"(\w+)\s+or\s+(\w+)(?:\s|$|\?)", query_lower)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]

    return []


def _find_l4_document(player_name: str, chunks: list[dict]) -> dict | None:
    """Find a player's L4 tournament summary document in chunks."""
    player_lower = player_name.lower()

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("level") == "4":
            chunk_player = chunk.get("player_name", "").lower()
            if player_lower in chunk_player or player_lower in chunk.get("text", "").lower()[:100]:
                return chunk

    return None


def _ensure_comparison_entities(
    query: str,
    results: list[dict],
    k: int,
) -> list[dict]:
    """
    Ensure that when a query compares two entities, both entities' L4
    tournament summary documents are included in the final top-k results.

    Strategy: check only the top-k (not all candidates), and if an entity's
    L4 doc is missing from top-k, find it in the chunk store and prepend it.
    """
    entities = _detect_comparison_entities(query)
    if len(entities) < 2:
        return results

    # Load chunks for L4 lookup
    chunks = _load_chunks()

    # Check current top-k for existing L4 docs
    top_k = results[:k]
    additions = []

    for entity in entities:
        entity_lower = entity.lower()

        # Check if entity's L4 doc is already in top-k
        has_l4_in_topk = any(
            r.get("metadata", {}).get("level") == "4" and
            entity_lower in r.get("metadata", {}).get("player_name", "").lower()
            for r in top_k
        )

        if not has_l4_in_topk:
            # Find entity's L4 chunk-0 (the main summary) in all chunks
            for chunk in chunks:
                if chunk.get("level") == "4" and chunk["chunk_id"].endswith("-chunk-0"):
                    player_name = (chunk.get("player_name") or
                                   chunk.get("metadata", {}).get("player_name", "")).lower()
                    if entity_lower in player_name:
                        additions.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "metadata": {
                                "document_id": chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"),
                                "level": "4",
                                "player_name": chunk.get("player_name") or chunk.get("metadata", {}).get("player_name"),
                                "team_name": chunk.get("team_name") or chunk.get("metadata", {}).get("team_name"),
                            },
                            "score": 0.01,
                            "rrf_score": 0.01,
                            "source": "comparison_boost",
                        })
                        break

    # Prepend additions so they appear in top-k
    if additions:
        results = additions + results

    # Prepend additions and deduplicate
    if additions:
        seen_ids = set()
        deduped = []
        for r in additions + results:
            if r["chunk_id"] not in seen_ids:
                deduped.append(r)
                seen_ids.add(r["chunk_id"])
        return deduped

    return results


# ---------------------------------------------------------------------------
# Team Style Query Detection
# ---------------------------------------------------------------------------

_STYLE_KEYWORDS = {"style", "formation", "play pattern", "tactics", "approach",
                   "playing style", "how they play", "how they played"}


def _detect_team_style_query(query: str) -> str | None:
    """
    Detect if a query is asking about a team's playing style.
    Returns team name if detected, None otherwise.
    """
    query_lower = query.lower().strip()

    # Check for style keywords
    has_style = any(kw in query_lower for kw in _STYLE_KEYWORDS)
    if not has_style:
        return None

    # Extract team name using common patterns
    import re
    patterns = [
        r"(?:what\s+was|how\s+did|describe)\s+(.+?)(?:'s|s')\s+(?:playing\s+)?style",
        r"(.+?)(?:'s|s')\s+(?:playing\s+)?style",
        r"(?:what\s+was|how\s+did)\s+(.+?)\s+play",
        r"(.+?)\s+(?:formation|tactics|approach)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            team_raw = match.group(1).strip()
            # Clean up common words
            for word in ["the", "a", "an", "in", "during", "at"]:
                team_raw = team_raw.replace(f" {word} ", " ").strip()
            if len(team_raw) > 2:
                return team_raw.title()

    return None


def _ensure_team_style_doc(
    query: str,
    results: list[dict],
    k: int,
) -> list[dict]:
    """
    When a query asks about a team's playing style, ensure the team-level
    analysis document is included in the top-k results.
    """
    team_name = _detect_team_style_query(query)
    if not team_name:
        return results

    top_k = results[:k]
    team_lower = team_name.lower()

    # Check if team doc is already in top-k
    has_team_in_topk = any(
        r.get("metadata", {}).get("level") == "team" and
        team_lower in (r.get("metadata", {}).get("team_name") or r.get("team_name") or "").lower()
        for r in top_k
    )

    if has_team_in_topk:
        return results

    # Find team doc in chunks and prepend
    chunks = _load_chunks()
    for chunk in chunks:
        if chunk.get("level") == "team":
            chunk_team = (chunk.get("team_name") or
                          chunk.get("metadata", {}).get("team_name", "")).lower()
            if team_lower in chunk_team and chunk["chunk_id"].endswith("-chunk-0"):
                addition = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"),
                        "level": "team",
                        "team_name": chunk.get("team_name") or chunk.get("metadata", {}).get("team_name"),
                    },
                    "score": 0.01,
                    "rrf_score": 0.01,
                    "source": "team_style_boost",
                }
                # Deduplicate: prepend and remove any existing copy
                seen_ids = set()
                deduped = [addition]
                seen_ids.add(addition["chunk_id"])
                for r in results:
                    if r["chunk_id"] not in seen_ids:
                        deduped.append(r)
                        seen_ids.add(r["chunk_id"])
                results = deduped
                break

    return results


# ---------------------------------------------------------------------------
# Match-Level Query Detection
# ---------------------------------------------------------------------------

_MATCH_QUERY_PATTERNS = [
    r"how\s+did\s+(.+?)\s+(?:perform|play|do)\b",
    r"what\s+happened\s+in\s+(?:the\s+)?(?:match\s+)?(?:between\s+)?(.+?)\s+(?:and|vs)",
    r"how\s+did\s+(.+?)\s+fare",
    r"describe\s+(.+?)(?:'s|s')\s+(?:match|game|performance)",
    r"(.+?)(?:'s|s')\s+(?:match|game)\s+(?:against|vs)",
]

_STAGE_KEYWORDS = {
    "semi": "Semi-finals",
    "semi-final": "Semi-finals",
    "quarter": "Quarter-finals",
    "quarter-final": "Quarter-finals",
    "final": "Final",
    "round of 16": "Round of 16",
    "group": "Group Stage",
    "group stage": "Group Stage",
    "3rd place": "3rd Place Final",
}


def _detect_match_query(query: str) -> tuple[str | None, str | None]:
    """
    Detect if a query is asking about a specific match.
    Returns (team_name, stage) if detected, (None, None) otherwise.
    """
    query_lower = query.lower().strip()

    # Extract stage
    stage = None
    for keyword, stage_name in _STAGE_KEYWORDS.items():
        if keyword in query_lower:
            stage = stage_name
            break

    # Extract team name
    for pattern in _MATCH_QUERY_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            team_raw = match.group(1).strip()
            # Clean up common words
            for word in ["the", "a", "an", "in", "during", "at", "did"]:
                team_raw = team_raw.replace(f" {word} ", " ").strip()
            if len(team_raw) > 2:
                return team_raw.title(), stage

    return None, stage


def _ensure_match_summary(
    query: str,
    results: list[dict],
    k: int,
) -> list[dict]:
    """
    When a query asks about a specific match, ensure the Level-1 Match Summary
    document is included in the top-k results.
    """
    team_name, stage = _detect_match_query(query)
    if not team_name and not stage:
        return results

    top_k = results[:k]
    team_lower = (team_name or "").lower()
    stage_lower = (stage or "").lower()

    # Check if L1 doc matching the team/stage is already in top-k
    # Use exact stage matching to avoid "Semi-finals" matching "final"
    def _matches_stage(text: str, stage: str) -> bool:
        text_lower = text.lower()[:300]
        stage_lower = stage.lower()
        # Exact match: "the Final between" or "Final," etc.
        if stage_lower == "final":
            return "final between" in text_lower or "final," in text_lower or "the final" in text_lower
        return stage_lower in text_lower

    has_l1_in_topk = any(
        r.get("metadata", {}).get("level") == "1" and
        (team_lower in (r.get("text", "") or "").lower()[:300] if team_lower else True) and
        (_matches_stage(r.get("text", ""), stage) if stage else True)
        for r in top_k
    )

    if has_l1_in_topk:
        return results

    # Find matching L1 doc in chunks
    chunks = _load_chunks()
    for chunk in chunks:
        if chunk.get("level") != "1":
            continue

        text = chunk.get("text", "")
        text_lower = text.lower()[:300]
        team_match = team_lower in text_lower if team_lower else True
        stage_match = _matches_stage(text, stage) if stage else True

        if team_match and stage_match:
            addition = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "document_id": chunk.get("document_id") or chunk.get("metadata", {}).get("document_id"),
                    "level": "1",
                    "match_id": chunk.get("match_id") or chunk.get("metadata", {}).get("match_id"),
                    "home_team": chunk.get("metadata", {}).get("home_team"),
                    "away_team": chunk.get("metadata", {}).get("away_team"),
                },
                "score": 0.01,
                "rrf_score": 0.01,
                "source": "match_summary_boost",
            }
            # Deduplicate: prepend and remove any existing copy
            seen_ids = set()
            deduped = [addition]
            seen_ids.add(addition["chunk_id"])
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    deduped.append(r)
                    seen_ids.add(r["chunk_id"])
            return deduped

    return results


# ---------------------------------------------------------------------------
# Hybrid Search — Orchestrator (LAB 8 — Step 6)
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    k: int = 5,
    bm25_k: int = 20,
    dense_k: int = 20,
    level_filter: str | None = None,
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

    Returns:
        Top-K results with RRF scores.
    """
    # Step 1: BM25 retrieval
    bm25_results = bm25_search(query, k=bm25_k)

    # Step 2: Dense retrieval
    dense_results = dense_search(query, k=dense_k, level_filter=level_filter)

    # Step 3: Merge via RRF
    merged = reciprocal_rank_fusion([bm25_results, dense_results])

    # Step 4: Re-rank
    reranked = rerank(merged, query)

    # Step 5: Ensure comparison entities' L4 docs are included
    reranked = _ensure_comparison_entities(query, reranked, k)

    # Step 6: Ensure team-level doc for style queries
    reranked = _ensure_team_style_doc(query, reranked, k)

    # Step 7: Ensure Level-1 match summary for match-level queries
    reranked = _ensure_match_summary(query, reranked, k)

    # Step 8: Return Top-K
    return reranked[:k]


# ---------------------------------------------------------------------------
# Backward Compatibility — Dense-only Search (LAB 7)
# ---------------------------------------------------------------------------


def semantic_search(query: str, persist_dir: Path = CHROMA_DIR,
                    collection_name: str = COLLECTION_NAME,
                    k: int = 5, level_filter: str | None = None) -> list[dict]:
    """
    Dense-only search using ChromaDB (backward compatibility).

    Use hybrid_search() for the full pipeline.
    """
    from chromadb import PersistentClient
    from src.cache import get_embedding_model

    model = get_embedding_model(EMBEDDING_MODEL)
    client = PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    where = None
    if level_filter:
        where = {"level": level_filter}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where=where,
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
        })

    return formatted


# ---------------------------------------------------------------------------
# Context Building (LAB 9 — unchanged)
# ---------------------------------------------------------------------------


def build_context(chunks: list[dict], max_length: int = 3000) -> str:
    """
    Build a context string from retrieved chunks.

    Formats chunks with metadata and truncates to max_length.
    """
    if not chunks:
        return "No relevant documents found."

    context_parts = []
    current_length = 0

    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        level = meta.get("level", "unknown")

        # Add header with metadata
        header = f"[Document {i+1} - Level {level}"
        if meta.get("player_name"):
            header += f", Player: {meta['player_name']}"
        if meta.get("team_name"):
            header += f", Team: {meta['team_name']}"
        if meta.get("match_id"):
            header += f", Match: {meta['match_id']}"

        # Show RRF score if available, otherwise show score
        score_key = "rrf_score" if "rrf_score" in chunk else "score"
        header += f", Score: {chunk.get(score_key, 0):.4f}]"

        text = chunk["text"]
        entry = f"{header}\n{text}\n"

        if current_length + len(entry) > max_length:
            break

        context_parts.append(entry)
        current_length += len(entry)

    return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# Convenience (LAB 8 — Step 7)
# ---------------------------------------------------------------------------


def retrieve_context(
    query: str,
    k: int = 5,
    max_length: int = 3000,
    level_filter: str | None = None,
    mode: str = "hybrid",
) -> dict:
    """
    Retrieve context for a query.

    Parameters:
        query: User query
        k: Number of results
        max_length: Max context length
        level_filter: Optional filter by document level
        mode: "hybrid" (default) or "semantic" (dense-only)

    Returns:
        {query, context, chunks, num_chunks, mode}
    """
    if mode == "hybrid":
        chunks = hybrid_search(query, k=k, level_filter=level_filter)
    else:
        chunks = semantic_search(query, k=k, level_filter=level_filter)

    context = build_context(chunks, max_length)

    return {
        "query": query,
        "context": context,
        "chunks": chunks,
        "num_chunks": len(chunks),
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid retrieval test")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    parser.add_argument("--mode", choices=["hybrid", "semantic"], default="hybrid",
                        help="Retrieval mode")
    parser.add_argument("--level", help="Filter by document level")
    parser.add_argument("--max-length", type=int, default=3000,
                        help="Max context length")
    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Retrieving {args.k} chunks...")
    print()

    result = retrieve_context(
        args.query,
        k=args.k,
        max_length=args.max_length,
        level_filter=args.level,
        mode=args.mode,
    )

    print(f"Retrieved {result['num_chunks']} chunks:")
    print("=" * 60)
    print(result["context"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
