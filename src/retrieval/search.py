"""
src/retrieval/search.py -- Retrieval: BM25 + dense + hybrid search.

Structural Cleanup Phase A: mechanically extracted from
06_retrieve_context.py's retrieval half (everything before its former
"Query Router" section) -- no logic changes. See 06_retrieve_context.py
for the temporary compatibility wrapper that re-exports these symbols for
existing callers.

Combines:
- Lexical (BM25) and semantic (dense) retrieval via Reciprocal Rank Fusion
- Retrieval-side entity/team/match evidence boosting
- Context building for the LLM prompt

Does NOT classify query intent, parse StructuredQuery, execute structured
queries, perform comparison orchestration, or call LLMs -- see
src/query/router.py for that.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from src.artifacts import ArtifactPaths
from src.embedding_config import resolve_embedding_config
from src.retrieval.chunk_selector import select_relevant_chunks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDICES_DIR = Path("output/indices")
CHUNKS_PATH = Path("output/chunks.json")
CHROMA_DIR = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
EMBEDDING_MODEL = resolve_embedding_config().hf_name  # legacy default (MiniLM) -- see src.embedding_config

# RRF constant (standard from the original RRF paper)
RRF_K = 60


# ---------------------------------------------------------------------------
# Index Loading (LAB 8 — Step 1)
# ---------------------------------------------------------------------------

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
# Lexical Retrieval — BM25 (LAB 8 — Step 2)
# ---------------------------------------------------------------------------


def bm25_search(query: str, k: int = 20, artifact_paths: ArtifactPaths | None = None) -> list[dict]:
    """
    Lexical retrieval using BM25.

    `artifact_paths` selects a namespaced dataset's bm25.pkl/chunks.json
    (see src/artifacts.py) instead of the module-level legacy defaults.
    Defaults to None -- unchanged legacy WC2022 behavior.

    Returns list of {chunk_id, text, metadata, score, rank}.
    Retrieves more candidates (k=20) for fusion.
    """
    bm25_path = artifact_paths.bm25_index if artifact_paths is not None else None
    chunks_path = artifact_paths.chunks if artifact_paths is not None else None
    bm25 = _load_bm25_index(bm25_path)
    chunks = _load_chunks(chunks_path)
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
                 level_filter: str | None = None,
                 persist_dir: Path = CHROMA_DIR,
                 collection_name: str = COLLECTION_NAME,
                 artifact_paths: ArtifactPaths | None = None) -> list[dict]:
    """
    Dense retrieval using ChromaDB embeddings.

    `persist_dir`/`collection_name` default to the legacy module-level constants.
    When `artifact_paths` is given, it selects the namespaced Chroma
    directory, that dataset's deterministic collection name, AND the
    embedding model tied to that identity (artifact_paths.embedding_model_id)
    -- the exact model the index was built with -- preventing both
    cross-competition collection reuse and a query embedded with the wrong
    model. If that model's collection was never built, Chroma's
    get_collection() raises a clear error rather than silently returning
    results from an unrelated (or nonexistent) index.

    Returns list of {chunk_id, text, metadata, score, rank}.
    Retrieves more candidates (k=20) for fusion.
    """
    from chromadb import PersistentClient
    from src.cache import get_embedding_model

    if artifact_paths is not None:
        persist_dir = artifact_paths.chroma_dir
        collection_name = artifact_paths.chroma_collection_name
        embedding_model_name = resolve_embedding_config(artifact_paths.embedding_model_id).hf_name
    else:
        embedding_model_name = EMBEDDING_MODEL

    # Use cached model (loaded once per resolved model name)
    model = get_embedding_model(embedding_model_name)
    client = PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

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
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Ensure that when a query compares two entities, both entities' L4
    tournament summary documents are included in the final top-k results.

    Strategy: check only the top-k (not all candidates), and if an entity's
    L4 doc is missing from top-k, find it in the chunk store and prepend it.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
    """
    entities = _detect_comparison_entities(query)
    if len(entities) < 2:
        return results

    # Load chunks for L4 lookup
    chunks = _load_chunks(artifact_paths.chunks if artifact_paths is not None else None)

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

_STYLE_KEYWORDS = {
    "style",
    "formation",
    "formations",
    "play pattern",
    "play patterns",
    "passing pattern",
    "passing patterns",
    "tactics",
    "approach",
    "playing style",
    "how they play",
    "how they played",
}


def _detect_team_style_query(query: str) -> str | None:
    """Return the team name for a team-style query, otherwise None."""
    query_lower = query.lower().strip()

    if not any(keyword in query_lower for keyword in _STYLE_KEYWORDS):
        return None

    patterns = [
        (
            r"^(?:what\s+(?:was|were)\s+|describe\s+|how\s+did\s+)?"
            r"(.+?)(?:['?]s|s')\s+"
            r"(?:passing\s+patterns?|(?:most\s+common\s+)?formations?|"
            r"tactics|approach|(?:playing\s+)?style)\b"
        ),
        r"^(.+?)(?:['?]s|s')\s+(?:playing\s+)?style\b",
        r"^(?:what\s+(?:was|were)|how\s+did)\s+(.+?)\s+play\b",
        r"^(?:describe\s+)?(.+?)\s+(?:formations?|tactics|approach)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if not match:
            continue

        team_name = match.group(1).strip(" ,.?")
        team_name = re.sub(
            r"^(?:what\s+(?:was|were)|how\s+did|describe)\s+",
            "",
            team_name,
        ).strip()

        if len(team_name) > 2:
            return team_name.title()

    return None


def _ensure_team_style_doc(
    query: str,
    results: list[dict],
    k: int,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    When a query asks about a team's playing style, ensure the team-level
    analysis document is included in the top-k results.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
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
    chunks = _load_chunks(artifact_paths.chunks if artifact_paths is not None else None)
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
    """Return the detected team and stage for a specific-match query."""
    query_lower = query.lower().strip()

    stage = None
    for keyword, stage_name in _STAGE_KEYWORDS.items():
        if keyword in query_lower:
            stage = stage_name
            break

    head_to_head_patterns = [
        (
            r"\b(?:in|between)\s+(?:the\s+)?"
            r"([a-z?-?][a-z?-? .'-]+?)\s+"
            r"(?:vs\.?|versus|and)\s+"
            r"([a-z?-?][a-z?-? .'-]+?)"
            r"(?=\s+(?:final|semi-finals?|semi-final|quarter-finals?|"
            r"quarter-final|round of 16|group stage|3rd place final)\b|\?|$)"
        ),
        (
            r"^([a-z?-?][a-z?-? .'-]+?)\s+"
            r"(?:vs\.?|versus)\s+"
            r"([a-z?-?][a-z?-? .'-]+?)"
            r"(?=\s+(?:final|semi-finals?|semi-final|quarter-finals?|"
            r"quarter-final|round of 16|group stage|3rd place final)\b|\?|$)"
        ),
    ]

    for pattern in head_to_head_patterns:
        match = re.search(pattern, query_lower)
        if match:
            team_name = match.group(1).strip(" ,.?")
            if len(team_name) > 2:
                return team_name.title(), stage

    for pattern in _MATCH_QUERY_PATTERNS:
        match = re.search(pattern, query_lower)
        if not match:
            continue

        team_name = match.group(1).strip()
        for word in ["the", "a", "an", "in", "during", "at", "did"]:
            team_name = team_name.replace(f" {word} ", " ").strip()

        if len(team_name) > 2:
            return team_name.title(), stage

    return None, stage


def _ensure_match_summary(
    query: str,
    results: list[dict],
    k: int,
    artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Include an L1 match summary only for a genuine match query, without
    displacing stronger retrieved results.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py.
    """
    team_name, stage = _detect_match_query(query)
    if not team_name and not stage:
        return results

    query_lower = query.lower().strip()
    explicit_match_intent = any(
        marker in query_lower
        for marker in (
            " match",
            " game",
            "between ",
            " vs ",
            " versus ",
            "what happened",
            "key events",
        )
    )

    if team_name is None and not explicit_match_intent:
        return results

    chunks = _load_chunks(artifact_paths.chunks if artifact_paths is not None else None)
    team_lower = (team_name or "").lower()

    if team_name:
        known_teams = {
            (
                chunk.get("team_name")
                or chunk.get("metadata", {}).get("team_name")
                or ""
            ).strip().lower()
            for chunk in chunks
            if chunk.get("level") == "team"
        }
        known_teams.discard("")

        is_known_team = any(
            team_lower == known
            or team_lower in known
            or known in team_lower
            for known in known_teams
        )
        if not is_known_team:
            return results

    def _matches_stage(text: str, requested_stage: str) -> bool:
        text_lower = (text or "").lower()[:300]
        requested = requested_stage.lower()

        if requested == "final":
            exclusions = (
                "semi-final",
                "semi final",
                "quarter-final",
                "quarter final",
                "3rd place final",
                "third place final",
            )
            if any(value in text_lower for value in exclusions):
                return False
            return re.search(r"\bfinal\b", text_lower) is not None

        return requested in text_lower

    top_k = results[:k]
    already_present = any(
        item.get("metadata", {}).get("level") == "1"
        and (
            team_lower in (item.get("text", "") or "").lower()[:300]
            if team_lower
            else True
        )
        and (
            _matches_stage(item.get("text", ""), stage)
            if stage
            else True
        )
        for item in top_k
    )
    if already_present:
        return results

    for chunk in chunks:
        if chunk.get("level") != "1":
            continue

        text = chunk.get("text", "")
        team_match = team_lower in text.lower()[:300] if team_lower else True
        stage_match = _matches_stage(text, stage) if stage else True

        if not (team_match and stage_match):
            continue

        if k <= 1:
            return results

        addition = {
            "chunk_id": chunk["chunk_id"],
            "text": text,
            "metadata": {
                "document_id": (
                    chunk.get("document_id")
                    or chunk.get("metadata", {}).get("document_id")
                ),
                "level": "1",
                "match_id": (
                    chunk.get("match_id")
                    or chunk.get("metadata", {}).get("match_id")
                ),
                "home_team": chunk.get("metadata", {}).get("home_team"),
                "away_team": chunk.get("metadata", {}).get("away_team"),
            },
            "score": 0.01,
            "rrf_score": 0.01,
            "source": "match_summary_boost",
        }

        existing = [
            item
            for item in results
            if item.get("chunk_id") != addition["chunk_id"]
        ]

        insert_at = min(k - 1, len(existing))
        return existing[:insert_at] + [addition] + existing[insert_at:]

    return results


def _expand_query_entity_siblings(
    query: str, results: list[dict], artifact_paths: ArtifactPaths | None = None,
) -> list[dict]:
    """
    Add sibling chunks for candidate documents whose entity appears in query.

    `artifact_paths` selects a namespaced dataset's chunks.json instead of
    the legacy default -- see src/artifacts.py. This safeguard must never
    silently reload the legacy WC2022 chunks for another dataset's query.
    """
    query_lower = query.casefold()
    entity_fields = ("team_name", "player_name", "home_team", "away_team")
    target_document_ids: set[str] = set()

    for result in results:
        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        entity_values = [
            metadata.get(field) or result.get(field)
            for field in entity_fields
        ]
        entity_matches = any(
            str(value).strip().casefold() in query_lower
            for value in entity_values
            if value
        )

        if entity_matches:
            document_id = result.get("document_id") or metadata.get("document_id")
            if document_id:
                target_document_ids.add(str(document_id))

    if not target_document_ids:
        return results

    expanded = list(results)
    seen_chunk_ids = {item.get("chunk_id") for item in expanded}

    chunks_path = artifact_paths.chunks if artifact_paths is not None else None
    for raw_chunk in _load_chunks(chunks_path):
        raw_metadata = raw_chunk.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}

        document_id = (
            raw_chunk.get("document_id")
            or raw_metadata.get("document_id")
        )
        chunk_id = raw_chunk.get("chunk_id")

        if document_id not in target_document_ids or chunk_id in seen_chunk_ids:
            continue

        metadata = dict(raw_metadata)
        metadata.setdefault("document_id", document_id)
        for field in ("level", "match_id", "player_name", "team_name", "home_team", "away_team"):
            if metadata.get(field) is None and raw_chunk.get(field) is not None:
                metadata[field] = raw_chunk.get(field)

        expanded.append({
            "chunk_id": chunk_id,
            "text": raw_chunk.get("text", ""),
            "metadata": metadata,
            "score": 0.0,
            "rrf_score": 0.0,
            "source": "sibling_expansion",
        })
        seen_chunk_ids.add(chunk_id)

    return expanded

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

        # Add a stable source label that matches the generation prompt's
        # citation contract and preserves exact chunk-level traceability.
        header = f"[Source {i+1}: Level {level}"
        if chunk.get("chunk_id"):
            header += f", chunk_id={chunk['chunk_id']}"
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
