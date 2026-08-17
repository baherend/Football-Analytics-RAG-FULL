"""
src/retrieval/dense.py -- Dense (vector) retrieval via ChromaDB.

Migration Step 2 (Retrieval Split): mechanically extracted from
src/retrieval/search.py's Dense Retrieval and Backward-Compatibility
Dense-only Search sections -- no logic changes. See src/retrieval/search.py
for the compatibility re-exports existing callers keep using.

Does NOT do BM25, fusion, safeguards, or orchestration -- see
src/retrieval/bm25.py, fusion.py, safeguards.py, service.py for those.
"""

from __future__ import annotations

from pathlib import Path

from src.artifacts import ArtifactPaths
from src.embedding_config import resolve_embedding_config

CHROMA_DIR = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
EMBEDDING_MODEL = resolve_embedding_config().hf_name  # legacy default (MiniLM) -- see src.embedding_config


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


def semantic_search(query: str, persist_dir: Path = CHROMA_DIR,
                    collection_name: str = COLLECTION_NAME,
                    k: int = 5, level_filter: str | None = None) -> list[dict]:
    """
    Dense-only search using ChromaDB (backward compatibility).

    Use hybrid_search() for the full pipeline (see src/retrieval/service.py).
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
