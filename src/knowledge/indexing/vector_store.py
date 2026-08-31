"""
src/knowledge/indexing/vector_store.py -- Chroma vector-store construction.

Migration Step 6: extracted verbatim from 05_create_chroma_store.py, which
keeps its CLI `main()` and re-exports these for compatibility.

knowledge/ *writes* the vector store; rag/ *reads* it (`src/retrieval/dense.py`
opens the collection at query time). The collection name and embedding model
are resolved through src/artifacts.py and src/embedding_config.py, so a new
competition/season is added by configuration, never by copying this module.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from src.artifacts import resolve_chroma_collection_name
from src.embedding_config import resolve_embedding_config

__all__ = ["DB_PATH", "COLLECTION_NAME", "MODEL_NAME",
           "create_vector_store", "get_collection"]

DB_PATH = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
MODEL_NAME = resolve_embedding_config().hf_name  # legacy default (MiniLM) -- see src.embedding_config


def create_vector_store(chunks: list[dict] | None = None,
                        persist_dir: Path = DB_PATH,
                        collection_name: str = COLLECTION_NAME,
                        embedding_model_id: str | None = None) -> chromadb.Collection:
    """
    Create or load ChromaDB collection from chunks.

    `embedding_model_id` selects which registered model (see
    src.embedding_config) embeds the documents; None resolves to the
    project default (MiniLM), unchanged from prior behavior. Building a
    collection with one model never touches another model's collection,
    since callers are expected to pass a `collection_name` that already
    encodes the model identity (see ArtifactPaths.chroma_collection_name)
    -- this function itself only deletes/creates the exact `collection_name`
    given to it.
    """
    from sentence_transformers import SentenceTransformer

    if chunks is None:
        chunks_path = Path("output/chunks.json")
        if chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            chunks = import_module("03_chunking").chunks

    embedding_config = resolve_embedding_config(embedding_model_id)
    print(f"Loading embedding model: {embedding_config.hf_name}")
    model = SentenceTransformer(embedding_config.hf_name)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Football competition documents"}
    )

    texts = [c.get("search_text", c["text"]) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {"document_id": c["document_id"], "level": c.get("level", "unknown")}
        if c.get("match_id"):
            meta["match_id"] = c["match_id"]
        if c.get("player_name"):
            meta["player_name"] = c["player_name"]
        if c.get("team_name"):
            meta["team_name"] = c["team_name"]
        chunk_metadata = c.get("metadata", {})
        if isinstance(chunk_metadata, dict):
            if chunk_metadata.get("home_team"):
                meta["home_team"] = chunk_metadata["home_team"]
            if chunk_metadata.get("away_team"):
                meta["away_team"] = chunk_metadata["away_team"]
            if chunk_metadata.get("match_date"):
                meta["match_date"] = chunk_metadata["match_date"]
        if c.get("match_date") and "match_date" not in meta:
            meta["match_date"] = c["match_date"]
        metadatas.append(meta)

    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"ChromaDB collection created: {collection_name} ({len(texts)} documents)")
    return collection


def get_collection(persist_dir: Path = DB_PATH,
                   collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Load existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(Path(persist_dir).resolve()))
    return client.get_collection(collection_name)

