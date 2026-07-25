"""
05_create_chroma_store.py — Stage 5: ChromaDB Vector Store

Creates or loads a persistent ChromaDB collection for semantic retrieval.
Uses sentence-transformers embeddings (all-MiniLM-L6-v2).

Input: chunks from 03_chunking.py
Output: ChromaDB persistent store at output/chroma_db/
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Store Creation
# ---------------------------------------------------------------------------

def create_vector_store(chunks: list[dict] | None = None,
                        persist_dir: Path = DB_PATH,
                        collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Create or load ChromaDB collection from chunks."""
    from sentence_transformers import SentenceTransformer

    if chunks is None:
        chunks_path = Path("output/chunks.json")
        if chunks_path.exists():
            with open(chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            chunks = import_module("03_chunking").chunks

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "FIFA World Cup 2022 documents"}
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


# ---------------------------------------------------------------------------
# Store Loading
# ---------------------------------------------------------------------------

def get_collection(persist_dir: Path = DB_PATH,
                   collection_name: str = COLLECTION_NAME) -> chromadb.Collection:
    """Load existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(collection_name)


if __name__ == "__main__":
    create_vector_store()
    print("Done.")
