"""
06_create_chroma_store.py — Phase 4: ChromaDB Vector Store

Creates persistent ChromaDB collection for semantic retrieval.

Input: chunks.json
Output: chroma_db/ directory
"""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# ChromaDB creation
# ---------------------------------------------------------------------------


def create_chroma_store(chunks: list[dict], persist_dir: Path,
                        collection_name: str = "wc2022_documents",
                        model_name: str = "all-MiniLM-L6-v2") -> int:
    """
    Create ChromaDB collection from chunks.

    Returns number of documents added.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create persistent client
    persist_dir.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "FIFA World Cup 2022 documents"}
    )

    # Prepare data
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        meta = {
            "document_id": c["document_id"],
            "level": c["level"],
        }
        if c.get("match_id"):
            meta["match_id"] = c["match_id"]
        if c.get("player_name"):
            meta["player_name"] = c["player_name"]
        if c.get("team_name"):
            meta["team_name"] = c["team_name"]
        metadatas.append(meta)

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # Add to collection in batches
    batch_size = 100
    total_added = 0

    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end].tolist(),
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )
        total_added += end - i
        print(f"  Added {total_added}/{len(texts)} chunks")

    print(f"\nChromaDB collection created: {collection_name}")
    print(f"  Total documents: {total_added}")
    print(f"  Persist directory: {persist_dir}")

    return total_added


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_chroma(query: str, persist_dir: Path,
                  collection_name: str = "wc2022_documents",
                  k: int = 5, level_filter: str | None = None) -> list[dict]:
    """
    Search ChromaDB collection.

    Returns list of {text, metadata, distance}.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    # Load model and client
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(persist_dir))
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
        formatted.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return formatted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Create ChromaDB vector store")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--collection", default="wc2022_documents",
                        help="Collection name")
    args = parser.parse_args()

    input_path = Path("output/chunks.json")
    persist_dir = Path("output/chroma_db")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 03_chunking.py first.")
        return 1

    with open(input_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Creating ChromaDB store from {len(chunks)} chunks...")
    total = create_chroma_store(chunks, persist_dir, args.collection, args.model)

    print(f"\nChromaDB store creation complete:")
    print(f"  Collection: {args.collection}")
    print(f"  Total documents: {total}")
    print(f"  Persist directory: {persist_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
