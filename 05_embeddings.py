"""
05_embeddings.py — Phase 4: Sentence Embeddings

Generates dense vector embeddings for semantic retrieval.

Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)

Input: chunks.json
Output: embeddings.npy, embedding_metadata.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(chunks: list[dict], model_name: str = "all-MiniLM-L6-v2",
                        batch_size: int = 64) -> np.ndarray:
    """
    Generate embeddings for all chunks.

    Returns numpy array of shape (n_chunks, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_embeddings(embeddings: np.ndarray, chunks: list[dict], output_dir: Path):
    """Save embeddings and metadata to disk."""
    output_dir.mkdir(exist_ok=True)

    # Save embeddings
    np.save(output_dir / "embeddings.npy", embeddings)

    # Save metadata
    metadata = {
        "total_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "document_ids": [c["document_id"] for c in chunks],
    }
    with open(output_dir / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings to {output_dir}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")


def load_embeddings(output_dir: Path) -> tuple:
    """Load embeddings and metadata from disk."""
    embeddings = np.load(output_dir / "embeddings.npy")
    with open(output_dir / "embedding_metadata.json") as f:
        metadata = json.load(f)
    return embeddings, metadata


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def embedding_search(query: str, model, embeddings: np.ndarray,
                     chunks: list[dict], k: int = 5) -> list[dict]:
    """Search using cosine similarity on embeddings."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding.T).flatten()

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            })
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_embeddings(input_path: Path, output_dir: Path,
                     model_name: str = "all-MiniLM-L6-v2") -> dict:
    """Build and save embeddings."""
    with open(input_path, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = generate_embeddings(chunks, model_name)
    save_embeddings(embeddings, chunks, output_dir)

    return {
        "total_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate sentence embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    input_path = Path("output/chunks.json")
    output_dir = Path("output/embeddings")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 03_chunking.py first.")
        return 1

    print(f"Building embeddings from {input_path}...")
    stats = build_embeddings(input_path, output_dir, args.model)

    print(f"\nEmbedding generation complete:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Embedding dim: {stats['embedding_dim']}")
    print(f"  Model: {stats['model']}")
    print(f"\nOutput: {output_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
