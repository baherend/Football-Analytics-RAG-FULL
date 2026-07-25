"""
04_vector_representation.py — Phase 4: Vector Representations

Builds all vector representations for retrieval:
- TF-IDF sparse vectors (keyword-based)
- BM25Okapi index (keyword-based)
- Sentence embeddings via all-MiniLM-L6-v2 (dense, 384 dimensions)
- ChromaDB persistent vector store

Input: chunks.json
Output: output/indices/ (TF-IDF, BM25), output/embeddings/ (dense vectors),
        output/chroma_db/ (ChromaDB store)

Usage:
    python 04_vector_representation.py
    python 04_vector_representation.py --skip-chroma
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_PATH = Path("output/chunks.json")
INDICES_DIR = Path("output/indices")
EMBEDDINGS_DIR = Path("output/embeddings")
CHROMA_DIR = Path("output/chroma_db")
COLLECTION_NAME = "wc2022_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Tokenization (shared by TF-IDF and BM25)
# ---------------------------------------------------------------------------


def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if len(t) > 1]  # skip single chars


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------


def build_tfidf_index(chunks: list[dict]) -> tuple:
    """
    Build TF-IDF index from chunks.

    Returns (vectorizer, tfidf_matrix, chunk_ids).
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    vectorizer = TfidfVectorizer(
        tokenizer=simple_tokenize,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix, chunk_ids


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


def build_bm25_index(chunks: list[dict]) -> tuple:
    """
    Build BM25 index from chunks.

    Returns (bm25, chunk_ids, tokenized_corpus).
    """
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    tokenized_corpus = [simple_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, chunk_ids, tokenized_corpus


# ---------------------------------------------------------------------------
# Sentence Embeddings
# ---------------------------------------------------------------------------


def generate_embeddings(chunks: list[dict], model_name: str = EMBEDDING_MODEL,
                        batch_size: int = 64) -> np.ndarray:
    """
    Generate dense embeddings for all chunks.

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
# ChromaDB Store
# ---------------------------------------------------------------------------


def create_chroma_store(chunks: list[dict], persist_dir: Path,
                        collection_name: str = COLLECTION_NAME,
                        model_name: str = EMBEDDING_MODEL) -> int:
    """
    Create ChromaDB collection from chunks.

    Returns number of documents added.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model for ChromaDB: {model_name}")
    model = SentenceTransformer(model_name)

    persist_dir.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "FIFA World Cup 2022 documents"}
    )

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

    print(f"Generating embeddings for {len(texts)} chunks (ChromaDB)...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

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
    return total_added


# ---------------------------------------------------------------------------
# Search Functions
# ---------------------------------------------------------------------------


def tfidf_search(query: str, vectorizer, tfidf_matrix, chunks: list[dict],
                 k: int = 5) -> list[dict]:
    """Search using TF-IDF similarity."""
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().flatten()

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            })
    return results


def bm25_search(query: str, bm25, chunks: list[dict],
                k: int = 5) -> list[dict]:
    """Search using BM25."""
    query_tokens = simple_tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(scores[idx]),
            })
    return results


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
# Persistence
# ---------------------------------------------------------------------------


def save_indices(vectorizer, tfidf_matrix, bm25, chunk_ids, output_dir: Path):
    """Save TF-IDF and BM25 indices to disk."""
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(output_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(output_dir / "chunk_ids.json", "w") as f:
        json.dump(chunk_ids, f)

    print(f"Saved indices to {output_dir}")


def load_indices(output_dir: Path) -> tuple:
    """Load TF-IDF and BM25 indices from disk."""
    with open(output_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(output_dir / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(output_dir / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(output_dir / "chunk_ids.json") as f:
        chunk_ids = json.load(f)

    return vectorizer, tfidf_matrix, bm25, chunk_ids


def save_embeddings(embeddings: np.ndarray, chunks: list[dict], output_dir: Path):
    """Save embeddings and metadata to disk."""
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)

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


def load_embeddings(output_dir: Path) -> tuple:
    """Load embeddings and metadata from disk."""
    embeddings = np.load(output_dir / "embeddings.npy")
    with open(output_dir / "embedding_metadata.json") as f:
        metadata = json.load(f)
    return embeddings, metadata


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def build_all_representations(chunks: list[dict], skip_chroma: bool = False) -> dict:
    """Build and save all vector representations."""
    # TF-IDF + BM25
    print(f"Building TF-IDF index for {len(chunks)} chunks...")
    vectorizer, tfidf_matrix, chunk_ids = build_tfidf_index(chunks)

    print(f"Building BM25 index...")
    bm25, _, _ = build_bm25_index(chunks)

    save_indices(vectorizer, tfidf_matrix, bm25, chunk_ids, INDICES_DIR)

    # Dense embeddings
    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL)
    save_embeddings(embeddings, chunks, EMBEDDINGS_DIR)

    # ChromaDB
    chroma_count = 0
    if not skip_chroma:
        chroma_count = create_chroma_store(chunks, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL)

    return {
        "total_chunks": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "tfidf_shape": tfidf_matrix.shape,
        "embedding_dim": embeddings.shape[1],
        "chroma_documents": chroma_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build vector representations")
    parser.add_argument("--skip-chroma", action="store_true",
                        help="Skip ChromaDB store creation")
    parser.add_argument("--model", default=EMBEDDING_MODEL,
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    if not CHUNKS_PATH.exists():
        print(f"Error: {CHUNKS_PATH} not found. Run 03_chunking.py first.")
        return 1

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Building representations from {len(chunks)} chunks...")
    stats = build_all_representations(chunks, skip_chroma=args.skip_chroma)

    print(f"\nRepresentation building complete:")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Vocabulary size:  {stats['vocabulary_size']}")
    print(f"  TF-IDF shape:     {stats['tfidf_shape']}")
    print(f"  Embedding dim:    {stats['embedding_dim']}")
    if stats['chroma_documents']:
        print(f"  ChromaDB docs:    {stats['chroma_documents']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
