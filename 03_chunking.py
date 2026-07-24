"""
03_chunking.py — Phase 4: Document Chunking

Splits long documents into chunks for better retrieval.
Uses sentence-based chunking with overlap.

Input: processed_documents.json
Output: chunks.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------

# Maximum chunk size in characters
MAX_CHUNK_SIZE = 500

# Overlap between chunks (in characters)
CHUNK_OVERLAP = 50

# Sentence boundary patterns
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# Chunking functions
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = SENTENCE_END.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(doc: dict, max_size: int = MAX_CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split a document into chunks.

    Strategy:
    - If document is short enough (<= max_size), return as single chunk.
    - Otherwise, split by sentences and group into chunks.
    - Overlap: include last sentence of previous chunk in next chunk.
    """
    text = doc["cleaned_text"]
    doc_id = doc["document_id"]

    # Short document — single chunk
    if len(text) <= max_size:
        return [{
            "chunk_id": f"{doc_id}-chunk-0",
            "document_id": doc_id,
            "level": doc["level"],
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": text,
            "metadata": doc["metadata"],
        }]

    # Split into sentences
    sentences = split_sentences(text)
    if not sentences:
        return [{
            "chunk_id": f"{doc_id}-chunk-0",
            "document_id": doc_id,
            "level": doc["level"],
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": text,
            "metadata": doc["metadata"],
        }]

    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for i, sentence in enumerate(sentences):
        sentence_len = len(sentence)

        # If adding this sentence exceeds max size, finalize current chunk
        if current_length + sentence_len > max_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
                "document_id": doc_id,
                "level": doc["level"],
                "match_id": doc.get("match_id"),
                "player_name": doc.get("player_name"),
                "team_name": doc.get("team_name"),
                "text": chunk_text,
                "metadata": doc["metadata"],
            })
            chunk_idx += 1

            # Overlap: keep last sentence for context
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len

    # Finalize last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
            "document_id": doc_id,
            "level": doc["level"],
            "match_id": doc.get("match_id"),
            "player_name": doc.get("player_name"),
            "team_name": doc.get("team_name"),
            "text": chunk_text,
            "metadata": doc["metadata"],
        })

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def chunk_documents(input_path: Path, output_path: Path) -> dict:
    """
    Chunk all processed documents.

    Returns statistics about the chunking.
    """
    with open(input_path, encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []
    stats = {
        "total_documents": len(documents),
        "total_chunks": 0,
        "by_level": {},
        "avg_chunk_length": 0,
    }

    total_length = 0

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

        level = doc["level"]
        stats["by_level"][level] = stats["by_level"].get(level, 0) + len(chunks)
        total_length += sum(len(c["text"]) for c in chunks)

    stats["total_chunks"] = len(all_chunks)
    stats["avg_chunk_length"] = total_length / len(all_chunks) if all_chunks else 0

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    input_path = Path("output/processed_documents.json")
    output_path = Path("output/chunks.json")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run 02_preprocessing.py first.")
        return 1

    print(f"Chunking {input_path}...")
    stats = chunk_documents(input_path, output_path)

    print(f"\nChunking complete:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  By level: {stats['by_level']}")
    print(f"  Avg chunk length: {stats['avg_chunk_length']:.0f} chars")
    print(f"\nOutput: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
