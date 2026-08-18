"""
src/knowledge/chunking.py -- document -> retrievable chunks.

Migration Step 6: extracted verbatim from 03_chunking.py, which keeps its CLI
`main()` and re-exports these for compatibility.

Pure transformation: documents in, chunk dicts out. Chunk IDs and metadata
shape are unchanged -- they are the contract every downstream index and every
citation depends on. No filesystem access here; the CLI orchestrator loads the
selected competition/season's documents and writes its chunks.json.
"""

from __future__ import annotations

import re

__all__ = ["MAX_CHUNK_SIZE", "CHUNK_OVERLAP", "SENTENCE_END",
           "split_sentences", "chunk_document", "build_chunks"]

MAX_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP = 50     # characters
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def _make_chunk(doc: dict, chunk_idx: int, chunk_text: str) -> dict:
    """
    Build a single chunk dict for a document.

    This is the one place that defines what a "chunk" looks like — every
    call site below (short doc, no-sentences fallback, mid-loop, final
    leftover) goes through here instead of repeating the same dict literal.
    """
    doc_id = doc["document_id"]
    return {
        "chunk_id": f"{doc_id}-chunk-{chunk_idx}",
        "document_id": doc_id,
        "level": doc.get("level", "unknown"),
        "match_id": doc.get("match_id"),
        "player_name": doc.get("player_name"),
        "team_name": doc.get("team_name"),
        "text": chunk_text,
        "search_text": chunk_text,
        "metadata": doc.get("metadata", {}),
    }


def chunk_document(doc: dict, max_size: int = MAX_CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split a document into chunks."""
    text = doc.get("cleaned_text") or doc.get("text", "")

    # Short document, or no sentence boundaries found — single chunk.
    sentences = split_sentences(text) if len(text) > max_size else []
    if len(text) <= max_size or not sentences:
        return [_make_chunk(doc, 0, text)]

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(_make_chunk(doc, chunk_idx, chunk_text))
            chunk_idx += 1

            if overlap > 0:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(_make_chunk(doc, chunk_idx, chunk_text))

    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """Build chunks from all documents.

    Migration Step 6: `documents` is now required. The previous default
    (`None` -> `import_module("01_documents").documents`) reached back into a
    root pipeline script and triggered its import-time filesystem read of the
    hardcoded legacy `output/` path, which is not competition-portable. The
    CLI wrapper in 03_chunking.py preserves that fallback for its own callers.
    """
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
