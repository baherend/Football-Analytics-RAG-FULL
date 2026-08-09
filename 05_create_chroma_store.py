"""
05_create_chroma_store.py — Stage 5: ChromaDB Vector Store

Creates or loads a persistent ChromaDB collection for semantic retrieval.
Uses sentence-transformers embeddings (all-MiniLM-L6-v2).

Input: chunks from 03_chunking.py
Output: ChromaDB persistent store at output/chroma_db/

The WC2022 default (competition_id=43, season_id=106) reads chunks from
and persists Chroma to the legacy flat output/ layout; any other
competition/season uses its own namespaced
output/competitions/<id>/<id>/chroma_db/ directory (see
src/artifacts.py's resolve_output_dir()). Pass --namespaced to explicitly
use output/competitions/43/106/ for a WC2022 build instead of the legacy
flat layout. The Chroma COLLECTION NAME itself stays the shared default
for now -- collection-name namespacing is deferred to a later batch.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import chromadb

from src.artifacts import resolve_output_dir, resolve_chroma_collection_name
from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    collection_name = resolve_chroma_collection_name(
        args.competition_id,
        args.season_id,
        legacy_name=COLLECTION_NAME,
        legacy_default=not args.namespaced,
    )
    chunks_path = output_dir / "chunks.json"

    # Explicit failure rather than letting create_vector_store()'s own
    # chunks=None fallback silently read the legacy output/chunks.json --
    # that would be exactly the cross-competition fallback this batch must
    # prevent for any non-WC2022 dataset.
    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run the chunking step for "
              f"competition_id={args.competition_id}, season_id={args.season_id} first.")
        return 1

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    create_vector_store(
        chunks=chunks,
        persist_dir=output_dir / "chroma_db",
        collection_name=collection_name,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
