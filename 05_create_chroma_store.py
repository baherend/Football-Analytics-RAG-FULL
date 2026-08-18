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
from src.embedding_config import resolve_embedding_config
from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

# Migration Step 6: vector-store construction moved to
# src/knowledge/indexing/vector_store.py. This script keeps its CLI (argument
# parsing, competition/season path resolution, artifact I/O) and re-exports
# the moved symbols so existing importers and tests are unaffected.
from src.knowledge.indexing.vector_store import (  # noqa: F401  (compatibility re-exports)
    COLLECTION_NAME,
    DB_PATH,
    MODEL_NAME,
    create_vector_store,
    get_collection,
)


def main() -> int:
    from src.embedding_config import DEFAULT_EMBEDDING_MODEL_ID, EMBEDDING_MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--embedding-model", default=None, choices=sorted(EMBEDDING_MODELS),
                        help=f"Registered embedding model id to build the index with "
                             f"(default: {DEFAULT_EMBEDDING_MODEL_ID}). A non-default model "
                             f"for the WC2022 default forces the namespaced output directory, "
                             f"same as --namespaced, so it can never write into the production "
                             f"MiniLM Chroma directory.")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced,
                                    embedding_model_id=args.embedding_model)
    collection_name = resolve_chroma_collection_name(
        args.competition_id,
        args.season_id,
        legacy_name=COLLECTION_NAME,
        legacy_default=not args.namespaced,
        embedding_model_id=args.embedding_model,
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
        embedding_model_id=args.embedding_model,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
