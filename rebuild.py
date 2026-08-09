"""
rebuild.py — Regenerate all pipeline artifacts from raw data.

Usage:
    python rebuild.py                                   # WC2022 default
    python rebuild.py --quick                            # Skip embeddings/ChromaDB
    python rebuild.py --competition-id 2 --season-id 27   # Another competition
    python rebuild.py --namespaced                        # Explicit WC2022 namespace

The WC2022 default (competition_id=43, season_id=106) writes to the legacy
flat output/ layout; any other competition/season writes to its own
namespaced output/competitions/<id>/<id>/ directory (see
src/artifacts.py's resolve_output_dir()) -- printed artifact locations
below reflect whichever was selected. Pass --namespaced to explicitly
build WC2022 under output/competitions/43/106/ instead of the legacy flat
layout.

Every artifact-producing phase (extraction, rendering, preprocessing,
chunking, vector representation, Chroma) receives the same
competition/season identity and reads/writes only that dataset's
namespace -- no phase falls back to another dataset's artifacts.

Prerequisites:
    1. pip install -r requirements.txt
    2. Download StatsBomb open data to open-data-master/data/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.artifacts import resolve_output_dir
from src.extraction.match_facts import COMPETITION_ID, SEASON_ID


def run(command: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(command, capture_output=False)

    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False

    print(f"\n✓ Complete: {description}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild all pipeline artifacts")
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quick", action="store_true",
                        help="Skip embeddings and ChromaDB (faster for testing)")
    args = parser.parse_args()

    # Check prerequisites
    if not Path("open-data-master/data").exists():
        print("Error: open-data-master/data/ not found.")
        print("Download StatsBomb open data first.")
        return 1

    identity_args = ["--competition-id", str(args.competition_id), "--season-id", str(args.season_id)]
    if args.namespaced:
        identity_args.append("--namespaced")

    steps: list[tuple[list[str], str]] = [
        ([sys.executable, "-m", "src.extraction.extract", *identity_args],
         "Phase 1: Extract structured facts"),
        ([sys.executable, "generate_documents.py", *identity_args, "--quiet"],
         "Phase 2: Render documents"),
        ([sys.executable, "02_preprocessing.py", *identity_args, "--quiet"],
         "Phase 3: Preprocess text"),
        ([sys.executable, "03_chunking.py", *identity_args, "--quiet"],
         "Phase 3: Chunk documents"),
        ([sys.executable, "04_vector_representation.py", *identity_args, "--quiet"],
         "Phase 4: Build all vector representations"),
    ]

    if not args.quick:
        steps.append(
            ([sys.executable, "05_create_chroma_store.py", *identity_args],
             "Phase 4: Create ChromaDB store")
        )

    for command, description in steps:
        if not run(command, description):
            return 1

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)

    print(f"\n{'='*60}")
    print("  REBUILD COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated artifacts under {output_dir.resolve()}:")
    print(f"  {output_dir / 'match_facts.json'}         — Structured facts")
    print(f"  {output_dir / 'documents.json'}           — Rendered documents")
    print(f"  {output_dir / 'processed_documents.json'} — Cleaned text")
    print(f"  {output_dir / 'chunks.json'}               — Chunked documents")
    print(f"  {output_dir / 'indices'}                   — BM25/TF-IDF indices")

    if not args.quick:
        print(f"  {output_dir / 'embeddings'}                — Sentence embeddings")
        print(f"  {output_dir / 'chroma_db'}                 — ChromaDB vector store")

    print("\nTo test the system:")
    print("  python chat.py 'How many goals did Messi score?'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
