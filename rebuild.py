"""
rebuild.py — Regenerate all pipeline artifacts from raw data.

Usage:
    python rebuild.py          # Full rebuild
    python rebuild.py --quick  # Skip embeddings/ChromaDB (for testing)

Prerequisites:
    1. pip install -r requirements.txt
    2. Download StatsBomb open data to open-data-master/data/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        return False

    print(f"\n✓ Complete: {description}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild all pipeline artifacts")
    parser.add_argument("--quick", action="store_true",
                        help="Skip embeddings and ChromaDB (faster for testing)")
    args = parser.parse_args()

    # Check prerequisites
    if not Path("open-data-master/data").exists():
        print("Error: open-data-master/data/ not found.")
        print("Download StatsBomb open data first.")
        return 1

    steps = [
        ("extract.py",              "Phase 1: Extract structured facts"),
        ("generate_documents.py",   "Phase 2: Generate documents"),
        ("02_preprocessing.py",     "Phase 3: Preprocess text"),
        ("03_chunking.py",          "Phase 3: Chunk documents"),
        ("04_representation.py",    "Phase 3: Build BM25/TF-IDF indices"),
    ]

    if not args.quick:
        steps.extend([
            ("05_embeddings.py",        "Phase 4: Generate embeddings"),
            ("06_create_chroma_store.py", "Phase 4: Create ChromaDB"),
        ])

    for script, description in steps:
        if not run(script, description):
            return 1

    print(f"\n{'='*60}")
    print("  REBUILD COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated artifacts:")
    print("  output/match_facts.json    — Structured facts")
    print("  output/documents.json      — Rendered documents")
    print("  output/processed_documents.json — Cleaned text")
    print("  output/chunks.json         — Chunked documents")
    print("  output/indices/            — BM25/TF-IDF indices")

    if not args.quick:
        print("  output/embeddings/         — Sentence embeddings")
        print("  output/chroma_db/          — ChromaDB vector store")

    print("\nTo test the system:")
    print("  python chat.py 'How many goals did Messi score?'")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
