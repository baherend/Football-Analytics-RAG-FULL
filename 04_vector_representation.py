"""
04_vector_representation.py — Stage 4: Vector Representation

Builds sparse (BM25/TF-IDF) and dense (sentence embeddings) representations
for hybrid retrieval.

Input: chunks from 03_chunking.py
Output: BM25 index, TF-IDF matrix, sentence embeddings

The hybrid_search() function combines BM25 lexical matching with dense
embedding similarity using min-max normalization and weighted fusion.

Run directly, reads the selected competition/season's chunks.json and
writes its indices/bm25.pkl and embeddings/embeddings.npy (see
src/artifacts.py's resolve_output_dir()). The WC2022 default
(competition_id=43, season_id=106) reads/writes the legacy flat output/
layout; any other competition/season uses its own namespaced
output/competitions/<id>/<id>/ directory. Pass --namespaced to explicitly
use the namespaced layout for a WC2022 build.

Usage:
    python 04_vector_representation.py [--competition-id ID] [--season-id ID] [--namespaced] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np

# Migration Step 6: index construction moved to src/knowledge/indexing/. This
# script keeps its CLI (argument parsing, competition/season path resolution,
# artifact I/O) and re-exports the moved symbols so existing importers and
# tests are unaffected.
#
# Also removed here: a duplicate query-time retrieval implementation
# (`_ensure_loaded`, `hybrid_search`, `min_max_normalize`, plus lazy module
# globals). It was a knowledge -> rag responsibility violation, it hardcoded
# the legacy `output/` paths so it was NOT competition-portable, it read the
# dead `embeddings.npy`, and it was measurably dead: zero references outside
# this file, with tests touching only `main()`. Query-time hybrid retrieval is
# owned by src/retrieval/. See PROJECT_MEMORY.md.
from src.knowledge.indexing.bm25 import (  # noqa: F401  (compatibility re-exports)
    _build_bm25,
    simple_tokenize,
)
from src.knowledge.indexing.embeddings import (  # noqa: F401
    MODEL_NAME,
    _build_embeddings,
)

ALPHA = 0.6  # weight for dense scores (1-ALPHA for BM25); retained for callers


def main() -> int:
    from src.artifacts import resolve_output_dir
    from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build lexical indices only; skip sentence embeddings",
    )
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    chunks_path = output_dir / "chunks.json"

    if not chunks_path.exists():
        print(f"Error: {chunks_path} not found. Run 03_chunking.py first.")
        return 1

    if not args.quiet:
        print(f"Loading chunks from {chunks_path}")
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not args.quiet:
        print(f"Building BM25 index for {len(chunks_data)} chunks...")
    bm25_index = _build_bm25(chunks_data)

    indices_dir = output_dir / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    with open(indices_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    if not args.quiet:
        print(f"Wrote {indices_dir / 'bm25.pkl'}")

    if not args.skip_embeddings:
        if not args.quiet:
            print(f"Generating embeddings ({MODEL_NAME})...")
        embeddings = _build_embeddings(chunks_data)

        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_dir / "embeddings.npy", embeddings)

        if not args.quiet:
            print(f"Wrote {embeddings_dir / 'embeddings.npy'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
