"""
02_preprocessing.py — Stage 2: Text Preprocessing

Cleans and normalizes document text for embedding and retrieval.
Applies Unicode normalization, punctuation normalization, control character
removal, and whitespace collapsing.

Reads the selected competition/season's documents.json and writes its
processed_documents.json (see src/artifacts.py's resolve_output_dir()).
The WC2022 default (competition_id=43, season_id=106) reads/writes the
legacy flat output/ layout; any other competition/season uses its own
namespaced output/competitions/<id>/<id>/ directory. Pass --namespaced to
explicitly use the namespaced layout for a WC2022 build.

Usage:
    python 02_preprocessing.py [--competition-id ID] [--season-id ID] [--namespaced] [--quiet]
"""

from __future__ import annotations

import argparse
import json

# Migration Step 6: the text-normalization logic moved to
# src/knowledge/preprocessing.py. This script keeps its CLI (argument parsing,
# competition/season path resolution, artifact I/O) and re-exports the moved
# functions so existing importers and tests are unaffected.
from src.knowledge.preprocessing import (  # noqa: F401  (compatibility re-exports)
    clean_whitespace,
    normalize_punctuation,
    normalize_unicode,
    preprocess_text,
    remove_control_chars,
)


def main() -> int:
    from src.artifacts import resolve_output_dir
    from src.extraction.match_facts import COMPETITION_ID, SEASON_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", type=int, default=COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=SEASON_ID)
    parser.add_argument("--namespaced", action="store_true",
                        help="Use output/competitions/<id>/<id>/ even for the WC2022 default")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.competition_id, args.season_id,
                                    legacy_default=not args.namespaced)
    documents_path = output_dir / "documents.json"
    output_path = output_dir / "processed_documents.json"

    if not documents_path.exists():
        print(f"Error: {documents_path} not found. Run generate_documents.py first.")
        return 1

    if not args.quiet:
        print(f"Loading documents from {documents_path}")
    documents = json.loads(documents_path.read_text(encoding="utf-8"))

    for doc in documents:
        doc["cleaned_text"] = preprocess_text(doc.get("text", ""))

    output_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote {len(documents)} processed documents to {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
