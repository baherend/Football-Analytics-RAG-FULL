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
import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (NFC form)."""
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single space, strip ends."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_punctuation(text: str) -> str:
    """Normalize smart quotes and special punctuation."""
    replacements = {
        '‘': "'", '’': "'",  # smart single quotes
        '“': '"', '”': '"',  # smart double quotes
        '–': '-', '—': '-',  # en/em dashes
        '…': '...',               # ellipsis
        ' ': ' ',                 # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def remove_control_chars(text: str) -> str:
    """Remove control characters except newline and tab."""
    return ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != 'C' or ch in '\n\t'
    )


def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to text."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = remove_control_chars(text)
    text = clean_whitespace(text)
    return text


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
