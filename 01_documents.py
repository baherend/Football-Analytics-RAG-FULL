"""
01_documents.py — Phase 1: Document Generation Pipeline

Thin CLI wrapper that orchestrates the extraction and document generation
pipeline using the canonical implementations in src/.

Pipeline:
    1. src/extraction/extract.py  →  output/match_facts.json
    2. generate_documents.py      →  output/documents.json
    3. Regression tests (§8)      →  verify correctness

All extraction logic lives in src/extraction/match_facts.py.
All rendering logic lives in src/rendering/render.py.
This file exists to satisfy the numbered-file convention and to run
the full pipeline end-to-end with regression verification.

Usage:
    python 01_documents.py
    python 01_documents.py --skip-tests
    python 01_documents.py --tests-only
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_extraction(verbose: bool = True) -> dict:
    """Run Phase 1 extraction: StatsBomb JSON → match_facts.json."""
    from src.extraction.match_facts import extract_all, persist, DATA_ROOT

    if verbose:
        print(f"Extracting structured facts from {DATA_ROOT.resolve()}")

    result = extract_all(DATA_ROOT, verbose=verbose)
    diag = result["diagnostics"]

    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Matches processed:     {diag['matches_processed']}")
        print(f"  Player-match facts:    {diag['total_player_facts']}")
        print(f"  Match facts:           {len(result['match_facts'])}")
        print(f"  Team-match facts:      {diag['total_team_facts']}")
        print(f"  Card parity failures:  {diag['card_count_mismatches'] or 'none'}")

    output_path = persist(result)
    if verbose:
        print(f"\nPersisted to {output_path.resolve()}")

    return result


def run_document_generation(verbose: bool = True) -> list:
    """Run Phase 2 rendering: match_facts.json → documents.json."""
    from src.rendering.render import render_all, persist

    facts_path = Path("output/match_facts.json")
    output_path = Path("output/documents.json")

    if not facts_path.exists():
        print(f"Error: {facts_path} not found. Run extraction first.")
        sys.exit(1)

    if verbose:
        print(f"Loading structured facts from {facts_path}")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))

    if verbose:
        print(f"Rendering documents...")
    documents = render_all(facts)

    if verbose:
        counts = Counter(d.level for d in documents)
        print(f"\nDocument counts by level")
        for level in ("1", "2", "3", "4", "team"):
            label = {"1": "Level 1  Match Summary", "2": "Level 2  Key Events",
                     "3": "Level 3  Player / match", "4": "Level 4  Player / tournament",
                     "team": "Team-level Analysis"}[level]
            print(f"  {label:<32} {counts.get(level, 0)}")
        print(f"  {'TOTAL':<32} {len(documents)}")

    persist(documents, output_path)
    if verbose:
        print(f"\nWrote {len(documents)} documents to {output_path.resolve()}")

    return documents


def run_regression_tests() -> int:
    """Run §8 regression tests from tests/test_extraction.py."""
    from tests.test_extraction import run_all_tests

    print("\n§8 regression tests")
    failures = run_all_tests()
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    skip_tests = "--skip-tests" in sys.argv
    tests_only = "--tests-only" in sys.argv

    if not tests_only:
        run_extraction(verbose=True)
        run_document_generation(verbose=True)

    if not skip_tests:
        failures = run_regression_tests()
        if failures:
            print(f"\n{failures} regression test(s) failed.")
            return 1
        print("  all regression tests passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
