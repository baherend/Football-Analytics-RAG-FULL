"""
test_multilingual_retrieval_cases.py -- Multilingual Retrieval Baseline:
proves the EN/MSA/EGY query variants are correctly bound to the existing
WC2022 Semantic Ground Truth cases (src/evaluation/ground_truth/semantic.py) before
any evaluation is run against them.

This is a variant-integrity test, not a retrieval-quality test. It does
not call any retrieval function.
"""

from __future__ import annotations

from src.evaluation.ground_truth.multilingual import (
    LANGUAGES,
    MULTILINGUAL_QUERY_VARIANTS,
    MultilingualQueryVariant,
    build_ground_truth_bundle,
    build_translated_cases,
)
from src.evaluation.ground_truth.semantic import SEMANTIC_GROUND_TRUTH, EXPECTED_CASE_IDS


def test_multilingual_variants_reference_existing_semantic_cases():
    known_case_ids = {c["id"] for c in SEMANTIC_GROUND_TRUTH}
    assert known_case_ids == set(EXPECTED_CASE_IDS), (
        "Sanity check: SEMANTIC_GROUND_TRUTH case IDs no longer match "
        "EXPECTED_CASE_IDS -- the multilingual benchmark's coverage "
        "assumptions below would be checked against the wrong set."
    )

    # 1. Every variant's source_case_id must reference a real semantic case.
    unknown_sources = sorted({
        v.source_case_id for v in MULTILINGUAL_QUERY_VARIANTS
        if v.source_case_id not in known_case_ids
    })
    assert not unknown_sources, (
        f"Multilingual variants reference unknown source_case_id(s): {unknown_sources}"
    )

    # 2. Exactly one variant per (case_id, language) -- no duplicates, no gaps.
    seen: dict[tuple[str, str], int] = {}
    for v in MULTILINGUAL_QUERY_VARIANTS:
        key = (v.source_case_id, v.language)
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(k for k, count in seen.items() if count > 1)
    assert not duplicates, f"Duplicate (case_id, language) variant pairs: {duplicates}"

    for case_id in known_case_ids:
        for language in LANGUAGES:
            assert (case_id, language) in seen, (
                f"Missing multilingual variant: case_id={case_id!r}, language={language!r}"
            )

    # 3. Full coverage: 24 English semantic cases x 3 languages = 72 variants.
    expected_total = len(known_case_ids) * len(LANGUAGES)
    assert len(MULTILINGUAL_QUERY_VARIANTS) == expected_total, (
        f"Expected {expected_total} multilingual variants "
        f"({len(known_case_ids)} cases x {len(LANGUAGES)} languages), "
        f"found {len(MULTILINGUAL_QUERY_VARIANTS)}"
    )

    # 4. No empty/whitespace-only translated query.
    empty_queries = sorted(
        f"{v.source_case_id}/{v.language}"
        for v in MULTILINGUAL_QUERY_VARIANTS
        if not v.query.strip()
    )
    assert not empty_queries, f"Empty translated query for: {empty_queries}"

    # 5. The English variant must be the ORIGINAL query, unchanged.
    english_query_by_case = {c["id"]: c["query"] for c in SEMANTIC_GROUND_TRUTH}
    mismatched_english = sorted(
        v.source_case_id
        for v in MULTILINGUAL_QUERY_VARIANTS
        if v.language == "en" and v.query != english_query_by_case.get(v.source_case_id)
    )
    assert not mismatched_english, (
        f"English variant query does not match the original semantic case "
        f"query verbatim for: {mismatched_english}"
    )


def test_build_translated_cases_reuses_english_relevance_truth():
    """
    build_translated_cases() must swap ONLY the query field -- relevance
    truth (relevant_document_ids, case_group, primary_level, etc.) must be
    byte-identical to the English source case, for every language.
    """
    english_by_id = {c["id"]: c for c in SEMANTIC_GROUND_TRUTH}

    for language in LANGUAGES:
        translated = build_translated_cases(language)
        assert len(translated) == len(SEMANTIC_GROUND_TRUTH), (
            f"build_translated_cases({language!r}) returned {len(translated)} cases, "
            f"expected {len(SEMANTIC_GROUND_TRUTH)}"
        )
        for case in translated:
            source = english_by_id[case["id"]]
            for field in (
                "case_group", "expected_route", "answerability", "primary_level",
                "acceptable_levels", "relevant_document_ids",
                "optional_relevant_document_ids", "dataset_id",
            ):
                assert case[field] == source[field], (
                    f"[{language}/{case['id']}] field {field!r} diverged from the "
                    "English source case -- relevance truth must never be "
                    "re-authored per language variant."
                )


def test_ground_truth_bundle_passes_existing_evaluator_validation():
    """
    build_ground_truth_bundle() must produce a bundle the existing
    retrieval evaluator's own validate_ground_truth_and_chunks() accepts
    unmodified -- proving the evaluator-reuse seam actually works end to
    end (structure + chunks-hash integrity), without running retrieval.
    """
    from src.evaluation.retrieval_evaluator import validate_ground_truth_and_chunks

    for language in LANGUAGES:
        bundle = build_ground_truth_bundle(language)
        metadata, cases, document_levels = validate_ground_truth_and_chunks(
            chunks_path="output/chunks.json", ground_truth=bundle,
        )
        assert len(cases) == len(SEMANTIC_GROUND_TRUTH)
        assert document_levels, "document_levels must be resolved from the real chunks.json"
