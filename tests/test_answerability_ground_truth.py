"""Tests for the standalone answerability ground-truth dataset."""

from __future__ import annotations

import pytest

from src.context.answerability import assess_answerability
from src.evaluation.ground_truth.answerability import (
    ANSWERABILITY_GROUND_TRUTH,
    ANSWERABILITY_GROUND_TRUTH_METADATA,
    EXPECTED_ANSWERABILITY_CASE_IDS,
)
from src.evaluation.ground_truth.semantic import (
    SEMANTIC_GROUND_TRUTH,
    SEMANTIC_GROUND_TRUTH_METADATA,
    index_chunks_by_document_id,
    load_chunks,
)


@pytest.fixture(scope="module")
def chunks_by_doc():
    chunks = load_chunks(SEMANTIC_GROUND_TRUTH_METADATA["chunks_path"])
    return index_chunks_by_document_id(chunks)


def test_answerability_ground_truth_contract():
    assert ANSWERABILITY_GROUND_TRUTH_METADATA["expected_case_count"] == 6
    assert ANSWERABILITY_GROUND_TRUTH_METADATA["expected_status_counts"] == {
        "answerable": 2,
        "partially_answerable": 2,
        "unanswerable": 2,
    }

    assert len(ANSWERABILITY_GROUND_TRUTH) == 6
    assert {case["id"] for case in ANSWERABILITY_GROUND_TRUTH} == set(
        EXPECTED_ANSWERABILITY_CASE_IDS
    )

    retrieval_ids = {case["id"] for case in SEMANTIC_GROUND_TRUTH}
    answerability_ids = {case["id"] for case in ANSWERABILITY_GROUND_TRUTH}
    assert retrieval_ids.isdisjoint(answerability_ids)

    required_fields = {
        "id",
        "dataset_id",
        "query",
        "expected_answerability",
        "evidence_document_ids",
        "evidence_snippets",
        "notes",
    }

    status_counts = {
        "answerable": 0,
        "partially_answerable": 0,
        "unanswerable": 0,
    }

    for case in ANSWERABILITY_GROUND_TRUTH:
        assert required_fields <= set(case)
        assert case["dataset_id"] == ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"]
        assert case["query"].strip()
        assert case["expected_answerability"] in status_counts
        assert case["evidence_document_ids"]
        assert case["evidence_snippets"]
        assert case["notes"].strip()
        status_counts[case["expected_answerability"]] += 1

    assert status_counts == ANSWERABILITY_GROUND_TRUTH_METADATA[
        "expected_status_counts"
    ]


def test_answerability_evidence_exists_verbatim(chunks_by_doc):
    for case in ANSWERABILITY_GROUND_TRUTH:
        for document_id in case["evidence_document_ids"]:
            assert document_id in chunks_by_doc

        document_text = "\n".join(
            chunk.get("text", "")
            for document_id in case["evidence_document_ids"]
            for chunk in chunks_by_doc[document_id]
        )

        for snippet in case["evidence_snippets"]:
            assert snippet in document_text, (
                f"{case['id']}: evidence snippet not found verbatim: {snippet}"
            )


def test_answerability_ground_truth_matches_runtime(chunks_by_doc):
    entity_fields = ("team_name", "player_name", "home_team", "away_team")

    for case in ANSWERABILITY_GROUND_TRUTH:
        selected_chunks = []

        for index, snippet in enumerate(case["evidence_snippets"]):
            source_chunk = next(
                chunk
                for document_id in case["evidence_document_ids"]
                for chunk in chunks_by_doc[document_id]
                if snippet in chunk.get("text", "")
            )

            metadata = dict(source_chunk.get("metadata", {}))
            metadata["document_id"] = source_chunk.get("document_id")

            for field in entity_fields:
                value = source_chunk.get(field) or metadata.get(field)
                if value:
                    metadata[field] = value

            selected_chunks.append(
                {
                    "chunk_id": f"{case['id']}-evidence-{index}",
                    "text": snippet,
                    "metadata": metadata,
                }
            )

        assessment = assess_answerability(case["query"], selected_chunks)

        assert assessment.status == case["expected_answerability"], (
            f"{case['id']}: expected {case['expected_answerability']}, "
            f"got {assessment.status}; matched={assessment.matched_terms}; "
            f"missing={assessment.missing_terms}"
        )
