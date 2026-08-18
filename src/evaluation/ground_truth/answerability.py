"""Standalone ground truth for deterministic answerability evaluation."""

from __future__ import annotations

from src.evaluation.ground_truth.semantic import SEMANTIC_GROUND_TRUTH_METADATA


ANSWERABILITY_GROUND_TRUTH_METADATA: dict = {
    "schema_version": "1.0",
    "dataset_id": "statsbomb-fifa-world-cup-2022-answerability",
    "source_dataset_id": SEMANTIC_GROUND_TRUTH_METADATA["dataset_id"],
    "chunks_path": SEMANTIC_GROUND_TRUTH_METADATA["chunks_path"],
    "chunks_sha256": SEMANTIC_GROUND_TRUTH_METADATA["chunks_sha256"],
    "expected_case_count": 6,
    "expected_status_counts": {
        "answerable": 2,
        "partially_answerable": 2,
        "unanswerable": 2,
    },
    "notes": (
        "Standalone evidence-set cases for answerability evaluation. "
        "These cases are excluded from retrieval-ranking evaluation."
    ),
}


EXPECTED_ANSWERABILITY_CASE_IDS: tuple[str, ...] = (
    "answerability-answerable-team-01",
    "answerability-answerable-team-02",
    "answerability-partial-team-01",
    "answerability-partial-team-02",
    "answerability-unanswerable-team-01",
    "answerability-unanswerable-team-02",
)


_FRANCE_PLAY_PATTERN = (
    "Their play patterns were dominated by regular play 38.8%, "
    "from throw in 24.2%, from free kick 16.1%, and from goal kick 10.1%."
)

_FRANCE_FORMATIONS = (
    "Their most common shapes, counted across Starting XI and Tactical Shift "
    "events, were 4231 (10 formation records), 433 (4 formation records), "
    "and 442 (2 formation records)."
)

_FRANCE_PASSES = (
    "Of 3926 passes, 83.2% were standard open-play passes and 16.8% came "
    "from set pieces or restarts, chiefly recovery 7.2%, throw-in 3.5%, "
    "free kick 2.4%, and goal kick 1.7%."
)

_ARGENTINA_PLAY_PATTERN = (
    "Their play patterns were dominated by regular play 40.0%, "
    "from throw in 23.3%, from free kick 20.3%, and from goal kick 5.5%."
)

_ARGENTINA_FORMATIONS = (
    "Their most common shapes, counted across Starting XI and Tactical Shift "
    "events, were 352 (8 formation records), 433 (8 formation records), "
    "and 442 (5 formation records)."
)


ANSWERABILITY_GROUND_TRUTH: tuple[dict, ...] = (
    {
        "id": "answerability-answerable-team-01",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were France's pass patterns and most common formations?",
        "expected_answerability": "answerable",
        "evidence_document_ids": ["TEAM-771"],
        "evidence_snippets": [
            _FRANCE_PLAY_PATTERN,
            _FRANCE_FORMATIONS,
            _FRANCE_PASSES,
        ],
        "notes": "Evidence covers pass, pattern, and formation terms for France.",
    },
    {
        "id": "answerability-answerable-team-02",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were Argentina's play patterns and most common formations?",
        "expected_answerability": "answerable",
        "evidence_document_ids": ["TEAM-779"],
        "evidence_snippets": [
            _ARGENTINA_PLAY_PATTERN,
            _ARGENTINA_FORMATIONS,
        ],
        "notes": "Evidence covers both requested Argentina team-style facets.",
    },
    {
        "id": "answerability-partial-team-01",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were France's pass patterns and most common formations?",
        "expected_answerability": "partially_answerable",
        "evidence_document_ids": ["TEAM-771"],
        "evidence_snippets": [_FRANCE_FORMATIONS],
        "notes": "Formation evidence exists, but pass-pattern evidence is absent.",
    },
    {
        "id": "answerability-partial-team-02",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were Argentina's play patterns and most common formations?",
        "expected_answerability": "partially_answerable",
        "evidence_document_ids": ["TEAM-779"],
        "evidence_snippets": [_ARGENTINA_PLAY_PATTERN],
        "notes": "Play-pattern evidence exists, but formation evidence is absent.",
    },
    {
        "id": "answerability-unanswerable-team-01",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were France's play patterns and most common formations?",
        "expected_answerability": "unanswerable",
        "evidence_document_ids": ["TEAM-779"],
        "evidence_snippets": [
            _ARGENTINA_PLAY_PATTERN,
            _ARGENTINA_FORMATIONS,
        ],
        "notes": "Evidence concerns Argentina rather than the requested France entity.",
    },
    {
        "id": "answerability-unanswerable-team-02",
        "dataset_id": ANSWERABILITY_GROUND_TRUTH_METADATA["dataset_id"],
        "query": "What were Argentina's play patterns and most common formations?",
        "expected_answerability": "unanswerable",
        "evidence_document_ids": ["TEAM-771"],
        "evidence_snippets": [
            _FRANCE_PLAY_PATTERN,
            _FRANCE_FORMATIONS,
        ],
        "notes": "Evidence concerns France rather than the requested Argentina entity.",
    },
)
