from __future__ import annotations

from src.context.answerability import assess_answerability


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "document_id": "DOC-1",
            "level": "team",
            "team_name": "Example FC",
        },
    }


def test_multi_facet_evidence_is_answerable():
    assessment = assess_answerability(
        query="What were Example FC's passing patterns and common formations?",
        chunks=[
            _chunk(
                "DOC-1-part-1",
                "Their play patterns included regular play, throw ins, and free kicks.",
            ),
            _chunk(
                "DOC-1-part-2",
                "Most passes were open-play passes and the common formations were 433 and 4231.",
            ),
        ],
    )

    assert assessment.status == "answerable"


def test_missing_one_query_facet_is_partially_answerable():
    assessment = assess_answerability(
        query="What were Example FC's passing patterns and common formations?",
        chunks=[
            _chunk(
                "DOC-1-part-1",
                "The common formations were 433 and 4231.",
            ),
        ],
    )

    assert assessment.status == "partially_answerable"


def test_entity_mention_without_answer_evidence_is_unanswerable():
    assessment = assess_answerability(
        query="What formations did Alpha United commonly use?",
        chunks=[
            _chunk(
                "DOC-1-part-0",
                "Example FC played seven matches in the tournament.",
            ),
        ],
    )

    assert assessment.status == "unanswerable"


def test_empty_chunks_are_unanswerable():
    assessment = assess_answerability(
        query="How did Example FC build attacks?",
        chunks=[],
    )

    assert assessment.status == "unanswerable"

def test_matching_facets_for_different_entity_are_unanswerable():
    assessment = assess_answerability(
        query="What formations did Alpha United commonly use?",
        chunks=[
            {
                "chunk_id": "DOC-2-part-1",
                "text": "The common formations were 433 and 4231.",
                "metadata": {
                    "document_id": "DOC-2",
                    "level": "team",
                    "team_name": "Beta City",
                },
            },
        ],
    )

    assert assessment.status == "unanswerable"

def test_semantic_route_exposes_answerability_assessment(monkeypatch):
    """
    Structural Cleanup Phase A: execute_route() now lives in
    src.query.router (moved from 06_retrieve_context.py) -- hybrid_search()
    is patched there directly, since execute_route() resolves it via that
    module's own globals, not the compatibility wrapper's.
    """
    from importlib import import_module

    retrieval = import_module("src.query.router")

    query = "What formations did Alpha United commonly use?"
    chunks = [
        {
            "chunk_id": "DOC-1-part-0",
            "text": "Example FC played seven matches in the tournament.",
            "metadata": {
                "document_id": "DOC-1",
                "level": "team",
                "team_name": "Example FC",
            },
        },
    ]

    monkeypatch.setattr(
        retrieval,
        "hybrid_search",
        lambda query, k=3, artifact_paths=None: chunks,
    )

    result = retrieval.execute_route(
        retrieval.Route(
            path="semantic",
            confidence=0.9,
            reason="test",
            semantic_query=query,
        ),
        semantic_k=1,
        original_query=query,
    )

    assert result.answerability.status == "unanswerable"


def test_pass_and_passes_normalize_consistently():
    assessment = assess_answerability(
        query="What pass patterns did Example FC use?",
        chunks=[
            _chunk(
                "DOC-1-part-1",
                "Their passes followed regular play patterns.",
            ),
        ],
    )

    assert assessment.status == "answerable"
