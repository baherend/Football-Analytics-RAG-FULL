"""
Faithfulness / Grounded Generation - Step 1 baseline.

06_retrieve_context.py's execute_route() already computes a deterministic
AnswerabilityAssessment (src/retrieval/answerability.py) for every semantic
or hybrid route and attaches it to RoutedResult.answerability. Nothing in
the generation path (chat.py::process_query, 07_prompting.py::answer_question)
reads that field before calling the LLM -- so a query the router itself has
already flagged "unanswerable" still reaches generation.

This test reproduces that gap deterministically, without a live LLM call:
it stubs execute_route() to return the same "unanswerable" assessment the
real router would produce for an off-dataset entity, and stubs
generate_answer() only to observe whether it gets invoked.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from src.artifacts import ArtifactPaths
from src.conversation_memory import ConversationMemory
from src.retrieval.answerability import AnswerabilityAssessment


def test_chat_process_query_stops_before_generation_when_context_is_unanswerable(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.mode = "hybrid"

    unanswerable = AnswerabilityAssessment(
        status="unanswerable",
        matched_terms=(),
        missing_terms=("alpha", "united", "formations"),
    )

    monkeypatch.setattr(
        chat.router_mod,
        "route_query",
        lambda q, artifact_paths=None: SimpleNamespace(
            path="semantic", confidence=1.0, reason="test", semantic_query=q,
        ),
    )
    monkeypatch.setattr(
        chat.router_mod,
        "execute_route",
        lambda route, semantic_k=5, artifact_paths=None: SimpleNamespace(
            structured_result=None,
            # Evidence was retrieved, but about an unrelated entity -- the
            # same shape the router's own answerability check flags as
            # "unanswerable" (see tests/test_answerability.py).
            semantic_chunks=[{
                "chunk_id": "DOC-1-part-0",
                "text": "Example FC played seven matches in the tournament.",
                "metadata": {"document_id": "DOC-1", "level": "team", "team_name": "Example FC"},
            }],
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        chat.prompting_mod,
        "generate_answer",
        lambda *a, **kw: (generation_calls.append(1), "fabricated answer")[1],
        raising=False,
    )

    chat.process_query("What formations did Alpha United commonly use?")

    assert not generation_calls, (
        "chat.process_query() invoked the LLM even though execute_route()'s "
        "own answerability assessment was 'unanswerable' -- the router "
        "computes this signal but the generation path never consults it "
        "before calling generate_answer(), so an unsupported query can "
        "still produce a fabricated-looking answer."
    )


def test_answer_question_stops_before_generation_when_context_is_unanswerable(monkeypatch):
    prompting = import_module("07_prompting")
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    unanswerable = AnswerabilityAssessment(
        status="unanswerable",
        matched_terms=(),
        missing_terms=("alpha", "united", "formations"),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=None,
            semantic_chunks=[{
                "chunk_id": "DOC-1-part-0",
                "text": "Example FC played seven matches in the tournament.",
                "metadata": {
                    "document_id": "DOC-1",
                    "level": "team",
                    "team_name": "Example FC",
                },
            }],
            context="Example FC played seven matches in the tournament.",
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: (
            generation_calls.append(1),
            "fabricated answer",
        )[1],
    )

    answer, _ = prompting.answer_question(
        "What formations did Alpha United commonly use?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert not generation_calls, (
        "answer_question() invoked the LLM even though the routed evidence "
        "was explicitly assessed as unanswerable."
    )


def test_process_query_does_not_block_valid_structured_result_when_semantic_unanswerable(monkeypatch):
    """
    Policy boundary: semantic-only answerability must never become a global
    answerability verdict. When a usable structured result exists (status
    "resolved"/"partial" with an explanation), generation must proceed even
    though the semantic evidence alone was flagged "unanswerable".
    """
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.mode = "hybrid"

    unanswerable = AnswerabilityAssessment(
        status="unanswerable",
        matched_terms=(),
        missing_terms=("alpha", "united"),
    )

    monkeypatch.setattr(
        chat.router_mod,
        "route_query",
        lambda q, artifact_paths=None: SimpleNamespace(
            path="hybrid", confidence=0.9, reason="test", semantic_query=q,
        ),
    )
    monkeypatch.setattr(
        chat.router_mod,
        "execute_route",
        lambda route, semantic_k=5, artifact_paths=None: SimpleNamespace(
            structured_result=SimpleNamespace(
                status="resolved",
                explanation="Messi scored 5 goals.",
                aggregated_value=5,
                query=SimpleNamespace(metric="goals"),
            ),
            # Semantic evidence for an unrelated entity is still flagged
            # "unanswerable" by the router -- must not veto the structured
            # answer above.
            semantic_chunks=[{
                "chunk_id": "DOC-1-part-0",
                "text": "Example FC played seven matches in the tournament.",
                "metadata": {"document_id": "DOC-1", "level": "team", "team_name": "Example FC"},
            }],
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        chat.prompting_mod,
        "generate_answer",
        lambda *a, **kw: (generation_calls.append(1), "Messi scored 5 goals.")[1],
        raising=False,
    )

    chat.process_query("How many goals did Messi score?")

    assert generation_calls, (
        "process_query() blocked generation for a query with a valid, usable "
        "structured result -- semantic-only answerability must not override "
        "an authoritative structured answer."
    )


def test_answer_question_unanswerable_refuses_even_without_api_key(monkeypatch):
    prompting = import_module("07_prompting")
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    unanswerable = AnswerabilityAssessment(
        status="unanswerable",
        matched_terms=(),
        missing_terms=("alpha", "united", "formations"),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=None,
            semantic_chunks=[],
            context="No relevant context found.",
            answerability=unanswerable,
        ),
    )
    monkeypatch.setattr(prompting, "GROQ_API_KEY", None)

    generation_calls = []
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: (
            generation_calls.append(1),
            "should not be generated",
        )[1],
    )

    answer, _ = prompting.answer_question(
        "What formations did Alpha United commonly use?",
        api_key=None,
        artifact_paths=selected,
    )

    assert answer == prompting.INSUFFICIENT_CONTEXT_MESSAGE
    assert not generation_calls
