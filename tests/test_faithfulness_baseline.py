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

prompting = import_module("07_prompting")


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


def test_format_context_for_prompt_distinguishes_chunks_with_identical_display_metadata():
    """
    Evidence-attribution gap: format_context_for_prompt() (chat.py's semantic
    evidence formatter, used for the "[Source N]" blocks the SYSTEM_PROMPT
    tells the LLM to cite) currently keys each header only on level/player/
    team/match_id. Two different retrieved chunks that share all four --
    e.g. two distinct excerpts about the same player from the same match,
    a realistic shape once sibling-expansion/comparison-boost add related
    chunks -- render identical header text except for the "[Source N]"
    position count, so a citation can't be traced back to which underlying
    chunk actually supported it. The stable retrieval identity (chunk_id)
    is available on every chunk but is never included.

    This also pins ordering: [Source 1] must correspond to the first chunk
    and [Source 2] to the second, matching the order answer_question()
    returns as `sources`.
    """
    chunk_a = {
        "chunk_id": "DOC-1-chunk-0",
        "text": "Messi opened the scoring with a low finish from the edge of the box.",
        "metadata": {
            "document_id": "DOC-1",
            "level": "4",
            "player_name": "Lionel Messi",
            "team_name": "Argentina",
            "match_id": "8658",
        },
    }
    chunk_b = {
        "chunk_id": "DOC-2-chunk-0",
        "text": "Messi later converted a penalty to double Argentina's lead.",
        "metadata": {
            "document_id": "DOC-2",
            "level": "4",
            "player_name": "Lionel Messi",
            "team_name": "Argentina",
            "match_id": "8658",
        },
    }

    formatted = prompting.format_context_for_prompt([chunk_a, chunk_b])
    lines = formatted.splitlines()

    source_1_header = next(line for line in lines if line.startswith("[Source 1"))
    source_2_header = next(line for line in lines if line.startswith("[Source 2"))

    assert chunk_a["chunk_id"] in source_1_header
    assert chunk_b["chunk_id"] in source_2_header
    assert chunk_a["chunk_id"] not in source_2_header
    assert chunk_b["chunk_id"] not in source_1_header


def test_retrieval_build_context_uses_source_labels_with_chunk_identity():
    retrieval = import_module("06_retrieve_context")

    chunks = [
        {
            "chunk_id": "L2-match-3869685-chunk-0",
            "text": "First evidence block.",
            "metadata": {
                "document_id": "L2-match-3869685",
                "level": "2",
                "match_id": 3869685,
                "team_name": "Argentina",
            },
            "score": 0.9,
        },
        {
            "chunk_id": "L2-match-3869685-chunk-1",
            "text": "Second evidence block.",
            "metadata": {
                "document_id": "L2-match-3869685",
                "level": "2",
                "match_id": 3869685,
                "team_name": "Argentina",
            },
            "score": 0.8,
        },
    ]

    context = retrieval.build_context(chunks)

    assert "[Source 1:" in context
    assert "[Source 2:" in context
    assert "L2-match-3869685-chunk-0" in context
    assert "L2-match-3869685-chunk-1" in context
    assert "[Document 1" not in context


def test_answer_question_corrects_structured_numeric_contradiction(monkeypatch):
    """
    Structured generation validation parity: chat.py::process_query() already
    runs validate_answer() against a usable structured result (status
    "resolved"/"partial" with an explanation) and swaps in the corrected
    answer on contradiction. answer_question() -- the Streamlit production
    path -- never calls validate_answer() at all, so a generated answer that
    numerically contradicts verified structured data reaches the user
    unchanged even though the exact same authoritative fact was available.
    """
    prompting = import_module("07_prompting")
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    structured_result = SimpleNamespace(
        status="resolved",
        explanation="Jamie Vardy's total goals is 24.",
        aggregated_value=24,
        query=SimpleNamespace(metric="goals"),
    )
    # A usable structured result must never be blocked by the Step 1 gate,
    # regardless of semantic answerability -- give it a harmless status so
    # this test can't accidentally pass for the wrong reason (the gate
    # refusing before generation rather than validation correcting it).
    answerable = AnswerabilityAssessment(
        status="answerable", matched_terms=("goals",), missing_terms=(),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=structured_result,
            semantic_chunks=[],
            context="Jamie Vardy's total goals is 24.",
            answerability=answerable,
        ),
    )
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: "Jamie Vardy scored 20 goals.",
    )

    answer, _ = prompting.answer_question(
        "How many goals did Jamie Vardy score?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert "20" not in answer, (
        "answer_question() returned a generated answer that contradicts "
        "verified structured data (Jamie Vardy's actual total is 24 goals) "
        "unchanged -- the Streamlit generation path never runs "
        "validate_answer(), unlike chat.py::process_query()."
    )
    assert "24" in answer


def test_answer_question_leaves_correct_structured_answer_unchanged(monkeypatch):
    """
    New answer_question() validation wiring must be a no-op when the
    generated answer already agrees with the structured fact -- only a
    detected contradiction should ever alter the returned text.
    """
    prompting = import_module("07_prompting")
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    structured_result = SimpleNamespace(
        status="resolved",
        explanation="Jamie Vardy's total goals is 24.",
        aggregated_value=24,
        query=SimpleNamespace(metric="goals"),
    )
    answerable = AnswerabilityAssessment(
        status="answerable", matched_terms=("goals",), missing_terms=(),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=structured_result,
            semantic_chunks=[],
            context="Jamie Vardy's total goals is 24.",
            answerability=answerable,
        ),
    )
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: "Jamie Vardy scored 24 goals.",
    )

    answer, _ = prompting.answer_question(
        "How many goals did Jamie Vardy score?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert answer == "Jamie Vardy scored 24 goals."


def test_answer_question_pure_semantic_skips_structured_validation(monkeypatch):
    """
    Structured boundary: a pure semantic query (structured_result=None) must
    never be routed through structured contradiction correction -- only a
    usable structured result should trigger validate_answer() at all.
    """
    prompting = import_module("07_prompting")
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    answerable = AnswerabilityAssessment(
        status="answerable", matched_terms=("goals",), missing_terms=(),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=None,
            semantic_chunks=[{
                "chunk_id": "DOC-1-part-0",
                "text": "The match finished 3-2 after a late winner.",
                "metadata": {"document_id": "DOC-1", "level": "1", "match_id": "8658"},
            }],
            context="The match finished 3-2 after a late winner.",
            answerability=answerable,
        ),
    )
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: "The match finished 3-2 after a late winner.",
    )

    validate_calls = []
    monkeypatch.setattr(
        prompting,
        "validate_answer",
        lambda **kw: validate_calls.append(kw),
    )

    answer, _ = prompting.answer_question(
        "What happened in the match?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert not validate_calls, (
        "answer_question() ran structured contradiction validation for a "
        "pure semantic query with no structured_result."
    )
    assert answer == "The match finished 3-2 after a late winner."
