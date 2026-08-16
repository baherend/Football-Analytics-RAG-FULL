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
from src.query.query_schema import ComparisonResult, ComparisonValue
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


def test_answer_question_blocks_incomplete_partial_comparison(monkeypatch):
    """
    Comparison Engine Step 2H: a ComparisonResult with one entity missing
    a usable value gets status="partial" (Step 2G) -- but
    is_unsupported_query()'s existing status in ("resolved", "partial")
    check treats "partial" identically to a fully "resolved" result, so
    this incomplete two-sided comparison is currently presented to the
    LLM as fully authoritative structured evidence ("VERIFIED and must be
    used EXACTLY") even though the comparison itself cannot actually be
    completed (difference/outcome are both None -- Step 2F already
    proved neither is fabricated). When semantic evidence is also
    unanswerable, this must trigger the same deterministic refusal an
    unsupported query already gets, not reach generation.
    """
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    incomplete_comparison = ComparisonResult(
        status="partial",
        metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=10),
            ComparisonValue(entity_name="Beta Player", value=None),
        ],
        explanation="Alpha Player: 10. | Beta Player: No data available.",
    )
    unanswerable = AnswerabilityAssessment(
        status="unanswerable", matched_terms=(), missing_terms=("beta", "player"),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=incomplete_comparison,
            semantic_chunks=[],
            context="Alpha Player: 10. | Beta Player: No data available.",
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: (generation_calls.append(1), "should not be generated")[1],
    )

    answer, _ = prompting.answer_question(
        "Who scored more goals, Alpha Player or Beta Player?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert not generation_calls, (
        "answer_question() invoked the LLM for an incomplete comparison (one "
        "entity has no value) even though semantic evidence was also "
        "unanswerable -- an incomplete comparison must not be treated as "
        "fully authoritative structured evidence."
    )
    assert answer == prompting.INSUFFICIENT_CONTEXT_MESSAGE


def test_chat_process_query_blocks_incomplete_partial_comparison(monkeypatch):
    """
    Comparison Engine Step 2H, CLI parity: chat.py::process_query() must
    apply the exact same completeness-aware usability rule as
    answer_question() (both now delegate to the shared
    prompting_mod.is_usable_structured_result()) -- an incomplete
    "partial" comparison must not behave differently between the two
    generation entry points.
    """
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.mode = "hybrid"

    incomplete_comparison = ComparisonResult(
        status="partial",
        metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=10),
            ComparisonValue(entity_name="Beta Player", value=None),
        ],
        explanation="Alpha Player: 10. | Beta Player: No data available.",
    )
    unanswerable = AnswerabilityAssessment(
        status="unanswerable", matched_terms=(), missing_terms=("beta", "player"),
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
            structured_result=incomplete_comparison,
            semantic_chunks=[],
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        chat.prompting_mod,
        "generate_answer",
        lambda *a, **kw: (generation_calls.append(1), "should not be generated")[1],
        raising=False,
    )

    answer = chat.process_query("Who scored more goals, Alpha Player or Beta Player?")

    assert not generation_calls, (
        "chat.process_query() invoked the LLM for an incomplete comparison "
        "(one entity has no value) even though semantic evidence was also "
        "unanswerable."
    )
    assert answer == chat.prompting_mod.INSUFFICIENT_CONTEXT_MESSAGE


def test_answer_question_complete_partial_comparison_remains_usable(monkeypatch):
    """
    Comparison Engine Step 2H safety boundary: a "partial" comparison is
    not automatically incomplete -- when both entities produced a real
    numeric value (the partial status came from one underlying entity's
    own StructuredResult, e.g. a dropped filter, not from a missing
    side), the comparison must remain usable structured evidence. It
    must not be blocked merely because its status string is "partial",
    and its status must not be silently upgraded to "resolved".
    """
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    complete_partial_comparison = ComparisonResult(
        status="partial",
        metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=10),
            ComparisonValue(entity_name="Beta Player", value=20),
        ],
        explanation="Alpha Player: 10 (Note: could not apply filter(s): period). | Beta Player: 20.",
    )
    # Even if semantic answerability looks unanswerable, a usable
    # structured result must take precedence (the same boundary Steps
    # 1-2G already established) -- confirms this isn't blocked "for the
    # right reason" (structured usability), not by coincidence.
    unanswerable = AnswerabilityAssessment(
        status="unanswerable", matched_terms=(), missing_terms=("x",),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=complete_partial_comparison,
            semantic_chunks=[],
            context="Alpha Player: 10. | Beta Player: 20.",
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: (generation_calls.append(1), "Alpha Player: 10, Beta Player: 20.")[1],
    )

    answer, _ = prompting.answer_question(
        "Who scored more goals, Alpha Player or Beta Player?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert generation_calls, (
        "answer_question() blocked a complete comparison (both entities have "
        "usable values) merely because its status is 'partial' -- a partial "
        "status with both values present must remain usable structured evidence."
    )
    assert complete_partial_comparison.status == "partial", (
        "complete_partial_comparison.status was mutated -- a complete comparison "
        "with an underlying partial caveat must not be silently upgraded to "
        "'resolved'."
    )
    assert [v.value for v in complete_partial_comparison.values] == [10, 20]


def test_answer_question_ordinary_partial_structured_result_remains_usable(monkeypatch):
    """
    Comparison Engine Step 2H non-regression: an ordinary single-entity
    StructuredResult(status="partial", aggregated_value=<number>) --
    e.g. a real value with a dropped-filter caveat, per
    src/query/resolver.py -- must keep its existing usable/authoritative
    behavior. The new completeness rule is comparison-specific (detected
    via the presence of a `.values` list) and must not affect ordinary
    structured results, which have no `.values` attribute at all.
    """
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    ordinary_partial = SimpleNamespace(
        status="partial",
        aggregated_value=10,
        explanation="Alpha Player's total minutes is 10 (Note: could not apply filter(s): period).",
        dropped_filters=["period"],
        data=[],
    )
    unanswerable = AnswerabilityAssessment(
        status="unanswerable", matched_terms=(), missing_terms=("x",),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=ordinary_partial,
            semantic_chunks=[],
            context=ordinary_partial.explanation,
            answerability=unanswerable,
        ),
    )

    generation_calls = []
    monkeypatch.setattr(
        prompting,
        "ask_groq",
        lambda *a, **kw: (generation_calls.append(1), "Alpha Player played 10 minutes.")[1],
    )

    answer, _ = prompting.answer_question(
        "How many minutes did Alpha Player play in the first half?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert generation_calls, (
        "answer_question() blocked an ordinary partial StructuredResult with a "
        "real aggregated_value -- this is the pre-existing, unrelated 'usable "
        "value with a scope caveat' contract and must not be affected by the "
        "comparison-specific completeness rule."
    )


def test_answer_question_corrects_comparison_outcome_contradiction(monkeypatch):
    """
    Comparison Engine Step 2I: a complete ComparisonResult is treated as
    fully authoritative structured evidence (Step 2H), but the existing
    validate_answer() only understands a single scalar aggregated_value --
    it has no notion of two entities, an outcome, or a difference. A
    generated comparison answer that directly contradicts the
    authoritative outcome (which entity actually scored more) currently
    reaches the user unchanged, because validate_answer() cannot detect
    the contradiction at all in this shape.
    """
    retrieval = import_module("06_retrieve_context")
    selected = ArtifactPaths(2, 27)

    comparison = ComparisonResult(
        status="resolved",
        metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=25),
            ComparisonValue(entity_name="Beta Player", value=24),
        ],
        explanation="Alpha Player: Alpha Player's total goals is 25. | Beta Player: Beta Player's total goals is 24.",
    )
    # difference/outcome are derived automatically by ComparisonResult.__post_init__.
    assert comparison.difference == 1
    assert comparison.outcome == "entity_a_higher"

    answerable = AnswerabilityAssessment(
        status="answerable", matched_terms=("goals",), missing_terms=(),
    )

    monkeypatch.setattr(
        retrieval,
        "route_and_execute",
        lambda q, artifact_paths=None: SimpleNamespace(
            structured_result=comparison,
            semantic_chunks=[],
            context=comparison.explanation,
            answerability=answerable,
        ),
    )

    wrong_answer = "Beta Player scored more goals than Alpha Player."
    monkeypatch.setattr(prompting, "ask_groq", lambda *a, **kw: wrong_answer)

    answer, _ = prompting.answer_question(
        "Who scored more goals, Alpha Player or Beta Player?",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert answer != wrong_answer, (
        "answer_question() returned a comparison answer that contradicts the "
        "authoritative outcome (Alpha Player is higher, not Beta Player) "
        "unchanged -- validate_answer() cannot validate ComparisonResult's "
        "two-entity shape."
    )
    assert "alpha player" in answer.lower(), (
        f"the corrected answer must name the actual higher entity (Alpha Player), got: {answer!r}"
    )


def test_chat_process_query_corrects_comparison_outcome_contradiction(monkeypatch):
    """
    Comparison Engine Step 2I, CLI parity: chat.py::process_query() must
    apply the exact same comparison validation as answer_question() --
    both now delegate to the shared prompting_mod.validate_structured_answer().
    """
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.mode = "hybrid"

    comparison = ComparisonResult(
        status="resolved",
        metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=25),
            ComparisonValue(entity_name="Beta Player", value=24),
        ],
        explanation="Alpha Player: Alpha Player's total goals is 25. | Beta Player: Beta Player's total goals is 24.",
    )
    answerable = AnswerabilityAssessment(
        status="answerable", matched_terms=("goals",), missing_terms=(),
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
            structured_result=comparison,
            semantic_chunks=[],
            answerability=answerable,
        ),
    )

    wrong_answer = "Beta Player scored more goals than Alpha Player."
    monkeypatch.setattr(
        chat.prompting_mod, "generate_answer", lambda *a, **kw: wrong_answer, raising=False,
    )

    answer = chat.process_query("Who scored more goals, Alpha Player or Beta Player?")

    assert answer != wrong_answer, (
        "chat.process_query() returned a comparison answer that contradicts the "
        "authoritative outcome unchanged."
    )
    assert "alpha player" in answer.lower(), (
        f"the corrected answer must name the actual higher entity (Alpha Player), got: {answer!r}"
    )


def test_validate_comparison_answer_cases():
    """
    Comparison Engine Step 2I regression: the smallest meaningful safety
    matrix for validate_comparison_answer(), calling it directly since
    detection is pure logic given an already-computed ComparisonResult --
    no execute_route()/answer_question() machinery needed to prove it.
    """
    comparison = ComparisonResult(
        status="resolved", metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=25),
            ComparisonValue(entity_name="Beta Player", value=24),
        ],
        explanation="Alpha Player: 25 | Beta Player: 24",
    )
    tie_comparison = ComparisonResult(
        status="resolved", metric="goals",
        values=[
            ComparisonValue(entity_name="Alpha Player", value=10),
            ComparisonValue(entity_name="Beta Player", value=10),
        ],
        explanation="Alpha Player: 10 | Beta Player: 10",
    )

    cases = [
        # (comparison_result, generated_answer, expect_valid)
        (comparison, "Alpha Player scored more than Beta Player.", True),           # A: correct ordering
        (comparison, "Beta Player scored more than Alpha Player.", False),          # B: wrong ordering
        (comparison, "Alpha Player had 25 goals and Beta Player had 24.", True),    # C: correct values
        (comparison, "Alpha Player had 20 goals and Beta Player had 24.", False),   # D: wrong entity value
        (comparison, "Alpha Player led Beta Player by 3 goals.", False),            # E: wrong difference
        (tie_comparison, "Alpha Player scored more than Beta Player.", False),      # F: tie claimed as a win
        (comparison, "Alpha Player and Beta Player had a great tournament.", True), # omission: no explicit claim
        # Negation: interpreting a negated comparison's actual truth value
        # would require real language understanding, which this
        # deterministic validator intentionally does not attempt. Both
        # negated cases below must be treated as "no clear directional
        # claim" (an omission) rather than a literal, negation-blind
        # pattern match -- critically, a truly correct negated statement
        # (the second case) must never be flagged as a contradiction.
        (comparison, "Alpha Player did not score more goals than Beta Player.", True),
        (comparison, "Beta Player did not score more goals than Alpha Player.", True),
    ]
    for comparison_result, generated_answer, expect_valid in cases:
        validation = prompting.validate_comparison_answer(generated_answer, comparison_result)
        assert validation.is_valid == expect_valid, (
            f"validate_comparison_answer({generated_answer!r}) against "
            f"outcome={comparison_result.outcome!r}, difference={comparison_result.difference!r}: "
            f"is_valid={validation.is_valid}, expected {expect_valid}"
        )
        if not expect_valid:
            assert validation.corrected_answer, (
                "an invalid comparison answer must produce a deterministic correction"
            )
