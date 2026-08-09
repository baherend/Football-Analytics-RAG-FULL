"""
Competition Portability — Batch 8: Conversation Memory & Memory Search.

These tests define the conversation-memory contract in isolation:

* relevant previous turns can be searched
* irrelevant turns don't dominate the result
* memory is strictly dataset-scoped (no cross-dataset leakage)
* switching dataset can't reuse the old dataset's memory
* a bounded, generic pronoun-resolution mechanism helps follow-up retrieval
* conversation context is formatted so it is never mistaken for verified
  football evidence

None of this exercises a real LLM, Chroma, or BM25 index — it is pure,
deterministic logic over in-memory conversation turns.
"""

from __future__ import annotations

from src.artifacts import ArtifactPaths
from src.conversation_memory import (
    ConversationMemory,
    ConversationTurn,
    format_conversation_context,
    resolve_pronoun_references,
)


DATASET_A = ArtifactPaths(2, 27)
DATASET_B = ArtifactPaths(11, 90)


# ---------------------------------------------------------------------------
# 1. Relevant memory retrieval
# ---------------------------------------------------------------------------


def test_search_returns_turn_matching_shared_entity():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Tell me about Messi's tournament.", "Messi played well.")

    results = memory.search(DATASET_A, "How many shots did he have?")

    assert len(results) == 1
    assert results[0].question == "Tell me about Messi's tournament."


def test_search_returns_previous_comparison_context_for_generic_followup():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Compare Argentina and France.", "Argentina edged France.")

    results = memory.search(DATASET_A, "Which team had higher xG?")

    assert len(results) == 1
    assert "Argentina" in results[0].question and "France" in results[0].question


# ---------------------------------------------------------------------------
# 2. Irrelevant memory exclusion
# ---------------------------------------------------------------------------


def test_search_prioritizes_matching_entity_over_unrelated_turns():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Tell me about Messi's tournament.", "Messi played well.")
    memory.add_turn(DATASET_A, "Tell me about Ronaldo's tournament.", "Ronaldo played well.")
    memory.add_turn(DATASET_A, "Tell me about the referees.", "Referees were consistent.")

    results = memory.search(DATASET_A, "How many goals did Messi score?", k=3)

    assert results[0].question == "Tell me about Messi's tournament."
    assert not any("referees" in r.question.lower() for r in results[:1])


def test_search_caps_results_at_k():
    memory = ConversationMemory()
    for i in range(5):
        memory.add_turn(DATASET_A, f"Tell me about Player{i}.", f"Player{i} played well.")

    results = memory.search(DATASET_A, "Which team had higher xG?", k=2)

    assert len(results) <= 2


def test_generic_entity_free_followup_does_not_fallback_to_unrelated_single_entity_turns():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Tell me about Messi.", "Messi played well.")
    memory.add_turn(DATASET_A, "Tell me about Ronaldo.", "Ronaldo played well.")

    results = memory.search(DATASET_A, "Which team had higher xG?")

    assert results == []


# ---------------------------------------------------------------------------
# 3. Dataset isolation
# ---------------------------------------------------------------------------


def test_memory_from_dataset_a_not_returned_for_dataset_b():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Tell me about Messi's tournament.", "Messi played well.")

    results = memory.search(DATASET_B, "How many shots did he have?")

    assert results == []


def test_legacy_flat_wc2022_and_explicit_namespaced_wc2022_are_isolated():
    memory = ConversationMemory()
    legacy_wc2022 = None  # resolve_runtime_artifact_paths(43, 106) -> None
    namespaced_wc2022 = ArtifactPaths(43, 106)

    memory.add_turn(legacy_wc2022, "Tell me about Messi's tournament.", "Messi played well.")

    assert memory.search(namespaced_wc2022, "How many shots did he have?") == []
    assert len(memory.search(legacy_wc2022, "How many shots did he have?")) == 1


# ---------------------------------------------------------------------------
# 4. Dataset-change behavior
# ---------------------------------------------------------------------------


def test_clearing_one_dataset_does_not_affect_another():
    memory = ConversationMemory()
    memory.add_turn(DATASET_A, "Tell me about Messi's tournament.", "Messi played well.")
    memory.add_turn(DATASET_B, "Tell me about Kane's tournament.", "Kane played well.")

    memory.clear(DATASET_A)

    assert memory.search(DATASET_A, "How many shots did he have?") == []
    assert len(memory.search(DATASET_B, "How many shots did he have?")) == 1


# ---------------------------------------------------------------------------
# 5. Query integration — pronoun resolution feeds retrieval, not the visible question
# ---------------------------------------------------------------------------


def test_resolve_pronoun_references_substitutes_last_entity():
    turns = [ConversationTurn(question="Tell me about Messi.", answer="Messi played well.",
                               entities=("Messi",))]

    expanded = resolve_pronoun_references("How many shots did he have?", turns)

    assert expanded == "How many shots did Messi have?"


def test_resolve_references_enriches_generic_followup_with_multi_entity_context():
    turns = [ConversationTurn(
        question="Compare Argentina and France.",
        answer="The comparison is ready.",
        entities=("Argentina", "France"),
    )]

    expanded = resolve_pronoun_references("Which team had higher xG?", turns)

    assert "Which team had higher xG?" in expanded
    assert "Argentina" in expanded
    assert "France" in expanded



def test_search_and_resolve_preserve_previous_topic_for_what_about_new_entity():
    memory = ConversationMemory()
    memory.add_turn(
        DATASET_A,
        "How many goals did Messi score?",
        "Messi scored several goals.",
    )

    turns = memory.search(DATASET_A, "What about Mbappe?")
    expanded = resolve_pronoun_references("What about Mbappe?", turns)

    assert turns
    assert "Mbappe" in expanded
    assert "goals" in expanded.lower()



def test_plural_pronoun_with_multi_entity_context_does_not_collapse_to_first_entity():
    turns = [ConversationTurn(
        question="Compare Argentina and France.",
        answer="The comparison is ready.",
        entities=("Argentina", "France"),
    )]

    expanded = resolve_pronoun_references("How did they perform?", turns)

    assert "Argentina" in expanded
    assert "France" in expanded
    assert expanded != "How did Argentina perform?"



def test_assistant_only_entity_cannot_become_reference_target():
    memory = ConversationMemory()
    memory.add_turn(
        DATASET_A,
        "Who was the top scorer?",
        "Hallucinated Player was the top scorer.",
    )

    turns = memory.search(DATASET_A, "How many shots did he have?")

    assert turns == []


def test_resolve_pronoun_references_is_noop_with_no_relevant_turns():
    assert resolve_pronoun_references("How many shots did he have?", []) == "How many shots did he have?"


# ---------------------------------------------------------------------------
# 6. No-memory compatibility
# ---------------------------------------------------------------------------


def test_search_on_empty_memory_returns_empty_list():
    memory = ConversationMemory()

    assert memory.search(DATASET_A, "How many goals did Messi score?") == []


# ---------------------------------------------------------------------------
# 7. Legacy compatibility — None is a valid, isolated dataset scope
# ---------------------------------------------------------------------------


def test_none_dataset_scope_is_a_valid_isolated_scope():
    memory = ConversationMemory()
    memory.add_turn(None, "Tell me about Messi's tournament.", "Messi played well.")

    results = memory.search(None, "How many shots did he have?")

    assert len(results) == 1


# ---------------------------------------------------------------------------
# 8. Memory is not authoritative football evidence
# ---------------------------------------------------------------------------


def test_formatted_conversation_context_is_never_labeled_authoritative():
    turns = [ConversationTurn(question="Tell me about Messi.", answer="Messi played well.",
                               entities=("Messi",))]

    formatted = format_conversation_context(turns)

    lowered = formatted.lower()
    assert "authoritative" not in lowered
    assert "is verified" not in lowered and "are verified" not in lowered
    assert "not verified" in lowered or "not a source" in lowered or "reference only" in lowered


def test_formatted_conversation_context_is_empty_string_for_no_turns():
    assert format_conversation_context([]) == ""
