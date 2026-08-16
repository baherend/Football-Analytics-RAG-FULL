"""
Competition Portability — Batch 8: Conversation Memory integration boundary.

Verifies that 07_prompting.answer_question wires ConversationMemory in
without becoming a football-fact source itself:

* memory is searched under the *selected* dataset scope (Batch 7 boundary)
* a follow-up question's retrieval query gets pronoun-resolved via memory,
  while the *visible* question passed to the prompt stays exactly what the
  user typed
* relevant memory is surfaced in the prompt context, clearly labeled as
  non-authoritative, alongside (never in place of) retrieval evidence
* each turn is recorded after generation, under the same dataset scope
* omitting `memory` entirely reproduces the exact Batch 7 behavior
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from src.artifacts import ArtifactPaths
from src.conversation_memory import ConversationMemory


def _patch_router(monkeypatch, prompting, fake_route_and_execute):
    """
    Structural Cleanup Phase B: answer_question() now imports
    route_and_execute directly from src.query.router (`from
    src.query.router import route_and_execute`), so it resolves the name
    from 07_prompting.py's own module globals -- patch it there.
    """
    monkeypatch.setattr(prompting, "route_and_execute", fake_route_and_execute)


def test_answer_question_searches_memory_under_selected_dataset_scope(monkeypatch):
    prompting = import_module("07_prompting")
    selected = ArtifactPaths(2, 27)
    memory = ConversationMemory()
    memory.add_turn(selected, "Tell me about Messi's tournament.", "Messi played well.")

    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["question"] = question
        return SimpleNamespace(context="evidence context", semantic_chunks=[])

    _patch_router(monkeypatch, prompting, fake_route_and_execute)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    prompting.answer_question(
        "How many shots did he have?",
        api_key="test-key",
        artifact_paths=selected,
        memory=memory,
    )

    # The pronoun "he" is resolved for retrieval using the memory entity.
    assert captured["question"] == "How many shots did Messi have?"


def test_answer_question_does_not_leak_other_dataset_memory(monkeypatch):
    prompting = import_module("07_prompting")
    dataset_a = ArtifactPaths(2, 27)
    dataset_b = ArtifactPaths(11, 90)
    memory = ConversationMemory()
    memory.add_turn(dataset_a, "Tell me about Messi's tournament.", "Messi played well.")

    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["question"] = question
        return SimpleNamespace(context="evidence context", semantic_chunks=[])

    _patch_router(monkeypatch, prompting, fake_route_and_execute)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    prompting.answer_question(
        "How many shots did he have?",
        api_key="test-key",
        artifact_paths=dataset_b,
        memory=memory,
    )

    # No relevant memory in dataset_b's scope -> pronoun left unresolved.
    assert captured["question"] == "How many shots did he have?"


def test_answer_question_surfaces_memory_as_labeled_context_not_authoritative(monkeypatch):
    prompting = import_module("07_prompting")
    selected = ArtifactPaths(2, 27)
    memory = ConversationMemory()
    memory.add_turn(selected, "Tell me about Messi's tournament.", "Messi played well.")

    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        return SimpleNamespace(context="Authoritative structured evidence goes here.", semantic_chunks=[])

    _patch_router(monkeypatch, prompting, fake_route_and_execute)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    real_build_prompt = prompting.build_prompt

    def spy_build_prompt(question, context, has_structured=False):
        captured["question"] = question
        captured["context"] = context
        return real_build_prompt(question, context, has_structured=has_structured)

    monkeypatch.setattr(prompting, "build_prompt", spy_build_prompt)

    prompting.answer_question(
        "How many shots did he have?",
        api_key="test-key",
        artifact_paths=selected,
        memory=memory,
    )

    # The visible question stays exactly what the user typed.
    assert captured["question"] == "How many shots did he have?"
    # Both conversation context and retrieval evidence reach the prompt...
    assert "Conversation Context" in captured["context"]
    assert "Authoritative structured evidence goes here." in captured["context"]
    # ...but memory is explicitly not labeled as verified/authoritative.
    assert "not verified" in captured["context"].lower()


def test_answer_question_records_turn_after_generation(monkeypatch):
    prompting = import_module("07_prompting")
    selected = ArtifactPaths(2, 27)
    memory = ConversationMemory()

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        return SimpleNamespace(context="evidence context", semantic_chunks=[])

    _patch_router(monkeypatch, prompting, fake_route_and_execute)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "Messi scored 5 goals.")

    prompting.answer_question(
        "How many goals did Messi score?",
        api_key="test-key",
        artifact_paths=selected,
        memory=memory,
    )

    recorded = memory.search(selected, "How many goals did Messi score?")
    assert len(recorded) == 1
    assert recorded[0].answer == "Messi scored 5 goals."


def test_answer_question_without_memory_matches_batch7_behavior(monkeypatch):
    prompting = import_module("07_prompting")
    selected = ArtifactPaths(2, 27)
    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["question"] = question
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(context="test context", semantic_chunks=[])

    _patch_router(monkeypatch, prompting, fake_route_and_execute)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    answer, sources = prompting.answer_question(
        "test question",
        api_key="test-key",
        artifact_paths=selected,
    )

    assert answer == "test answer"
    assert sources == []
    assert captured["question"] == "test question"
    assert captured["artifact_paths"] is selected
