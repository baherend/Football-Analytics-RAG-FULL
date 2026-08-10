"""
Competition Portability — Batch 8: conversation memory wired into chat.py.

chat.py already threads conversation across turns (Batch 7's ad-hoc
`state.history` text dump) but that dump was neither dataset-scoped nor
searched -- it fed every turn verbatim into the prompt and would silently
leak across a runtime dataset switch. These tests pin the replacement
behavior: a dataset-scoped, searched `state.memory`.

`chat` is a plain top-level module cached in sys.modules once imported, so
`state` is a process-wide singleton across the whole test session (the
existing Batch 7 tests already rely on this and reset the fields they
care about). Each test below resets `state.memory`/`state.artifact_paths`
itself for the same reason.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from src.artifacts import ArtifactPaths
from src.conversation_memory import ConversationMemory


def _stub_router(monkeypatch, chat, route_query=None, execute_route=None):
    monkeypatch.setattr(
        chat.router_mod,
        "route_query",
        route_query or (lambda q, artifact_paths=None: SimpleNamespace(
            path="semantic", confidence=1.0, reason="test", semantic_query=q,
        )),
    )
    monkeypatch.setattr(
        chat.router_mod,
        "execute_route",
        execute_route or (lambda route, semantic_k=5, artifact_paths=None: SimpleNamespace(
            structured_result=None, semantic_chunks=[],
        )),
    )


def _stub_generation(monkeypatch, chat, answer="answer"):
    monkeypatch.setattr(chat.prompting_mod, "build_prompt", lambda *a, **kw: "prompt")
    monkeypatch.setattr(chat.prompting_mod, "generate_answer", lambda *a, **kw: answer, raising=False)


def test_chat_process_query_resolves_pronoun_via_memory_for_routing(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.memory.add_turn(selected, "Tell me about Messi's tournament.", "Messi played well.")

    captured = {}

    def fake_route_query(query, artifact_paths=None):
        captured["routed_query"] = query
        return SimpleNamespace(path="semantic", confidence=1.0, reason="test", semantic_query=query)

    _stub_router(monkeypatch, chat, route_query=fake_route_query)
    _stub_generation(monkeypatch, chat)

    chat.process_query("How many shots did he have?")

    assert captured["routed_query"] == "How many shots did Messi have?"


def test_chat_process_query_leaves_query_unresolved_without_relevant_memory(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()

    captured = {}

    def fake_route_query(query, artifact_paths=None):
        captured["routed_query"] = query
        return SimpleNamespace(path="semantic", confidence=1.0, reason="test", semantic_query=query)

    _stub_router(monkeypatch, chat, route_query=fake_route_query)
    _stub_generation(monkeypatch, chat)

    chat.process_query("How many shots did he have?")

    assert captured["routed_query"] == "How many shots did he have?"


def test_chat_switching_dataset_does_not_reuse_previous_dataset_memory(monkeypatch):
    chat = import_module("chat")
    dataset_a = ArtifactPaths(2, 27)
    dataset_b = ArtifactPaths(11, 90)
    chat.state.memory = ConversationMemory()
    chat.state.artifact_paths = dataset_a

    _stub_router(monkeypatch, chat)
    _stub_generation(monkeypatch, chat)

    chat.process_query("Tell me about Messi's tournament.")

    chat.configure_runtime_dataset(11, 90)
    assert chat.state.artifact_paths == dataset_b

    captured = {}

    def fake_route_query(query, artifact_paths=None):
        captured["routed_query"] = query
        return SimpleNamespace(path="semantic", confidence=1.0, reason="test", semantic_query=query)

    _stub_router(monkeypatch, chat, route_query=fake_route_query)

    chat.process_query("How many shots did he have?")

    assert captured["routed_query"] == "How many shots did he have?"


def test_chat_process_query_surfaces_memory_as_labeled_context(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()
    chat.state.memory.add_turn(selected, "Tell me about Messi's tournament.", "Messi played well.")

    _stub_router(monkeypatch, chat)

    captured = {}

    def spy_build_prompt(question, context, has_structured=False):
        captured["question"] = question
        captured["context"] = context
        return "prompt"

    monkeypatch.setattr(chat.prompting_mod, "build_prompt", spy_build_prompt)
    monkeypatch.setattr(chat.prompting_mod, "generate_answer", lambda *a, **kw: "answer", raising=False)

    chat.process_query("How many shots did he have?")

    assert captured["question"] == "How many shots did he have?"
    assert "Conversation Context" in captured["context"]
    assert "not verified" in captured["context"].lower()


def test_chat_process_query_records_turn_in_memory(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.memory = ConversationMemory()

    _stub_router(monkeypatch, chat)
    _stub_generation(monkeypatch, chat, answer="Messi scored 5 goals.")

    chat.process_query("How many goals did Messi score?")

    recorded = chat.state.memory.search(selected, "How many goals did Messi score?")
    assert len(recorded) == 1
    assert recorded[0].answer == "Messi scored 5 goals."
