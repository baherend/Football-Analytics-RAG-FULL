"""
Competition Portability — Batch 7: Runtime Dataset Selection / Entry-Point Portability.

These tests define the runtime-boundary contract only. They deliberately do
not retest retrieval internals already covered by Batch 5/6.
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

from src.artifacts import ArtifactPaths


def test_runtime_dataset_resolution_keeps_zero_arg_wc2022_legacy():
    from src.artifacts import resolve_runtime_artifact_paths

    assert resolve_runtime_artifact_paths(43, 106) is None


def test_runtime_dataset_resolution_namespaces_non_wc_dataset():
    from src.artifacts import resolve_runtime_artifact_paths

    assert resolve_runtime_artifact_paths(2, 27) == ArtifactPaths(2, 27)


def test_runtime_dataset_resolution_can_explicitly_namespace_wc2022():
    from src.artifacts import resolve_runtime_artifact_paths

    assert resolve_runtime_artifact_paths(43, 106, legacy_default=False) == ArtifactPaths(43, 106)


def test_answer_question_threads_selected_artifact_paths(monkeypatch):
    prompting = import_module("07_prompting")
    selected = ArtifactPaths(2, 27)
    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["question"] = question
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(context="test context", semantic_chunks=[])

    fake_router = SimpleNamespace(route_and_execute=fake_route_and_execute)
    real_import_module = prompting.import_module

    def fake_import_module(name):
        if name == "06_retrieve_context":
            return fake_router
        return real_import_module(name)

    monkeypatch.setattr(prompting, "import_module", fake_import_module)
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


def test_answer_question_default_remains_legacy_compatible(monkeypatch):
    prompting = import_module("07_prompting")
    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(context="test context", semantic_chunks=[])

    fake_router = SimpleNamespace(route_and_execute=fake_route_and_execute)
    real_import_module = prompting.import_module

    def fake_import_module(name):
        if name == "06_retrieve_context":
            return fake_router
        return real_import_module(name)

    monkeypatch.setattr(prompting, "import_module", fake_import_module)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    prompting.answer_question("test question", api_key="test-key")

    assert captured["artifact_paths"] is None


def test_generation_system_prompt_is_competition_neutral():
    prompting = import_module("07_prompting")

    assert "FIFA World Cup 2022" not in prompting.SYSTEM_PROMPT
    assert "football analytics assistant" in prompting.SYSTEM_PROMPT.lower()

def test_chat_configures_runtime_dataset_once():
    chat = import_module("chat")

    chat.configure_runtime_dataset(2, 27)

    assert chat.state.artifact_paths == ArtifactPaths(2, 27)


def test_chat_process_query_threads_configured_artifact_paths(monkeypatch):
    chat = import_module("chat")
    selected = ArtifactPaths(2, 27)
    chat.state.artifact_paths = selected
    chat.state.history = []
    captured = {}

    route = SimpleNamespace(
        path="semantic",
        confidence=1.0,
        reason="test",
        semantic_query="test question",
    )

    def fake_route_query(question, artifact_paths=None):
        captured["route_artifact_paths"] = artifact_paths
        return route

    monkeypatch.setattr(chat.router_mod, "route_query", fake_route_query)

    def fake_execute_route(route, semantic_k=5, original_query="", artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(structured_result=None, semantic_chunks=[])

    monkeypatch.setattr(chat.router_mod, "execute_route", fake_execute_route)
    monkeypatch.setattr(chat.prompting_mod, "build_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(chat.prompting_mod, "generate_answer", lambda *args, **kwargs: "answer", raising=False)

    assert chat.process_query("test question") == "answer"
    assert captured["route_artifact_paths"] is selected
    assert captured["artifact_paths"] is selected


def test_chat_main_selects_namespaced_dataset_without_polluting_question(monkeypatch):
    import sys

    chat = import_module("chat")
    captured = {}

    monkeypatch.setattr(chat.prompting_mod, "get_api_key", lambda: "test-key", raising=False)
    monkeypatch.setattr(
        chat,
        "single_question_mode",
        lambda question: captured.setdefault("question", question),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chat.py",
            "--competition-id", "2",
            "--season-id", "27",
            "Who scored the most goals?",
        ],
    )

    assert chat.main() == 0
    assert captured["question"] == "Who scored the most goals?"
    assert chat.state.artifact_paths == ArtifactPaths(2, 27)


def test_chat_main_without_dataset_flags_keeps_legacy_runtime(monkeypatch):
    import sys

    chat = import_module("chat")
    captured = {}
    chat.state.artifact_paths = ArtifactPaths(2, 27)

    monkeypatch.setattr(chat.prompting_mod, "get_api_key", lambda: "test-key", raising=False)
    monkeypatch.setattr(
        chat,
        "single_question_mode",
        lambda question: captured.setdefault("question", question),
    )
    monkeypatch.setattr(sys, "argv", ["chat.py", "Who scored the most goals?"])

    assert chat.main() == 0
    assert captured["question"] == "Who scored the most goals?"
    assert chat.state.artifact_paths is None

def _streamlit_source() -> str:
    from pathlib import Path

    return Path("streamlit_app.py").read_text(encoding="utf-8")


def test_streamlit_selected_catalog_entry_supplies_runtime_artifact_paths():
    """The catalog-selected entry must directly supply the runtime contract."""
    import ast

    tree = ast.parse(_streamlit_source())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "artifact_paths"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "artifact_paths"
    assert isinstance(value.value, ast.Name)
    assert value.value.id == "selected_entry"


def test_streamlit_passes_selected_artifact_paths_to_answer_question():
    import ast

    tree = ast.parse(_streamlit_source())
    answer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "answer_question"
    ]

    assert answer_calls
    assert any(
        any(keyword.arg == "artifact_paths" for keyword in call.keywords)
        for call in answer_calls
    )


def test_streamlit_runtime_ui_is_competition_neutral():
    source = _streamlit_source()

    assert "FIFA World Cup 2022" not in source


def test_streamlit_resets_chat_history_when_dataset_changes():
    source = _streamlit_source()

    expected_reset = """if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = dataset_key
elif st.session_state.dataset_key != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.messages = []"""
    assert expected_reset in source

def test_runtime_entry_points_have_no_wc2022_specific_user_facing_residue():
    from pathlib import Path

    chat_source = Path("chat.py").read_text(encoding="utf-8")
    streamlit_source = Path("streamlit_app.py").read_text(encoding="utf-8")
    prompting_source = Path("07_prompting.py").read_text(encoding="utf-8")

    assert "FIFA World Cup 2022" not in chat_source
    assert "2,835+ documents" not in streamlit_source
    assert 'question = "How many goals did Messi score in the tournament?"' not in prompting_source
