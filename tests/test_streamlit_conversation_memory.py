"""
Competition Portability — Batch 8: conversation memory wired into
streamlit_app.py.

Batch 7's `tests/test_runtime_dataset_selection.py` already verifies
streamlit_app.py's runtime-dataset-selection contract via lightweight
source inspection (ast parsing / substring checks) rather than executing
the Streamlit script, since driving `st.session_state` end-to-end needs a
running app. These Batch 8 tests follow the same established pattern for
the conversation-memory wiring.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _source() -> str:
    return Path("streamlit_app.py").read_text(encoding="utf-8")


def test_streamlit_initializes_session_scoped_conversation_memory():
    source = _source()

    assert "ConversationMemory" in source
    assert "st.session_state.memory" in source


def test_streamlit_passes_memory_to_answer_question():
    tree = ast.parse(_source())
    answer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "answer_question"
    ]

    assert answer_calls
    assert any(
        any(keyword.arg == "memory" for keyword in call.keywords)
        for call in answer_calls
    )


def test_streamlit_resets_memory_alongside_messages_when_dataset_changes():
    source = _source()

    expected_reset = """if "dataset_key" not in st.session_state:
    st.session_state.dataset_key = dataset_key
elif st.session_state.dataset_key != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.messages = []
    st.session_state.memory = ConversationMemory()"""
    assert expected_reset in source
