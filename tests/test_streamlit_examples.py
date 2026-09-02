"""Streamlit example-question registry and submission-flow contracts."""

from __future__ import annotations

import ast
from pathlib import Path


def _streamlit_source() -> str:
    return Path("streamlit_app.py").read_text(encoding="utf-8")


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == name
    ]


def test_example_registry_is_competition_specific_and_bilingual():
    from src.example_questions import EXAMPLE_QUESTIONS

    wc = EXAMPLE_QUESTIONS[(43, 106)]
    epl = EXAMPLE_QUESTIONS[(2, 27)]

    assert tuple(wc) == ("en", "ar")
    assert tuple(epl) == ("en", "ar")
    assert len(wc["en"]) == len(wc["ar"]) > 0
    assert len(epl["en"]) == len(epl["ar"]) > 0
    assert wc != epl
    assert all("World Cup" not in question for question in epl["en"])
    assert all("كأس العالم" not in question for question in epl["ar"])


def test_displayed_examples_match_current_router_support():
    from src.example_questions import EXAMPLE_QUESTIONS
    from src.query.router import route_query

    expected_paths = ("structured", "structured", "semantic")
    for question_set in EXAMPLE_QUESTIONS.values():
        for language in ("en", "ar"):
            assert tuple(
                route_query(question).path
                for question in question_set[language]
            ) == expected_paths


def test_streamlit_selects_examples_by_competition_and_season():
    source = _streamlit_source()

    assert "selected_entry.competition_id" in source
    assert "selected_entry.season_id" in source
    assert "EXAMPLE_QUESTIONS.get" in source


def test_streamlit_renders_english_left_and_arabic_right():
    source = _streamlit_source()

    assert "english_col, arabic_col = st.columns(2)" in source
    assert "with english_col:" in source
    assert "with arabic_col:" in source


def test_streamlit_scopes_rtl_to_arabic_chat_and_example_markers():
    source = _streamlit_source()

    assert '[data-testid="stChatMessage"]:has(.rtl-message-marker)' in source
    assert '[data-testid="stColumn"]:has(.rtl-example-marker)' in source
    assert "direction: rtl" in source
    assert "text-align: right" in source
    assert "unicode-bidi: isolate" in source
    assert '.rtl-message-marker, .rtl-example-marker' in source


def test_streamlit_chat_renderer_marks_only_arabic_text():
    tree = ast.parse(_streamlit_source())
    renderer = _function(tree, "render_chat_content")

    arabic_checks = [
        call
        for call in ast.walk(renderer)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "contains_arabic"
    ]
    assert len(arabic_checks) == 1

    markdown_calls = [
        call
        for call in ast.walk(renderer)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "markdown"
    ]
    assert any(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "content"
        and not call.keywords
        for call in markdown_calls
    )

    renderer_calls = _calls_named(tree, "render_chat_content")
    assert len(renderer_calls) == 3


def test_mixed_script_examples_remain_unmodified():
    from src.example_questions import EXAMPLE_QUESTIONS
    from src.generation.prompt import contains_arabic

    wc_arabic = EXAMPLE_QUESTIONS[(43, 106)]["ar"]
    epl_arabic = EXAMPLE_QUESTIONS[(2, 27)]["ar"]

    assert "كيف لعبت Argentina في المباراة النهائية؟" in wc_arabic
    assert "كيف لعب Leicester City خلال الموسم؟" in epl_arabic
    assert all(contains_arabic(question) for question in wc_arabic + epl_arabic)
    assert contains_arabic("How did Argentina play in the final?") is False


def test_example_button_uses_same_submission_pipeline_as_chat_input():
    tree = ast.parse(_streamlit_source())
    submit = _function(tree, "submit_question")

    answer_calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "rag"
        and call.func.attr == "answer_question"
    ]
    assert len(answer_calls) == 1
    assert answer_calls[0] in list(ast.walk(submit))

    chat_input_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.NamedExpr)
        and isinstance(node.test.value, ast.Call)
        and isinstance(node.test.value.func, ast.Attribute)
        and node.test.value.func.attr == "chat_input"
    ]
    assert len(chat_input_branches) == 1
    assert len(_calls_named(chat_input_branches[0], "submit_question")) == 1

    button_branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == "button"
    ]
    assert button_branches
    for branch in button_branches:
        submit_calls = _calls_named(branch, "submit_question")
        assert len(submit_calls) == 1
        assert isinstance(submit_calls[0].args[0], ast.Name)
        assert submit_calls[0].args[0].id == "example"

        direct_calls = [
            statement.value
            for statement in branch.body
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
        ]
        submit_index = direct_calls.index(submit_calls[0])
        rerun_index = next(
            index
            for index, call in enumerate(direct_calls)
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "rerun"
        )
        assert submit_index < rerun_index
