"""
test_orchestration_boundary.py -- Phase B contracts.

B0: one canonical module identity for 07_prompting.py.
B1: the shared answer policy lives in src/orchestration/ and both runtime
    entry points use it, without erasing genuine interface differences.
"""

from __future__ import annotations

import ast
import pathlib
from importlib import import_module
from types import SimpleNamespace

import pytest

from src.orchestration.policy import (
    NO_CONTEXT_MESSAGE,
    assemble_context,
    finalize_answer,
    should_refuse,
)


# --- B0: one module, one identity, one instance -----------------------------


def test_chat_and_direct_import_share_one_module_instance():
    """chat.py used to load 07_prompting.py by file path as "prompting",
    producing a second module object: module-level state and monkeypatches
    then applied to only one copy."""
    chat = import_module("chat")
    prompting = import_module("07_prompting")

    assert chat.prompting_mod is prompting, (
        "chat.py holds a different module instance of 07_prompting.py -- the "
        "dual identity is back."
    )
    assert chat.prompting_mod.__name__ == "07_prompting"


def test_no_stale_prompting_alias_is_registered():
    import sys

    import_module("chat")
    assert "prompting" not in sys.modules, (
        'sys.modules["prompting"] is registered again -- the file-path loader '
        "or an equivalent alias has returned."
    )


def test_chat_no_longer_defines_a_file_path_loader():
    chat = import_module("chat")
    assert not hasattr(chat, "load_module_from_path"), (
        "the file-path module loader that created the duplicate instance is back"
    )


# --- B1: both entry points use the shared policy ----------------------------


def _imports_of(path: str) -> set[str]:
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


@pytest.mark.parametrize("entry_point", ["chat.py", "07_prompting.py"])
def test_both_entry_points_import_the_shared_policy(entry_point):
    assert any(m.startswith("src.orchestration") for m in _imports_of(entry_point)), (
        f"{entry_point} no longer routes through the shared orchestration policy"
    )


def test_authoritative_header_is_defined_once():
    """The banner used to be duplicated verbatim in both entry points."""
    banner = "## Authoritative Data (Verified from Match Facts)"
    offenders = [
        f for f in ("chat.py", "07_prompting.py")
        if banner in pathlib.Path(f).read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"the authoritative-data banner is inlined again in {offenders} instead "
        "of coming from src/orchestration/policy.py"
    )


# --- B1: dependency direction ----------------------------------------------


def test_runtime_layers_do_not_import_orchestration():
    """Orchestration sits above the runtime stages; nothing below may import it."""
    offenders = []
    for root in ("src/retrieval", "src/query", "src/context", "src/generation",
                 "src/verification", "src/knowledge", "src/evaluation"):
        for path in pathlib.Path(root).rglob("*.py"):
            for module in _imports_of(str(path)):
                if module.startswith("src.orchestration"):
                    offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, f"reverse dependency into orchestration: {offenders}"


def test_orchestration_does_not_import_interfaces_or_evaluation():
    """It must not depend on CLI/Streamlit internals or on evaluation."""
    offenders = []
    for path in pathlib.Path("src/orchestration").rglob("*.py"):
        for module in _imports_of(str(path)):
            if module.startswith(("src.evaluation", "chat", "streamlit")):
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, f"orchestration reached upward/sideways: {offenders}"


def test_orchestration_does_not_own_routing_prompting_or_generation():
    """Those carry interface-level monkeypatch contracts and stay in adapters.
    Guards against the package growing into a god orchestrator."""
    forbidden = {"src.query.router", "src.retrieval.search", "src.generation.provider"}
    offenders = []
    for path in pathlib.Path("src/orchestration").rglob("*.py"):
        for module in _imports_of(str(path)):
            if module in forbidden:
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, (
        f"orchestration absorbed routing/generation machinery: {offenders}"
    )


# --- B1: policy behavior ----------------------------------------------------


def _routed(structured=None, chunks=None, answerability=None, context=""):
    return SimpleNamespace(structured_result=structured, semantic_chunks=chunks or [],
                           answerability=answerability, context=context)


def test_structured_block_is_marked_authoritative_and_comes_first():
    sr = SimpleNamespace(status="resolved", explanation="Messi scored 8 goals.", values=None)
    out = assemble_context(_routed(structured=sr))
    assert out.has_structured is True
    assert out.context.startswith("## Authoritative Data")
    assert "Messi scored 8 goals." in out.context


def test_conversation_block_is_prepended_not_merged_into_authoritative():
    sr = SimpleNamespace(status="resolved", explanation="X.", values=None)
    out = assemble_context(_routed(structured=sr), conversation_context="## Earlier turns\nfoo")
    assert out.full_context.startswith("## Earlier turns")
    assert out.full_context.index("## Earlier turns") < out.full_context.index("## Authoritative Data")
    assert out.context.startswith("## Authoritative Data"), (
        "conversation text leaked into the authoritative evidence block"
    )


def test_fallback_context_preserves_each_interface_difference():
    """CLI falls back to its own message; answer_question falls back to the
    RoutedResult's context. This difference is real and must survive."""
    empty = _routed(context="No relevant documents found.")
    assert assemble_context(empty).context == NO_CONTEXT_MESSAGE
    assert assemble_context(empty, fallback_context=empty.context).context == (
        "No relevant documents found."
    )


def test_refusal_gate_matches_the_generation_policy():
    assert should_refuse(_routed(answerability=SimpleNamespace(status="unanswerable"))) is True
    usable = SimpleNamespace(status="resolved", explanation="ok", values=None)
    assert should_refuse(_routed(structured=usable,
                                 answerability=SimpleNamespace(status="unanswerable"))) is False


def test_finalize_corrects_contradicted_numeric_claims():
    sr = SimpleNamespace(status="resolved", explanation="Jamie Vardy's total goals is 24.",
                         aggregated_value=24, query=SimpleNamespace(metric="goals"), values=None)
    out = finalize_answer("Jamie Vardy scored 11 goals.", _routed(structured=sr), True)
    assert out.corrected is True
    assert out.validation is not None and out.validation.is_valid is False
    assert out.answer != "Jamie Vardy scored 11 goals."


def test_finalize_leaves_supported_answers_alone():
    sr = SimpleNamespace(status="resolved", explanation="Jamie Vardy's total goals is 24.",
                         aggregated_value=24, query=SimpleNamespace(metric="goals"), values=None)
    out = finalize_answer("Jamie Vardy scored 24 goals.", _routed(structured=sr), True)
    assert out.corrected is False
    assert out.answer == "Jamie Vardy scored 24 goals."


def test_finalize_verification_failure_never_breaks_the_pipeline():
    """Best-effort contract inherited from both original implementations."""
    exploding = SimpleNamespace(status="resolved", explanation=None,
                                query=property(lambda self: 1 / 0), values=None)
    out = finalize_answer("some answer", _routed(structured=exploding), True)
    assert out.answer == "some answer"


def test_citations_only_come_from_retrieved_evidence():
    chunks = [{"chunk_id": "c1", "text": "t",
               "metadata": {"document_id": "d1", "level": "team", "team_name": "France"}}]
    out = finalize_answer("answer", _routed(chunks=chunks), False)
    assert {c.get("chunk_id") for c in out.citations if c.get("chunk_id")} <= {"c1"}
    assert finalize_answer("answer", _routed(), False).citations == []


# --- Provider / model vocabulary: single source of truth ---------------------
#
# The Streamlit model list used to be hardcoded inside streamlit_app.py's
# selectbox, disconnected from MODELS, so the two could drift silently. It now
# comes from src/generation/provider.py. These pin that, and pin the dispatch
# rule both interfaces depend on.


def test_streamlit_model_list_comes_from_the_provider_module():
    """The UI must not hardcode model IDs again."""
    source = pathlib.Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "GROQ_DIRECT_MODELS" in source, (
        "streamlit_app.py no longer sources its model list from the provider "
        "module -- the vocabulary can drift again."
    )
    for hardcoded in ("llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"):
        assert f'"{hardcoded}"' not in source, (
            f"{hardcoded} is hardcoded in streamlit_app.py again"
        )


def test_every_offered_model_resolves_to_a_known_provider():
    """Whatever any interface offers must dispatch to a provider that has a
    declared API-key environment variable."""
    from src.generation.provider import PROVIDER_KEYS, offered_models, resolve_model

    for model in offered_models():
        provider, provider_model_id = resolve_model(model)
        assert provider in PROVIDER_KEYS, f"{model} -> unknown provider {provider}"
        assert provider_model_id, f"{model} resolved to an empty provider model id"


def test_registry_entries_are_well_formed():
    from src.generation.provider import MODELS, PROVIDER_KEYS

    for key, config in MODELS.items():
        assert set(config) == {"provider", "model"}, f"{key} has unexpected fields"
        assert config["provider"] in PROVIDER_KEYS, f"{key} names an unknown provider"
        assert config["model"], f"{key} has an empty provider model id"


def test_registry_keys_win_over_the_raw_groq_fallthrough():
    """A registry key must map through the registry; an unknown name falls
    through to Groq as a raw model id (the documented escape hatch that lets a
    Streamlit deployment set GROQ_MODEL from secrets)."""
    from src.generation.provider import resolve_model

    assert resolve_model("haiku") == ("openrouter", "anthropic/claude-3.5-haiku")
    assert resolve_model("llama") == ("groq", "llama-3.3-70b-versatile")
    assert resolve_model("some-unlisted-groq-model") == ("groq", "some-unlisted-groq-model")


def test_friendly_key_and_raw_id_resolve_to_the_same_groq_model():
    """Evidence for keeping the two menus separate rather than merging them:
    merging would list the same underlying model twice."""
    from src.generation.provider import resolve_model

    assert resolve_model("llama") == resolve_model("llama-3.3-70b-versatile")
    assert resolve_model("llama-8b") == resolve_model("llama-3.1-8b-instant")


def test_cli_model_menu_still_comes_from_the_registry():
    source = pathlib.Path("chat.py").read_text(encoding="utf-8")
    assert "prompting_mod.MODELS" in source, (
        "chat.py's /model menu no longer derives from the registry"
    )


def test_model_selection_cannot_reach_code_execution():
    """Model IDs are configuration, not executable input: resolve_model must
    only ever return data, never import or evaluate anything."""
    from src.generation import provider

    source = pathlib.Path(provider.__file__).read_text(encoding="utf-8")
    for unsafe in ("eval(", "exec(", "__import__", "importlib", "subprocess"):
        assert unsafe not in source, f"{unsafe} present in provider dispatch"
    # A hostile "model id" is passed through as an opaque string only.
    hostile = "'; import os; os.system('x')  #"
    assert provider.resolve_model(hostile) == ("groq", hostile)
