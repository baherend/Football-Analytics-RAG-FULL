"""
test_evaluation_boundary.py -- Migration Step 7 contracts for the evaluation
layer.

Pins the cross-cutting dependency rule and the integrity of the protected
benchmark baseline. Metric *behavior* is covered by test_retrieval_evaluator.py
and is not duplicated here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
from importlib import import_module

import pytest

EVALUATION_PACKAGE = "src.evaluation"

# Every runtime layer. None of these may reach into evaluation.
RUNTIME_ROOTS = [
    pathlib.Path("src/retrieval"),
    pathlib.Path("src/query"),
    pathlib.Path("src/context"),
    pathlib.Path("src/generation"),
    pathlib.Path("src/verification"),
    pathlib.Path("src/knowledge"),
]
RUNTIME_SCRIPTS = [
    pathlib.Path("chat.py"),
    pathlib.Path("streamlit_app.py"),
    pathlib.Path("07_prompting.py"),
    pathlib.Path("01_documents.py"),
    pathlib.Path("02_preprocessing.py"),
    pathlib.Path("03_chunking.py"),
    pathlib.Path("04_vector_representation.py"),
    pathlib.Path("05_create_chroma_store.py"),
    pathlib.Path("generate_documents.py"),
    pathlib.Path("rebuild.py"),
]


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _runtime_files() -> list[pathlib.Path]:
    files = [p for root in RUNTIME_ROOTS for p in root.rglob("*.py")]
    files += [p for p in RUNTIME_SCRIPTS if p.exists()]
    files += [p for p in pathlib.Path("src").glob("*.py")]      # flat src modules
    return files


# --- The cross-cutting rule -------------------------------------------------


def test_runtime_never_imports_evaluation():
    """evaluation -> runtime is allowed; runtime -> evaluation never is.
    Evaluation must not become part of the online query path.
    """
    offenders = []
    for path in _runtime_files():
        for module in _imported_modules(path):
            if module.startswith(EVALUATION_PACKAGE):
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, f"runtime imported evaluation: {offenders}"


def test_runtime_never_imports_the_test_package_either():
    """The evaluation library used to live in tests/, which is why three
    src/retrieval modules documented a contract against a test-folder path.
    Runtime must not import `tests.*` either."""
    offenders = []
    for path in _runtime_files():
        for module in _imported_modules(path):
            if module == "tests" or module.startswith("tests."):
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, f"runtime imported the test package: {offenders}"


def test_evaluation_modules_import_cleanly():
    """No cycles, and the package is importable without pytest running."""
    for name in (
        "src.evaluation",
        "src.evaluation.retrieval_evaluator",
        "src.evaluation.diagnostics",
        "src.evaluation.benchmark",
        "src.evaluation.ground_truth",
        "src.evaluation.ground_truth.semantic",
        "src.evaluation.ground_truth.answerability",
        "src.evaluation.ground_truth.multilingual",
        "src.evaluation.ground_truth.registry",
    ):
        assert import_module(name) is not None


def test_no_evaluation_library_modules_left_in_tests():
    """tests/ should now contain only test modules (plus __init__)."""
    strays = [
        p.name for p in pathlib.Path("tests").glob("*.py")
        if not p.name.startswith("test_") and p.name != "__init__.py"
    ]
    assert not strays, f"non-test modules still in tests/: {strays}"


def test_no_stale_references_to_the_old_tests_locations():
    """Every import site and patch-target literal must have been rewritten."""
    old = [
        "tests.retrieval_evaluator", "tests.semantic_ground_truth",
        "tests.answerability_ground_truth", "tests.multilingual_retrieval_cases",
        "tests.ground_truth_registry", "tests.multilingual_diagnostics",
        "tests.run_multilingual_diagnostics", "tests.run_phase4_phase5",
        "tests.evaluation_benchmark",
    ]
    this_file = pathlib.Path(__file__).resolve()
    offenders = []
    for path in list(pathlib.Path("tests").glob("*.py")) + list(pathlib.Path("src").rglob("*.py")):
        if path.resolve() == this_file:
            continue          # this guard necessarily spells the old names out
        source = path.read_text(encoding="utf-8")
        for name in old:
            if name in source:
                offenders.append(f"{path.as_posix()}: {name}")
    assert not offenders, f"stale references to pre-Step-7 locations: {offenders}"


# --- Protected benchmark baseline -------------------------------------------


def test_semantic_ground_truth_is_unchanged():
    """The WC2022 baseline is protected data: 24 cases, fixed IDs, fixed
    payload. It is never edited to make a test pass."""
    from src.evaluation.ground_truth.semantic import (
        EXPECTED_CASE_IDS,
        SEMANTIC_GROUND_TRUTH,
    )

    assert len(SEMANTIC_GROUND_TRUTH) == 24
    assert len(EXPECTED_CASE_IDS) == 24

    # Payload hash captured from the pre-move baseline, so any edit to the
    # protected benchmark data fails loudly rather than silently shifting
    # every downstream metric.
    payload = hashlib.sha256(
        json.dumps(SEMANTIC_GROUND_TRUTH, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert payload.startswith("8664cce1bd6d21f79663"), (
        f"semantic ground-truth payload changed (sha256 now {payload})"
    )


def test_ground_truth_registry_still_keys_the_wc2022_identity():
    import src.evaluation.ground_truth.registry as registry

    assert sorted(registry._REGISTRY.keys()) == [(43, 106)]


def test_multilingual_bundle_still_builds_24_cases():
    from src.evaluation.ground_truth.multilingual import build_ground_truth_bundle

    for language in ("en", "ar_msa", "ar_egy"):
        bundle = build_ground_truth_bundle(language)
        assert len(bundle.cases) == 24, f"{language} bundle changed size"


@pytest.mark.parametrize(
    "metric,args,expected",
    [
        ("hit_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 3), 1.0),
        ("hit_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 1), 0.0),
        ("recall_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 3), 0.5),
        ("recall_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 5), 1.0),
        ("all_required_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 5), 1.0),
        ("all_required_at_k", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 2), 0.0),
        ("reciprocal_rank", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}), 0.5),
        ("first_relevant_document_rank", (["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}), 2),
    ],
)
def test_metric_values_unchanged_by_the_move(metric, args, expected):
    module = import_module("src.evaluation.retrieval_evaluator")
    assert getattr(module, metric)(*args) == expected


def test_ndcg_unchanged_by_the_move():
    from src.evaluation.retrieval_evaluator import ndcg_at_k

    assert round(ndcg_at_k(["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"}, 5), 8) == 0.62405052


# --- Artifact-safety contract still reachable -------------------------------


def test_chroma_artifact_safety_helpers_moved_with_the_evaluator():
    """AGENT_RULES.md §5 points agents at temporary_chroma_copy(); it must
    still exist at the documented (new) location."""
    module = import_module("src.evaluation.retrieval_evaluator")
    for name in ("temporary_chroma_copy", "reset_retrieval_caches",
                 "load_retrieval_module", "LegacyTemporaryChromaArtifactPaths",
                 "TemporaryChromaArtifactPaths"):
        assert hasattr(module, name), f"retrieval_evaluator lost {name}"
