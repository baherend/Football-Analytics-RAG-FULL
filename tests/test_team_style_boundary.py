"""
test_team_style_boundary.py -- Phase 5 contracts for the shared team-style
classifier.

Before this phase `src/query/intent.py` imported `_detect_team_style_query`
from `src.retrieval.search`, so the UNDERSTAND stage depended on RETRIEVE --
backwards relative to the runtime flow (UNDERSTAND -> PLAN -> RETRIEVE).
Moving the classifier into `src/query/` would only have flipped the arrow, as
`src/retrieval/safeguards.py` uses the same detectors for its team-style
document boost. The classifier is pure text classification belonging to
neither layer, so it now lives in the neutral shared module `src/team_style.py`
-- the convention `src/stage_taxonomy.py` and `src/artifacts.py` already use.

These tests pin the *direction* and the *purity*, not the classification rules
themselves: those are exhaustively covered by tests/test_arabic_safeguards.py,
which reaches the same functions through the `src.retrieval.search` compat
re-export and therefore also guards that the re-export keeps working.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


def _imports_of(path: str) -> set[str]:
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


# --- A. The reverse dependency stays closed ---------------------------------


def test_understanding_does_not_import_retrieval():
    """THE edge this phase closed. src/query/intent.py classifies queries; it
    must not reach into the retrieval package to do so."""
    offenders = [m for m in _imports_of("src/query/intent.py")
                 if m.startswith("src.retrieval")]
    assert not offenders, (
        f"src/query/intent.py imports retrieval again: {offenders} -- the "
        "understanding -> retrieval reverse dependency has reopened."
    )


def test_no_query_module_imports_retrieval_for_classification():
    """router.py legitimately imports `hybrid_search` (execution calling the
    RETRIEVE stage follows the runtime flow). It must not import the
    *classifier* from retrieval -- that is the reverse edge in disguise."""
    offenders = []
    for path in pathlib.Path("src/query").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src.retrieval"):
                for alias in node.names:
                    if "team_style" in alias.name:
                        offenders.append(f"{path.as_posix()}: {node.module}.{alias.name}")
    assert not offenders, f"team-style classification imported from retrieval: {offenders}"


def test_shared_module_depends_on_neither_layer():
    """A shared module that imports a layer is not shared -- it would just
    relocate the cycle."""
    offenders = [m for m in _imports_of("src/team_style.py")
                 if m.startswith(("src.retrieval", "src.query", "src.context",
                                  "src.generation", "src.orchestration"))]
    assert not offenders, f"src/team_style.py reached into a pipeline layer: {offenders}"


def test_shared_module_is_pure_text_classification():
    """No retrieval, no I/O, no chunk store: the property that made relocation
    safe in the first place."""
    source = pathlib.Path("src/team_style.py").read_text(encoding="utf-8")
    for forbidden in ("chromadb", "pickle", "bm25", "dense_search", "_get_chunks",
                      "_load_chunks", "ArtifactPaths", "open(", "Path("):
        assert forbidden not in source, f"{forbidden} appeared in src/team_style.py"


# --- B. One definition, reached through every surviving path ----------------


def test_classifier_is_defined_once():
    """The alternative to moving it was duplicating it into both layers. That
    would drift; this pins that it did not happen."""
    defs = []
    for path in pathlib.Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                    "_detect_team_style_query", "_detect_team_style_entities"):
                defs.append(f"{path.as_posix()}::{node.name}")
    assert sorted(defs) == [
        "src/team_style.py::_detect_team_style_entities",
        "src/team_style.py::_detect_team_style_query",
    ], f"team-style detection is defined in unexpected places: {defs}"


@pytest.mark.parametrize("name", [
    "_LATIN_ENTITY_SPAN",
    "_STYLE_KEYWORDS",
    "_STYLE_KEYWORDS_AR",
    "_detect_team_style_entities",
    "_detect_team_style_query",
    "_extract_latin_entity_spans",
    "_normalize_arabic_for_matching",
])
def test_retrieval_compat_reexport_still_resolves(name):
    """tests/ and src/query/router.py reach these through src.retrieval.search;
    the relocation must not break that surface."""
    import src.retrieval.search as search
    import src.team_style as team_style

    assert getattr(search, name) is getattr(team_style, name), (
        f"search.{name} no longer refers to the shared definition"
    )


def test_all_call_sites_agree():
    """intent.py, router.py, safeguards.py and the compat re-export must all be
    the same object -- otherwise a monkeypatch or a fix would reach only some."""
    import src.query.intent as intent
    import src.query.router as router
    import src.retrieval.safeguards as safeguards
    import src.retrieval.search as search
    import src.team_style as team_style

    assert intent._detect_team_style_query is team_style._detect_team_style_query
    assert router._detect_team_style_query is team_style._detect_team_style_query
    assert search._detect_team_style_query is team_style._detect_team_style_query
    assert safeguards._detect_team_style_entities is team_style._detect_team_style_entities


# --- C. Behavior is unchanged by the move -----------------------------------
#
# Characterization spot-checks. Full coverage lives in
# tests/test_arabic_safeguards.py; these run against the shared module directly
# so a future removal of the compat re-export cannot leave the logic untested.


@pytest.mark.parametrize("query,expected", [
    ("Describe Argentina's playing style", "Argentina"),
    ("What were France's passing patterns and most common formations?", "France"),
    ("How many goals did Messi score?", None),
    ("Who won the final?", None),
])
def test_detection_behavior_survived_the_move(query, expected):
    from src.team_style import _detect_team_style_query

    assert _detect_team_style_query(query) == expected


def test_arabic_normalization_survived_the_move():
    from src.team_style import _normalize_arabic_for_matching

    # Alef variants unify; alef maksura -> ya; whitespace runs collapse.
    assert _normalize_arabic_for_matching("أإآ") == "ااا"
    assert _normalize_arabic_for_matching("ى") == "ي"
    assert _normalize_arabic_for_matching("a  \t b") == "a b"


def test_comparison_safeguard_still_uses_the_shared_normalizer():
    """The normalizer moved alongside the classifier because both need it.
    _detect_comparison_entities() stayed in retrieval and must still work."""
    import src.retrieval.search as search

    # Entities come back lowercased -- _detect_comparison_entities() matches
    # against query.lower(). Unchanged by this phase (pinned here as-is rather
    # than "fixed", which would be an unrelated behavior change).
    assert search._detect_comparison_entities("Compare Messi and Mbappe") == ["messi", "mbappe"]
