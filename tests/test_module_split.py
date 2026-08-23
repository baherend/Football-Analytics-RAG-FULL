"""
test_module_split.py -- Structural Cleanup Phases A & B: proves the final
retrieval/routing module architecture actually exists as real, importable
Python modules (src/retrieval/search.py, src/query/router.py), and that the
legacy 06_retrieve_context.py compatibility wrapper is gone -- not just a
proposed target on paper.

This is a compatibility/architecture test, not a behavior test -- behavior
equivalence is proven separately by the existing test_router.py /
test_artifact_paths.py / test_faithfulness_baseline.py / test_structured.py
suites, run against the real module locations production code now imports
directly.
"""

from __future__ import annotations

from pathlib import Path


def test_retrieval_and_router_modules_expose_expected_public_symbols():
    from src.retrieval.search import hybrid_search, bm25_search, dense_search, build_context
    from src.query.router import (
        Route, RoutedResult, route_query, execute_route, route_and_execute,
    )

    assert callable(hybrid_search)
    assert callable(bm25_search)
    assert callable(dense_search)
    assert callable(build_context)
    assert callable(route_query)
    assert callable(execute_route)
    assert callable(route_and_execute)
    assert Route is not None
    assert RoutedResult is not None


def test_legacy_wrapper_no_longer_exists():
    """
    Structural Cleanup Phase B: 06_retrieve_context.py was a temporary
    compatibility wrapper re-exporting symbols now owned by
    src/retrieval/search.py and src/query/router.py. Once every production
    caller (chat.py, 07_prompting.py) and test migrated to import those
    modules directly, the wrapper was deleted -- this pins that it stays
    deleted, rather than silently reappearing.
    """
    assert not Path("06_retrieve_context.py").exists(), (
        "06_retrieve_context.py exists again -- Structural Cleanup Phase B "
        "removed it once no production code, test, or dev script depended "
        "on it any longer; reintroducing it recreates the compatibility "
        "wrapper this phase eliminated."
    )


# ---------------------------------------------------------------------------
# Why the test suite patches `src.retrieval.search` and not the owner modules
# ---------------------------------------------------------------------------
#
# Phase 6 set out to retarget the 16 `monkeypatch.setattr(search, ...)` sites
# onto the modules that own each symbol (bm25.py / dense.py / ...). Inspection
# stopped it: those patches are not a transitional artifact, they are the only
# *effective* target, for two independently measured reasons.
#
#   1. `_load_chunks` (14 of the 16 sites) is DEFINED in search.py -- it is not
#      a re-export at all, so there is no owner module to retarget to. bm25.py
#      and safeguards.py deliberately lazy-import back into search.py to call
#      it, because src/evaluation/retrieval_evaluator.py resets
#      `_bm25_cache`/`_chunks_cache` by reassigning them on the search module
#      object. That is a documented architecture decision, not drift.
#
#   2. `bm25_search`/`dense_search` ARE re-exports, but `hybrid_search()` is
#      defined in search.py and calls them as bare globals, so Python resolves
#      them through search.py's namespace. Rebinding the owner module has no
#      effect on that lookup -- measured end-to-end: an owner-module stub was
#      invoked 0 times while real production retrieval ran.
#
# The danger this pins is silence: retargeting would leave these tests GREEN
# while they stopped stubbing anything, quietly exercising the real index. The
# tests below assert the mechanism directly (no retrieval, no I/O) so a future
# facade-removal phase cannot re-make that mistake by inspection alone.


def test_load_chunks_is_owned_by_search_not_re_exported():
    """The 14 `_load_chunks` patch sites already target the defining module."""
    import ast

    import src.retrieval.search as search

    tree = ast.parse(Path("src/retrieval/search.py").read_text(encoding="utf-8"))
    defined_here = {
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_load_chunks" in defined_here, (
        "_load_chunks moved out of search.py -- the evaluator's cache-reset "
        "contract (reassigning _chunks_cache on this module) and every test "
        "patching search._load_chunks depend on it being defined here."
    )
    assert search._load_chunks.__module__ == "src.retrieval.search"


def test_hybrid_search_resolves_retrievers_through_search_globals():
    """`hybrid_search` is defined in search.py, so its global lookups go
    through search.py's namespace -- which is why tests patch it there."""
    import src.retrieval.search as search

    assert search.hybrid_search.__globals__ is vars(search), (
        "hybrid_search no longer resolves globals through src.retrieval.search"
    )
    for name in ("bm25_search", "dense_search"):
        assert name in search.hybrid_search.__globals__


def test_patching_the_owner_module_does_not_reach_hybrid_search(monkeypatch):
    """THE finding that stopped Phase 6: rebinding bm25.bm25_search leaves
    hybrid_search's resolved binding untouched, so retargeting these patches
    would silently stop stubbing anything instead of failing loudly."""
    import src.retrieval.bm25 as bm25
    import src.retrieval.search as search

    sentinel = object()
    monkeypatch.setattr(bm25, "bm25_search", sentinel)

    assert search.hybrid_search.__globals__["bm25_search"] is not sentinel, (
        "patching the owner module now DOES reach hybrid_search -- the "
        "retrieval lookup path changed; Phase 6's STOP verdict should be "
        "re-evaluated."
    )


def test_patching_the_facade_does_reach_hybrid_search(monkeypatch):
    """The positive half: patching search is what the runtime actually sees.
    Keeps the test above from passing for a trivial reason."""
    import src.retrieval.search as search

    sentinel = object()
    monkeypatch.setattr(search, "bm25_search", sentinel)

    assert search.hybrid_search.__globals__["bm25_search"] is sentinel
