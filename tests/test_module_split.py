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
