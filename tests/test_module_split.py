"""
test_module_split.py -- Structural Cleanup Phase A: proves the retrieval/
routing module boundary described in the architecture review actually
exists as real, importable Python modules (src/retrieval/search.py,
src/query/router.py), not just a proposed target on paper.

This is a compatibility/architecture test, not a behavior test -- behavior
equivalence is proven separately by the existing test_router.py /
test_artifact_paths.py / test_faithfulness_baseline.py / test_structured.py
suites, unmodified in intent, run against the new module locations.
"""

from __future__ import annotations


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
