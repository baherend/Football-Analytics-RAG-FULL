"""
test_retrieval_context_boundary.py -- Phase 4 characterization + boundary
contracts for the retrieval / Context Engineering split.

Written BEFORE the split, to pin current behavior.

## Why these tests are hermetic rather than pinning absolute chunk IDs

The pre-change assessment measured that `hybrid_search()`'s output for the
team-style query varies run-to-run (documented in PROJECT_MEMORY.md as
order-dependent retrieval variance). Pinning absolute production chunk IDs
would therefore be flaky and would hit the production Chroma store on every
run. Instead these tests pin the *invariants the split must preserve*, using a
stubbed retrieval pipeline so they are deterministic:

    select_relevant_chunks(q, hybrid_candidates(q, k), k) == hybrid_search(q, k)

The absolute 8-case x 4-k production IDs were captured separately as
pre/post evidence for the phase (0/32 mismatches), which is the measurement the
expensive multilingual benchmark would otherwise have provided.

## The k-coupling this guards

`k` is NOT purely a context-engineering concern: three retrieval safeguards
consume it to test top-k membership (`results[:k]`) and to choose an insertion
position (`_ensure_match_summary` inserts at `min(k-1, len(existing))`). The
candidate pool is therefore **k-dependent** (measured: 42 candidates at k=1 vs
44 at k>=3 for one query). Composing as `select(candidates(q, k), k)` is
behavior-preserving; "retrieve one deep pool, then select at several k" is NOT
-- that produced 2/21 mismatches in the assessment.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import src.retrieval.search as search
from src.context.selection import select_relevant_chunks


# --- Deterministic stub pipeline --------------------------------------------


def _chunk(chunk_id, rank, text, **meta):
    # `rank` is required by reciprocal_rank_fusion(); `source` is required by
    # its `appears_in` bookkeeping. Shapes match what bm25_search() emits.
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {"document_id": f"{chunk_id}-doc", "level": "1", **meta},
        "score": 0.5,
        "rank": rank,
        "source": "bm25",
    }


# Deliberately varied: distinct facets so coverage selection has real work to
# do, and a shared entity so entity grounding participates.
_POOL = [
    _chunk("c1", 1, "Argentina pressed high and controlled possession in the final.",
           team_name="Argentina"),
    _chunk("c2", 2, "Argentina scored twice from open play in the second half.",
           team_name="Argentina"),
    _chunk("c3", 3, "France countered quickly through the wings.", team_name="France"),
    _chunk("c4", 4, "Argentina committed several tactical fouls in midfield.",
           team_name="Argentina"),
    _chunk("c5", 5, "The final went to penalties after extra time.", team_name="Argentina"),
    _chunk("c6", 6, "France substituted three attackers before extra time.", team_name="France"),
    _chunk("c7", 7, "Argentina goalkeeper made two decisive saves.", team_name="Argentina"),
]

_QUERIES = [
    "How did Argentina play in the final?",
    "Argentina possession and fouls",
    "penalties after extra time",
    "France counter attacks wings",
    "Argentina goalkeeper saves in the final",
    "tactical fouls midfield possession penalties",
    "unrelated cricket scorecard",
    "Argentina France final substitutions and saves",
]

K_VALUES = (1, 3, 5, 10)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Make steps 1-8 deterministic while leaving their real wiring intact.

    Only the leaf retrievers are stubbed; RRF, rerank and every safeguard run
    for real, so the k-dependent safeguard behavior this phase must preserve is
    still exercised.
    """
    monkeypatch.setattr(search, "bm25_search",
                        lambda query, k=20, artifact_paths=None: list(_POOL))
    monkeypatch.setattr(search, "dense_search",
                        lambda query, k=20, level_filter=None, artifact_paths=None: [])
    # Safeguards need the chunk store; give them the same deterministic pool.
    monkeypatch.setattr(search, "_load_chunks", lambda path=None: list(_POOL))
    return None


# --- A. Characterization: composition identity ------------------------------


@pytest.mark.parametrize("query", _QUERIES)
@pytest.mark.parametrize("k", K_VALUES)
def test_selection_over_candidates_equals_hybrid_search(query, k, stub_pipeline):
    """THE contract this phase rests on: splitting candidate generation from
    selection must not change the selected chunks, for every k."""
    composed = [c["chunk_id"] for c in search.hybrid_search(query, k=k)]
    candidates = search.hybrid_candidates(query, k=k)
    inverted = [c["chunk_id"] for c in select_relevant_chunks(query, candidates, max_chunks=k)]
    assert inverted == composed, (
        f"split changed selection for {query!r} at k={k}: "
        f"composed={composed} inverted={inverted}"
    )


@pytest.mark.parametrize("k", K_VALUES)
def test_hybrid_search_never_exceeds_k(k, stub_pipeline):
    """hybrid_search stays a SELECTED result, never the raw candidate pool --
    the regression that would blow up prompt context and the benchmark."""
    for query in _QUERIES:
        assert len(search.hybrid_search(query, k=k)) <= k


def test_candidate_pool_is_larger_than_the_selection(stub_pipeline):
    """Sanity: the two really are different stages, so the test above is not
    vacuously comparing a pool to itself."""
    query = "Argentina possession and fouls"
    assert len(search.hybrid_candidates(query, k=3)) > len(search.hybrid_search(query, k=3))


# --- B. Ordering and object identity ----------------------------------------


def test_candidate_ordering_is_preserved_through_selection(stub_pipeline):
    """Selection may drop candidates but must not reorder the pool it is given,
    and its output order must be a decision of selection alone."""
    query = "Argentina possession and fouls"
    candidates = search.hybrid_candidates(query, k=5)
    before = [c["chunk_id"] for c in candidates]
    select_relevant_chunks(query, candidates, max_chunks=5)
    assert [c["chunk_id"] for c in candidates] == before, "candidate pool was reordered in place"


def test_selection_returns_the_same_objects_not_copies(stub_pipeline):
    """Identity matters: selection dedupes with id(chunk), and downstream
    EvidencePack keeps the original dicts verbatim for provenance."""
    query = "Argentina possession and fouls"
    candidates = search.hybrid_candidates(query, k=5)
    selected = select_relevant_chunks(query, candidates, max_chunks=5)
    for item in selected:
        assert any(item is candidate for candidate in candidates), (
            "selection returned a copy -- provenance/identity contract broken"
        )


def test_candidates_are_not_mutated_by_selection(stub_pipeline):
    import copy

    query = "Argentina possession and fouls"
    candidates = search.hybrid_candidates(query, k=5)
    snapshot = copy.deepcopy(candidates)
    select_relevant_chunks(query, candidates, max_chunks=5)
    assert candidates == snapshot, "selection mutated candidate dicts"


# --- C. k stays a retrieval parameter ---------------------------------------


def test_candidate_generation_still_consumes_k(stub_pipeline):
    """`k` must remain a retrieval parameter: the safeguards use it for top-k
    membership and insertion position. If hybrid_candidates stopped taking k,
    the 'retrieve deep once, select at many k' anti-pattern would silently
    become possible -- it measured 2/21 mismatches in the assessment."""
    import inspect

    assert "k" in inspect.signature(search.hybrid_candidates).parameters


# --- D. Public contracts the evaluator and callers depend on ----------------


def test_hybrid_search_remains_the_composed_public_api():
    """src/evaluation/retrieval_evaluator.py calls hybrid_search() per K and
    treats the whole return as the ranked set. It must stay composed."""
    assert callable(search.hybrid_search)
    assert callable(search.hybrid_candidates)


def test_retrieve_context_returns_selected_chunks_not_candidates(stub_pipeline):
    result = search.retrieve_context("Argentina possession and fouls", k=3)
    assert result["num_chunks"] <= 3
    assert len(result["chunks"]) == result["num_chunks"]


def test_no_candidate_type_or_wrapper_was_introduced():
    """Condition: the existing list[dict] contract is sufficient."""
    source = pathlib.Path("src/retrieval/search.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert not classes, f"a new type was introduced in search.py: {classes}"


def test_selection_module_still_owns_selection_only():
    """context/selection.py must remain pure: no retrieval work, no I/O."""
    source = pathlib.Path("src/context/selection.py").read_text(encoding="utf-8")
    for forbidden in ("chromadb", "pickle", "bm25", "dense_search", "open(", "Path("):
        assert forbidden not in source, f"{forbidden} appeared in context/selection.py"
