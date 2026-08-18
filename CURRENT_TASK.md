# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Phase 4 (retrieval/context inversion) — COMPLETE, uncommitted

`src/retrieval/search.py` now separates the two stages:

    hybrid_candidates(q, k, ...)   steps 1-8, pure retrieval, no context import
    hybrid_search(q, k, ...)       select_relevant_chunks(q, candidates, k)

Only `src/retrieval/search.py` changed in production code.

## Measured finding that shaped the design

`k` is **not** purely a context budget — three safeguards consume it for
top-k membership and insertion position, so the candidate pool is k-dependent
(42 candidates at k=1 vs 44 at k≥3 for one query). Therefore:

- `select(hybrid_candidates(q, k), k) == hybrid_search(q, k)` → **0/32**
- `select(hybrid_candidates(q, 10), k) != hybrid_search(q, k)` → **2/21**

The second composition is forbidden and guarded by test.

## What was deliberately NOT done

`search.py` still imports `context.selection`: `hybrid_search()` must stay
composed because `retrieval_evaluator.py` calls it per-K and treats the whole
return as the ranked set. Removing the edge entirely means moving composition
to `orchestration/` and repointing the evaluator — a change to the
protected-baseline harness. No `Candidate` type introduced.

## Verification

0/32 real-data parity vs pre-patch baseline; ordering + object identity
preserved; candidates unmutated; `retrieve_context()` still returns selected
chunks; router gets 5 selected (pool 44), context 2628 chars < 3000 cap;
provenance intact; roles still `['system','user']`. Targeted 292; full suite
**786 passed / 5 skipped**. Expensive benchmark **skipped** — justified below.

## Remaining debt

`search.py → context.selection` import edge (isolated to one composition
line); provider model liveness (product question); `retrieval_evaluator.py`
split; Step 3 `intent.py → retrieval`; transitional boundaries; no
`domain/`/`infrastructure/`; `resolve_output_dir` ID validation; BM25 pickle;
4 library `print()` paths.

## Next Step

Ask the product owner which Groq models are live (smallest remaining item).
The `search.py → context.selection` edge should only be closed together with
repointing the evaluator, in its own phase with a benchmark re-run.

## Out of Scope

Step 8 observability (rejected on evidence), evaluator repointing,
`retrieval_evaluator.py` split, `output/competitions/`, committing/pushing.
