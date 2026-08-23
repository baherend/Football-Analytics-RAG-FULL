# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Phase 7 (`hybrid_search()` ownership reassessment) — STOPPED on evidence

**Decision: H2 — `hybrid_search()` stays in `src/retrieval/search.py`.**
No production code changed. This was an assessment; nothing was half-migrated.

## Why H1 (move to `src/orchestration/`) is not viable

Phase 4 did make the function thin — it is now only:

    select_relevant_chunks(hybrid_candidates(q, k), k)

So the *body* is extractable. The blocker is the dependency direction. The
committed Phase B1 contract `test_runtime_layers_do_not_import_orchestration`
forbids importing `src.orchestration` from **all seven** runtime roots:
`retrieval`, `query`, `context`, `generation`, `verification`, `knowledge`,
`evaluation`. Every route out of `search.py` needs exactly that import:

| Route | Required import | Verdict |
| --- | --- | --- |
| `retrieve_context()` stays and calls it (search.py:357) | `retrieval → orchestration` | forbidden |
| `router.py` calls it (2 sites) | `query → orchestration` | forbidden |
| evaluator binds `retrieval_module.hybrid_search` | evaluator change, or a re-export that itself imports orchestration | forbidden / protected |
| keep a compat re-export in `search.py` | `retrieval → orchestration` | forbidden, and shim-only cleanliness |

There is no ordering of these that avoids the violation.

## Two findings that shrink the case for moving it at all

1. **The move would not close the `retrieval → context` inversion.** `search.py`
   still imports `build_context` for `retrieve_context()` (search.py:373)
   whether or not `hybrid_search()` leaves. The architectural gain is smaller
   than the Phase 6 write-up implied.
2. **No runtime import cycle would occur** — the
   `orchestration → verification → src.query.query_schema` chain ends at a leaf
   dataclass module. So this is a *layering-contract* violation, not an
   `ImportError`. That distinction matters: it means the constraint is a
   deliberate decision, not a technical accident, and is already machine-enforced.

## Verification

- **Production diff: empty.** `git diff -- src/ *.py` returns nothing.
- **Full suite: 809 passed / 5 skipped** — unchanged baseline.
- **Characterization parity (8 × 4 k) not run, deliberately**: no production
  code changed, so it would compare identical bytecode to itself. Per §9, the
  expensive multilingual benchmark was also skipped — no retrieval semantics,
  ordering, candidate generation, evaluator contract, or cache behavior changed.
- **No new guard test added**: the constraint is already enforced by the
  committed `test_runtime_layers_do_not_import_orchestration`.

## Prerequisites before this can be reconsidered

1. The evaluator's contract that **one** module (`src.retrieval.search`) owns
   `hybrid_search` + `bm25_search` + `dense_search` + `CHROMA_DIR` + both caches.
2. `retrieve_context()`'s residence in `search.py`.

Until those two move, `search.py` legitimately owns `hybrid_search()` and the
facade is not merely transitional.

## Not committed

No `git add`, `git commit`, or `git push` was run. `output/competitions/`
untouched.
