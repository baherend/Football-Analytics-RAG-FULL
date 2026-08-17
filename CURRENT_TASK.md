# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Migration Step 2 (Retrieval Split) — COMPLETE, uncommitted

`src/retrieval/search.py` was split into `bm25.py`, `dense.py`, `fusion.py`,
`safeguards.py` inside the existing `src/retrieval/` package. A `service.py`
extraction of `hybrid_search()` was tried, then reverted on a follow-up
architecture review (it required a permanent reverse call-time dependency
back into `search.py` to stay test-monkeypatch compatible — more coupling
debt than keeping the function in `search.py`, for the module meant to be
this migration's centerpiece). `search.py` is `src/retrieval/`'s
transitional compatibility boundary: re-export layer + index loading +
`hybrid_search()` orchestration + context-building — not final architecture.
Verified after both the original split and the correction: targeted tests
316/316 (baseline match), full regression 555 passed / 5 skipped (baseline
match), EN/MSA/EGY document IDs byte-identical pre/post, EN/MSA/EGY
Hit@K/AllReq@K/NDCG@K/MRR at K=1/3/5/10 exactly reproduced, all 11 tracked
`output/chroma_db/` files match HEAD, `output/competitions/` untouched. Not
committed — awaiting user review. See `PROJECT_MEMORY.md`'s Architecture
Decisions and Completed Milestones for the durable record; details are not
repeated here.

## Next Step

Migration Step 3 (per `docs/architecture/overview.md` §11): split
`src/query/router.py`'s query classification from its execution/
orchestration. Before starting: decide, with fresh repository evidence,
whether `router.py` has the same kind of test-monkeypatch coupling that
shaped this phase's design (grep for `import src.query.router as` /
`router\.<name>` module-attribute patches first, the same way this phase's
key finding was discovered, rather than assuming a clean split is safe).

## Out of Scope (unchanged discipline, next phase too)

Any algorithm change, Arabic feature addition, entity-normalization work,
embedding change, BM25/Dense/RRF tuning, prompt/generation change, unrelated
cleanup, dependency change, moving to a `rag/` namespace prematurely,
touching `output/competitions/`, committing or pushing without being asked.
