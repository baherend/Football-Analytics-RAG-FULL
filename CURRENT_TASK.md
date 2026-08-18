# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Migration Step 7 (Evaluation Organization) — COMPLETE, uncommitted

Nine evaluation library modules moved from `tests/` to a new `src/evaluation/`
package, byte-identically (`git mv`, content verified). 112 references
rewritten across 17 files. `tests/` now holds only `test_*.py` + `__init__.py`.
No compatibility shims — verified nothing outside `tests/` imported them.

`runtime → evaluation` is now **structurally enforced**: an AST test proves no
runtime layer or root script imports `src.evaluation` or `tests.*`.

Verified: ground-truth payload SHA-256, 24 semantic cases, registry key
`(43, 106)`, EN/MSA/EGY bundles (24 each) and 9 metric values all unchanged;
targeted 112 + 18 boundary tests; full suite **714 passed / 5 skipped /
0 failed**; `output/chroma_db/` and `output/competitions/` untouched.

## Evidence gathered before patching

- **The dependency contract already holds**: no `src/` module and no root
  script imports `tests.*`. `runtime → evaluation` is already NO — but it is
  nowhere *enforced*, so nothing stops a future regression.
- **9 non-`test_*` modules in `tests/` are evaluation library code**, not
  tests (~5,586 lines): `semantic_ground_truth.py` (2439),
  `retrieval_evaluator.py` (1561), `evaluation_benchmark.py` (421),
  `run_multilingual_diagnostics.py` (368), `multilingual_retrieval_cases.py`
  (295), `multilingual_diagnostics.py` (271), `answerability_ground_truth.py`
  (134), `run_phase4_phase5.py` (103), `ground_truth_registry.py` (34).
- **The mislocation is a real architectural smell, not aesthetics**:
  `AGENT_RULES.md` §5 instructs agents to use
  `tests/retrieval_evaluator.py::temporary_chroma_copy()`, and three `src/`
  modules (`retrieval/search.py`, `retrieval/bm25.py`, `retrieval/safeguards.py`)
  document a load-bearing cache-reset contract against
  `tests/retrieval_evaluator.py`. Production design is constrained by a module
  living in the test folder. `docs/architecture/overview.md` §4 already states
  the goal: make these "importable without reaching into `tests/`".
- **Migration surface is fully bounded and self-contained**: 69 import
  statements and 34 string-literal patch targets, **all inside `tests/`**.
  No external consumer → no compatibility shim needed (verified, not assumed).
- **`retrieval_evaluator.py` is internally multi-responsibility** (metrics,
  case evaluation, aggregation, runtime-module loading, Chroma artifact
  safety, CLI — 12 sections). Splitting it is a *separate* concern from
  relocating it; doing both at once would violate MOVE → REWIRE → VERIFY.

## Scope

Move the 9 modules to `src/evaluation/`, byte-identically, with a
`ground_truth/` subpackage for the 4 dataset/registry modules. Rewrite the
imports and patch-target literals. Add structural boundary tests. Update the
6 documentation/comment references (including `AGENT_RULES.md` §5 — a path
change is exactly the kind of explicit architecture decision that file
requires).

## Deliberately NOT done

Splitting `retrieval_evaluator.py` internally (metrics / benchmarks /
reporting sublayers) — deferred with no current consumer forcing it; no new
evaluation contract types; no change to ground truth, thresholds, expected
IDs, or benchmark definitions.

## Next Step

Migration Step 8 (observability/cache reorganization) — the plan marks it
"only if evaluation shows it's warranted; not scheduled by default". Before
starting it, decide from evidence whether it is justified at all; the more
valuable candidates are the open debts below.

## Success Criteria (met)

Ground-truth file bytes and payload hashes unchanged; 24 semantic cases and
registry key `(43, 106)` unchanged; metric outputs unchanged; full suite
≥696 passed / 5 skipped; `runtime → evaluation` proven absent by test;
`output/chroma_db/` and `output/competitions/` untouched.

## Out of Scope

Migration Step 8; the 11 known open debts; retrieval/embedding/Arabic work;
rebuilding indexes; new dependencies; committing or pushing.
