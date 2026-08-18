"""
src/evaluation/ -- the evaluation layer: ground truth, metrics, benchmark
runners and diagnostics.

Migration Step 7. These modules were previously in `tests/`, where evaluation
*library* code was indistinguishable from the test suite that consumes it —
and where three `src/retrieval/` modules and AGENT_RULES.md §5 pointed at a
load-bearing contract (`temporary_chroma_copy`, `reset_retrieval_caches`)
living under a test folder. All nine modules moved byte-identically.

    ground_truth/              -- benchmark datasets + registry (protected data)
    retrieval_evaluator.py     -- metrics, case evaluation, aggregation,
                                  Chroma artifact safety, baseline CLI
    diagnostics.py             -- multilingual retrieval diagnostics
    benchmark.py               -- evaluation benchmark
    run_multilingual_diagnostics.py, run_phase4_phase5.py -- runners

**Dependency direction (enforced by tests/test_evaluation_boundary.py):**

    evaluation -> runtime      allowed
    runtime    -> evaluation   NEVER

Evaluation is a cross-cutting observer. It may import retrieval, query,
context, generation, verification and knowledge to exercise them; nothing in
those layers, and no root runtime script, may import this package. Evaluation
must never become part of the online query path.

Deliberately NOT split further this phase: `retrieval_evaluator.py` mixes
metrics, orchestration, artifact safety and CLI (12 sections). Relocating and
re-splitting at once would violate MOVE -> REWIRE -> VERIFY; the internal
split is deferred with no current consumer forcing it.
"""

__all__: list[str] = []
