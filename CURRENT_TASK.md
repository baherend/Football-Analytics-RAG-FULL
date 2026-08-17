# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Migration Step 4 (Context Engineering / Evidence Pack) — COMPLETE, uncommitted

New top-level `src/context/` package owns choosing and presenting evidence:
`selection.py` (moved from `retrieval/chunk_selector.py`), `answerability.py`
(moved from `retrieval/`), `rendering.py` (`build_context` moved from
`search.py`), and `evidence.py` (new `EvidenceItem`/`EvidencePack`). Both
moves were `git mv` with byte-identical content. `src/retrieval/__init__.py`'s
exports of the moved symbols were dropped rather than shimmed — nothing
outside the package imported them.

`EvidencePack` is minimal and lossless: `to_chunks()` returns the *same*
objects `from_chunks()` received (identity-checked), so citations, prompt
text, and answerability get byte-identical input. Real consumers today:
`execute_route()` (SELECT EVIDENCE → ANSWERABILITY handoff, exposed as
`RoutedResult.evidence`) and `rendering.render_pack()`.

Verified: 188 targeted tests, full suite **610 passed / 5 skipped / 0 failed**
(587 + 23 new contract tests). Candidate IDs, selected evidence IDs, context
text, entity coverage, and answerability status **byte-identical** across the
8-case baseline. All 11 tracked `output/chroma_db/` files match HEAD;
`output/competitions/` untouched.

## Deliberately not built (measured, not assumed)

- **Deduplicate** — candidate-pool duplicate count was **0 in all 8 cases**;
  retrieval safeguards already dedupe by `chunk_id`. Would be a no-op box.
- **Token budget** — one already exists implicitly (`max_chunks` +
  `max_length`). No token-aware need proven.
- **Compression** — `DEFERRED, NOT YET JUSTIFIED`: no demonstrated overflow,
  no consumer, no way to evaluate information loss.

## Open debt carried forward

1. **Retrieval → context inversion**: `hybrid_search()` step 9 still calls
   `select_relevant_chunks()`. Moving it to the orchestrator changes the
   most-benchmarked function's output → needs its own phase with a full
   multilingual benchmark re-run.
2. **Two divergent context renderers**: `context/rendering.py::build_context`
   vs `07_prompting.py::format_context_for_prompt` render the same evidence
   differently, and which one reaches the LLM depends on route + entry point.
   Unify during Step 5 (changing it alters generation semantics).
3. `intent.py → src.retrieval.search._detect_team_style_query`
   (understanding → retrieval), still open from Step 3.
4. `build_context()` returns `""` rather than the "No relevant documents
   found." sentinel when the first chunk alone exceeds `max_length`
   (pre-existing, moved verbatim).

## Next Step

Migration Step 5: Generation + verification split — separate
`07_prompting.py`'s prompt building, LLM calls, and answer validation into
generation/ and verification/. That phase is the right place to resolve open
debt #2, since it owns the competing renderer.

## Out of Scope (unchanged discipline)

Retrieval tuning, BM25/Dense/RRF, embeddings, entity normalization, Arabic
features, Ground Truth/benchmark changes, generation semantics, LLM
compression, dependencies, `output/competitions/`, committing or pushing.
