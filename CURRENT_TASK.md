# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Migration Step 5 (Generation + Verification Split) — COMPLETE, uncommitted

`07_prompting.py` reduced 940 → 219 lines: it is now the coordinator
(`answer_question()`) plus compatibility re-exports. Implementations moved
verbatim to:

- `src/generation/` — `prompt.py` (policy, canonical rendering, prompt +
  role-separated message construction), `provider.py` (model invocation),
  `policy.py` (pre-generation refusal gate), `citations.py` (provenance →
  sources)
- `src/verification/` — `validation.py` (numeric-claim checks),
  `comparison.py` (comparison/structured checks)

Option C was chosen on evidence: 7 test modules monkeypatch generation
dependencies *on the `07_prompting` module* and `chat.py` reaches 11
attributes through it, so the module had to remain the seam.

Verified: 73 prompting-path tests, 32 new contract tests, full suite
**642 passed / 5 skipped / 0 failed**. All 11 tracked `output/chroma_db/`
files match HEAD; `output/competitions/` untouched.

## Two deliberate, reported changes

1. **Trust boundary (security)**: both providers previously sent one
   `{"role": "user"}` message containing policy + evidence + question, so
   retrieved text sat at the same privilege level as the rules. They now send
   `[{"role":"system"}, {"role":"user"}]`. `build_prompt()` still returns the
   byte-identical legacy string (pinned equal to the concatenated messages).
2. **Canonical renderer (closes Step 4 debt #2)**:
   `format_context_for_prompt()` delegates to `context/rendering.py::build_context()`.
   Measured effect: on structured/hybrid routes the source header gains
   `Player:`/`Team:` labels and `Score:` (player-comparison prompt
   2773 → 2862 chars). Semantic-only routes unchanged. No answer-quality
   claim — only that rendering is single-sourced.

## Follow-up security fix — DONE (same working tree, uncommitted)

The pre-existing ReDoS in `extract_numeric_claims()` is closed. Three claim
patterns began with an unbounded lazy entity capture `(\w[\w\s]*?)`; with
`re.finditer()` retrying at every position this was O(N²) (~4.0x per doubling;
40 KB = 159 s). Fixed by bounding those three captures to
`_MAX_ENTITY_SPAN = 200` — evidence-chosen, not copied from the 60 used
elsewhere (that bound covers a single token; this span can hold a whole
sentence prefix, measured worst realistic case 109 chars). The two already-
linear patterns were left untouched.

Now linear (2.0x per doubling): 40 KB **159 s → 0.39 s**. Validation behavior
proven identical to HEAD across supported / contradicted / decimal /
multi-claim / long-prefix / accented / Arabic cases, including answer
correction. Rejected: narrowing the character class (still quadratic on
letters/words/newlines) and input truncation (complexity unchanged below the
cap). 39 tests in `tests/test_verification_security.py`. Suite: 642 → **681
passed / 5 skipped**. Detail in `PROJECT_MEMORY.md` → Security Findings.

Also hardened `test_comparison_scaling_is_not_quadratic` earlier in Step 5
(false "quadratic" failure under full-suite load).

## Open debt carried forward

1. `hybrid_search()` still calls `select_relevant_chunks()` (retrieval →
   context inversion) — needs its own phase with a benchmark re-run.
2. `intent.py → retrieval.search._detect_team_style_query` (understanding →
   retrieval), open since Step 3.
3. `build_context()` returns `""` when the first chunk alone exceeds
   `max_length` (pre-existing).
4. `07_prompting.py`, `router.py`, `search.py` remain transitional
   compatibility boundaries.

## Next Step

Migration Step 6: knowledge pipeline organization (group `extraction/`,
`rendering/`, `03_chunking.py`, `04_vector_representation.py`,
`05_create_chroma_store.py` under `knowledge/`). Not started.

## Out of Scope (unchanged discipline)

Retrieval tuning, BM25/Dense/RRF, embeddings, entity normalization, Arabic
features, Ground Truth, provider defaults, prompt rewording for style,
dependencies, `output/competitions/`, committing or pushing.
