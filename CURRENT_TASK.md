# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Phase 5 (shared team-style classifier relocation) — COMPLETE, uncommitted

Closed the last *understanding → retrieval* reverse dependency. Before:

    src/query/intent.py  ──imports _detect_team_style_query──►  src/retrieval/search.py

After — both layers depend downward on neutral shared vocabulary:

    src/query/intent.py       ──►  src/team_style.py  ◄──  src/retrieval/safeguards.py
    src/query/router.py       ──►                          src/retrieval/search.py (re-export)

`src/team_style.py` imports **only `re`**. It owns the 7 relocated symbols:
`_detect_team_style_query`, `_detect_team_style_entities`, `_STYLE_KEYWORDS`,
`_STYLE_KEYWORDS_AR`, `_normalize_arabic_for_matching`,
`_extract_latin_entity_spans`, `_LATIN_ENTITY_SPAN`.

## Why the normalizer moved too

`_normalize_arabic_for_matching()` is shared by a *moving* function
(`_detect_team_style_entities`) and a *staying* one
(`_detect_comparison_entities`, which remains in `safeguards.py`). Leaving it
behind would have created `team_style → retrieval` — the very edge being
removed. Duplicating it would let two copies drift. It moved, and
`safeguards.py` imports it back. This is the one naming compromise in the
phase: a general Arabic helper lives in a module named for team style.
Disclosed in the module docstring and in `PROJECT_MEMORY.md`.

## Files changed

| File | Change |
| --- | --- |
| `src/team_style.py` | **new** — 7 symbols, moved AST-verbatim |
| `src/retrieval/safeguards.py` | −181 lines; imports the 2 symbols it still *uses* |
| `src/retrieval/search.py` | compat re-exports repointed at the new owner |
| `src/query/intent.py` | imports the classifier from `src.team_style` |
| `src/query/router.py` | classifier from `src.team_style`; keeps `hybrid_search` from retrieval |
| `tests/test_team_style_boundary.py` | **new** — 19 guard tests |

## Verification

- **Verbatim**: all 7 symbols AST-compared source-segment-identical to their
  pre-move form; `team_style.py` contains no other top-level definitions.
- **Characterization**: 96/96 cases identical to the pre-change baseline across
  `classify` / `comparison` / `detect_entities` / `detect_query` /
  `latin_spans` / `normalize` (EN + MSA + EGY).
- **Real-data parity**: `hybrid_search(k=5)` and `_ensure_team_style_doc(k=3,5)`
  on the production index — identical pre vs post when run through the *same*
  script against stashed pre-change code. (An initial mismatch was a defect in
  the replay recipe, not in the code; confirmed by that apples-to-apples run
  plus a determinism check.)
- **Suite**: **805 passed / 5 skipped** (786 baseline + the 19 new tests).
- `git diff --check` clean. `output/chroma_db/chroma.sqlite3` bookkeeping touch
  restored (31,444,992 bytes). `output/competitions/` untouched.

## Retained compatibility edges (deliberate)

- `src/retrieval/search.py` still re-exports all 7 symbols — `tests/` and
  `router.py` reach them through it. The re-exports now point at
  `src.team_style` (the real owner) rather than laundering through
  `safeguards.py` a second time; identity is pinned by test.
- `src/query/router.py → src/retrieval/search.py::hybrid_search` remains. This
  is execution calling the RETRIEVE stage — forward in the runtime flow, not a
  reverse edge.

## Not committed

No `git add`, `git commit`, or `git push` was run.
