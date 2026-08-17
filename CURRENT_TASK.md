# Current Task

> Rewritten at the start of each phase — do not append history here. Anything
> worth keeping past this phase belongs in `PROJECT_MEMORY.md` instead.

## Status: Migration Step 3 (Query Understanding + Planning Split) — COMPLETE, uncommitted

`src/query/router.py` (933 lines) split into `intent.py` (classification +
comparison understanding), `parsing.py` (StructuredQuery + filters + stage
vocabulary), and `planning.py` (`Route` + `route_query()`). `router.py` keeps
execution (`RoutedResult`, `execute_route`, `route_and_execute`, CLI) and is
the query package's transitional compatibility boundary — not final
architecture.

Dependency direction is strictly one-way (`router.py → planning.py →
parsing.py + intent.py`); no cycle, no reverse dependency, no lazy-import-back
trick — `Route` lives in `planning.py` specifically to keep it that way.

Verified: 163 targeted tests (baseline match), full regression 555 passed /
5 skipped (baseline match), 18-case route/classify/parse snapshot identical,
7-case end-to-end snapshot identical (routes, structured results, EN/MSA/EGY
document IDs, context). All 11 tracked `output/chroma_db/` files match HEAD;
`output/competitions/` untouched. Not committed — awaiting review.

## Follow-up security fix — DONE (same working tree, uncommitted)

The pre-existing ReDoS surfaced by the split is closed: `intent.py`'s
`vs`/`versus` comparison patterns bounded to `(\w{1,60})`. Complexity went
from 4.0× per doubling (quadratic) to exactly 2.0× (linear); 20 KB accented
input 16.5 s → 0.034 s. Bound justified on corpus evidence (longest single
`\w` token in 713 entity names is 14 chars), not copied blindly. 32 permanent
tests in `tests/test_query_intent_security.py`. Route/parse snapshot still
identical to the pre-migration baseline. Suite now 587 passed / 5 skipped.
Detail in `PROJECT_MEMORY.md` → Security Findings.

## Next Step

Migration Step 4: Context Engineering / Evidence Pack (`chunk_selector.py` +
`answerability.py` → a named `rag/context/` home, plus the
Candidates→Evidence-Pack contract). Step 4 is also where the deferred richer
plan model (evidence_requirements / coverage_requirements) should be
reconsidered, since that step creates its first real consumer.

Also still open: the `intent.py → src.retrieval.search._detect_team_style_query`
understanding→retrieval dependency (deferred architecture debt, recorded in
`PROJECT_MEMORY.md`).

## Out of Scope (unchanged discipline)

Routing algorithm redesign, entity normalization, Arabic feature expansion,
retrieval tuning, embedding changes, prompt/generation changes, dead-code
cleanup, dependency changes, competition-specific branches,
`output/competitions/`, committing or pushing without being asked.
