# Project Memory — Football Analytics RAG

Structured, durable memory. Compact and factual — not a transcript. **Read
this file by scanning the section headings below first, then retrieving
only the section(s) relevant to the current task** — do not load the whole
file into every prompt by default. Keeping every entry compact (facts and
pointers, not narrative) is what makes that partial-read pattern viable; if
an entry starts reading like a transcript, trim it. Update in place when a
fact changes; move stale entries to history rather than deleting silently
where traceability matters.

Repo: `D:\Football-Analytics-RAG-git` · Branch: `competition-portability`

---

## Current State

- **Production embedding default**: `minilm` (`sentence-transformers/all-MiniLM-L6-v2`), selected via `src/embedding_config.py`. `mpnet-multilingual` is registered and available but is the *current best tested candidate*, not the default — see Unresolved Questions.
- **Production dataset default**: WC2022 (`competition_id=43, season_id=106`), flat `output/` layout, for zero-argument CLI calls. Any other competition/season is fully namespaced under `output/competitions/<id>/<id>/` via `src/artifacts.py::ArtifactPaths` — no competition-specific code forks.
- **Runtime entry points**: `chat.py` (CLI, interactive + single-question), `streamlit_app.py` (web UI, calls into `07_prompting.py`). Both are thin interface layers over the same underlying pipeline.
- **Offline pipeline** (numbered scripts, run via `rebuild.py`): `01_documents.py`(loader shim) → `src/extraction/` → `src/rendering/` → `02_preprocessing.py` → `03_chunking.py` → `04_vector_representation.py` (BM25 + legacy embeddings.npy) → `05_create_chroma_store.py` (Chroma index, the one actually queried at runtime).
- **Online query pipeline**: `chat.py`/`streamlit_app.py` → `src/query/router.py` (classify + route) → `src/query/resolver.py` (structured) and/or `src/retrieval/search.py::hybrid_search()` (semantic: BM25 + Dense + RRF + safeguards + sibling-expansion + `src/retrieval/chunk_selector.py`) → `07_prompting.py` (prompt build + LLM call + numeric-claim validation + citations).
- **`docs/architecture/overview.md` is the single canonical architecture source.** Root `ARCHITECTURE.md` was stale (referenced files that no longer exist) and has been replaced with a short compatibility pointer to `overview.md` — it no longer carries independent content, so there is no second architecture description to keep in sync.
- **Test suite**: 39 files under `tests/`, ~555 tests passing, 5 pre-existing unrelated skips (`test_extraction.py`, missing `documents.json` fixture in this environment).
- **Dependencies** (as of the last verification pass): Python 3.11.5, torch 2.2.2+cpu (no CUDA — see Unresolved Questions), sentence-transformers 2.7.0, transformers 4.40.2, chromadb 1.5.9, huggingface_hub 0.36.2. No dependency has changed across the last ~10 phases.

---

## Architecture Decisions

- **Competition portability via `ArtifactPaths`, not per-competition branching**: `src/artifacts.py` centralizes all dataset-scoped paths; WC2022 keeps a legacy flat-layout exception (`legacy_default=True`) so zero-argument calls stay unchanged, but `ArtifactPaths` itself makes no competition-ID special cases.
- **Embedding model selection via a small registry, not scattered literals**: `src/embedding_config.py::EmbeddingModelConfig`/`resolve_embedding_config()` replaced four independently hardcoded `"all-MiniLM-L6-v2"` strings. Dense-index identity is tied to embedding-model identity through `ArtifactPaths.chroma_collection_name` (suffixed only when non-default, so existing collection names are unchanged for the default model).
- **Structural split of the old `06_retrieve_context.py`** into `src/retrieval/search.py` (BM25/Dense/RRF/safeguards/context-building) and `src/query/router.py` (intent classification + structured/semantic/hybrid execution) — mechanical extraction, no behavior change, done in two phases (A: new modules created alongside the old file; B: callers migrated, old file deleted).
- **`rag/` must not depend on `knowledge/` internals** (corrected in the architecture contract after initial drafting): `knowledge/` produces indexes/stores; `rag/` consumes them only through stable storage/index-access contracts declared in `infrastructure/`. `infrastructure/` carries technical contracts only, never football business logic, though it may reference `domain/` types. `domain/football/` remains the lowest project-specific layer (depends on nothing, everything else may depend on it). See `docs/architecture/overview.md` §4–5.
- **One canonical architecture document, not two**: root `ARCHITECTURE.md` was replaced with a compatibility pointer rather than being independently corrected, to avoid maintaining two competing descriptions going forward.
- **Arabic retrieval support is safeguard-layer, not embedding-layer**: MSA/Egyptian team-style and comparison intent detection live in `src/retrieval/search.py` as parallel phrase/regex sets to the existing English ones, deliberately not a language-detection or translation layer. Entity names stay in canonical Latin form by design; Arabic-script entity transliteration is explicitly out of scope for that layer.
- **Chunk selector is now Unicode-aware but the real corpus is English-only**: `src/retrieval/chunk_selector.py`'s tokenizer recognizes Arabic script, proven correct via synthetic-content tests, but has zero measured effect on the current benchmark because every chunk's `text` field and every entity-metadata field in this dataset is English/Latin-only — confirmed by direct inspection (0/6594 chunks contain Arabic).
- **Verification discipline**: every phase in this project's history has followed baseline → competing hypotheses → evidence → surgical patch → targeted verify → regression → RAG evaluation → security → review, with an explicit final commit-safety verdict. This is now formalized in `AGENT_RULES.md`.

---

## Completed Milestones

(newest first; each was verified with regression + evaluation before being marked done)

1. Arabic-aware chunk selector (`e9e4974`) — Unicode tokenizer fix, zero measured benchmark effect, real correctness fix for a pure-Arabic-no-entity edge case.
2. Arabic retrieval safeguards + adversarial security audit (`2b88916`) — MSA/Egyptian team-style + comparison intent detection; found and fixed a ReDoS vulnerability (2 pre-existing English patterns + 1 new Arabic pattern) and a multi-entity collision bug (team-style safeguard silently picking only the first of several named teams).
3. Configurable multilingual embeddings (`5fce790`) — `src/embedding_config.py` registry; MiniLM stays default; MPNet registered as selectable.
4. Multilingual retrieval root-cause diagnosis — isolated 8 named contributors to Arabic retrieval degradation via controlled ablation (Dense-only/BM25-only/raw-RRF/full-Hybrid × EN/MSA/EGY); confirmed English-only retrieval safeguards and ASCII-only tokenization as primary causes, independent of embedding quality.
5. Multilingual retrieval baseline — 72-query EN/MSA/EGY benchmark built on top of the existing 24-case semantic ground truth; found severe Arabic degradation (Hybrid Hit@5: EN 91.7%, MSA 54.2%, EGY 41.7%) before any of the above fixes.
6. README architecture correction — fixed a misleading pipeline diagram (split Build Flow vs Query Flow). Note: this did **not** touch the separate, more deeply stale `ARCHITECTURE.md`.
7. Structural cleanup Phase A/B — extracted `src/retrieval/search.py` + `src/query/router.py` from `06_retrieve_context.py`, then migrated all callers and deleted the compatibility wrapper.
8. Naming/responsibility cleanup, user-facing citations, team/player comparison correctness fixes — earlier hardening phases (details not repeated here; see git log for exact commits).
9. Competition portability — `src/artifacts.py`, `ArtifactPaths`, namespaced pipeline for non-WC2022 datasets.

---

## Benchmark Baselines

Dense Hit@5 / All-Required@5, MiniLM, 24-case semantic ground truth ×3 language variants (most recent measurement, post all fixes above):

| Model | EN Hit@5 | MSA Hit@5 | EGY Hit@5 | EN AllReq@5 | MSA AllReq@5 | EGY AllReq@5 |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM (prod) | 0.833 | 0.792 | 0.708 | 0.750 | 0.708 | 0.583 |
| BGE-M3 | 0.875 | 0.833 | 0.625 | 0.708 | 0.708 | 0.500 |
| MPNet-multilingual | 0.833 | 0.792 | 0.708 | 0.750 | 0.708 | 0.583 |

Hybrid (full pipeline, post Arabic-safeguards + chunk-selector fixes), Hit@5 / AllReq@5:

| Lang | Hit@5 | AllReq@5 | NDCG@5 | MRR |
|---|---:|---:|---:|---:|
| EN | 0.917 | 0.792 | 0.791 | 0.795 |
| MSA | 0.750 | 0.667 | 0.573 | 0.553 |
| EGY | 0.625 | 0.500 | 0.494 | 0.524 |

Full suite: 555 passed, 5 skipped (pre-existing, unrelated), 0 failed, as of `e9e4974`.

---

## Known Bugs

- Pre-existing English regex bug in `_detect_team_style_query()` (`src/retrieval/search.py`): `"What formation did Germany use?"` → extracts `"What"` instead of `"Germany"` (non-greedy capture stops too early). Not Arabic-related, not introduced by any recent phase. Not fixed — out of scope wherever it's been found.
- `_ensure_comparison_entities()`'s L4-document lookup uses plain substring matching, so an extracted entity spelled without diacritics (`"mbappe"`) never matches a chunk store `player_name` spelled with them (`"Kylian Mbappé Lottin"`). Language-independent (affects English equally). Requires entity normalization — deferred.
- `rerank()` in `src/retrieval/search.py` is a documented no-op pass-through (comment says "future: cross-encoder reranker").
- `resolve_from_text()` in `src/query/resolver.py` is not called by production code (`router.py` doesn't use it) — only exercised by one test. Dead code, not yet removed.
- `04_vector_representation.py` writes `output/embeddings/embeddings.npy`, which nothing in the live query path reads (`dense_search()` queries Chroma directly). Confirmed dead artifact, not yet removed from the pipeline.

## Security Findings

- **ReDoS, found and fixed** (Arabic safeguards phase): `_detect_comparison_entities()` had 3 vulnerable regex shapes (2 pre-existing English patterns — `X vs Y`, `X or Y` — plus 1 new Arabic pattern) that combined `re.search()`'s per-position retry with an unbounded quantifier, producing O(N²) behavior on long non-matching input (100k-character payloads hung >60s). Fixed by bounding every entity-capturing quantifier to 60 characters; verified sub-second on the same payloads afterward. 8 permanent regression tests protect this.
- **SQL/shell/path/Chroma-filter injection**: repeatedly, adversarially tested across the retrieval safeguard layer and the chunk selector — confirmed inert in all cases. Query text and extracted entities only ever reach `.lower()`/substring comparisons, embedding vectors, or fixed-shape `{"level": ...}` Chroma filters; never raw SQL, shell commands, or filesystem paths. No SQL is constructed anywhere in the retrieval path (Chroma's own SQLite usage is internal to its parameterized API).

## Deferred Work

- Entity-script normalization (Arabic-transliterated names, e.g. `ميسي` ↔ `Messi`) — confirmed independent blocker from the root-cause diagnosis, not yet started.
- Stage/match-summary Arabic support (`_STAGE_KEYWORDS`/`_detect_match_query()`) — deliberately deferred in the Arabic safeguards phase; no real benchmark case needed it, and full support would require Arabic sentence-structure parsing.
- MPNet production-index validation (build + validate a real MPNet Chroma index end-to-end before considering it as a default-model candidate).
- CUDA-enabled PyTorch environment (separate from the standard CPU environment) — physical GPU (RTX 2000 Ada, 16GB) present but unused; current torch wheel is CPU-only. Not installed; would only matter for faster large-model experimentation, not correctness.
- Dead-artifact cleanup: `embeddings.npy`, `resolve_from_text()`.
- Modular Monolith migration itself (see `docs/architecture/overview.md` for the staged plan) — this phase is the contract only, no code moved yet.

## Unresolved Questions

- Should MPNet-multilingual become the production default? Current evidence: it wins on Egyptian-Arabic metrics (the project's stated priority population) but loses to BGE-M3 on EN/MSA; no production index has been built or validated for it yet.
- What triggers building a namespaced (non-WC2022) competition/season in practice — is there a concrete second dataset planned, or is portability being built ahead of demand?
- Should the ASCII-only-corpus limitation found in the chunk-selector work (chunk text is English-only even for Arabic queries) be addressed by translating/localizing rendered documents, or is English-only content an accepted, permanent property of this data source?

## Important Constraints

- Never change the production embedding default away from `minilm` without an explicit, separate task asking for it.
- Never touch `output/competitions/` unless the current task explicitly requires reading or writing it.
- Never rebuild or modify `output/chroma_db/` in place; use `temporary_chroma_copy()` for any experimental Chroma access.
- Never install or upgrade dependencies without first explaining why the standard library / already-installed packages are insufficient.
- Never commit or push without explicit instruction in that turn.
- WC2022 (`43`/`106`) zero-argument behavior must never silently change.
