# Architecture Overview — Target Contract

**Status**: migration in progress — steps 1-7 of §11 are done (retrieval
split, understanding/planning split, context engineering, generation/
verification split, knowledge pipeline, evaluation organization); step 8
(observability/cache reorganization, only if justified) remains.
Code HAS moved; §4's "today:" annotations and §11 track where each stage
actually lives. This document describes where the codebase is going and why;
`PROJECT_MEMORY.md` describes where it actually is right now, including the
open compatibility/architecture debts. When code moves, update the "today:"
annotations here — do not let this drift the way the root `ARCHITECTURE.md`
did (see `PROJECT_MEMORY.md` → Known Bugs).

§1 below is a point-in-time snapshot of the pre-migration layout, kept for
the rationale in §2; it is deliberately not rewritten as code moves.

---

## 1. Current Architecture (as inspected, not as previously documented)

```
Football-Analytics-RAG-git/
├── 01_documents.py, 02_preprocessing.py, 03_chunking.py,
│   04_vector_representation.py, 05_create_chroma_store.py   -- offline pipeline stages
├── generate_documents.py, rebuild.py                        -- pipeline entry points
├── chat.py, streamlit_app.py                                 -- interfaces (CLI, web)
├── 07_prompting.py                                            -- generation + validation + citations (940 lines, multi-responsibility)
├── src/
│   ├── artifacts.py, embedding_config.py, cache.py, dataset_catalog.py,
│   │   conversation_memory.py, stage_taxonomy.py             -- flat, mixed-purpose
│   ├── extraction/        -- raw StatsBomb JSON -> match_facts.json
│   ├── rendering/          -- match_facts.json -> documents.json (prose)
│   ├── query/              -- router.py (933), resolver.py (886), vocab.py (697), query_schema.py
│   └── retrieval/          -- search.py (1244, BM25+Dense+RRF+safeguards+context), chunk_selector.py, answerability.py
└── tests/                  -- 39 files, flat, no package structure mirrors src/
```

No `domain/`, `knowledge/`, `rag/`, `infrastructure/`, `evaluation/`, or
`interfaces/` packages exist. The codebase already shows the *shape* of the
target (extraction/rendering/query/retrieval are already separated
sub-packages) but not the layering: retrieval mechanics, safeguards, and
context-building all live in one 1244-line file; generation, validation,
and citation-rendering all live in one 940-line top-level script;
orchestration (`chat.py::process_query()`) directly calls low-level
functions from both instead of going through a stable pipeline interface.

## 2. Competing Options

Evaluated against: maintainability, football-domain growth, multi-competition
support, multilingual support, local/online model support, testability,
migration risk, development complexity. Evidence drawn from the repository
itself, not from general architecture preference.

| Option | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| **Modular Monolith** (layered packages, one process) | Repo is already a single local process (CLI + Streamlit, no network boundary between components today); the project has *already* executed 3+ successful incremental modular extractions with full regression protection (competition portability, embedding config, the `06_retrieve_context.py` → `search.py`/`router.py` split) — direct proof this migration style works in this codebase | Requires discipline to keep layer boundaries honest (no shortcut imports) | **Selected** |
| **Feature/package-oriented** (vertical slices per feature, e.g. `structured_queries/`, `semantic_search/`, `comparison/`) | Would colocate everything about one feature | RAG here is a *shared pipeline* (retrieve→rerank→select→generate) that structured, semantic, and hybrid queries all flow through — the natural seams are pipeline stages, not features; vertical slicing would duplicate pipeline-stage code across features or force a shared-kernel anyway | Rejected |
| **Microservices now** | Would allow independent scaling/deployment | No current multi-tenant, high-QPS, or independent-scaling requirement anywhere in the codebase or its tests; would add network/serialization boundaries to a system that is currently one local process end to end, for no evidenced benefit | Rejected — "fashionable, not evidenced" |
| **Status quo** (numbered scripts + flat `src/`) | Zero migration risk, works today | Already causing real, observed pain: a 940-line multi-responsibility `07_prompting.py`, a 1244-line `search.py` mixing retrieval mechanics with safeguards and context-building, and a root `ARCHITECTURE.md` that drifted out of sync with reality within a few phases — direct evidence that ad-hoc structure doesn't stay documented or navigable as the system grows | Rejected |

## 3. Selected Architecture

**Modular Monolith**, one deployable Python process, packages organized by
**pipeline layer** (not by feature), each with an explicit, narrow public
interface and one-directional dependency rules (§5).

```
src/
├── domain/          -- football concepts: Competition, Season, Stage, Match, Team, Player, Event
├── knowledge/        -- ingestion, rendering, chunking, indexing (offline)
├── rag/               -- understanding, planning, retrieval, reranking, context, answerability, generation, verification, orchestration (online)
├── infrastructure/    -- cache, config, artifact paths, external clients (embedding models, LLM providers)
├── evaluation/         -- ground truth, benchmark runners, diagnostics
└── interfaces/          -- CLI, web UI -- thin, no business logic
```

## 4. Module Boundaries

### `src/domain/football/`

Football concepts as typed data, no I/O, no framework dependencies:
`Competition`, `Season`, `Stage`, `Match`, `Team`, `Player`, `Event`,
`StructuredStatistic`. Today these concepts exist only implicitly (as dict
shapes and ID conventions scattered through `src/extraction/match_facts.py`,
`src/query/query_schema.py`, `src/stage_taxonomy.py`). Target: one place
that defines what these things *are*, independent of how they're extracted,
rendered, or queried. Multi-competition support depends on this existing —
it's what lets `knowledge/` and `rag/` stay competition-agnostic.

### `src/knowledge/`

Offline pipeline. Partially realized as of §11 step 6:

```
preprocessing.py         -- today: src/knowledge/preprocessing.py   (moved from 02_preprocessing.py)
chunking.py               -- today: src/knowledge/chunking.py        (moved from 03_chunking.py)
indexing/bm25.py           -- today: src/knowledge/indexing/bm25.py         (moved from 04_vector_representation.py)
indexing/embeddings.py      -- today: src/knowledge/indexing/embeddings.py   (moved from 04_vector_representation.py)
indexing/vector_store.py     -- today: src/knowledge/indexing/vector_store.py (moved from 05_create_chroma_store.py)
ingestion/                    -- today: src/extraction/  -- already cohesive; rename deferred (churn, no responsibility change)
rendering/                     -- today: src/rendering/   -- already cohesive; rename deferred
```

The root numbered scripts remain as thin CLI orchestrators owning argument
parsing, competition/season path resolution and artifact I/O, and re-export
the moved symbols — `rebuild.py` and four test modules depend on that surface.
Structured statistics are indexed as a first-class store here, not bolted on
— `src/query/resolver.py`'s `FactStore` is the existing proof this already
works, it just isn't organized under a shared `knowledge/` umbrella yet.

`knowledge/indexing/` **writes** the Dense/BM25/structured-fact stores
through the storage contracts defined in `infrastructure/` (§ below) — it
does not expose its own ingestion/chunking/index-building internals to
anything outside `knowledge/`. Nothing outside `knowledge/` may import
`knowledge/ingestion/`, `knowledge/chunking/`, or `knowledge/indexing/`
directly; the produced indexes/stores are the only thing that crosses the
boundary, and they cross it via `infrastructure/`, not via a direct
`rag/ -> knowledge/` import.

### `src/rag/`

Online pipeline, one sub-package per stage:

```
understanding/  -- today: src/query/intent.py (classification, comparison understanding) + src/query/parsing.py (StructuredQuery, filters, stage vocabulary) -- split done, see §11 step 3
planning/        -- today: src/query/planning.py (Route + route_query() strategy selection); a richer plan model (evidence/coverage requirements) is still deferred, see §11 step 4
retrieval/        -- today: src/retrieval/{bm25,dense,fusion,safeguards}.py + search.py::hybrid_search() (split done, see §11 step 2; hybrid_search() and index loading stay in search.py -- a transitional compatibility boundary, not yet renamed rag/retrieval/ -- see PROJECT_MEMORY.md's Architecture Decisions for why)
reranking/         -- today: src/retrieval/fusion.py::rerank() (currently a no-op)
context/            -- today: src/context/ (selection.py + rendering.py + evidence.py) -- split done, see §11 step 4
answerability/       -- today: src/context/answerability.py
generation/           -- today: src/generation/ (prompt.py, provider.py, policy.py, citations.py) -- split done, see §11 step 5
verification/          -- today: src/verification/ (validation.py, comparison.py) -- split done, see §11 step 5
orchestration/          -- today: chat.py::process_query() + router.py::execute_route() (execution + the query package's transitional compatibility boundary), currently mixed together
```

Runtime flow this layering targets:

```
USER QUERY
  -> UNDERSTAND        (intent, entities, language)
  -> PLAN               (structured vs semantic vs hybrid vs comparison)
  -> RETRIEVE            (BM25 + Dense + structured facts)
  -> RERANK               (currently a no-op; extension point)
  -> SELECT EVIDENCE        (dedup, coverage, entity grounding -- chunk_selector today)
  -> ANSWERABILITY            (enough evidence? -- answerability.py today)
  -> enough evidence?
       NO  -> REFINE -> RETRIEVE
       YES -> GENERATE -> VERIFY -> FINAL ANSWER
```

Context engineering (inside SELECT EVIDENCE, expanded):

```
Candidates -> Rerank -> Deduplicate -> Coverage -> Context Budget -> Compress -> Order -> Evidence Pack
```

Principle: **smallest high-signal context, not maximum context** — this is
already `chunk_selector.py`'s actual design intent (marginal-coverage
selection over "return everything"), just not yet named or organized as a
distinct pipeline stage.

### `src/infrastructure/`

Cross-cutting technical concerns with no football-domain knowledge:
`src/cache.py`, `src/embedding_config.py`, `src/artifacts.py`, LLM provider
clients (today: the HTTP/API parts of `07_prompting.py`).

`infrastructure/` also defines the **stable storage/index-access
contracts** — vector-store, lexical-index, and fact-store interfaces — that
`knowledge/indexing/` writes through and `rag/retrieval/` reads through.
This is the sole boundary between the offline and online pipelines: it
holds *technical* access contracts only (open a collection, load an index,
fetch a record by key), never football business logic (no team-style
detection, no metric resolution, no chunk-coverage selection — those stay
in `rag/` and `knowledge/` respectively, using domain types where typing
is needed). Infrastructure modules may depend on `domain/` for shared
types in these contracts, but must never import from `rag/` or
`knowledge/`'s ingestion/chunking/indexing internals, and must never
contain business logic itself.

### `src/evaluation/`

Ground truth, benchmark runners, diagnostics. **Realized in §11 step 7** —
moved out of `tests/`, where evaluation library code was indistinguishable
from the test suite consuming it:

```
ground_truth/semantic.py       -- 24-case WC2022 semantic ground truth (protected data)
ground_truth/answerability.py  -- answerability expectations
ground_truth/multilingual.py    -- EN/MSA/EGY case construction
ground_truth/registry.py         -- dataset identity -> ground-truth bundle
retrieval_evaluator.py            -- metrics, case evaluation, aggregation,
                                     Chroma artifact safety, baseline CLI
diagnostics.py                     -- multilingual retrieval diagnostics
benchmark.py, run_*.py              -- benchmark/diagnostic runners
```

Dependency rule, enforced by `tests/test_evaluation_boundary.py`:
`evaluation -> runtime` is allowed, `runtime -> evaluation` never is.
Evaluation is a cross-cutting observer and must never join the online query
path. `retrieval_evaluator.py` remains internally multi-responsibility; that
split is deferred (see `PROJECT_MEMORY.md`).

### `src/interfaces/`

`chat.py`, `streamlit_app.py`. Thin: parse input, call one orchestration
entry point in `rag/orchestration/`, render output. No retrieval logic, no
prompt construction, no validation logic — those already leak into
`chat.py::process_query()` today (structured/semantic context assembly is
inline in the interface layer, not delegated).

## 5. Dependency Rules

```
interfaces      -> rag/orchestration only
rag/*            -> domain, infrastructure
infrastructure    -> domain (types only, for storage contracts)
knowledge          -> domain, infrastructure
evaluation          -> everything (it exercises the whole system)
domain                -> (nothing)               [lowest project-specific layer]
```

`domain/football/` is the lowest project-specific layer — every other
layer may depend on it, it depends on nothing. **`rag/` must never depend
on `knowledge/`.** The online runtime does not import
`knowledge/ingestion/`, `knowledge/chunking/`, or `knowledge/indexing/`
internals — it has no reason to know *how* an index was built. `knowledge/`
**produces** indexes/stores; `rag/` **consumes** them, and the only thing
that connects the two is the stable set of storage/index-access contracts
declared in `infrastructure/` (§4). `infrastructure/` itself carries no
business logic — it is the shared technical contract, not a place for
football-specific rules to leak into.

**Orchestration coordinates, it does not implement.** `rag/orchestration/`
calls `understand()`, `plan()`, `retrieve()`, `rerank()`, `build_evidence()`,
`check_answerability()`, `generate()`, `verify()` — it must never itself
contain retrieval logic, prompt strings, or validation rules, the way
`chat.py::process_query()` currently does. This is the single most
important rule for this migration; violating it silently recreates the
current 1244-line/940-line monoliths under new directory names.

## 6. Project Memory Design

Three files replace "re-read everything every session":

- **`AGENT_RULES.md`** — stable process rules; stable and rarely changed —
  modifications require an explicit architecture/process decision, not a
  per-phase edit.
- **`PROJECT_MEMORY.md`** — durable, structured, compact facts (current
  state, decisions, milestones, baselines, bugs, security findings,
  deferred work, constraints). Updated in place as facts change.
- **`CURRENT_TASK.md`** — small, fully rewritten at the start of each phase.
  Never accumulates history.

Composition for any future session: **Stable Rules + retrieved relevant
`PROJECT_MEMORY.md` section(s) + `CURRENT_TASK.md`** — not the full
conversation history, not the full repository.

## 7. Agent / Token-Efficiency Design

Development-agent architecture is kept conceptually separate from the
production RAG system (§10 below covers the production trust boundary).
When sub-agents are used for development work:

```
Main Agent
├── Retrieval Agent   -- retrieval-quality investigation only
├── Code Agent         -- implementation only
├── Security Agent       -- adversarial/injection/ReDoS review only
└── Evaluation Agent       -- benchmark execution only
```

Each receives only the files and prior findings relevant to its role — not
the whole repository, not the whole session history. Tool-efficiency rules
(also captured in `AGENT_RULES.md` §6–7): grep/targeted-read over full-file
dumps; targeted pytest node before full suite; compact command output;
reuse `PROJECT_MEMORY.md` instead of rebuilding history by re-reading
source or re-running already-recorded experiments.

## 8. Football-Specific Adaptation

This is not generic document RAG. `domain/football/` makes `Competition`,
`Season`, `Stage`, `Match`, `Team`, `Player`, `Event`, and
`StructuredStatistic` first-class types, not implicit dict shapes. Structured
facts remain a first-class retrieval path alongside Dense and BM25 — never
demoted to a fallback. Multi-competition/season support must come from
`domain/` + `ArtifactPaths` namespacing, never from competition-specific code
branches (the existing portability work already enforces this; the target
architecture keeps enforcing it explicitly rather than as tribal knowledge).

## 9. Optional / Extension-Point Components

Not mandatory, not built now, remain pure extension points until justified
by evaluation or a concrete use case: Graph RAG, a SQL database, *runtime*
conversation memory (distinct from the existing dataset-scoped
`src/conversation_memory.py`, which stays), an autonomous multi-agent
runtime, a microservices split, tenant/auth infrastructure.

## 10. Trust Boundary

Retrieved documents, external/web content, and tool output are **untrusted
data**, architecturally — not instructions, regardless of where they were
retrieved from or how authoritative they look. `rag/generation/` and
`rag/verification/` must treat retrieved chunk text as content to reason
about, never as directives to follow. This is a structural requirement, not just a prompting
convention — see `AGENT_RULES.md` §9 for the same principle applied to agent
work.

**As implemented (step 5)**: `src/generation/prompt.py::build_messages()`
emits `[{"role": "system", <policy>}, {"role": "user", <evidence + question>}]`
and both provider adapters transmit that separation. Delimiters alone were
previously the *only* barrier — every call sent one `user` message containing
policy, evidence, and question together, so retrieved text sat at the same
privilege level as the rules above it. The boundary is now the message role,
which the model actually distinguishes. Hostile evidence text is **contained,
not deleted**: it stays readable as evidence so the model can reason about it,
it simply cannot be promoted to an instruction.

## 11. Migration Strategy

No big-bang refactor. Each step: **Baseline → Surgical migration → Targeted
tests → Regression → RAG evaluation → Security → Review**, exactly the
workflow already used for every phase in `PROJECT_MEMORY.md`'s Completed
Milestones. Each step must be behavior-preserving — verified the same way
this phase verified it (byte-identical source outside the intended change,
full regression, artifact-integrity hashing).

1. **Architecture contract** (done) — no code moves.
2. **Retrieval split** (done) — `src/retrieval/search.py`'s BM25/Dense/RRF/
   safeguards split into `bm25.py`/`dense.py`/`fusion.py`/`safeguards.py`,
   mechanically, inside `src/retrieval/` (not yet renamed to
   `rag/retrieval/` — no other target packages exist yet, so that rename is
   deferred to whichever step actually stands up the `rag/` namespace). A
   `service.py` extraction of `hybrid_search()` orchestration was tried and
   reverted on architecture review (it needed a permanent reverse
   call-time dependency back into `search.py` to stay test-monkeypatch
   compatible — see `PROJECT_MEMORY.md`'s Architecture Decisions for the
   comparison). `search.py` remains `src/retrieval/`'s transitional
   compatibility boundary: re-export layer + index loading +
   `hybrid_search()` orchestration + context-building; see
   `PROJECT_MEMORY.md`'s Architecture Decisions and Deferred Work for why,
   and for the removal plan.
3. **Query understanding + planning** (done) — `src/query/router.py`'s
   classification, parsing, and route selection split into `intent.py`,
   `parsing.py`, and `planning.py`; `router.py` keeps execution
   (`execute_route`/`route_and_execute`/CLI) and becomes the query package's
   transitional compatibility boundary. Dependency direction is strictly
   one-way (`router.py → planning.py → parsing.py + intent.py`) — `Route`
   lives in `planning.py` precisely so no cycle forms. A formal
   `ContextPlan` (evidence/coverage requirements) was deliberately NOT
   introduced: no current consumer. See `PROJECT_MEMORY.md`'s Architecture
   Decisions.
4. **Context engineering / evidence pack** (done) — evidence selection,
   rendering, and answerability moved out of `src/retrieval/` into a new
   top-level `src/context/` package (`selection.py`, `rendering.py`,
   `answerability.py`, `evidence.py`), with a minimal lossless `EvidencePack`
   as the Candidates→Evidence-Pack contract. Named `src/context/` rather than
   `rag/context/` for the same reason as steps 2-3: the `rag/` move is one
   uniform rename of every package later. Deduplication, a token budget, and
   compression were deliberately NOT built — measured as no-ops or without a
   consumer; see `PROJECT_MEMORY.md`. Two OPEN debts remain: `hybrid_search()`
   still calls the selector (retrieval → context inversion), and two divergent
   context renderers still exist (unify in step 5).
5. **Generation + verification split** (done) — `07_prompting.py` (940 lines)
   reduced to a 219-line coordinator; implementations moved to
   `src/generation/` and `src/verification/`. Option C (coordinator +
   extracted packages) was chosen because seven test modules monkeypatch
   generation dependencies *on that module* and `chat.py` reaches eleven
   attributes through it. Also closed step 4's divergent-renderer debt (one
   canonical renderer) and established the prompt trust boundary as a real
   `system`/`user` role separation rather than markdown delimiters. See
   `PROJECT_MEMORY.md`.
6. **Knowledge pipeline organization** (done, deliberately smaller than
   originally sketched) — the transformation logic of `02_preprocessing.py`,
   `03_chunking.py`, `04_vector_representation.py` and
   `05_create_chroma_store.py` moved into `src/knowledge/`
   (`preprocessing.py`, `chunking.py`, `indexing/{bm25,embeddings,vector_store}.py`);
   the numbered scripts remain thin CLI orchestrators because `rebuild.py`
   and four test modules depend on their CLI/import surface. `src/extraction/`
   and `src/rendering/` were NOT renamed into `knowledge/`: both are already
   cohesive packages, so renaming changes no responsibility while touching
   every pipeline script and two query modules. A duplicate query-time
   `hybrid_search()` was removed from the indexing script (a real
   knowledge → rag violation). No `domain/` or `infrastructure/` package was
   created — neither is justified by current evidence. See
   `PROJECT_MEMORY.md`.
7. **Evaluation organization** (done) — the nine evaluation library modules
   moved from `tests/` to `src/evaluation/` byte-identically, with a
   `ground_truth/` subpackage for the datasets and registry. No compatibility
   shims (nothing outside `tests/` imported them). `runtime -> evaluation` is
   now structurally enforced by test. `retrieval_evaluator.py` was NOT split
   internally — relocating and re-splitting at once would violate
   MOVE -> REWIRE -> VERIFY. See `PROJECT_MEMORY.md`.
8. **Observability/cache reorganization** — only if evaluation shows it's
   warranted; not scheduled by default.

Each step is its own phase with its own `CURRENT_TASK.md`, not a checklist
to rush through in one sitting.

## 12. Not Yet Decided

- Exact package for `evaluation/`: under `src/` (importable, but blurs
  production/test boundary) or a parallel top-level root. Decide during
  step 7, with evidence from how `tests/` actually imports from it today.
- Whether `interfaces/` needs a shared thin abstraction over "ask a
  question, get an answer" before `streamlit_app.py` and `chat.py` both
  route through it, or whether each keeps its own thin orchestration call.
