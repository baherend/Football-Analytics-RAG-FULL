# FIFA World Cup 2022 RAG System — Architecture Documentation

> Generated from the implemented source code. Every diagram and description
> reflects the actual codebase — no invented components.

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Source"
        RAW["StatsBomb Open Data<br/>(JSON files)"]
    end

    subgraph "Offline Pipeline"
        direction TB
        EXT["Phase 1: Extraction<br/>src/extraction/match_facts.py"]
        RND["Phase 2: Rendering<br/>src/rendering/render.py"]
        PRE["Phase 3a: Preprocessing<br/>02_preprocessing.py"]
        CHK["Phase 3b: Chunking<br/>03_chunking.py"]
        REP["Phase 3c: Sparse Indices<br/>04_representation.py"]
        EMB["Phase 3d: Embeddings<br/>05_embeddings.py"]
        CHR["Phase 3e: Vector Store<br/>06_create_chroma_store.py"]

        EXT -->|"match_facts.json"| RND
        RND -->|"documents.json"| PRE
        PRE -->|"processed_documents.json"| CHK
        CHK -->|"chunks.json"| REP
        CHK -->|"chunks.json"| EMB
        EMB -->|"embeddings.npy"| CHR
    end

    subgraph "Storage Artifacts"
        MF["output/match_facts.json"]
        DOC["output/documents.json"]
        CHUNKS["output/chunks.json"]
        BM25_IDX["output/indices/bm25.pkl"]
        TFIDF_IDX["output/indices/tfidf_vectorizer.pkl"]
        EMB_NPY["output/embeddings/embeddings.npy"]
        CHROMA_DB["output/chroma_db/"]
    end

    subgraph "Online Query Pipeline"
        direction TB
        CHAT["Chat Interface<br/>chat.py"]
        ROUTER["Query Router<br/>08_router.py"]
        RESOLVER["Structured Resolver<br/>src/query/resolver.py"]
        RETRIEVE["Hybrid Retrieval<br/>07_retrieve_context.py"]
        PROMPT["Prompt Builder<br/>src/generation/prompt_builder.py"]
        LLM["LLM Generator<br/>src/generation/llm.py"]
        CACHE["Cache Manager<br/>src/cache.py"]

        CHAT -->|"user query"| ROUTER
        ROUTER -->|"structured query"| RESOLVER
        ROUTER -->|"semantic query"| RETRIEVE
        ROUTER -->|"hybrid query"| RESOLVER
        ROUTER -->|"hybrid query"| RETRIEVE
        RESOLVER -->|"structured result"| PROMPT
        RETRIEVE -->|"context chunks"| PROMPT
        PROMPT -->|"prompt"| LLM
        CACHE -.->|"caches models<br/>& results"| RESOLVER
        CACHE -.->|"caches embedding<br/>model"| RETRIEVE
    end

    subgraph "External Services"
        OPENROUTER["OpenRouter API<br/>(Claude, GPT, Mistral)"]
    end

    RAW --> EXT
    MF -.->|"read by"| RESOLVER
    CHUNKS -.->|"read by"| RETRIEVE
    BM25_IDX -.->|"read by"| RETRIEVE
    CHROMA_DB -.->|"read by"| RETRIEVE
    EMB_NPY -.->|"read by"| CHR
    LLM -->|"HTTP POST"| OPENROUTER

    style EXT fill:#4CAF50,color:#fff
    style RND fill:#4CAF50,color:#fff
    style PRE fill:#2196F3,color:#fff
    style CHK fill:#2196F3,color:#fff
    style REP fill:#2196F3,color:#fff
    style EMB fill:#2196F3,color:#fff
    style CHR fill:#2196F3,color:#fff
    style ROUTER fill:#FF9800,color:#fff
    style RESOLVER fill:#FF9800,color:#fff
    style RETRIEVE fill:#FF9800,color:#fff
    style PROMPT fill:#9C27B0,color:#fff
    style LLM fill:#9C27B0,color:#fff
    style CHAT fill:#E91E63,color:#fff
    style CACHE fill:#607D8B,color:#fff
```

---

## 2. Offline Pipeline (Indexing)

```mermaid
graph LR
    subgraph "Phase 1 — Extraction"
        A1["Raw StatsBomb<br/>events/*.json"] --> A2["match_facts.py<br/>extract_all()"]
        A2 --> A3["match_facts.json<br/>1995 player records<br/>64 match records<br/>128 team records"]
    end

    subgraph "Phase 2 — Rendering"
        B1["match_facts.json"] --> B2["render.py<br/>render_all()"]
        B2 --> B3["documents.json<br/>L1: 64 match summaries<br/>L2: 64 key events<br/>L3: ~1995 player/match<br/>L4: ~500 player/tournament<br/>Team: 32 team docs"]
    end

    subgraph "Phase 3 — Text Processing"
        C1["documents.json"] --> C2["02_preprocessing.py"]
        C2 --> C3["processed_documents.json"]
        C3 --> C4["03_chunking.py<br/>500 char max, 50 overlap"]
        C4 --> C5["chunks.json"]
    end

    subgraph "Phase 3 — Index Building"
        D1["chunks.json"] --> D2["04_representation.py"]
        D2 --> D3["bm25.pkl<br/>(BM25Okapi index)"]
        D2 --> D4["tfidf_vectorizer.pkl<br/>(TF-IDF vectorizer)"]
        D2 --> D5["chunk_ids.json"]

        E1["chunks.json"] --> E2["05_embeddings.py<br/>all-MiniLM-L6-v2"]
        E2 --> E3["embeddings.npy<br/>(384-dim vectors)"]
        E2 --> E4["embedding_metadata.json"]
        E3 --> E3b["06_create_chroma_store.py"]
        E3b --> E5["chroma_db/<br/>(persistent ChromaDB)"]
    end

    style A2 fill:#4CAF50,color:#fff
    style B2 fill:#4CAF50,color:#fff
    style C2 fill:#2196F3,color:#fff
    style C4 fill:#2196F3,color:#fff
    style D2 fill:#2196F3,color:#fff
    style E2 fill:#2196F3,color:#fff
    style E3b fill:#2196F3,color:#fff
```

**Pipeline orchestrator:** `rebuild.py` runs all steps sequentially. `--quick` flag skips embeddings/ChromaDB.

---

## 3. Online Query Pipeline

```mermaid
graph TB
    subgraph "User Input"
        U["User Question"]
    end

    subgraph "Chat Interface (chat.py)"
        C1["process_query()"]
        C2["ChatState<br/>mode, model, history"]
    end

    subgraph "Router (08_router.py)"
        R1["classify_query()<br/>regex patterns +<br/>keyword scoring"]
        R2["parse_structured_query()<br/>regex extraction"]
        R3["route_query()<br/>→ Route"]
        R4["execute_route()<br/>→ RoutedResult"]
    end

    subgraph "Structured Path"
        S1["resolve()"]
        S2["FactStore<br/>indexed lookups"]
        S3["MetricKind-aware<br/>period slicing"]
        S4["StructuredResult<br/>status, value, explanation"]
    end

    subgraph "Semantic Path"
        SE1["bm25_search()<br/>k=20 candidates"]
        SE2["dense_search()<br/>ChromaDB k=20"]
        SE3["reciprocal_rank_fusion()<br/>RRF k=60"]
        SE4["rerank()<br/>(pass-through)"]
        SE5["build_context()<br/>→ context string"]
    end

    subgraph "Hybrid Path"
        HY1["Structured resolution"]
        HY2["Semantic retrieval"]
        HY3["Combine results"]
    end

    subgraph "Generation"
        G1["build_prompt()<br/>system + context + question"]
        G2["generate_answer()<br/>OpenRouter API"]
        G3["Answer"]
    end

    U --> C1
    C1 --> R1
    R1 --> R2
    R2 --> R3
    R3 -->|"structured"| R4
    R3 -->|"semantic"| R4
    R3 -->|"hybrid"| R4

    R4 -->|"structured"| S1
    S1 --> S2
    S2 --> S3
    S3 --> S4

    R4 -->|"semantic"| SE1
    R4 -->|"semantic"| SE2
    SE1 --> SE3
    SE2 --> SE3
    SE3 --> SE4
    SE4 --> SE5

    R4 -->|"hybrid"| HY1
    R4 -->|"hybrid"| HY2
    HY1 --> HY3
    HY2 --> HY3

    S4 --> G1
    SE5 --> G1
    HY3 --> G1
    G1 --> G2
    G2 --> G3

    style C1 fill:#E91E63,color:#fff
    style R3 fill:#FF9800,color:#fff
    style S1 fill:#FF9800,color:#fff
    style SE3 fill:#FF9800,color:#fff
    style G2 fill:#9C27B0,color:#fff
```

---

## 4. Sequence Diagram — Complete Query Flow

```mermaid
sequenceDiagram
    actor User
    participant Chat as chat.py
    participant Router as 08_router.py
    participant Resolver as resolver.py
    participant Retrieve as 07_retrieve_context.py
    participant Prompt as prompt_builder.py
    participant LLM as llm.py
    participant API as OpenRouter API
    participant Cache as cache.py

    User->>Chat: "How many goals did Messi score?"

    Note over Chat: process_query(question)

    rect rgb(255, 243, 224)
        Note over Chat,Router: Routing Phase
        Chat->>Router: route_query(question)
        Router->>Router: classify_query() → "structured" (0.9)
        Router->>Router: parse_structured_query()
        Router-->>Chat: Route(path="structured",<br/>structured_query=StructuredQuery(<br/>intent="numeric", entity="player",<br/>metric="goals", entity_name="Messi"))
    end

    rect rgb(227, 242, 253)
        Note over Chat,Resolver: Structured Execution
        Chat->>Router: execute_route(route)
        Router->>Resolver: resolve(structured_query, data)
        Resolver->>Cache: get_cached_structured_result(key)
        Cache-->>Resolver: None (cache miss)
        Resolver->>Resolver: _resolve_player_name("Messi")
        Resolver->>Resolver: _apply_filter(), _aggregate()
        Resolver->>Cache: set_cached_structured_result(key, result)
        Resolver-->>Router: StructuredResult(status="resolved",<br/>aggregated_value=7,<br/>explanation="Messi's total goals is 7.")
        Router-->>Chat: RoutedResult(structured_result=...,<br/>semantic_chunks=None)
    end

    Note over Chat: Build context from structured result

    rect rgb(243, 229, 245)
        Note over Chat,LLM: Generation Phase
        Chat->>Prompt: build_prompt(question, context)
        Prompt-->>Chat: Full prompt string
        Chat->>LLM: generate_answer(prompt, model="haiku")
        LLM->>API: POST /chat/completions
        API-->>LLM: {"choices": [{"message": {"content": "..."}}]}
        LLM-->>Chat: "Lionel Messi scored 7 goals..."
        Chat-->>User: Answer
    end
```

---

## 5. Component Responsibilities

### 5.1 Data Layer

| Module | File | Purpose | Output |
|--------|------|---------|--------|
| **Extraction** | `src/extraction/match_facts.py` | Parses raw StatsBomb JSON into typed records. Handles minutes-played algorithm, card detection, possession calculation, per-period breakdowns. | `match_facts.json` |
| **Extraction CLI** | `src/extraction/extract.py` | Entry point: calls `extract_all()` and `persist()` | `match_facts.json` |
| **Minutes Played** | `src/extraction/minutes_played.py` | Computes per-player minutes from lineups position segments. Handles phantom segments, red cards, inverted periods, duplicate segments. | In-memory |
| **Rendering** | `src/rendering/render.py` | Pure renderer: reads `match_facts.json`, produces natural-language documents. No statistics computation. | `documents.json` |
| **Render CLI** | `generate_documents.py` | Entry point: calls `render_all()` and `persist()` | `documents.json` |

### 5.2 Semantic Pipeline (Phase 3/4)

| Module | File | Purpose | Output |
|--------|------|---------|--------|
| **Preprocessing** | `02_preprocessing.py` | Unicode normalization, whitespace cleaning, punctuation normalization, control char removal. | `processed_documents.json` |
| **Chunking** | `03_chunking.py` | Sentence-based chunking (500 char max, 50 char overlap). | `chunks.json` |
| **Sparse Indices** | `04_representation.py` | Builds TF-IDF vectorizer and BM25Okapi index from chunks. | `bm25.pkl`, `tfidf_vectorizer.pkl`, `chunk_ids.json` |
| **Embeddings** | `05_embeddings.py` | Generates 384-dim sentence embeddings using `all-MiniLM-L6-v2`. | `embeddings.npy`, `embedding_metadata.json` |
| **Vector Store** | `06_create_chroma_store.py` | Creates persistent ChromaDB collection `wc2022_documents`. | `chroma_db/` directory |

### 5.3 Query Layer

| Module | File | Purpose |
|--------|------|---------|
| **Query Schema** | `src/query/query_schema.py` | Defines `Filter`, `StructuredQuery`, `StructuredResult` dataclasses. Intent types: numeric, superlative, slice, aggregation. |
| **Vocabulary** | `src/query/vocab.py` | Defines 27+ metrics with `MetricKind` (STORED, DERIVED_RATIO, MATCH_ONLY). Maps synonyms (goals→scored, xg→expected goals). Validates queries before execution. |
| **Resolver** | `src/query/resolver.py` | Executes `StructuredQuery` against `match_facts.json`. Indexed lookups via `FactStore`. Period-aware metric reading. Case-insensitive entity resolution with accent handling. |
| **Router** | `08_router.py` | Classifies queries as structured/semantic/hybrid using regex patterns + keyword scoring. Extracts stage filters. Parses structured queries from natural language. |

### 5.4 Retrieval Layer

| Module | File | Purpose |
|--------|------|---------|
| **Hybrid Retrieval** | `07_retrieve_context.py` | Combines BM25 lexical search (k=20) + ChromaDB dense search (k=20) via Reciprocal Rank Fusion (k=60). Includes comparison entity detection, team style query detection, match-level query detection. |

### 5.5 Generation Layer

| Module | File | Purpose |
|--------|------|---------|
| **Prompt Builder** | `src/generation/prompt_builder.py` | Builds prompts with system instructions (8 rules), retrieved context, and user question. Formats chunk metadata as source citations. Supports honest failure prompts. |
| **LLM Generator** | `src/generation/llm.py` | HTTP client for OpenRouter API. Supports 5 models: Claude Haiku, Claude Sonnet, GPT-4, GPT-3.5, Mistral. Returns generated text. |

### 5.6 Infrastructure

| Module | File | Purpose |
|--------|------|---------|
| **Cache** | `src/cache.py` | In-memory caching for embedding model (avoids reload) and structured query results (keyed by data hash, auto-invalidates). |
| **Chat** | `chat.py` | Interactive terminal interface. Commands: /context, /prompt, /route, /mode, /model, /history, /clear. Mode override (hybrid/semantic/structured). Conversation history for follow-ups. |
| **Rebuild** | `rebuild.py` | Orchestrates full pipeline rebuild. `--quick` skips embeddings/ChromaDB. |

---

## 6. Router Decision Logic

The router (`08_router.py`) classifies queries using a two-stage approach:

### Stage 1: Pattern Matching (high confidence)

| Pattern Type | Examples | Classification |
|-------------|----------|----------------|
| Numeric | "How many goals did Messi score?" | structured (0.9) |
| Superlative | "Who scored the most goals?" | structured (0.9) |
| Which-team | "Which team had the highest xG?" | structured (0.9) |
| Direct metric | "Messi goals" | structured (0.9) |
| Descriptive | "How did France play?" | semantic (0.9) |
| Tell-me | "Tell me about the final" | semantic (0.9) |
| Compare | "Compare Messi and Mbappé" | semantic (0.9) |

### Stage 2: Keyword Scoring (lower confidence)

If no pattern matches, count structured vs semantic keywords:
- `structured_pct > 0.7` → structured
- `semantic_pct > 0.7` → semantic
- Otherwise → hybrid (0.6)

### Execution Paths

| Path | Structured | Semantic | Hybrid |
|------|-----------|----------|--------|
| Resolver | ✅ | ❌ | ✅ |
| Retrieval | ❌ | ✅ | ✅ |
| Context source | `StructuredResult.explanation` | chunk text | Both combined |
| LLM prompt | Structured data first, labeled "authoritative" | Retrieved chunks | Structured + chunks |

---

## 7. Storage Artifacts

### Actually present in the code

| Artifact | Created By | Read By | Format |
|----------|-----------|---------|--------|
| `output/match_facts.json` | `src/extraction/extract.py` | `resolver.py`, `render.py`, `cache.py` | JSON (3 record types) |
| `output/documents.json` | `generate_documents.py` | `02_preprocessing.py` | JSON (5 doc levels) |
| `output/processed_documents.json` | `02_preprocessing.py` | `03_chunking.py` | JSON |
| `output/chunks.json` | `03_chunking.py` | `04_representation.py`, `05_embeddings.py`, `07_retrieve_context.py` | JSON |
| `output/indices/bm25.pkl` | `04_representation.py` | `07_retrieve_context.py` | Pickle (BM25Okapi) |
| `output/indices/tfidf_vectorizer.pkl` | `04_representation.py` | — (unused at query time) | Pickle |
| `output/indices/chunk_ids.json` | `04_representation.py` | — (unused at query time) | JSON |
| `output/embeddings/embeddings.npy` | `05_embeddings.py` | `06_create_chroma_store.py` | NumPy array |
| `output/embeddings/embedding_metadata.json` | `05_embeddings.py` | — (metadata only) | JSON |
| `output/chroma_db/` | `06_create_chroma_store.py` | `07_retrieve_context.py` | ChromaDB persistent |

### Legacy artifact (not in pipeline)

| Artifact | Created By | Status |
|----------|-----------|--------|
| `output/documents.json` | `01_documents.py` | **Deprecated** — superseded by `src/rendering/render.py` |

---

## 8. Architecture Review & Improvement Suggestions

### 8.1 Inconsistencies Found

| Issue | Location | Description |
|-------|----------|-------------|
| **`tfidf_vectorizer.pkl` is unused** | `04_representation.py` | Built during indexing but never loaded at query time. BM25 is used instead. Dead artifact. |
| **`chunk_ids.json` is unused** | `04_representation.py` | Built during indexing but never loaded at query time. Dead artifact. |
| **`embeddings.npy` intermediate** | `05_embeddings.py` → `06_create_chroma_store.py` | Embeddings are saved to `.npy` then re-loaded by ChromaDB creation. The `.npy` file is not needed at query time — ChromaDB stores its own embeddings. |
| **Legacy `01_documents.py`** | Root directory | 1560-line monolith that conflates Phase 1 and Phase 2. Marked deprecated but still present. |
| **`rerank()` is pass-through** | `07_retrieve_context.py:257` | Comment says "future: cross-encoder reranker" but implementation is `return results`. |
| **`resolve_from_text()` duplicated** | `resolver.py:663-720` | Duplicates regex parsing logic from `08_router.py`. Never called by the router. Dead code. |
| **Rebuild step numbering** | `rebuild.py:53-64` | Comments say "Phase 3" for preprocessing/chunking/representation and "Phase 4" for embeddings/ChromaDB, but ARCHITECTURE.md calls them Phase 4. Inconsistent numbering. |
| **`by_period` not in JSON** | `match_facts.json` | The extraction code populates `by_period` with string keys, but the shipped artifact has no `by_period` field. Extraction has not been re-run with updated code. |

### 8.2 Missing Connections

| Gap | Description |
|-----|-------------|
| **No structured→semantic fallback** | If the structured path returns "empty", the router does not retry with semantic search. The user gets "I don't have enough data" even if semantic chunks exist. |
| **Chat mode not wired to retrieval** | `state.mode` is now respected for routing, but `semantic_k` is hardcoded to 5. Mode-specific tuning (e.g., structured mode could skip retrieval entirely) is incomplete. |
| **No answer validation** | The LLM response is never validated against the structured data. If the LLM hallucinates a different number, there's no correction. |
| **History not used for context** | Conversation history is included in the prompt as text, but the retrieval step doesn't use prior turns to improve query formulation. |

### 8.3 Recommendations

| Priority | Recommendation |
|----------|---------------|
| 🔴 High | Re-run extraction (`python -m src.extraction.extract`) to regenerate `match_facts.json` with `by_period` data. This enables period-sliceable queries. |
| 🔴 High | Add structured→semantic fallback: if structured returns "empty" and the query is hybrid, execute semantic retrieval as backup. |
| 🟠 Medium | Implement cross-encoder reranker in `rerank()` using `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| 🟠 Medium | Remove dead artifacts: `tfidf_vectorizer.pkl`, `chunk_ids.json`, `embeddings.npy` from the pipeline output. |
| 🟠 Medium | Delete or fully deprecate `01_documents.py` and `resolve_from_text()`. |
| 🟡 Low | Add conversation-history-aware query reformulation: use prior turns to expand follow-up queries before retrieval. |
| 🟡 Low | Add answer validation: compare LLM output against structured data and flag discrepancies. |
| 🟡 Low | Standardize phase numbering across `rebuild.py` comments and `ARCHITECTURE.md`. |

---

## 9. Technology Stack

| Component | Technology | Version/Model |
|-----------|-----------|---------------|
| Embedding Model | sentence-transformers | `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | ChromaDB | Persistent client |
| Sparse Retrieval | rank_bm25 | BM25Okapi |
| LLM API | OpenRouter | Claude Haiku/Sonnet, GPT-4, GPT-3.5, Mistral |
| HTTP Client | httpx | Sync POST requests |
| Data Format | JSON | match_facts.json, documents.json, chunks.json |
| Test Framework | pytest | Unit tests for extraction, routing, structured queries |
| Language | Python 3.14+ | Type hints, dataclasses, pathlib |

---

## 10. Document Levels

| Level | Grain | Count | Content |
|-------|-------|-------|---------|
| **L1** | One per match | 64 | Match summary: teams, score, stage, key events |
| **L2** | One per match | 64 | Key events: goals, cards, substitutions with build-up narration |
| **L3** | One per player per match | ~1995 | Player match performance: minutes, stats, context |
| **L4** | One per player | ~500 | Tournament aggregates: total goals, xG, minutes across all matches |
| **Team** | One per team | 32 | Team tournament aggregates: possession, formations, style |

---

## 11. MetricKind System

| Kind | Description | Period-Sliceable | Examples |
|------|-------------|-----------------|----------|
| **STORED** | Directly stored in records, sum across periods works | ✅ | goals, assists, xG, shots, passes, tackles, carries |
| **DERIVED_RATIO** | Computed from numerator/denominator | ✅ | pass_completion_pct, pass_pct |
| **MATCH_ONLY** | Match-grain value, no per-period breakdown | ❌ | minutes, possession_share, first_shot_minute, first_goal_minute |
