# ⚽ Football Analytics RAG System

A production-oriented **Retrieval-Augmented Generation (RAG)** system for football analytics built on StatsBomb data. The project combines structured football facts, hybrid retrieval, multilingual and Arabic query support, competition portability, answerability checks, and ground-truth evaluation.

> **Current scope:** FIFA World Cup 2022 baseline with portability support for additional competitions, including EPL 2015/16.

## 🚀 Key Features

- **Competition Portability** — Dataset-aware architecture with competition and season-specific artifacts.
- **Hybrid Retrieval** — BM25 lexical retrieval + dense embeddings + Reciprocal Rank Fusion (RRF).
- **Structured Querying** — Deterministic handling of numeric, comparison, ranking, player, team, and match questions.
- **Query Routing** — Automatically selects the appropriate structured or semantic retrieval path.
- **Multilingual & Arabic Support** — Retrieval safeguards and Arabic-aware tokenization for cross-language football queries.
- **Evidence-Aware Context Selection** — Selects relevant chunks while reducing duplicate or weak evidence.
- **Answerability Checks** — Validates whether retrieved evidence is sufficient before generation.
- **Ground-Truth Evaluation** — Evaluates BM25, dense, and hybrid retrieval using reproducible test cases and ranking metrics.
- **StatsBomb Integration** — Structured extraction and document generation directly from football event data.
- **CLI + Streamlit Interface** — Supports both command-line analysis and an interactive web application.

## Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd football-analytics-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download StatsBomb open data
#    Download from: https://github.com/statsbomb/open-data
#    Extract to: open-data-master/data/

# 4. Rebuild all artifacts
python rebuild.py

# 5. Set your API key (for LLM generation)
export OPENROUTER_API_KEY="your-key-here"

# 6. Ask a question
python chat.py "How many goals did Messi score?"
```

## Project Structure

```text
football-analytics-rag/
|-- 01_documents.py              # Load rendered documents
|-- 02_preprocessing.py          # Text preprocessing and cleaning
|-- 03_chunking.py               # Document chunking
|-- 04_vector_representation.py  # BM25 index and sentence embeddings
|-- 05_create_chroma_store.py    # ChromaDB vector index
|-- 07_prompting.py              # Prompting, generation, validation, citations, and answer orchestration
|-- chat.py                      # Interactive CLI
|-- streamlit_app.py             # Streamlit web interface
|-- generate_documents.py        # Document generation CLI
|-- rebuild.py                   # Rebuild dataset artifacts
|
|-- src/
|   |-- artifacts.py             # Dataset-specific artifact paths
|   |-- cache.py                 # Model and runtime caching
|   |-- conversation_memory.py   # Conversation state and memory
|   |-- dataset_catalog.py       # Dataset metadata/catalog
|   |-- stage_taxonomy.py        # Competition-stage taxonomy
|   |
|   |-- extraction/
|   |   |-- extract.py           # StatsBomb extraction entry point
|   |   |-- match_facts.py       # Structured fact extraction and validation
|   |   `-- minutes_played.py    # Minutes-played calculation
|   |
|   |-- rendering/
|   |   `-- render.py            # Structured facts to text documents
|   |
|   |-- retrieval/
|   |   |-- search.py            # BM25, dense, semantic, and hybrid retrieval
|   |   |-- answerability.py     # Deterministic answerability checks
|   |   `-- chunk_selector.py    # Evidence-aware chunk selection
|   |
|   `-- query/
|       |-- router.py            # Query classification, routing, execution, and CLI entry point
|       |-- resolver.py          # Deterministic structured-query resolution
|       |-- query_schema.py      # Structured query/result schemas
|       `-- vocab.py             # Metrics and query vocabularies
|
|-- tests/                       # Automated regression and evaluation tests
|-- output/                      # Generated dataset/index artifacts
|-- requirements.txt
|-- ARCHITECTURE.md
`-- README.md
```

## Pipeline Architecture

Build-time (indexing) and query-time (answering) are separate flows. The
query-time flow reads the artifacts the build-time flow produces, but
nothing in `src/retrieval/search.py` calls anything in `src/query/router.py`
— the router decides which path a query takes and calls into retrieval
(and the structured resolver), never the other way around.

### Build Flow (indexing)

```text
Raw StatsBomb JSON
        |
        v
src/extraction/extract.py
        |
        v
match_facts.json
        |
        v
generate_documents.py
        |
        v
documents.json
        |
        v
02_preprocessing.py
        |
        v
processed_documents.json
        |
        v
03_chunking.py
        |
        v
chunks.json
        |
        +-------------------------------+
        |                               |
        v                               v
04_vector_representation.py     05_create_chroma_store.py
(BM25 index + embeddings)       (ChromaDB vector index)
```

### Query Flow (answering)

```text
User Query
        |
        v
src/query/router.py
        |
        +--> src/query/resolver.py     (structured evidence)
        |
        +--> src/retrieval/search.py   (semantic / hybrid evidence,
        |                               reads the BM25/Chroma indices
        |                               the build flow produced)
        v
    Evidence
        |
        v
07_prompting.py
        |
        +------+------+
        |             |
        v             v
     chat.py    streamlit_app.py
```

## How to Run

### Interactive Chat

```bash
python chat.py
```

Commands:
- `/context` — Show retrieved context
- `/prompt` — Show full prompt
- `/route` — Show routing decision
- `/model` — Switch LLM model
- `/help` — Show help
- `/quit` — Exit

### Single Question

```bash
python chat.py "How many goals did Messi score?"
```

### Rebuild Artifacts

```bash
python rebuild.py          # Full rebuild
python rebuild.py --quick  # Skip embeddings/ChromaDB
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | For LLM generation | OpenRouter API key |

## Example Queries

| Category | Query |
|----------|-------|
| **Numeric** | "How many goals did Messi score?" |
| **Superlative** | "Who scored the most goals?" |
| **Comparison** | "Compare Mbappé and Messi's performance" |
| **Match-level** | "How did France play in the final?" |
| **Player-specific** | "How did Griezmann perform against Poland?" |
| **Team-level** | "What was Argentina's playing style?" |
| **Stage-filtered** | "Mbappé assists in the semi-final" |

## Data Source

This system uses [StatsBomb Open Data](https://github.com/statsbomb/open-data) for the FIFA World Cup 2022 (competition_id=43, season_id=106).

**Note:** The raw data is not included in this repository. Download it separately and extract to `open-data-master/data/`.


<!-- PROJECT_STATUS_START -->
## Current Project Status - August 2026

The project is being upgraded from a FIFA World Cup 2022-specific RAG system into a competition-portable football analytics platform.

### Competition Portability

The pipeline now supports runtime dataset selection using `competition_id` and `season_id`, with namespaced artifacts and Chroma collections.

Validated datasets:

- FIFA World Cup 2022 - competition `43`, season `106` (legacy baseline)
- English Premier League 2015/16 - competition `2`, season `27` (portability validation)

The EPL dataset was successfully processed through extraction, validation, rendering, chunking, indexing, retrieval, and structured querying without competition-specific code changes.

### Structured Correctness

Completed fixes include:

- Period filter normalization across integer, string, and alias inputs.
- Correct aggregation of derived ratios from combined numerator/denominator components instead of averaging per-match percentages.
- Dataset integrity validation for competition-specific artifacts.

### Retrieval Quality

The retrieval layer uses BM25, dense retrieval, Hybrid Reciprocal Rank Fusion, answer-aware safeguards, sibling expansion, and query-aware chunk selection.

The current WC2022 semantic ground-truth benchmark contains 24 evaluation cases.

Final Hybrid retrieval results at `K=5`:

| Metric | Hybrid Before | Hybrid Final |
| --- | ---: | ---: |
| Hit@5 | 58.3% | **91.7%** |
| Recall@5 | 47.0% | **83.8%** |
| All Required@5 | 41.7% | **79.2%** |
| NDCG@5 | 48.7% | **75.4%** |

The final Hybrid configuration also outperformed the Dense baseline at `K=5`.

Key selector fixes:

- Backfill ranked candidates when lexical/query coverage is exhausted before `max_chunks`.
- Treat missing entity metadata as unknown rather than as an entity conflict.

Relevant retrieval regression suite: **100 tests passed**.

> Retrieval quality metrics above are benchmarked on the WC2022 semantic ground truth. EPL 2015/16 has been validated for portability and runtime correctness but does not yet have an equivalent semantic ground-truth benchmark.

### Faithfulness / Grounded Generation

Generation is grounded against retrieved evidence and, where available, verified structured facts.

- Unsupported semantic queries are stopped before the LLM is called when no authoritative structured answer exists, returning a deterministic refusal: `I don't have enough data to answer this question.`
- A valid structured result is never blocked merely because semantic answerability is unanswerable - structured evidence takes precedence.
- Semantic sources are formatted as `[Source N]` with a `chunk_id`, giving a deterministic mapping from cited sources back to the exact retrieved chunk. The CLI and Streamlit generation paths share this same source-attribution contract.
- When structured evidence is available, it is presented to the LLM as explicitly authoritative. Generated numeric claims are checked against it, and a contradicting claim is corrected using the existing validation layer before the answer is returned.
- Pure-semantic responses are not routed through structured numeric validation.

Faithfulness regression: 9 dedicated tests passed; 82 relevant regression tests passed.

> Scope: general semantic claim verification such as LLM-as-judge, NLI-based hallucination detection, or sentence-level claim verification is not part of the current production scope.

### Comparison Engine

The current comparison core supports two-player, single-metric structured comparisons (for example, "Who scored more goals, Harry Kane or Jamie Vardy?" or "Compare Messi and Mbappe by assists").

- Comparison entities are extracted generically from the query, with clean names free of trailing metric-clause contamination (e.g. a stray "by goals").
- The requested metric is preserved and resolved against the existing metric vocabulary; an explicitly requested but unsupported metric is never silently treated as goals.
- Each comparison produces a machine-readable result holding both entities' authoritative structured values, a deterministic non-negative difference, and a deterministic outcome (entity A higher, entity B higher, or tie) - computed from verified structured values, not inferred by the LLM.
- Result status (`resolved` / `partial` / `empty`) reflects how completely the comparison is backed by structured evidence. An incomplete comparison, where one side has no usable value, does not cross the fully-authoritative generation boundary.
- Generated comparison answers are checked against the structured result: an explicit wrong winner, wrong entity value, wrong stated difference, or a false winner claim over an actual tie is corrected before the final answer is returned. CLI and Streamlit share the same validation path.

Example (EPL 2015/16, competition `2`, season `27`): Harry Kane = 25 goals, Jamie Vardy = 24 goals, difference = 1, Harry Kane higher - a functional structured-comparison validation example, not a benchmark evaluation.

Faithfulness test suite, including comparison-specific coverage: **16 tests passed** (not the full repository test count).

> Scope: the current comparison core supports two-player, single-metric structured comparisons. Team comparison and multi-metric comparison are not yet implemented.

### Development Roadmap

Completed:

- Competition portability
- Dataset integrity validation
- Structured query correctness
- Chunking / Retrieval Quality
- Faithfulness / Grounded Generation
- Two-player / One-metric Comparison Engine Core

Next:

- Multi-Source Football Data Ingestion
- Common Football Schema
- Provenance and Freshness
- Source Adapters
- Current-season API / Web ingestion
- Incremental updates
- Multi-source retrieval and evaluation
- Team comparison
- Multi-metric comparison
- Temporal analytics
- User interface refinement
- Final documentation

Current development branch: `competition-portability`.
<!-- PROJECT_STATUS_END -->

## License

The source code in this repository is licensed under the **MIT License**. See [LICENSE](./LICENSE).

StatsBomb open data is **not covered by this repository's MIT License** and remains subject to StatsBomb's own open-data terms and attribution requirements.
