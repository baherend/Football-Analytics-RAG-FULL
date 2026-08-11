# FIFA World Cup 2022 — RAG Analytics System

A Retrieval-Augmented Generation (RAG) system over StatsBomb open data for the FIFA World Cup 2022. Answers natural-language questions about matches, players, and teams using hybrid retrieval (BM25 + dense embeddings + RRF fusion).

## Features

- **Hybrid Retrieval** — BM25 lexical search + sentence embeddings with Reciprocal Rank Fusion
- **Structured Queries** — Numeric/superlative questions answered directly from structured data
- **Semantic Queries** — Descriptive questions answered via vector search
- **Query Router** — Automatically routes questions to the best retrieval path
- **2,835 documents** — Match summaries, key events, player stats, team analysis
- **64 matches** — Complete FIFA World Cup 2022 coverage

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

```
football-analytics-rag/
├── 01_documents.py              # Legacy pipeline entry
├── 02_preprocessing.py          # Text cleaning
├── 03_chunking.py               # Document chunking
├── 04_representation.py         # TF-IDF / BM25 indices
├── 05_embeddings.py             # Sentence embeddings
├── 06_create_chroma_store.py    # ChromaDB vector store
├── 07_retrieve_context.py       # Hybrid retrieval pipeline
├── 08_router.py                 # Query routing
├── chat.py                      # Interactive CLI
├── generate_documents.py        # Document generation CLI
├── extract.py                   # Structured extraction CLI
├── rebuild.py                   # Rebuild all artifacts
│
├── src/                         # Modular source code
│   ├── cache.py                 # Model & result caching
│   ├── extraction/              # Phase 1: Structured extraction
│   │   ├── minutes_played.py    #   Minutes algorithm
│   │   ├── match_facts.py       #   Schema + extraction logic
│   │   └── extract.py           #   CLI entry point
│   ├── rendering/               # Phase 2: Document rendering
│   │   └── render.py            #   Pure renderer
│   ├── query/                   # Phase 3: Structured queries
│   │   ├── vocab.py             #   Metric/dimension vocabularies
│   │   ├── query_schema.py      #   Query/Result schemas
│   │   └── resolver.py          #   Query executor
│   └── generation/              # Phase 6: LLM generation
│       ├── prompt_builder.py    #   Prompt construction
│       └── llm.py               #   OpenRouter integration
│
├── tests/                       # Unit tests
│   ├── test_extraction.py
│   ├── test_structured.py
│   └── test_router.py
│
├── output/                      # Generated artifacts (not committed)
│   ├── match_facts.json
│   ├── documents.json
│   ├── chunks.json
│   ├── indices/
│   ├── embeddings/
│   └── chroma_db/
│
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## Pipeline Architecture

```
Raw StatsBomb JSON
    │
    ▼
extract.py ──→ match_facts.json (structured layer)
    │
    ▼
generate_documents.py ──→ documents.json (rendered text)
    │
    ▼
02_preprocessing.py ──→ processed_documents.json
    │
    ▼
03_chunking.py ──→ chunks.json
    │
    ├──→ 04_representation.py ──→ BM25/TF-IDF indices
    │
    └──→ 05_embeddings.py ──→ Sentence embeddings
            │
            ▼
        06_create_chroma_store.py ──→ ChromaDB
            │
            ▼
        07_retrieve_context.py (hybrid retrieval)
            │
            ▼
        08_router.py (query routing)
            │
            ▼
        chat.py (interactive CLI)
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

### Development Roadmap

Completed:

- Competition portability
- Dataset integrity validation
- Structured query correctness
- Chunking / Retrieval Quality

Next:

- Faithfulness and grounded generation
- Comparison Engine
- User interface refinement
- Final documentation

Current development branch: `competition-portability`.
<!-- PROJECT_STATUS_END -->

## License

This project is for educational purposes. StatsBomb data is subject to their open data license.
