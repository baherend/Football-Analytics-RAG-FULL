"""
src/knowledge/ -- the offline knowledge pipeline: raw data -> queryable indexes.

Migration Step 6. Stage modules extracted from the root-level numbered
pipeline scripts, which remain as thin CLI orchestrators (they own argument
parsing, competition/season path resolution, and artifact I/O) and re-export
these symbols for their existing callers and tests:

    preprocessing.py     <- 02_preprocessing.py   (text normalization)
    chunking.py          <- 03_chunking.py        (documents -> chunks)
    indexing/bm25.py         <- 04_vector_representation.py
    indexing/embeddings.py   <- 04_vector_representation.py
    indexing/vector_store.py <- 05_create_chroma_store.py

Deliberately NOT moved here, on evidence rather than to complete a diagram:

    src/extraction/  -- already a cohesive package (raw StatsBomb -> match
                        facts); renaming it to knowledge/ingestion/ would be
                        pure churn and would touch every pipeline script plus
                        src/query/parsing.py and src/query/router.py.
    src/rendering/   -- already a cohesive package (facts -> prose documents).

Both already satisfy the knowledge boundary; see PROJECT_MEMORY.md.

Dependency direction: knowledge -> src/artifacts.py + src/embedding_config.py
(technical contracts) only. **Nothing here imports src/retrieval/, src/query/,
src/context/, src/generation/ or src/verification/** -- knowledge produces what
the runtime consumes, never the reverse.
"""

__all__: list[str] = []
