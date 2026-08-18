"""
src/knowledge/indexing/embeddings.py -- chunk embedding generation.

Migration Step 6: extracted verbatim from 04_vector_representation.py, which
keeps its CLI `main()` and re-exports this for compatibility.

The model is selected through src/embedding_config.py, so a competition/season
build never hardcodes a model name.

NOTE: the array this produces (`embeddings.npy`) is a **dead artifact** on the
query path -- `src/retrieval/dense.py` queries Chroma directly (see
05_create_chroma_store.py / src/knowledge/indexing/vector_store.py). It is kept
because the CLI still offers it and removing it is a pipeline behavior change,
not a Step 6 relocation. Recorded in PROJECT_MEMORY.md as deferred cleanup.
"""

from __future__ import annotations

import numpy as np

from src.embedding_config import resolve_embedding_config

__all__ = ["MODEL_NAME", "build_embeddings"]

MODEL_NAME = resolve_embedding_config().hf_name  # legacy default (MiniLM) -- see src.embedding_config


def _build_embeddings(chunks: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    """Generate sentence embeddings for all chunks."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c.get("search_text", c["text"]) for c in chunks]
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

build_embeddings = _build_embeddings
