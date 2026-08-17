"""
embedding_config.py -- Single source of truth for embedding-model
configuration: which models are available, their Hugging Face identifiers,
and the short filesystem/Chroma-collection-name-safe alias used to
identify them.

Consumed by index creation (05_create_chroma_store.py), query-time dense
retrieval (src/retrieval/search.py), the embedding-model cache
(src/cache.py), and src.artifacts.ArtifactPaths (which ties dense-index
identity to embedding-model identity via chroma_collection_name).

Before this module existed, "all-MiniLM-L6-v2" was hardcoded independently
in four places (src/cache.py, src/retrieval/search.py,
04_vector_representation.py, 05_create_chroma_store.py). This registry
replaces all four with one lookup.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """One registered embedding model."""

    model_id: str  # short, deterministic alias -- safe in filesystem paths and Chroma collection names
    hf_name: str  # full Hugging Face model identifier passed to SentenceTransformer(...)
    dimensions: int  # informational -- output embedding vector size
    normalize_embeddings: bool = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# normalize_embeddings is fixed True for every registered model, not a
# per-model choice: Chroma's "l2" distance default ranks unit-normalized
# vectors identically to cosine similarity (proven during the multilingual
# root-cause investigation), so changing it would be a similarity-function
# experiment, not a model-configuration concern -- out of scope here.

MINILM = EmbeddingModelConfig(
    model_id="minilm",
    hf_name="sentence-transformers/all-MiniLM-L6-v2",
    dimensions=384,
)

MPNET_MULTILINGUAL = EmbeddingModelConfig(
    model_id="mpnet-multilingual",
    hf_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    dimensions=768,
)

EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    MINILM.model_id: MINILM,
    MPNET_MULTILINGUAL.model_id: MPNET_MULTILINGUAL,
}

# The production default -- every existing zero-argument caller must keep
# resolving to this model.
DEFAULT_EMBEDDING_MODEL_ID = MINILM.model_id


def resolve_embedding_config(model_id: str | None = None) -> EmbeddingModelConfig:
    """
    Resolve a short model_id alias to its EmbeddingModelConfig.

    `model_id=None` resolves to the project default (MiniLM) -- the
    zero-argument production behavior every existing caller must keep.
    An unknown alias fails clearly (ValueError) rather than silently
    falling back to the default or to some other model.
    """
    resolved_id = model_id if model_id is not None else DEFAULT_EMBEDDING_MODEL_ID
    try:
        return EMBEDDING_MODELS[resolved_id]
    except KeyError:
        available = ", ".join(sorted(EMBEDDING_MODELS))
        raise ValueError(
            f"Unknown embedding model id {resolved_id!r}. Available: {available}."
        ) from None
