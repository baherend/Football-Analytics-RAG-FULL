"""
test_embedding_config.py -- src.embedding_config is the single source of
truth for embedding-model configuration (replacing four independently
hardcoded "all-MiniLM-L6-v2" string literals). Proves: MiniLM resolves as
the default, MPNet resolves explicitly, and an unknown model id fails
clearly rather than silently falling back to some other model.
"""

from __future__ import annotations

import pytest

from src.embedding_config import (
    DEFAULT_EMBEDDING_MODEL_ID,
    EMBEDDING_MODELS,
    MINILM,
    MPNET_MULTILINGUAL,
    resolve_embedding_config,
)


def test_default_resolves_minilm():
    config = resolve_embedding_config()
    assert config is MINILM
    assert config.model_id == "minilm"
    assert config.hf_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_EMBEDDING_MODEL_ID == "minilm"


def test_explicit_none_matches_default():
    assert resolve_embedding_config(None) == resolve_embedding_config()


def test_explicit_mpnet_resolves_correctly():
    config = resolve_embedding_config("mpnet-multilingual")
    assert config is MPNET_MULTILINGUAL
    assert config.hf_name == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    assert config.dimensions == 768


def test_minilm_and_mpnet_are_distinct_registrations():
    assert MINILM.model_id != MPNET_MULTILINGUAL.model_id
    assert MINILM.hf_name != MPNET_MULTILINGUAL.hf_name
    assert MINILM.dimensions != MPNET_MULTILINGUAL.dimensions


def test_invalid_embedding_id_fails_clearly():
    with pytest.raises(ValueError, match="Unknown embedding model id"):
        resolve_embedding_config("gpt-4-embeddings-that-do-not-exist")


def test_registry_contains_exactly_minilm_and_mpnet():
    """The immediate integration target is MiniLM + MPNet -- BGE-M3 (also
    measured in the multilingual diagnostics) is deliberately not
    registered here per this phase's explicit scope."""
    assert set(EMBEDDING_MODELS) == {"minilm", "mpnet-multilingual"}


def test_every_registered_model_normalizes_embeddings():
    """normalize_embeddings is a fixed project-wide invariant (Chroma's
    l2 default ranks unit-normalized vectors identically to cosine), not a
    per-model choice -- see src.embedding_config's module docstring."""
    for config in EMBEDDING_MODELS.values():
        assert config.normalize_embeddings is True
