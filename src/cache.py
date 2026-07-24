"""
cache.py — Shared caching for models and computed results.

Caching policy:
- ✅ Embedding models: cache in memory (expensive to load)
- ✅ Structured query results: cache keyed by data hash (self-invalidates)
- ❌ Generated text answers: NEVER cache (cache-coherence risk)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Embedding Model Cache
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Get or load the embedding model. Cached in memory after first load.

    This avoids reloading the model on every query (~1-2 seconds per load).
    """
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name} (cached after first load)")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Structured Result Cache
# ---------------------------------------------------------------------------

_structured_cache: dict[str, Any] = {}
_data_hash_cache: str | None = None


def _compute_data_hash(data_path: Path = Path("output/match_facts.json")) -> str:
    """
    Compute hash of match_facts.json for cache invalidation.

    Uses file modification time + size for fast hashing (not full content hash).
    """
    global _data_hash_cache

    if _data_hash_cache is not None:
        return _data_hash_cache

    if not data_path.exists():
        _data_hash_cache = "no_data"
        return _data_hash_cache

    stat = data_path.stat()
    # Fast hash: mtime + size
    hash_input = f"{stat.st_mtime}:{stat.st_size}"
    _data_hash_cache = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return _data_hash_cache


def get_cached_structured_result(query_key: str) -> Any | None:
    """
    Get a cached structured query result.

    Returns None if not cached or if data has changed.
    """
    data_hash = _compute_data_hash()
    cache_key = f"{data_hash}:{query_key}"

    if cache_key in _structured_cache:
        return _structured_cache[cache_key]

    return None


def set_cached_structured_result(query_key: str, result: Any) -> None:
    """
    Cache a structured query result.

    Automatically invalidated when match_facts.json changes.
    """
    data_hash = _compute_data_hash()
    cache_key = f"{data_hash}:{query_key}"
    _structured_cache[cache_key] = result


def clear_all_caches() -> None:
    """Clear all caches (useful for testing)."""
    global _data_hash_cache
    _model_cache.clear()
    _structured_cache.clear()
    _data_hash_cache = None
