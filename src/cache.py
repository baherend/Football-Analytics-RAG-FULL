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


def _compute_data_hash(data_path: Path = Path("output/match_facts.json")) -> str:
    """Compute a current, path-aware signature for cache invalidation."""
    resolved_path = data_path.resolve()

    if not data_path.exists():
        hash_input = f"{resolved_path}:missing"
    else:
        stat = data_path.stat()
        hash_input = f"{resolved_path}:{stat.st_mtime_ns}:{stat.st_size}"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def get_cached_structured_result(
    query_key: str, data_path: Path = Path("output/match_facts.json"),
) -> Any | None:
    """
    Get a cached structured query result.

    `data_path` binds the cache entry to the actual match_facts.json that
    was selected (e.g. a namespaced competition/season artifact) rather
    than always the legacy default -- see src/artifacts.py. Two different
    paths never share a cache entry, even if their contents coincide.

    Returns None if not cached or if the file at `data_path` has changed.
    """
    data_hash = _compute_data_hash(data_path)
    cache_key = f"{data_hash}:{query_key}"

    if cache_key in _structured_cache:
        return _structured_cache[cache_key]

    return None


def set_cached_structured_result(
    query_key: str, result: Any, data_path: Path = Path("output/match_facts.json"),
) -> None:
    """
    Cache a structured query result.

    `data_path` binds the cache entry to the actual match_facts.json that
    was selected -- see get_cached_structured_result(). Automatically
    invalidated when that file changes.
    """
    data_hash = _compute_data_hash(data_path)
    cache_key = f"{data_hash}:{query_key}"
    _structured_cache[cache_key] = result


def clear_all_caches() -> None:
    """Clear all caches (useful for testing)."""
    _model_cache.clear()
    _structured_cache.clear()
