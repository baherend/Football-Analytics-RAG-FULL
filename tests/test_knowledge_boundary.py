"""
test_knowledge_boundary.py -- Migration Step 6 contracts for the offline
knowledge pipeline.

Pins architectural properties that had no test before this phase: the
dependency direction (knowledge produces, rag consumes -- never the reverse),
the compatibility re-exports the root pipeline scripts still owe their
callers, and the removal of the duplicate query-time retrieval that used to
live inside the indexing script.

Stage *behavior* (chunk shape, tokenization, path resolution) is covered by
the existing test_artifact_paths.py suite and is not duplicated here.
"""

from __future__ import annotations

import ast
import pathlib
from importlib import import_module

import pytest

RAG_PACKAGES = (
    "src.retrieval",
    "src.query",
    "src.context",
    "src.generation",
    "src.verification",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


# --- Dependency direction ---------------------------------------------------


def test_knowledge_never_imports_rag():
    """knowledge/ *produces* indexes; rag/ *consumes* them. A knowledge -> rag
    import would invert the offline/online boundary -- exactly the violation
    that 04_vector_representation.py's duplicate hybrid_search() represented.
    """
    offenders = []
    for path in pathlib.Path("src/knowledge").rglob("*.py"):
        for module in _imported_modules(path):
            if any(module.startswith(pkg) for pkg in RAG_PACKAGES):
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, f"knowledge must not import rag layers: {offenders}"


def test_knowledge_depends_only_on_technical_contracts():
    """Permitted src dependencies are the technical ones (artifact paths,
    embedding registry). Football business logic must not leak in from the
    runtime layers."""
    allowed = {"src.artifacts", "src.embedding_config"}
    offenders = []
    for path in pathlib.Path("src/knowledge").rglob("*.py"):
        for module in _imported_modules(path):
            if not module.startswith("src."):
                continue
            if module.startswith("src.knowledge"):
                continue          # intra-package imports are fine
            if module not in allowed:
                offenders.append(f"{path.as_posix()}: {module}")
    assert not offenders, (
        f"unexpected src dependency in knowledge/ (allowed: {sorted(allowed)}): {offenders}"
    )


def test_knowledge_modules_import_standalone():
    """No import cycles, and no filesystem side effects at import time."""
    for name in (
        "src.knowledge.preprocessing",
        "src.knowledge.chunking",
        "src.knowledge.indexing.bm25",
        "src.knowledge.indexing.embeddings",
        "src.knowledge.indexing.vector_store",
    ):
        assert import_module(name) is not None


def test_knowledge_stage_modules_do_no_import_time_file_io():
    """The root scripts still read files at import (legacy, documented debt);
    the extracted stage modules must not -- that is what makes them testable
    and competition-portable."""
    offenders = []
    for path in pathlib.Path("src/knowledge").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:                      # module level only
            if isinstance(node, (ast.Expr, ast.Assign)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                        if sub.func.id in {"open", "load_documents", "build_chunks"}:
                            offenders.append(f"{path.as_posix()}: {sub.func.id}()")
    assert not offenders, f"import-time file I/O in knowledge/: {offenders}"


# --- Duplicate retrieval removed -------------------------------------------


@pytest.mark.parametrize("symbol", ["hybrid_search", "_ensure_loaded", "min_max_normalize"])
def test_indexing_script_no_longer_implements_query_time_retrieval(symbol):
    """04_vector_representation.py carried a full duplicate of the query path
    (hardcoded legacy output/ paths, so not competition-portable, and reading
    the dead embeddings.npy). It was measurably dead -- zero references
    outside the file -- and query-time retrieval belongs to src/retrieval/.
    """
    vector_rep = import_module("04_vector_representation")
    assert not hasattr(vector_rep, symbol), (
        f"04_vector_representation.{symbol} is back -- query-time retrieval "
        "belongs to src/retrieval/, not the offline indexing script."
    )


def test_query_time_retrieval_still_lives_in_rag():
    from src.retrieval.search import hybrid_search

    assert callable(hybrid_search)


# --- Compatibility surface the root scripts still owe -----------------------


@pytest.mark.parametrize(
    "script,symbols",
    [
        ("02_preprocessing", ["preprocess_text", "normalize_unicode", "clean_whitespace",
                              "normalize_punctuation", "remove_control_chars", "main"]),
        ("03_chunking", ["build_chunks", "chunk_document", "split_sentences",
                         "MAX_CHUNK_SIZE", "CHUNK_OVERLAP", "chunks", "main"]),
        ("04_vector_representation", ["_build_bm25", "_build_embeddings",
                                      "simple_tokenize", "MODEL_NAME", "main"]),
        ("05_create_chroma_store", ["create_vector_store", "get_collection",
                                    "DB_PATH", "COLLECTION_NAME", "MODEL_NAME", "main"]),
    ],
)
def test_pipeline_scripts_keep_their_public_surface(script, symbols):
    """rebuild.py shells out to these scripts and four test modules import
    them by name; the CLI + symbol surface must survive the extraction."""
    module = import_module(script)
    missing = [s for s in symbols if not hasattr(module, s)]
    assert not missing, f"{script} lost {missing}"


def test_script_symbols_are_the_knowledge_implementations():
    """The re-exports must delegate, not duplicate -- otherwise two copies of
    the logic can drift apart."""
    from src.knowledge.chunking import chunk_document
    from src.knowledge.indexing.bm25 import simple_tokenize
    from src.knowledge.indexing.vector_store import create_vector_store
    from src.knowledge.preprocessing import preprocess_text

    assert import_module("02_preprocessing").preprocess_text is preprocess_text
    assert import_module("03_chunking").chunk_document is chunk_document
    assert import_module("04_vector_representation").simple_tokenize is simple_tokenize
    assert import_module("05_create_chroma_store").create_vector_store is create_vector_store


# --- Competition portability ------------------------------------------------


def test_no_hardcoded_competition_identity_in_knowledge_stages():
    """A new competition must be addable by configuration/data, not by editing
    pipeline code. The stage modules must not embed WC2022's identity."""
    offenders = []
    for path in pathlib.Path("src/knowledge").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for marker in ("competition_id=43", "season_id=106", "43, 106"):
            if marker in source:
                offenders.append(f"{path.as_posix()}: {marker}")
    assert not offenders, f"hardcoded competition identity in knowledge/: {offenders}"


def test_vector_store_collection_name_is_resolved_not_hardcoded():
    """The Chroma collection name must come from src/artifacts.py so each
    competition/season gets its own collection."""
    source = pathlib.Path("src/knowledge/indexing/vector_store.py").read_text(encoding="utf-8")
    assert "resolve_chroma_collection_name" in source
