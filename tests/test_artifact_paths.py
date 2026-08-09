"""
test_artifact_paths.py -- Competition Portability Batch 5: Artifact Namespace

Proves that generated runtime artifacts (match_facts.json, documents.json,
chunks.json, BM25/embeddings indices, Chroma directory) are competition/
season isolated via src.artifacts.ArtifactPaths, so multiple competitions
can coexist without overwriting or accidentally reading each other's data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.artifacts import ArtifactPaths


# ---------------------------------------------------------------------------
# 1-4. ArtifactPaths itself: deterministic, isolated, custom root, consistent
# ---------------------------------------------------------------------------


def test_namespace_is_deterministic_for_wc2022():
    paths = ArtifactPaths(competition_id=43, season_id=106)
    assert paths.root == Path("output") / "competitions" / "43" / "106"
    assert paths.match_facts == Path("output") / "competitions" / "43" / "106" / "match_facts.json"


def test_different_competitions_are_isolated():
    wc2022 = ArtifactPaths(competition_id=43, season_id=106)
    other = ArtifactPaths(competition_id=2, season_id=27)

    assert wc2022.root != other.root
    assert wc2022.match_facts != other.match_facts
    assert wc2022.documents != other.documents
    assert wc2022.chunks != other.chunks
    assert wc2022.indices_dir != other.indices_dir
    assert wc2022.bm25_index != other.bm25_index
    assert wc2022.embeddings_dir != other.embeddings_dir
    assert wc2022.embeddings_file != other.embeddings_file
    assert wc2022.chroma_dir != other.chroma_dir

    # Neither root is a parent of the other -- fully separate subtrees.
    assert not str(other.root).startswith(str(wc2022.root))
    assert not str(wc2022.root).startswith(str(other.root))


def test_different_seasons_of_the_same_competition_are_isolated():
    season_a = ArtifactPaths(competition_id=2, season_id=27)
    season_b = ArtifactPaths(competition_id=2, season_id=28)

    assert season_a.root != season_b.root
    assert season_a.match_facts != season_b.match_facts


def test_custom_output_root_works():
    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("/tmp/custom"))
    assert paths.root == Path("/tmp/custom") / "competitions" / "2" / "27"
    assert paths.match_facts == Path("/tmp/custom") / "competitions" / "2" / "27" / "match_facts.json"


def test_all_dataset_paths_are_under_the_same_root():
    paths = ArtifactPaths(competition_id=2, season_id=27)
    root_str = str(paths.root)

    assert str(paths.match_facts).startswith(root_str)
    assert str(paths.documents).startswith(root_str)
    assert str(paths.processed_documents).startswith(root_str)
    assert str(paths.chunks).startswith(root_str)
    assert str(paths.indices_dir).startswith(root_str)
    assert str(paths.bm25_index).startswith(root_str)
    assert str(paths.embeddings_dir).startswith(root_str)
    assert str(paths.embeddings_file).startswith(root_str)
    assert str(paths.chroma_dir).startswith(root_str)

    # Index/embedding files belong under their respective directories.
    assert str(paths.bm25_index).startswith(str(paths.indices_dir))
    assert str(paths.embeddings_file).startswith(str(paths.embeddings_dir))


def test_ids_only_in_path_no_names():
    """Competition/season NAMES must never appear in filesystem paths --
    only the stable numeric IDs."""
    paths = ArtifactPaths(competition_id=43, season_id=106)
    assert "FIFA" not in str(paths.root)
    assert "World Cup" not in str(paths.root)


# ---------------------------------------------------------------------------
# 5. Extraction persist() can write to a namespaced destination
# ---------------------------------------------------------------------------


def test_extraction_persist_writes_to_namespaced_destination(tmp_path):
    from src.extraction.match_facts import persist, DatasetIdentity

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    result = {"player_match_facts": [], "match_facts": [], "team_match_facts": []}
    identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )

    output_path = persist(
        result, output_dir=paths.root, competition_id=2, season_id=27,
        dataset_identity=identity,
    )

    assert output_path == paths.match_facts
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["metadata"]["competition_id"] == 2
    assert data["metadata"]["competition_name"] == "Test League"


def test_extraction_persist_for_competition_b_does_not_touch_competition_a(tmp_path):
    """No new artifact generated for competition B overwrites competition A."""
    from src.extraction.match_facts import persist, DatasetIdentity

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)

    result = {"player_match_facts": [], "match_facts": [], "team_match_facts": []}
    identity_a = DatasetIdentity(competition_id=2, competition_name="League A", season_id=27, season_name="2025/2026")
    identity_b = DatasetIdentity(competition_id=3, competition_name="League B", season_id=1, season_name="1999")

    persist(result, output_dir=paths_a.root, competition_id=2, season_id=27, dataset_identity=identity_a)
    persist(result, output_dir=paths_b.root, competition_id=3, season_id=1, dataset_identity=identity_b)

    data_a = json.loads(paths_a.match_facts.read_text(encoding="utf-8"))
    data_b = json.loads(paths_b.match_facts.read_text(encoding="utf-8"))

    assert data_a["metadata"]["competition_name"] == "League A"
    assert data_b["metadata"]["competition_name"] == "League B"
    assert paths_a.match_facts != paths_b.match_facts


# ---------------------------------------------------------------------------
# 6-7. Structured resolver loads a namespaced match_facts artifact, and its
#      structured-result cache binds to the actual selected path.
# ---------------------------------------------------------------------------


def _write_match_facts(path: Path, player_name: str, goals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metadata": {"competition_id": 2, "competition_name": "Test League",
                     "season_id": 27, "season_name": "2025/2026"},
        "player_match_facts": [{
            "player_id": 1, "player_name": player_name, "team_name": "Test FC",
            "team_id": 1, "match_id": 1, "match_date": "2026-01-01",
            "stage": "Matchday 1", "opponent": "Rival FC", "is_knockout": False,
            "was_dismissed": False, "roles": [], "minutes": 90.0, "goals": goals,
        }],
        "match_facts": [], "team_match_facts": [],
    }), encoding="utf-8")


def test_resolver_loads_specified_namespaced_match_facts_artifact(tmp_path):
    from src.query.resolver import _load_data

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, "Test Player", goals=5)

    data = _load_data(paths.match_facts)

    assert data["player_match_facts"][0]["goals"] == 5
    assert data["metadata"]["competition_name"] == "Test League"


def test_resolve_accepts_data_path_for_namespaced_artifact(tmp_path):
    from src.query.query_schema import StructuredQuery
    from src.query.resolver import resolve

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, "Test Player", goals=5)

    q = StructuredQuery(intent="numeric", entity="player", metric="goals",
                        aggregation="sum", entity_name="Test Player")
    result = resolve(q, data_path=paths.match_facts)

    assert result.aggregated_value == 5


def test_resolve_structured_cache_binds_to_selected_data_path(tmp_path):
    """
    Structured cache invalidation must use the ACTUAL selected match_facts
    path, not always the legacy output/match_facts.json. Two namespaced
    datasets answering the identical query through the data_path-aware
    cache must never collide.
    """
    from src.query.query_schema import StructuredQuery
    from src.query.resolver import resolve

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)
    _write_match_facts(paths_a.match_facts, "Test Player", goals=1)
    _write_match_facts(paths_b.match_facts, "Test Player", goals=9)

    q = StructuredQuery(intent="numeric", entity="player", metric="goals",
                        aggregation="sum", entity_name="Test Player")

    result_a = resolve(q, data_path=paths_a.match_facts)
    result_b = resolve(q, data_path=paths_b.match_facts)

    assert result_a.aggregated_value == 1
    assert result_b.aggregated_value == 9

    # Re-querying A again (cache hit) must still return A's value, not B's.
    result_a_again = resolve(q, data_path=paths_a.match_facts)
    assert result_a_again.aggregated_value == 1


# ---------------------------------------------------------------------------
# 9. Chroma persistent directory selection is dataset-isolated
#    (collection-name namespacing is explicitly deferred to the next batch)
# ---------------------------------------------------------------------------


class _FakeEmbeddingModel:
    """Stands in for SentenceTransformer so this test doesn't pay real
    model-load/embedding cost -- only directory isolation is under test."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, **kwargs):
        import numpy as np
        return np.zeros((len(texts), 8), dtype="float32")


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id, "document_id": f"{chunk_id}-doc",
        "level": "1", "text": text, "search_text": text,
    }


def test_chroma_persistent_directory_is_dataset_isolated(tmp_path, monkeypatch):
    from importlib import import_module

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEmbeddingModel)
    chroma_mod = import_module("05_create_chroma_store")

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)

    assert paths_a.chroma_dir != paths_b.chroma_dir

    collection_a = chroma_mod.create_vector_store(
        chunks=[_chunk("a1", "League A content")],
        persist_dir=paths_a.chroma_dir,
        collection_name="dataset_a",
    )
    collection_b = chroma_mod.create_vector_store(
        chunks=[_chunk("b1", "League B content"), _chunk("b2", "More league B content")],
        persist_dir=paths_b.chroma_dir,
        collection_name="dataset_b",
    )

    assert paths_a.chroma_dir.exists()
    assert paths_b.chroma_dir.exists()
    assert collection_a.count() == 1
    assert collection_b.count() == 2

    # Loading back via get_collection() at each namespaced directory returns
    # only that dataset's collection -- the two ChromaDB stores never share
    # data merely because they were built in the same test/process.
    reloaded_a = chroma_mod.get_collection(persist_dir=paths_a.chroma_dir, collection_name="dataset_a")
    reloaded_b = chroma_mod.get_collection(persist_dir=paths_b.chroma_dir, collection_name="dataset_b")
    assert reloaded_a.count() == 1
    assert reloaded_b.count() == 2


# ---------------------------------------------------------------------------
# 8. Retrieval BM25/chunk loaders can be configured to read one dataset
#    namespace without sharing path-dependent process caches with another
# ---------------------------------------------------------------------------


def _write_chunks(path: Path, chunk_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [_chunk(cid, f"Text for {cid}") for cid in chunk_ids]
    path.write_text(json.dumps(chunks), encoding="utf-8")


def _write_bm25_index(path: Path, chunk_ids: list[str]) -> None:
    import pickle
    from rank_bm25 import BM25Okapi

    path.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [[cid.lower()] for cid in chunk_ids]
    with open(path, "wb") as f:
        pickle.dump(BM25Okapi(tokenized), f)


def test_chunk_loader_reads_one_namespace_without_leaking_into_another(tmp_path):
    from importlib import import_module
    router = import_module("06_retrieve_context")

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)
    _write_chunks(paths_a.chunks, ["a1", "a2"])
    _write_chunks(paths_b.chunks, ["b1"])

    chunks_a = router._load_chunks(paths_a.chunks)
    chunks_b = router._load_chunks(paths_b.chunks)

    assert [c["chunk_id"] for c in chunks_a] == ["a1", "a2"]
    assert [c["chunk_id"] for c in chunks_b] == ["b1"]

    # Re-reading A again (process-cache hit) must still return A's chunks.
    chunks_a_again = router._load_chunks(paths_a.chunks)
    assert [c["chunk_id"] for c in chunks_a_again] == ["a1", "a2"]


def test_bm25_loader_reads_one_namespace_without_leaking_into_another(tmp_path):
    from importlib import import_module
    router = import_module("06_retrieve_context")

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)
    _write_bm25_index(paths_a.bm25_index, ["a1", "a2"])
    _write_bm25_index(paths_b.bm25_index, ["b1"])

    bm25_a = router._load_bm25_index(paths_a.bm25_index)
    bm25_b = router._load_bm25_index(paths_b.bm25_index)

    assert bm25_a.corpus_size == 2
    assert bm25_b.corpus_size == 1

    # Re-reading A again (process-cache hit) must still return A's index.
    bm25_a_again = router._load_bm25_index(paths_a.bm25_index)
    assert bm25_a_again.corpus_size == 2


# ---------------------------------------------------------------------------
# 10-11. resolve_output_dir(): the one narrow, documented legacy-WC2022
#        boundary. Legacy default keeps working; non-WC never falls back.
# ---------------------------------------------------------------------------


def test_resolve_output_dir_uses_legacy_flat_layout_for_wc2022_default(tmp_path):
    from src.artifacts import resolve_output_dir

    assert resolve_output_dir(43, 106, output_root=tmp_path) == tmp_path


def test_resolve_output_dir_namespaces_any_other_competition(tmp_path):
    from src.artifacts import resolve_output_dir

    result = resolve_output_dir(2, 27, output_root=tmp_path)

    assert result == ArtifactPaths(2, 27, output_root=tmp_path).root
    assert result != tmp_path  # never the shared legacy flat directory


def test_resolve_output_dir_non_wc_never_shares_wc2022_root(tmp_path):
    from src.artifacts import resolve_output_dir

    wc2022_dir = resolve_output_dir(43, 106, output_root=tmp_path)
    other_dir = resolve_output_dir(2, 27, output_root=tmp_path)
    another_dir = resolve_output_dir(3, 1, output_root=tmp_path)

    assert wc2022_dir != other_dir
    assert other_dir != another_dir
    assert wc2022_dir != another_dir


# ---------------------------------------------------------------------------
# Gap 1 fix: explicit namespaced WC2022 builds must be possible.
# ArtifactPaths itself stays competition-agnostic; resolve_output_dir()'s
# legacy_default flag is the only opt-out of the legacy boundary.
# ---------------------------------------------------------------------------


def test_artifact_paths_wc2022_root_is_always_namespaced(tmp_path):
    """ArtifactPaths itself makes no legacy exception -- 43/106 namespaces
    exactly like any other competition when constructed directly."""
    paths = ArtifactPaths(competition_id=43, season_id=106, output_root=tmp_path)
    assert paths.root == tmp_path / "competitions" / "43" / "106"


def test_resolve_output_dir_defaults_wc2022_to_legacy_flat_layout(tmp_path):
    from src.artifacts import resolve_output_dir

    # legacy_default defaults to True -- unchanged existing behavior.
    assert resolve_output_dir(43, 106, output_root=tmp_path) == tmp_path


def test_resolve_output_dir_supports_explicit_namespaced_wc2022(tmp_path):
    """An explicitly requested namespaced WC2022 build must be possible --
    legacy_default=False opts out of the legacy flat layout even for 43/106."""
    from src.artifacts import resolve_output_dir

    result = resolve_output_dir(43, 106, output_root=tmp_path, legacy_default=False)

    assert result == ArtifactPaths(43, 106, output_root=tmp_path).root
    assert result != tmp_path


def test_resolve_output_dir_legacy_default_flag_has_no_effect_on_non_wc_datasets():
    """A non-WC dataset must never fall back to legacy artifacts, regardless
    of legacy_default -- the flag only ever affects the WC2022 IDs."""
    from src.artifacts import resolve_output_dir

    with_legacy_true = resolve_output_dir(2, 27, legacy_default=True)
    with_legacy_false = resolve_output_dir(2, 27, legacy_default=False)

    assert with_legacy_true == with_legacy_false == ArtifactPaths(2, 27).root
    assert with_legacy_true != Path("output")


# ---------------------------------------------------------------------------
# extract.py CLI threads the namespaced output_dir through to persist()
# ---------------------------------------------------------------------------


def test_extract_cli_persists_to_namespaced_output_dir_for_custom_competition(monkeypatch, tmp_path):
    import sys
    import src.extraction.extract as extract_cli

    calls = {}

    def fake_extract_all(data_root, verbose=True, competition_id=None, season_id=None):
        return {
            "player_match_facts": [], "match_facts": [], "team_match_facts": [],
            "diagnostics": {"matches_processed": 0, "total_player_facts": 0,
                            "total_team_facts": 0, "card_count_mismatches": []},
        }

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None,
                      dataset_identity=None):
        calls["output_dir"] = output_dir
        return output_dir / "match_facts.json"

    monkeypatch.setattr(extract_cli, "extract_all", fake_extract_all)
    monkeypatch.setattr(extract_cli, "persist", fake_persist)
    monkeypatch.setattr(sys, "argv", ["extract.py", "--competition-id", "2", "--season-id", "27", "--quiet"])

    assert extract_cli.main() == 0
    assert calls["output_dir"] == ArtifactPaths(2, 27).root


def test_extract_cli_persists_to_legacy_dir_for_wc2022_default(monkeypatch, tmp_path):
    import sys
    import src.extraction.extract as extract_cli

    calls = {}

    def fake_extract_all(data_root, verbose=True, competition_id=None, season_id=None):
        return {
            "player_match_facts": [], "match_facts": [], "team_match_facts": [],
            "diagnostics": {"matches_processed": 0, "total_player_facts": 0,
                            "total_team_facts": 0, "card_count_mismatches": []},
        }

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None,
                      dataset_identity=None):
        calls["output_dir"] = output_dir
        return output_dir / "match_facts.json"

    monkeypatch.setattr(extract_cli, "extract_all", fake_extract_all)
    monkeypatch.setattr(extract_cli, "persist", fake_persist)
    monkeypatch.setattr(sys, "argv", ["extract.py", "--quiet"])

    assert extract_cli.main() == 0
    assert calls["output_dir"] == Path("output")


# ---------------------------------------------------------------------------
# generate_documents.py CLI reads/writes the namespaced directory
# ---------------------------------------------------------------------------


def test_generate_documents_cli_uses_namespaced_directory(monkeypatch, tmp_path):
    """
    Sandboxes CWD to tmp_path (generate_documents.py's output_dir is
    derived relative to the process CWD via resolve_output_dir's default
    output_root=Path("output")) so this test can never touch the real
    repository's output/ directory.
    """
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.match_facts.write_text(json.dumps({
        "metadata": {"competition_id": 2, "competition_name": "Test League",
                     "season_id": 27, "season_name": "2025/2026"},
        "player_match_facts": [], "match_facts": [], "team_match_facts": [],
    }), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["generate_documents.py", "--competition-id", "2",
                                       "--season-id", "27", "--quiet"])

    import generate_documents
    importlib.reload(generate_documents)

    assert generate_documents.main() == 0
    assert paths.documents.exists()
    written = json.loads(paths.documents.read_text(encoding="utf-8"))
    assert written == []  # no match_facts records in this minimal fixture

    # The real repo output/ directory (relative to the ORIGINAL cwd) must
    # never be touched by this run -- verified by construction, since CWD
    # was sandboxed to tmp_path for the entire call.


# ---------------------------------------------------------------------------
# 05_create_chroma_store.py CLI: namespaced directory, no legacy fallback
# ---------------------------------------------------------------------------


def test_chroma_store_cli_errors_instead_of_falling_back_to_legacy_chunks(monkeypatch, tmp_path):
    """
    A non-WC2022 competition with no namespaced chunks.json must fail
    explicitly, never silently read the legacy output/chunks.json.
    """
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["05_create_chroma_store.py",
                                       "--competition-id", "2", "--season-id", "27"])

    chroma_mod = importlib.import_module("05_create_chroma_store")
    importlib.reload(chroma_mod)

    assert chroma_mod.main() == 1  # explicit failure, no chunks.json present
    assert not (tmp_path / "output" / "chunks.json").exists()


def test_chroma_store_cli_uses_namespaced_chunks_and_persist_dir(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEmbeddingModel)
    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    _write_chunks(paths.chunks, ["a1", "a2"])

    monkeypatch.setattr(sys, "argv", ["05_create_chroma_store.py",
                                       "--competition-id", "2", "--season-id", "27"])

    chroma_mod = importlib.import_module("05_create_chroma_store")
    importlib.reload(chroma_mod)

    assert chroma_mod.main() == 0
    assert paths.chroma_dir.exists()

    collection = chroma_mod.get_collection(persist_dir=paths.chroma_dir, collection_name=paths.chroma_collection_name)
    assert collection.count() == 2


# ---------------------------------------------------------------------------
# rebuild.py threads competition/season identity to child pipeline steps
# ---------------------------------------------------------------------------


def test_rebuild_threads_identity_to_extraction_and_rendering_and_chroma_steps(monkeypatch, tmp_path):
    """
    Verifies command construction only -- does not actually execute the
    (slow, real-data) pipeline. Confirms:
    - --competition-id/--season-id reach the extraction, rendering, and
      Chroma steps;
    - quick mode still skips the Chroma step.
    """
    import rebuild

    (tmp_path / "open-data-master" / "data").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    commands_run = []

    def fake_run(command, description):
        commands_run.append(command)
        return True

    monkeypatch.setattr(rebuild, "run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["rebuild.py", "--competition-id", "2", "--season-id", "27"],
    )

    assert rebuild.main() == 0

    extraction_cmd = next(c for c in commands_run if "src.extraction.extract" in c)
    rendering_cmd = next(c for c in commands_run if "generate_documents.py" in c)
    preprocessing_cmd = next(c for c in commands_run if "02_preprocessing.py" in c)
    chunking_cmd = next(c for c in commands_run if "03_chunking.py" in c)
    vector_rep_cmd = next(c for c in commands_run if "04_vector_representation.py" in c)
    chroma_cmd = next(c for c in commands_run if "05_create_chroma_store.py" in c)

    for cmd in (extraction_cmd, rendering_cmd, preprocessing_cmd, chunking_cmd,
                vector_rep_cmd, chroma_cmd):
        assert "--competition-id" in cmd and "2" in cmd
        assert "--season-id" in cmd and "27" in cmd


def test_rebuild_namespaced_flag_reaches_every_artifact_producing_step(monkeypatch, tmp_path):
    import rebuild

    (tmp_path / "open-data-master" / "data").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    commands_run = []
    monkeypatch.setattr(rebuild, "run", lambda command, description: commands_run.append(command) or True)
    monkeypatch.setattr("sys.argv", ["rebuild.py", "--namespaced"])

    assert rebuild.main() == 0

    # Every step except the bare `-m src.extraction.extract` module prefix
    # should carry --namespaced through.
    steps_with_identity = [c for c in commands_run if "--competition-id" in c]
    assert len(steps_with_identity) == 6  # extract, render, preprocess, chunk, vectorize, chroma
    for cmd in steps_with_identity:
        assert "--namespaced" in cmd


def test_rebuild_quick_mode_skips_chroma_step(monkeypatch, tmp_path):
    import rebuild

    (tmp_path / "open-data-master" / "data").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    commands_run = []
    monkeypatch.setattr(rebuild, "run", lambda command, description: commands_run.append(command) or True)
    monkeypatch.setattr("sys.argv", ["rebuild.py", "--quick"])

    assert rebuild.main() == 0
    assert not any("05_create_chroma_store.py" in c for c in commands_run)


# ---------------------------------------------------------------------------
# Gap 3 fix: 02_preprocessing.py, 03_chunking.py, 04_vector_representation.py
# participate in the namespaced rebuild end to end.
# ---------------------------------------------------------------------------


def _fake_document(doc_id: str, text: str) -> dict:
    return {
        "document_id": doc_id, "level": "1", "text": text,
        "metadata": {}, "match_id": None, "player_name": None, "team_name": None,
    }


def test_preprocessing_cli_reads_and_writes_namespaced_documents(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.documents.write_text(json.dumps([
        _fake_document("d1", "  Hello\u2019s   world  "),
    ]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["02_preprocessing.py", "--competition-id", "2",
                                       "--season-id", "27", "--quiet"])
    preprocessing = importlib.import_module("02_preprocessing")
    importlib.reload(preprocessing)

    assert preprocessing.main() == 0
    assert paths.processed_documents.exists()
    written = json.loads(paths.processed_documents.read_text(encoding="utf-8"))
    assert written[0]["cleaned_text"] == "Hello's world"


def test_chunking_cli_reads_and_writes_namespaced_chunks(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    paths.root.mkdir(parents=True, exist_ok=True)
    long_text = "Sentence one. " * 60  # forces multi-chunk splitting
    doc = _fake_document("d1", long_text.strip())
    doc["cleaned_text"] = doc["text"]
    paths.processed_documents.write_text(json.dumps([doc]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["03_chunking.py", "--competition-id", "2",
                                       "--season-id", "27", "--quiet"])
    chunking = importlib.import_module("03_chunking")
    importlib.reload(chunking)

    assert chunking.main() == 0
    assert paths.chunks.exists()
    written = json.loads(paths.chunks.read_text(encoding="utf-8"))
    assert len(written) >= 1
    assert all(c["document_id"] == "d1" for c in written)


def test_chunking_cli_falls_back_to_documents_when_processed_absent(monkeypatch, tmp_path):
    """Chunking reads processed_documents when present, else documents.json
    from the SAME namespace -- never the legacy directory."""
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.documents.write_text(json.dumps([_fake_document("d1", "Short text.")]), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["03_chunking.py", "--competition-id", "2",
                                       "--season-id", "27", "--quiet"])
    chunking = importlib.import_module("03_chunking")
    importlib.reload(chunking)

    assert chunking.main() == 0
    assert paths.chunks.exists()
    assert not paths.processed_documents.exists()


def test_vector_representation_cli_writes_namespaced_indices_and_embeddings(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEmbeddingModel)
    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    _write_chunks(paths.chunks, ["a1", "a2", "a3"])

    monkeypatch.setattr(sys, "argv", ["04_vector_representation.py", "--competition-id", "2",
                                       "--season-id", "27", "--quiet"])
    vector_rep = importlib.import_module("04_vector_representation")
    importlib.reload(vector_rep)

    assert vector_rep.main() == 0
    assert paths.bm25_index.exists()
    assert paths.embeddings_file.exists()

    import pickle
    with open(paths.bm25_index, "rb") as f:
        bm25 = pickle.load(f)
    assert bm25.corpus_size == 3

    import numpy as np
    embeddings = np.load(paths.embeddings_file)
    assert embeddings.shape[0] == 3


def test_namespaced_pipeline_end_to_end_for_competition_b_never_touches_legacy(monkeypatch, tmp_path):
    """
    Full documents -> processed -> chunks -> indices/embeddings chain for a
    non-WC2022 competition, proving no step falls back to (or writes into)
    the legacy output/ layout.
    """
    import sys
    import importlib

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEmbeddingModel)
    monkeypatch.chdir(tmp_path)

    paths = ArtifactPaths(competition_id=3, season_id=1, output_root=Path("output"))
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.documents.write_text(json.dumps([_fake_document("d1", "Competition B content.")]), encoding="utf-8")

    common_argv_tail = ["--competition-id", "3", "--season-id", "1", "--quiet"]

    monkeypatch.setattr(sys, "argv", ["02_preprocessing.py", *common_argv_tail])
    preprocessing = importlib.import_module("02_preprocessing")
    importlib.reload(preprocessing)
    assert preprocessing.main() == 0

    monkeypatch.setattr(sys, "argv", ["03_chunking.py", *common_argv_tail])
    chunking = importlib.import_module("03_chunking")
    importlib.reload(chunking)
    assert chunking.main() == 0

    monkeypatch.setattr(sys, "argv", ["04_vector_representation.py", *common_argv_tail])
    vector_rep = importlib.import_module("04_vector_representation")
    importlib.reload(vector_rep)
    assert vector_rep.main() == 0

    assert paths.processed_documents.exists()
    assert paths.chunks.exists()
    assert paths.bm25_index.exists()
    assert paths.embeddings_file.exists()

    # The legacy flat layout (relative to sandboxed cwd) must never have
    # been created by this competition-B run.
    assert not (tmp_path / "output" / "processed_documents.json").exists()
    assert not (tmp_path / "output" / "chunks.json").exists()
    assert not (tmp_path / "output" / "indices").exists()


# ---------------------------------------------------------------------------
# Gap 2 fix: runtime retrieval (bm25_search, dense_search, hybrid_search,
# retrieve_context, execute_route, route_and_execute, and the sibling-
# expansion safeguard) is namespace-aware end to end.
# ---------------------------------------------------------------------------


def _build_full_dataset(paths: ArtifactPaths, chroma_mod, chunk_texts: dict) -> None:
    """Populate one namespaced dataset's chunks.json, bm25.pkl, and Chroma
    collection from {chunk_id: text}."""
    chunk_ids = list(chunk_texts)
    paths.root.mkdir(parents=True, exist_ok=True)
    chunk_dicts = [_chunk(cid, text) for cid, text in chunk_texts.items()]
    paths.chunks.write_text(json.dumps(chunk_dicts), encoding="utf-8")
    _write_bm25_index(paths.bm25_index, chunk_ids)
    chroma_mod.create_vector_store(
        chunks=chunk_dicts, persist_dir=paths.chroma_dir, collection_name=paths.chroma_collection_name,
    )


def test_hybrid_search_uses_only_selected_dataset_artifacts(monkeypatch, tmp_path):
    """
    Proof 1 + 4: hybrid retrieval passes competition-B BM25/chunks/Chroma
    paths all the way through, and two dataset namespaces used
    sequentially in one process do not cross-read each other's artifacts.
    """
    from importlib import import_module
    router = import_module("06_retrieve_context")
    chroma_mod = import_module("05_create_chroma_store")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeEmbeddingModel)

    paths_a = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)

    _build_full_dataset(paths_a, chroma_mod, {"a1": "Alpha content about a match", "a2": "More alpha text"})
    _build_full_dataset(paths_b, chroma_mod, {"b1": "Beta content about a match", "b2": "More beta text"})

    results_b = router.hybrid_search("content about a match", k=5, artifact_paths=paths_b)
    result_ids_b = {r["chunk_id"] for r in results_b}
    assert result_ids_b, "expected at least one result from dataset B"
    assert result_ids_b <= {"b1", "b2"}
    assert not (result_ids_b & {"a1", "a2"})

    # Sequential re-query of A in the same process must still return only A.
    results_a = router.hybrid_search("content about a match", k=5, artifact_paths=paths_a)
    result_ids_a = {r["chunk_id"] for r in results_a}
    assert result_ids_a, "expected at least one result from dataset A"
    assert result_ids_a <= {"a1", "a2"}
    assert not (result_ids_a & {"b1", "b2"})

    # And B again -- must still be B, not contaminated by the intervening A call.
    results_b_again = router.hybrid_search("content about a match", k=5, artifact_paths=paths_b)
    assert {r["chunk_id"] for r in results_b_again} <= {"b1", "b2"}


def test_execute_route_structured_path_uses_selected_match_facts(tmp_path):
    """Proof 2: execute_route's structured path resolves against the
    selected competition/season's match_facts, not the legacy default."""
    from importlib import import_module
    router = import_module("06_retrieve_context")

    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)
    _write_match_facts(paths_b.match_facts, "Test Player", goals=9)

    route = router.Route(
        path="structured", confidence=0.9, reason="test",
        structured_query=router.StructuredQuery(
            intent="numeric", entity="player", metric="goals",
            aggregation="sum", entity_name="Test Player",
        ),
    )

    result = router.execute_route(route, artifact_paths=paths_b)

    assert result.structured_result is not None
    assert result.structured_result.aggregated_value == 9


def test_sibling_expansion_uses_selected_chunks_not_legacy(monkeypatch, tmp_path):
    """
    Proof 3: the sibling-expansion safeguard (_expand_query_entity_siblings)
    must load the SELECTED dataset's chunks, never silently defaulting to
    the legacy path, when artifact_paths is given.
    """
    from importlib import import_module
    router = import_module("06_retrieve_context")

    paths_b = ArtifactPaths(competition_id=3, season_id=1, output_root=tmp_path)
    _write_chunks(paths_b.chunks, ["b1", "b2"])

    seen_paths = []
    real_load_chunks = router._load_chunks

    def spy_load_chunks(path=None):
        seen_paths.append(path)
        return real_load_chunks(path)

    monkeypatch.setattr(router, "_load_chunks", spy_load_chunks)

    results = [{"chunk_id": "existing", "metadata": {"team_name": "Team B"}, "team_name": "Team B",
                "document_id": "b1-doc"}]

    router._expand_query_entity_siblings("Team B history", results, artifact_paths=paths_b)

    assert seen_paths == [paths_b.chunks]
    assert None not in seen_paths


def test_default_runtime_calls_remain_legacy_compatible(monkeypatch):
    """
    Proof 5: calling the runtime retrieval functions with no artifact_paths
    argument must keep using the legacy module-level defaults (path=None
    reaching the loaders, persist_dir/collection_name at their defaults) --
    unchanged behavior for every existing zero-argument production caller.
    """
    from importlib import import_module
    router = import_module("06_retrieve_context")

    seen = {}

    monkeypatch.setattr(router, "_load_bm25_index", lambda path=None: seen.setdefault("bm25_path", path) or object())
    monkeypatch.setattr(router, "_load_chunks", lambda path=None: seen.setdefault("chunks_path", path) or [])

    def fake_dense_search(query, k=20, level_filter=None, persist_dir=router.CHROMA_DIR,
                          collection_name=router.COLLECTION_NAME, artifact_paths=None):
        seen["dense_persist_dir"] = persist_dir
        seen["dense_collection_name"] = collection_name
        return []

    monkeypatch.setattr(router, "dense_search", fake_dense_search)

    class _FakeBM25:
        def get_scores(self, tokens):
            import numpy as np
            return np.array([])

    monkeypatch.setattr(router, "_load_bm25_index", lambda path=None: (seen.setdefault("bm25_path", path), _FakeBM25())[1])

    router.hybrid_search("legacy default query", k=3)

    assert seen["bm25_path"] is None
    assert seen["chunks_path"] is None
    assert seen["dense_persist_dir"] == router.CHROMA_DIR
    assert seen["dense_collection_name"] == router.COLLECTION_NAME


# ---------------------------------------------------------------------------
# Final gap fix: 06_retrieve_context.py's CLI (main()) can select a dataset
# and threads it into execute_route() as artifact_paths.
# ---------------------------------------------------------------------------


def _stub_execute_route(router, monkeypatch):
    """Replace execute_route with a spy that records artifact_paths and
    returns a minimal RoutedResult, so main() never touches real Chroma
    or artifacts."""
    captured = {}

    def fake_execute_route(route, semantic_k=3, original_query="", artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return router.RoutedResult(route=route, explanation="stub")

    monkeypatch.setattr(router, "execute_route", fake_execute_route)
    return captured


def test_cli_default_query_passes_artifact_paths_none(monkeypatch):
    """Plain existing CLI usage (no flags) must remain backward-compatible
    and pass artifact_paths=None -- i.e. query the legacy flat WC2022
    output/ directory."""
    import sys
    from importlib import import_module

    router = import_module("06_retrieve_context")
    captured = _stub_execute_route(router, monkeypatch)

    monkeypatch.setattr(sys, "argv", ["06_retrieve_context.py", "How many goals did Messi score?"])

    assert router.main() == 0
    assert captured["artifact_paths"] is None


def test_cli_non_wc_competition_passes_matching_artifact_paths(monkeypatch):
    """--competition-id/--season-id for a non-WC dataset must pass
    ArtifactPaths(competition_id, season_id), never falling back to legacy
    WC2022 artifacts."""
    import sys
    from importlib import import_module

    router = import_module("06_retrieve_context")
    captured = _stub_execute_route(router, monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "06_retrieve_context.py", "How many goals did Messi score?",
        "--competition-id", "2", "--season-id", "27",
    ])

    assert router.main() == 0
    assert captured["artifact_paths"] == ArtifactPaths(2, 27)


def test_cli_namespaced_flag_forces_wc2022_artifact_paths(monkeypatch):
    """--namespaced for the WC2022 default must pass ArtifactPaths(43, 106)
    instead of None."""
    import sys
    from importlib import import_module

    router = import_module("06_retrieve_context")
    captured = _stub_execute_route(router, monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "06_retrieve_context.py", "How many goals did Messi score?", "--namespaced",
    ])

    assert router.main() == 0
    assert captured["artifact_paths"] == ArtifactPaths(43, 106)


def test_cli_namespaced_flag_is_a_no_op_for_non_wc_competition(monkeypatch):
    """--namespaced on a non-WC dataset must not change the result -- they
    are namespaced regardless of the flag."""
    import sys
    from importlib import import_module

    router = import_module("06_retrieve_context")
    captured = _stub_execute_route(router, monkeypatch)

    monkeypatch.setattr(sys, "argv", [
        "06_retrieve_context.py", "How many goals did Messi score?",
        "--competition-id", "2", "--season-id", "27", "--namespaced",
    ])

    assert router.main() == 0
    assert captured["artifact_paths"] == ArtifactPaths(2, 27)

# ---------------------------------------------------------------------------
# Competition Portability Batch 6: Chroma collection namespace
# ---------------------------------------------------------------------------

def test_chroma_collection_name_is_deterministic_from_competition_and_season():
    paths = ArtifactPaths(competition_id=2, season_id=27)
    assert paths.chroma_collection_name == "competition_2_season_27_documents"


def test_different_competitions_have_different_chroma_collection_names():
    a = ArtifactPaths(competition_id=2, season_id=27)
    b = ArtifactPaths(competition_id=3, season_id=1)
    assert a.chroma_collection_name != b.chroma_collection_name


def test_different_seasons_have_different_chroma_collection_names():
    a = ArtifactPaths(competition_id=2, season_id=27)
    b = ArtifactPaths(competition_id=2, season_id=28)
    assert a.chroma_collection_name != b.chroma_collection_name


def test_namespaced_wc2022_has_generic_dataset_collection_name():
    paths = ArtifactPaths(competition_id=43, season_id=106)
    assert paths.chroma_collection_name == "competition_43_season_106_documents"
    assert paths.chroma_collection_name != "wc2022_documents"

def test_chroma_store_cli_non_wc_uses_dataset_collection_name(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)
    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=Path("output"))
    _write_chunks(paths.chunks, ["a1"])

    chroma_mod = importlib.import_module("05_create_chroma_store")
    importlib.reload(chroma_mod)

    captured = {}

    def fake_create_vector_store(*, chunks, persist_dir, collection_name=chroma_mod.COLLECTION_NAME):
        captured["persist_dir"] = persist_dir
        captured["collection_name"] = collection_name
        return object()

    monkeypatch.setattr(chroma_mod, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(sys, "argv", [
        "05_create_chroma_store.py",
        "--competition-id", "2",
        "--season-id", "27",
    ])

    assert chroma_mod.main() == 0
    assert captured["persist_dir"] == paths.chroma_dir
    assert captured["collection_name"] == paths.chroma_collection_name


def test_chroma_store_cli_namespaced_wc2022_uses_dataset_collection_name(monkeypatch, tmp_path):
    import sys
    import importlib

    monkeypatch.chdir(tmp_path)
    paths = ArtifactPaths(competition_id=43, season_id=106, output_root=Path("output"))
    _write_chunks(paths.chunks, ["wc1"])

    chroma_mod = importlib.import_module("05_create_chroma_store")
    importlib.reload(chroma_mod)

    captured = {}

    def fake_create_vector_store(*, chunks, persist_dir, collection_name=chroma_mod.COLLECTION_NAME):
        captured["collection_name"] = collection_name
        return object()

    monkeypatch.setattr(chroma_mod, "create_vector_store", fake_create_vector_store)
    monkeypatch.setattr(sys, "argv", [
        "05_create_chroma_store.py",
        "--namespaced",
    ])

    assert chroma_mod.main() == 0
    assert captured["collection_name"] == paths.chroma_collection_name


def test_dense_search_uses_dataset_collection_name_when_artifact_paths_are_given(monkeypatch, tmp_path):
    from importlib import import_module

    router = import_module("06_retrieve_context")
    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)

    seen = {}

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    class FakeClient:
        def __init__(self, path):
            seen["persist_dir"] = Path(path)

        def get_collection(self, name):
            seen["collection_name"] = name
            return FakeCollection()

    class FakeModel:
        def encode(self, values, normalize_embeddings=True):
            return type("Encoded", (), {"tolist": lambda self: [[0.0]]})()

    monkeypatch.setattr("chromadb.PersistentClient", FakeClient)
    monkeypatch.setattr("src.cache.get_embedding_model", lambda *args, **kwargs: FakeModel())

    router.dense_search("test", artifact_paths=paths)

    assert seen["persist_dir"] == paths.chroma_dir
    assert seen["collection_name"] == paths.chroma_collection_name


def test_default_dense_search_keeps_legacy_wc2022_collection_name(monkeypatch):
    from importlib import import_module

    router = import_module("06_retrieve_context")
    seen = {}

    class FakeCollection:
        def query(self, **kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    class FakeClient:
        def __init__(self, path):
            pass

        def get_collection(self, name):
            seen["collection_name"] = name
            return FakeCollection()

    class FakeModel:
        def encode(self, values, normalize_embeddings=True):
            return type("Encoded", (), {"tolist": lambda self: [[0.0]]})()

    monkeypatch.setattr("chromadb.PersistentClient", FakeClient)
    monkeypatch.setattr("src.cache.get_embedding_model", lambda *args, **kwargs: FakeModel())

    router.dense_search("test")

    assert seen["collection_name"] == "wc2022_documents"

def test_retrieve_context_semantic_mode_uses_dataset_collection_name(monkeypatch, tmp_path):
    from importlib import import_module

    router = import_module("06_retrieve_context")
    paths = ArtifactPaths(competition_id=2, season_id=27, output_root=tmp_path)
    seen = {}

    def fake_semantic_search(query, persist_dir=router.CHROMA_DIR,
                             collection_name=router.COLLECTION_NAME,
                             k=5, level_filter=None):
        seen["persist_dir"] = persist_dir
        seen["collection_name"] = collection_name
        return []

    monkeypatch.setattr(router, "semantic_search", fake_semantic_search)

    router.retrieve_context(
        "test",
        mode="semantic",
        artifact_paths=paths,
    )

    assert seen["persist_dir"] == paths.chroma_dir
    assert seen["collection_name"] == paths.chroma_collection_name
