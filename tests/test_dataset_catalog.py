"""
Competition Portability -- Batch 9: Dataset Catalog / Friendly Runtime Identity.

Proves that available datasets (legacy flat WC2022 and any namespaced
competition/season) can be *discovered* from real artifacts on disk --
never from a hardcoded competition registry -- and exposed with a
human-readable label that degrades honestly when no real metadata name is
available. Also proves the catalog's selected entry reaches the existing
`ArtifactPaths | None` runtime contract that 07_prompting.answer_question
and retrieval already depend on (Batch 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.artifacts import ArtifactPaths
from src.dataset_catalog import DatasetEntry, discover_datasets


def _write_match_facts(path: Path, competition_id: int, season_id: int,
                        competition_name: str | None = None,
                        season_name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"competition_id": competition_id, "season_id": season_id}
    if competition_name is not None:
        metadata["competition_name"] = competition_name
    if season_name is not None:
        metadata["season_name"] = season_name
    path.write_text(json.dumps({
        "metadata": metadata,
        "player_match_facts": [], "match_facts": [], "team_match_facts": [],
    }), encoding="utf-8")


def _write_chunks(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([]), encoding="utf-8")



def _write_runtime_retrieval_artifacts(root: Path) -> None:
    _write_chunks(root / "chunks.json")
    (root / "indices").mkdir(parents=True, exist_ok=True)
    (root / "indices" / "bm25.pkl").write_bytes(b"test-bm25")
    (root / "chroma_db").mkdir(parents=True, exist_ok=True)
    (root / "chroma_db" / "chroma.sqlite3").write_bytes(b"test-chroma")


# ---------------------------------------------------------------------------
# 1. Namespaced discovery
# ---------------------------------------------------------------------------


def test_discovers_namespaced_dataset_directory(tmp_path):
    (tmp_path / "competitions" / "2" / "27").mkdir(parents=True)

    entries = discover_datasets(output_root=tmp_path)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.competition_id == 2
    assert entry.season_id == 27
    assert entry.artifact_paths == ArtifactPaths(2, 27, output_root=tmp_path)


# ---------------------------------------------------------------------------
# 2. Multiple datasets discovered independently and deterministically
# ---------------------------------------------------------------------------


def test_discovers_multiple_datasets_deterministically(tmp_path):
    for competition_id, season_id in [(11, 90), (2, 27), (43, 106)]:
        (tmp_path / "competitions" / str(competition_id) / str(season_id)).mkdir(parents=True)

    entries_first = discover_datasets(output_root=tmp_path)
    entries_second = discover_datasets(output_root=tmp_path)

    ids_first = [(e.competition_id, e.season_id) for e in entries_first]
    ids_second = [(e.competition_id, e.season_id) for e in entries_second]

    assert ids_first == ids_second  # deterministic across repeated calls
    assert ids_first == sorted(ids_first)  # sorted output
    assert set(ids_first) == {(11, 90), (2, 27), (43, 106)}


# ---------------------------------------------------------------------------
# 3. Invalid directory names must not crash discovery
# ---------------------------------------------------------------------------


def test_invalid_directory_names_are_skipped(tmp_path):
    (tmp_path / "competitions" / "not-a-number" / "27").mkdir(parents=True)
    (tmp_path / "competitions" / "2" / "also-not-a-number").mkdir(parents=True)
    (tmp_path / "competitions" / "2" / "27").mkdir(parents=True)
    # A stray file sitting where a directory is expected must not crash.
    (tmp_path / "competitions" / "2" / "stray-file.txt").write_text("x", encoding="utf-8")

    entries = discover_datasets(output_root=tmp_path)

    ids = [(e.competition_id, e.season_id) for e in entries]
    assert ids == [(2, 27)]


# ---------------------------------------------------------------------------
# 4. Incomplete namespaces: discovered, but not falsely reported as ready
# ---------------------------------------------------------------------------


def test_incomplete_namespace_is_discovered_but_not_ready(tmp_path):
    paths = ArtifactPaths(2, 27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, 2, 27, "Test League", "2025/2026")
    # chunks.json intentionally absent -- semantic retrieval has nothing to read.

    entries = discover_datasets(output_root=tmp_path)

    assert len(entries) == 1
    assert entries[0].is_ready is False


def test_complete_namespace_is_ready(tmp_path):
    paths = ArtifactPaths(2, 27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, 2, 27, "Test League", "2025/2026")
    _write_runtime_retrieval_artifacts(paths.root)

    entries = discover_datasets(output_root=tmp_path)

    assert len(entries) == 1
    assert entries[0].is_ready is True


# ---------------------------------------------------------------------------
# 5. Friendly metadata is used when available
# ---------------------------------------------------------------------------


def test_friendly_metadata_used_when_present(tmp_path):
    paths = ArtifactPaths(2, 27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, 2, 27, "Premier League", "2025/26")
    _write_chunks(paths.chunks)

    entries = discover_datasets(output_root=tmp_path)

    assert entries[0].label == "Premier League — 2025/26"


# ---------------------------------------------------------------------------
# 6. Honest fallback label when names are unavailable -- no invented identity
# ---------------------------------------------------------------------------


def test_honest_fallback_label_when_names_unavailable(tmp_path):
    (tmp_path / "competitions" / "11" / "90").mkdir(parents=True)

    entries = discover_datasets(output_root=tmp_path)

    assert entries[0].label == "Competition 11 — Season 90"
    assert entries[0].identity is None


def test_honest_fallback_label_when_match_facts_has_ids_only(tmp_path):
    """A non-WC2022 dataset whose match_facts.json exists but has no
    competition_name/season_name must not invent a name."""
    paths = ArtifactPaths(11, 90, output_root=tmp_path)
    _write_match_facts(paths.match_facts, 11, 90)  # ids only, no names
    _write_runtime_retrieval_artifacts(paths.root)

    entries = discover_datasets(output_root=tmp_path)

    assert entries[0].label == "Competition 11 — Season 90"
    assert entries[0].is_ready is True


# ---------------------------------------------------------------------------
# 7. Legacy WC2022 remains discoverable/selectable where its artifacts exist
# ---------------------------------------------------------------------------


def test_legacy_wc2022_discovered_when_flat_artifacts_present(tmp_path):
    _write_match_facts(tmp_path / "match_facts.json", 43, 106)  # real shipped shape: ids only
    _write_runtime_retrieval_artifacts(tmp_path)

    entries = discover_datasets(output_root=tmp_path)

    legacy = next(e for e in entries if e.artifact_paths is None)
    assert legacy.competition_id == 43
    assert legacy.season_id == 106
    assert legacy.is_ready is True
    assert legacy.label == "FIFA World Cup — 2022 (legacy)"


def test_legacy_wc2022_not_discovered_when_absent(tmp_path):
    entries = discover_datasets(output_root=tmp_path)
    assert entries == []


# ---------------------------------------------------------------------------
# 8. Legacy flat WC2022 vs explicitly namespaced WC2022 remain distinct
# ---------------------------------------------------------------------------


def test_legacy_and_namespaced_wc2022_are_distinct_entries(tmp_path):
    _write_match_facts(tmp_path / "match_facts.json", 43, 106)
    _write_chunks(tmp_path / "chunks.json")

    namespaced_paths = ArtifactPaths(43, 106, output_root=tmp_path)
    _write_match_facts(namespaced_paths.match_facts, 43, 106, "FIFA World Cup", "2022")
    _write_runtime_retrieval_artifacts(namespaced_paths.root)

    entries = discover_datasets(output_root=tmp_path)

    assert len(entries) == 2
    legacy = next(e for e in entries if e.artifact_paths is None)
    namespaced = next(e for e in entries if e.artifact_paths is not None)

    assert legacy.artifact_paths is None
    assert namespaced.artifact_paths == ArtifactPaths(43, 106, output_root=tmp_path)
    assert legacy != namespaced


# ---------------------------------------------------------------------------
# 9. ArtifactPaths integration -- catalog never duplicates path rules
# ---------------------------------------------------------------------------


def test_namespaced_entry_artifact_paths_matches_artifact_paths_directly(tmp_path):
    (tmp_path / "competitions" / "5" / "3").mkdir(parents=True)

    entries = discover_datasets(output_root=tmp_path)

    assert entries[0].artifact_paths == ArtifactPaths(5, 3, output_root=tmp_path)
    assert entries[0].artifact_paths.match_facts == ArtifactPaths(5, 3, output_root=tmp_path).match_facts


# ---------------------------------------------------------------------------
# 13. Empty catalog: no discoverable datasets, no crash, no fabrication
# ---------------------------------------------------------------------------


def test_empty_catalog_returns_empty_list(tmp_path):
    assert discover_datasets(output_root=tmp_path) == []


def test_empty_catalog_when_output_root_does_not_exist(tmp_path):
    assert discover_datasets(output_root=tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# 10. Runtime selection: the selected catalog entry reaches
#     07_prompting.answer_question / retrieval via the SAME artifact_paths
#     value discover_datasets() produced -- not a re-derived lookalike.
# ---------------------------------------------------------------------------


def test_selected_catalog_entry_reaches_answer_question(monkeypatch, tmp_path):
    from importlib import import_module
    from types import SimpleNamespace

    paths = ArtifactPaths(2, 27, output_root=tmp_path)
    _write_match_facts(paths.match_facts, 2, 27, "Test League", "2025/2026")
    _write_runtime_retrieval_artifacts(paths.root)

    entries = discover_datasets(output_root=tmp_path)
    assert len(entries) == 1
    selected = entries[0]
    assert selected.is_ready

    prompting = import_module("07_prompting")
    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(context="test context", semantic_chunks=[])

    fake_router = SimpleNamespace(route_and_execute=fake_route_and_execute)
    real_import_module = prompting.import_module

    def fake_import_module(name):
        if name == "06_retrieve_context":
            return fake_router
        return real_import_module(name)

    monkeypatch.setattr(prompting, "import_module", fake_import_module)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    prompting.answer_question(
        "test question",
        api_key="test-key",
        artifact_paths=selected.artifact_paths,
    )

    # The exact object discovery produced is what reached retrieval --
    # not a re-derived ArtifactPaths that merely compares equal.
    assert captured["artifact_paths"] is selected.artifact_paths


def test_selected_legacy_entry_reaches_answer_question_as_none(monkeypatch, tmp_path):
    from importlib import import_module
    from types import SimpleNamespace

    _write_match_facts(tmp_path / "match_facts.json", 43, 106)
    _write_chunks(tmp_path / "chunks.json")

    entries = discover_datasets(output_root=tmp_path)
    selected = next(e for e in entries if e.artifact_paths is None)

    prompting = import_module("07_prompting")
    captured = {}

    def fake_route_and_execute(question, semantic_k=3, artifact_paths=None):
        captured["artifact_paths"] = artifact_paths
        return SimpleNamespace(context="test context", semantic_chunks=[])

    fake_router = SimpleNamespace(route_and_execute=fake_route_and_execute)
    real_import_module = prompting.import_module

    def fake_import_module(name):
        if name == "06_retrieve_context":
            return fake_router
        return real_import_module(name)

    monkeypatch.setattr(prompting, "import_module", fake_import_module)
    monkeypatch.setattr(prompting, "ask_groq", lambda *args, **kwargs: "test answer")

    prompting.answer_question(
        "test question",
        api_key="test-key",
        artifact_paths=selected.artifact_paths,
    )

    assert captured["artifact_paths"] is None


# ---------------------------------------------------------------------------
# 12. CLI backward compatibility: --list-datasets is additive, existing
#     --competition-id/--season-id flags are untouched (already covered by
#     tests/test_runtime_dataset_selection.py).
# ---------------------------------------------------------------------------


def test_chat_list_datasets_reports_discovered_datasets(monkeypatch, tmp_path, capsys):
    import sys
    from importlib import import_module

    # Import chat.py (and its module-level pipeline loading, which resolves
    # "06_retrieve_context.py" relative to CWD) BEFORE sandboxing CWD to
    # tmp_path -- module-level code only runs once per process, so this
    # must not depend on test execution order against other test files.
    chat = import_module("chat")
    monkeypatch.chdir(tmp_path)
    _write_match_facts(tmp_path / "output" / "match_facts.json", 43, 106)
    _write_runtime_retrieval_artifacts(tmp_path / "output")

    monkeypatch.setattr(sys, "argv", ["chat.py", "--list-datasets"])

    assert chat.main() == 0
    out = capsys.readouterr().out
    assert "competition_id=43" in out
    assert "season_id=106" in out
    assert "ready" in out


def test_chat_list_datasets_honest_when_empty(monkeypatch, tmp_path, capsys):
    import sys
    from importlib import import_module

    chat = import_module("chat")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["chat.py", "--list-datasets"])

    assert chat.main() == 0
    out = capsys.readouterr().out
    assert "No datasets discovered" in out


# ---------------------------------------------------------------------------
# 11. Streamlit dataset switch: selecting another catalog entry must still
#     reset messages/memory (Batch 7/8 behavior) and never fabricate a
#     dataset when discovery is empty. Streamlit itself can't be driven
#     headlessly here, so -- following the established pattern in
#     tests/test_runtime_dataset_selection.py and
#     tests/test_streamlit_conversation_memory.py -- these inspect the
#     actual source, strengthened with an AST check that `dataset_key` is
#     assigned from the catalog-selected entry's artifact_paths (not a
#     hand-rolled tuple), so the reset block provably keys off the same
#     ArtifactPaths | None value passed to answer_question.
# ---------------------------------------------------------------------------


def _streamlit_source() -> str:
    return Path("streamlit_app.py").read_text(encoding="utf-8")


def test_streamlit_uses_dataset_catalog_for_selection():
    source = _streamlit_source()

    assert "discover_datasets" in source
    assert "DatasetEntry" not in source  # imports the function, not the class -- no need to name it


def test_streamlit_fails_honestly_when_no_datasets_discovered():
    source = _streamlit_source()

    assert "st.stop()" in source
    assert "No runtime-ready datasets" in source


def test_streamlit_dataset_key_is_derived_from_selected_artifact_paths():
    """
    dataset_key (what the reset block in streamlit_app.py compares against
    session state) must be assigned directly from the selected catalog
    entry's artifact_paths -- the same value threaded to answer_question --
    not a separately-typed tuple that could drift out of sync with it.
    """
    import ast

    tree = ast.parse(_streamlit_source())
    assigns = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "dataset_key" for t in node.targets)
    ]

    assert len(assigns) == 1
    value = assigns[0].value
    assert isinstance(value, ast.Name) and value.id == "artifact_paths"


def test_match_facts_and_chunks_alone_are_not_runtime_ready(tmp_path):
    root = tmp_path / "output"
    dataset = root / "competitions" / "2" / "27"
    dataset.mkdir(parents=True)
    (dataset / "match_facts.json").write_text('{"metadata": {}}', encoding="utf-8")
    (dataset / "chunks.json").write_text("[]", encoding="utf-8")

    entries = discover_datasets(root)

    assert len(entries) == 1
    assert entries[0].is_ready is False
