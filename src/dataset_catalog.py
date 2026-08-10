"""
dataset_catalog.py -- Local dataset discovery and friendly runtime identity.

Answers exactly one question: "which football datasets are actually
available to this runtime, and what should a human call them?" It sits
between the artifact filesystem (src/artifacts.py) and runtime entry points
(streamlit_app.py, chat.py) so neither has to scan directories or
reconstruct paths by hand.

Discovery has two sources, both derived from real artifacts on disk --
never a hardcoded competition registry:

- The legacy flat WC2022 layout directly under `output_root`
  (output/match_facts.json, ...), when present. Resolves to
  artifact_paths=None -- the exact value src.artifacts and
  src.conversation_memory already treat as "the legacy dataset".
- Namespaced competition/season directories under
  output_root/competitions/<competition_id>/<season_id>/, each resolving
  to its own ArtifactPaths(competition_id, season_id, output_root).

Friendly names come from match_facts.json's own "metadata" block
(competition_name/season_name), the same field src.rendering.render already
reads to pick a DatasetIdentity for rendering. When that block has no names
-- including the real, already-shipped output/match_facts.json, which
predates this field and has IDs only -- the one narrow, already-established
legacy WC2022 boundary (WC2022_DATASET_IDENTITY, competition_id=43,
season_id=106) is reused, exactly as src.rendering.render and
src.artifacts already do. Any other dataset with unknown names degrades
honestly to "Competition <id> — Season <id>" rather than inventing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.artifacts import ArtifactPaths
from src.extraction.match_facts import DatasetIdentity, WC2022_DATASET_IDENTITY

_LEGACY_COMPETITION_ID = WC2022_DATASET_IDENTITY.competition_id
_LEGACY_SEASON_ID = WC2022_DATASET_IDENTITY.season_id


@dataclass(frozen=True)
class DatasetEntry:
    """
    One runtime-selectable dataset: identity + artifact location + readiness.

    `artifact_paths` is the exact `ArtifactPaths | None` value the existing
    runtime contract (src.artifacts.resolve_runtime_artifact_paths,
    07_prompting.answer_question, src.conversation_memory.ConversationMemory)
    already keys on -- selecting this entry and passing `artifact_paths`
    straight through requires no translation step.
    """

    competition_id: int
    season_id: int
    artifact_paths: ArtifactPaths | None
    identity: DatasetIdentity | None
    is_ready: bool

    @property
    def is_legacy(self) -> bool:
        return self.artifact_paths is None

    @property
    def label(self) -> str:
        if self.identity is not None:
            base = f"{self.identity.competition_name} — {self.identity.season_name}"
        else:
            base = f"Competition {self.competition_id} — Season {self.season_id}"
        return f"{base} (legacy)" if self.is_legacy else base


def _parse_id(name: str) -> int | None:
    if not name.isdigit():
        return None
    return int(name)


def _read_identity(match_facts_path: Path, competition_id: int, season_id: int) -> DatasetIdentity | None:
    """
    Best-effort DatasetIdentity from match_facts.json's metadata block.

    Never raises: an unreadable/missing/malformed file is treated the same
    as "no names available" rather than failing discovery for the whole
    catalog over one bad dataset.
    """
    metadata: dict = {}
    if match_facts_path.exists():
        try:
            with open(match_facts_path, encoding="utf-8") as handle:
                metadata = json.load(handle).get("metadata") or {}
        except (OSError, json.JSONDecodeError):
            metadata = {}

    competition_name = metadata.get("competition_name")
    season_name = metadata.get("season_name")
    if competition_name and season_name:
        return DatasetIdentity(
            competition_id=competition_id,
            competition_name=competition_name,
            season_id=season_id,
            season_name=season_name,
        )

    if (competition_id, season_id) == (_LEGACY_COMPETITION_ID, _LEGACY_SEASON_ID):
        return WC2022_DATASET_IDENTITY

    return None


def _is_ready(
    match_facts_path: Path,
    chunks_path: Path,
    bm25_path: Path,
    chroma_dir: Path,
) -> bool:
    """
    Filesystem-level runtime readiness for the existing query pipeline.

    Structured queries require match_facts.json. Hybrid retrieval requires
    chunks.json plus the BM25 index, while dense retrieval opens the
    dataset's persisted Chroma store. Discovery stays lightweight: it checks
    the persisted artifacts exist, but does not open Chroma or load models.
    """
    return (
        match_facts_path.is_file()
        and chunks_path.is_file()
        and bm25_path.is_file()
        and (chroma_dir / "chroma.sqlite3").is_file()
    )


def _build_entry(
    competition_id: int,
    season_id: int,
    artifact_paths: ArtifactPaths | None,
    match_facts_path: Path,
    chunks_path: Path,
    bm25_path: Path,
    chroma_dir: Path,
) -> DatasetEntry:
    return DatasetEntry(
        competition_id=competition_id,
        season_id=season_id,
        artifact_paths=artifact_paths,
        identity=_read_identity(match_facts_path, competition_id, season_id),
        is_ready=_is_ready(match_facts_path, chunks_path, bm25_path, chroma_dir),
    )


def discover_datasets(output_root: Path = Path("output")) -> list[DatasetEntry]:
    """
    Discover runtime-selectable datasets under `output_root`.

    Returns the legacy flat WC2022 entry first (if its artifacts are
    present), followed by namespaced competition/season entries sorted
    deterministically by (competition_id, season_id). Directory names that
    are not plain non-negative integers are silently skipped -- discovery
    never crashes on unrelated files/directories under output/competitions/.
    """
    entries: list[DatasetEntry] = []

    legacy_match_facts = output_root / "match_facts.json"
    if legacy_match_facts.exists():
        entries.append(_build_entry(
            competition_id=_LEGACY_COMPETITION_ID,
            season_id=_LEGACY_SEASON_ID,
            artifact_paths=None,
            match_facts_path=legacy_match_facts,
            chunks_path=output_root / "chunks.json",
            bm25_path=output_root / "indices" / "bm25.pkl",
            chroma_dir=output_root / "chroma_db",
        ))

    competitions_dir = output_root / "competitions"
    namespaced: list[DatasetEntry] = []
    if competitions_dir.is_dir():
        for competition_dir in competitions_dir.iterdir():
            if not competition_dir.is_dir():
                continue
            competition_id = _parse_id(competition_dir.name)
            if competition_id is None:
                continue
            for season_dir in competition_dir.iterdir():
                if not season_dir.is_dir():
                    continue
                season_id = _parse_id(season_dir.name)
                if season_id is None:
                    continue
                paths = ArtifactPaths(competition_id, season_id, output_root=output_root)
                namespaced.append(_build_entry(
                    competition_id=competition_id,
                    season_id=season_id,
                    artifact_paths=paths,
                    match_facts_path=paths.match_facts,
                    chunks_path=paths.chunks,
                    bm25_path=paths.bm25_index,
                    chroma_dir=paths.chroma_dir,
                ))

    namespaced.sort(key=lambda entry: (entry.competition_id, entry.season_id))
    entries.extend(namespaced)
    return entries
