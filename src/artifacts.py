"""
artifacts.py -- Competition/season artifact path namespace.

Centralizes where each competition/season's pipeline artifacts live on
disk (output/competitions/<competition_id>/<season_id>/...) so scripts
stop hardcoding shared "output/..." paths directly. Production code
threads competition_id/season_id through to construct one ArtifactPaths
instead of duplicating path construction across scripts.

ArtifactPaths itself makes NO competition-ID branching decisions -- every
(competition_id, season_id) pair, including WC2022's 43/106, maps
uniformly to its own namespaced directory. The one narrow, documented
legacy-compatibility exception -- existing production entry points
(extract.py, generate_documents.py, 05_create_chroma_store.py, rebuild.py)
defaulting to the pre-existing flat output/ layout for the WC2022 default
when no other competition/season is requested -- is consolidated in
resolve_output_dir() below, rather than duplicated at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """
    Deterministic artifact locations for one competition/season dataset.

    root = output_root / "competitions" / <competition_id> / <season_id>

    IDs (not competition/season names) are used in the filesystem path --
    they are stable integers with no sanitization or collision concerns,
    unlike free-text names.
    """

    competition_id: int
    season_id: int
    output_root: Path = Path("output")

    @property
    def root(self) -> Path:
        return self.output_root / "competitions" / str(self.competition_id) / str(self.season_id)

    @property
    def match_facts(self) -> Path:
        return self.root / "match_facts.json"

    @property
    def documents(self) -> Path:
        return self.root / "documents.json"

    @property
    def processed_documents(self) -> Path:
        return self.root / "processed_documents.json"

    @property
    def chunks(self) -> Path:
        return self.root / "chunks.json"

    @property
    def indices_dir(self) -> Path:
        return self.root / "indices"

    @property
    def bm25_index(self) -> Path:
        return self.indices_dir / "bm25.pkl"

    @property
    def embeddings_dir(self) -> Path:
        return self.root / "embeddings"

    @property
    def embeddings_file(self) -> Path:
        return self.embeddings_dir / "embeddings.npy"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma_db"

    @property
    def chroma_collection_name(self) -> str:
        return f"competition_{self.competition_id}_season_{self.season_id}_documents"


# ---------------------------------------------------------------------------
# The one narrow, documented legacy-compatibility boundary.
# ---------------------------------------------------------------------------

# The repository already ships legacy WC2022 artifacts directly under the
# flat output/ layout (output/match_facts.json, output/chunks.json, ...),
# and legacy-default production read paths (06_retrieve_context.py,
# src.query.resolver's default DATA_PATH, src.cache's default data_path)
# still expect that exact layout. These two IDs identify that one legacy
# dataset -- see resolve_output_dir().
_LEGACY_WC2022_COMPETITION_ID = 43
_LEGACY_WC2022_SEASON_ID = 106


def resolve_runtime_artifact_paths(
    competition_id: int = _LEGACY_WC2022_COMPETITION_ID,
    season_id: int = _LEGACY_WC2022_SEASON_ID,
    output_root: Path = Path("output"),
    legacy_default: bool = True,
) -> ArtifactPaths | None:
    """Resolve one runtime dataset selection to the retrieval contract."""
    is_wc2022 = (
        competition_id,
        season_id,
    ) == (
        _LEGACY_WC2022_COMPETITION_ID,
        _LEGACY_WC2022_SEASON_ID,
    )
    if legacy_default and is_wc2022:
        return None
    return ArtifactPaths(competition_id, season_id, output_root)

def resolve_output_dir(
    competition_id: int,
    season_id: int,
    output_root: Path = Path("output"),
    legacy_default: bool = True,
) -> Path:
    """
    Resolve the output directory a pipeline entry point should write to /
    read from for one competition/season.

    This is the single, narrowly-scoped legacy-compatibility exception in
    the artifact namespace: with `legacy_default=True` (the default), the
    WC2022 default (competition_id=43, season_id=106) resolves to the
    pre-existing flat `output_root` layout, preserving the existing
    extract -> render -> chat flow unchanged for the plain no-argument
    workflow. Any other competition/season resolves to its own namespaced
    ArtifactPaths(...).root and NEVER falls back to the shared legacy
    location -- a non-WC2022 dataset can never silently read or overwrite
    WC2022's artifacts, or vice versa.

    `legacy_default` only ever affects the WC2022 IDs -- it has no effect
    on any other competition/season. Set it to False to explicitly request
    a namespaced WC2022 build (output/competitions/43/106/) instead of the
    legacy flat layout, e.g. to build WC2022 side-by-side with another
    competition without the two colliding.
    """
    is_wc2022 = (competition_id, season_id) == (_LEGACY_WC2022_COMPETITION_ID, _LEGACY_WC2022_SEASON_ID)
    if legacy_default and is_wc2022:
        return output_root
    return ArtifactPaths(competition_id, season_id, output_root).root


def resolve_chroma_collection_name(
    competition_id: int,
    season_id: int,
    legacy_name: str,
    legacy_default: bool = True,
) -> str:
    """
    Resolve the Chroma collection name a pipeline entry point should use,
    mirroring resolve_output_dir()'s legacy-compatibility boundary: with
    `legacy_default=True` (the default), the WC2022 default keeps its
    pre-existing shared `legacy_name`. Any other competition/season -- and
    an explicitly namespaced WC2022 build (`legacy_default=False`) -- gets
    its own ArtifactPaths(...).chroma_collection_name and never shares a
    collection with another dataset.
    """
    is_wc2022 = (competition_id, season_id) == (_LEGACY_WC2022_COMPETITION_ID, _LEGACY_WC2022_SEASON_ID)
    if legacy_default and is_wc2022:
        return legacy_name
    return ArtifactPaths(competition_id, season_id).chroma_collection_name
