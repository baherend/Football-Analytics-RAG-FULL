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

from src.embedding_config import DEFAULT_EMBEDDING_MODEL_ID


@dataclass(frozen=True)
class ArtifactPaths:
    """
    Deterministic artifact locations for one competition/season/embedding-
    model dataset.

    root = output_root / "competitions" / <competition_id> / <season_id>

    IDs (not competition/season names) are used in the filesystem path --
    they are stable integers with no sanitization or collision concerns,
    unlike free-text names.

    `embedding_model_id` ties dense-index identity to embedding-model
    identity (see src.embedding_config) via chroma_collection_name below,
    so a dense index built with one model can never be silently queried
    with another. It defaults to the project-wide default model
    (DEFAULT_EMBEDDING_MODEL_ID) -- every existing call site that doesn't
    pass this argument keeps producing the exact same collection name as
    before this field was added.
    """

    competition_id: int
    season_id: int
    output_root: Path = Path("output")
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID

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
        base = f"competition_{self.competition_id}_season_{self.season_id}_documents"
        if self.embedding_model_id == DEFAULT_EMBEDDING_MODEL_ID:
            return base
        return f"competition_{self.competition_id}_season_{self.season_id}_{self.embedding_model_id}_documents"


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
    embedding_model_id: str | None = None,
) -> ArtifactPaths | None:
    """
    Resolve one runtime dataset selection to the retrieval contract.

    `embedding_model_id=None` resolves to the project default (MiniLM),
    matching every existing zero-argument caller exactly. Selecting a
    non-default embedding model for the WC2022 default forces namespacing
    (returns ArtifactPaths(43, 106, ...) instead of None) for the same
    reason `legacy_default=False` does: a non-default index must never be
    silently built into or read from the shared legacy flat directory.
    """
    is_wc2022 = (
        competition_id,
        season_id,
    ) == (
        _LEGACY_WC2022_COMPETITION_ID,
        _LEGACY_WC2022_SEASON_ID,
    )
    resolved_model_id = embedding_model_id if embedding_model_id is not None else DEFAULT_EMBEDDING_MODEL_ID
    if legacy_default and is_wc2022 and resolved_model_id == DEFAULT_EMBEDDING_MODEL_ID:
        return None
    return ArtifactPaths(competition_id, season_id, output_root, embedding_model_id=resolved_model_id)

def resolve_output_dir(
    competition_id: int,
    season_id: int,
    output_root: Path = Path("output"),
    legacy_default: bool = True,
    embedding_model_id: str | None = None,
) -> Path:
    """
    Resolve the output directory a pipeline entry point should write to /
    read from for one competition/season/embedding-model selection.

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

    `embedding_model_id` extends this same boundary: selecting a
    non-default embedding model for the WC2022 default also opts out of
    the flat layout (exactly as `legacy_default=False` does), so building
    an alternate-model index for WC2022 can never write into the same
    directory as the production MiniLM Chroma store.
    """
    is_wc2022 = (competition_id, season_id) == (_LEGACY_WC2022_COMPETITION_ID, _LEGACY_WC2022_SEASON_ID)
    resolved_model_id = embedding_model_id if embedding_model_id is not None else DEFAULT_EMBEDDING_MODEL_ID
    if legacy_default and is_wc2022 and resolved_model_id == DEFAULT_EMBEDDING_MODEL_ID:
        return output_root
    return ArtifactPaths(competition_id, season_id, output_root, embedding_model_id=resolved_model_id).root


def resolve_chroma_collection_name(
    competition_id: int,
    season_id: int,
    legacy_name: str,
    legacy_default: bool = True,
    embedding_model_id: str | None = None,
) -> str:
    """
    Resolve the Chroma collection name a pipeline entry point should use,
    mirroring resolve_output_dir()'s legacy-compatibility boundary: with
    `legacy_default=True` (the default) and the default embedding model,
    the WC2022 default keeps its pre-existing shared `legacy_name`. Any
    other competition/season, an explicitly namespaced WC2022 build
    (`legacy_default=False`), or a non-default embedding model -- gets its
    own ArtifactPaths(...).chroma_collection_name and never shares a
    collection with another dataset or model.
    """
    is_wc2022 = (competition_id, season_id) == (_LEGACY_WC2022_COMPETITION_ID, _LEGACY_WC2022_SEASON_ID)
    resolved_model_id = embedding_model_id if embedding_model_id is not None else DEFAULT_EMBEDDING_MODEL_ID
    if legacy_default and is_wc2022 and resolved_model_id == DEFAULT_EMBEDDING_MODEL_ID:
        return legacy_name
    return ArtifactPaths(competition_id, season_id, embedding_model_id=resolved_model_id).chroma_collection_name
