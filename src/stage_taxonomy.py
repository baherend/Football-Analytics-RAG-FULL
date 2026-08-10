"""
stage_taxonomy.py — Competition Stage Taxonomy

Explicit stage vocabulary, knockout classification, and group-stage
classification for a competition.

Knockout and group-stage status are both properties of the taxonomy, never
inferred from each other and never inferred by comparing a stage name
against a literal (e.g. `stage != "Group Stage"` or `not is_knockout`). A
league competition has no knockout phase AND no group phase — it supplies
empty `knockout_stages` and `group_stages` sets, and every match is
correctly classified as neither, regardless of what its stage is named
("Regular Season", "Matchday 1", ...).

WC2022 (StatsBomb competition_id=43, season_id=106) ships as the default
taxonomy so existing behavior is unchanged wherever no taxonomy is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class StageTaxonomy:
    """
    stages:           every valid stage name for this competition. Used to
                       validate structured-query stage filters.
    knockout_stages:  the subset of `stages` that are knockout rounds.
                       Empty for a competition with no knockout phase.
    group_stages:     the subset of `stages` that are group-stage rounds.
                       Empty for a competition with no group phase. A stage
                       may not appear in both `knockout_stages` and
                       `group_stages`.
    stage_order:      optional stage progression (earliest -> latest), used
                       only for "furthest stage reached" reporting. Empty
                       means no fixed order is defined.
    """

    stages: frozenset[str]
    knockout_stages: frozenset[str] = field(default_factory=frozenset)
    group_stages: frozenset[str] = field(default_factory=frozenset)
    stage_order: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        overlap = self.knockout_stages & self.group_stages
        if overlap:
            raise ValueError(
                f"stage(s) {sorted(overlap)} cannot be both knockout and "
                "group stages"
            )

    def is_knockout(self, stage: str) -> bool:
        return stage in self.knockout_stages

    def is_group_stage(self, stage: str) -> bool:
        return stage in self.group_stages

    def is_valid_stage(self, stage: str) -> bool:
        return stage in self.stages

    def to_dict(self) -> dict:
        """Serialize the complete taxonomy semantics to JSON-safe values."""
        return {
            "stages": sorted(self.stages),
            "knockout_stages": sorted(self.knockout_stages),
            "group_stages": sorted(self.group_stages),
            "stage_order": list(self.stage_order),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageTaxonomy":
        """Reconstruct a taxonomy from persisted metadata."""
        return cls(
            stages=frozenset(data.get("stages", ())),
            knockout_stages=frozenset(data.get("knockout_stages", ())),
            group_stages=frozenset(data.get("group_stages", ())),
            stage_order=tuple(data.get("stage_order", ())),
        )

    @classmethod
    def discover(
        cls,
        stages: Iterable[str],
        knockout_stages: Iterable[str] = (),
        group_stages: Iterable[str] = (),
        stage_order: tuple[str, ...] = (),
    ) -> "StageTaxonomy":
        """
        Build a taxonomy from stage names observed in competition data (e.g.
        the distinct `match["competition_stage"]["name"]` values).

        Knockout and group-stage semantics are never guessed from the
        discovered names — callers must state `knockout_stages` and
        `group_stages` explicitly. Both default to empty, which is correct
        for a league with neither phase.
        """
        return cls(
            stages=frozenset(stages),
            knockout_stages=frozenset(knockout_stages),
            group_stages=frozenset(group_stages),
            stage_order=tuple(stage_order),
        )


WC2022_STAGE_TAXONOMY = StageTaxonomy(
    stages=frozenset({
        "Group Stage", "Round of 16", "Quarter-finals",
        "Semi-finals", "3rd Place Final", "Final",
    }),
    knockout_stages=frozenset({
        "Round of 16", "Quarter-finals", "Semi-finals",
        "3rd Place Final", "Final",
    }),
    group_stages=frozenset({"Group Stage"}),
    stage_order=(
        "Group Stage", "Round of 16", "Quarter-finals",
        "Semi-finals", "3rd Place Final", "Final",
    ),
)
