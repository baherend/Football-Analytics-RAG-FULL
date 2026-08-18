import pytest

from src.evaluation.ground_truth.registry import (
    GroundTruthNotRegisteredError,
    resolve_ground_truth_bundle,
)
from src.evaluation.retrieval_evaluator import GroundTruthBundle
from src.evaluation.ground_truth.semantic import (
    SEMANTIC_GROUND_TRUTH,
    SEMANTIC_GROUND_TRUTH_METADATA,
    validate_semantic_ground_truth,
)


def test_wc2022_resolves_to_existing_semantic_ground_truth():
    bundle = resolve_ground_truth_bundle(competition_id=43, season_id=106)

    assert isinstance(bundle, GroundTruthBundle)
    assert bundle.metadata is SEMANTIC_GROUND_TRUTH_METADATA
    assert bundle.cases is SEMANTIC_GROUND_TRUTH
    assert bundle.validate_fn is validate_semantic_ground_truth


def test_unregistered_dataset_does_not_fall_back_to_wc2022():
    with pytest.raises(
        GroundTruthNotRegisteredError,
        match=r"competition_id=2.*season_id=27",
    ):
        resolve_ground_truth_bundle(competition_id=2, season_id=27)

def test_evaluator_uses_registry_for_selected_dataset():
    from src.artifacts import resolve_runtime_artifact_paths
    from src.evaluation.retrieval_evaluator import run_retrieval_baseline

    artifact_paths = resolve_runtime_artifact_paths(
        competition_id=2,
        season_id=27,
        legacy_default=False,
    )

    with pytest.raises(
        GroundTruthNotRegisteredError,
        match=r"competition_id=2.*season_id=27",
    ):
        run_retrieval_baseline(artifact_paths=artifact_paths)
