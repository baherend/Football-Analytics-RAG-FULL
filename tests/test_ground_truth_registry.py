import pytest

from src.evaluation.ground_truth.registry import (
    GroundTruthNotRegisteredError,
    resolve_ground_truth_bundle,
)
from src.evaluation.ground_truth.epl_2015_16 import (
    EPL_2015_16_GROUND_TRUTH,
    EPL_2015_16_GROUND_TRUTH_METADATA,
    validate_epl_2015_16_ground_truth,
)
from src.evaluation.ground_truth.semantic import (
    SEMANTIC_GROUND_TRUTH,
    SEMANTIC_GROUND_TRUTH_METADATA,
    validate_semantic_ground_truth,
)
from src.evaluation.retrieval_evaluator import GroundTruthBundle


def test_wc2022_resolves_to_existing_semantic_ground_truth():
    bundle = resolve_ground_truth_bundle(competition_id=43, season_id=106)

    assert isinstance(bundle, GroundTruthBundle)
    assert bundle.metadata is SEMANTIC_GROUND_TRUTH_METADATA
    assert bundle.cases is SEMANTIC_GROUND_TRUTH
    assert bundle.validate_fn is validate_semantic_ground_truth


def test_epl_2015_16_resolves_to_registered_ground_truth():
    bundle = resolve_ground_truth_bundle(competition_id=2, season_id=27)

    assert isinstance(bundle, GroundTruthBundle)
    assert bundle.metadata is EPL_2015_16_GROUND_TRUTH_METADATA
    assert bundle.cases is EPL_2015_16_GROUND_TRUTH
    assert bundle.validate_fn is validate_epl_2015_16_ground_truth


def test_unregistered_dataset_does_not_fall_back():
    with pytest.raises(
        GroundTruthNotRegisteredError,
        match=r"competition_id=999.*season_id=999",
    ):
        resolve_ground_truth_bundle(competition_id=999, season_id=999)
