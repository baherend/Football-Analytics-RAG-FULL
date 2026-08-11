from tests.retrieval_evaluator import GroundTruthBundle, RetrievalEvaluationError
from tests.semantic_ground_truth import (
    SEMANTIC_GROUND_TRUTH,
    SEMANTIC_GROUND_TRUTH_METADATA,
    validate_semantic_ground_truth,
)


class GroundTruthNotRegisteredError(RetrievalEvaluationError):
    """Raised when no Semantic Ground Truth benchmark is registered for a dataset."""


_REGISTRY = {
    (43, 106): GroundTruthBundle(
        metadata=SEMANTIC_GROUND_TRUTH_METADATA,
        cases=SEMANTIC_GROUND_TRUTH,
        validate_fn=validate_semantic_ground_truth,
    ),
}


def resolve_ground_truth_bundle(
    competition_id: int,
    season_id: int,
) -> GroundTruthBundle:
    key = (competition_id, season_id)

    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise GroundTruthNotRegisteredError(
            "No Semantic Ground Truth benchmark is registered for "
            f"competition_id={competition_id}, season_id={season_id}."
        ) from exc
