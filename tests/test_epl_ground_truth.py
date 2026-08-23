from collections import Counter

from src.evaluation.ground_truth.epl_2015_16 import (
    EPL_2015_16_GROUND_TRUTH,
    EPL_2015_16_GROUND_TRUTH_METADATA,
    validate_epl_2015_16_ground_truth,
)
from src.evaluation.ground_truth.semantic import (
    index_chunks_by_document_id,
    load_chunks,
)


def test_epl_ground_truth_metadata_identity():
    metadata = EPL_2015_16_GROUND_TRUTH_METADATA

    assert metadata["competition_id"] == 2
    assert metadata["season_id"] == 27
    assert metadata["tournament_name"] == "Premier League"
    assert metadata["season_name"] == "2015/2016"
    assert metadata["expected_case_count"] == 24


def test_epl_ground_truth_has_exactly_24_unique_cases():
    case_ids = [case["id"] for case in EPL_2015_16_GROUND_TRUTH]

    assert len(case_ids) == 24
    assert len(set(case_ids)) == 24


def test_epl_ground_truth_has_four_cases_per_group():
    counts = Counter(
        case["case_group"]
        for case in EPL_2015_16_GROUND_TRUTH
    )

    assert counts == {
        "l1": 4,
        "l2": 4,
        "l3": 4,
        "l4": 4,
        "team": 4,
        "multi": 4,
    }


def test_all_epl_relevant_and_optional_documents_exist():
    chunks = load_chunks(
        EPL_2015_16_GROUND_TRUTH_METADATA["chunks_path"]
    )
    chunks_by_document = index_chunks_by_document_id(chunks)

    referenced_document_ids = {
        document_id
        for case in EPL_2015_16_GROUND_TRUTH
        for field in (
            "relevant_document_ids",
            "optional_relevant_document_ids",
        )
        for document_id in case[field]
    }

    missing = sorted(
        document_id
        for document_id in referenced_document_ids
        if document_id not in chunks_by_document
    )

    assert missing == []


def test_epl_ground_truth_validation_succeeds():
    errors = validate_epl_2015_16_ground_truth(
        EPL_2015_16_GROUND_TRUTH_METADATA,
        EPL_2015_16_GROUND_TRUTH,
        EPL_2015_16_GROUND_TRUTH_METADATA["chunks_path"],
    )

    assert errors == []


def test_epl_ground_truth_excludes_known_name_risk_player_ids():
    risky_player_ids = {
        "3054",
        "3961",
        "4429",
        "13248",
        "16890",
    }

    selected_player_ids = set()

    for case in EPL_2015_16_GROUND_TRUTH:
        for document_id in case["relevant_document_ids"]:
            if document_id.startswith("L4-player-"):
                selected_player_ids.add(
                    document_id.removeprefix("L4-player-")
                )
            elif document_id.startswith("L3-match-") and "-player-" in document_id:
                selected_player_ids.add(
                    document_id.rsplit("-player-", 1)[1]
                )

    assert selected_player_ids.isdisjoint(risky_player_ids)
