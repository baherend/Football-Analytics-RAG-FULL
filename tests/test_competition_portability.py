import json
from pathlib import Path

import pytest

import src.extraction.match_facts as match_facts
from src.extraction.match_facts import DatasetIdentity, WC2022_DATASET_IDENTITY


def test_extract_all_uses_requested_competition_and_season(monkeypatch):
    requested_paths = []

    def fake_load_json(path):
        requested_paths.append(Path(path))
        return []

    monkeypatch.setattr(match_facts, "load_json", fake_load_json)

    match_facts.extract_all(
        data_root=Path("dataset"),
        competition_id=2,
        season_id=27,
        verbose=False,
    )

    assert requested_paths == [Path("dataset") / "matches" / "2" / "27.json"]


def test_extract_all_defaults_to_wc2022(monkeypatch):
    requested_paths = []

    def fake_load_json(path):
        requested_paths.append(Path(path))
        return []

    monkeypatch.setattr(match_facts, "load_json", fake_load_json)

    match_facts.extract_all(
        data_root=Path("dataset"),
        verbose=False,
    )

    assert requested_paths == [Path("dataset") / "matches" / "43" / "106.json"]


def test_persist_writes_requested_competition_and_season(tmp_path):
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }
    identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )

    output_path = match_facts.persist(
        result,
        output_dir=tmp_path,
        competition_id=2,
        season_id=27,
        dataset_identity=identity,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["metadata"]["competition_id"] == 2
    assert data["metadata"]["season_id"] == 27


def test_persist_defaults_to_wc2022(tmp_path):
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }

    output_path = match_facts.persist(result, output_dir=tmp_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["metadata"]["competition_id"] == 43
    assert data["metadata"]["season_id"] == 106
    # No dataset_identity was passed — the WC2022 default (43/106) is the
    # one case where persist() safely auto-fills names on its own.
    assert data["metadata"]["competition_name"] == "FIFA World Cup"
    assert data["metadata"]["season_name"] == "2022"


def test_extract_cli_threads_competition_and_season(monkeypatch, tmp_path):
    import sys
    import src.extraction.extract as extract_cli

    calls = {}

    def fake_extract_all(data_root, verbose=True, competition_id=None, season_id=None):
        calls["extract"] = {
            "data_root": data_root,
            "verbose": verbose,
            "competition_id": competition_id,
            "season_id": season_id,
        }
        return {
            "player_match_facts": [],
            "match_facts": [],
            "team_match_facts": [],
            "diagnostics": {
                "matches_processed": 0,
                "total_player_facts": 0,
                "total_team_facts": 0,
                "card_count_mismatches": [],
            },
        }

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None,
                      dataset_identity=None):
        calls["persist"] = {
            "competition_id": competition_id,
            "season_id": season_id,
            "dataset_identity": dataset_identity,
        }
        return tmp_path / "match_facts.json"

    monkeypatch.setattr(extract_cli, "extract_all", fake_extract_all)
    monkeypatch.setattr(extract_cli, "persist", fake_persist)
    monkeypatch.setattr(
        sys,
        "argv",
        ["extract.py", "--competition-id", "2", "--season-id", "27", "--quiet"],
    )

    assert extract_cli.main() == 0
    assert calls["extract"]["competition_id"] == 2
    assert calls["extract"]["season_id"] == 27
    assert calls["persist"]["competition_id"] == 2
    assert calls["persist"]["season_id"] == 27


def test_extract_cli_defaults_to_wc2022_when_args_omitted(monkeypatch, tmp_path):
    import sys
    import src.extraction.extract as extract_cli

    calls = {}

    def fake_extract_all(data_root, verbose=True, competition_id=None, season_id=None):
        calls["extract"] = {
            "competition_id": competition_id,
            "season_id": season_id,
        }
        return {
            "player_match_facts": [],
            "match_facts": [],
            "team_match_facts": [],
            "diagnostics": {
                "matches_processed": 0,
                "total_player_facts": 0,
                "total_team_facts": 0,
                "card_count_mismatches": [],
            },
        }

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None,
                      dataset_identity=None):
        calls["persist"] = {
            "competition_id": competition_id,
            "season_id": season_id,
            "dataset_identity": dataset_identity,
        }
        return tmp_path / "match_facts.json"

    monkeypatch.setattr(extract_cli, "extract_all", fake_extract_all)
    monkeypatch.setattr(extract_cli, "persist", fake_persist)
    monkeypatch.setattr(sys, "argv", ["extract.py", "--quiet"])

    assert extract_cli.main() == 0
    assert calls["extract"]["competition_id"] == 43
    assert calls["extract"]["season_id"] == 106
    assert calls["persist"]["competition_id"] == 43
    assert calls["persist"]["season_id"] == 106


# ---------------------------------------------------------------------------
# Dataset identity (competition_name / season_name derived from StatsBomb
# match records, never typed by hand — see match_facts.DatasetIdentity)
# ---------------------------------------------------------------------------


def _synthetic_match(match_id, competition_id=2, competition_name="Test League",
                      season_id=27, season_name="2025/2026", stage_name="Matchday 1"):
    """A minimal StatsBomb-shaped match record for identity-derivation tests."""
    return {
        "match_id": match_id,
        "match_date": "2026-01-01",
        "competition_stage": {"name": stage_name},
        "competition": {"competition_id": competition_id, "competition_name": competition_name},
        "season": {"season_id": season_id, "season_name": season_name},
        "stadium": {"name": "Test Stadium"},
        "home_team": {"home_team_name": "Home FC", "home_team_id": 1},
        "away_team": {"away_team_name": "Away FC", "away_team_id": 2},
        "home_score": 1,
        "away_score": 0,
    }


def test_resolve_dataset_identity_derives_name_from_matches():
    matches = [_synthetic_match(1), _synthetic_match(2)]

    identity = match_facts._resolve_dataset_identity(matches, requested_competition_id=2,
                                                       requested_season_id=27)

    assert identity == DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )
    assert identity.display_name == "Test League 2025/2026"


def test_resolve_dataset_identity_returns_none_for_no_matches():
    assert match_facts._resolve_dataset_identity([], 2, 27) is None


def test_resolve_dataset_identity_raises_on_inconsistent_competition():
    matches = [
        _synthetic_match(1, competition_id=2, competition_name="Test League"),
        _synthetic_match(2, competition_id=2, competition_name="Renamed League"),
    ]

    with pytest.raises(ValueError):
        match_facts._resolve_dataset_identity(matches, requested_competition_id=2,
                                               requested_season_id=27)


def test_resolve_dataset_identity_raises_on_inconsistent_season():
    matches = [
        _synthetic_match(1, season_id=27, season_name="2025/2026"),
        _synthetic_match(2, season_id=27, season_name="2026/2027"),
    ]

    with pytest.raises(ValueError):
        match_facts._resolve_dataset_identity(matches, requested_competition_id=2,
                                               requested_season_id=27)


def test_resolve_dataset_identity_raises_on_requested_id_mismatch():
    """The matches file's own identity must match what was actually requested."""
    matches = [_synthetic_match(1, competition_id=2, season_id=27)]

    with pytest.raises(ValueError):
        match_facts._resolve_dataset_identity(matches, requested_competition_id=999,
                                               requested_season_id=27)


def test_resolve_dataset_identity_matches_real_wc2022_data():
    """The real WC2022 matches file is internally consistent and self-describing."""
    matches = match_facts.load_json(
        match_facts.DATA_ROOT / "matches" / str(match_facts.COMPETITION_ID)
        / f"{match_facts.SEASON_ID}.json"
    )

    identity = match_facts._resolve_dataset_identity(
        matches, match_facts.COMPETITION_ID, match_facts.SEASON_ID,
    )

    assert identity == WC2022_DATASET_IDENTITY
    assert identity.display_name == "FIFA World Cup 2022"


def test_extract_all_returns_dataset_identity(monkeypatch):
    matches = [_synthetic_match(1), _synthetic_match(2)]

    def fake_load_json(path):
        path = Path(path)
        if path.name == "27.json":
            return matches
        return []  # events/lineups

    monkeypatch.setattr(match_facts, "load_json", fake_load_json)

    result = match_facts.extract_all(
        data_root=Path("dataset"), competition_id=2, season_id=27, verbose=False,
    )

    assert result["dataset_identity"] == DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )


def test_persist_writes_dataset_identity_name_fields(tmp_path):
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }
    identity = DatasetIdentity(
        competition_id=2, competition_name="Test League",
        season_id=27, season_name="2025/2026",
    )

    output_path = match_facts.persist(
        result, output_dir=tmp_path, competition_id=2, season_id=27,
        dataset_identity=identity,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["metadata"]["competition_id"] == 2
    assert data["metadata"]["competition_name"] == "Test League"
    assert data["metadata"]["season_id"] == 27
    assert data["metadata"]["season_name"] == "2025/2026"


def test_persist_raises_for_custom_ids_without_dataset_identity(tmp_path):
    """
    Safety policy: persist() refuses to write a knowingly incomplete
    identity (competition_name/season_name = None) for any competition
    other than the WC2022 default. The only production caller
    (src/extraction/extract.py) always has dataset_identity available from
    extract_all(), so this never breaks a real extraction — it only
    prevents a caller from silently creating a mislabeled artifact.
    """
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }

    with pytest.raises(ValueError):
        match_facts.persist(result, output_dir=tmp_path, competition_id=2, season_id=27)


def test_persist_raises_when_dataset_identity_ids_mismatch_requested_ids(tmp_path):
    """persist() must never write metadata mixing two different identities."""
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
    }
    mismatched_identity = DatasetIdentity(
        competition_id=999, competition_name="Other League",
        season_id=1, season_name="1999",
    )

    with pytest.raises(ValueError):
        match_facts.persist(
            result, output_dir=tmp_path, competition_id=2, season_id=27,
            dataset_identity=mismatched_identity,
        )

def test_extract_all_discovers_conservative_taxonomy_for_non_wc_dataset(monkeypatch):
    matches = [
        _synthetic_match(1, stage_name="League Phase"),
        _synthetic_match(2, stage_name="Final"),
    ]

    def fake_load_json(path):
        path = Path(path)
        if path.name == "27.json":
            return matches
        return []

    monkeypatch.setattr(match_facts, "load_json", fake_load_json)

    result = match_facts.extract_all(
        data_root=Path("dataset"),
        competition_id=2,
        season_id=27,
        verbose=False,
    )

    taxonomy = result["stage_taxonomy"]

    assert taxonomy.stages == frozenset({"League Phase", "Final"})
    assert taxonomy.knockout_stages == frozenset()
    assert taxonomy.group_stages == frozenset()
    assert taxonomy.stage_order == ()

    by_stage = {fact.stage: fact for fact in result["match_facts"]}
    assert by_stage["League Phase"].is_knockout is False
    assert by_stage["Final"].is_knockout is False

def test_persist_writes_stage_taxonomy_metadata(tmp_path):
    from src.stage_taxonomy import StageTaxonomy

    taxonomy = StageTaxonomy(
        stages=frozenset({"League Phase", "Final"}),
        knockout_stages=frozenset({"Final"}),
        group_stages=frozenset({"League Phase"}),
        stage_order=("League Phase", "Final"),
    )
    identity = DatasetIdentity(
        competition_id=2,
        competition_name="Test League",
        season_id=27,
        season_name="2025/2026",
    )
    result = {
        "player_match_facts": [],
        "match_facts": [],
        "team_match_facts": [],
        "stage_taxonomy": taxonomy,
    }

    output_path = match_facts.persist(
        result,
        output_dir=tmp_path,
        competition_id=2,
        season_id=27,
        dataset_identity=identity,
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))

    assert data["metadata"]["stage_taxonomy"] == taxonomy.to_dict()
