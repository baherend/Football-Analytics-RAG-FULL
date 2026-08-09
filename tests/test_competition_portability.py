import json
from pathlib import Path

import src.extraction.match_facts as match_facts


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

    output_path = match_facts.persist(
        result,
        output_dir=tmp_path,
        competition_id=2,
        season_id=27,
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

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None):
        calls["persist"] = {
            "competition_id": competition_id,
            "season_id": season_id,
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

    def fake_persist(result, output_dir=Path("output"), competition_id=None, season_id=None):
        calls["persist"] = {
            "competition_id": competition_id,
            "season_id": season_id,
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
