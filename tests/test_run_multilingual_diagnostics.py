"""
test_run_multilingual_diagnostics.py -- CLI dispatch tests for
run_multilingual_diagnostics.py's --phase argument. Verifies routing only
(that --phase entity/language-entity call the correct existing phase
function with the correct data) -- never runs real embedding/retrieval
work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from src.evaluation.run_multilingual_diagnostics import PROJECT_ROOT, main
from src.evaluation.run_phase4_phase5 import PHASE4_PAIRS, PHASE5_VARIANTS


def _run_main_with_phase(phase, out_dir):
    argv = ["run_multilingual_diagnostics.py", "--phase", phase, "--out-dir", str(out_dir)]
    with patch.object(sys, "argv", argv):
        main()


def test_phase_entity_invokes_run_entity_phase(tmp_path):
    with patch("src.evaluation.run_multilingual_diagnostics.run_entity_phase") as mock_run:
        _run_main_with_phase("entity", tmp_path)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["minimal_pairs"] == PHASE4_PAIRS


def test_phase_language_entity_invokes_run_language_entity_phase(tmp_path):
    with patch("src.evaluation.run_multilingual_diagnostics.run_language_entity_phase") as mock_run:
        _run_main_with_phase("language-entity", tmp_path)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["variant_cases"] == PHASE5_VARIANTS


def test_phase_models_forwards_dataset_selection(tmp_path):
    argv = [
        "run_multilingual_diagnostics.py", "--phase", "models",
        "--out-dir", str(tmp_path), "--models", "model-a",
        "--competition-id", "2", "--season-id", "27",
        "--artifact-output-root", str(tmp_path / "artifacts"),
    ]
    with patch.object(sys, "argv", argv), patch(
        "src.evaluation.run_multilingual_diagnostics.run_model_phase"
    ) as mock_run:
        main()

    mock_run.assert_called_once_with(
        tmp_path,
        ["model-a"],
        competition_id=2,
        season_id=27,
        artifact_output_root=str(tmp_path / "artifacts"),
    )


def test_project_root_resolves_repository_root():
    assert (Path(PROJECT_ROOT) / "src" / "evaluation").is_dir()
