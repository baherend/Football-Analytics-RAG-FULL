"""
test_run_multilingual_diagnostics.py -- CLI dispatch tests for
run_multilingual_diagnostics.py's --phase argument. Verifies routing only
(that --phase entity/language-entity call the correct existing phase
function with the correct data) -- never runs real embedding/retrieval
work.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from tests.run_multilingual_diagnostics import main
from tests.run_phase4_phase5 import PHASE4_PAIRS, PHASE5_VARIANTS


def _run_main_with_phase(phase, out_dir):
    argv = ["run_multilingual_diagnostics.py", "--phase", phase, "--out-dir", str(out_dir)]
    with patch.object(sys, "argv", argv):
        main()


def test_phase_entity_invokes_run_entity_phase(tmp_path):
    with patch("tests.run_multilingual_diagnostics.run_entity_phase") as mock_run:
        _run_main_with_phase("entity", tmp_path)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["minimal_pairs"] == PHASE4_PAIRS


def test_phase_language_entity_invokes_run_language_entity_phase(tmp_path):
    with patch("tests.run_multilingual_diagnostics.run_language_entity_phase") as mock_run:
        _run_main_with_phase("language-entity", tmp_path)

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["variant_cases"] == PHASE5_VARIANTS
