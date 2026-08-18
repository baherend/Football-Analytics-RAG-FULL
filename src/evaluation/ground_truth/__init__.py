"""
src/evaluation/ground_truth/ -- benchmark datasets and their registry.

Migration Step 7: moved byte-identically from tests/ (see PROJECT_MEMORY.md).

    semantic.py       -- the 24-case WC2022 semantic ground truth
    answerability.py  -- answerability expectations
    multilingual.py   -- EN/MSA/EGY case construction over the semantic set
    registry.py       -- dataset identity -> ground-truth bundle

This is **protected baseline data**. It is not edited to make a test pass:
expected document IDs, rankings, labels and thresholds change only through a
deliberate, separately-justified benchmark decision.
"""

__all__: list[str] = []
