"""
Driver for Phase 4 (expanded entity-script minimal pairs) and Phase 5
(language-vs-entity A-E variants). Not a general CLI -- the specific case
selection and translated query text for these two diagnostics, kept
separate from run_multilingual_diagnostics.py's reusable phase functions
(which it imports and calls).
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.run_multilingual_diagnostics import run_entity_phase, run_language_entity_phase

DEFAULT_OUT_DIR = Path(tempfile.gettempdir()) / "football_analytics_rag_diagnostics"

# ---------------------------------------------------------------------------
# Phase 4: 12 expanded minimal pairs (Egyptian Arabic sentence, Latin vs
# Arabic-transliterated entity only) -- 8 player-entity cases, 4 team-entity
# cases.
# ---------------------------------------------------------------------------

PHASE4_PAIRS = [
    # (case_id, latin_query, arabic_query, entity_kind)
    ("gt-pilot-l3-01", "Messi لعب ازاي قدام Croatia في نص النهائي؟", "ميسي لعب ازاي قدام كرواتيا في نص النهائي؟", "player"),
    ("gt-pilot-l4-01", "كلمني عن أداء Messi في البطولة كلها.", "كلمني عن أداء ميسي في البطولة كلها.", "player"),
    ("gt-l3-02", "Antoine Griezmann عمل كام باسة في النهائي؟", "أنطوان جريزمان عمل كام باسة في النهائي؟", "player"),
    ("gt-l4-02", "Kylian Mbappé لعب ازاي في البطولة كلها؟", "كيليان مبابي لعب ازاي في البطولة كلها؟", "player"),
    ("gt-l3-03", "Kylian Mbappé لعب ازاي قدام Poland في دور الـ16؟", "كيليان مبابي لعب ازاي قدام بولندا في دور الـ16؟", "player"),
    ("gt-l4-03", "Antoine Griezmann لعب ازاي في البطولة كلها؟", "أنطوان جريزمان لعب ازاي في البطولة كلها؟", "player"),
    ("gt-l3-04", "Enzo Fernández لعب دفاع ازاي في نهائي كأس العالم؟", "إنزو فرنانديز لعب دفاع ازاي في نهائي كأس العالم؟", "player"),
    ("gt-l4-04", "Enzo Fernández لعب ازاي في البطولة كلها؟", "إنزو فرنانديز لعب ازاي في البطولة كلها؟", "player"),
    ("gt-pilot-team-01", "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟", "الأرجنتين كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟", "team"),
    ("gt-team-02", "Morocco لعبت ازاي في البطولة، وايه أكتر تشكيل لعبوا بيه؟", "المغرب لعبت ازاي في البطولة، وايه أكتر تشكيل لعبوا بيه؟", "team"),
    ("gt-team-03", "France كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟", "فرنسا كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟", "team"),
    ("gt-team-04", "Germany لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟", "ألمانيا لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟", "team"),
]

# ---------------------------------------------------------------------------
# Phase 5: A=EN+Latin, B=MSA+Latin, C=EGY+Latin, D=MSA+Arabic-entity,
# E=EGY+Arabic-entity, for 6 cases (3 player, 3 team).
# ---------------------------------------------------------------------------

PHASE5_VARIANTS = {
    "gt-pilot-l3-01": {
        "A": "How did Messi perform against Croatia in the semi-final?",
        "B": "كيف كان أداء Messi أمام Croatia في نصف النهائي؟",
        "C": "Messi لعب ازاي قدام Croatia في نص النهائي؟",
        "D": "كيف كان أداء ميسي أمام كرواتيا في نصف النهائي؟",
        "E": "ميسي لعب ازاي قدام كرواتيا في نص النهائي؟",
    },
    "gt-pilot-team-01": {
        "A": "What was Argentina's playing style and most common formation?",
        "B": "ما هو أسلوب لعب Argentina والتشكيل الأكثر استخدامًا؟",
        "C": "Argentina كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟",
        "D": "ما هو أسلوب لعب الأرجنتين والتشكيل الأكثر استخدامًا؟",
        "E": "الأرجنتين كانت بتلعب ازاي وايه أكتر تشكيل استخدموه؟",
    },
    "gt-l4-02": {
        "A": "Describe Kylian Mbappé's overall World Cup tournament performance.",
        "B": "صف أداء Kylian Mbappé العام في بطولة كأس العالم.",
        "C": "Kylian Mbappé لعب ازاي في البطولة كلها؟",
        "D": "صف أداء كيليان مبابي العام في بطولة كأس العالم.",
        "E": "كيليان مبابي لعب ازاي في البطولة كلها؟",
    },
    "gt-team-03": {
        "A": "What were France's passing patterns and most common formations?",
        "B": "ما هي أنماط التمرير لدى France والتشكيلات الأكثر استخدامًا؟",
        "C": "France كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟",
        "D": "ما هي أنماط التمرير لدى فرنسا والتشكيلات الأكثر استخدامًا؟",
        "E": "فرنسا كانت بتلعب باسات ازاي وايه أكتر تشكيل استخدموه؟",
    },
    "gt-l3-04": {
        "A": "How did Enzo Fernández perform defensively in the World Cup Final?",
        "B": "كيف كان أداء Enzo Fernández الدفاعي في نهائي كأس العالم؟",
        "C": "Enzo Fernández لعب دفاع ازاي في نهائي كأس العالم؟",
        "D": "كيف كان أداء إنزو فرنانديز الدفاعي في نهائي كأس العالم؟",
        "E": "إنزو فرنانديز لعب دفاع ازاي في نهائي كأس العالم؟",
    },
    "gt-team-04": {
        "A": "How did Germany play in the tournament, and which formations did they use?",
        "B": "كيف لعبت Germany في البطولة، وما هي التشكيلات التي استخدمتها؟",
        "C": "Germany لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟",
        "D": "كيف لعبت ألمانيا في البطولة، وما هي التشكيلات التي استخدمتها؟",
        "E": "ألمانيا لعبت ازاي في البطولة، وايه التشكيلات اللي استخدموها؟",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver for Phase 4 and Phase 5 multilingual diagnostics")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                         help="Directory to write phase4_entity_script.json and phase5_language_entity.json "
                              "(default: an OS-managed temp directory)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    print("=== Phase 4: expanded entity-script minimal pairs ===", flush=True)
    run_entity_phase(out_dir, minimal_pairs=PHASE4_PAIRS)

    print("\n=== Phase 5: language-vs-entity A-E variants ===", flush=True)
    run_language_entity_phase(out_dir, variant_cases=PHASE5_VARIANTS)
