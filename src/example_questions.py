"""Competition-specific example questions for the Streamlit UI."""

from __future__ import annotations


EXAMPLE_QUESTIONS: dict[tuple[int, int], dict[str, tuple[str, ...]]] = {
    (43, 106): {
        "en": (
            "Who scored the most goals?",
            "Which team scored the most goals?",
            "How did Argentina play in the final?",
        ),
        "ar": (
            "من هو هداف كأس العالم 2022 وكم هدفًا سجل؟",
            "ما الفريق الأكثر تسجيلًا للأهداف؟",
            "كيف لعبت Argentina في المباراة النهائية؟",
        ),
    },
    (2, 27): {
        "en": (
            "Who scored the most goals?",
            "Which team scored the most goals?",
            "How did Leicester City play during the season?",
        ),
        "ar": (
            "من سجل أكبر عدد من الأهداف؟",
            "ما الفريق الأكثر تسجيلًا للأهداف؟",
            "كيف لعب Leicester City خلال الموسم؟",
        ),
    },
}
