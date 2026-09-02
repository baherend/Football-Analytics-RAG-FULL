"""Representative Arabic query-understanding coverage matrix."""

from __future__ import annotations

import pytest

from src.query.parsing import parse_structured_query
from src.query.router import route_query


@pytest.mark.parametrize(
    ("question", "entity"),
    [
        ("من هو هداف كأس العالم 2022 وكم هدفًا سجل؟", "player"),
        ("من سجل أكبر عدد من الأهداف؟", "player"),
        ("ما الفريق الأكثر تسجيلًا للأهداف؟", "team"),
    ],
)
def test_arabic_goal_superlatives_are_structured(question: str, entity: str):
    route = route_query(question)
    parsed = parse_structured_query(question)

    assert route.path == "structured"
    assert parsed is not None
    assert parsed.entity == entity
    assert parsed.metric == "goals"
    assert parsed.aggregation == "max"


@pytest.mark.parametrize(
    "question",
    [
        "كيف لعبت Argentina في المباراة النهائية؟",
        "كيف لعب Leicester City خلال الموسم؟",
    ],
)
def test_arabic_qualitative_questions_remain_semantic(question: str):
    route = route_query(question)

    assert route.path == "semantic"
    assert parse_structured_query(question) is None


@pytest.mark.parametrize(
    ("question", "entity_name"),
    [
        ("كم هدفًا سجل Kylian Mbappé؟", "Kylian Mbappé"),
        ("كم هدفًا سجل Lionel Messi؟", "Lionel Messi"),
        ("كم اهداف سجل Jamie Vardy؟", "Jamie Vardy"),
        ("كم عدد أهداف Harry Kane؟", "Harry Kane"),
    ],
)
def test_arabic_named_player_numeric_questions_are_structured(
    question: str,
    entity_name: str,
):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.intent == "numeric"
    assert route.structured_query.entity == "player"
    assert route.structured_query.entity_name == entity_name
    assert route.structured_query.metric == "goals"


@pytest.mark.parametrize(
    ("question", "entity", "metric"),
    [
        ("من صنع أكبر عدد من الأهداف؟", "player", "assists"),
        ("من لديه أكبر عدد من التمريرات الحاسمة؟", "player", "assists"),
        ("من سدد أكبر عدد من التسديدات؟", "player", "shots"),
        ("من لديه أعلى xG؟", "player", "xg"),
    ],
)
def test_arabic_non_goal_superlatives_are_structured(
    question: str,
    entity: str,
    metric: str,
):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.intent == "superlative"
    assert route.structured_query.entity == entity
    assert route.structured_query.metric == metric
    assert route.structured_query.aggregation == "max"


@pytest.mark.parametrize(
    ("question", "entity_name", "metric"),
    [
        ("كم تمريرة حاسمة صنع Lionel Messi؟", "Lionel Messi", "assists"),
        ("كم تمريرةً حاسمةً صنع Lionel Messi؟", "Lionel Messi", "assists"),
        ("كم تسديدة سدد Kylian Mbappé؟", "Kylian Mbappé", "shots"),
        ("كم بلغ xG Lionel Messi؟", "Lionel Messi", "xg"),
    ],
)
def test_arabic_non_goal_named_player_questions_are_structured(
    question: str,
    entity_name: str,
    metric: str,
):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.intent == "numeric"
    assert route.structured_query.entity == "player"
    assert route.structured_query.entity_name == entity_name
    assert route.structured_query.metric == metric
    assert route.structured_query.aggregation == "sum"


@pytest.mark.parametrize(
    ("question", "metric"),
    [
        ("كم تسديدة صنع Lionel Messi؟", "shots"),
        ("كم تمريرة حاسمة سدد Lionel Messi؟", "assists"),
        ("من صنع أكبر عدد من التسديدات؟", "shots"),
        ("من سدد أكبر عدد من التمريرات الحاسمة؟", "assists"),
    ],
)
def test_arabic_explicit_metric_phrase_wins_over_mismatched_verb(
    question: str,
    metric: str,
):
    route = route_query(question)

    assert route.path == "structured"
    assert route.structured_query is not None
    assert route.structured_query.metric == metric


@pytest.mark.parametrize(
    "question",
    [
        "ما الفريق صاحب أعلى xG؟",
        "كم بلغ xG فريق Leicester City؟",
    ],
)
def test_unsupported_arabic_team_xg_questions_remain_semantic(question: str):
    route = route_query(question)

    assert route.path == "semantic"
    assert parse_structured_query(question) is None


@pytest.mark.xfail(
    strict=True,
    reason="Arabic compositional dependency parsing is not implemented",
)
def test_arabic_compositional_dependency_routes_hybrid():
    route = route_query("كيف لعب الفريق الأكثر تسجيلًا للأهداف؟")

    assert route.path == "hybrid"
    assert route.dependency_query is not None
    assert route.dependency_query.entity == "team"
    assert route.dependency_query.metric == "goals"


@pytest.mark.xfail(
    strict=True,
    reason="Arabic comparison routing is not implemented",
)
@pytest.mark.parametrize(
    "question",
    [
        "قارن بين Lionel Messi و Kylian Mbappe في الأهداف.",
        "قارن بين Harry Kane و Jamie Vardy في الأهداف.",
    ],
)
def test_arabic_comparisons_route_hybrid(question: str):
    assert route_query(question).path == "hybrid"
