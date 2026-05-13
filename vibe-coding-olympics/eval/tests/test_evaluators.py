from evaluators import _coerce_score


def test_coerce_score_accepts_unit_interval() -> None:
    assert _coerce_score(0.75) == 0.75


def test_coerce_score_accepts_display_scale() -> None:
    assert _coerce_score(7.5) == 0.75


def test_coerce_score_clamps_after_display_scale_conversion() -> None:
    assert _coerce_score(12) == 1.0
