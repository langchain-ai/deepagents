from __future__ import annotations

from aggregate import aggregate


def test_aggregate_counts_missing_known_axis_as_zero() -> None:
    score = aggregate(
        {"color": 1.0, "layout": None},
        weights={"color": 1.0, "layout": 1.0},
    )

    assert score == 0.5


def test_aggregate_ignores_unknown_axes() -> None:
    score = aggregate(
        {"color": 1.0, "unknown": 0.0},
        weights={"color": 1.0},
    )

    assert score == 1.0


def test_aggregate_returns_zero_when_no_weighted_axes_present() -> None:
    assert aggregate({"unknown": 1.0}, weights={"color": 1.0}) == 0.0
