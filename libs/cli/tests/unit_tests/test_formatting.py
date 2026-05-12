"""Tests for formatting module."""

from __future__ import annotations

import pytest

from deepagents_cli.formatting import format_duration


class TestFormatDuration:
    """Tests for format_duration() helper."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (1, "1s"),
            (5, "5s"),
            (59, "59s"),
        ],
    )
    def test_whole_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.5, "0.5s"),
            (1.3, "1.3s"),
            (59.9, "59.9s"),
        ],
    )
    def test_fractional_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (60, "1m 0s"),
            (61, "1m 1s"),
            (90, "1m 30s"),
            (125, "2m 5s"),
            (3599, "59m 59s"),
        ],
    )
    def test_minutes(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (3600, "1h 0m 0s"),
            (3661, "1h 1m 1s"),
            (7384, "2h 3m 4s"),
        ],
    )
    def test_hours(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    def test_boundary_rounds_up_to_minute(self) -> None:
        """59.95 rounds to 60.0 which should render as 1m 0s."""
        assert format_duration(59.95) == "1m 0s"

    def test_whole_float_renders_without_decimal(self) -> None:
        """A float like 5.0 should render as '5s', not '5.0s'."""
        assert format_duration(5.0) == "5s"
