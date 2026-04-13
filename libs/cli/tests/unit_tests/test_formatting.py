"""Unit tests for deepagents_cli.formatting."""

from __future__ import annotations

import pytest

from deepagents_cli.formatting import format_duration


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0s"),
            (1, "1s"),
            (59, "59s"),
            (59.0, "59s"),
            (5.0, "5s"),
        ],
    )
    def test_whole_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.5, "0.5s"),
            (1.3, "1.3s"),
            (10.7, "10.7s"),
            (59.4, "59.4s"),
        ],
    )
    def test_fractional_seconds(self, seconds: float, expected: str) -> None:
        assert format_duration(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (60, "1m 0s"),
            (90, "1m 30s"),
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

    def test_boundary_rounding_to_minute(self) -> None:
        # 59.95 rounds to 60.0 → should display as "1m 0s"
        assert format_duration(59.95) == "1m 0s"
