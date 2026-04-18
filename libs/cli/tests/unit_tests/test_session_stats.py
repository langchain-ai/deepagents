"""Tests for _session_stats module."""

from __future__ import annotations

import pytest

from deepagents_cli._session_stats import (
    ModelStats,
    SessionStats,
    format_token_count,
)


class TestFormatTokenCount:
    """Tests for format_token_count()."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (999, "999"),
        ],
    )
    def test_small_counts(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1000, "1.0K"),
            (1500, "1.5K"),
            (12_500, "12.5K"),
            (999_999, "1000.0K"),
        ],
    )
    def test_thousands(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1_000_000, "1.0M"),
            (1_200_000, "1.2M"),
            (10_000_000, "10.0M"),
        ],
    )
    def test_millions(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected


class TestModelStats:
    """Tests for ModelStats dataclass."""

    def test_defaults(self) -> None:
        stats = ModelStats()
        assert stats.request_count == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0


class TestSessionStats:
    """Tests for SessionStats accumulation logic."""

    def test_defaults(self) -> None:
        stats = SessionStats()
        assert stats.request_count == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.wall_time_seconds == 0.0
        assert stats.per_model == {}

    def test_record_request_increments_totals(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4o", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_request_accumulates(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4o", 100, 50)
        stats.record_request("gpt-4o", 200, 75)
        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 125

    def test_record_request_populates_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4o", 100, 50)
        assert "gpt-4o" in stats.per_model
        model = stats.per_model["gpt-4o"]
        assert model.request_count == 1
        assert model.input_tokens == 100
        assert model.output_tokens == 50

    def test_record_request_multiple_models(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4o", 100, 50)
        stats.record_request("claude-sonnet-4-5", 200, 75)
        assert len(stats.per_model) == 2
        assert stats.per_model["gpt-4o"].input_tokens == 100
        assert stats.per_model["claude-sonnet-4-5"].input_tokens == 200
        assert stats.request_count == 2
        assert stats.input_tokens == 300

    def test_record_request_empty_model_skips_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.per_model == {}

    def test_merge_combines_totals(self) -> None:
        a = SessionStats(
            request_count=1,
            input_tokens=100,
            output_tokens=50,
            wall_time_seconds=1.5,
        )
        b = SessionStats(
            request_count=2,
            input_tokens=200,
            output_tokens=75,
            wall_time_seconds=2.0,
        )
        a.merge(b)
        assert a.request_count == 3
        assert a.input_tokens == 300
        assert a.output_tokens == 125
        assert a.wall_time_seconds == pytest.approx(3.5)

    def test_merge_combines_per_model(self) -> None:
        a = SessionStats()
        a.record_request("gpt-4o", 100, 50)

        b = SessionStats()
        b.record_request("gpt-4o", 200, 75)
        b.record_request("claude-sonnet-4-5", 300, 100)

        a.merge(b)
        assert a.per_model["gpt-4o"].input_tokens == 300
        assert a.per_model["gpt-4o"].request_count == 2
        assert a.per_model["claude-sonnet-4-5"].input_tokens == 300

    def test_merge_empty_into_populated(self) -> None:
        a = SessionStats(request_count=5, input_tokens=500)
        b = SessionStats()
        a.merge(b)
        assert a.request_count == 5
        assert a.input_tokens == 500
