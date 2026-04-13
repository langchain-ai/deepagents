"""Unit tests for deepagents_cli._session_stats."""

from __future__ import annotations

import pytest

from deepagents_cli._session_stats import ModelStats, SessionStats, format_token_count


class TestFormatTokenCount:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0"),
            (1, "1"),
            (500, "500"),
            (999, "999"),
        ],
    )
    def test_sub_thousand(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1000, "1.0K"),
            (1500, "1.5K"),
            (12500, "12.5K"),
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
            (10_500_000, "10.5M"),
        ],
    )
    def test_millions(self, count: int, expected: str) -> None:
        assert format_token_count(count) == expected


class TestModelStatsDefaults:
    def test_default_values(self) -> None:
        ms = ModelStats()
        assert ms.request_count == 0
        assert ms.input_tokens == 0
        assert ms.output_tokens == 0


class TestSessionStatsRecordRequest:
    def test_increments_totals(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_accumulates_multiple_requests(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("gpt-4", 200, 75)
        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 125

    def test_per_model_breakdown(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("gpt-4", 200, 75)
        assert "gpt-4" in stats.per_model
        ms = stats.per_model["gpt-4"]
        assert ms.request_count == 2
        assert ms.input_tokens == 300
        assert ms.output_tokens == 125

    def test_multiple_models(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-3", 80, 40)
        assert len(stats.per_model) == 2
        assert stats.request_count == 2
        assert stats.input_tokens == 180
        assert stats.output_tokens == 90

    def test_empty_model_name_skips_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert stats.per_model == {}


class TestSessionStatsMerge:
    def test_merge_totals(self) -> None:
        a = SessionStats(request_count=1, input_tokens=100, output_tokens=50, wall_time_seconds=1.0)
        b = SessionStats(request_count=2, input_tokens=200, output_tokens=75, wall_time_seconds=2.0)
        a.merge(b)
        assert a.request_count == 3
        assert a.input_tokens == 300
        assert a.output_tokens == 125
        assert a.wall_time_seconds == 3.0

    def test_merge_per_model_same_model(self) -> None:
        a = SessionStats()
        a.record_request("gpt-4", 100, 50)
        b = SessionStats()
        b.record_request("gpt-4", 200, 75)
        a.merge(b)
        ms = a.per_model["gpt-4"]
        assert ms.request_count == 2
        assert ms.input_tokens == 300
        assert ms.output_tokens == 125

    def test_merge_per_model_different_models(self) -> None:
        a = SessionStats()
        a.record_request("gpt-4", 100, 50)
        b = SessionStats()
        b.record_request("claude-3", 80, 40)
        a.merge(b)
        assert "gpt-4" in a.per_model
        assert "claude-3" in a.per_model

    def test_merge_empty_other(self) -> None:
        a = SessionStats(request_count=3, input_tokens=300, output_tokens=150)
        a.merge(SessionStats())
        assert a.request_count == 3
        assert a.input_tokens == 300
        assert a.output_tokens == 150
