"""Tests for _session_stats module."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from deepagents_code._session_stats import (
    ModelStats,
    SessionStats,
    format_cost,
    format_token_count,
    print_usage_table,
)


class TestFormatCost:
    """Tests for compact USD formatting."""

    @pytest.mark.parametrize(
        ("cost_usd", "expected"),
        [
            (0.0, "$0.00"),
            (-1.0, "$0.00"),
            (0.0001, "<$0.01"),
            (0.009, "<$0.01"),
            (0.01, "$0.01"),
            (0.42, "$0.42"),
            (12.5, "$12.50"),
        ],
    )
    def test_format(self, cost_usd: float, expected: str) -> None:
        assert format_cost(cost_usd) == expected


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
        assert stats.cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 0
        assert stats.provider == ""


class TestSessionStats:
    """Tests for SessionStats accumulation logic."""

    def test_defaults(self) -> None:
        stats = SessionStats()
        assert stats.request_count == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.total_cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 0
        assert stats.wall_time_seconds == pytest.approx(0.0)
        assert stats.per_model == {}

    def test_record_request_increments_totals(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_record_request_accumulates(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        stats.record_request("gpt-5.5", 200, 75)
        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 125

    def test_record_request_populates_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        assert ("", "gpt-5.5") in stats.per_model
        model = stats.per_model["", "gpt-5.5"]
        assert model.request_count == 1
        assert model.input_tokens == 100
        assert model.output_tokens == 50
        assert model.model_name == "gpt-5.5"

    def test_record_request_multiple_models(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50)
        stats.record_request("claude-sonnet-4-5", 200, 75)
        assert len(stats.per_model) == 2
        assert stats.per_model["", "gpt-5.5"].input_tokens == 100
        assert stats.per_model["", "claude-sonnet-4-5"].input_tokens == 200
        assert stats.request_count == 2
        assert stats.input_tokens == 300

    def test_record_request_records_provider(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, provider="openai")
        assert stats.per_model["openai", "gpt-5.5"].provider == "openai"

    def test_record_request_splits_same_model_by_provider(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, provider="openai")
        stats.record_request("gpt-5.5", 200, 75, provider="azure")

        assert len(stats.per_model) == 2
        assert stats.per_model["openai", "gpt-5.5"].input_tokens == 100
        assert stats.per_model["azure", "gpt-5.5"].input_tokens == 200

    def test_record_request_empty_model_skips_per_model(self) -> None:
        stats = SessionStats()
        stats.record_request("", 100, 50)
        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.per_model == {}

    def test_record_request_accumulates_cost(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        stats.record_request("gpt-5.5", 200, 75, cost_usd=0.02)
        assert stats.total_cost_usd == pytest.approx(0.03)
        assert stats.priced_request_count == 2
        assert stats.per_model["", "gpt-5.5"].cost_usd == pytest.approx(0.03)

    def test_missing_cost_does_not_inflate_total(self) -> None:
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        stats.record_request("unknown", 200, 75, cost_usd=None)
        assert stats.total_cost_usd == pytest.approx(0.01)
        assert stats.priced_request_count == 1
        assert stats.per_model["", "unknown"].priced_request_count == 0

    def test_literal_zero_cost_is_recorded_but_does_not_inflate_total(self) -> None:
        stats = SessionStats()
        stats.record_request("free-model", 100, 50, cost_usd=0.0)
        assert stats.total_cost_usd == pytest.approx(0.0)
        assert stats.priced_request_count == 1
        assert stats.per_model["", "free-model"].priced_request_count == 1

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

    def test_merge_combines_cost(self) -> None:
        first = SessionStats()
        first.record_request("gpt-5.5", 100, 50, cost_usd=0.01)
        second = SessionStats()
        second.record_request("gpt-5.5", 200, 75, cost_usd=0.02)

        first.merge(second)

        assert first.total_cost_usd == pytest.approx(0.03)
        assert first.priced_request_count == 2
        assert first.per_model["", "gpt-5.5"].cost_usd == pytest.approx(0.03)

    def test_merge_combines_per_model(self) -> None:
        a = SessionStats()
        a.record_request("gpt-5.5", 100, 50)

        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75)
        b.record_request("claude-sonnet-4-5", 300, 100)

        a.merge(b)
        assert a.per_model["", "gpt-5.5"].input_tokens == 300
        assert a.per_model["", "gpt-5.5"].request_count == 2
        assert a.per_model["", "claude-sonnet-4-5"].input_tokens == 300

    def test_merge_carries_provider(self) -> None:
        a = SessionStats()
        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75, provider="openai")

        a.merge(b)
        assert a.per_model["openai", "gpt-5.5"].provider == "openai"

    def test_merge_splits_same_model_by_provider(self) -> None:
        a = SessionStats()
        a.record_request("gpt-5.5", 100, 50, provider="openai")

        b = SessionStats()
        b.record_request("gpt-5.5", 200, 75, provider="azure")

        a.merge(b)
        assert len(a.per_model) == 2
        assert a.per_model["openai", "gpt-5.5"].input_tokens == 100
        assert a.per_model["azure", "gpt-5.5"].input_tokens == 200

    def test_merge_empty_into_populated(self) -> None:
        a = SessionStats(request_count=5, input_tokens=500)
        b = SessionStats()
        a.merge(b)
        assert a.request_count == 5
        assert a.input_tokens == 500


class TestPrintUsageTable:
    """Tests for `print_usage_table` output."""

    def test_no_model_called_skips_unknown_row(self) -> None:
        """When no model was called, the table should not show 'unknown'."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=1.5, console=console)
        output = buf.getvalue()
        assert "unknown" not in output
        assert "Usage Stats" not in output
        assert "Agent active" in output

    def test_single_model_shows_name(self) -> None:
        """Single-model session should display the model name."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, cost_usd=0.42)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Cost" in output
        assert "$0.42" in output
        assert "unknown" not in output

    def test_unpriced_model_does_not_render_zero_cost(self) -> None:
        stats = SessionStats()
        stats.record_request("self-hosted", 100, 50, cost_usd=None)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "Cost" in output
        assert "$0.00" not in output
        assert "—" in output

    def test_shows_provider_name(self) -> None:
        """The table should include the provider for each model."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "Provider" in output
        assert "openai" in output
        assert "gpt-4" in output

    def test_multi_model_shows_all_names_and_total(self) -> None:
        """Multi-model session should show each model and a Total row."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "claude-opus-4-6" in output
        assert "Total" in output
        assert "unknown" not in output

    def test_same_model_with_different_providers_shows_separate_rows(self) -> None:
        """Same-name models from different providers should render separately."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50, provider="openai")
        stats.record_request("gpt-4", 200, 80, provider="azure")
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "openai" in output
        assert "azure" in output
        assert "Total" in output
        # Two distinct rows, not a collapsed one: each provider's per-row token
        # counts must appear (100/50 and 200/80), alongside the 300/130 totals.
        assert "100" in output
        assert "50" in output
        assert "200" in output
        assert "80" in output

    def test_tokens_with_no_wall_time_omits_timing_line(self) -> None:
        """Token table should print but timing line should be absent."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Agent active" not in output

    def test_no_requests_no_time_prints_nothing(self) -> None:
        """Empty stats with negligible wall time should print nothing."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.01, console=console)
        output = buf.getvalue()
        assert output.strip() == ""
