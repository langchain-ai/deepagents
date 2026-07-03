"""Tests for LLM cost estimation helpers in `_cost`."""

from __future__ import annotations

import pytest

from deepagents_code._cost import estimate_request_cost, format_cost


class TestEstimateRequestCost:
    """Tests for estimate_request_cost()."""

    def test_known_model_returns_positive_cost(self) -> None:
        cost = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        assert cost is not None
        assert cost > 0

    def test_known_model_without_provider(self) -> None:
        # genai-prices can infer the provider from a well-known model name.
        cost = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="gpt-4o",
        )
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self) -> None:
        cost = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="totally-made-up-model-xyz",
            provider="nonexistent-provider",
        )
        assert cost is None

    def test_empty_model_returns_none(self) -> None:
        assert (
            estimate_request_cost(input_tokens=1000, output_tokens=100, model_name="")
            is None
        )

    def test_zero_tokens_returns_none(self) -> None:
        assert (
            estimate_request_cost(
                input_tokens=0,
                output_tokens=0,
                model_name="claude-sonnet-4-5",
                provider="anthropic",
            )
            is None
        )

    def test_more_output_costs_more(self) -> None:
        # Output tokens are billed at a higher rate than input for most models,
        # so a request with more output should never be cheaper.
        low = estimate_request_cost(
            input_tokens=1000,
            output_tokens=10,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        high = estimate_request_cost(
            input_tokens=1000,
            output_tokens=1000,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        assert low is not None
        assert high is not None
        assert high > low


class TestFormatCost:
    """Tests for format_cost()."""

    @pytest.mark.parametrize(
        ("cost", "expected"),
        [
            (0.0, "$0"),
            (-1.0, "$0"),
            (0.0045, "$0.0045"),
            (0.5, "$0.5000"),
            (1.0, "$1.00"),
            (1.234, "$1.23"),
            (12.5, "$12.50"),
        ],
    )
    def test_formatting(self, cost: float, expected: str) -> None:
        assert format_cost(cost) == expected
