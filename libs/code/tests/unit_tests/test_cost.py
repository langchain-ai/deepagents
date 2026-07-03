"""Tests for LLM cost estimation helpers in `_cost`."""

from __future__ import annotations

import pytest

from deepagents_code._cost import (
    cache_tokens_from_usage,
    estimate_request_cost,
    format_cost,
)


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

    def test_cache_read_is_cheaper_than_uncached_input(self) -> None:
        # Cache reads are billed at a reduced rate, so pricing 900 of 1000 input
        # tokens as cache reads must cost less than pricing all 1000 at full
        # rate. This is the divergence that made local estimates undershoot
        # LangSmith before cache tokens were priced.
        uncached = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        with_cache_reads = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            cache_read_tokens=900,
        )
        assert uncached is not None
        assert with_cache_reads is not None
        assert with_cache_reads < uncached

    def test_cache_write_is_pricier_than_uncached_input(self) -> None:
        # Cache writes carry a premium over ordinary input, so moving input
        # tokens into the cache-write bucket must not reduce the cost.
        uncached = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        with_cache_writes = estimate_request_cost(
            input_tokens=1000,
            output_tokens=100,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            cache_write_tokens=900,
        )
        assert uncached is not None
        assert with_cache_writes is not None
        assert with_cache_writes > uncached

    def test_cache_tokens_not_double_counted(self) -> None:
        # `input_tokens` is the grand total including cached tokens. Supplying a
        # cache_read that equals the full input must not exceed pricing the same
        # input entirely at the (higher) uncached rate.
        all_uncached = estimate_request_cost(
            input_tokens=1000,
            output_tokens=0,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
        )
        all_cache_read = estimate_request_cost(
            input_tokens=1000,
            output_tokens=0,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            cache_read_tokens=1000,
        )
        assert all_uncached is not None
        assert all_cache_read is not None
        # Cache reads are cheaper, and none of the 1000 tokens is billed twice.
        assert all_cache_read < all_uncached

    def test_cache_tokens_exceeding_input_clamped(self) -> None:
        # Defensive: if cache counts exceed input_tokens (inconsistent provider
        # reporting), uncached input clamps to zero rather than going negative.
        cost = estimate_request_cost(
            input_tokens=100,
            output_tokens=0,
            model_name="claude-sonnet-4-5",
            provider="anthropic",
            cache_read_tokens=500,
        )
        assert cost is not None
        assert cost > 0


class TestCacheTokensFromUsage:
    """Tests for cache_tokens_from_usage()."""

    def test_none_usage_returns_zeros(self) -> None:
        assert cache_tokens_from_usage(None) == (0, 0)

    def test_missing_details_returns_zeros(self) -> None:
        assert cache_tokens_from_usage({"input_tokens": 100}) == (0, 0)

    def test_extracts_cache_read_and_creation(self) -> None:
        usage = {
            "input_tokens": 350,
            "input_token_details": {"cache_creation": 200, "cache_read": 100},
        }
        # Returns (cache_read, cache_write); cache_write maps to cache_creation.
        assert cache_tokens_from_usage(usage) == (100, 200)

    def test_partial_details(self) -> None:
        usage = {"input_token_details": {"cache_read": 42}}
        assert cache_tokens_from_usage(usage) == (42, 0)


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
