"""Tests for the cost_tracking module (estimate_cost + CostTrackingMiddleware)."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.cost_tracking import (
    CostTrackingMiddleware,
    estimate_cost,
)

# A model + provider known to genai-prices' bundled price data.
KNOWN_MODEL = "claude-sonnet-4-5"
KNOWN_PROVIDER = "anthropic"


def _runtime() -> SimpleNamespace:
    """Build a stand-in `Runtime` (the middleware ignores it)."""
    return SimpleNamespace(context=None)


def _usage(input_tokens: int, output_tokens: int, **extra: object) -> dict:
    """Build a usage_metadata dict with the required total_tokens field."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        **extra,
    }


def _ai(
    usage: dict | None, *, model: str = KNOWN_MODEL, provider: str = KNOWN_PROVIDER
) -> AIMessage:
    """Build an AIMessage with usage + model metadata for pricing."""
    return AIMessage(
        content="hi",
        usage_metadata=usage,  # type: ignore[arg-type]
        response_metadata={"model_name": model, "model_provider": provider},
    )


class TestEstimateCost:
    """Tests for estimate_cost()."""

    def test_known_model_returns_positive_decimal(self) -> None:
        cost = estimate_cost(
            {"input_tokens": 1000, "output_tokens": 500},
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert isinstance(cost, Decimal)
        assert cost > 0

    def test_provider_inferred_when_omitted(self) -> None:
        cost = estimate_cost({"input_tokens": 1000, "output_tokens": 500}, KNOWN_MODEL)
        assert cost is not None
        assert cost > 0

    def test_unknown_model_returns_none(self) -> None:
        assert (
            estimate_cost(
                {"input_tokens": 1000, "output_tokens": 500},
                "definitely-not-a-real-model-xyz",
                KNOWN_PROVIDER,
            )
            is None
        )

    def test_missing_usage_returns_none(self) -> None:
        assert estimate_cost(None, KNOWN_MODEL, KNOWN_PROVIDER) is None

    def test_missing_model_returns_none(self) -> None:
        assert estimate_cost({"input_tokens": 100, "output_tokens": 50}, "") is None

    def test_zero_tokens_returns_none(self) -> None:
        assert (
            estimate_cost(
                {"input_tokens": 0, "output_tokens": 0}, KNOWN_MODEL, KNOWN_PROVIDER
            )
            is None
        )

    def test_cache_read_tokens_reduce_cost(self) -> None:
        # Cached input reads are cheaper than uncached input, so pricing the same
        # token total with most of it served from cache must cost strictly less.
        base = estimate_cost(
            {"input_tokens": 1000, "output_tokens": 0},
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        cached = estimate_cost(
            {
                "input_tokens": 1000,
                "output_tokens": 0,
                "input_token_details": {"cache_read": 900},
            },
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert base is not None
        assert cached is not None
        assert cached < base


class TestCostTrackingMiddleware:
    """Tests for CostTrackingMiddleware.after_model."""

    def test_accumulates_session_cost(self) -> None:
        mw = CostTrackingMiddleware()
        usage = _usage(1000, 500)

        first = mw.after_model({"messages": [_ai(usage)]}, _runtime())  # ty: ignore
        assert first is not None
        cost1 = first["_session_cost_usd"]
        assert cost1 > 0

        # Second call carries the prior cumulative total in state.
        second = mw.after_model(
            {"messages": [HumanMessage("q"), _ai(usage)], "_session_cost_usd": cost1},
            _runtime(),  # ty: ignore
        )
        assert second is not None
        assert second["_session_cost_usd"] > cost1

    def test_prices_latest_ai_message(self) -> None:
        mw = CostTrackingMiddleware()
        state: dict[str, Any] = {
            "messages": [
                _ai(_usage(10, 5)),
                HumanMessage("follow up"),
                _ai(_usage(1000, 500)),
            ]
        }
        result = mw.after_model(state, _runtime())  # ty: ignore
        expected = estimate_cost(_usage(1000, 500), KNOWN_MODEL, KNOWN_PROVIDER)
        assert result is not None
        assert expected is not None
        assert result["_session_cost_usd"] == pytest.approx(float(expected))

    def test_no_usage_returns_none(self) -> None:
        mw = CostTrackingMiddleware()
        assert mw.after_model({"messages": [_ai(None)]}, _runtime()) is None  # ty: ignore

    def test_unpriceable_model_returns_none(self) -> None:
        mw = CostTrackingMiddleware()
        msg = _ai(_usage(100, 50), model="no-such-model-xyz")
        assert mw.after_model({"messages": [msg]}, _runtime()) is None  # ty: ignore

    def test_no_messages_returns_none(self) -> None:
        mw = CostTrackingMiddleware()
        assert mw.after_model({"messages": []}, _runtime()) is None  # ty: ignore

    def test_model_spec_fallback_from_state(self) -> None:
        # No model info on the message; resolved from the `_model_spec` channel.
        mw = CostTrackingMiddleware()
        msg = AIMessage(
            content="hi",
            usage_metadata=_usage(1000, 500),  # type: ignore[arg-type]
        )
        state: dict[str, Any] = {
            "messages": [msg],
            "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
        }
        result = mw.after_model(state, _runtime())  # ty: ignore
        assert result is not None
        assert result["_session_cost_usd"] > 0
