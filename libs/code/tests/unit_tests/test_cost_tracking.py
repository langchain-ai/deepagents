"""Tests for cost estimation and graph-side cumulative cost persistence."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.cost_tracking import (
    CostState,
    CostTrackingMiddleware,
    estimate_cost,
    resolve_message_model,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

KNOWN_MODEL = "claude-sonnet-4-5"
KNOWN_PROVIDER = "anthropic"


def _usage(
    input_tokens: int = 1_000,
    output_tokens: int = 100,
    *,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, Any]:
    """Build LangChain usage metadata for a completed request."""
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read or cache_write:
        usage["input_token_details"] = {
            "cache_read": cache_read,
            "cache_creation": cache_write,
        }
    return usage


def _message(
    usage: dict[str, Any] | None,
    *,
    model: str = KNOWN_MODEL,
    provider: str = KNOWN_PROVIDER,
) -> AIMessage:
    """Build an AI message carrying model and usage metadata."""
    return AIMessage(
        content="response",
        usage_metadata=usage,  # ty: ignore[invalid-argument-type]
        response_metadata={"model_name": model, "model_provider": provider},
    )


def _runtime() -> Runtime[Any]:
    """Build the runtime shape required by the middleware hook."""
    return cast("Runtime[Any]", SimpleNamespace(context=None))


class TestEstimateCost:
    """Tests for the shared `genai-prices` adapter."""

    def test_known_model_returns_positive_cost(self) -> None:
        cost_usd = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        assert cost_usd is not None
        assert cost_usd > 0

    def test_unknown_model_returns_none(self) -> None:
        assert (
            estimate_cost(
                _usage(),
                "definitely-not-a-real-model",
                "unknown-provider",
            )
            is None
        )

    def test_aggregate_only_usage_returns_none(self) -> None:
        assert (
            estimate_cost(
                {"total_tokens": 1_100},
                KNOWN_MODEL,
                KNOWN_PROVIDER,
            )
            is None
        )

    def test_azure_openai_uses_azure_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import genai_prices

        provider_ids: list[str | None] = []

        def fake_calc_price(
            usage: object,
            model_ref: str,
            *,
            provider_id: str | None = None,
        ) -> SimpleNamespace:
            assert usage is not None
            assert model_ref == "gpt-5.5"
            provider_ids.append(provider_id)
            return SimpleNamespace(total_price=0.42)

        monkeypatch.setattr(genai_prices, "calc_price", fake_calc_price)

        assert estimate_cost(_usage(), "gpt-5.5", "azure_openai") == pytest.approx(0.42)
        assert provider_ids == ["azure"]

    def test_malformed_price_result_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import genai_prices

        monkeypatch.setattr(
            genai_prices,
            "calc_price",
            lambda *_args, **_kwargs: SimpleNamespace(total_price=object()),
        )

        assert estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER) is None

    def test_azure_fallback_overrides_generic_openai_metadata(self) -> None:
        message = _message(_usage(), model="gpt-5.5", provider="openai")

        _, provider = resolve_message_model(
            message,
            fallback_model="gpt-5.5",
            fallback_provider="azure_openai",
        )

        assert provider == "azure_openai"

    def test_codex_subscription_usage_is_not_priced_as_openai_api(self) -> None:
        assert estimate_cost(_usage(), "gpt-5.4", "openai_codex") is None

    def test_cache_read_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        cached = estimate_cost(
            _usage(cache_read=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert cached is not None
        assert cached < uncached

    def test_cache_write_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        cached = estimate_cost(
            _usage(cache_write=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert cached is not None
        assert cached > uncached

    def test_cache_write_alias_is_priced_separately(self) -> None:
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        usage = _usage()
        usage["input_token_details"] = {"cache_write": 900}
        cached = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert uncached is not None
        assert cached is not None
        assert cached > uncached

    def test_anthropic_detailed_cache_writes_are_priced_separately(self) -> None:
        """Anthropic zeroes `cache_creation` when TTL breakdown fields are set."""
        uncached = estimate_cost(_usage(), KNOWN_MODEL, KNOWN_PROVIDER)
        usage = _usage()
        usage["input_token_details"] = {
            "cache_creation": 0,
            "ephemeral_5m_input_tokens": 600,
            "ephemeral_1h_input_tokens": 300,
        }
        cached = estimate_cost(usage, KNOWN_MODEL, KNOWN_PROVIDER)

        assert uncached is not None
        assert cached is not None
        assert cached > uncached
        # Same total as pricing the sum through the generic cache-write field.
        assert cached == estimate_cost(
            _usage(cache_write=900),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )

    def test_cache_tokens_are_not_double_counted(self) -> None:
        uncached = estimate_cost(
            _usage(output_tokens=0),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        all_cache_read = estimate_cost(
            _usage(output_tokens=0, cache_read=1_000),
            KNOWN_MODEL,
            KNOWN_PROVIDER,
        )
        assert uncached is not None
        assert all_cache_read is not None
        assert all_cache_read < uncached

    def test_module_import_does_not_import_genai_prices(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import deepagents_code.cost_tracking; "
                    "assert 'genai_prices' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestCostTrackingMiddleware:
    """Tests for cumulative cost writes on the model checkpoint path."""

    def test_cost_channel_is_private(self) -> None:
        hints = get_type_hints(CostState, include_extras=True)
        metadata = getattr(hints["_session_cost_usd"], "__metadata__", ())
        assert PrivateStateAttr in metadata

    def test_accumulates_onto_prior_checkpoint_total(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [HumanMessage("hello"), _message(_usage())],
            "_session_cost_usd": 1.25,
        }
        result = middleware.after_model(state, _runtime())
        assert result is not None
        assert result["_session_cost_usd"] > 1.25

    def test_overflowing_prior_total_is_treated_as_zero(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage())],
            "_session_cost_usd": 10**1000,
        }

        result = middleware.after_model(state, _runtime())

        assert result is not None
        assert 0 < result["_session_cost_usd"] < 1

    def test_uses_persisted_model_spec_when_message_metadata_is_absent(self) -> None:
        middleware = CostTrackingMiddleware()
        message = AIMessage(
            content="response",
            usage_metadata=_usage(),  # ty: ignore[invalid-argument-type]
        )
        state: CostState = {
            "messages": [message],
            "_model_spec": f"{KNOWN_PROVIDER}:{KNOWN_MODEL}",
        }
        result = middleware.after_model(state, _runtime())
        assert result is not None
        assert result["_session_cost_usd"] > 0

    def test_codex_model_spec_overrides_generic_openai_message_metadata(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [_message(_usage(), model="gpt-5.4", provider="openai")],
            "_model_spec": "openai_codex:gpt-5.4",
            "_session_cost_usd": 2.5,
        }

        assert middleware.after_model(state, _runtime()) is None

    def test_unpriceable_model_leaves_prior_total_unchanged(self) -> None:
        middleware = CostTrackingMiddleware()
        state: CostState = {
            "messages": [
                _message(
                    _usage(),
                    model="definitely-not-a-real-model",
                    provider="unknown-provider",
                )
            ],
            "_session_cost_usd": 2.5,
        }
        assert middleware.after_model(state, _runtime()) is None

    @pytest.mark.parametrize(
        "messages",
        [[], [HumanMessage("hello")], [AIMessage(content="no usage")]],
    )
    def test_no_priceable_message_is_a_noop(self, messages: list[Any]) -> None:
        middleware = CostTrackingMiddleware()
        state = cast("CostState", {"messages": messages})
        assert middleware.after_model(state, _runtime()) is None
