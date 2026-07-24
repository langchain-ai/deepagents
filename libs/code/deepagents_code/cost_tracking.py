"""Estimate and persist cumulative model cost for each thread.

This module uses a dedicated `CostTrackingMiddleware` rather than extending
`ResumeStateMiddleware`. The separate middleware keeps resume-state token and
model bookkeeping focused while still writing `_session_cost_usd` from inside
the graph, so each cost update rides the model checkpoint and works for local,
headless, and remote graph execution without a client-side state update.

Every caller uses `estimate_cost`, the only function that imports or calls
`genai-prices`. The import is lazy so the package and its bundled pricing data
stay off the CLI startup path. Unsupported models and malformed usage return
`None`; pricing must never interrupt a model turn.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage

from deepagents_code.resume_state import ResumeState

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_PROVIDER_ALIASES: dict[str, str] = {
    "azure_openai": "azure",
    "bedrock": "aws",
    "google_genai": "google",
    "google_vertexai": "google",
    "mistralai": "mistral",
    "xai": "x-ai",
}
"""Map LangChain provider names to the identifiers used by `genai-prices`."""

_UNPRICEABLE_PROVIDERS: frozenset[str] = frozenset({"openai_codex"})
"""Providers whose access model is not equivalent to per-token API billing."""


def _token_count(value: object) -> int:
    """Return a non-negative integer token count for a metadata value."""
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _cache_write_tokens(details: Mapping[str, Any]) -> int:
    """Return cache-write tokens from LangChain `input_token_details`.

    LangChain Anthropic zeroes the generic `cache_creation` field when the
    response includes a TTL breakdown (`ephemeral_5m_input_tokens` /
    `ephemeral_1h_input_tokens`). Sum those detailed fields when present so
    tokens are priced as cache writes rather than ordinary input. Fall back to
    `cache_creation` or the `cache_write` alias used by some other providers.
    `genai-prices` exposes a single cache-write rate, so 5-minute and 1-hour
    writes share that catalog price.
    """
    detailed = _token_count(details.get("ephemeral_5m_input_tokens")) + _token_count(
        details.get("ephemeral_1h_input_tokens")
    )
    if detailed:
        return detailed
    return _token_count(details.get("cache_creation") or details.get("cache_write"))


def _coerce_cost_usd(value: object) -> float:
    """Coerce a cumulative cost to a finite non-negative float.

    Returns:
        The valid cost, or `0.0` for malformed and overflowing values.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    try:
        cost_usd = float(value)
    except (OverflowError, ValueError):
        return 0.0
    return cost_usd if math.isfinite(cost_usd) and cost_usd >= 0 else 0.0


def estimate_cost(
    usage_metadata: Mapping[str, Any] | None,
    model_name: str,
    provider: str = "",
) -> float | None:
    """Estimate one model request's cost in USD from LangChain usage metadata.

    LangChain's `input_tokens` is the full input count, including cache reads and
    writes. `genai-prices` receives that inclusive total plus the two cache
    buckets; it subtracts the cache buckets before applying the ordinary input
    rate, then prices reads and writes separately so tokens are not double-counted.

    Args:
        usage_metadata: The request's LangChain `usage_metadata` mapping.
        model_name: Model identifier used for the request.
        provider: LangChain provider identifier. An empty value lets
            `genai-prices` infer the provider from `model_name`.

    Returns:
        Estimated cost in USD, or `None` when usage or pricing is unavailable.
    """
    model_ref = model_name.strip()
    provider_key = provider.strip().lower()
    if not usage_metadata or not model_ref:
        return None
    if provider_key in _UNPRICEABLE_PROVIDERS:
        logger.debug(
            "Cost estimate unavailable for non-API provider=%r model=%r",
            provider,
            model_ref,
        )
        return None

    input_tokens = _token_count(usage_metadata.get("input_tokens"))
    output_tokens = _token_count(usage_metadata.get("output_tokens"))
    if not input_tokens and not output_tokens:
        # `total_tokens` combines input and output, which normally have different
        # rates. Without the split there is no defensible estimate.
        return None

    details = usage_metadata.get("input_token_details")
    if isinstance(details, Mapping):
        cache_read_tokens = _token_count(details.get("cache_read"))
        cache_write_tokens = _cache_write_tokens(details)
    else:
        cache_read_tokens = 0
        cache_write_tokens = 0

    # Provider metadata can occasionally report cache parts that exceed the
    # inclusive input total. Clamp the parts so pricing still produces a safe
    # estimate instead of failing the model turn with negative uncached input.
    cache_read_tokens = min(cache_read_tokens, input_tokens)
    cache_write_tokens = min(cache_write_tokens, input_tokens - cache_read_tokens)

    provider_id = _PROVIDER_ALIASES.get(provider_key, provider_key) or None
    try:
        from genai_prices import Usage, calc_price

        price = calc_price(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens or None,
                cache_write_tokens=cache_write_tokens or None,
            ),
            model_ref=model_ref,
            provider_id=provider_id,
        )
        cost_usd = float(price.total_price)
    except Exception:
        logger.debug(
            "Cost estimate unavailable for model=%r provider=%r",
            model_ref,
            provider,
            exc_info=True,
        )
        return None

    return cost_usd if math.isfinite(cost_usd) and cost_usd >= 0 else None


def resolve_message_model(
    message: object,
    *,
    fallback_model: str = "",
    fallback_provider: str = "",
) -> tuple[str, str]:
    """Resolve the model and provider attached to a streamed model message.

    Args:
        message: An AI message or chunk with optional `response_metadata`.
        fallback_model: Model to use when message metadata does not name one.
        fallback_provider: Provider to use when message metadata does not name one.
            Known provider aliases and non-API providers override generic
            response metadata.

    Returns:
        The `(model_name, provider)` pair used for pricing.
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    model_name = metadata.get("model_name") or metadata.get("model") or fallback_model
    provider = (
        metadata.get("model_provider") or metadata.get("provider") or fallback_provider
    )
    resolved_model = model_name if isinstance(model_name, str) else fallback_model
    resolved_provider = provider if isinstance(provider, str) else fallback_provider
    fallback_provider_key = fallback_provider.strip().lower()
    if fallback_provider_key in _PROVIDER_ALIASES or (
        fallback_provider_key in _UNPRICEABLE_PROVIDERS
    ):
        resolved_provider = fallback_provider
    return resolved_model, resolved_provider


class CostState(ResumeState):
    """Agent state extended with the private cumulative thread-cost channel."""

    _session_cost_usd: Annotated[NotRequired[float], PrivateStateAttr]
    """Cumulative estimated USD cost for all priceable calls in this thread."""


class CostTrackingMiddleware(AgentMiddleware[CostState, ContextT]):
    """Accumulate priceable model usage into `_session_cost_usd` checkpoints."""

    state_schema = CostState

    def after_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: CostState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Add the latest model response's estimate to the prior thread total.

        Args:
            state: Current state containing messages and prior session cost.
            runtime: LangGraph runtime required by the middleware interface.

        Returns:
            The new cumulative cost, or `None` when the latest response cannot be
            priced. Returning `None` leaves the prior checkpoint value unchanged.
        """
        for message in reversed(state.get("messages") or []):
            if not isinstance(message, AIMessage):
                continue

            fallback_model = ""
            fallback_provider = ""
            model_spec = state.get("_model_spec")
            if isinstance(model_spec, str) and model_spec:
                fallback_provider, separator, fallback_model = model_spec.partition(":")
                if not separator:
                    fallback_model = fallback_provider
                    fallback_provider = ""

            model_name, provider = resolve_message_model(
                message,
                fallback_model=fallback_model,
                fallback_provider=fallback_provider,
            )
            if not model_name:
                from deepagents_code.config import settings

                model_name = settings.model_name or ""
                provider = provider or settings.model_provider or ""

            cost_usd = estimate_cost(
                getattr(message, "usage_metadata", None),
                model_name,
                provider,
            )
            if cost_usd is None:
                return None

            prior_cost_usd = _coerce_cost_usd(state.get("_session_cost_usd"))
            return {"_session_cost_usd": prior_cost_usd + cost_usd}

        return None
