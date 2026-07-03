"""Estimate and accumulate the USD cost of model calls.

Cost is priced with [`genai-prices`](https://github.com/pydantic/genai-prices),
which ships bundled offline price data for many providers and understands
Anthropic cache tokens. Two consumers share the single `estimate_cost` helper so
they can never drift:

- `CostTrackingMiddleware` runs *inside the graph* and accumulates cumulative
  USD into the private `_session_cost_usd` checkpoint channel. Writing from the
  model node (rather than a client-side `aupdate_state`) rides the same
  checkpoint as the response, so cost is persisted, restored on resume, works in
  headless / non-interactive mode, and behaves identically against remote
  graphs — the same rationale as `ResumeStateMiddleware` for `_context_tokens`.
- The TUI client (`textual_adapter.py`) calls `estimate_cost` directly on the
  `usage_metadata` it already pulls off the stream, updating the live status-bar
  cost with no extra state round-trip.

Unknown / unpriceable models degrade to `None` (no cost recorded) rather than
raising, so a pricing gap never breaks a turn.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from decimal import Decimal

    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


def estimate_cost(
    usage_metadata: dict[str, Any] | None,
    model_name: str,
    provider: str = "",
) -> Decimal | None:
    """Estimate the USD cost of a single model call from its token usage.

    Args:
        usage_metadata: LangChain `AIMessage.usage_metadata`. Its `input_tokens`
            already includes cached tokens (matching what genai-prices expects);
            cache detail is read from `input_token_details`
            (`cache_read` / `cache_creation`).
        model_name: Model identifier to price (e.g. `claude-sonnet-4-5`).
        provider: Provider id (e.g. `anthropic`). An empty string lets
            genai-prices infer the provider from the model name.

    Returns:
        Total price as a `Decimal`, or `None` when usage/model is missing or
        genai-prices has no price for the model. Never raises — lookup failures
        are logged at debug and return `None`.
    """
    if not usage_metadata or not model_name:
        return None
    input_tokens = usage_metadata.get("input_tokens") or 0
    output_tokens = usage_metadata.get("output_tokens") or 0
    if not input_tokens and not output_tokens:
        return None
    details = usage_metadata.get("input_token_details") or {}
    try:
        from genai_prices import Usage, calc_price

        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=details.get("cache_read") or None,
            cache_write_tokens=details.get("cache_creation") or None,
        )
        price = calc_price(usage, model_ref=model_name, provider_id=provider or None)
    except Exception:
        logger.debug(
            "cost estimate failed for model=%r provider=%r",
            model_name,
            provider,
            exc_info=True,
        )
        return None
    return price.total_price


class CostState(AgentState):
    """Agent state extended with the cumulative session-cost channel."""

    _session_cost_usd: Annotated[NotRequired[float], PrivateStateAttr]
    """Cumulative estimated USD cost across all model calls in the thread.

    Written by `CostTrackingMiddleware.after_model`, read back by the CLI on
    resume to seed the status-bar cost. Stored as `float` (not `Decimal`) so it
    round-trips cleanly through checkpoint JSON serialization.
    """


def _model_ref_and_provider(msg: AIMessage, state: CostState) -> tuple[str, str]:
    """Resolve `(model_ref, provider)` for pricing a model response in-graph.

    Prefers the message's own `response_metadata`, then the `_model_spec`
    (`provider:model`) channel written by `ConfigurableModelMiddleware`, then the
    client `settings`. The returned provider may be `""`, in which case
    genai-prices infers it from the model name.

    Returns:
        A `(model_ref, provider)` pair; either element may be `""` when unknown.
    """
    meta = getattr(msg, "response_metadata", None) or {}
    model_ref = meta.get("model_name") or meta.get("model") or ""
    provider = meta.get("model_provider") or ""

    spec = state.get("_model_spec")
    if isinstance(spec, str) and ":" in spec:
        spec_provider, spec_model = spec.split(":", 1)
        model_ref = model_ref or spec_model
        provider = provider or spec_provider

    if not model_ref or not provider:
        from deepagents_code.config import settings

        model_ref = model_ref or (settings.model_name or "")
        provider = provider or (settings.model_provider or "")
    return model_ref, provider


class CostTrackingMiddleware(AgentMiddleware[CostState, ContextT]):
    """Accumulate estimated USD cost into the checkpoint after each model call.

    Mirrors `ResumeStateMiddleware.after_model`: the write happens inside the
    model node so it rides the response's checkpoint (persisted, resumable,
    remote-safe) instead of a standalone `aupdate_state`. Pricing goes through
    `estimate_cost`; unpriceable models simply contribute nothing to the total.
    """

    state_schema = CostState

    def after_model(  # noqa: PLR6301 — AgentMiddleware hook must be an instance method.
        self,
        state: CostState,
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Add the latest model call's estimated cost to `_session_cost_usd`.

        Args:
            state: Current agent state; the most recent `AIMessage` is priced.
            runtime: LangGraph runtime required by the middleware interface.

        Returns:
            State update with the new cumulative cost, or `None` when the latest
            call has no priceable usage.
        """
        for msg in reversed(state.get("messages") or []):
            if isinstance(msg, AIMessage):
                model_ref, provider = _model_ref_and_provider(msg, state)
                cost = estimate_cost(
                    getattr(msg, "usage_metadata", None), model_ref, provider
                )
                if cost is None:
                    return None
                current = state.get("_session_cost_usd") or 0.0
                return {"_session_cost_usd": current + float(cost)}
        return None
