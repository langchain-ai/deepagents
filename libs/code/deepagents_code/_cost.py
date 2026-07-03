"""LLM cost estimation backed by `genai-prices`.

Isolated from `_session_stats.py` so the latter stays free of heavy imports (it
is imported at module level by `app.py`). `genai_prices` is imported lazily
inside `estimate_request_cost` so merely importing this module stays cheap and
off the startup hot path.

Prices are indicative only — see the `genai-prices` project for caveats. When a
model or provider is not found in the bundled price data, `estimate_request_cost`
returns `None` and callers leave the running total unchanged rather than crash.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


def cache_tokens_from_usage(usage: Mapping | None) -> tuple[int, int]:
    """Extract cache read/write token counts from LangChain `usage_metadata`.

    Args:
        usage: A message's `usage_metadata` mapping, or `None`.

    Returns:
        A `(cache_read_tokens, cache_write_tokens)` tuple. Missing details
        yield zeros. `cache_write_tokens` maps to `cache_creation` in
        LangChain's `input_token_details`.
    """
    if not usage:
        return 0, 0
    details = usage.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0) or 0
    cache_write = details.get("cache_creation", 0) or 0
    return cache_read, cache_write


def estimate_request_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    model_name: str,
    provider: str = "",
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimate the USD cost of a single LLM request.

    Cached tokens are billed at different rates than uncached input (cache reads
    are cheaper; cache writes carry a premium), so they are priced separately by
    `genai-prices`. Both LangChain's `usage_metadata` and `genai-prices` treat
    `input_tokens` as the grand total of input — inclusive of any cached
    portion — and `genai-prices` subtracts the cache counts internally to derive
    the uncached (full-rate) input. The totals are therefore forwarded as-is.

    Args:
        input_tokens: Total input (prompt) tokens for the request, inclusive of
            any cached tokens per LangChain's `usage_metadata` convention.
        output_tokens: Output (completion) tokens for the request. Reasoning
            tokens are already included here by LangChain convention.
        model_name: Model that served the request (e.g. `claude-sonnet-4-5`).
        provider: Provider id that served the model (e.g. `anthropic`). When
            empty, `genai-prices` infers the provider from the model name.
        cache_read_tokens: Tokens read from the prompt cache (billed at a
            reduced rate). Sourced from `input_token_details.cache_read`.
        cache_write_tokens: Tokens written to the prompt cache (billed at a
            premium). Sourced from `input_token_details.cache_creation`.

    Returns:
        The estimated total price in USD, or `None` when the model/provider is
        not in the bundled price data (or pricing otherwise fails). Returning
        `None` lets callers skip the request without disturbing a running total.
    """
    if not model_name:
        return None
    if input_tokens <= 0 and output_tokens <= 0:
        return None

    # `genai-prices` treats `input_tokens` as the grand total and subtracts the
    # cache counts internally to derive uncached input, rejecting a negative
    # result. Clamp the cache counts to `input_tokens` to stay robust to
    # inconsistent provider reporting where the parts exceed the whole.
    if input_tokens > 0:
        cache_read_tokens = min(cache_read_tokens, input_tokens)
        cache_write_tokens = min(cache_write_tokens, input_tokens - cache_read_tokens)

    try:
        from genai_prices import Usage, calc_price

        price = calc_price(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens or None,
                cache_write_tokens=cache_write_tokens or None,
            ),
            model_ref=model_name,
            provider_id=provider or None,
        )
    except LookupError:
        # Model or provider not present in the bundled price data. Expected for
        # newer/self-hosted models; degrade quietly.
        logger.debug(
            "No price data for model=%s provider=%s; skipping cost estimate",
            model_name,
            provider,
        )
        return None
    except Exception:
        # Never let a pricing failure interrupt agent streaming.
        logger.warning(
            "Cost estimation failed for model=%s provider=%s",
            model_name,
            provider,
            exc_info=True,
        )
        return None

    return float(price.total_price)


def format_cost(cost: float) -> str:
    """Format a USD cost into a compact status-bar string.

    Uses more decimal places for small amounts so sub-cent sessions still show a
    meaningful value instead of `$0.00`.

    Args:
        cost: Cost in USD.

    Returns:
        Formatted string like `'$0.0042'`, `'$1.23'`, or `'$0'`.
    """
    if cost <= 0:
        return "$0"
    if cost < 1:  # sub-dollar amounts get extra precision
        return f"${cost:.4f}"
    return f"${cost:.2f}"
