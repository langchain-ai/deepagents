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

logger = logging.getLogger(__name__)


def estimate_request_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    model_name: str,
    provider: str = "",
) -> float | None:
    """Estimate the USD cost of a single LLM request.

    Args:
        input_tokens: Input (prompt) tokens for the request.
        output_tokens: Output (completion) tokens for the request.
        model_name: Model that served the request (e.g. `claude-sonnet-4-5`).
        provider: Provider id that served the model (e.g. `anthropic`). When
            empty, `genai-prices` infers the provider from the model name.

    Returns:
        The estimated total price in USD, or `None` when the model/provider is
        not in the bundled price data (or pricing otherwise fails). Returning
        `None` lets callers skip the request without disturbing a running total.
    """
    if not model_name:
        return None
    if input_tokens <= 0 and output_tokens <= 0:
        return None

    try:
        from genai_prices import Usage, calc_price

        price = calc_price(
            Usage(input_tokens=input_tokens, output_tokens=output_tokens),
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
