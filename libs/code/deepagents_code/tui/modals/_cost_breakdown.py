"""Plain-text formatting for the token and cost breakdown modal."""

import math
from collections.abc import Mapping


def format_cost_breakdown_table(
    total_usd: float, breakdown: Mapping[str, object] | None
) -> str:
    """Build the copyable entire-thread estimated token/cost table.

    Returns:
        A plain-text table, or an empty string when historical detail is missing.
    """
    if (
        not isinstance(breakdown, Mapping)
        or breakdown.get("version") != 1
        or breakdown.get("historical_complete") is not True
    ):
        return ""

    def _number(key: str) -> float | None:
        value = breakdown.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None

    def _tokens(key: str, complete_key: str | None = None) -> str:
        value = breakdown.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return "unavailable"
        if complete_key and breakdown.get(complete_key) is not True:
            return f"{value} (partial)"
        return str(value)

    def _cost(key: str, complete_key: str | None = None) -> str:
        value = _number(key)
        if value is None:
            return "unavailable"
        text = repr(value)
        if complete_key and breakdown.get(complete_key) is not True:
            return f"{text} (partial)"
        return text

    def _percent(key: str) -> str:
        value = _number(key)
        if value is None or total_usd <= 0:
            return "n/a"
        return f"{value / total_usd * 100:.6g}%"

    rows = [
        (
            "Input",
            _percent("input_cost_usd"),
            _tokens("input_tokens", "input_tokens_complete"),
            _cost("input_cost_usd", "input_cost_complete"),
        ),
        (
            "  cache creation",
            _percent("cache_creation_cost_usd"),
            _tokens("cache_creation_tokens", "cache_creation_tokens_complete"),
            _cost("cache_creation_cost_usd", "cache_creation_cost_complete"),
        ),
        (
            "  cache read",
            _percent("cache_read_cost_usd"),
            _tokens("cache_read_tokens", "cache_read_tokens_complete"),
            _cost("cache_read_cost_usd", "cache_read_cost_complete"),
        ),
        (
            "Output",
            _percent("output_cost_usd"),
            _tokens("output_tokens", "output_tokens_complete"),
            _cost("output_cost_usd", "output_cost_complete"),
        ),
        (
            "  reasoning",
            _percent("reasoning_cost_usd"),
            _tokens("reasoning_tokens", "reasoning_tokens_complete"),
            _cost("reasoning_cost_usd", "reasoning_cost_complete"),
        ),
        (
            "Total",
            "100%" if total_usd > 0 else "n/a",
            str(int(_number("input_tokens") or 0) + int(_number("output_tokens") or 0)),
            repr(total_usd),
        ),
    ]
    widths = [
        max(
            len(row[index])
            for row in [("Category", "% cost", "Tokens", "Cost (USD)"), *rows]
        )
        for index in range(4)
    ]
    rendered = [
        "  ".join(
            value.ljust(widths[index])
            for index, value in enumerate(
                ("Category", "% cost", "Tokens", "Cost (USD)")
            )
        ).rstrip()
    ]
    rendered.append("  ".join("-" * width for width in widths))
    rendered.extend(
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()
        for row in rows
    )
    notes: list[str] = ["Parent rows are inclusive; indented rows are subsets."]
    attributed = (_number("input_cost_usd") or 0.0) + (
        _number("output_cost_usd") or 0.0
    )
    if not math.isclose(attributed, total_usd, rel_tol=1e-12, abs_tol=1e-15):
        notes.append(
            f"Partial attribution: {max(total_usd - attributed, 0.0)!r} USD is "
            "directionless/unattributed."
        )
    if breakdown.get("priced_request_count") != breakdown.get("request_count"):
        notes.append("Some requests were unpriceable; costs are partial.")
    return "Entire-thread estimated breakdown\n" + "\n".join(rendered + notes)
