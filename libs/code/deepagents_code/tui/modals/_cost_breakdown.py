"""Markdown formatting for the token and cost breakdown modal."""

import math
from collections.abc import Mapping

_SMALL_COST = 0.0001
"""Smallest cost the four-decimal table format can show without rounding to 0."""

_CATEGORIES = (
    ("**Input**", "input"),
    ("cache creation", "cache_creation"),
    ("cache read", "cache_read"),
    ("**Output**", "output"),
    ("reasoning", "reasoning"),
)
"""Table label and breakdown key prefix for each priced row, in display order."""

_UNPRICED_CATEGORIES = (
    ("**Unpriced input**", "input"),
    ("**Unpriced output**", "output"),
)
"""Rows that hold usage from requests with no price data, when it is separable."""


def _format_usd(value: float) -> str:
    """Format a table cost with fixed precision and no scientific notation.

    Returns:
        A four-decimal amount, or `<0.0001` for a positive amount below that.
    """
    return "<0.0001" if 0 < value < _SMALL_COST else f"{value:.4f}"


class _BreakdownView:
    """Validated reads from an untrusted breakdown mapping."""

    def __init__(self, breakdown: Mapping[str, object], total_usd: float) -> None:
        self._data = breakdown
        self._total_usd = total_usd
        self.has_unpriced = breakdown.get("priced_request_count") != breakdown.get(
            "request_count"
        )
        # Only breakdowns that recorded unpriced usage separately can keep it
        # out of the priced rows. Older checkpoints fall back to a marker.
        self.split = (
            self.has_unpriced and breakdown.get("unpriced_tokens_known") is True
        )

    def number(self, key: str) -> float | None:
        value = self._data.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None

    def count(self, key: str) -> int | None:
        value = self._data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
        return value

    def complete(self, key: str) -> bool:
        return self._data.get(key) is True

    def priced_tokens(self, category: str) -> str:
        """Tokens for one category, excluding unpriced usage when possible.

        Returns:
            The formatted count, with a marker when it is partial or mixed.
        """
        value = self.count(f"{category}_tokens")
        if value is None:
            return "unavailable"
        suffix = ""
        if self.split:
            value = max(value - (self.count(f"unpriced_{category}_tokens") or 0), 0)
        elif self.has_unpriced:
            suffix = " (incl. unpriced)"
        if not self.complete(f"{category}_tokens_complete"):
            suffix = " (partial)"
        return f"{value:,}{suffix}"

    def unpriced_tokens(self, category: str) -> str:
        value = self.count(f"unpriced_{category}_tokens") or 0
        partial = not self.complete(f"{category}_tokens_complete")
        return f"{value:,}{' (partial)' if partial else ''}"

    def cost(self, category: str) -> str:
        """Cost for one category, marked partial when it misses some usage.

        Returns:
            The formatted cost, or `unavailable` when it is invalid.
        """
        value = self.number(f"{category}_cost_usd")
        if value is None:
            return "unavailable"
        flags = self._data.get("priced_cost_complete")
        complete = (
            flags.get(category) is True
            if self.split and isinstance(flags, Mapping)
            else self.complete(f"{category}_cost_complete")
        )
        if complete:
            return _format_usd(value)
        return f"{_format_usd(value)} (partial)"

    def percent(self, category: str) -> str:
        value = self.number(f"{category}_cost_usd")
        if value is None or self._total_usd <= 0:
            return "n/a"
        return f"{value / self._total_usd * 100:.1f}%"


_Row = tuple[str, str, str, str]
"""Category, % cost, tokens, and cost cells for one table row."""


def _rows(view: _BreakdownView, total_usd: float) -> list[_Row]:
    """Build the table rows in display order.

    Returns:
        One tuple of cell text per row.
    """
    rows: list[_Row] = [
        (label, view.percent(key), view.priced_tokens(key), view.cost(key))
        for label, key in _CATEGORIES
    ]
    if view.split:
        rows.extend(
            (label, "n/a", view.unpriced_tokens(key), "no price data")
            for label, key in _UNPRICED_CATEGORIES
        )
    total_tokens = int(view.number("input_tokens") or 0) + int(
        view.number("output_tokens") or 0
    )
    if view.has_unpriced:
        total_percent, total_cost = "n/a", f"{_format_usd(total_usd)} (partial)"
    else:
        total_percent = "100%" if total_usd > 0 else "n/a"
        total_cost = _format_usd(total_usd)
    rows.append(("**Total**", total_percent, f"{total_tokens:,}", total_cost))
    return rows


def _notes(view: _BreakdownView, total_usd: float) -> list[str]:
    """Explain how to read the table and what it leaves out.

    Returns:
        Note paragraphs to print below the table.
    """
    notes = [
        (
            "Input and Output are inclusive; cache and reasoning rows are subsets. "
            "% cost is each row's share of the priced cost."
        )
    ]
    if view.split:
        notes.append(
            "Unpriced rows hold usage from requests with no price data; "
            "the rows above exclude it."
        )
    attributed = (view.number("input_cost_usd") or 0.0) + (
        view.number("output_cost_usd") or 0.0
    )
    if not math.isclose(attributed, total_usd, rel_tol=1e-12, abs_tol=1e-15):
        notes.append(
            f"Partial attribution: {_format_usd(max(total_usd - attributed, 0.0))} "
            "USD is directionless/unattributed."
        )
    if view.has_unpriced:
        priced = view.count("priced_request_count")
        requests = view.count("request_count")
        counts = (
            "Some requests"
            if priced is None or requests is None
            else f"{priced} of {requests} requests"
        )
        notes.append(
            f"{counts} in this thread, including side questions and thread "
            "names, had price data; costs are partial."
        )
    return notes


def format_cost_breakdown_table(
    total_usd: float, breakdown: Mapping[str, object] | None
) -> str:
    """Build the copyable entire-thread estimated token/cost table.

    Returns:
        A Markdown table, or an empty string when historical detail is missing.
    """
    if (
        not isinstance(breakdown, Mapping)
        or breakdown.get("version") != 1
        or breakdown.get("historical_complete") is not True
    ):
        return ""
    view = _BreakdownView(breakdown, total_usd)
    rendered = [
        "| Category | % cost | Tokens | Cost (USD) |",
        "| :--- | ---: | ---: | ---: |",
        *("| " + " | ".join(row) + " |" for row in _rows(view, total_usd)),
    ]
    return (
        "Entire-thread estimated breakdown\n\n"
        + "\n".join(rendered)
        + "\n\n"
        + "\n\n".join(_notes(view, total_usd))
    )
