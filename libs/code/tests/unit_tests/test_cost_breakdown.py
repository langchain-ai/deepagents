"""Tests for the copyable cost breakdown presentation."""

from deepagents_code.cost_tracking import _empty_cost_breakdown
from deepagents_code.tui.modals.cost_breakdown import format_cost_breakdown_table


def test_readable_markdown_preserves_small_costs_and_partial_notes() -> None:
    breakdown = _empty_cost_breakdown()
    breakdown.update(
        input_tokens=123456,
        input_tokens_complete=False,
        input_cost_usd=0.000000123,
        input_cost_complete=False,
        request_count=2,
        priced_request_count=1,
    )
    table = format_cost_breakdown_table(0.001, breakdown)

    assert "| Category | % cost | Tokens | Cost (USD) |" in table
    assert "| **Input** | 0.0% | 123,456 (partial) | 1.23e-07 (partial) |" in table
    assert "| **Total** | 100% | 123,456 | 0.0010 |" in table
    assert "Partial attribution: 0.0010 USD is directionless/unattributed." in table
    assert "Some requests were unpriceable; costs are partial." in table


def test_zero_cost_and_unavailable_details() -> None:
    breakdown: dict[str, object] = dict(_empty_cost_breakdown())
    breakdown.update(cache_read_tokens=None, cache_read_cost_usd=None)
    table = format_cost_breakdown_table(0, breakdown)

    assert "| cache read | n/a | unavailable | unavailable |" in table
    assert "| **Total** | n/a | 0 | 0.0000 |" in table
    assert format_cost_breakdown_table(0, None) == ""
    breakdown["historical_complete"] = False
    assert format_cost_breakdown_table(0, breakdown) == ""
