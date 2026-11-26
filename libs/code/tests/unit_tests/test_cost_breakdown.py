"""Tests for the copyable cost breakdown presentation."""

from typing import cast

from deepagents_code.cost_tracking import (
    CostBreakdown,
    _breakdown_for_estimate,
    _CostEstimate,
    _empty_cost_breakdown,
    _merge_cost_breakdowns,
)
from deepagents_code.tui.modals.cost_breakdown import format_cost_breakdown_table


def _priced_side_request() -> CostBreakdown:
    return _breakdown_for_estimate(
        _CostEstimate(
            total_cost_usd=0.00029,
            input_cost_usd=0.0002,
            output_cost_usd=0.00009,
            input_tokens=106,
            output_tokens=9,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            reasoning_tokens=0,
            cache_creation_cost_usd=0.0,
            cache_read_cost_usd=0.0,
            reasoning_cost_usd=0.0,
        )
    )


def _unpriced_main_request() -> CostBreakdown:
    return _breakdown_for_estimate(
        None,
        usage_metadata={
            "input_tokens": 35_098,
            "output_tokens": 11,
            "input_token_details": {"cache_creation": 35_095},
            "output_token_details": {"reasoning": 0},
        },
        provider="openai",
    )


def test_unpriced_usage_stays_out_of_priced_rows() -> None:
    breakdown = _merge_cost_breakdowns(_unpriced_main_request(), _priced_side_request())
    table = format_cost_breakdown_table(0.00029, breakdown)

    assert "| **Input** | 69.0% | 106 | 0.0002 |" in table
    assert "| cache creation | 0.0% | 0 | 0.0000 |" in table
    assert "| **Output** | 31.0% | 9 | <0.0001 |" in table
    assert "| **Unpriced input** | n/a | 35,098 | no price data |" in table
    assert "| **Unpriced output** | n/a | 11 | no price data |" in table
    assert "| **Total** | n/a | 35,224 | 0.0003 (partial) |" in table
    assert "1 of 2 requests in this thread" in table


def test_breakdown_without_unpriced_counts_marks_mixed_tokens() -> None:
    """Breakdowns checkpointed before the split still render honestly."""
    legacy = cast(
        "CostBreakdown",
        {
            key: value
            for key, value in _unpriced_main_request().items()
            if not key.startswith("unpriced_")
        },
    )
    breakdown = _merge_cost_breakdowns(legacy, _priced_side_request())
    table = format_cost_breakdown_table(0.00029, breakdown)

    assert "| **Input** | 69.0% | 35,204 (incl. unpriced) | 0.0002 (partial) |" in table
    assert "Unpriced input" not in table


def test_fully_priced_breakdown_totals_one_hundred_percent() -> None:
    breakdown = _merge_cost_breakdowns(_priced_side_request(), None)
    table = format_cost_breakdown_table(0.00029, breakdown)

    assert "| **Total** | 100% | 115 | 0.0003 |" in table
    assert "Unpriced" not in table
    assert "costs are partial" not in table


def test_sub_display_costs_avoid_scientific_notation() -> None:
    breakdown = _empty_cost_breakdown()
    breakdown.update(input_cost_usd=0.00009, output_cost_usd=0.000000123)
    table = format_cost_breakdown_table(0.000090123, breakdown)

    assert "| **Input** | 99.9% | 0 | <0.0001 |" in table
    assert "| **Output** | 0.1% | 0 | <0.0001 |" in table
    assert "e-0" not in table


def test_zero_cost_and_unavailable_details() -> None:
    breakdown: dict[str, object] = dict(_empty_cost_breakdown())
    breakdown.update(cache_read_tokens=None, cache_read_cost_usd=None)
    table = format_cost_breakdown_table(0, breakdown)

    assert "| cache read | n/a | unavailable | unavailable |" in table
    assert "| **Total** | n/a | 0 | 0.0000 |" in table
    assert format_cost_breakdown_table(0, None) == ""
    breakdown["historical_complete"] = False
    assert format_cost_breakdown_table(0, breakdown) == ""
