"""Combine independently reported main-task and side-question usage for display."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

if TYPE_CHECKING:
    from deepagents_code.cost_tracking import CostBreakdown


class SessionCost(TypedDict):
    """Presentation total derived from the latest independent subtotals."""

    total: float
    """Combined main-task and side-question spend in USD."""

    breakdown: CostBreakdown | None
    """Combined usage and costs from available source breakdowns, if any."""

    cached: NotRequired[bool]
    """No fresh graph checkpoint was available to settle provisional usage."""


@dataclass
class SessionCostTracker:
    """Retain each source independently so delayed updates cannot erase spend."""

    graph_total: float | None = None
    """Highest accepted main-task spend in USD, or `None` before usage arrives."""

    graph_breakdown: CostBreakdown | None = None
    """Usage and costs accompanying the retained main-task total, if supplied."""

    side_breakdown: CostBreakdown | None = None
    """Newest cumulative side-question usage and costs, if reported."""

    def update_graph(self, total: object, breakdown: object) -> bool:
        """Adopt a main-task total without replacing newer streamed usage.

        Args:
            total: Main-task spend from a checkpoint or stream event.
            breakdown: Structured main-task usage, when available.

        Returns:
            Whether the source supplied a valid main-task total.
        """
        if isinstance(total, bool) or not isinstance(total, int | float):
            return False
        if not math.isfinite(total):
            return False
        if self.graph_total is None or total >= self.graph_total:
            self.graph_total = max(float(total), 0.0)
            self.graph_breakdown = (
                cast("CostBreakdown", breakdown)
                if isinstance(breakdown, Mapping)
                else None
            )
        return True

    def update_side(self, incoming: CostBreakdown | None) -> None:
        """Keep the newest cumulative side usage, including free requests.

        Args:
            incoming: Persisted side-question subtotal from an HTTP response.
        """
        if incoming is None:
            return
        incoming_key = (incoming["request_count"], incoming["total_cost_usd"])
        current = self.side_breakdown
        if current is None or incoming_key >= (
            current["request_count"],
            current["total_cost_usd"],
        ):
            self.side_breakdown = incoming

    def snapshot(self, *, cached: bool = False) -> SessionCost | None:
        """Combine subtotals only when producing a presentation value.

        Args:
            cached: Preserve provisional usage because graph state was unavailable.

        Returns:
            Display totals, or `None` when neither source has reported usage.
        """
        from deepagents_code.cost_tracking import _merge_cost_breakdowns

        if self.graph_total is None and self.side_breakdown is None:
            return None
        side = self.side_breakdown
        return {
            "total": (self.graph_total or 0.0)
            + (side["total_cost_usd"] if side else 0),
            "breakdown": (
                _merge_cost_breakdowns(self.graph_breakdown, side)
                if side is not None
                else self.graph_breakdown
            ),
            "cached": cached,
        }
