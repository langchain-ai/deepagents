"""Persist side-question spend separately from the running graph's checkpoints.

The sessions database owns this subtotal. Readers add it to the graph's total;
it is never fed back into the graph's cost recorder or checkpoint channels.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, Any, cast

from deepagents_code.cost_tracking import (
    _RECORDER_VAR,
    CostBreakdown,
    CostState,
    _merge_cost_breakdowns,
    _SessionCostRecorder,
    prepare_operation_cost,
)
from deepagents_code.workspace import _database_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping


def _read_cost(conn: sqlite3.Connection, thread_id: str) -> CostBreakdown | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'dcode_btw_costs'"
    ).fetchone()
    if not exists:
        return None
    row = conn.execute(
        "SELECT breakdown FROM dcode_btw_costs WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    return cast("CostBreakdown", json.loads(row[0])) if row else None


def load_cost(thread_id: str) -> CostBreakdown | None:
    """Read the durable subtotal, including after a server restart.

    Args:
        thread_id: Thread whose side questions were charged.

    Returns:
        The saved breakdown, or `None` when no side usage has been saved.
    """
    path = _database_path()
    if not path.exists():
        return None
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as conn:
        return _read_cost(conn, thread_id)


def include_cost(
    values: Mapping[str, Any], cost: CostBreakdown | None
) -> dict[str, Any]:
    """Combine persisted subtotals for presentation without changing graph state.

    Args:
        values: Original graph state values.
        cost: Independently persisted side-question subtotal.

    Returns:
        A state view with combined cost and usage totals.
    """
    if cost is None:
        return dict(values)
    return {
        **values,
        "_session_cost_usd": values.get("_session_cost_usd", 0.0)
        + cost["total_cost_usd"],
        "_session_cost_breakdown": _merge_cost_breakdowns(
            values.get("_session_cost_breakdown"), cost
        ),
    }


def _persist_cost(thread_id: str, state: CostState) -> CostBreakdown | None:
    prepared = prepare_operation_cost(state, thread_id)
    try:
        if not prepared.breakdown["request_count"]:
            prepared.commit()
            return None
        # A short SQLite transaction serializes side completions across server
        # processes without changing checkpoints or cancelling the main run.
        with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dcode_btw_costs "
                "(thread_id TEXT PRIMARY KEY NOT NULL, breakdown TEXT NOT NULL)"
            )
            previous = _read_cost(conn, thread_id)
            total = _merge_cost_breakdowns(previous, prepared.breakdown)
            conn.execute(
                "INSERT INTO dcode_btw_costs VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET breakdown = excluded.breakdown",
                (thread_id, json.dumps(total)),
            )
    except BaseException:
        prepared.rollback()
        raise
    else:
        prepared.commit()
        return total


async def answer_with_cost(
    answer: Awaitable[str],
    *,
    thread_id: str,
    state: CostState,
) -> tuple[str, CostBreakdown | None]:
    """Save completed usage before delivering an answer or finishing cancellation.

    Args:
        answer: Tool-free side-question generation.
        thread_id: Thread that owns this request.
        state: Checkpoint metadata used as a pricing fallback.

    Returns:
        Answer text and the persisted side-question subtotal, when available.
    """
    from deepagents_code.offload_api import _join_task_deferring_cancellation

    token = _RECORDER_VAR.set(_SessionCostRecorder())
    try:
        try:
            text = await answer
        finally:
            # A disconnect can arrive after the provider completed. Finish the
            # database write even then, but allow cancellation during generation.
            settlement = asyncio.create_task(
                asyncio.to_thread(_persist_cost, thread_id, state)
            )
            cancellation = await _join_task_deferring_cancellation(settlement)
            total = settlement.result()
            if cancellation is not None:
                raise cancellation
        return text, total
    finally:
        _RECORDER_VAR.reset(token)
