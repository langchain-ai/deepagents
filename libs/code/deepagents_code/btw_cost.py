"""Persist side-question spend separately from the running graph's checkpoints.

The sessions database owns this subtotal. Readers add it to the graph's total;
it is never fed back into the graph's cost recorder or checkpoint channels.
Failed writes remain owned in memory until a later settlement or cost read
retries them. Only successfully persisted charges survive a server restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections import deque
from contextlib import closing
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

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

    import aiosqlite


_PENDING_COSTS: dict[str, deque[tuple[_SessionCostRecorder, CostState]]] = {}
_SETTLEMENT_LOCK = threading.Lock()
"""Serialize retries and retain failed recorders until their writes succeed."""

logger = logging.getLogger(__name__)


class SessionCost(TypedDict):
    """Server-owned presentation total, separate from graph checkpoint values.

    Components let clients refresh side spend while preserving live graph costs.
    They are optional for compatibility with older servers' combined totals.
    """

    total: float
    breakdown: CostBreakdown | None
    graph_total: NotRequired[float]
    graph_breakdown: NotRequired[CostBreakdown | None]
    side_breakdown: NotRequired[CostBreakdown | None]


def combine_session_cost(
    graph_total: float,
    graph_breakdown: CostBreakdown | None,
    side_breakdown: CostBreakdown | None,
) -> SessionCost:
    """Combine independent spend while retaining its sources for live refreshes.

    Args:
        graph_total: Main graph's cumulative spend, possibly uncommitted.
        graph_breakdown: Main graph's structured usage.
        side_breakdown: Separately persisted side-question usage.

    Returns:
        Presentation total with the independent accounting components.
    """
    return {
        "total": graph_total
        + (side_breakdown["total_cost_usd"] if side_breakdown else 0),
        "breakdown": (
            _merge_cost_breakdowns(graph_breakdown, side_breakdown)
            if side_breakdown is not None
            else graph_breakdown
        ),
        "graph_total": graph_total,
        "graph_breakdown": graph_breakdown,
        "side_breakdown": side_breakdown,
    }


async def delete_cost(conn: aiosqlite.Connection, thread_id: str) -> None:
    """Erase spend in the caller's thread-deletion transaction.

    Retain only an ID tombstone to discard pending retries and late provider
    completions, including answers that have not written their first charge.

    Args:
        conn: Sessions connection whose transaction owns thread deletion.
        thread_id: Thread whose charges must be erased.
    """
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS dcode_btw_costs "
        "(thread_id TEXT PRIMARY KEY NOT NULL, breakdown TEXT NOT NULL)"
    )
    await conn.execute(
        "INSERT INTO dcode_btw_costs VALUES (?, 'null') "
        "ON CONFLICT(thread_id) DO UPDATE SET breakdown = 'null'",
        (thread_id,),
    )


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
    """Retry pending settlements and read the durable subtotal.

    Args:
        thread_id: Thread whose side questions were charged.

    Returns:
        The saved breakdown, or `None` when no side usage has been saved.
    """
    with _SETTLEMENT_LOCK:
        _retry_pending_costs(thread_id)
        return _load_saved_cost(thread_id)


def _load_saved_cost(thread_id: str) -> CostBreakdown | None:
    """Read without taking the settlement lock or retrying writes.

    Returns:
        The persisted subtotal, or `None` when none has been saved.
    """
    path = _database_path()
    if not path.exists():
        return None
    with closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=0.05)
    ) as conn:
        return _read_cost(conn, thread_id)


def session_cost(values: Mapping[str, Any], thread_id: str) -> SessionCost:
    """Read the combined total without modifying or settling graph state.

    Live graph events use this read-only path so a pending side settlement
    cannot delay a main turn. Explicit cost reads retry settlements separately.

    Args:
        values: Graph-owned cost values, optionally including an uncommitted delta.
        thread_id: Thread that owns any separately persisted side spend.

    Returns:
        Combined cost and usage for presentation only.
    """
    cost = _load_saved_cost(thread_id) if thread_id else None
    total = values.get("_session_cost_usd", 0.0)
    breakdown = values.get("_session_cost_breakdown")
    return combine_session_cost(total, breakdown, cost)


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
            # Deletion leaves only an ID tombstone. It prevents a late provider
            # completion or an in-memory retry from resurrecting deleted spend.
            deleted = conn.execute(
                "SELECT 1 FROM dcode_btw_costs "
                "WHERE thread_id = ? AND breakdown = 'null'",
                (thread_id,),
            ).fetchone()
            if deleted:
                prepared.commit()
                return None
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


def _retry_pending_costs(thread_id: str) -> CostBreakdown | None:
    """Drain a thread's queue under `_SETTLEMENT_LOCK`, keeping failures owned.

    Returns:
        The latest persisted subtotal, or `None` when no usage was written.
    """
    pending = _PENDING_COSTS.get(thread_id)
    total = None
    while pending:
        recorder, state = pending[0]
        token = _RECORDER_VAR.set(recorder)
        try:
            persisted = _persist_cost(thread_id, state)
        finally:
            _RECORDER_VAR.reset(token)
        pending.popleft()
        if persisted is not None:
            total = persisted
    _PENDING_COSTS.pop(thread_id, None)
    return total


def _settle_cost(
    thread_id: str, state: CostState, recorder: _SessionCostRecorder
) -> CostBreakdown | None:
    """Transfer ownership before writing so a failed request remains retryable.

    Returns:
        The latest persisted subtotal, or `None` when no usage was written.
    """
    with _SETTLEMENT_LOCK:
        _PENDING_COSTS.setdefault(thread_id, deque()).append((recorder, state))
        return _retry_pending_costs(thread_id)


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

    recorder = _SessionCostRecorder()
    token = _RECORDER_VAR.set(recorder)
    try:
        try:
            text = await answer
        finally:
            # A disconnect can arrive after the provider completed. Finish the
            # database write even then, but allow cancellation during generation.
            settlement = asyncio.create_task(
                asyncio.to_thread(_settle_cost, thread_id, state, recorder)
            )
            cancellation = await _join_task_deferring_cancellation(settlement)
            try:
                total = settlement.result()
            except Exception:
                logger.warning(
                    "Could not save side-question costs; settlement remains pending",
                    exc_info=True,
                )
                total = None
            if cancellation is not None:
                raise cancellation
        return text, total
    finally:
        _RECORDER_VAR.reset(token)
