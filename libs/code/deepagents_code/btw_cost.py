"""Persist side-question spend separately from the running graph's checkpoints.

The sessions database owns this subtotal. Readers add it to the graph's total;
it is never fed back into the graph's cost recorder or checkpoint channels.
Failed charges retain only usage and pricing metadata, never the conversation.
Once priced, a charge is retried without recalculating its price. Only
successfully persisted charges survive a server restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections import deque
from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from deepagents_code._async import _join_task_deferring_cancellation
from deepagents_code.cost_tracking import (
    _RECORDER_VAR,
    CostBreakdown,
    CostState,
    _checkpointed_model_spec,
    _has_legacy_cost_history,
    _merge_cost_breakdowns,
    _ModelCallRecord,
    _price_operation_records,
    _SessionCostRecorder,
)
from deepagents_code.workspace import _database_path

if TYPE_CHECKING:
    from collections.abc import Awaitable

    import aiosqlite


@dataclass(frozen=True, slots=True)
class _UnpricedCost:
    """Only usage and pricing metadata survive a failed pricing attempt."""

    records: list[_ModelCallRecord]
    fallback: tuple[str, str]
    historical_complete: bool


_PENDING_COSTS: dict[str, deque[_UnpricedCost | CostBreakdown]] = {}
_SETTLEMENT_LOCK = threading.Lock()
"""Serialize retries and keep each charge owned until its write succeeds."""

logger = logging.getLogger(__name__)


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


def _persist_cost(thread_id: str, charge: CostBreakdown) -> CostBreakdown | None:
    if not charge["request_count"]:
        return None
    # A short transaction serializes completions across server processes.
    # The queue retains the priced charge if this transaction rolls back.
    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dcode_btw_costs "
            "(thread_id TEXT PRIMARY KEY NOT NULL, breakdown TEXT NOT NULL)"
        )
        # A tombstone prevents late completions from resurrecting deleted spend.
        deleted = conn.execute(
            "SELECT 1 FROM dcode_btw_costs WHERE thread_id = ? AND breakdown = 'null'",
            (thread_id,),
        ).fetchone()
        if deleted:
            return None
        total = _merge_cost_breakdowns(_read_cost(conn, thread_id), charge)
        conn.execute(
            "INSERT INTO dcode_btw_costs VALUES (?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET breakdown = excluded.breakdown",
            (thread_id, json.dumps(total)),
        )
    return total


def _retry_pending_costs(thread_id: str) -> CostBreakdown | None:
    """Drain a thread's queue under `_SETTLEMENT_LOCK`, keeping failures owned.

    Returns:
        The latest persisted subtotal, or `None` when no usage was written.
    """
    pending = _PENDING_COSTS.get(thread_id)
    total = None
    while pending:
        charge = pending[0]
        if isinstance(charge, _UnpricedCost):
            _, charge = _price_operation_records(
                charge.records,
                fallback=charge.fallback,
                historical_complete=charge.historical_complete,
            )
            # A database retry must not reprice a charge or retain raw records.
            pending[0] = charge
        persisted = _persist_cost(thread_id, charge)
        pending.popleft()
        if persisted is not None:
            total = persisted
    _PENDING_COSTS.pop(thread_id, None)
    return total


def _settle_cost(
    thread_id: str,
    recorder: _SessionCostRecorder,
    *,
    fallback: tuple[str, str],
    historical_complete: bool,
) -> CostBreakdown | None:
    """Transfer ownership before writing so a failed request remains retryable.

    Returns:
        The latest persisted subtotal, or `None` when no usage was written.
    """
    with _SETTLEMENT_LOCK:
        _PENDING_COSTS.setdefault(thread_id, deque()).append(
            _UnpricedCost(recorder.drain(thread_id), fallback, historical_complete)
        )
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
    fallback = _checkpointed_model_spec(state)
    historical_complete = not _has_legacy_cost_history(state)
    del state  # Even a logged settlement traceback must not retain the transcript.
    recorder = _SessionCostRecorder()
    token = _RECORDER_VAR.set(recorder)
    try:
        try:
            text = await answer
        finally:
            # A disconnect can arrive after the provider completed. Finish the
            # database write even then, but allow cancellation during generation.
            settlement = asyncio.create_task(
                asyncio.to_thread(
                    _settle_cost,
                    thread_id,
                    recorder,
                    fallback=fallback,
                    historical_complete=historical_complete,
                )
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
