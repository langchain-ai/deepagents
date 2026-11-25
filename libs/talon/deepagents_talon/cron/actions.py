"""Durable tool intents for objective runs with interrupted destination writes."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from deepagents_talon.store_records import StoreRecords, Write


@dataclass
class ActionRun:
    """Bind tool receipts and successful destination reads to one objective check."""

    records: StoreRecords
    job: str
    run: str
    current: Callable[[], bool] | None = field(default=None, kw_only=True)
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)
    reads: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    applied: set[tuple[str, str]] = field(default_factory=set)

    @staticmethod
    def arguments(args: dict[str, object]) -> str:
        """Bound historical arguments and normalize key order across retries."""
        return json.dumps(args, default=str, sort_keys=True)[:4000]

    async def already_applied(self, name: str, args: dict[str, object]) -> bool:
        """Check destination receipts and owner-retired writes."""
        args_key = self.arguments(args)
        if (name, args_key) in self.applied:
            return True
        async with self.records.access():
            retired = await self.records.get("retired:" + self.job) or {}
            return [name, args_key] in retired.values()

    async def unresolved(self) -> list[dict[str, str]]:
        """Return interrupted calls requiring a destination lookup."""
        async with self.records.access():
            intents = await self.records.get("actions:" + self.job) or {}
            return [
                row
                for key, row in intents.items()
                if key not in self.active and isinstance(row, dict)
            ]

    @asynccontextmanager
    async def dispatch(
        self, call_id: str, name: str, args: dict[str, object]
    ) -> AsyncGenerator[None]:
        """Commit intent before dispatch and retain it across interruption."""
        key = "actions:" + self.job
        async with self.records.access():
            intents = await self.records.get(key) or {}
            if call_id in intents:
                msg = "Action already recorded"
                raise ValueError(msg)
            intents[call_id] = {
                "run": self.run,
                "call_id": call_id,
                "tool": name,
                "arguments": self.arguments(args),
            }
            await self.records.commit([(key, intents)])
            self.active.add(call_id)
        try:
            yield
        finally:
            self.active.discard(call_id)

    async def receipt(
        self, call_id: str, note: str, *, reconcile: bool = False
    ) -> dict[str, str] | None:
        """Commit an observed outcome and clear its intent together."""
        key = "actions:" + self.job
        async with self.records.access():
            intents = await self.records.get(key) or {}
            intent = intents.get(call_id)
            if not isinstance(intent, dict) or (not reconcile and intent["run"] != self.run):
                return None
            del intents[call_id]
            writes: list[Write] = [(key, intents)]
            check = await self.records.get(self.run)
            if check:
                check["work"] = (str(check["work"]) + "\n" + note)[-4000:]
                writes.append((self.run, check))
            await self.records.commit(writes)
            return intent


CURRENT_ACTIONS: ContextVar[ActionRun | None] = ContextVar("talon_actions", default=None)


@tool
async def reconcile_job_action(
    call_id: str, read_call_id: str, outcome: Literal["applied", "absent"], reference: str
) -> str:
    """Record a conclusive destination read for an interrupted action before further writes.

    Use applied for a matching destination effect and absent only for authoritative absence.
    Supply the successful read tool call ID and the destination evidence reference.
    """
    run = CURRENT_ACTIONS.get()
    if run is None or read_call_id not in run.reads or not reference.strip():
        return "A successful destination read and its evidence reference are required."
    if call_id in run.active:
        return "Action is still executing."
    note = f"Reconciled action ({call_id}): {outcome} at {reference[:1000]}"
    row = await run.receipt(call_id, note, reconcile=True)
    if row is None:
        return "No matching unresolved action."
    if outcome == "applied":
        run.applied.add((row["tool"], row["arguments"]))
    return f"Recorded: {note}."


def successful(result: object, call_id: str) -> bool:
    """Whether a tool returned an unambiguous success receipt."""
    if isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages")
        if not isinstance(messages, list):
            return False
        matches = [m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == call_id]
        if len(matches) != 1:
            return False
        result = matches[0]
    return (
        isinstance(result, ToolMessage)
        and result.tool_call_id == call_id
        and result.status == "success"
        and not (isinstance(result.content, str) and result.content.startswith("Error:"))
    )
