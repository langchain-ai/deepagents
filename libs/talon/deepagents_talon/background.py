"""Bounded local workers with independent checkpoint threads and explicit recovery."""

from __future__ import annotations

import asyncio
import contextvars
import json
import sqlite3
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4
from weakref import finalize

from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves injected annotations
)
from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langgraph.types import StateSnapshot

_MAX_WORKERS = 4
_MAX_TASKS = 1000
_MAX_INPUT_BYTES = 1024 * 1024
_MAX_RESULT_CHARS = 64_000
_TIMEOUT = 3600
_TERMINAL = frozenset({"success", "error", "cancelled"})


class BackgroundGraph(Protocol):
    """Graph operations needed by the supervisor."""

    async def ainvoke(self, value: object, config: RunnableConfig) -> Mapping[str, object]:
        """Run or resume a child checkpoint thread."""

    async def aupdate_state(
        self, config: RunnableConfig, values: object, *, as_node: str
    ) -> RunnableConfig:
        """Persist the initial child input before a worker can perform effects."""

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        """Inspect pending approval interrupts before explicit recovery."""


class LocalTaskSupervisor:
    """Persist local task ownership separately from the cancellable parent turn.

    Args:
        path: Private SQLite task database, or `None` for an in-memory runtime.
        factory: Build a graph from a pinned local agent definition.
    """

    def __init__(
        self, path: Path | None, factory: Callable[[dict[str, str]], BackgroundGraph]
    ) -> None:
        """Open the task store without starting workers."""
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.is_symlink():
                msg = "Local task database must not be a symlink"
                raise ValueError(msg)
            path.touch(mode=0o600, exist_ok=True)
            path.chmod(0o600)
        self._database = sqlite3.connect(path or ":memory:")
        self._finalize = finalize(self, self._database.close)
        self._database.row_factory = sqlite3.Row
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS tasks ("
            "id TEXT PRIMARY KEY, owner TEXT NOT NULL, definition TEXT NOT NULL, "
            "status TEXT NOT NULL, result TEXT NOT NULL DEFAULT '', "
            "origin TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0, "
            "revision INTEGER NOT NULL DEFAULT 0)"
        )
        self._database.execute(
            "UPDATE tasks SET status = 'interrupted', revision = revision + 1, "
            "result = 'Process stopped; inspect the checkpoint before explicitly resuming.' "
            "WHERE status IN ('running', 'cancelling')"
        )
        self._database.commit()
        self._factory = factory
        self._lock = asyncio.Lock()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._graphs: dict[str, BackgroundGraph] = {}

    async def start_task(
        self,
        definition: dict[str, str],
        description: str,
        runtime: ToolRuntime,
        *,
        factory: Callable[[dict[str, str]], BackgroundGraph] | None = None,
    ) -> dict[str, str]:
        """Create a task record before starting an independent worker.

        Args:
            definition: Pinned effective prompt and model for the child.
            description: Delegated task.
            runtime: Parent tool context, including conversation ownership.
            factory: Factory pinned to the launching graph's tool configuration.

        Returns:
            Task ID and launch status, or a capacity error.
        """
        async with self._lock:
            if self._full():
                return {
                    "error": "Local capacity reached: four active workers or 1000 retained tasks"
                }
            messages = [*runtime.state.get("messages", []), HumanMessage(content=description)]
            if len(json.dumps(messages_to_dict(messages)).encode()) > _MAX_INPUT_BYTES:
                return {"error": "Conversation is too large to fork; summarize it first"}
            definition = dict(definition)
            graph = (factory or self._factory)(definition)
            task_id = f"talon-local-{uuid4().hex}"
            await graph.aupdate_state(
                {"configurable": {"thread_id": task_id}},
                {"messages": messages},
                as_node="__start__",
            )
            origin = runtime.config.get("configurable", {}).get("talon_origin") or {}
            self._database.execute(
                "INSERT INTO tasks (id, owner, definition, status, origin) VALUES (?, ?, ?, ?, ?)",
                (task_id, _owner(runtime), json.dumps(definition), "running", json.dumps(origin)),
            )
            self._database.commit()
            self._schedule(task_id, graph, None)
            return {"task_id": task_id, "status": "running"}

    def _full(self) -> bool:
        count = self._database.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        return len(self._workers) >= _MAX_WORKERS or count >= _MAX_TASKS

    def _schedule(self, task_id: str, graph: BackgroundGraph, value: object) -> None:
        self._graphs[task_id] = graph
        worker = asyncio.create_task(
            self._run(task_id, graph, value), name=task_id, context=contextvars.Context()
        )
        self._workers[task_id] = worker
        worker.add_done_callback(lambda done: self._finished(task_id, done))

    def _finished(self, task_id: str, worker: asyncio.Task[None]) -> None:
        self._workers.pop(task_id, None)
        if worker.cancelled():
            row = self._database.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            status = "cancelled" if row["status"] == "cancelling" else "interrupted"
            self._set_status(task_id, status)
            if status == "cancelled":
                self._graphs.pop(task_id, None)

    async def _run(self, task_id: str, graph: BackgroundGraph, value: object) -> None:
        try:
            async with asyncio.timeout(_TIMEOUT):
                state = await graph.ainvoke(value, {"configurable": {"thread_id": task_id}})
            status = "interrupted" if state.get("__interrupt__") else "success"
            messages = state.get("messages", [])
            last = messages[-1] if isinstance(messages, list) and messages else None
            interrupts = state.get("__interrupt__", ())
            result = (
                str([getattr(item, "value", "") for item in interrupts])
                if isinstance(interrupts, (list, tuple)) and interrupts
                else str(getattr(last, "content", ""))
            )[:_MAX_RESULT_CHARS]
        except asyncio.CancelledError:
            row = self._database.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            status = "cancelled" if row["status"] == "cancelling" else "interrupted"
            result = (
                "Worker stopped. Explicitly resume interrupted tasks after reviewing their state."
            )
        except Exception:  # noqa: BLE001  # isolate worker failures without logging prompt or secrets
            status, result = (
                "error",
                "Local subagent failed; inspect its checkpoint before retrying.",
            )
        self._set_status(task_id, status, result)
        if status in _TERMINAL:
            self._graphs.pop(task_id, None)

    def _set_status(self, task_id: str, status: str, result: str = "") -> None:
        self._database.execute(
            "UPDATE tasks SET status = ?, result = ?, delivered = 0, revision = revision + 1 "
            "WHERE id = ?",
            (status, result, task_id),
        )
        self._database.commit()

    def _task(self, task_id: str, owner: str) -> sqlite3.Row:
        row = self._database.execute(
            "SELECT * FROM tasks WHERE id = ? AND owner = ?", (task_id, owner)
        ).fetchone()
        if row is None:
            msg = "Unknown local task for this conversation"
            raise ValueError(msg)
        return row

    async def cancel(self, task_id: str, owner: str) -> dict[str, str]:
        """Cancel only a task owned by the requesting conversation.

        Args:
            task_id: Exact local task identifier.
            owner: Parent conversation identifier.

        Returns:
            Cancellation status; a timeout never claims the worker has stopped.
        """
        async with self._lock:
            row = self._task(task_id, owner)
            worker = self._workers.get(task_id)
            if worker is None or worker.done():
                if row["status"] not in _TERMINAL:
                    self._set_status(task_id, "cancelled")
                return {"task_id": task_id, "status": "stopped"}
            self._set_status(task_id, "cancelling")
            worker.cancel()
            done, _pending = await asyncio.wait([worker], timeout=10)
            return {"task_id": task_id, "status": "cancelled" if done else "cancelling"}

    async def resume(
        self, task_id: str, owner: str, decision: Literal["approve", "reject"] | None
    ) -> dict[str, str]:
        """Explicitly resume a checkpoint, optionally answering one tool approval.

        Args:
            task_id: Exact local task identifier.
            owner: Parent conversation identifier.
            decision: Approval decision for a paused child, or `None` after process interruption.

        Returns:
            Resumed status, or a capacity/state error.
        """
        async with self._lock:
            row = self._task(task_id, owner)
            if row["status"] != "interrupted" or task_id in self._workers:
                return {"error": "Only stopped, interrupted tasks can be resumed"}
            if len(self._workers) >= _MAX_WORKERS:
                return {"error": "Local worker capacity reached"}
            graph = self._graphs.get(task_id) or self._factory(json.loads(row["definition"]))
            snapshot = await graph.aget_state({"configurable": {"thread_id": task_id}})
            if not snapshot.values:
                return {"error": "Task checkpoint is unavailable; refusing to restart it"}
            if snapshot.interrupts and decision is None:
                return {"error": "Inspect the paused tool calls, then provide approve or reject"}
            if not snapshot.interrupts and decision is not None:
                return {"error": "This task has no pending tool approval"}
            value = (
                None
                if decision is None
                else Command(
                    resume={
                        item.id: {
                            "decisions": [{"type": decision}] * len(item.value["action_requests"])
                        }
                        for item in snapshot.interrupts
                    }
                )
            )
            self._set_status(task_id, "running")
            self._schedule(task_id, graph, value)
            return {"task_id": task_id, "status": "running"}

    async def close(self) -> None:
        """Stop workers before the runtime closes its shared checkpointer."""
        workers = list(self._workers.values())
        for worker in workers:
            worker.cancel()
        if workers:
            _done, pending = await asyncio.wait(workers, timeout=10)
            if pending:
                msg = "Local workers did not stop; checkpoint resources remain open"
                raise RuntimeError(msg)
        self._finalize()

    async def update(self, task_id: str, owner: str, message: str) -> dict[str, str]:
        """Safely interrupt a worker and send follow-up instructions on its existing thread.

        Args:
            task_id: Exact local task identifier.
            owner: Parent conversation identifier.
            message: New instructions to append to the child conversation.

        Returns:
            Updated status, or an error if the worker cannot stop or needs approval.
        """
        async with self._lock:
            row = self._task(task_id, owner)
            graph = self._graphs.get(task_id) or self._factory(json.loads(row["definition"]))
            worker = self._workers.get(task_id)
            if worker is not None and not worker.done():
                worker.cancel()
                _done, pending = await asyncio.wait([worker], timeout=10)
                if pending:
                    return {"error": "Worker has not stopped; follow-up was not started"}
            if len(self._workers) >= _MAX_WORKERS:
                return {"error": "Local worker capacity reached"}
            config: RunnableConfig = {"configurable": {"thread_id": task_id}}
            snapshot = await graph.aget_state(config)
            if snapshot.interrupts:
                return {"error": "Resolve the pending tool approval before sending a follow-up"}
            if not snapshot.values:
                return {"error": "Task checkpoint is unavailable"}
            await graph.aupdate_state(
                config, {"messages": [HumanMessage(content=message)]}, as_node="__start__"
            )
            self._set_status(task_id, "running")
            self._schedule(task_id, graph, None)
            return {"task_id": task_id, "status": "running"}

    def tools(
        self,
        definitions: Mapping[str, dict[str, str]],
        *,
        factory: Callable[[dict[str, str]], BackgroundGraph] | None = None,
    ) -> list[BaseTool]:
        """Bind launch tools to one immutable configuration snapshot.

        Args:
            definitions: Current local agent definitions keyed by public name.
            factory: Factory capturing this graph's configured runtime tools.

        Returns:
            Launch, inspection, cancellation, and explicit recovery tools.
        """

        @tool
        async def start_local_task(
            subagent_type: str, description: str, runtime: ToolRuntime
        ) -> dict[str, str]:
            """Run a local subagent in the background and return its task ID immediately.

            The child inherits conversation context. It continues when the parent is interrupted.
            Use check_local_task to inspect results or pending tool approvals.
            """
            definition = definitions.get(subagent_type)
            if definition is None:
                return {"error": "Unknown local subagent type"}
            return await self.start_task(definition, description, runtime, factory=factory)

        @tool
        async def check_local_task(task_id: str, runtime: ToolRuntime) -> dict[str, str]:
            """Read a local task's status and result, including any paused tool call."""
            row = self._task(task_id, _owner(runtime))
            return {"task_id": row["id"], "status": row["status"], "result": row["result"]}

        @tool
        async def list_local_tasks(runtime: ToolRuntime) -> list[dict[str, str]]:
            """List this conversation's local background tasks, including after a restart."""
            rows = self._database.execute(
                "SELECT id, status FROM tasks WHERE owner = ? ORDER BY rowid DESC LIMIT 100",
                (_owner(runtime),),
            )
            return [{"task_id": row["id"], "status": row["status"]} for row in rows]

        @tool
        async def cancel_local_task(task_id: str, runtime: ToolRuntime) -> dict[str, str]:
            """Explicitly stop a local worker; stopping the parent alone leaves workers running."""
            return await self.cancel(task_id, _owner(runtime))

        @tool
        async def update_local_task(
            task_id: str, message: str, runtime: ToolRuntime
        ) -> dict[str, str]:
            """Interrupt a local worker and send new instructions on the same checkpoint thread."""
            return await self.update(task_id, _owner(runtime), message)

        @tool
        async def resume_local_task(
            task_id: str, runtime: ToolRuntime, decision: Literal["approve", "reject"] | None = None
        ) -> dict[str, str]:
            """Resume an interrupted local task after inspecting its checkpoint/result.

            Set decision only for a paused tool approval. After a process interruption, resuming
            may retry the last unfinished tool; confirm its effects before resuming.
            """
            return await self.resume(task_id, _owner(runtime), decision)

        start_local_task.description += "\nAvailable agents: " + ", ".join(definitions)
        return [
            start_local_task,
            check_local_task,
            list_local_tasks,
            cancel_local_task,
            update_local_task,
            resume_local_task,
        ]

    def pending_results(self) -> list[dict[str, str]]:
        """Return undelivered worker results for the host to route when the parent is idle."""
        rows = self._database.execute(
            "SELECT id, owner, result, origin, status, revision FROM tasks "
            "WHERE delivered = 0 AND status IN ('success', 'error', 'interrupted') LIMIT 100"
        )
        return [{key: str(value) for key, value in dict(row).items()} for row in rows]

    def result_is_current(self, task_id: str, revision: str) -> bool:
        """Check that a queued notification still refers to the same task result.

        Args:
            task_id: Task being delivered.
            revision: Result revision captured by the delivery loop.

        Returns:
            Whether the task has stayed unchanged since the result was read.
        """
        return (
            self._database.execute(
                "SELECT 1 FROM tasks WHERE id = ? AND revision = ?", (task_id, revision)
            ).fetchone()
            is not None
        )

    def acknowledge(self, task_id: str, *, revision: str | None = None) -> None:
        """Mark one successfully delivered worker result as delivered.

        Args:
            task_id: Task whose result was delivered by the host.
            revision: Only acknowledge this result revision, preserving newer notifications.
        """
        self._database.execute(
            "UPDATE tasks SET delivered = 1 WHERE id = ? AND (? IS NULL OR revision = ?)",
            (task_id, revision, revision),
        )
        self._database.commit()


def _owner(runtime: ToolRuntime) -> str:
    owner = runtime.config.get("configurable", {}).get("thread_id")
    if not isinstance(owner, str) or not owner:
        msg = "Local task tools require a conversation thread"
        raise ValueError(msg)
    return owner
