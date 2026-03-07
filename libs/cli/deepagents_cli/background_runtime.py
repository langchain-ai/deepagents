"""Background task runtime backed by TaskIQ in-memory broker."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from taskiq import InMemoryBroker

if TYPE_CHECKING:
    from taskiq.task import AsyncTaskiqTask

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(StrEnum):
    """Lifecycle status for a background task."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"


class BackgroundApprovalDecision(StrEnum):
    """Approval decision values for background HITL events."""

    APPROVE = "approve"
    REJECT = "reject"


@dataclass(slots=True)
class BackgroundTaskRecord:
    """Runtime-visible task metadata and latest status."""

    task_id: str
    command: str
    status: BackgroundTaskStatus
    created_at: datetime
    updated_at: datetime
    result_text: str | None = None
    error_text: str | None = None
    exit_code: int | None = None


@dataclass(slots=True)
class BackgroundHitlEvent:
    """HITL approval event emitted by background task execution."""

    event_id: str
    task_id: str
    action_requests: list[dict[str, Any]]
    assistant_id: str | None = None


class BackgroundRuntime:
    """TaskIQ-backed runtime for CLI background shell tasks."""

    def __init__(
        self,
        *,
        mode: Literal["inmemory"] = "inmemory",
        poll_interval_seconds: float = 0.1,
        require_hitl_for_shell: bool = True,
    ) -> None:
        """Initialize runtime internals.

        Args:
            mode: Runtime mode. Currently only `inmemory` is supported.
            poll_interval_seconds: Poll interval used by UI bridge loops.
            require_hitl_for_shell: Whether shell tasks require approval.

        Raises:
            ValueError: If `mode` is unsupported.
        """
        if mode != "inmemory":
            msg = f"Unsupported background runtime mode: {mode}"
            raise ValueError(msg)

        self._mode = mode
        self._poll_interval_seconds = poll_interval_seconds
        self._require_hitl_for_shell = require_hitl_for_shell

        self._broker = InMemoryBroker()
        self._started = False

        self._lock = threading.RLock()
        self._records: dict[str, BackgroundTaskRecord] = {}
        self._task_handles: dict[str, AsyncTaskiqTask[Any]] = {}
        self._wait_events: dict[str, asyncio.Event] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._killed: set[str] = set()
        self._pending_updates: deque[str] = deque()
        self._pending_tui_notifications: deque[str] = deque()

        self._hitl_events: deque[BackgroundHitlEvent] = deque()
        self._hitl_waiters: dict[
            str, asyncio.Future[tuple[BackgroundApprovalDecision, str | None]]
        ] = {}
        self._task_hitl_event_ids: dict[str, str] = {}
        self._hitl_idle_event = asyncio.Event()
        self._hitl_idle_event.set()

        @self._broker.task(task_name="deepagents.background.execute_shell")
        async def _execute_background_shell(
            task_id: str,
            command: str,
        ) -> dict[str, Any]:
            return await self._run_shell_task(task_id=task_id, command=command)

        self._execute_background_shell = _execute_background_shell

    @property
    def poll_interval_seconds(self) -> float:
        """Return the recommended poll interval for app loops."""
        return self._poll_interval_seconds

    async def start(self) -> None:
        """Start the underlying TaskIQ broker."""
        if self._started:
            return
        await self._broker.startup()
        self._started = True

    async def shutdown(self) -> None:
        """Shutdown broker and cancel active runtime tasks."""
        with self._lock:
            monitor_tasks = list(self._monitor_tasks.values())
            process_ids = list(self._records.keys())

        for task_id in process_ids:
            await self.kill_task(task_id)

        for monitor in monitor_tasks:
            monitor.cancel()

        for monitor in monitor_tasks:
            try:
                await monitor
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.debug(
                    "Background monitor task failed during shutdown",
                    exc_info=True,
                )

        for event_id in list(self._hitl_waiters.keys()):
            self.resolve_hitl_event(
                event_id,
                decision=BackgroundApprovalDecision.REJECT,
                message="Runtime shutdown",
            )

        if self._started:
            await self._broker.shutdown()
            self._started = False

    async def submit_shell_task(
        self,
        command: str,
        *,
        assistant_id: str | None = None,
    ) -> str:
        """Submit a shell command for background execution.

        Args:
            command: Shell command to execute.
            assistant_id: Optional assistant ID for approval display.

        Returns:
            Runtime task ID.
        """
        task_id = uuid.uuid4().hex[:12]
        now = datetime.now(UTC)
        record = BackgroundTaskRecord(
            task_id=task_id,
            command=command,
            status=BackgroundTaskStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._records[task_id] = record
            self._wait_events[task_id] = asyncio.Event()
            self._pending_updates.append(f"Task `{task_id}` queued: `{command}`")

        task = await self._execute_background_shell.kiq(
            task_id=task_id,
            command=command,
        )

        with self._lock:
            self._task_handles[task_id] = task

        monitor = asyncio.create_task(
            self._monitor_task_result(task_id, task, assistant_id),
            name=f"background-monitor-{task_id}",
        )
        with self._lock:
            self._monitor_tasks[task_id] = monitor

        return task_id

    def list_tasks(self) -> list[BackgroundTaskRecord]:
        """Return snapshot of all background tasks."""
        with self._lock:
            records = list(self._records.values())
        return sorted(records, key=lambda item: item.created_at)

    def get_task(self, task_id: str) -> BackgroundTaskRecord | None:
        """Return a task snapshot by ID."""
        with self._lock:
            return self._records.get(task_id)

    async def kill_task(self, task_id: str) -> bool:
        """Best-effort kill a background task.

        Returns:
            `True` if the task was transitioned to killed, else `False`.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return False
            if record.status in {
                BackgroundTaskStatus.SUCCEEDED,
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.KILLED,
            }:
                return False

            self._killed.add(task_id)
            record.status = BackgroundTaskStatus.KILLED
            record.updated_at = datetime.now(UTC)
            record.error_text = "Killed by user (best-effort cancellation)."
            self._pending_updates.append(f"Task `{task_id}` marked as killed.")
            self._pending_tui_notifications.append(f"Background task {task_id} killed.")
            wait_event = self._wait_events.get(task_id)
            process = self._processes.get(task_id)
            hitl_event_id = self._task_hitl_event_ids.get(task_id)

        if hitl_event_id is not None:
            self.resolve_hitl_event(
                hitl_event_id,
                decision=BackgroundApprovalDecision.REJECT,
                message="Task killed before approval",
            )

        if process is not None and process.returncode is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            except OSError:
                logger.debug(
                    "Failed to terminate process for task %s",
                    task_id,
                    exc_info=True,
                )

        if wait_event is not None:
            wait_event.set()
        return True

    async def wait_task(
        self,
        task_id: str,
        timeout_seconds: float | None = None,
    ) -> BackgroundTaskRecord:
        """Wait for a task to reach terminal state.

        Args:
            task_id: Task ID to wait for.
            timeout_seconds: Optional timeout.

        Returns:
            Final task record.

        Raises:
            ValueError: If task ID is unknown.
        """
        with self._lock:
            record = self._records.get(task_id)
            event = self._wait_events.get(task_id)
            if record is None or event is None:
                msg = f"Unknown background task: {task_id}"
                raise ValueError(msg)
            if record.status in {
                BackgroundTaskStatus.SUCCEEDED,
                BackgroundTaskStatus.FAILED,
                BackgroundTaskStatus.KILLED,
            }:
                return record

        await asyncio.wait_for(event.wait(), timeout=timeout_seconds)

        with self._lock:
            final_record = self._records.get(task_id)
            if final_record is None:
                msg = f"Unknown background task after wait: {task_id}"
                raise ValueError(msg)
            return final_record

    def pop_hitl_event(self) -> BackgroundHitlEvent | None:
        """Pop one pending HITL event for foreground approval UI.

        Returns:
            One pending event, or `None` when the queue is empty.
        """
        with self._lock:
            if not self._hitl_events:
                return None
            return self._hitl_events.popleft()

    def pending_hitl_count(self) -> int:
        """Return number of pending HITL approvals."""
        with self._lock:
            return len(self._hitl_waiters)

    async def wait_for_no_pending_hitl(self) -> None:
        """Wait until all pending HITL approvals are resolved."""
        with self._lock:
            if not self._hitl_waiters:
                return
            idle_event = self._hitl_idle_event
        await idle_event.wait()

    def resolve_hitl_event(
        self,
        event_id: str,
        *,
        decision: BackgroundApprovalDecision,
        message: str | None = None,
    ) -> None:
        """Resolve a pending HITL event."""
        with self._lock:
            waiter = self._hitl_waiters.pop(event_id, None)
            self._sync_hitl_idle_event_locked()
        if waiter is None or waiter.done():
            return
        waiter.set_result((decision, message))

    def consume_status_updates(self) -> list[str]:
        """Drain and return status updates since last poll.

        Returns:
            Collected runtime update strings.
        """
        with self._lock:
            updates = list(self._pending_updates)
            self._pending_updates.clear()
        return updates

    def consume_tui_notifications(self) -> list[str]:
        """Drain and return user-facing TUI notification messages.

        Returns:
            Collected notification strings for foreground display.
        """
        with self._lock:
            notifications = list(self._pending_tui_notifications)
            self._pending_tui_notifications.clear()
        return notifications

    async def _monitor_task_result(
        self,
        task_id: str,
        task: AsyncTaskiqTask[Any],
        assistant_id: str | None,
    ) -> None:
        """Await task completion and map TaskIQ result to runtime state.

        Raises:
            asyncio.CancelledError: If the monitor task is cancelled.
        """
        del assistant_id  # reserved for future use

        try:
            result = await task.wait_result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record = self._records.get(task_id)
                if record and record.status != BackgroundTaskStatus.KILLED:
                    record.status = BackgroundTaskStatus.FAILED
                    record.updated_at = datetime.now(UTC)
                    record.error_text = str(exc)
                    self._pending_updates.append(f"Task `{task_id}` failed: {exc}")
                    self._pending_tui_notifications.append(
                        f"Background task {task_id} failed: {exc}"
                    )
                    event = self._wait_events.get(task_id)
                    if event is not None:
                        event.set()
            return
        finally:
            with self._lock:
                self._monitor_tasks.pop(task_id, None)

        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return

            event = self._wait_events.get(task_id)
            if record.status == BackgroundTaskStatus.KILLED:
                if event is not None:
                    event.set()
                return

            if result.is_err:
                record.status = BackgroundTaskStatus.FAILED
                record.error_text = str(result.error)
                record.updated_at = datetime.now(UTC)
                self._pending_updates.append(
                    f"Task `{task_id}` failed: {record.error_text}"
                )
                self._pending_tui_notifications.append(
                    f"Background task {task_id} failed."
                )
            else:
                payload = result.return_value
                if isinstance(payload, dict):
                    record.exit_code = _safe_int(payload.get("exit_code"))
                    record.result_text = _safe_str(payload.get("stdout"))
                    stderr_text = _safe_str(payload.get("stderr"))
                    if record.exit_code == 0:
                        record.status = BackgroundTaskStatus.SUCCEEDED
                        if stderr_text:
                            record.error_text = stderr_text
                        self._pending_updates.append(f"Task `{task_id}` succeeded.")
                        self._pending_tui_notifications.append(
                            f"Background task {task_id} completed."
                        )
                    else:
                        record.status = BackgroundTaskStatus.FAILED
                        record.error_text = stderr_text or "Background command failed"
                        self._pending_updates.append(
                            f"Task `{task_id}` failed with exit code "
                            f"{record.exit_code}."
                        )
                        self._pending_tui_notifications.append(
                            f"Background task {task_id} failed "
                            f"(exit {record.exit_code})."
                        )
                    record.updated_at = datetime.now(UTC)
                else:
                    record.status = BackgroundTaskStatus.SUCCEEDED
                    record.result_text = str(payload)
                    record.updated_at = datetime.now(UTC)
                    self._pending_updates.append(f"Task `{task_id}` succeeded.")
                    self._pending_tui_notifications.append(
                        f"Background task {task_id} completed."
                    )

            if event is not None:
                event.set()

    async def _run_shell_task(self, *, task_id: str, command: str) -> dict[str, Any]:
        """TaskIQ task body for executing shell commands.

        Returns:
            Result payload containing `exit_code`, `stdout`, and `stderr`.
        """
        await self._mark_running(task_id)

        if self._require_hitl_for_shell:
            decision, reason = await self._request_shell_approval(task_id, command)
            if decision == BackgroundApprovalDecision.REJECT:
                with self._lock:
                    record = self._records.get(task_id)
                    if (
                        record is not None
                        and record.status != BackgroundTaskStatus.KILLED
                    ):
                        record.status = BackgroundTaskStatus.FAILED
                        record.error_text = reason or "Rejected by user"
                        record.updated_at = datetime.now(UTC)
                        self._pending_updates.append(
                            f"Task `{task_id}` rejected by approval."
                        )
                        self._pending_tui_notifications.append(
                            f"Background task {task_id} rejected by approval."
                        )
                        event = self._wait_events.get(task_id)
                        if event is not None:
                            event.set()
                return {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": reason or "Rejected by user",
                }

        with self._lock:
            if task_id in self._killed:
                return {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "Killed before command execution",
                }

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )
        with self._lock:
            self._processes[task_id] = process

        try:
            stdout_bytes, stderr_bytes = await process.communicate()
        finally:
            with self._lock:
                self._processes.pop(task_id, None)

        return {
            "exit_code": process.returncode,
            "stdout": (stdout_bytes or b"").decode(errors="replace").strip(),
            "stderr": (stderr_bytes or b"").decode(errors="replace").strip(),
        }

    async def _mark_running(self, task_id: str) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            if record.status == BackgroundTaskStatus.QUEUED:
                record.status = BackgroundTaskStatus.RUNNING
                record.updated_at = datetime.now(UTC)
                self._pending_updates.append(f"Task `{task_id}` is running.")

    async def _request_shell_approval(
        self,
        task_id: str,
        command: str,
    ) -> tuple[BackgroundApprovalDecision, str | None]:
        """Emit HITL event and wait for foreground decision.

        Returns:
            Decision tuple: `(decision, optional_message)`.
        """
        event_id = uuid.uuid4().hex
        approval_event = BackgroundHitlEvent(
            event_id=event_id,
            task_id=task_id,
            action_requests=[
                {
                    "name": "execute",
                    "args": {"command": command},
                    "description": (
                        f"Background task {task_id} requests shell command approval"
                    ),
                }
            ],
        )

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[tuple[BackgroundApprovalDecision, str | None]] = (
            loop.create_future()
        )

        with self._lock:
            self._hitl_waiters[event_id] = waiter
            self._task_hitl_event_ids[task_id] = event_id
            self._hitl_events.append(approval_event)
            self._sync_hitl_idle_event_locked()
            self._pending_updates.append(
                f"Task `{task_id}` awaiting approval for shell execution."
            )

        try:
            return await waiter
        finally:
            with self._lock:
                self._task_hitl_event_ids.pop(task_id, None)

    def _sync_hitl_idle_event_locked(self) -> None:
        if self._hitl_waiters:
            self._hitl_idle_event.clear()
            return
        self._hitl_idle_event.set()


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
