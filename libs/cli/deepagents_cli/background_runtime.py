"""Background task runtime backed by TaskIQ in-memory broker."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import signal
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

from taskiq import InMemoryBroker

if TYPE_CHECKING:
    from deepagents.backends.protocol import SandboxBackendProtocol
    from taskiq.task import AsyncTaskiqTask

logger = logging.getLogger(__name__)


class BackgroundTaskStatus(StrEnum):
    """Lifecycle status for a background task."""

    QUEUED = "queued"
    """Task registered but not yet running."""

    RUNNING = "running"
    """Task execution in progress."""

    SUCCEEDED = "succeeded"
    """Task completed with exit code 0."""

    FAILED = "failed"
    """Task completed with non-zero exit code or runtime error."""

    REJECTED = "rejected"
    """Task rejected by user via HITL approval."""

    KILLED = "killed"
    """Task cancelled by user (best-effort)."""


TERMINAL_STATUSES: frozenset[BackgroundTaskStatus] = frozenset(
    {
        BackgroundTaskStatus.SUCCEEDED,
        BackgroundTaskStatus.FAILED,
        BackgroundTaskStatus.REJECTED,
        BackgroundTaskStatus.KILLED,
    }
)
"""States from which a task will never transition again."""

_VALID_TRANSITIONS: dict[BackgroundTaskStatus, frozenset[BackgroundTaskStatus]] = {
    BackgroundTaskStatus.QUEUED: frozenset(
        {BackgroundTaskStatus.RUNNING, BackgroundTaskStatus.KILLED}
    ),
    BackgroundTaskStatus.RUNNING: frozenset(
        {
            BackgroundTaskStatus.SUCCEEDED,
            BackgroundTaskStatus.FAILED,
            BackgroundTaskStatus.KILLED,
            BackgroundTaskStatus.REJECTED,
        }
    ),
}
"""Allowed status transitions. Terminal statuses have no outgoing edges."""


class BackgroundApprovalDecision(StrEnum):
    """Approval decision values for background HITL events."""

    APPROVE = "approve"
    """Allow the background task to proceed."""

    REJECT = "reject"
    """Deny the background task; it transitions to `REJECTED`."""


@dataclass(slots=True)
class BackgroundTaskRecord:
    """Runtime-visible task metadata and latest status."""

    task_id: str
    """Unique 12-character hex identifier."""

    command: str
    """Shell command submitted for execution."""

    status: BackgroundTaskStatus
    """Current lifecycle status."""

    created_at: datetime
    """Timestamp when the task was submitted."""

    updated_at: datetime
    """Timestamp of the most recent status change."""

    result_text: str | None = None
    """Captured stdout on completion."""

    stderr_text: str | None = None
    """Captured stderr, or descriptive message for non-success states."""

    exit_code: int | None = None
    """Process exit code when available."""


class BackgroundActionRequest(TypedDict):
    """Typed structure for HITL action request payloads."""

    name: str
    """Tool name requiring approval (e.g. `execute`)."""

    args: dict[str, Any]
    """Tool arguments (e.g. `{"command": "ls"}`)."""

    description: str
    """Human-readable description shown in the approval UI."""


class _ShellResult(TypedDict):
    """Internal result payload from shell task execution."""

    exit_code: int | None
    """Process exit code, or `None` if unavailable."""

    stdout: str
    """Captured standard output."""

    stderr: str
    """Captured standard error."""


@dataclass(slots=True, frozen=True)
class BackgroundHitlEvent:
    """HITL approval event emitted by background task execution."""

    event_id: str
    """Unique 32-character hex identifier for this approval request."""

    task_id: str
    """Background task that triggered this event."""

    action_requests: list[BackgroundActionRequest]
    """Tool calls requiring user approval."""

    assistant_id: str | None = None
    """Optional assistant context for the approval UI."""


_BACKGROUND_TIMEOUT = 86400
"""Timeout in seconds for background tasks via remote backends (24h)."""


class BackgroundRuntime:
    """TaskIQ-backed runtime for CLI background shell tasks."""

    def __init__(
        self,
        *,
        mode: Literal["inmemory"] = "inmemory",
        poll_interval_seconds: float = 0.1,
        require_hitl_for_shell: bool = True,
        backend: SandboxBackendProtocol | None = None,
    ) -> None:
        """Initialize runtime internals.

        Args:
            mode: Runtime mode. Currently only `inmemory` is supported.
            poll_interval_seconds: Poll interval used by UI bridge loops.
            require_hitl_for_shell: Whether shell tasks require approval.
            backend: Execution backend for running commands.

                When `None`, task submissions fail with a configuration error.
                Can be set after init via the `backend` property.

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

        self._backend = backend

        # NOTE: This lock protects dict/deque/set state. The asyncio primitives
        # (_wait_events, _hitl_waiters, _hitl_idle_event) stored here are
        # manipulated only from the event loop thread. Do NOT acquire this lock
        # from a non-event-loop thread if you intend to touch asyncio objects.
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
        ) -> _ShellResult:
            return await self._run_shell_task(task_id=task_id, command=command)

        self._execute_background_shell = _execute_background_shell

    @property
    def poll_interval_seconds(self) -> float:
        """Return the recommended poll interval for app loops."""
        return self._poll_interval_seconds

    @property
    def backend(self) -> SandboxBackendProtocol | None:
        """Return the configured execution backend."""
        return self._backend

    def set_backend(self, backend: SandboxBackendProtocol | None) -> None:
        """Set the execution backend for running commands.

        Args:
            backend: Backend to use for command execution, or `None`
                to clear.
        """
        self._backend = backend

    async def start(self) -> None:
        """Start the underlying TaskIQ broker."""
        if self._started:
            return
        await self._broker.startup()
        self._started = True

    async def __aenter__(self) -> Self:
        """Start runtime as an async context manager.

        Returns:
            The started runtime instance.
        """
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Shutdown runtime on context exit."""
        await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown broker and cancel active runtime tasks."""
        with self._lock:
            monitor_tasks = list(self._monitor_tasks.items())
            process_ids = list(self._records.keys())

        for task_id in process_ids:
            try:
                await self.kill_task(task_id)
            except Exception:
                logger.warning(
                    "Failed to kill task %s during shutdown",
                    task_id,
                    exc_info=True,
                )

        for _, monitor in monitor_tasks:
            monitor.cancel()

        for tid, monitor in monitor_tasks:
            try:
                await monitor
            except asyncio.CancelledError:
                continue
            except Exception:
                logger.warning(
                    "Background monitor task for %s failed during shutdown",
                    tid,
                    exc_info=True,
                )

        with self._lock:
            pending_event_ids = list(self._hitl_waiters.keys())

        for event_id in pending_event_ids:
            try:
                self.resolve_hitl_event(
                    event_id,
                    decision=BackgroundApprovalDecision.REJECT,
                    message="Runtime shutdown",
                )
            except Exception:
                logger.warning(
                    "Failed to resolve HITL event %s during shutdown",
                    event_id,
                    exc_info=True,
                )

        if self._started:
            await self._broker.shutdown()
            self._started = False

    async def submit_shell_task(
        self,
        command: str,
        *,
        assistant_id: str | None = None,  # noqa: ARG002
    ) -> str:
        """Submit a shell command for background execution.

        Returns immediately with a task ID. When `require_hitl_for_shell`
        is enabled, the background worker blocks on approval before shell
        execution begins.

        Args:
            command: Shell command to execute.
            assistant_id: Reserved for future use; currently unused.

        Returns:
            Runtime task ID.

        Raises:
            RuntimeError: If the runtime has not been started.
        """
        if not self._started:
            msg = "BackgroundRuntime has not been started; call start() first"
            raise RuntimeError(msg)

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
            self._monitor_task_result(task_id, task),
            name=f"background-monitor-{task_id}",
        )
        with self._lock:
            self._monitor_tasks[task_id] = monitor

        return task_id

    def list_tasks(self) -> list[BackgroundTaskRecord]:
        """Return snapshot of all background tasks."""
        with self._lock:
            records = [dataclasses.replace(r) for r in self._records.values()]
        return sorted(records, key=lambda item: item.created_at)

    def get_task(self, task_id: str) -> BackgroundTaskRecord | None:
        """Return a task snapshot by ID."""
        with self._lock:
            record = self._records.get(task_id)
            return dataclasses.replace(record) if record else None

    async def kill_task(self, task_id: str) -> bool:
        """Best-effort kill a background task.

        For local backends, terminates the subprocess group via SIGTERM.
        For remote sandbox backends, only marks the task as killed — the
        remote command may continue to completion.

        Args:
            task_id: Task ID to kill.

        Returns:
            `True` if the task was transitioned to killed, else `False`.
        """
        with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return False
            if record.status in TERMINAL_STATUSES:
                return False

            self._killed.add(task_id)
            record.status = BackgroundTaskStatus.KILLED
            record.updated_at = datetime.now(UTC)
            record.stderr_text = "Killed by user (best-effort cancellation)."
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

        # Terminate subprocess if running (local backend path only).
        # For sandbox backends no process handle is stored, so this
        # is skipped and kill is best-effort.
        if process is not None and process.returncode is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                logger.debug(
                    "Process for task %s already exited before kill signal",
                    task_id,
                )
            except OSError:
                logger.warning(
                    "Failed to terminate process for task %s; "
                    "the process may still be running",
                    task_id,
                    exc_info=True,
                )
                with self._lock:
                    self._pending_tui_notifications.append(
                        f"Warning: kill signal for task {task_id} may not "
                        "have been delivered. The process might still be running."
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
            TimeoutError: If `timeout_seconds` is not `None` and the task
                does not reach a terminal state in time.
        """
        with self._lock:
            record = self._records.get(task_id)
            event = self._wait_events.get(task_id)
            if record is None or event is None:
                msg = f"Unknown background task: {task_id}"
                raise ValueError(msg)
            if record.status in TERMINAL_STATUSES:
                return dataclasses.replace(record)

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
        except TimeoutError as exc:
            msg = f"Task {task_id} did not complete within {timeout_seconds}s"
            raise TimeoutError(msg) from exc

        with self._lock:
            final_record = self._records.get(task_id)
            if final_record is None:
                msg = f"Unknown background task after wait: {task_id}"
                raise ValueError(msg)
            return dataclasses.replace(final_record)

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
        """Resolve a pending HITL event.

        Silently no-ops if the event is unknown or already resolved.

        Args:
            event_id: HITL event identifier to resolve.
            decision: Approval or rejection decision.
            message: Optional reason string forwarded to the waiting task.
        """
        with self._lock:
            waiter = self._hitl_waiters.pop(event_id, None)
            self._sync_hitl_idle_event_locked()
        if waiter is None:
            logger.debug(
                "resolve_hitl_event called for unknown event %s "
                "(possibly already resolved or expired)",
                event_id,
            )
            return
        if waiter.done():
            logger.debug(
                "resolve_hitl_event called for already-resolved event %s",
                event_id,
            )
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
    ) -> None:
        """Await task completion and map TaskIQ result to runtime state.

        Raises:
            asyncio.CancelledError: If the monitor task is cancelled.
        """
        try:
            result = await task.wait_result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record = self._records.get(task_id)
                if record and record.status not in TERMINAL_STATUSES:
                    record.status = BackgroundTaskStatus.FAILED
                    record.updated_at = datetime.now(UTC)
                    record.stderr_text = str(exc)
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
                logger.warning(
                    "Task %s completed but record was unexpectedly removed",
                    task_id,
                )
                event = self._wait_events.get(task_id)
                if event is not None:
                    event.set()
                return

            event = self._wait_events.get(task_id)
            if record.status in TERMINAL_STATUSES:
                if event is not None:
                    event.set()
                return

            if result.is_err:
                record.status = BackgroundTaskStatus.FAILED
                record.stderr_text = str(result.error)
                record.updated_at = datetime.now(UTC)
                self._pending_updates.append(
                    f"Task `{task_id}` failed: {record.stderr_text}"
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
                            record.stderr_text = stderr_text
                        self._pending_updates.append(f"Task `{task_id}` succeeded.")
                        self._pending_tui_notifications.append(
                            f"Background task {task_id} completed."
                        )
                    else:
                        record.status = BackgroundTaskStatus.FAILED
                        record.stderr_text = stderr_text or "Background command failed"
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

    async def _run_shell_task(self, *, task_id: str, command: str) -> _ShellResult:
        """TaskIQ task body for executing commands via the configured backend.

        Local backends use direct async subprocess for process-level kill
        and no execution timeout. Remote sandbox backends delegate to
        `backend.aexecute()`.

        Returns:
            Result payload containing `exit_code`, `stdout`, and `stderr`.
        """
        if not await self._mark_running(task_id):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "Task record missing; aborted",
            }

        if self._require_hitl_for_shell:
            decision, reason = await self._request_shell_approval(task_id, command)
            if decision == BackgroundApprovalDecision.REJECT:
                with self._lock:
                    record = self._records.get(task_id)
                    if (
                        record is not None
                        and record.status != BackgroundTaskStatus.KILLED
                    ):
                        record.status = BackgroundTaskStatus.REJECTED
                        record.stderr_text = reason or "Rejected by user"
                        record.updated_at = datetime.now(UTC)
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

        if self._backend is None:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "No execution backend configured",
            }

        # Local backends use direct subprocess for process-level kill
        # and no timeout. Remote backends delegate to aexecute().
        from deepagents.backends import LocalShellBackend

        if isinstance(self._backend, LocalShellBackend):
            return await self._execute_local(task_id, command)
        return await self._execute_via_backend(task_id, command)

    async def _execute_local(
        self,
        task_id: str,
        command: str,
    ) -> _ShellResult:
        """Execute via direct async subprocess for local backends.

        Uses `asyncio.create_subprocess_shell` to preserve process handles
        for kill support and avoid the backend's default execution timeout.

        Returns:
            Result payload containing `exit_code`, `stdout`, and `stderr`.
        """
        # Match the local backend's execution environment.
        raw_cwd = getattr(self._backend, "cwd", None)
        cwd = str(raw_cwd) if raw_cwd is not None else None
        env: dict[str, str] | None = getattr(self._backend, "_env", None)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name != "nt"),
                cwd=cwd,
                env=env,
            )
        except OSError as exc:
            logger.warning(
                "Failed to create subprocess for task %s: %s",
                task_id,
                exc,
            )
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Failed to start subprocess: {exc}",
            }
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

    async def _execute_via_backend(
        self,
        task_id: str,
        command: str,
    ) -> _ShellResult:
        """Execute via `backend.aexecute()` for remote sandbox backends.

        Uses a generous timeout to avoid the backend's default timeout,
        which is typically tuned for foreground tool calls.

        Returns:
            Result payload containing `exit_code`, `stdout`, and `stderr`.
        """
        backend = self._backend
        if backend is None:  # Defensive; caller checks
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "No execution backend configured",
            }

        try:
            response = await backend.aexecute(command, timeout=_BACKGROUND_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Backend execution failed for task %s: %s",
                task_id,
                exc,
            )
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Execution failed: {exc}",
            }

        output = response.output
        if response.truncated:
            output += "\n[output truncated]"

        # aexecute() returns combined stdout+stderr. Route to the
        # appropriate field based on exit code so the monitor logic
        # populates result_text / stderr_text correctly.
        failed = response.exit_code is not None and response.exit_code != 0
        return {
            "exit_code": response.exit_code,
            "stdout": output if not failed else "",
            "stderr": output if failed else "",
        }

    def _transition_status(
        self,
        task_id: str,
        new_status: BackgroundTaskStatus,
    ) -> bool:
        """Validate and apply a status transition.

        Must be called with `_lock` held.

        Args:
            task_id: Task to transition.
            new_status: Target status.

        Returns:
            `True` if the transition was applied, `False` otherwise.
        """
        record = self._records.get(task_id)
        if record is None:
            logger.warning(
                "Cannot transition unknown task %s to %s",
                task_id,
                new_status,
            )
            return False
        valid = _VALID_TRANSITIONS.get(record.status, frozenset())
        if new_status not in valid:
            logger.debug(
                "Ignoring invalid transition %s -> %s for task %s",
                record.status,
                new_status,
                task_id,
            )
            return False
        record.status = new_status
        record.updated_at = datetime.now(UTC)
        return True

    async def _mark_running(self, task_id: str) -> bool:
        """Transition task from QUEUED to RUNNING.

        Args:
            task_id: Task to mark as running.

        Returns:
            `True` if the task was successfully marked, `False` if the
            record is missing or not in QUEUED state.
        """
        with self._lock:
            if not self._transition_status(task_id, BackgroundTaskStatus.RUNNING):
                return False
            self._pending_updates.append(f"Task `{task_id}` is running.")
            return True

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
        """Update idle event based on pending HITL waiters. Must hold `_lock`."""
        if self._hitl_waiters:
            self._hitl_idle_event.clear()
            return
        self._hitl_idle_event.set()


def _safe_str(value: object) -> str | None:
    """Coerce a value to str, returning `None` for `None` input.

    Returns:
        String representation or `None`.
    """
    if value is None:
        return None
    return str(value)


def _safe_int(value: object) -> int | None:
    """Coerce a value to int, returning `None` on failure.

    Returns:
        Integer value or `None`.
    """
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
            logger.warning("Could not parse exit code value %r as int", value)
            return None
    logger.warning("Unexpected exit code type %s: %r", type(value).__name__, value)
    return None


__all__ = [
    "TERMINAL_STATUSES",
    "BackgroundActionRequest",
    "BackgroundApprovalDecision",
    "BackgroundHitlEvent",
    "BackgroundRuntime",
    "BackgroundTaskRecord",
    "BackgroundTaskStatus",
]
