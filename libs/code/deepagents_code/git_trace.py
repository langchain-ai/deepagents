"""Reconcile observed checkout work onto the current turn's traces."""

from __future__ import annotations

import asyncio
import logging
import operator
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    TracePolicy,
    omit_payload,
)
from langgraph.config import get_config

from deepagents_code import _env_vars
from deepagents_code.git_evidence import (
    GitSnapshot,
    _candidate_paths,
    aprobe_git,
    changed_snapshots,
    probe_git,
)
from deepagents_code.turn_trace import TurnTrace, capture_turn

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from deepagents.backends.protocol import BackendProtocol
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from langgraph.types import Command


logger = logging.getLogger(__name__)
_MAX_THREADS = 64
_FAILURE_LIMIT = 3
_COOLDOWN_SECONDS = 30.0


def _observes_checkout(request: ToolCallRequest) -> bool:
    name = request.tool_call["name"]
    if name in {"read_file", "ls", "glob", "grep"}:
        return False
    metadata = (request.tool.metadata or {}) if request.tool is not None else {}
    if (
        metadata.get("readOnlyHint") is True
        and metadata.get("destructiveHint") is not True
    ):
        return False
    if metadata.get("destructiveHint") is True or name in {
        "execute",
        "write_file",
        "edit_file",
        "apply_patch",
        "js",
        "task",
    }:
        return True
    return any(
        isinstance(request.tool_call["args"].get(key), str)
        for key in ("file_path", "path", "cwd", "command")
    )


@dataclass
class _Turn:
    trace: TurnTrace
    targets: dict[tuple[str, str], tuple[int, GitSnapshot]] = field(
        default_factory=dict
    )
    baseline: dict[str, tuple[int, GitSnapshot]] = field(default_factory=dict)
    covered: set[str] = field(default_factory=set)
    sequence: int = 0
    probes: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def covers(self, paths: tuple[str, ...] | None) -> bool:
        with self.lock:
            return paths is not None and self.covered.issuperset(paths)

    def start(self) -> int:
        with self.lock:
            self.probes += 1
            return self.probes

    def observe(
        self,
        snapshots: list[GitSnapshot],
        probe: int,
        order: int,
        *,
        created: bool,
    ) -> None:
        """Attribute changes since the newest older snapshot of each checkout."""
        with self.lock:
            fresh = [
                snapshot
                for snapshot in snapshots
                if (known := self.baseline.get(snapshot.root)) is None
                or known[0] < probe
            ]
            previous = [
                self.baseline[snapshot.root][1]
                for snapshot in fresh
                if snapshot.root in self.baseline
            ]
            candidates = [
                snapshot
                for snapshot in fresh
                if created or snapshot.root in self.baseline
            ]
            for snapshot in changed_snapshots(previous, candidates):
                if snapshot.repository is not None:
                    key = (snapshot.repository.name, snapshot.branch)
                    self.targets.setdefault(key, (order, snapshot))
            for snapshot in fresh:
                self.baseline[snapshot.root] = (probe, snapshot)

    def metadata(self) -> dict[str, object]:
        with self.lock:
            targets = [
                item[1]
                for item in sorted(self.targets.values(), key=operator.itemgetter(0))
            ]
        if not targets:
            return {}
        selected = targets[0]
        repository = selected.repository
        if repository is None:
            return {}
        metadata: dict[str, object] = {
            "ls_git_repo": repository.name,
            "ls_repo_provider": repository.provider,
            "open_swe_git_targets": [
                {"repo": target.repository.name, "branch": target.branch}
                for target in targets
                if target.repository is not None
            ],
        }
        if selected.branch and selected.branch != "HEAD":
            metadata["ls_git_branch"] = selected.branch
        return metadata


class GitTraceMiddleware(AgentMiddleware[AgentState, ContextT]):
    """Observe backend work and backfill only the initiating user turn.

    Each observing tool call compares a snapshot taken after it runs with the
    newest earlier snapshot from the same turn, so a probe before the call is
    only needed when its arguments reach paths the turn has not inspected yet.
    Repeated probe failures back off and then disable probing for the session.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(self, backend: Callable[[], BackendProtocol], cwd: str) -> None:
        """Bind observations to the execution backend and its working directory."""
        self._backend = backend
        self._cwd = cwd
        self._turns: OrderedDict[str, tuple[str, _Turn]] = OrderedDict()
        self._lock = threading.RLock()
        self._failures = 0
        self._resume_at = 0.0
        self._disabled = False

    def _turn(self, config: RunnableConfig) -> _Turn | None:
        try:
            trace = capture_turn(config)
            metadata = config.get("metadata", {})
            thread, turn = metadata.get("thread_id"), metadata.get("turn_id")
            if (
                trace is None
                or not isinstance(thread, str)
                or not isinstance(turn, str)
            ):
                return None
            with self._lock:
                previous = self._turns.get(thread)
                if previous is None or previous[0] != turn:
                    self._turns[thread] = (turn, _Turn(trace))
                self._turns.move_to_end(thread)
                while len(self._turns) > _MAX_THREADS:
                    self._turns.popitem(last=False)
                return self._turns[thread][1]
        except Exception:
            logger.debug("Could not capture git attribution turn", exc_info=True)
            return None

    def _ready(self) -> bool:
        with self._lock:
            return not self._disabled and time.monotonic() >= self._resume_at

    def _succeeded(self) -> None:
        with self._lock:
            self._failures = 0

    def _failed(self, error: Exception) -> None:
        logger.debug("Could not inspect git attribution evidence", exc_info=error)
        with self._lock:
            self._failures += 1
            if self._disabled or (
                self._failures < _FAILURE_LIMIT
                and not isinstance(error, NotImplementedError)
            ):
                cooldown = _COOLDOWN_SECONDS * 2 ** (self._failures - 1)
                self._resume_at = time.monotonic() + cooldown
                return
            self._disabled = True
        logger.info(
            "Git trace attribution disabled for this session after %d failed "
            "probe(s) (%s); set %s=false or [tracing].git_attribution = false "
            "to skip it entirely",
            self._failures,
            type(error).__name__,
            _env_vars.GIT_ATTRIBUTION,
        )

    def _location(self) -> tuple[BackendProtocol, str]:
        backend = self._backend()
        return backend, self._cwd or str(getattr(backend, "cwd", ""))

    def _paths(self, args: Mapping[str, object]) -> tuple[str, ...] | None:
        try:
            _, cwd = self._location()
            return tuple(_candidate_paths(cwd, args)) if cwd else None
        except Exception:
            logger.debug("Could not resolve git attribution paths", exc_info=True)
            return None

    def _probe(
        self, args: Mapping[str, object] | None = None
    ) -> list[GitSnapshot] | None:
        if not self._ready():
            return None
        try:
            backend, cwd = self._location()
            if not cwd:
                return None
            snapshots = probe_git(backend, cwd, args)
        except Exception as error:  # noqa: BLE001  # Telemetry never breaks tools.
            self._failed(error)
            return None
        self._succeeded()
        return snapshots

    async def _aprobe(
        self, args: Mapping[str, object] | None = None
    ) -> list[GitSnapshot] | None:
        if not self._ready():
            return None
        try:
            backend, cwd = self._location()
            if not cwd:
                return None
            snapshots = await asyncio.wait_for(
                aprobe_git(backend, cwd, args), timeout=10
            )
        except Exception as error:  # noqa: BLE001  # Telemetry never breaks tools.
            self._failed(error)
            return None
        self._succeeded()
        return snapshots

    @staticmethod
    def _update(turn: _Turn) -> None:
        try:
            turn.trace.update(
                turn.metadata(),
                remove=("ls_git_repo", "ls_git_branch", "ls_repo_provider"),
            )
        except Exception:
            logger.debug("Could not reconcile git attribution traces", exc_info=True)

    @staticmethod
    def _order(turn: _Turn) -> int:
        with turn.lock:
            turn.sequence += 1
            return turn.sequence

    @staticmethod
    def _observe(
        turn: _Turn,
        snapshots: list[GitSnapshot] | None,
        probe: int,
        order: int,
        paths: tuple[str, ...] | None,
        *,
        created: bool,
    ) -> None:
        if snapshots is None:
            return
        try:
            turn.observe(snapshots, probe, order, created=created)
            if paths is not None:
                with turn.lock:
                    turn.covered.update(paths)
        except Exception:
            logger.debug("Could not retain git attribution evidence", exc_info=True)

    def _begin(self, request: ToolCallRequest) -> _Turn | None:
        if not _observes_checkout(request) or not self._ready():
            return None
        return self._turn(request.runtime.config)

    @override
    def before_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> None:
        """Register this invocation alongside any approval-resume siblings."""
        self._turn(get_config())

    @override
    async def abefore_agent(
        self, state: AgentState, runtime: Runtime[ContextT]
    ) -> None:
        """Register this invocation without changing graph state."""
        self.before_agent(state, runtime)

    @override
    def after_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> None:
        """Merge final work attribution into this turn's earlier calls."""
        if turn := self._turn(get_config()):
            self._update(turn)

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime[ContextT]) -> None:
        """Keep optional trace transport off the server event loop."""
        if turn := self._turn(get_config()):
            await asyncio.to_thread(self._update, turn)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[object]],
    ) -> ToolMessage | Command[object]:
        """Compare actual backend checkouts after tool execution.

        Returns:
            The unchanged tool result.
        """
        turn = self._begin(request)
        if turn is None:
            return handler(request)
        order = self._order(turn)
        args = request.tool_call["args"]
        paths = self._paths(args)
        if not turn.covers(paths):
            probe = turn.start()
            self._observe(turn, self._probe(args), probe, order, paths, created=False)
        created = turn.covers(paths)
        result = handler(request)
        probe = turn.start()
        self._observe(turn, self._probe(args), probe, order, paths, created=created)
        return result

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
    ) -> ToolMessage | Command[object]:
        """Observe asynchronous tools without changing their outputs.

        Returns:
            The unchanged tool result.
        """
        turn = self._begin(request)
        if turn is None:
            return await handler(request)
        order = self._order(turn)
        args = request.tool_call["args"]
        paths = self._paths(args)
        if not turn.covers(paths):
            probe = turn.start()
            snapshots = await self._aprobe(args)
            self._observe(turn, snapshots, probe, order, paths, created=False)
        created = turn.covers(paths)
        result = await handler(request)
        probe = turn.start()
        snapshots = await self._aprobe(args)
        self._observe(turn, snapshots, probe, order, paths, created=created)
        return result
