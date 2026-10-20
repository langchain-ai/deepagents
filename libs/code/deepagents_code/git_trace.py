"""Reconcile observed checkout work onto the current turn's traces."""

from __future__ import annotations

import asyncio
import logging
import operator
import threading
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

from deepagents_code.git_evidence import (
    GitSnapshot,
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


@dataclass
class _Turn:
    trace: TurnTrace
    targets: dict[tuple[str, str], tuple[int, GitSnapshot]] = field(
        default_factory=dict
    )
    sequence: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record(
        self, before: list[GitSnapshot], after: list[GitSnapshot], order: int
    ) -> None:
        with self.lock:
            for snapshot in changed_snapshots(before, after):
                if snapshot.repository is not None:
                    key = (snapshot.repository.name, snapshot.branch)
                    self.targets.setdefault(key, (order, snapshot))

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
    """Observe backend work and backfill only the initiating user turn."""

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(self, backend: Callable[[], BackendProtocol], cwd: str) -> None:
        """Bind observations to the execution backend and its working directory."""
        self._backend = backend
        self._cwd = cwd
        self._turns: OrderedDict[str, tuple[str, _Turn]] = OrderedDict()
        self._lock = threading.RLock()

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

    def _probe(
        self, args: Mapping[str, object] | None = None
    ) -> list[GitSnapshot] | None:
        try:
            backend = self._backend()
            cwd = self._cwd or str(getattr(backend, "cwd", ""))
            return probe_git(backend, cwd, args) if cwd else None
        except Exception:
            logger.debug("Could not inspect git attribution evidence", exc_info=True)
            return None

    async def _aprobe(
        self, args: Mapping[str, object] | None = None
    ) -> list[GitSnapshot] | None:
        try:
            backend = self._backend()
            cwd = self._cwd or str(getattr(backend, "cwd", ""))
            if not cwd:
                return None
            return await asyncio.wait_for(aprobe_git(backend, cwd, args), timeout=10)
        except Exception:
            logger.debug("Could not inspect git attribution evidence", exc_info=True)
            return None

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
    def _record(
        turn: _Turn,
        before: list[GitSnapshot] | None,
        after: list[GitSnapshot] | None,
        order: int,
    ) -> None:
        if before is None or after is None:
            return
        try:
            turn.record(before, after, order)
        except Exception:
            logger.debug("Could not retain git attribution evidence", exc_info=True)

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
        """Compare actual backend checkouts around tool execution.

        Returns:
            The unchanged tool result.
        """
        turn = self._turn(request.runtime.config)
        if turn is None:
            return handler(request)
        order = self._order(turn)
        args = request.tool_call["args"]
        before = self._probe(args)
        try:
            return handler(request)
        finally:
            self._record(turn, before, self._probe(args), order)

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
        turn = self._turn(request.runtime.config)
        if turn is None:
            return await handler(request)
        order = self._order(turn)
        args = request.tool_call["args"]
        before = await self._aprobe(args)
        try:
            return await handler(request)
        finally:
            self._record(turn, before, await self._aprobe(args), order)
