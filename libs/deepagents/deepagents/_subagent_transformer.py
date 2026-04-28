"""Surface declared subagents as typed `run.subagents` handles.

Pregel's `tasks` event names a child task by its parent's node name
(typically `"tools"`) plus a Pregel-assigned task id. The actual
`subagent_type` and the user-facing `tool_call_id` only live in the
**parent** task's `input` payload (the list of tool calls). This
transformer does two things:

1. At parent scope, intercept `tasks` start events whose `name ==
   "tools"` and `input` is a list containing one or more `task` tool
   calls. For each such tool call, record
   ``parent_task_id → (subagent_type, tool_call_id)``.
2. When a direct-child `tasks` start fires (segment ``"tools:<id>"``),
   look up `id` in the pending map. If it resolves to a declared
   subagent name, build a `SubagentRunStream` (or async variant)
   wrapping a child mini-mux and push it onto the `subagents` log.
   The handle reports `graph_name` as the subagent's declared type
   (e.g. ``"researcher"``) and `trigger_call_id` as the user-facing
   tool call id (e.g. ``"call-parent-1"``).

A subagent therefore shows up on **both** `run.subgraphs` (untyped,
superset, keyed by the raw Pregel segment) and `run.subagents`
(typed, declared-only, with user-friendly identifiers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream._event_log import EventLog
from langgraph.stream.run_stream import (
    AsyncSubgraphRunStream,
    SubgraphRunStream,
)
from langgraph.stream.transformers import (
    SubgraphStatus,
    ValuesTransformer,
    _TasksLifecycleBase,
)

if TYPE_CHECKING:
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent


class SubagentRunStream(SubgraphRunStream):
    """Typed sync handle for a declared subagent execution."""

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "tool_call_id": self.trigger_call_id}


class AsyncSubagentRunStream(AsyncSubgraphRunStream):
    """Typed async handle for a declared subagent execution."""

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "tool_call_id": self.trigger_call_id}


class SubagentTransformer(_TasksLifecycleBase):
    """Promote declared subagents into typed handles on `run.subagents`."""

    _native: ClassVar[bool] = True

    def __init__(
        self,
        scope: tuple[str, ...] = (),
        *,
        subagent_names: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(scope)
        self._names = subagent_names
        self._log: EventLog[SubagentRunStream | AsyncSubagentRunStream] = EventLog()
        self._handles: dict[tuple[str, ...], SubagentRunStream | AsyncSubagentRunStream] = {}
        self._mux: StreamMux | None = None
        # parent_task_id -> {"subagent_type": ..., "tool_call_id": ...}
        self._pending: dict[str, dict[str, str]] = {}

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _capture_pending_from_parent(self, event: ProtocolEvent) -> None:
        """Record subagent metadata from a parent-scope `tools`-task start.

        Pregel emits a `tasks` start at the parent ns whose `input` is
        the list of tool calls being dispatched. Each ``task`` tool
        call carries the user-visible ``tool_call_id`` and the
        declared ``subagent_type`` we need at child-task time.
        """
        ns = tuple(event["params"]["namespace"])
        if ns != self.scope:
            return
        data = event["params"]["data"]
        if "result" in data or data.get("name") != "tools":
            return
        parent_task_id = data.get("id")
        tool_calls = data.get("input")
        if not isinstance(parent_task_id, str) or not isinstance(tool_calls, list):
            return
        for tc in tool_calls:
            if not isinstance(tc, dict) or tc.get("name") != "task":
                continue
            args = tc.get("args") or {}
            subagent_type = args.get("subagent_type")
            tool_call_id = tc.get("id")
            if not isinstance(subagent_type, str):
                continue
            self._pending[parent_task_id] = {
                "subagent_type": subagent_type,
                "tool_call_id": tool_call_id if isinstance(tool_call_id, str) else "",
            }
            # First task-typed call wins; multiple `task` calls under
            # the same parent task aren't expected in the current
            # scheduling model.
            return

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,  # noqa: ARG002
        trigger_call_id: str | None,
    ) -> None:
        if trigger_call_id is None:
            return
        info = self._pending.pop(trigger_call_id, None)
        if info is None:
            return
        subagent_type = info["subagent_type"]
        if subagent_type not in self._names:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)
        except RuntimeError:
            return
        values_t = child_mux.transformer_by_key("values")
        if not isinstance(values_t, ValuesTransformer):
            return
        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            values_transformer=values_t,
            path=ns,
            graph_name=subagent_type,
            trigger_call_id=info["tool_call_id"] or None,
        )
        self._handles[ns] = handle
        self._log.push(handle)

    def _on_terminal(
        self,
        ns: tuple[str, ...],
        status: SubgraphStatus,
        error: str | None,
    ) -> None:
        handle = self._handles.get(ns)
        if handle is None or handle._seen_terminal:
            return
        handle.status = status
        if error is not None and handle.error is None:
            handle.error = error
        handle._seen_terminal = True
        if handle._mux is None or handle._mux._events._closed:
            return
        if status == "failed":
            handle._mux.fail(RuntimeError(error or "Subagent failed"))
        else:
            handle._mux.close()

    def _child_mux_for_event(self, event: ProtocolEvent) -> StreamMux | None:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:
            return None
        return handle._mux

    def process(self, event: ProtocolEvent) -> bool:
        if event.get("method") == "tasks":
            self._capture_pending_from_parent(event)
        keep = super().process(event)
        child_mux = self._child_mux_for_event(event)
        if child_mux is not None:
            child_mux.push(event)
        return keep
