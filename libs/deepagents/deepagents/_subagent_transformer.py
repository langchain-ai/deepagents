"""Surface declared subagents as typed `run.subagents` handles.

Each `task` tool call dispatched by `langchain.agents.create_agent`'s
Send fan-out (`Send("tools", [tool_call])`) becomes its own pregel
task — a per-call dispatched task whose `input` is a single-element
list of tool-call dicts. The child subagent subgraph starts under a
namespace of the form ``["tools:<pregel_task_id>"]``, so the
namespace tail's task id is the same `task.id` the per-call
dispatched task carries.

This transformer:

1. On every `tasks` start event (anywhere in the run), inspects the
   task's `input` for the per-call envelope. If it parses, records
   ``task_id → {"subagent_type", "task_input"}`` keyed by the
   pregel task id. Mirrors what
   `langgraph.stream.transformers._TasksLifecycleBase._record_invocation_metadata`
   does on the wire, but with a deepagents-specific dict shape so
   we can plumb it straight into our typed handles.
2. When `_on_started` fires for a tracked namespace, looks up the
   pending entry by `trigger_call_id` (the parsed namespace tail).
   If it resolves to a declared subagent name, builds a
   `SubagentRunStream` (or async variant) wrapping a child mini-mux
   and pushes it onto the `subagents` log.

A subagent therefore shows up on **both** `run.subgraphs` (untyped,
superset, keyed by the raw Pregel segment) and `run.subagents`
(typed, declared-only). The typed handle's `cause` exposes
`trigger_call_id` (the pregel task id) — not the model-side
`tool_call_id`, which is no longer used for correlation because it
conflated parallel `task` calls dispatched in the same parent step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from langgraph.stream.run_stream import (
    AsyncSubgraphRunStream,
    SubgraphRunStream,
)
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    SubgraphStatus,
    _TasksLifecycleBase,
)

if TYPE_CHECKING:
    from langgraph.stream._mux import StreamMux
    from langgraph.stream._types import ProtocolEvent


class SubagentRunStream(SubgraphRunStream):
    """Typed sync handle for a declared subagent execution."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        task_input: str | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self.task_input = task_input

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        """Invocation descriptor for in-process consumers.

        Exposes the pregel task id under the `trigger_call_id` key
        (not the model-side `tool_call_id`, which conflated parallel
        `task` calls under the same parent task). The `type` tag stays
        camelCase (`"toolCall"`) for in-process consumer compatibility;
        the wire-side `lifecycle.started.cause` uses snake_case
        (`"tool_call"`).
        """
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "trigger_call_id": self.trigger_call_id}


class AsyncSubagentRunStream(AsyncSubgraphRunStream):
    """Typed async handle for a declared subagent execution."""

    def __init__(
        self,
        mux: StreamMux,
        *,
        path: tuple[str, ...],
        graph_name: str | None = None,
        trigger_call_id: str | None = None,
        task_input: str | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self.task_input = task_input

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> dict[str, str] | None:
        """Invocation descriptor for in-process consumers.

        Exposes the pregel task id under the `trigger_call_id` key
        (not the model-side `tool_call_id`, which conflated parallel
        `task` calls under the same parent task). The `type` tag stays
        camelCase (`"toolCall"`) for in-process consumer compatibility;
        the wire-side `lifecycle.started.cause` uses snake_case
        (`"tool_call"`).
        """
        if self.trigger_call_id is None:
            return None
        return {"type": "toolCall", "trigger_call_id": self.trigger_call_id}


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
        self._log: StreamChannel[SubagentRunStream | AsyncSubagentRunStream] = StreamChannel()
        self._handles: dict[tuple[str, ...], SubagentRunStream | AsyncSubagentRunStream] = {}
        self._mux: StreamMux | None = None
        # Maps trigger_call_id (per-call dispatched task id, parsed from
        # namespace tail) -> {"subagent_type": ..., "task_input": ...}.
        # Mined from the per-call task's ToolCallWithContext input
        # envelope (`{"tool_call": {"args": {"subagent_type": ...,
        # "description": ...}}, ...}`). Each Send-dispatched call has a
        # unique task id, so parallel `task` calls each get their own
        # entry — no conflation.
        self._pending: dict[str, dict[str, str]] = {}

    def init(self) -> dict[str, Any]:
        return {"subagents": self._log}

    def _on_register(self, mux: StreamMux) -> None:
        self._mux = mux

    def _should_track(self, ns: tuple[str, ...]) -> bool:
        depth = len(self.scope)
        return len(ns) == depth + 1 and ns[:depth] == self.scope

    def _capture_per_call_metadata(self, data: dict[str, Any]) -> None:
        """Capture subagent metadata from a per-call dispatched task.

        `langchain.agents.create_agent` Send-fans out tool calls as
        ``Send("tools", [tool_call])`` — one pregel task per call —
        so the per-call dispatched task's `input` is always a
        single-element list of one tool-call dict
        (``Send("tools", [tool_call])`` shape). Anything else is not
        a per-call envelope we recognise and is ignored. We mine
        `subagent_type` and `description` off that single `task`
        tool call and stash them keyed by the per-call dispatched
        task's `id`. The child subgraph's first lifecycle event
        will carry that same id as `trigger_call_id` (parsed from
        the ``tools:<id>`` namespace tail), so `_on_started` joins
        on it directly.

        Multiple `task` calls dispatched in the same model turn each
        produce a separate per-call task with its own unique id, so
        keying by the pregel id disambiguates parallel calls without
        relying on the model-side `tool_call_id` (which previously
        conflated calls when emitted in the same batch).
        """
        task_id = data.get("id")
        if not isinstance(task_id, str):
            return
        payload = data.get("input")
        if isinstance(payload, list):
            if len(payload) != 1:
                return
            tool_call = payload[0]
            if not isinstance(tool_call, dict) or tool_call.get("name") != "task":
                return
            args = tool_call.get("args")
            if not isinstance(args, dict):
                return
            subagent_type = args.get("subagent_type")
            if not isinstance(subagent_type, str):
                return
            if subagent_type not in self._names:
                return
            description = args.get("description")
            self._pending[task_id] = {
                "subagent_type": subagent_type,
                "task_input": description if isinstance(description, str) else "",
            }

    def _on_started(
        self,
        ns: tuple[str, ...],
        graph_name: str | None,  # noqa: ARG002
        trigger_call_id: str | None,
        invocation_metadata: dict[str, str] | None = None,  # noqa: ARG002
    ) -> None:
        # Pair the started namespace to its captured metadata via
        # trigger_call_id (the pregel task id, parsed from the
        # namespace tail). Each per-call dispatched task has a unique
        # id, so parallel `task` calls under the same parent each get
        # their own pending entry — no conflation. The
        # `invocation_metadata` arg from the base class is unused here
        # because we maintain our own typed handle state via
        # `_pending`; we forward `trigger_call_id` directly to the
        # handle constructor as the in-process correlation id.
        if trigger_call_id is None:
            return
        info = self._pending.pop(trigger_call_id, None)
        if info is None:
            return
        if self._mux is None or ns in self._handles:
            return
        try:
            child_mux = self._mux._make_child(ns)
        except RuntimeError:
            return
        handle_cls = AsyncSubagentRunStream if child_mux.is_async else SubagentRunStream
        handle = handle_cls(
            mux=child_mux,
            path=ns,
            graph_name=info["subagent_type"],
            trigger_call_id=trigger_call_id,
            task_input=info["task_input"] or None,
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

    def _handle_for_event(self, event: ProtocolEvent) -> SubagentRunStream | AsyncSubagentRunStream | None:
        ns = tuple(event["params"]["namespace"])
        depth = len(self.scope)
        if len(ns) < depth + 1:
            return None
        handle = self._handles.get(ns[: depth + 1])
        if handle is None or handle._mux is None or handle._mux._events._closed:
            return None
        return handle

    def process(self, event: ProtocolEvent) -> bool:
        if event.get("method") == "tasks":
            data = event.get("params", {}).get("data", {})
            if isinstance(data, dict) and "result" not in data:
                self._capture_per_call_metadata(data)
        keep = super().process(event)
        handle = self._handle_for_event(event)
        if handle is not None:
            # Mirror SubgraphTransformer.process: observe the event on
            # the handle (so `_latest` is populated for `output()`)
            # before forwarding it to the child mini-mux.
            handle._observe_event(event)
            handle._mux.push(event)
        return keep
