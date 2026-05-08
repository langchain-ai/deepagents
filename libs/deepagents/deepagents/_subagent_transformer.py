"""Surface declared subagents as typed `run.subagents` handles.

Each `task` tool call dispatched by `langchain.agents.create_agent`'s
Send fan-out (`Send("tools", [tool_call])`) becomes its own pregel
task — a per-call dispatched task whose `input` is a single-element
list containing one tool-call dict
(``[{"id": ..., "name": "task", "args": {...}}]``). The child subagent
subgraph starts under a namespace of the form
``["tools:<pregel_task_id>"]``, so the namespace tail's task id is the
same `task.id` the per-call dispatched task carries.

This transformer:

1. On `tasks` start events at the transformer's own scope, inspects
   the task's `input` for the per-call envelope. If it parses (single
   tool-call dict named ``"task"`` with a declared `subagent_type`),
   records ``task_id → {"subagent_type", ...}`` keyed by the pregel
   task id. Mirrors what
   `langgraph.stream.transformers._TasksLifecycleBase._record_invocation_metadata`
   does on the wire, but with a deepagents-specific dict shape so we
   can plumb it straight into our typed handles. Scoped capture keeps
   `_pending` from absorbing per-call tasks dispatched by inner
   subagents under deeper namespaces.
2. When `_on_started` fires for a tracked namespace, looks up the
   pending entry by `trigger_call_id` (the parsed namespace tail).
   If it resolves to a declared subagent name, builds a
   `SubagentRunStream` (or async variant) wrapping a child mini-mux
   and pushes it onto the `subagents` log.

A subagent therefore shows up on **both** `run.subgraphs` (untyped,
superset, keyed by the raw Pregel segment) and `run.subagents`
(typed, declared-only). The typed handle's `cause` exposes
`trigger_call_id` (the pregel task id, used for identity-level
correlation across lifecycle events for this invocation) and, when
the dispatch came through a recognised tool-call envelope, also
`tool_call_id` — the model-side id. Because the per-call Send fan-out
gives each tool_call its own per-call task, `tool_call_id` here is
1:1 with `trigger_call_id` and disambiguates parallel `task` calls
without the conflation that affected the old batched-Send layout.
UI consumers use `tool_call_id` to anchor the lifecycle event back
to the originating AI message tool call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NotRequired, TypedDict

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

logger = logging.getLogger(__name__)


class SubagentCause(TypedDict):
    """In-process invocation descriptor for a subagent run handle.

    The `type` tag stays camelCase (`"toolCall"`) for in-process consumer
    compatibility; the wire-side `lifecycle.started.cause` uses snake_case
    (`"tool_call"`). Anyone matching this dict against a value pulled from
    JSON logs should compare the wire form, not this one.
    """

    type: Literal["toolCall"]
    trigger_call_id: str
    tool_call_id: NotRequired[str]


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
        tool_call_id: str | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self.task_input = task_input
        self.tool_call_id = tool_call_id

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> SubagentCause | None:
        """Invocation descriptor for in-process consumers.

        `trigger_call_id` is the pregel task id (identity-level
        correlation key across lifecycle events). `tool_call_id`, when
        present, is the model-side id of the originating tool call —
        1:1 with `trigger_call_id` under the per-call Send fan-out, used
        by UI consumers to anchor the invocation back to the AI message.
        See `SubagentCause` for the camelCase/snake_case wire divergence.
        """
        if self.trigger_call_id is None:
            return None
        result: SubagentCause = {
            "type": "toolCall",
            "trigger_call_id": self.trigger_call_id,
        }
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


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
        tool_call_id: str | None = None,
    ) -> None:
        super().__init__(
            mux,
            path=path,
            graph_name=graph_name,
            trigger_call_id=trigger_call_id,
        )
        self.task_input = task_input
        self.tool_call_id = tool_call_id

    @property
    def name(self) -> str | None:
        return self.graph_name

    @property
    def cause(self) -> SubagentCause | None:
        """Invocation descriptor for in-process consumers.

        `trigger_call_id` is the pregel task id (identity-level
        correlation key across lifecycle events). `tool_call_id`, when
        present, is the model-side id of the originating tool call —
        1:1 with `trigger_call_id` under the per-call Send fan-out, used
        by UI consumers to anchor the invocation back to the AI message.
        See `SubagentCause` for the camelCase/snake_case wire divergence.
        """
        if self.trigger_call_id is None:
            return None
        result: SubagentCause = {
            "type": "toolCall",
            "trigger_call_id": self.trigger_call_id,
        }
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


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
        # namespace tail) -> {"subagent_type", optionally "task_input",
        # optionally "tool_call_id"}. Mined from the per-call task's
        # `input` — a single-element list shaped
        # ``[{"id": <tool_call_id>, "name": "task", "args":
        # {"subagent_type": ..., "description": ...}}]`` (the shape
        # `langchain.agents.create_agent` v1 emits via
        # ``Send("tools", [tool_call])``). Each Send-dispatched call has
        # a unique task id, so parallel `task` calls each get their own
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
        single-element list containing one tool-call dict shaped
        ``{"id": <tool_call_id>, "name": "task", "args":
        {"subagent_type": ..., "description": ...}}``. Anything else is
        not a per-call envelope we recognise and is ignored. We mine
        `subagent_type`, `description`, and the model-side
        `tool_call_id` off that single tool-call dict and stash them
        keyed by the per-call dispatched task's `id`. The child
        subgraph's first lifecycle event will carry that same id as
        `trigger_call_id` (parsed from the ``tools:<id>`` namespace
        tail), so `_on_started` joins on it directly.

        Identity-level correlation across lifecycle events still uses
        `trigger_call_id` (the pregel task id, unique per-call). The
        captured `tool_call_id` rides alongside as anchoring metadata
        for UI consumers — under the per-call Send fan-out it's 1:1
        with `trigger_call_id`, so it disambiguates parallel `task`
        calls cleanly (the conflation that affected the old batched-Send
        layout doesn't apply here).
        """
        task_id = data.get("id")
        if not isinstance(task_id, str):
            return
        payload = data.get("input")
        if not isinstance(payload, list):
            return
        if len(payload) != 1:
            # Per-call Send fan-out always emits a single-element list.
            # A multi-element list signals an upstream contract change
            # (a different agent factory, or batched Sends) — silently
            # ignoring it would make subagent handles vanish, so log it.
            logger.debug(
                "SubagentTransformer: ignoring tasks input of unexpected length %d "
                "(expected 1 for per-call Send fan-out); subagent surface will not "
                "fire for this dispatch",
                len(payload),
            )
            return
        tool_call = payload[0]
        if not isinstance(tool_call, dict) or tool_call.get("name") != "task":
            return
        args = tool_call.get("args")
        if not isinstance(args, dict):
            return
        subagent_type = args.get("subagent_type")
        if not isinstance(subagent_type, str) or subagent_type not in self._names:
            return
        entry: dict[str, str] = {"subagent_type": subagent_type}
        description = args.get("description")
        if isinstance(description, str) and description:
            entry["task_input"] = description
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str):
            entry["tool_call_id"] = tool_call_id
        self._pending[task_id] = entry

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
            task_input=info.get("task_input"),
            tool_call_id=info.get("tool_call_id"),
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
            ns = tuple(event["params"]["namespace"])
            # Per-call dispatched `tasks` start events arrive at the
            # parent agent's own scope (`Send("tools", [tool_call])`
            # creates the child task one level deeper, but the start
            # event itself is emitted at the parent ns). Gating on
            # `len(ns) == len(self.scope)` keeps `_pending` from
            # absorbing per-call tasks dispatched by inner subagents.
            if len(ns) == len(self.scope) and ns == self.scope:
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
