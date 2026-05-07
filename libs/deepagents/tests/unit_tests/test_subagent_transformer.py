"""Unit tests for `SubagentTransformer`.

Pushes synthetic `tasks` protocol events into a `StreamMux` configured
with `SubagentTransformer` alongside the usual native transformers.
Mirrors the structure of langgraph's `test_stream_subgraph_transformer.py`.

The realistic event sequence (post-namespace-keyed refactor):

1. Per-call dispatched `tasks` start at parent ns with ``input``
   matching the `ToolCallWithContext` envelope
   (`{"tool_call": {"id": tc, "name": "task", "args":
   {"subagent_type": ..., "description": ...}}}`). The task's `id`
   is the pregel task id assigned by Send fan-out — the transformer
   keys ``_pending`` by this id (== `trigger_call_id` parsed from
   the child's namespace tail), so parallel `task` calls each get a
   distinct entry.
2. Child-scope `tasks` start at ``["tools:<parent_task_id>"]`` —
   triggers `_on_started`, which looks up the pending entry and
   builds a `SubagentRunStream` if the type is declared.
3. Parent-scope `tasks` result with the matching ``id`` — closes the
   child handle.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphInterrupt
from langgraph.stream._mux import StreamMux
from langgraph.stream.stream_channel import StreamChannel
from langgraph.stream.transformers import (
    LifecycleTransformer,
    MessagesTransformer,
    SubgraphTransformer,
    ValuesTransformer,
)

from deepagents import SubagentRunStream, SubagentTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

TS = int(time.time() * 1000)


def _per_call_tasks_start(
    namespace: list[str],
    *,
    parent_task_id: str,
    subagent_type: str,
    tool_call_id: str,
    description: str | None = None,
) -> ProtocolEvent:
    """A per-call dispatched `tasks` start with a single-element tool-call list.

    Mirrors what `langchain.agents.create_agent`'s
    ``Send("tools", [tool_call])`` fan-out emits: one pregel task per
    call whose `input` is a list containing a single tool-call dict.
    `parent_task_id` is the pregel task id (the same id that becomes
    `trigger_call_id` when parsed off the child subgraph's namespace
    tail). `tool_call_id` is the model-side id; it travels in the
    tool-call dict but is no longer used for correlation.
    """
    args: dict[str, Any] = {"subagent_type": subagent_type}
    if description is not None:
        args["description"] = description
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": parent_task_id,
                "name": "tools",
                "input": [
                    {
                        "id": tool_call_id,
                        "name": "task",
                        "args": args,
                    }
                ],
                "triggers": [],
            },
        },
    }


def _child_tasks_start(
    namespace: list[str],
    *,
    task_id: str = "child-task",
    name: str = "PatchToolCallsMiddleware.before_agent",
) -> ProtocolEvent:
    """A child-scope `tasks` start (any inner node — kicks off the subagent).

    No `tool_call_id` plumbing on this event anymore; correlation is
    by `trigger_call_id` parsed from the namespace tail (which equals
    the per-call dispatched task's `id`).
    """
    data: dict[str, Any] = {
        "id": task_id,
        "name": name,
        "input": None,
        "triggers": [],
    }
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": data,
        },
    }


def _parent_tasks_result(
    namespace: list[str],
    *,
    parent_task_id: str,
    error: str | None = None,
    interrupts: list[dict[str, Any]] | None = None,
) -> ProtocolEvent:
    """A parent-scope `tasks` result that closes the dispatched subagent."""
    return {
        "type": "event",
        "method": "tasks",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": {
                "id": parent_task_id,
                "name": "tools",
                "error": error,
                "interrupts": interrupts or [],
                "result": {},
            },
        },
    }


def _spawn(
    mux: StreamMux,
    *,
    parent_task_id: str,
    subagent_type: str,
    tool_call_id: str,
    parent_ns: list[str] | None = None,
    description: str | None = None,
) -> None:
    """Push per-call + child start events that mimic a real subagent invocation."""
    parent_ns = parent_ns or []
    mux.push(
        _per_call_tasks_start(
            parent_ns,
            parent_task_id=parent_task_id,
            subagent_type=subagent_type,
            tool_call_id=tool_call_id,
            description=description,
        )
    )
    mux.push(
        _child_tasks_start(
            [*parent_ns, f"tools:{parent_task_id}"],
        )
    )


def _values(payload: dict[str, Any], *, namespace: list[str]) -> ProtocolEvent:
    return {
        "type": "event",
        "method": "values",
        "params": {
            "namespace": namespace,
            "timestamp": TS,
            "data": payload,
        },
    }


def _subscribe(log: StreamChannel) -> None:
    log._subscribed = True


def _pre_subscribe_handle(handle: SubagentRunStream) -> None:
    """Flip `_subscribed` on every StreamChannel inside the handle's mini-mux."""
    for value in handle._mux.extensions.values():
        if isinstance(value, StreamChannel):
            _subscribe(value)


def _factories(names: frozenset[str]):
    def subagent_factory(scope: tuple[str, ...]) -> SubagentTransformer:
        return SubagentTransformer(scope, subagent_names=names)

    return [
        ValuesTransformer,
        MessagesTransformer,
        LifecycleTransformer,
        SubgraphTransformer,
        subagent_factory,
    ]


def _handle_values_items(handle: SubagentRunStream) -> list:
    # StreamChannel._items now stores `(stamp, item)` tuples; strip the stamp.
    return [item for _stamp, item in handle._mux.extensions["values"]._items]


def _handle_subagents_items(handle: SubagentRunStream) -> list:
    return [item for _stamp, item in handle._mux.extensions["subagents"]._items]


class TestSubagentTransformerUnit:
    NAMES = frozenset({"researcher", "coder"})

    def _mux(self) -> tuple[StreamMux, SubagentTransformer]:
        mux = StreamMux(factories=_factories(self.NAMES), is_async=False)
        transformer = mux.transformer_by_key("subagents")
        assert isinstance(transformer, SubagentTransformer)
        _subscribe(transformer._log)
        return mux, transformer

    def _handle(self, transformer: SubagentTransformer) -> SubagentRunStream:
        # StreamChannel._items now stores `(stamp, item)` tuples; strip the stamp.
        (entry,) = list(transformer._log._items)
        _stamp, handle = entry
        return handle

    def test_nondeclared_subagent_type_is_ignored(self) -> None:
        """A `task` tool call with a non-declared subagent_type stays off `.subagents`."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="abc",
            subagent_type="plain_subagent",
            tool_call_id="tc-1",
        )

        assert list(transformer._log._items) == []
        assert transformer._handles == {}

    def test_declared_subagent_yields_handle(self) -> None:
        """A declared subagent_type produces a `SubagentRunStream` with proper metadata."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="abc",
            subagent_type="researcher",
            tool_call_id="tc-1",
            description="do research",
        )

        handle = self._handle(transformer)
        assert isinstance(handle, SubagentRunStream)
        assert handle.path == ("tools:abc",)
        assert handle.name == "researcher"
        # Cause now exposes `trigger_call_id` (pregel task id), not the
        # model-side `tool_call_id`.
        assert handle.cause == {"type": "toolCall", "trigger_call_id": "abc"}
        assert handle.task_input == "do research"
        assert handle.status == "started"

    def test_parallel_task_calls_get_distinct_handles(self) -> None:
        """Parallel `task` calls under the same parent step don't conflate.

        Each Send-dispatched per-call task has a unique pregel task id,
        so two `_spawn` invocations with distinct `parent_task_id`s
        produce two independent handles even though they share the
        parent ns and overlap in time.
        """
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="p1",
            subagent_type="researcher",
            tool_call_id="tc-shared",
        )
        _spawn(
            mux,
            parent_task_id="p2",
            subagent_type="coder",
            tool_call_id="tc-shared",
        )

        handles = [item for _stamp, item in transformer._log._items]
        assert len(handles) == 2
        names = sorted(h.name for h in handles)
        assert names == ["coder", "researcher"]
        causes = sorted(h.cause["trigger_call_id"] for h in handles if h.cause is not None)
        assert causes == ["p1", "p2"]
        # Pin the pairing: the future-bug this guards against is
        # cross-contamination of metadata across parallel calls under
        # the same parent step. If a regression pairs the wrong
        # subagent_type with the wrong trigger_call_id, this fails.
        by_id = {h.cause["trigger_call_id"]: h.graph_name for h in handles}
        assert by_id == {"p1": "researcher", "p2": "coder"}

    def test_status_transitions(self) -> None:
        """A parent-scope tasks-result closes the subagent and marks completed."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )
        mux.push(_parent_tasks_result([], parent_task_id="x"))

        handle = self._handle(transformer)
        assert handle.status == "completed"
        assert handle._mux.extensions["values"]._closed

    def test_failed_stores_error(self) -> None:
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )
        mux.push(_parent_tasks_result([], parent_task_id="x", error="boom"))

        handle = self._handle(transformer)
        assert handle.status == "failed"
        assert handle.error == "boom"

    def test_values_routed_into_handle(self) -> None:
        """Values events under the subagent's ns flow into its mini-mux."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )

        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        mux.push(_values({"k": 1}, namespace=["tools:x"]))
        mux.push(_values({"k": 2}, namespace=["tools:x"]))

        assert _handle_values_items(handle) == [{"k": 1}, {"k": 2}]
        assert handle.output == {"k": 2}

    def test_root_values_not_routed(self) -> None:
        """A values event at root ns must not leak into a child handle."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )
        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        mux.push(_values({"k": "root"}, namespace=[]))
        assert _handle_values_items(handle) == []

    def test_nested_subagent_surfaces_under_parent(self) -> None:
        """A subagent spawned from inside another subagent surfaces on the parent's `.subagents`."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="p",
            subagent_type="researcher",
            tool_call_id="tc-p",
        )

        parent = self._handle(transformer)
        _pre_subscribe_handle(parent)

        # Nested invocation: parent ns is the researcher's ns.
        _spawn(
            mux,
            parent_task_id="c",
            subagent_type="coder",
            tool_call_id="tc-c",
            parent_ns=["tools:p"],
        )

        (child,) = _handle_subagents_items(parent)
        assert isinstance(child, SubagentRunStream)
        assert child.path == ("tools:p", "tools:c")
        assert child.name == "coder"

    def test_nested_nondeclared_does_not_surface_on_subagents(self) -> None:
        """A nested non-declared subagent_type does NOT appear on `.subagents`."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="p",
            subagent_type="researcher",
            tool_call_id="tc-p",
        )

        parent = self._handle(transformer)
        _pre_subscribe_handle(parent)

        _spawn(
            mux,
            parent_task_id="c",
            subagent_type="plain_nested",
            tool_call_id="tc-c",
            parent_ns=["tools:p"],
        )

        assert _handle_subagents_items(parent) == []

    def test_finalize_closes_dangling(self) -> None:
        """A still-open child at finalize time is marked completed and its mux closed."""
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )

        handle = self._handle(transformer)
        mux.close()

        assert handle.status == "completed"
        assert handle._mux.extensions["values"]._closed
        assert handle._mux.extensions["subagents"]._closed

    def test_fail_with_graph_interrupt_marks_interrupted(self) -> None:
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )
        handle = self._handle(transformer)

        mux.fail(GraphInterrupt())
        assert handle.status == "interrupted"

    def test_fail_with_generic_error_marks_failed(self) -> None:
        mux, transformer = self._mux()
        _spawn(
            mux,
            parent_task_id="x",
            subagent_type="researcher",
            tool_call_id="tc-1",
        )
        handle = self._handle(transformer)

        mux.fail(RuntimeError("kaboom"))
        assert handle.status == "failed"
        assert handle.error == "kaboom"
