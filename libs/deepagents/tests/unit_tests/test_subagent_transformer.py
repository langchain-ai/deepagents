"""Unit tests for `SubagentTransformer` — synthetic lifecycle events.

These tests exercise the transformer in isolation by pushing protocol
events directly into a `StreamMux` configured with
`SubagentTransformer` alongside the usual `values` / `messages` /
`subgraphs` transformers. Mirrors the structure of langgraph's
`test_stream_subgraph_transformer.py`.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langgraph.errors import GraphInterrupt
from langgraph.stream._event_log import EventLog
from langgraph.stream._mux import StreamMux
from langgraph.stream.transformers import (
    MessagesTransformer,
    SubgraphTransformer,
    ValuesTransformer,
)

from deepagents import SubagentRunStream, SubagentTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

TS = int(time.time() * 1000)


def _lifecycle(
    event: str,
    *,
    namespace: list[str] | None = None,
    graph_name: str | None = None,
    trigger_call_id: str | None = None,
    error: str | None = None,
) -> ProtocolEvent:
    data: dict[str, Any] = {"event": event}
    if graph_name is not None:
        data["graph_name"] = graph_name
    if trigger_call_id is not None:
        data["trigger_call_id"] = trigger_call_id
    if error is not None:
        data["error"] = error
    return {
        "type": "event",
        "method": "lifecycle",
        "params": {
            "namespace": namespace or [],
            "timestamp": TS,
            "data": data,
        },
    }


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


def _subscribe(log: EventLog) -> None:
    """Flip `_subscribed = True` so pushes retain items for test inspection."""
    log._subscribed = True


def _pre_subscribe_handle(handle: SubagentRunStream) -> None:
    """Flip `_subscribed` on every EventLog inside the handle's mini-mux.

    The child mini-mux is built via `make_child` with the full factory
    list, so `values` / `messages` / `subagents` logs exist as
    projections. Tests that feed events directly need them subscribed
    so pushes retain items for `_items` inspection.
    """
    for value in handle._mux.extensions.values():
        if isinstance(value, EventLog):
            _subscribe(value)


def _factories(names: frozenset[str]):
    """Return a factory list matching DeepAgentStreamer's minimal shape.

    Omits the agent-layer transformers (ToolCall / Middleware) — they
    don't interact with subagent routing and adding them here would
    just pull in langchain-side imports.
    """

    def subagent_factory(scope: tuple[str, ...]) -> SubagentTransformer:
        return SubagentTransformer(scope, subagent_names=names)

    return [
        ValuesTransformer,
        MessagesTransformer,
        SubgraphTransformer,
        subagent_factory,
    ]


def _handle_values_items(handle: SubagentRunStream) -> list:
    return list(handle._mux.extensions["values"]._items)


def _handle_subagents_items(handle: SubagentRunStream) -> list:
    return list(handle._mux.extensions["subagents"]._items)


class TestSubagentTransformerUnit:
    NAMES = frozenset({"researcher", "coder"})

    def _mux(self) -> tuple[StreamMux, SubagentTransformer]:
        mux = StreamMux(factories=_factories(self.NAMES), is_async=False)
        transformer = mux.transformer_by_key("subagents")
        assert isinstance(transformer, SubagentTransformer)
        _subscribe(transformer._log)
        return mux, transformer

    def _handle(self, transformer: SubagentTransformer) -> SubagentRunStream:
        (handle,) = list(transformer._log._items)
        return handle

    def test_nondeclared_graph_name_is_ignored(self) -> None:
        """A plain subgraph (graph_name not in declared names) stays off `.subagents`."""
        mux, transformer = self._mux()
        mux.push(
            _lifecycle(
                "started",
                namespace=["tools:abc"],
                graph_name="plain_tool_subgraph",
                trigger_call_id="abc",
            )
        )

        assert list(transformer._log._items) == []
        assert transformer._by_ns == {}

    def test_declared_graph_name_yields_handle(self) -> None:
        """A `started` event whose graph_name matches a declared name produces a SubagentRunStream."""
        mux, transformer = self._mux()
        mux.push(
            _lifecycle(
                "started",
                namespace=["tools:abc"],
                graph_name="researcher",
                trigger_call_id="abc",
            )
        )

        handle = self._handle(transformer)
        assert isinstance(handle, SubagentRunStream)
        assert handle.path == ("tools:abc",)
        assert handle.name == "researcher"
        assert handle.trigger_call_id == "abc"
        assert handle.status == "started"

    def test_status_transitions(self) -> None:
        """Started → running → completed updates the handle in place."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))
        mux.push(_lifecycle("running", namespace=["tools:x"]))
        mux.push(_lifecycle("completed", namespace=["tools:x"]))

        handle = self._handle(transformer)
        assert handle.status == "completed"
        # Terminal status closes the mini-mux so consumers unblock.
        assert handle._mux.extensions["values"]._closed

    def test_failed_stores_error(self) -> None:
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))
        mux.push(_lifecycle("failed", namespace=["tools:x"], error="boom"))

        handle = self._handle(transformer)
        assert handle.status == "failed"
        assert handle.error == "boom"

    def test_values_routed_into_handle(self) -> None:
        """Values events under the subagent's ns flow into its mini-mux."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))

        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        mux.push(_values({"k": 1}, namespace=["tools:x"]))
        mux.push(_values({"k": 2}, namespace=["tools:x"]))

        assert _handle_values_items(handle) == [{"k": 1}, {"k": 2}]
        assert handle.output == {"k": 2}

    def test_root_values_not_routed(self) -> None:
        """A values event at root ns must not leak into a child handle."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))
        handle = self._handle(transformer)
        _pre_subscribe_handle(handle)

        mux.push(_values({"k": "root"}, namespace=[]))
        assert _handle_values_items(handle) == []

    def test_nested_subagent_surfaces_under_parent(self) -> None:
        """A nested subagent under a subagent appears on the parent's `.subagents`."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:p"], graph_name="researcher"))

        parent = self._handle(transformer)
        _pre_subscribe_handle(parent)

        mux.push(
            _lifecycle(
                "started",
                namespace=["tools:p", "tools:c"],
                graph_name="coder",
            )
        )

        (child,) = _handle_subagents_items(parent)
        assert isinstance(child, SubagentRunStream)
        assert child.path == ("tools:p", "tools:c")
        assert child.name == "coder"

    def test_nested_nondeclared_subgraph_does_not_surface_on_subagents(self) -> None:
        """A nested plain subgraph (graph_name not declared) does NOT appear on `.subagents`."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:p"], graph_name="researcher"))

        parent = self._handle(transformer)
        _pre_subscribe_handle(parent)

        mux.push(
            _lifecycle(
                "started",
                namespace=["tools:p", "tools:c"],
                graph_name="plain_nested",
            )
        )

        assert _handle_subagents_items(parent) == []

    def test_finalize_closes_dangling(self) -> None:
        """A still-open child at finalize time is marked completed and its mux closed."""
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))

        handle = self._handle(transformer)
        mux.close()

        assert handle.status == "completed"
        assert handle._mux.extensions["values"]._closed
        assert handle._mux.extensions["subagents"]._closed

    def test_fail_with_graph_interrupt_marks_interrupted(self) -> None:
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))
        handle = self._handle(transformer)

        mux.fail(GraphInterrupt())
        assert handle.status == "interrupted"

    def test_fail_with_generic_error_marks_failed(self) -> None:
        mux, transformer = self._mux()
        mux.push(_lifecycle("started", namespace=["tools:x"], graph_name="researcher"))
        handle = self._handle(transformer)

        mux.fail(RuntimeError("kaboom"))
        assert handle.status == "failed"
        assert handle.error == "kaboom"
