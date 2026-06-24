"""Tests for live subagent lifecycle events emitted on the custom stream.

`call_subagent_task_tool` emits start/complete (or error) events via the
runtime's `stream_writer` so a UI can render a live fan-out panel. These tests
cover event shape, ordering, id propagation, truncation, and that telemetry
failures never break the underlying dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from langchain_quickjs._subagent import call_subagent_task_tool


@dataclass
class _FakeRuntime:
    """Minimal stand-in for the LangGraph ToolRuntime the bridge passes in."""

    tool_call_id: str = "eval_call_123"
    stream_writer: Any = None
    config: dict | None = None


class _FakeTaskTool:
    """Stand-in for the deepagents `task` tool."""

    name = "task"

    def __init__(
        self,
        result: str = "done",
        *,
        raise_exc: Exception | None = None,
    ) -> None:
        self._result = result
        self._raise = raise_exc
        self.seen_runtime_tool_call_id: str | None = None

    async def arun(self, args: dict[str, Any], **_kwargs: Any) -> str:
        self.seen_runtime_tool_call_id = args["runtime"].tool_call_id
        if self._raise is not None:
            raise self._raise
        return self._result


@dataclass
class _Recorder:
    events: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


async def test_emits_start_then_complete() -> None:
    rec = _Recorder()
    runtime = _FakeRuntime(stream_writer=rec)
    tool = _FakeTaskTool("hello world")

    out = await call_subagent_task_tool(
        tool,
        description="do the thing",
        subagent_type="researcher",
        label="lbl",
        response_schema=None,
        runtime=runtime,
    )

    assert out == "hello world"
    assert [e["phase"] for e in rec.events] == ["start", "complete"]
    start, complete = rec.events
    assert start["type"] == "subagent"
    assert start["eval_id"] == "eval_call_123"
    assert start["subagent_type"] == "researcher"
    assert start["label"] == "lbl"
    assert start["description"] == "do the thing"
    # The per-dispatch id is stable across start/complete and is the fresh
    # child tool_call_id (not the parent eval id).
    assert start["id"] == complete["id"]
    assert start["id"].startswith("ptc_task_")
    assert tool.seen_runtime_tool_call_id == start["id"]
    assert isinstance(complete["duration_ms"], int)


async def test_emits_error_event_and_reraises() -> None:
    rec = _Recorder()
    runtime = _FakeRuntime(stream_writer=rec)
    tool = _FakeTaskTool(raise_exc=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await call_subagent_task_tool(
            tool,
            description="x",
            subagent_type="t",
            label="lbl",
            response_schema=None,
            runtime=runtime,
        )

    assert [e["phase"] for e in rec.events] == ["start", "error"]
    error = rec.events[1]
    assert error["error"] == "boom"
    assert error["id"] == rec.events[0]["id"]
    assert isinstance(error["duration_ms"], int)


async def test_description_is_truncated_in_event() -> None:
    rec = _Recorder()
    runtime = _FakeRuntime(stream_writer=rec)

    await call_subagent_task_tool(
        _FakeTaskTool("r"),
        description="a" * 500,
        subagent_type="t",
        label="lbl",
        response_schema=None,
        runtime=runtime,
    )

    assert len(rec.events[0]["description"]) == 200


async def test_label_is_truncated_in_event() -> None:
    rec = _Recorder()
    runtime = _FakeRuntime(stream_writer=rec)

    await call_subagent_task_tool(
        _FakeTaskTool("r"),
        description="x",
        subagent_type="t",
        label="L" * 500,
        response_schema=None,
        runtime=runtime,
    )

    assert len(rec.events[0]["label"]) == 120


async def test_writer_failure_does_not_break_dispatch() -> None:
    def boom_writer(_event: dict[str, Any]) -> None:
        msg = "writer down"
        raise RuntimeError(msg)

    runtime = _FakeRuntime(stream_writer=boom_writer)
    out = await call_subagent_task_tool(
        _FakeTaskTool("still works"),
        description="x",
        subagent_type="t",
        label="lbl",
        response_schema=None,
        runtime=runtime,
    )
    assert out == "still works"


async def test_missing_writer_is_a_noop() -> None:
    runtime = _FakeRuntime(stream_writer=None)
    out = await call_subagent_task_tool(
        _FakeTaskTool("ok"),
        description="x",
        subagent_type="t",
        label="lbl",
        response_schema=None,
        runtime=runtime,
    )
    assert out == "ok"
