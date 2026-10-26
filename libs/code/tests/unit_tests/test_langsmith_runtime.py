"""Tests for the inline LangSmith runtime override in `_langsmith_runtime`."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import TYPE_CHECKING, Any

import pytest
from langsmith import Client, traceable, tracing_context
from langsmith._internal import _aiter
from langsmith.run_helpers import get_current_run_tree

from deepagents_code._langsmith_runtime import install_untraced_inline_override

if TYPE_CHECKING:
    from langsmith.run_trees import RunTree


@pytest.fixture
def no_executor(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every `run_in_executor` call on the running loop.

    Returns:
        The list that collects the submitted callables.
    """
    calls: list[object] = []
    original = asyncio.BaseEventLoop.run_in_executor

    def record(self: asyncio.BaseEventLoop, executor: Any, func: Any, *args: Any):  # noqa: ANN202, ANN401
        calls.append(func)
        return original(self, executor, func, *args)

    monkeypatch.setattr(asyncio.BaseEventLoop, "run_in_executor", record)
    return calls


@traceable(name="hook")
async def _hook() -> int:  # noqa: RUF029  # exercises the async `traceable` wrapper
    return 2


@traceable(name="failing_hook")
async def _failing_hook() -> None:  # noqa: RUF029  # exercises the async `traceable` wrapper
    msg = "boom"
    raise ValueError(msg)


async def test_untraced_call_skips_executor(no_executor: list[object]) -> None:
    install_untraced_inline_override()
    with tracing_context(enabled=False):
        assert await _hook() == 2
    assert no_executor == []


async def test_untraced_error_still_propagates(no_executor: list[object]) -> None:
    install_untraced_inline_override()
    with tracing_context(enabled=False), pytest.raises(ValueError, match="boom"):
        await _failing_hook()
    assert no_executor == []


async def test_other_calls_keep_thread_hop() -> None:
    """Calls that are not `traceable` bookkeeping can do network I/O."""
    install_untraced_inline_override()
    with tracing_context(enabled=False):
        worker = await _aio_to_thread(threading.get_ident)
    assert worker != threading.get_ident()


async def _aio_to_thread(func: Any) -> Any:  # noqa: ANN401
    return await _aiter.aio_to_thread(contextvars.copy_context(), func)


async def test_traced_runs_keep_nesting(no_executor: list[object]) -> None:
    """With tracing on, runs still hop to a thread and nest as before."""
    install_untraced_inline_override()
    finished: list[RunTree] = []
    seen_parent: list[object] = []

    @traceable(name="inner")
    async def inner() -> None:  # noqa: RUF029  # exercises the async `traceable` wrapper
        run = get_current_run_tree()
        seen_parent.append(run.parent_run_id if run else None)

    @traceable(name="outer")
    async def outer() -> None:
        await inner()

    with tracing_context(
        enabled="local", client=Client(api_url="http://localhost:1984")
    ):
        await outer(langsmith_extra={"on_end": finished.append})

    assert seen_parent == [finished[0].id]
    assert [child.name for child in finished[0].child_runs] == ["inner"]
    assert no_executor
