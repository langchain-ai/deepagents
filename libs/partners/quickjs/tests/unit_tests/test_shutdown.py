"""Interpreter release on runs that skip `after_agent`, and at process exit."""

from __future__ import annotations

import gc
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

# Turn 1 runs `eval`, which opens an interpreter. Turn 2 ends the run the way
# the scenario says: normally, through `interrupt()`, or with a tool exception.
_EXIT_SCRIPT = textwrap.dedent(
    """
    import sys

    from deepagents import create_deep_agent
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt

    from langchain_quickjs import CodeInterpreterMiddleware
    from tests._common import FakeChatModel


    @tool
    def ask_human(question: str) -> str:
        \"\"\"Suspend the run until a human answers.\"\"\"
        return interrupt({"question": question})


    @tool
    def broken_tool(x: str) -> str:
        \"\"\"Always fails.\"\"\"
        raise RuntimeError("boom")


    scenario = sys.argv[1]
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "eval", "args": {"code": "1 + 1"}}],
        )
    ]
    second = {"interrupt": "ask_human", "error": "broken_tool"}.get(scenario)
    if second:
        args = {"question": "ok?"} if second == "ask_human" else {"x": "1"}
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"id": "c2", "name": second, "args": args}],
            )
        )
    messages.append(AIMessage(content="done"))

    agent = create_deep_agent(
        model=FakeChatModel(messages=iter(messages)),
        tools=[ask_human, broken_tool],
        middleware=[CodeInterpreterMiddleware(ptc=[], timeout=5.0)],
        checkpointer=InMemorySaver(),
    )
    try:
        agent.invoke(
            {"messages": [("user", "go")]},
            config={"configurable": {"thread_id": "t1"}},
        )
    except RuntimeError:
        pass
    print("exiting", flush=True)
    """
)


@tool
def ask_human(question: str) -> str:
    """Suspend the run until a human answers."""
    return interrupt({"question": question})


def _worker_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name.startswith("quickjs-worker")}


def _interrupting_agent(
    middleware: CodeInterpreterMiddleware,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    messages = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "eval", "args": {"code": "1 + 1"}}],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c2", "name": "ask_human", "args": {"question": "ok?"}}
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    return create_deep_agent(
        model=FakeChatModel(messages=messages),
        tools=[ask_human],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )


_PACKAGE_ROOT = Path(__file__).parents[2]
_CONFIG = {"configurable": {"thread_id": "t1"}}


@pytest.mark.parametrize("scenario", ["complete", "interrupt", "error"])
def test_process_exits_after_run(scenario: str) -> None:
    proc = subprocess.run(  # noqa: S603  # fixed script run with this interpreter
        [sys.executable, "-c", _EXIT_SCRIPT, scenario],
        cwd=_PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "exiting" in proc.stdout


def test_close_releases_interpreters_after_interrupt() -> None:
    before = _worker_threads()
    middleware = CodeInterpreterMiddleware(ptc=[])
    result = _interrupting_agent(middleware).invoke(
        {"messages": [("user", "go")]}, config=_CONFIG
    )
    assert "__interrupt__" in result
    assert _worker_threads() - before

    middleware.close()
    assert not _worker_threads() - before
    middleware.close()


async def test_aclose_releases_interpreters_after_interrupt() -> None:
    before = _worker_threads()
    middleware = CodeInterpreterMiddleware(ptc=[])
    result = await _interrupting_agent(middleware).ainvoke(
        {"messages": [("user", "go")]}, config=_CONFIG
    )
    assert "__interrupt__" in result
    assert _worker_threads() - before

    await middleware.aclose()
    assert not _worker_threads() - before
    await middleware.aclose()


def test_garbage_collection_releases_interpreters() -> None:
    before = _worker_threads()
    middleware = CodeInterpreterMiddleware(ptc=[])
    agent = _interrupting_agent(middleware)
    agent.invoke({"messages": [("user", "go")]}, config=_CONFIG)
    assert _worker_threads() - before

    del agent, middleware
    gc.collect()
    assert not _worker_threads() - before
