"""Async end-to-end tests for ``REPLMiddleware`` with a fake LLM.

Covers the same integration surfaces as the prior quickjs e2e suite:
agent wiring, REPL execution, PTC tool calls, runtime propagation,
error surfacing, and concurrent agent runs.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # tool decorator resolves type hints at import time
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import Field

from langchain_quickjs import REPLMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel whose bind_tools returns self.

    Without the override, ``create_deep_agent``'s bind_tools call replaces
    the model with a RunnableBinding whose ``_generate`` no longer reads
    from the pre-scripted iterator.
    """

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _FakeChatModel:
        return self


@tool
def list_user_ids() -> list[str]:
    """Return example user identifiers for QuickJS bridging tests."""
    return ["user_1", "user_2", "user_3"]


@tool("sync_label")
def sync_label_tool(value: str) -> str:
    """Return a labeled value from a synchronous LangChain tool."""
    return f"sync:{value}"


@tool("async_label")
async def async_label_tool(value: str) -> str:
    """Return a labeled value from an asynchronous LangChain tool."""
    await asyncio.sleep(0)
    return f"async:{value}"


@tool("runtime_marker")
def runtime_marker(value: str, runtime: ToolRuntime) -> str:
    """Return runtime metadata for testing ToolRuntime injection."""
    return (
        f"{value}:{runtime.tool_call_id}:{runtime.config['metadata']['langgraph_node']}"
    )


@tool("runtime_configurable")
def runtime_configurable(value: str, runtime: ToolRuntime) -> str:
    """Return configurable runtime data for testing ToolRuntime context propagation."""
    return f"{value}:{runtime.config['configurable']['user_id']}"


@tool("always_fails")
async def always_fails(value: str) -> str:
    """Raise to verify host-call failures surface to the model."""
    await asyncio.sleep(0)
    msg = f"boom:{value}"
    raise RuntimeError(msg)


def _script(code: str, *, final_message: str = "done") -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content=final_message),
        ]
    )


def _make_agent(
    code: str,
    middleware: REPLMiddleware,
    *,
    final_message: str = "done",
) -> Any:
    return create_deep_agent(
        model=_FakeChatModel(messages=_script(code, final_message=final_message)),
        middleware=[middleware],
    )


def _eval_tool_message(result: dict[str, Any]) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _assert_result_contains(content: str, expected: str) -> None:
    assert "<error" not in content, content
    assert "<result" in content, content
    assert expected in content, content


async def test_deepagent_with_quickjs_interpreter() -> None:
    """Basic async test with QuickJS interpreter."""
    result = await _make_agent(
        "6 * 7",
        REPLMiddleware(),
        final_message="The answer is 42.",
    ).ainvoke(
        {"messages": [HumanMessage(content="Use the eval tool to calculate 6 * 7")]}
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "42")
    assert result["messages"][-1].content == "The answer is 42."


async def test_deepagent_with_quickjs_json_roundtrip_foreign_function() -> None:
    """Verify async eval can parse JSON strings returned from PTC tool calls."""
    code = (
        "const idsJson = await tools.listUserIds({});\n"
        "const ids = JSON.parse(idsJson);\n"
        "ids.join(',');"
    )
    result = await _make_agent(code, REPLMiddleware(ptc=[list_user_ids])).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to print the available user ids"
                )
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "user_1,user_2,user_3")


async def test_deepagent_with_quickjs_async_foreign_function() -> None:
    """Verify the eval tool can call sync and async LangChain tools in one run."""
    code = (
        "const [left, right] = await Promise.all([\n"
        "  tools.syncLabel({value: 'left'}),\n"
        "  tools.asyncLabel({value: 'right'}),\n"
        "]);\n"
        "`${left}|${right}`;"
    )
    result = await _make_agent(
        code,
        REPLMiddleware(ptc=[sync_label_tool, async_label_tool]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool to call sync and async tools")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "sync:left|async:right")


async def test_quickjs_async_timeout_error() -> None:
    """Verify the async eval path surfaces QuickJS eval timeouts."""
    result = await _make_agent(
        "while (true) {}",
        REPLMiddleware(timeout=1),
        final_message="timeout hit",
    ).ainvoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool and keep looping until timeout")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    assert '<error type="Timeout">' in tool_message.content
    assert result["messages"][-1].content == "timeout hit"


async def test_quickjs_async_tool_exception() -> None:
    """Verify async tool exceptions surface as eval errors."""
    result = await _make_agent(
        "await tools.alwaysFails({value: 'x'})",
        REPLMiddleware(ptc=[always_fails]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to call the async tool that raises"
                )
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    assert '<error type="HostError">' in tool_message.content
    assert "Host function failed" in tool_message.content


async def test_quickjs_async_toolruntime_foreign_function() -> None:
    """Verify async PTC tool calls inherit eval ToolRuntime metadata."""
    result = await _make_agent(
        "await tools.runtimeMarker({value: 'value'})",
        REPLMiddleware(ptc=[runtime_marker]),
    ).ainvoke(
        {"messages": [HumanMessage(content="Use the eval tool to inspect the runtime")]}
    )

    tool_message = _eval_tool_message(result)
    assert tool_message.content.startswith("<result>value:ptc_runtime_marker_")
    assert tool_message.content.endswith(":tools</result>")


async def test_quickjs_async_toolruntime_configurable_foreign_function() -> None:
    """Verify async PTC tool calls see configurable runtime data."""
    result = await _make_agent(
        "await tools.runtimeConfigurable({value: 'value'})",
        REPLMiddleware(ptc=[runtime_configurable]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to inspect configurable runtime"
                )
            ]
        },
        config={"configurable": {"user_id": "user-123"}},
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "value:user-123")


async def test_quickjs_async_parallel_agents() -> None:
    """Verify many async agents can run in parallel with REPL middleware."""

    async def _run_agent(index: int) -> tuple[int, dict[str, object]]:
        result = await _make_agent(
            f"{index} * 10",
            REPLMiddleware(),
            final_message=f"done-{index}",
        ).ainvoke(
            {
                "messages": [
                    HumanMessage(content=f"Use the eval tool to multiply {index} by 10")
                ]
            }
        )
        return index, result

    runs = await asyncio.gather(*(_run_agent(index) for index in range(25)))

    assert len(runs) == 25
    for index, result in runs:
        tool_message = _eval_tool_message(result)
        _assert_result_contains(tool_message.content, str(index * 10))
        assert result["messages"][-1].content == f"done-{index}"
