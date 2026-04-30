"""End-to-end tests for ``REPLMiddleware`` with a fake LLM.

Regression gate for the sync tool handler: before the worker-thread
refactor, sync ``invoke`` ran ``ctx.eval``, which cannot dispatch async
host functions (PTC bridges are ``is_async=True``). Any eval that
referenced ``tools.*`` surfaced as:

    <error type="ConcurrentEval">sync ctx.eval dispatched a registered
    async host function; use ctx.eval_async for code that awaits async
    host calls</error>

We reproduce the exact production snippet on both the sync and async
handlers and assert it no longer errors.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest
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

# The exact snippet a model produced in production when this regressed.
_EVAL_CODE = "var result = tools.listUserIds({}); result;"


class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel whose bind_tools returns self.

    Without the override, ``create_deep_agent``'s bind_tools call replaces
    the model with a RunnableBinding whose ``_generate`` no longer reads
    from our pre-scripted iterator.
    """

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _FakeChatModel:
        return self


@tool
def list_user_ids() -> list[int]:
    """List user IDs."""
    return [1, 21, 35, 41, 42, 43]


@tool("sync_label")
def sync_label_tool(value: str) -> str:
    """Return a labeled value from a synchronous LangChain tool."""
    return f"sync:{value}"


@tool("async_label")
async def async_label_tool(value: str) -> str:
    """Return a labeled value from an asynchronous LangChain tool."""
    await asyncio.sleep(0)
    return f"async:{value}"


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


@tool
def echo_foo(foo: str) -> str:
    """Echo the value of `foo`."""
    return f"got {foo}"


def _script(code: str, *, final_message: str = "Done.") -> Iterator[AIMessage]:
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
    final_message: str = "Done.",
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


def _assert_no_error(content: str) -> None:
    assert "ConcurrentEval" not in content, content
    assert "<error" not in content, content


def _assert_result_contains(content: str, expected: str) -> None:
    assert "<error" not in content, content
    assert "<result" in content, content
    assert expected in content, content


def test_deepagent_with_quickjs_interpreter_sync() -> None:
    """Basic sync test with QuickJS interpreter."""
    result = _make_agent(
        "6 * 7",
        REPLMiddleware(),
        final_message="The answer is 42.",
    ).invoke(
        {"messages": [HumanMessage(content="Use the eval tool to calculate 6 * 7")]}
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "42")
    assert result["messages"][-1].content == "The answer is 42."


def test_deepagent_with_quickjs_json_roundtrip_foreign_function_sync() -> None:
    """Verify sync eval can parse JSON strings returned from PTC tool calls."""
    code = (
        "const idsJson = await tools.listUserIds({});\n"
        "const ids = JSON.parse(idsJson);\n"
        "ids.join(',');"
    )
    result = _make_agent(code, REPLMiddleware(ptc=[list_user_ids])).invoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool to print the available user ids"
                )
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "1,21,35,41,42,43")


def test_deepagent_with_quickjs_mixed_foreign_function_sync() -> None:
    """Verify sync eval can call sync and async LangChain tools in one run."""
    code = (
        "const [left, right] = await Promise.all([\n"
        "  tools.syncLabel({value: 'left'}),\n"
        "  tools.asyncLabel({value: 'right'}),\n"
        "]);\n"
        "`${left}|${right}`;"
    )
    result = _make_agent(
        code,
        REPLMiddleware(ptc=[sync_label_tool, async_label_tool]),
    ).invoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool to call sync and async tools")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "sync:left|async:right")


def test_quickjs_sync_timeout_error() -> None:
    """Verify sync eval path surfaces QuickJS eval timeouts."""
    result = _make_agent(
        "while (true) {}",
        REPLMiddleware(timeout=1),
        final_message="timeout hit",
    ).invoke(
        {
            "messages": [
                HumanMessage(content="Use the eval tool and keep looping until timeout")
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    assert '<error type="Timeout">' in tool_message.content
    assert result["messages"][-1].content == "timeout hit"


def test_quickjs_sync_tool_exception() -> None:
    """Verify sync tool exceptions surface as eval errors."""
    result = _make_agent(
        "await tools.alwaysFails({value: 'x'})",
        REPLMiddleware(ptc=[always_fails]),
    ).invoke(
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


def test_quickjs_sync_host_call_budget_exceeded() -> None:
    """Verify sync eval surfaces per-eval PTC call budget exhaustion."""
    result = _make_agent(
        "await tools.syncLabel({value: 'a'});\n"
        "await tools.syncLabel({value: 'b'});\n"
        "'done'",
        REPLMiddleware(ptc=[sync_label_tool], max_ptc_calls=1),
    ).invoke(
        {"messages": [HumanMessage(content="Use eval and call sync_label twice.")]}
    )
    tool_message = _eval_tool_message(result)
    assert '<error type="PTCCallBudgetExceeded">' in tool_message.content
    assert "limit=1" in tool_message.content


def test_quickjs_sync_toolruntime_configurable_foreign_function() -> None:
    """Verify sync PTC tool calls see configurable runtime data."""
    result = _make_agent(
        "await tools.runtimeConfigurable({value: 'value'})",
        REPLMiddleware(ptc=[runtime_configurable]),
    ).invoke(
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


def test_quickjs_sync_parallel_agents_across_threads() -> None:
    """Verify many sync agents can run in parallel threads with REPL middleware."""

    def _run_agent(index: int) -> tuple[int, dict[str, object]]:
        result = _make_agent(
            f"{index} * 10",
            REPLMiddleware(),
            final_message=f"done-{index}",
        ).invoke(
            {
                "messages": [
                    HumanMessage(content=f"Use the eval tool to multiply {index} by 10")
                ]
            }
        )
        return index, result

    with ThreadPoolExecutor(max_workers=8) as executor:
        runs = list(executor.map(_run_agent, range(12)))

    assert len(runs) == 12
    for index, result in runs:
        tool_message = _eval_tool_message(result)
        _assert_result_contains(tool_message.content, str(index * 10))
        assert result["messages"][-1].content == f"done-{index}"


def test_sync_ptc_eval_through_repl() -> None:
    """``invoke`` path: the observed production snippet must not error."""
    result = _make_agent(_EVAL_CODE, REPLMiddleware(ptc=[list_user_ids])).invoke(
        {"messages": [HumanMessage(content="go")]}
    )
    _assert_no_error(_eval_tool_message(result).content)


async def test_async_ptc_eval_through_repl() -> None:
    """``ainvoke`` path: same guard on the async handler."""
    result = await _make_agent(
        _EVAL_CODE,
        REPLMiddleware(ptc=[list_user_ids]),
    ).ainvoke({"messages": [HumanMessage(content="go")]})
    _assert_no_error(_eval_tool_message(result).content)


@pytest.mark.xfail
def test_wrong_arg_name_surfaces_to_model() -> None:
    """Document what the model sees when JS calls a tool with a misspelled arg."""
    agent = create_deep_agent(
        model=_FakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "eval",
                                "args": {
                                    "code": 'await tools.echoFoo({not_foo: "x"})',
                                },
                                "id": "call_1",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        ),
        middleware=[REPLMiddleware(ptc=[echo_foo])],
    )
    result = agent.invoke({"messages": [HumanMessage(content="go")]})
    content = _eval_tool_message(result).content
    assert "Host function failed" not in content
