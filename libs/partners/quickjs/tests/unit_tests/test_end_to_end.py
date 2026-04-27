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

from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
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


@tool
def echo_foo(foo: str) -> str:
    """Echo the value of `foo`."""
    return f"got {foo}"


def _script() -> Iterator[AIMessage]:
    return iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": _EVAL_CODE},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done."),
        ]
    )


def _make_agent() -> Any:
    return create_deep_agent(
        model=_FakeChatModel(messages=_script()),
        middleware=[REPLMiddleware(ptc=[list_user_ids])],
    )


def _eval_tool_message(result: dict) -> ToolMessage:
    messages = [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "eval"
    ]
    assert messages, "expected at least one eval ToolMessage"
    return messages[-1]


def _assert_no_error(content: str) -> None:
    assert "ConcurrentEval" not in content, content
    assert "<error" not in content, content


def test_sync_ptc_eval_through_repl() -> None:
    """``invoke`` path: the observed production snippet must not error."""
    result = _make_agent().invoke({"messages": [HumanMessage(content="go")]})
    _assert_no_error(_eval_tool_message(result).content)


async def test_async_ptc_eval_through_repl() -> None:
    """``ainvoke`` path: same guard on the async handler."""
    result = await _make_agent().ainvoke({"messages": [HumanMessage(content="go")]})
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
