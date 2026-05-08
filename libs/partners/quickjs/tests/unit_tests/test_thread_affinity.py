"""Thread/loop-affinity regression tests for QuickJS PTC async dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves annotations at runtime
)
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from pydantic import Field

from langchain_quickjs import REPLMiddleware

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel whose bind_tools returns self."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> _FakeChatModel:
        del tools
        return self


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


@tool("current_loop_id")
async def current_loop_id() -> str:
    """Return the running loop id for loop-affinity assertions."""
    await asyncio.sleep(0)
    return str(id(asyncio.get_running_loop()))


async def test_quickjs_async_ptc_runs_tools_on_outer_loop() -> None:
    """PTC tool coroutines run on the caller loop, not the worker loop."""
    outer_loop_id = str(id(asyncio.get_running_loop()))
    result = await _make_agent(
        "await tools.currentLoopId({})",
        REPLMiddleware(ptc=[current_loop_id]),
    ).ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Use the eval tool and call currentLoopId via PTC."
                )
            ]
        }
    )

    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, outer_loop_id)


async def test_quickjs_async_ptc_task_subagent_loop_affinity_e2e() -> None:
    """E2E: PTC `task` runs compiled subagents on the caller loop."""
    outer_loop_id = id(asyncio.get_running_loop())
    owner_loop = asyncio.get_running_loop()
    subagent_loop_ids: list[int] = []

    def _subagent_sync(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="subagent-ok")]}

    async def _subagent_async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        subagent_loop_ids.append(id(asyncio.get_running_loop()))
        # Affine guard: awaiting this from a different loop raises the
        # same "Future attached to a different loop" runtime error.
        future = owner_loop.create_future()
        owner_loop.call_soon_threadsafe(future.set_result, None)
        await future
        return {"messages": [AIMessage(content="subagent-ok")]}

    subagent_runnable = RunnableLambda(_subagent_sync, afunc=_subagent_async)
    agent = create_agent(
        model=_FakeChatModel(
            messages=_script(
                "await tools.task({"
                "description: 'say hi', subagent_type: 'researcher'"
                "})",
                final_message="done",
            )
        ),
        middleware=[
            SubAgentMiddleware(
                backend=None,
                subagents=[
                    {
                        "name": "researcher",
                        "description": "returns one short answer",
                        "runnable": subagent_runnable,
                    }
                ],
            ),
            REPLMiddleware(ptc=["task"]),
        ],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use eval and call task for researcher")]},
        config={"configurable": {"thread_id": "ptc-task-loop-affinity"}},
    )
    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "subagent-ok")
    assert subagent_loop_ids
    assert all(loop_id == outer_loop_id for loop_id in subagent_loop_ids)
