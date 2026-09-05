"""Thread/loop-affinity regression tests for QuickJS PTC async dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves annotations at runtime
)
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware

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
    middleware: CodeInterpreterMiddleware,
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
        CodeInterpreterMiddleware(ptc=[current_loop_id]),
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


@pytest.mark.filterwarnings("ignore:The feature `forked subagents` is in beta")
def test_forked_subagent_clones_its_parent_repl() -> None:
    """A fork receives a copy of, not access to, its parent's REPL."""
    slot_ids: list[str] = []
    eval_outputs: list[str] = []

    class _RecordingMiddleware(CodeInterpreterMiddleware):
        def before_agent(
            self, state: dict[str, Any], runtime: Any
        ) -> dict[str, Any] | None:
            update = super().before_agent(state, runtime)
            slot_ids.append(self._slot_id({**state, **(update or {})}))
            return update

        def after_agent(
            self, state: dict[str, Any], runtime: Any
        ) -> dict[str, Any] | None:
            eval_outputs.extend(
                str(message.content)
                for message in state["messages"]
                if isinstance(message, ToolMessage) and message.name == "eval"
            )
            return super().after_agent(state, runtime)

    parent_model = _FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {"code": "globalThis.parentValue = 42"},
                            "id": "call_parent_eval",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Continue this conversation.",
                                "subagent_type": "worker",
                            },
                            "id": "call_worker",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {"code": "parentValue"},
                            "id": "call_parent_verify",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="parent done"),
            ]
        )
    )
    worker_model = _FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "eval",
                            "args": {"code": "parentValue += 1; parentValue"},
                            "id": "call_worker_eval",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="worker done"),
            ]
        )
    )
    agent = create_deep_agent(
        model=parent_model,
        checkpointer=InMemorySaver(),
        middleware=[_RecordingMiddleware()],
        system_prompt="",
        subagents=[
            {
                "name": "worker",
                "description": "Continues the current conversation.",
                "model": worker_model,
                "mode": "fork",
            }
        ],
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="start")]},
        config={
            "configurable": {"thread_id": "quickjs-fork-slot"},
            "recursion_limit": 100,
        },
    )

    assert result["messages"][-1].content == "parent done"
    assert any("43" in output for output in eval_outputs), eval_outputs
    assert any("42" in output for output in eval_outputs), eval_outputs
    assert len(slot_ids) == 2
    assert len(set(slot_ids)) == 2


async def test_quickjs_async_task_global_subagent_loop_affinity_e2e() -> None:
    """E2E: the top-level `task()` global runs compiled subagents on the caller loop.

    `task` is reserved and cannot be exposed via `ptc` (it is always the global),
    so the loop-affinity guarantee is exercised through the global dispatch path.
    """
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
    agent = create_deep_agent(
        model=_FakeChatModel(
            messages=_script(
                "await task({description: 'say hi', "
                "subagentType: 'researcher', label: 'greet'})",
                final_message="done",
            )
        ),
        subagents=[
            {
                "name": "researcher",
                "description": "returns one short answer",
                "runnable": subagent_runnable,
            }
        ],
        middleware=[
            CodeInterpreterMiddleware(),
        ],
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Use eval and call task for researcher")]},
        config={"configurable": {"thread_id": "task-global-loop-affinity"}},
    )
    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "subagent-ok")
    assert subagent_loop_ids
    assert all(loop_id == outer_loop_id for loop_id in subagent_loop_ids)


async def test_quickjs_task_global_available_without_ptc_e2e() -> None:
    """E2E: top-level `task()` works without exposing PTC tools."""

    def _subagent_sync(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="subagent-ok")]}

    async def _subagent_async(state: dict[str, Any], config: Any) -> dict[str, Any]:
        del state, config
        return {"messages": [AIMessage(content="subagent-ok")]}

    subagent_runnable = RunnableLambda(_subagent_sync, afunc=_subagent_async)
    agent = create_deep_agent(
        model=_FakeChatModel(
            messages=_script(
                "await task({description: 'say hi', "
                "subagentType: 'researcher', label: 'greet'})",
                final_message="done",
            )
        ),
        subagents=[
            {
                "name": "researcher",
                "description": "returns one short answer",
                "runnable": subagent_runnable,
            }
        ],
        middleware=[
            CodeInterpreterMiddleware(),
        ],
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content="Use eval and call the researcher through task.")
            ]
        },
        config={"configurable": {"thread_id": "task-global-no-ptc"}},
    )
    tool_message = _eval_tool_message(result)
    _assert_result_contains(tool_message.content, "subagent-ok")
