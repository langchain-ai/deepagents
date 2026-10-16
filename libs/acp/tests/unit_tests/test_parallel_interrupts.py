"""Permission routing for concurrent Deep Agents task interrupts."""

from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest
from acp.schema import (
    AllowedOutcome,
    PermissionOption,
    RequestPermissionResponse,
    TextContentBlock,
    ToolCallUpdate,
)
from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP
from tests.chat_model import GenericFakeChatModel
from tests.test_agent import FakeACPClient


class SelectiveACPClient(FakeACPClient):
    def __init__(self, rejected: tuple[str, ...]) -> None:
        super().__init__()
        self.rejected = rejected
        self.actions: list[str] = []

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        action = tool_call.raw_input["action"]
        self.actions.append(action)
        decision = "reject" if action in self.rejected else "approve"
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=decision)
        )


@pytest.mark.parametrize(
    ("action_counts", "rejected", "stream_interrupts"),
    [
        ((1,), (), "normal"),
        ((1, 1), (), "normal"),
        ((1, 2), (), "normal"),
        ((1, 2), ("right-1",), "normal"),
        ((1, 2), ("right-1",), "missing"),
        ((1, 2), ("right-1",), "repeated"),
    ],
    ids=["single", "parallel", "unequal", "mixed", "snapshot-fallback", "repeated-events"],
)
async def test_task_permissions_resume_each_interrupt(
    action_counts: tuple[int, ...],
    rejected: tuple[str, ...],
    stream_interrupts: Literal["normal", "missing", "repeated"],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    executed: list[str] = []

    @tool
    def record_action(action: str) -> str:
        """Record an approved action without external side effects."""
        executed.append(action)
        return action

    subagents: list[SubAgent] = []
    actions: list[str] = []
    for name, count in zip(("left", "right"), action_counts, strict=False):
        calls = [
            {
                "name": "record_action",
                "args": {"action": f"{name}-{index}"},
                "id": f"{name}-{index}",
            }
            for index in range(count)
        ]
        actions.extend(call["args"]["action"] for call in calls)
        subagents.append(
            {
                "name": name,
                "description": name,
                "tools": [record_action],
                "interrupt_on": {"record_action": True},
                "model": GenericFakeChatModel(
                    messages=iter(
                        [AIMessage(content="", tool_calls=calls), AIMessage(content="done")]
                    )
                ),
            }
        )
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {"description": agent["name"], "subagent_type": agent["name"]},
                            "id": f"task-{agent['name']}",
                        }
                        for agent in subagents
                    ],
                ),
                AIMessage(content="all done"),
            ]
        )
    )
    graph = create_deep_agent(model=model, subagents=subagents, checkpointer=MemorySaver())
    stream = graph.astream

    async def filtered_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for chunk in stream(*args, **kwargs):
            is_interrupt = chunk[1] == "updates" and "__interrupt__" in chunk[2]
            if is_interrupt and stream_interrupts == "missing":
                continue
            yield chunk
            if is_interrupt and stream_interrupts == "repeated":
                yield chunk

    monkeypatch.setattr(graph, "astream", filtered_stream)
    server = AgentServerACP(graph)
    client = SelectiveACPClient(rejected)
    server.on_connect(client)
    session = await server.new_session(cwd=str(tmp_path), mcp_servers=[])

    response = await server.prompt(
        [TextContentBlock(type="text", text="Run both tasks")], session.session_id
    )

    assert response.stop_reason == "end_turn"
    assert sorted(client.actions) == sorted(actions)
    assert sorted(executed) == sorted(action for action in actions if action not in rejected)
    state = await graph.aget_state({"configurable": {"thread_id": session.session_id}})
    assert not state.interrupts
    assert not state.next
