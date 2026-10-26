"""MCP deadlines across the CLI's production agent stacks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.agent import create_cli_agent
from deepagents_code.hooks.interrupt import (
    build_hook_resume_value,
    parse_hook_interrupt_payload,
)
from deepagents_code.hooks.models.domain import (
    HookEvent,
    PostToolUseFailureDecision,
    PostToolUseFailureEvent,
)
from deepagents_code.hooks.models.transport import HookInvocationResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ToolCallingModel:
        _ = tools, tool_choice, kwargs
        return self


@pytest.mark.parametrize(
    ("subagent", "forked"),
    [("researcher", "0"), ("general-purpose", "0"), ("general-purpose", "1")],
)
async def test_delegated_mcp_call_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subagent: str, forked: str
) -> None:
    """Both custom and general-purpose subagents recover from stalled MCP calls."""
    cancelled = asyncio.Event()

    @tool
    async def stalled_mcp() -> str:
        """Simulate an MCP server that never responds."""
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    stalled_mcp.metadata = {"_deepagents_code_mcp": True}
    monkeypatch.setenv("DEEPAGENTS_CODE_FORKED_SUBAGENTS", forked)
    monkeypatch.setattr(
        "deepagents_code.config_manifest.resolve_mcp_tool_timeout", lambda: 0.01
    )
    monkeypatch.setattr(
        "deepagents_code.agent.list_subagents",
        lambda **_kwargs: [
            {
                "name": "researcher",
                "description": "Research a task",
                "system_prompt": "Research the task.",
                "model": None,
            }
        ],
    )
    model = _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "research",
                                "subagent_type": subagent,
                            },
                            "id": "delegate",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "stalled_mcp", "args": {}, "id": "mcp"}],
                ),
                AIMessage(content="child recovered"),
                AIMessage(content="parent recovered"),
            ]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "mcp-timeout-test",
        cwd=tmp_path,
        tools=[stalled_mcp],
        mcp_tools=[stalled_mcp],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
    )

    result = await asyncio.wait_for(
        graph.ainvoke({"messages": [HumanMessage(content="delegate research")]}),
        timeout=3,
    )

    assert cancelled.is_set(), [(m.type, m.content) for m in result["messages"]]
    assert result["messages"][-1].content == "parent recovered"


async def test_mcp_timeout_triggers_failure_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout reaches failure hooks, and resuming does not rerun the tool."""
    calls = 0

    @tool
    async def stalled_mcp() -> str:
        """Simulate an MCP server that never responds."""
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()
        return "unreachable"

    stalled_mcp.metadata = {"_deepagents_code_mcp": True}
    monkeypatch.setattr(
        "deepagents_code.config_manifest.resolve_mcp_tool_timeout", lambda: 0.01
    )
    model = _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "stalled_mcp", "args": {}, "id": "mcp"}],
                ),
                AIMessage(content="recovered"),
            ]
        ),
        profile={"max_input_tokens": 200000},
        disable_streaming=True,
    )
    graph, _ = create_cli_agent(
        model,
        "mcp-failure-test",
        cwd=tmp_path,
        tools=[stalled_mcp],
        mcp_tools=[stalled_mcp],
        auto_approve=True,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "mcp-failure-test"}}
    context = CLIContextSchema(
        thread_id="mcp-failure-test",
        hooks_snapshot_id="snap",
        hooks_server_events=[HookEvent.POST_TOOL_USE_FAILURE.value],
    )
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="call MCP")]}, config=config, context=context
    )

    assert result.get("__interrupt__"), "Timeout skipped PostToolUseFailure"
    request = parse_hook_interrupt_payload(result["__interrupt__"][0].value)
    assert request is not None
    event = request.invocation.event
    assert isinstance(event, PostToolUseFailureEvent)
    assert event.call.id == "mcp"
    assert "timed out" in event.error
    response = HookInvocationResponse(
        protocol_version=1,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
        decision=PostToolUseFailureDecision(event=HookEvent.POST_TOOL_USE_FAILURE),
    )
    result = await graph.ainvoke(
        Command(resume=build_hook_resume_value(response)),
        config=config,
        context=context,
    )

    assert calls == 1
    assert result["messages"][-1].content == "recovered"
