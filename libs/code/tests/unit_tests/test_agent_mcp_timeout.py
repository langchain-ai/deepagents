"""MCP deadlines across the CLI's production agent stacks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deepagents_code.agent import create_cli_agent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

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
