"""Tests for tool exclusion middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deepagents.middleware._tool_exclusion import _ToolExclusionMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _request_with_prompt(prompt: str) -> ModelRequest:
    return ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        messages=[HumanMessage(content="hi")],
        system_message=SystemMessage(content=prompt),
        tools=[
            {"name": "ls"},
            {"name": "read_file"},
            {"name": "write_file"},
            {"name": "edit_file"},
            {"name": "glob"},
            {"name": "grep"},
            {"name": "execute"},
            {"name": "task"},
        ],
    )


def test_excluded_tools_are_removed_from_tools_and_system_prompt() -> None:
    middleware = _ToolExclusionMiddleware(excluded=frozenset({"write_file", "edit_file", "execute", "task"}))
    request = _request_with_prompt(
        """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

- ls: list files in a directory
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern
- grep: search for text within files

## Execute Tool `execute`

Run shell commands.

## `task` (subagent spawner)

Use subagents.

## Keep Me

Other instructions."""
    )
    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handler)

    assert [_tool["name"] for _tool in captured[0].tools] == ["ls", "read_file", "glob", "grep"]
    assert captured[0].system_message is not None
    prompt = captured[0].system_message.text
    assert "write_file" not in prompt
    assert "edit_file" not in prompt
    assert "Execute Tool" not in prompt
    assert "subagent spawner" not in prompt
    assert "## Keep Me" in prompt


@pytest.mark.asyncio
async def test_async_excluded_tools_are_removed_from_tools_and_system_prompt() -> None:
    middleware = _ToolExclusionMiddleware(excluded=frozenset({"write_file"}))
    request = _request_with_prompt(
        """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

- ls: list files in a directory
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem"""
    )
    captured: list[ModelRequest] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    await middleware.awrap_model_call(request, handler)

    assert [_tool["name"] for _tool in captured[0].tools] == ["ls", "read_file", "edit_file", "glob", "grep", "execute", "task"]
    assert captured[0].system_message is not None
    prompt = captured[0].system_message.text
    assert "write_file" not in prompt
    assert "edit_file" in prompt
