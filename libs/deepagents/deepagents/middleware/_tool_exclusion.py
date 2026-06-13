"""Middleware for filtering excluded tools from model requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import SystemMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.messages import AIMessage
    from langchain_core.tools import BaseTool

_FILESYSTEM_TOOL_NAMES = frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep"})


def _tool_name(tool: BaseTool | dict[str, str]) -> str | None:
    """Extract tool name from a `BaseTool` or dict tool."""
    if isinstance(tool, dict):
        name = tool.get("name")
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _remove_markdown_section(text: str, heading: str) -> str:
    """Remove a markdown section by exact heading."""
    start = text.find(heading)
    if start == -1:
        return text
    next_section = text.find("\n\n## ", start + len(heading))
    if next_section == -1:
        return text[:start].rstrip()
    return (text[:start].rstrip() + "\n\n" + text[next_section:].lstrip()).strip()


def _remove_filesystem_tool_prompt_entries(text: str, excluded: frozenset[str]) -> str:
    """Remove excluded filesystem tools from the filesystem prompt block."""
    filesystem_excluded = excluded & _FILESYSTEM_TOOL_NAMES
    if not filesystem_excluded:
        return text

    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Filesystem Tools "):
            available = [name for name in ("ls", "read_file", "write_file", "edit_file", "glob", "grep") if name not in filesystem_excluded]
            if not available:
                continue
            tools = ", ".join(f"`{name}`" for name in available)
            lines.append(f"## Filesystem Tools {tools}")
            continue
        if any(line.startswith(f"- {name}:") for name in filesystem_excluded):
            continue
        lines.append(line)
    return "\n".join(lines)


def _filter_system_prompt(system_message: SystemMessage | None, excluded: frozenset[str]) -> SystemMessage | None:
    """Remove built-in tool instructions for tools hidden from the model."""
    if system_message is None or not excluded:
        return system_message

    changed = False
    content_blocks: list[Any] = []
    for block in system_message.content_blocks:
        block_data: Any = block
        if not isinstance(block_data, dict) or block_data.get("type") != "text" or not isinstance(block_data.get("text"), str):
            content_blocks.append(block)
            continue

        text = block_data["text"]
        filtered_text = _remove_filesystem_tool_prompt_entries(text, excluded)
        if "execute" in excluded:
            filtered_text = _remove_markdown_section(filtered_text, "## Execute Tool `execute`")
        if "task" in excluded:
            filtered_text = _remove_markdown_section(filtered_text, "## `task` (subagent spawner)")

        if filtered_text != text:
            changed = True
        if filtered_text.strip():
            content_blocks.append({**block_data, "text": filtered_text})
        else:
            changed = True

    if not changed:
        return system_message
    return SystemMessage(content_blocks=content_blocks)


class _ToolExclusionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Middleware that filters excluded tools from the model request.

    Should be placed late in the middleware stack (after all
    tool-injecting middleware) so it can strip middleware-injected tools
    (filesystem, subagent, etc.) that the harness profile marks as excluded.

    Args:
        excluded: Tool names to remove before the model sees them.
    """

    def __init__(self, *, excluded: frozenset[str]) -> None:
        self._excluded = excluded

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Filter excluded tools before they reach the model."""
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(
                tools=filtered,
                system_message=_filter_system_prompt(request.system_message, self._excluded),
            )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        if self._excluded:
            filtered = [t for t in request.tools if _tool_name(t) not in self._excluded]
            request = request.override(
                tools=filtered,
                system_message=_filter_system_prompt(request.system_message, self._excluded),
            )
        return await handler(request)
