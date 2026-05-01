"""Helpers for inspecting and rewriting `create_deep_agent` tool inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def _tool_name(tool: BaseTool | Callable | dict[str, Any]) -> str | None:
    """Extract the tool name from any supported tool type.

    Args:
        tool: A tool in any of the forms accepted by `create_deep_agent`.

    Returns:
        The tool name, or `None` if it cannot be determined.
    """
    if isinstance(tool, dict):
        name = tool.get("name")  # ty: ignore[invalid-argument-type]  # Callable & dict intersection confuses ty
        return name if isinstance(name, str) else None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _apply_tool_description_overrides(
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    overrides: Mapping[str, str],
) -> list[BaseTool | Callable | dict[str, Any]] | None:
    """Apply description overrides without mutating caller-owned tools.

    Only dict tools and `BaseTool` instances are rewritten. Plain callables are
    returned unchanged because safely replacing their descriptions would require
    wrapping them in new tool objects.

    Args:
        tools: User-supplied tools to copy and possibly rewrite.
        overrides: Description overrides keyed by tool name.

    Returns:
        A copied tool list with supported overrides applied, or `None`.
    """
    if tools is None:
        return None

    copied_tools: list[BaseTool | Callable | dict[str, Any]] = []
    for tool in tools:
        name = _tool_name(tool)
        override = overrides.get(name) if name is not None else None
        if override is None:
            copied_tools.append(tool)
            continue
        if isinstance(tool, dict):
            rewritten_tool = cast("dict[str, Any]", tool).copy()
            rewritten_tool["description"] = override
            copied_tools.append(rewritten_tool)
            continue
        if isinstance(tool, BaseTool):
            copied_tools.append(tool.model_copy(update={"description": override}))
            continue
        copied_tools.append(tool)
    return copied_tools
