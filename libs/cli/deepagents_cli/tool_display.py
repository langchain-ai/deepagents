"""Formatting utilities for tool call display in the CLI.

This module handles rendering tool calls and tool messages for the TUI.

Imported at runtime (not at CLI startup), so it can safely depend
on heavier modules like `backends`.
"""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT

from deepagents_cli.config import MAX_ARG_LENGTH, get_glyphs


def _format_timeout(seconds: int) -> str:
    """Format timeout in human-readable units (e.g., 300 -> '5m', 3600 -> '1h').

    Args:
        seconds: The timeout value in seconds to format.

    Returns:
        Human-readable timeout string (e.g., '5m', '1h', '300s').
    """
    if seconds < 60:  # noqa: PLR2004  # Time unit boundary
        return f"{seconds}s"
    if seconds < 3600 and seconds % 60 == 0:  # noqa: PLR2004  # Time unit boundaries
        return f"{seconds // 60}m"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    # For odd values, just show seconds
    return f"{seconds}s"


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length.

    Returns:
        Truncated string with ellipsis suffix if exceeded, otherwise original.
    """
    if len(value) > max_length:
        return value[:max_length] + get_glyphs().ellipsis
    return value


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """Format tool calls for display with tool-specific smart formatting.

    Shows the most relevant information for each tool type rather than all arguments.

    Args:
        tool_name: Name of the tool being called
        tool_args: Dictionary of tool arguments

    Returns:
        Formatted string for display (e.g., "(*) read_file(config.py)" in ASCII mode)

    Examples:
        read_file(path="/long/path/file.py") → "<prefix> read_file(file.py)"
        web_search(query="how to code") → '<prefix> web_search("how to code")'
        execute(command="pip install foo") → '<prefix> execute("pip install foo")'
    """
    prefix = get_glyphs().tool_prefix

    def abbreviate_path(path_str: str, max_length: int = 60) -> str:
        """Abbreviate a file path intelligently - show basename or relative path.

        Returns:
            Shortened path string suitable for display.
        """
        try:
            path = Path(path_str)

            # If it's just a filename (no directory parts), return as-is
            if len(path.parts) == 1:
                return path_str

            # Try to get relative path from current working directory
            with suppress(
                ValueError,  # ValueError: path is not relative to cwd
                OSError,  # OSError: filesystem errors when resolving paths
            ):
                rel_path = path.relative_to(Path.cwd())
                rel_str = str(rel_path)
                # Use relative if it's shorter and not too long
                if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                    return rel_str

            # If absolute path is reasonable length, use it
            if len(path_str) <= max_length:
                return path_str
        except Exception:  # noqa: BLE001  # Fallback to original string on any path resolution error
            return truncate_value(path_str, max_length)
        else:
            # Otherwise, just show basename (filename only)
            return path.name

    # Tool-specific formatting - show the most important argument(s)
    if tool_name in {"read_file", "write_file", "edit_file"}:
        # File operations: show the primary file path argument (file_path or path)
        path_value = tool_args.get("file_path")
        if path_value is None:
            path_value = tool_args.get("path")
        if path_value is not None:
            path = abbreviate_path(str(path_value))
            return f"{prefix} {tool_name}({path})"

    elif tool_name == "web_search":
        # Web search: show the query string
        if "query" in tool_args:
            query = str(tool_args["query"])
            query = truncate_value(query, 100)
            return f'{prefix} {tool_name}("{query}")'

    elif tool_name == "grep":
        # Grep: show the search pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 70)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "execute":
        # Execute: show the command, and timeout only if non-default
        if "command" in tool_args:
            command = str(tool_args["command"])
            command = truncate_value(command, 120)
            timeout = tool_args.get("timeout")
            if timeout is not None and timeout != DEFAULT_EXECUTE_TIMEOUT:
                timeout_str = _format_timeout(timeout)
                return f'{prefix} {tool_name}("{command}", timeout={timeout_str})'
            return f'{prefix} {tool_name}("{command}")'

    elif tool_name == "ls":
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path = abbreviate_path(str(tool_args["path"]))
            return f"{prefix} {tool_name}({path})"
        return f"{prefix} {tool_name}()"

    elif tool_name == "glob":
        # Glob: show the pattern
        if "pattern" in tool_args:
            pattern = str(tool_args["pattern"])
            pattern = truncate_value(pattern, 80)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "http_request":
        # HTTP: show method and URL
        parts = []
        if "method" in tool_args:
            parts.append(str(tool_args["method"]).upper())
        if "url" in tool_args:
            url = str(tool_args["url"])
            url = truncate_value(url, 80)
            parts.append(url)
        if parts:
            return f"{prefix} {tool_name}({' '.join(parts)})"

    elif tool_name == "fetch_url":
        # Fetch URL: show the URL being fetched
        if "url" in tool_args:
            url = str(tool_args["url"])
            url = truncate_value(url, 80)
            return f'{prefix} {tool_name}("{url}")'

    elif tool_name == "task":
        # Task: show the task description
        if "description" in tool_args:
            desc = str(tool_args["description"])
            desc = truncate_value(desc, 100)
            return f'{prefix} {tool_name}("{desc}")'

    elif tool_name == "write_todos":
        # Todos: show count of items
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{prefix} {tool_name}({count} items)"

    # Fallback: generic formatting for unknown tools
    # Show all arguments in key=value format
    args_str = ", ".join(
        f"{k}={truncate_value(str(v), 50)}" for k, v in tool_args.items()
    )
    return f"{prefix} {tool_name}({args_str})"


def _format_content_block(block: dict) -> str:
    """Format a single content block dict for display.

    Replaces large binary payloads (e.g. base64 image data) with a
    human-readable placeholder so they don't flood the terminal.

    Args:
        block: An `ImageContentBlock` dictionary.

    Returns:
        A display-friendly string for the block.
    """
    if block.get("type") == "image" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "image")
        return f"[Image: {mime}, ~{size_kb}KB]"
    try:
        return json.dumps(block)
    except (TypeError, ValueError):
        return str(block)


def format_tool_message_content(content: Any) -> str:  # noqa: ANN401  # Content can be str, list, or dict
    """Convert `ToolMessage` content into a printable string.

    Returns:
        Formatted string representation of the tool message content.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_format_content_block(item))
            else:
                try:
                    parts.append(json.dumps(item))
                except (TypeError, ValueError):
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)
