"""Formatting utilities for tool call display in the CLI.

This module handles rendering tool calls and tool messages for the TUI.

Imported at runtime (not at CLI startup), so it can safely depend
on heavier modules like `backends`.
"""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from deepagents_cli.config import MAX_ARG_LENGTH, get_glyphs
from deepagents_cli.unicode_security import strip_dangerous_unicode

_HIDDEN_CHAR_MARKER = " [hidden chars removed]"
"""Marker appended to display values that had dangerous Unicode stripped, so
users know the value was modified for safety."""


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


def _coerce_timeout_seconds(timeout: int | str | None) -> int | None:
    """Normalize timeout values to seconds for display.

    Accepts integer values and numeric strings. Returns `None` for invalid
    values so display formatting never raises.

    Args:
        timeout: Raw timeout value from tool arguments.

    Returns:
        Integer timeout in seconds, or `None` if unavailable/invalid.
    """
    if type(timeout) is int:
        return timeout
    if isinstance(timeout, str):
        stripped = timeout.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """Truncate a string value if it exceeds max_length.

    Returns:
        Truncated string with ellipsis suffix if exceeded, otherwise original.
    """
    if len(value) > max_length:
        return value[:max_length] + get_glyphs().ellipsis
    return value


def _sanitize_display_value(value: object, *, max_length: int = MAX_ARG_LENGTH) -> str:
    """Sanitize a value for safe, compact terminal display.

    Hidden/deceptive Unicode controls are stripped. When stripping occurs, a
    marker is appended so users know the value changed for display safety.

    Args:
        value: Any value to display.
        max_length: Maximum display length before truncation.

    Returns:
        Sanitized display string.
    """
    raw = str(value)
    sanitized = strip_dangerous_unicode(raw)
    display = truncate_value(sanitized, max_length)
    if sanitized != raw:
        return display + _HIDDEN_CHAR_MARKER
    return display


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
            path_raw = strip_dangerous_unicode(str(path_value))
            path = abbreviate_path(path_raw)
            if path_raw != str(path_value):
                path += _HIDDEN_CHAR_MARKER
            return f"{prefix} {tool_name}({path})"

    elif tool_name == "web_search":
        # Web search: show the query string
        if "query" in tool_args:
            query = _sanitize_display_value(tool_args["query"], max_length=100)
            return f'{prefix} {tool_name}("{query}")'

    elif tool_name == "grep":
        # Grep: show the search pattern
        if "pattern" in tool_args:
            pattern = _sanitize_display_value(tool_args["pattern"], max_length=70)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "execute":
        # Execute: show the command, and timeout only if non-default
        if "command" in tool_args:
            command = _sanitize_display_value(tool_args["command"], max_length=120)
            timeout = _coerce_timeout_seconds(tool_args.get("timeout"))
            from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT

            if timeout is not None and timeout != DEFAULT_EXECUTE_TIMEOUT:
                timeout_str = _format_timeout(timeout)
                return f'{prefix} {tool_name}("{command}", timeout={timeout_str})'
            return f'{prefix} {tool_name}("{command}")'

    elif tool_name == "ls":
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path_raw = strip_dangerous_unicode(str(tool_args["path"]))
            path = abbreviate_path(path_raw)
            if path_raw != str(tool_args["path"]):
                path += _HIDDEN_CHAR_MARKER
            return f"{prefix} {tool_name}({path})"
        return f"{prefix} {tool_name}()"

    elif tool_name == "glob":
        # Glob: show the pattern
        if "pattern" in tool_args:
            pattern = _sanitize_display_value(tool_args["pattern"], max_length=80)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "http_request":
        # HTTP: show method and URL
        parts = []
        if "method" in tool_args:
            method = _sanitize_display_value(tool_args["method"], max_length=16)
            parts.append(method.upper())
        if "url" in tool_args:
            url = _sanitize_display_value(tool_args["url"], max_length=80)
            parts.append(url)
        if parts:
            return f"{prefix} {tool_name}({' '.join(parts)})"

    elif tool_name == "fetch_url":
        # Fetch URL: show the URL being fetched
        if "url" in tool_args:
            url = _sanitize_display_value(tool_args["url"], max_length=80)
            return f'{prefix} {tool_name}("{url}")'

    elif tool_name == "task":
        # Task: show the task description
        if "description" in tool_args:
            desc = _sanitize_display_value(tool_args["description"], max_length=100)
            return f'{prefix} {tool_name}("{desc}")'

    elif tool_name == "ask_user":
        if "questions" in tool_args and isinstance(tool_args["questions"], list):
            count = len(tool_args["questions"])
            label = "question" if count == 1 else "questions"
            return f"{prefix} {tool_name}({count} {label})"

    elif tool_name == "compact_conversation":
        return f"{prefix} {tool_name}()"

    elif tool_name == "write_todos":
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{prefix} {tool_name}({count} items)"

    # Fallback: generic formatting for unknown tools
    # Show all arguments in key=value format
    args_str = ", ".join(
        f"{_sanitize_display_value(k, max_length=30)}="
        f"{_sanitize_display_value(v, max_length=50)}"
        for k, v in tool_args.items()
    )
    return f"{prefix} {tool_name}({args_str})"


def _format_content_block(block: dict) -> str:
    """Format a single content block dict for display.

    Replaces large binary payloads (e.g. base64 image/video data) with a
    human-readable placeholder so they don't flood the terminal.

    Args:
        block: An `ImageContentBlock`, `VideoContentBlock`, or `FileContentBlock`
            dictionary.

    Returns:
        A display-friendly string for the block.
    """
    if block.get("type") == "image" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "image")
        return f"[Image: {mime}, ~{size_kb}KB]"
    if block.get("type") == "video" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "video")
        return f"[Video: {mime}, ~{size_kb}KB]"
    if block.get("type") == "file" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "file")
        return f"[File: {mime}, ~{size_kb}KB]"
    try:
        # Preserve non-ASCII characters (CJK, emoji, etc.) instead of \uXXXX escapes
        return json.dumps(block, ensure_ascii=False)
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
                    # Preserve non-ASCII characters (CJK, emoji, etc.)
                    parts.append(json.dumps(item, ensure_ascii=False))
                except (TypeError, ValueError):
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)
