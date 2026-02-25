"""Utility functions for converting ACP content blocks to LangChain formats."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acp.schema import (
        AudioContentBlock,
        EmbeddedResourceContentBlock,
        ImageContentBlock,
        ResourceContentBlock,
        TextContentBlock,
    )


def convert_text_block_to_content_blocks(block: TextContentBlock) -> list[dict[str, str]]:
    """Convert an ACP text block to LangChain content blocks."""
    return [{"type": "text", "text": block.text}]


def convert_image_block_to_content_blocks(block: ImageContentBlock) -> list[dict[str, object]]:
    """Convert an ACP image block to LangChain content blocks."""
    # Primary case: inline base64 data (data is already a base64 string)
    if block.data:
        data_uri = f"data:{block.mime_type};base64,{block.data}"
        return [{"type": "image_url", "image_url": {"url": data_uri}}]

    # No data available
    return [{"type": "text", "text": "[Image: no data available]"}]


def convert_audio_block_to_content_blocks(block: AudioContentBlock) -> list[dict[str, str]]:
    """Convert an ACP audio block to LangChain content blocks.

    Raises:
        NotImplementedError: Audio content is not yet supported.
    """
    msg = "Audio is not currently supported."
    raise NotImplementedError(msg)


def convert_resource_block_to_content_blocks(
    block: ResourceContentBlock,
    *,
    root_dir: str,
) -> list[dict[str, str]]:
    """Convert an ACP resource block to LangChain content blocks."""
    file_prefix = "file://"
    resource_text = f"[Resource: {block.name}"
    if block.uri:
        # Truncate root_dir from path while preserving file:// prefix
        uri = block.uri
        has_file_prefix = uri.startswith(file_prefix)
        path = uri[len(file_prefix) :] if has_file_prefix else uri

        # Remove root_dir prefix to get path relative to agent's working directory
        if path.startswith(root_dir):
            path = path[len(root_dir) :].lstrip("/")

        # Restore file:// prefix if it was present
        uri = f"file://{path}" if has_file_prefix else path
        resource_text += f"\nURI: {uri}"
    if block.description:
        resource_text += f"\nDescription: {block.description}"
    if block.mime_type:
        resource_text += f"\nMIME type: {block.mime_type}"
    resource_text += "]"
    return [{"type": "text", "text": resource_text}]


def convert_embedded_resource_block_to_content_blocks(
    block: EmbeddedResourceContentBlock,
) -> list[dict[str, str]]:
    """Convert an ACP embedded resource block to LangChain content blocks.

    Raises:
        ValueError: If the block has neither a ``text`` nor ``blob`` property.
    """
    resource = block.resource
    if hasattr(resource, "text"):
        mime_type = getattr(resource, "mime_type", "application/text")
        return [{"type": "text", "text": f"[Embedded {mime_type} resource: {resource.text}"}]
    if hasattr(resource, "blob"):
        mime_type = getattr(resource, "mime_type", "application/octet-stream")
        data_uri = f"data:{mime_type};base64,{resource.blob}"
        return [
            {
                "type": "text",
                "text": f"[Embedded resource: {data_uri}]",
            }
        ]
    msg = (
        "Could not parse embedded resource block. "
        "Block expected either a `text` or `blob` property."
    )
    raise ValueError(msg)


def extract_command_types(command: str) -> list[str]:  # noqa: C901, PLR0915  # Complex shell command parser with nested helper functions
    """Extract all command types from a shell command, handling && separators.

    For security-sensitive commands (python, node, npm, uv, etc.), includes the full
    signature to avoid over-permissioning. Each sensitive command has a dedicated handler
    that extracts the appropriate signature.

    Signature extraction strategy:
    - python/python3: Include module name for -m, just flag for -c
    - node: Just flag for -e/-p (code execution)
    - npm/yarn/pnpm: Include subcommand, and script name for "run"
    - uv: Include subcommand, and tool name for "run"
    - npx: Include package name
    - Others: Just the base command

    Args:
        command: The full shell command string

    Returns:
        List of command signatures (base command + subcommand/module for sensitive commands)

    Examples:
        >>> extract_command_types("npm install")
        ['npm install']
        >>> extract_command_types("cd /path && python -m pytest tests/")
        ['cd', 'python -m pytest']
        >>> extract_command_types("python -m pip install package")
        ['python -m pip']
        >>> extract_command_types("python -c 'print(1)'")
        ['python -c']
        >>> extract_command_types("node -e 'console.log(1)'")
        ['node -e']
        >>> extract_command_types("uv run pytest")
        ['uv run pytest']
        >>> extract_command_types("npm run build")
        ['npm run build']
        >>> extract_command_types("ls -la | grep foo")
        ['ls', 'grep']
        >>> extract_command_types("cd dir && npm install && npm test")
        ['cd', 'npm install', 'npm test']
    """
    if not command or not command.strip():
        return []

    def extract_python_signature(tokens: list[str]) -> str:
        """Extract signature for python/python3 commands."""
        base_cmd = tokens[0]
        if len(tokens) < 2:  # noqa: PLR2004  # Token count threshold for subcommand parsing
            return base_cmd

        # python -m <module> -> "python -m <module>"
        if tokens[1] == "-m" and len(tokens) > 2:  # noqa: PLR2004  # Token count threshold for module name
            return f"{base_cmd} -m {tokens[2]}"
        # python -c <code> -> "python -c" (code changes, just track the flag)
        if tokens[1] == "-c":
            return f"{base_cmd} -c"
        # python script.py -> "python" (just running a script)
        return base_cmd

    def extract_node_signature(tokens: list[str]) -> str:
        """Extract signature for node commands."""
        base_cmd = tokens[0]
        if len(tokens) < 2:  # noqa: PLR2004  # Token count threshold for subcommand parsing
            return base_cmd

        # node -e <code> -> "node -e" (code changes, just track the flag)
        if tokens[1] == "-e":
            return f"{base_cmd} -e"
        # node -p <code> -> "node -p" (code changes, just track the flag)
        if tokens[1] == "-p":
            return f"{base_cmd} -p"
        # node script.js -> "node" (just running a script)
        return base_cmd

    def extract_npm_signature(tokens: list[str]) -> str:
        """Extract signature for npm commands."""
        base_cmd = tokens[0]
        if len(tokens) < 2:  # noqa: PLR2004  # Token count threshold for subcommand parsing
            return base_cmd

        subcommand = tokens[1]
        # npm run <script> -> "npm run <script>" (include script name)
        if subcommand == "run" and len(tokens) > 2:  # noqa: PLR2004  # Token count threshold for script name
            return f"{base_cmd} run {tokens[2]}"
        # npm install/test/etc -> "npm <subcommand>"
        return f"{base_cmd} {subcommand}"

    def extract_uv_signature(tokens: list[str]) -> str:
        """Extract signature for uv commands."""
        base_cmd = tokens[0]
        if len(tokens) < 2:  # noqa: PLR2004  # Token count threshold for subcommand parsing
            return base_cmd

        subcommand = tokens[1]
        # uv run <tool> -> "uv run <tool>" (include tool name)
        if subcommand == "run" and len(tokens) > 2:  # noqa: PLR2004  # Token count threshold for tool name
            return f"{base_cmd} run {tokens[2]}"
        # uv pip/add/sync/etc -> "uv <subcommand>"
        return f"{base_cmd} {subcommand}"

    def extract_npx_signature(tokens: list[str]) -> str:
        """Extract signature for npx commands."""
        base_cmd = tokens[0]
        # npx <package> -> "npx <package>" (always include package name)
        if len(tokens) > 1:
            return f"{base_cmd} {tokens[1]}"
        return base_cmd

    def extract_yarn_pnpm_signature(tokens: list[str]) -> str:
        """Extract signature for yarn/pnpm commands."""
        base_cmd = tokens[0]
        if len(tokens) < 2:  # noqa: PLR2004  # Token count threshold for subcommand parsing
            return base_cmd

        subcommand = tokens[1]
        # yarn/pnpm run <script> -> "yarn run <script>" (include script name)
        if subcommand == "run" and len(tokens) > 2:  # noqa: PLR2004  # Token count threshold for script name
            return f"{base_cmd} run {tokens[2]}"
        # yarn/pnpm install/test/etc -> "yarn <subcommand>"
        return f"{base_cmd} {subcommand}"

    # Command handlers for sensitive commands
    command_handlers = {
        "python": extract_python_signature,
        "python3": extract_python_signature,
        "node": extract_node_signature,
        "npm": extract_npm_signature,
        "npx": extract_npx_signature,
        "yarn": extract_yarn_pnpm_signature,
        "pnpm": extract_yarn_pnpm_signature,
        "uv": extract_uv_signature,
    }

    command_types: list[str] = []

    # Split by && to handle chained commands
    and_segments = command.split("&&")

    for raw_segment in and_segments:
        segment = raw_segment.strip()
        if not segment:
            continue

        try:
            # Split by pipes and process all segments
            pipe_segments = segment.split("|")

            for raw_pipe_segment in pipe_segments:
                pipe_segment = raw_pipe_segment.strip()
                if not pipe_segment:
                    continue

                # Parse the segment to get the command
                tokens = shlex.split(pipe_segment)
                if not tokens:
                    continue

                base_cmd = tokens[0]

                # Use specific handler if available, otherwise just use base command
                if base_cmd in command_handlers:
                    signature = command_handlers[base_cmd](tokens)
                    command_types.append(signature)
                else:
                    # Non-sensitive commands - just use the base command
                    command_types.append(base_cmd)

        except (ValueError, IndexError):
            # If parsing fails, skip this segment
            continue

    return command_types


_MAX_DISPLAY_COMMAND_LENGTH = 120


def truncate_execute_command_for_display(command: str) -> str:
    """Truncate a command string to a maximum length for display."""
    if len(command) >= _MAX_DISPLAY_COMMAND_LENGTH:
        return command[:_MAX_DISPLAY_COMMAND_LENGTH] + "..."
    return command


def format_execute_result(command: str, result: str) -> str:
    """Format execute tool result for better display.

    Args:
        command: The shell command that was executed
        result: The raw result string from the execute tool

    Returns:
        Formatted string with command, output, and exit code
    """
    # Parse the result to extract output and exit code
    lines = result.split("\n")
    output_lines = []
    exit_code_line = None
    truncated_line = None

    for line in lines:
        if line.startswith("[Command ") and "exit code" in line:
            exit_code_line = line
        elif line.startswith("[Output was truncated"):
            truncated_line = line
        else:
            output_lines.append(line)

    # Join output lines and strip trailing whitespace
    output = "\n".join(output_lines).rstrip()

    # Build formatted result
    parts = []

    # Add command section
    parts.append(f"**Command:**\n```bash\n{command}\n```\n")

    # Add output section
    if output:
        parts.append(f"**Output:**\n```\n{output}\n```\n")
    else:
        parts.append("**Output:** _(empty)_\n")

    # Add status
    if exit_code_line:
        parts.append(f"**Status:** {exit_code_line.strip('[]')}")

    if truncated_line:
        parts.append(f"\n_{truncated_line.strip('[]')}_")

    return "\n".join(parts)
