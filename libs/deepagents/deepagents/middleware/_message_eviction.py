# ruff: noqa: E501
"""Shared helpers for evicting/clipping large message content with a head+tail preview.

Used by:

- `FilesystemMiddleware` — proactive per-tool-call offload when a tool result
    exceeds its configured size threshold.
- `SummarizationMiddleware` — reactive tail-clipping in the fallback
    summarization path after a `ContextOverflowError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

from langchain_core.messages import BaseMessage, ToolMessage

from deepagents.backends.utils import format_content_with_line_numbers, sanitize_tool_call_id

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock

    from deepagents.backends.protocol import BackendProtocol

TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

{preview_note}

{content_sample}
"""

TRUNCATION_MARKER_TEMPLATE = "... [{omitted_lines} lines truncated] ..."
"""Marker inserted between the head and tail of a preview when lines are omitted."""

_PREVIEW_NOTE = "Here is a preview of the {subject}:"
_PREVIEW_NOTE_WITH_MARKERS = "Here is a preview showing the head and tail of the {subject} (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):"


class ContentPreview(NamedTuple):
    """A rendered preview plus whether the producer omitted lines from its middle.

    `lines_omitted` is reported by whoever built `text`, so the marker
    explanation is never inferred from the previewed content itself -- content
    that happens to contain a literal `... [N lines truncated] ...` line cannot
    masquerade as an inserted marker.
    """

    text: str
    lines_omitted: bool


def _preview_note(*, lines_omitted: bool, subject: str = "result") -> str:
    """Build the sentence introducing a preview.

    Args:
        lines_omitted: Whether the preview omits lines behind a truncation marker.
        subject: Noun describing what is being previewed, e.g. `result`.

    Returns:
        A single-sentence note ending in a colon. The head/tail wording and the
        truncation-marker explanation are included only when `lines_omitted`.
    """
    template = _PREVIEW_NOTE_WITH_MARKERS if lines_omitted else _PREVIEW_NOTE
    return template.format(subject=subject)


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> ContentPreview:
    """Create a preview of content showing head and tail with truncation marker.

    Args:
        content_str: The full content string to preview.
        head_lines: Number of lines to show from the start.
        tail_lines: Number of lines to show from the end.

    Returns:
        The formatted preview with line numbers, and whether lines were omitted.
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines = [line[:1000] for line in lines]
        return ContentPreview(format_content_with_line_numbers(preview_lines, start_line=1), lines_omitted=False)

    # Show head and tail with truncation marker
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    marker = TRUNCATION_MARKER_TEMPLATE.format(omitted_lines=len(lines) - head_lines - tail_lines)
    truncation_notice = f"\n{marker}\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return ContentPreview(head_sample + truncation_notice + tail_sample, lines_omitted=True)


def _extract_text_from_message(message: BaseMessage) -> str:
    """Extract text from a message using its `content_blocks` property.

    Joins all text content blocks and ignores non-text blocks (images, audio, etc.)
    so that binary payloads don't inflate the size measurement.

    Args:
        message: The BaseMessage to extract text from.

    Returns:
        Joined text from all text content blocks, or stringified content as fallback.
    """
    texts = [block["text"] for block in message.content_blocks if block["type"] == "text"]
    return "\n".join(texts)


def _build_evicted_content(message: ToolMessage, replacement_text: str) -> str | list[ContentBlock]:
    """Build replacement content for an evicted message, preserving non-text blocks.

    For plain string content, returns the replacement text directly. For list content
    with mixed block types (e.g., text + image), replaces all text blocks with a single
    text block containing the replacement text while keeping non-text blocks intact.

    Args:
        message: The original ToolMessage being evicted.
        replacement_text: The truncation notice and preview text.

    Returns:
        Replacement content: a string or list of content blocks.
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        # All content is text, so a plain string replacement is sufficient.
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]


def _build_evicted_tool_message(message: ToolMessage, evicted_content: str | list[ContentBlock]) -> ToolMessage:
    """Build a replacement `ToolMessage` carrying `evicted_content`, preserving identity fields."""
    return ToolMessage(
        content=cast("str | list[str | dict]", evicted_content),
        tool_call_id=message.tool_call_id,
        name=message.name,
        id=message.id,
        artifact=message.artifact,
        status=message.status,
        additional_kwargs=dict(message.additional_kwargs),
        response_metadata=dict(message.response_metadata),
    )


def _offload_tool_message_content(
    message: ToolMessage,
    content_str: str,
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    """Write `content_str` to `{prefix}/{tool_call_id}` and return a clipped replacement.

    The replacement carries a head+tail preview and the offload path in
    `TOO_LARGE_TOOL_MSG` format so the agent can `read_file` the full content
    by tool_call_id. Returns `None` if the backend write fails — caller should
    keep the original message in that case.
    """
    sanitized_id = sanitize_tool_call_id(message.tool_call_id) if message.tool_call_id else "unknown"
    file_path = f"{large_tool_results_prefix}/{sanitized_id}"
    result = backend.write(file_path, content_str)
    if result is None or result.error:
        return None
    preview = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_TOOL_MSG.format(
        tool_call_id=message.tool_call_id,
        file_path=file_path,
        preview_note=_preview_note(lines_omitted=preview.lines_omitted),
        content_sample=preview.text,
    )
    return _build_evicted_tool_message(message, _build_evicted_content(message, replacement_text))


async def _aoffload_tool_message_content(
    message: ToolMessage,
    content_str: str,
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    """Async variant of `_offload_tool_message_content` using `backend.awrite`."""
    sanitized_id = sanitize_tool_call_id(message.tool_call_id) if message.tool_call_id else "unknown"
    file_path = f"{large_tool_results_prefix}/{sanitized_id}"
    result = await backend.awrite(file_path, content_str)
    if result is None or result.error:
        return None
    preview = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_TOOL_MSG.format(
        tool_call_id=message.tool_call_id,
        file_path=file_path,
        preview_note=_preview_note(lines_omitted=preview.lines_omitted),
        content_sample=preview.text,
    )
    return _build_evicted_tool_message(message, _build_evicted_content(message, replacement_text))
