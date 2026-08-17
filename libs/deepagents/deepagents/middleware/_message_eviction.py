# ruff: noqa: E501
"""Shared helpers for evicting/clipping large message content with a head+tail preview.

Used by:

- `FilesystemMiddleware` — proactive per-tool-call offload when a tool result
    exceeds its configured size threshold.
- `SummarizationMiddleware` — reactive tail-clipping in the fallback
    summarization path after a `ContextOverflowError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from langchain_core.messages import BaseMessage, ToolMessage

from deepagents.backends.utils import (
    TRUNCATION_MARKER_TEMPLATE as TRUNCATION_MARKER_TEMPLATE,
    format_content_with_line_numbers,
    sanitize_tool_call_id,
)

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock

    from deepagents.backends.protocol import BackendProtocol

_TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

{preview_note}

{content_sample}
"""

PREVIEW_LINE_CHAR_LIMIT: Final = 1000
"""Per-line character budget for preview lines.

Bounds a preview built from few but very long lines (a `.jsonl` dump, a minified
bundle). Clipping here drops content *within* a shown line rather than dropping
whole lines, so it is reported separately from `lines_omitted`.

Keep this below `backends.utils.MAX_LINE_LENGTH`, or clipped lines also acquire
that renderer's `N.1`-style continuation gutters and `_CAVEAT_CLIPPED_LINES` stops
describing what the model sees.
"""

_PREVIEW_NOTE_PLAIN = "Here is a preview of the {subject}"
_PREVIEW_NOTE_HEAD_TAIL = "Here is a preview showing the head and tail of the {subject}"

_CAVEAT_OMITTED_LINES = (
    f"lines of the form `{TRUNCATION_MARKER_TEMPLATE.format(omitted_lines='N')}` indicate omitted lines in the middle of the content"
)
_CAVEAT_CLIPPED_LINES = f"lines longer than {PREVIEW_LINE_CHAR_LIMIT} characters are clipped to their first {PREVIEW_LINE_CHAR_LIMIT} characters"


@dataclass(frozen=True, slots=True)
class ContentPreview:
    """A rendered preview plus what its producer elided to build it.

    Both flags are reported by whoever built `text`, never inferred from the
    rendered bytes. That is what stops previewed content from describing itself:
    a literal `... [N lines truncated] ...` line in the content cannot make the
    note claim lines were omitted.

    Neither flag means "the preview is complete" on its own -- check both. They
    describe two independent kinds of loss, and a preview can suffer both at
    once.
    """

    text: str
    """The rendered, line-numbered preview."""

    lines_omitted: bool
    """Whole lines were dropped from the middle, behind a truncation marker."""

    lines_clipped: bool = False
    """At least one shown line was clipped at `PREVIEW_LINE_CHAR_LIMIT` characters."""


def _preview_note(*, lines_omitted: bool, lines_clipped: bool = False, subject: str = "result") -> str:
    """Build the sentence introducing a preview.

    The note describes only what the preview actually did, so the model is never
    told to look for a truncation marker that was not inserted, nor led to
    believe the lines it can see are intact when they were clipped.

    Args:
        lines_omitted: Whether whole lines were dropped from the middle behind a
            truncation marker.
        lines_clipped: Whether any shown line was clipped at
            `PREVIEW_LINE_CHAR_LIMIT` characters.
        subject: Noun describing what is being previewed, e.g. `result`.

    Returns:
        A single-sentence note ending in a colon. Applicable caveats are
        appended in a parenthetical; with no caveats the sentence is a bare
        `Here is a preview of the {subject}:`.
    """
    base = _PREVIEW_NOTE_HEAD_TAIL if lines_omitted else _PREVIEW_NOTE_PLAIN
    caveats = [caveat for applies, caveat in ((lines_omitted, _CAVEAT_OMITTED_LINES), (lines_clipped, _CAVEAT_CLIPPED_LINES)) if applies]
    note = base.format(subject=subject)
    if caveats:
        note += f" ({'; '.join(caveats)})"
    return f"{note}:"


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> ContentPreview:
    """Create a preview of content showing head and tail with truncation marker.

    Args:
        content_str: The full content string to preview.
        head_lines: Number of lines to show from the start.
        tail_lines: Number of lines to show from the end.

    Returns:
        The formatted preview plus what was elided to build it: whether whole
        lines were dropped from the middle, and whether any shown line was
        clipped at `PREVIEW_LINE_CHAR_LIMIT` characters.
    """
    lines = content_str.splitlines()

    def _clip(shown: list[str]) -> tuple[list[str], bool]:
        """Clip each line to the per-line budget, reporting whether any was."""
        return [line[:PREVIEW_LINE_CHAR_LIMIT] for line in shown], any(len(line) > PREVIEW_LINE_CHAR_LIMIT for line in shown)

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines, clipped = _clip(lines)
        return ContentPreview(
            format_content_with_line_numbers(preview_lines, start_line=1),
            lines_omitted=False,
            lines_clipped=clipped,
        )

    # Show head and tail with truncation marker
    head, head_clipped = _clip(lines[:head_lines])
    tail, tail_clipped = _clip(lines[-tail_lines:])

    head_sample = format_content_with_line_numbers(head, start_line=1)
    marker = TRUNCATION_MARKER_TEMPLATE.format(omitted_lines=len(lines) - head_lines - tail_lines)
    truncation_notice = f"\n{marker}\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return ContentPreview(
        head_sample + truncation_notice + tail_sample,
        lines_omitted=True,
        lines_clipped=head_clipped or tail_clipped,
    )


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


def _render_too_large_tool_msg(*, tool_call_id: str, file_path: str, content_str: str) -> str:
    """Render the large-tool-result stub for `content_str`.

    Derive the preview and its matching note together so the note cannot
    describe a different preview than the one shown.

    Args:
        tool_call_id: Tool call whose result was offloaded.
        file_path: Path the full content was written to.
        content_str: The full content being previewed.

    Returns:
        The rendered stub, ready to use as message content.
    """
    preview = _create_content_preview(content_str)
    return _TOO_LARGE_TOOL_MSG.format(
        tool_call_id=tool_call_id,
        file_path=file_path,
        preview_note=_preview_note(lines_omitted=preview.lines_omitted, lines_clipped=preview.lines_clipped),
        content_sample=preview.text,
    )


def _offload_tool_message_content(
    message: ToolMessage,
    content_str: str,
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    """Write `content_str` to `{prefix}/{tool_call_id}` and return a clipped replacement.

    The replacement carries a head+tail preview and the offload path in
    large-tool-result format so the agent can `read_file` the full content
    by tool_call_id. Returns `None` if the backend write fails — caller should
    keep the original message in that case.
    """
    sanitized_id = sanitize_tool_call_id(message.tool_call_id) if message.tool_call_id else "unknown"
    file_path = f"{large_tool_results_prefix}/{sanitized_id}"
    result = backend.write(file_path, content_str)
    if result is None or result.error:
        return None
    replacement_text = _render_too_large_tool_msg(tool_call_id=message.tool_call_id, file_path=file_path, content_str=content_str)
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
    replacement_text = _render_too_large_tool_msg(tool_call_id=message.tool_call_id, file_path=file_path, content_str=content_str)
    return _build_evicted_tool_message(message, _build_evicted_content(message, replacement_text))
