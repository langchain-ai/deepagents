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
    TRUNCATION_MARKER_TEMPLATE,
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

Bounds previews of few-but-huge lines (a `.jsonl` dump, a minified bundle).
Clipping cuts within a shown line rather than dropping whole lines, so it is
reported separately from `lines_omitted`.

Keep below `backends.utils.MAX_LINE_LENGTH`, or clipped lines also pick up that
renderer's `N.1` continuation gutters and `_CAVEAT_CLIPPED_LINES` no longer
describes what the model sees.
"""

_PREVIEW_NOTE_PLAIN = "Here is a preview of the {subject}"
_PREVIEW_NOTE_HEAD_TAIL = "Here is a preview showing the head and tail of the {subject}"

_CAVEAT_OMITTED_LINES = (
    f"lines of the form `{TRUNCATION_MARKER_TEMPLATE.format(omitted_lines='N')}` indicate omitted lines in the middle of the content"
)
_CAVEAT_CLIPPED_LINES = f"the output contains lines longer than {PREVIEW_LINE_CHAR_LIMIT} characters; this preview shows only their first {PREVIEW_LINE_CHAR_LIMIT} characters"


@dataclass(frozen=True, slots=True)
class ContentPreview:
    """A rendered preview plus a record of what was left out to build it.

    The flags are reported by the code that built `text`, never inferred from
    the rendered bytes — a literal `... [N lines truncated] ...` line in the
    content would otherwise pass for a real marker. The two flags track
    independent kinds of loss; a preview can have both, so check both.
    """

    text: str
    """The rendered, line-numbered preview."""

    lines_omitted: bool
    """Whole lines were dropped from the middle, behind a truncation marker."""

    lines_clipped: bool
    """At least one shown line was clipped at `PREVIEW_LINE_CHAR_LIMIT` characters."""


def _preview_note(*, lines_omitted: bool, lines_clipped: bool = False, subject: str = "result") -> str:
    """Build the sentence introducing a preview.

    Mentions only losses the preview actually has, so the model is never told
    to look for a marker that was not inserted.

    Args:
        lines_omitted: Whole lines were dropped from the middle behind a
            truncation marker.
        lines_clipped: At least one shown line was clipped at
            `PREVIEW_LINE_CHAR_LIMIT` characters.
        subject: Noun for what is being previewed, e.g. `result`.

    Returns:
        The note, ending in a colon.

            For example:

            - No losses: `Here is a preview of the result:`
            - `lines_omitted`: `Here is a preview showing the head and tail of the
                result (lines of the form ... indicate omitted lines ...):`
            - Both losses: the head/tail sentence with both caveats in parentheses.
    """
    base = _PREVIEW_NOTE_HEAD_TAIL if lines_omitted else _PREVIEW_NOTE_PLAIN
    caveats = [caveat for applies, caveat in ((lines_omitted, _CAVEAT_OMITTED_LINES), (lines_clipped, _CAVEAT_CLIPPED_LINES)) if applies]
    note = base.format(subject=subject)
    if caveats:
        note += f" ({'; '.join(caveats)})"
    return f"{note}:"


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> ContentPreview:
    """Create a line-numbered preview of `content_str`.

    Shows all lines when they fit within `head_lines + tail_lines`, otherwise
    the head and tail around a `... [N lines truncated] ...` marker.

    Args:
        content_str: The full content string to preview.
        head_lines: Number of lines to show from the start.
        tail_lines: Number of lines to show from the end.

    Returns:
        The formatted preview plus a record of what was left out to build it.
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


def _render_preview_stub(template: str, preview: ContentPreview, *, subject: str = "result", **fields: str) -> str:
    """Render `template` around `preview`, deriving the note from that same preview.

    The only way to fill a `{preview_note}`/`{content_sample}` template, so the
    note cannot end up describing losses some other preview had.

    Args:
        template: Stub text with `{preview_note}` and `{content_sample}`
            placeholders, plus whatever `fields` supplies.
        preview: The preview to render and to derive the note from.
        subject: Noun for what is being previewed, e.g. `result`.
        fields: Remaining template placeholders, e.g. `file_path`.

    Returns:
        The rendered stub, ready to use as message content.
    """
    return template.format(
        preview_note=_preview_note(lines_omitted=preview.lines_omitted, lines_clipped=preview.lines_clipped, subject=subject),
        content_sample=preview.text,
        **fields,
    )


def _render_too_large_tool_msg(*, tool_call_id: str, file_path: str, content_str: str) -> str:
    """Render the large-tool-result stub for `content_str`.

    Args:
        tool_call_id: Tool call whose result was offloaded.
        file_path: Path the full content was written to.
        content_str: The full content being previewed.

    Returns:
        The rendered stub, ready to use as message content.
    """
    return _render_preview_stub(
        _TOO_LARGE_TOOL_MSG,
        _create_content_preview(content_str),
        tool_call_id=tool_call_id,
        file_path=file_path,
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
