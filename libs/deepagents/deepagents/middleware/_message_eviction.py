# ruff: noqa: E501
"""Shared helpers for evicting/clipping large message content with a head+tail preview.

Used by:

- `FilesystemMiddleware` — proactive per-tool-call offload when a tool result
    exceeds its configured size threshold, and inline-media offload out of
    `HumanMessage`s that are both oversized and no longer the current turn.
- `SummarizationMiddleware` — reactive tail-clipping in the fallback
    summarization path after a `ContextOverflowError`, and inline-media offload
    before archiving or summarizing.

The inline-media helpers are shared because the two middlewares need the same
block detection and the same content-hash storage layout, but they need different
*replacements*: an archive or summary prompt can carry a typed path reference
(`_media_reference_block`), while a live model request cannot and must get plain
pointer text (`_media_pointer_text`).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import mimetypes
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage, ToolMessage

from deepagents.backends.utils import (
    TRUNCATION_MARKER_TEMPLATE,
    format_content_with_line_numbers,
    sanitize_tool_call_id,
)

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock

    from deepagents.backends.protocol import BackendProtocol, FileUploadResponse

logger = logging.getLogger(__name__)

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

_VISIBLE_TOOL_CALL_ID_LIMIT: Final = 32
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


_OFFLOAD_FAILED_PLACEHOLDER = '<image error="failed_to_offload" />'
"""Text placeholder written when a media block cannot be offloaded.

Marks the spot so the saved history shows a block was present rather than
silently omitting it.
"""

_INLINE_MEDIA_POINTER: Final = "[{kind} offloaded to the filesystem at {path}; call read_file on that path to view it]"
"""Pointer text replacing inline media in a live model request.

Plain text, not a typed media block: `read_file` resolves the path back to a
real media block, whereas a typed block carrying a backend-relative `url` is not
fetchable by the model provider.
"""

_MEDIA_POINTER_KINDS: Final = frozenset({"image", "audio", "video"})


def _is_data_url(url: str) -> bool:
    """Return whether `url` is an inline `data:` URL.

    Any `data:` URL counts as inline media, in both base64
    (`data:<mime>;base64,<payload>`) and percent-encoded / plaintext
    (`data:<mime>,<payload>`, e.g. an inline SVG) forms; whether the payload
    actually decodes is left to `_decode_data_url`.
    """
    return url.startswith("data:")


def _inline_media_source(block: Any) -> str | None:  # noqa: ANN401
    """Return the stored inline-media payload of `block`, or `None`.

    Detects the three inline-data content-block shapes that appear across
    LangChain messages:

    1. A standard content block with an explicit `base64` field.
    2. A `data:` URL on the `url` field.
    3. An OpenAI-style `image_url` block whose `url` is a `data:` URL.

    Shape 3 is defensive: `content_blocks` normalizes most `image_url` blocks
    (a base64 `data:` URL becomes shape 1; an `https` URL becomes a plain `url`
    image block), so this branch rarely fires for normalized input.

    The payload is returned exactly as stored -- the base64 body for shape 1, the
    full `data:` URL for shapes 2 and 3 -- so callers that only measure or test a
    block never rebuild (and therefore never copy) its bytes. Use
    `_extract_data_url` when a normalized `data:` URL is required.

    Pure detection and never raises: it reports *whether* a block carries inline
    data, leaving decoding (which can fail) to `_decode_data_url`.

    Args:
        block: A single content block (usually a dict).

    Returns:
        The block's stored inline payload, or `None` if it carries none.
    """
    if not isinstance(block, dict):
        return None

    raw_b64 = block.get("base64")
    if raw_b64:
        return str(raw_b64)

    url = block.get("url", "")
    if isinstance(url, str) and _is_data_url(url):
        return url

    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        inner = image_url.get("url", "")
        if isinstance(inner, str) and _is_data_url(inner):
            return inner

    return None


def _extract_data_url(block: Any) -> str | None:  # noqa: ANN401
    """Return the `data:` URL for an inline-media content block.

    Normalizes all three shapes detected by `_inline_media_source` into a
    `data:<mime>[;base64],<payload>` URL ready for `_decode_data_url`. Covers any
    inline `data:` URL, base64 or percent-encoded/plaintext, because such payloads
    must be offloaded to a referenceable path rather than left inline.

    Args:
        block: A single content block (usually a dict).

    Returns:
        The block's `data:` URL, or `None` if the block carries no inline data.
    """
    source = _inline_media_source(block)
    if source is None:
        return None
    if _is_data_url(source):
        return source
    mime = block.get("mime_type") or "application/octet-stream"
    return f"data:{mime};base64,{source}"


def _decode_data_url(data_url: str) -> tuple[bytes, str, str] | None:
    """Decode a `data:` URL to raw bytes, a file extension, and a MIME type.

    Handles both encodings a `data:` URL can use: a `;base64,` payload is
    base64-decoded, while a plain `data:<mime>,<payload>` payload is treated as
    percent-encoded text (e.g. an inline SVG).

    Args:
        data_url: A `data:<mime>[;base64],<payload>` URL.

    Returns:
        A `(raw_bytes, extension, mime_type)` tuple, or `None` if decoding fails
            (including a malformed URL with no `,` payload separator). A failure
            is logged here and, like an upload failure, surfaces as a
            failed-offload placeholder that counts toward the caller's aggregate
            warning -- it is never swallowed silently.
    """
    try:
        header, payload = data_url.split(",", 1)
        mime = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        ext = (mimetypes.guess_extension(mime) or ".bin").lstrip(".")
        is_base64 = "base64" in header.lower().split(";")
        raw = base64.b64decode(payload) if is_base64 else urllib.parse.unquote_to_bytes(payload)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to decode data: content block (%s): %s", type(e).__name__, e)
        return None
    else:
        return raw, ext, mime


def _media_reference_block(path: str, mime: str) -> dict[str, Any]:
    """Build a content block referencing offloaded media by backend path.

    The block type is chosen so the XML history renderer serializes the
    reference: `image`, `audio`, and `video` map to their typed blocks, while
    any other MIME type falls back to a text block (the renderer has no generic
    file block and would otherwise drop it).

    These typed references are only safe where the text is *rendered* (archive
    markdown, summary prompts). A live model request must use
    `_media_pointer_text` instead.

    Args:
        path: Backend path where the media was stored.
        mime: MIME type of the original media, used to pick the block type.

    Returns:
        A content block carrying the path reference.
    """
    major = mime.split("/", 1)[0]
    if major in _MEDIA_POINTER_KINDS:
        return {"type": major, "url": path}
    return {"type": "text", "text": f'<file url="{path}" />'}


def _media_pointer_text(path: str, mime: str) -> str:
    """Return the model-facing pointer for media offloaded to `path`.

    The message carrying this pointer goes into a live model request, where a
    backend-relative `url` (e.g. `/conversation_history/media/<hash>.png`) is not
    fetchable by the provider. `read_file` resolves the path back to a real media
    block, so the model can still inspect the payload on demand in one extra tool
    call.

    Args:
        path: Backend path where the media was stored.
        mime: MIME type of the original media, used to name the block.

    Returns:
        Pointer text safe to send as request content.
    """
    major = mime.split("/", 1)[0]
    kind = major if major in _MEDIA_POINTER_KINDS else "file"
    return _INLINE_MEDIA_POINTER.format(kind=kind, path=path)


def _upload_response_error(responses: list[FileUploadResponse]) -> str | None:
    """Extract an error from a single-file batch upload result.

    Args:
        responses: Backend upload responses. `upload_files`/`aupload_files`
            are batch APIs that return one `FileUploadResponse` per input file
            in order. Media offloading passes exactly one file at a time, so the
            expected length is 1 and `responses[0]` maps to that file.

    Returns:
        The upload error, `"missing_upload_response"` if the backend returned
            no response, or `None` when the upload succeeded.
    """
    if not responses:
        return "missing_upload_response"
    error = responses[0].error
    if error is None:
        return None
    return str(error)


def _inline_media_payload_chars(block: Any) -> int:  # noqa: ANN401
    """Return the character length of `block`'s inline-media payload, else 0.

    Cheap enough to run over every message on every model call: unlike
    `_extract_data_url` it never rebuilds the payload, so measuring a
    multi-megabyte block does not copy its bytes.
    """
    source = _inline_media_source(block)
    return len(source) if source is not None else 0


def _message_char_size(message: BaseMessage) -> int:
    """Return `message`'s text length plus the length of its inline-media payloads.

    `_extract_text_from_message` deliberately ignores non-text blocks, so a short
    caption plus a multi-megabyte inline image measures as tiny and would never
    cross an eviction threshold. Eviction decisions use this measurement instead.

    The text written to the backend still comes from `_extract_text_from_message`:
    this measurement only decides *whether* a message is oversized.

    Args:
        message: The message to measure.

    Returns:
        Text characters plus inline-media payload characters.
    """
    media_chars = sum(_inline_media_payload_chars(block) for block in message.content_blocks)
    return len(_extract_text_from_message(message)) + media_chars


def _offload_inline_media_blocks(
    message: BaseMessage,
    backend: BackendProtocol,
    media_prefix: str,
) -> list[str | None] | None:
    """Upload `message`'s inline media and return positionally aligned pointers.

    The returned list is aligned to `message`'s non-text blocks in order: pointer
    text for a block that was uploaded, and `None` for a block left inline -- a
    remote `http(s)` reference, an undecodable payload, or a failed upload.
    Keeping a failed block inline is the fail-safe choice: no data is lost, the
    request simply does not shrink this turn.

    Unique media are deduped by content hash, so the same image repeated across
    blocks is uploaded once.

    Args:
        message: The message whose inline media should be offloaded.
        backend: Backend to upload media files to.
        media_prefix: Directory prefix the media files are written under.

    Returns:
        A list aligned to the message's non-text blocks, or `None` when nothing
            was offloaded (no inline media, or every upload failed) -- so callers
            can skip tagging the message entirely.
    """
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        return None

    path_map: dict[str, str] = {}  # content key -> backend path (successfully uploaded)
    pointers: list[str | None] = []
    for block in media_blocks:
        data_url = _extract_data_url(block)
        if data_url is None:
            pointers.append(None)
            continue
        decoded = _decode_data_url(data_url)
        if decoded is None:
            pointers.append(None)
            continue
        raw, ext, mime = decoded
        key = hashlib.sha256(raw).hexdigest()[:16]
        path = path_map.get(key)
        if path is None:
            path = f"{media_prefix}/{key}.{ext}"
            try:
                if error := _upload_response_error(backend.upload_files([(path, raw)])):
                    logger.warning("Failed to upload media %s to backend: %s", path, error)
                    pointers.append(None)
                    continue
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to upload media %s to backend: %s: %s", path, type(e).__name__, e)
                pointers.append(None)
                continue
            path_map[key] = path
        pointers.append(_media_pointer_text(path, mime))

    return pointers if any(pointer is not None for pointer in pointers) else None


async def _aoffload_inline_media_blocks(
    message: BaseMessage,
    backend: BackendProtocol,
    media_prefix: str,
) -> list[str | None] | None:
    """Async twin of `_offload_inline_media_blocks` using `aupload_files`.

    See `_offload_inline_media_blocks` for full documentation, including the
    positional-alignment and `None` return contracts.
    """
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        return None

    path_map: dict[str, str] = {}
    pointers: list[str | None] = []
    for block in media_blocks:
        data_url = _extract_data_url(block)
        if data_url is None:
            pointers.append(None)
            continue
        decoded = _decode_data_url(data_url)
        if decoded is None:
            pointers.append(None)
            continue
        raw, ext, mime = decoded
        key = hashlib.sha256(raw).hexdigest()[:16]
        path = path_map.get(key)
        if path is None:
            path = f"{media_prefix}/{key}.{ext}"
            try:
                if error := _upload_response_error(await backend.aupload_files([(path, raw)])):
                    logger.warning("Failed to upload media %s to backend: %s", path, error)
                    pointers.append(None)
                    continue
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to upload media %s to backend: %s: %s", path, type(e).__name__, e)
                pointers.append(None)
                continue
            path_map[key] = path
        pointers.append(_media_pointer_text(path, mime))

    return pointers if any(pointer is not None for pointer in pointers) else None


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


def _visible_tool_call_id(tool_call_id: str) -> str:
    """Abbreviate IDs only in model-visible offload notices."""
    return f"{tool_call_id[:_VISIBLE_TOOL_CALL_ID_LIMIT]}..." if len(tool_call_id) > _VISIBLE_TOOL_CALL_ID_LIMIT else tool_call_id


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
        tool_call_id=_visible_tool_call_id(tool_call_id),
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
    sanitized_id = sanitize_tool_call_id(message.tool_call_id) if message.tool_call_id else f"unknown-{uuid4().hex[:8]}"
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
    sanitized_id = sanitize_tool_call_id(message.tool_call_id) if message.tool_call_id else f"unknown-{uuid4().hex[:8]}"
    file_path = f"{large_tool_results_prefix}/{sanitized_id}"
    result = await backend.awrite(file_path, content_str)
    if result is None or result.error:
        return None
    replacement_text = _render_too_large_tool_msg(tool_call_id=message.tool_call_id, file_path=file_path, content_str=content_str)
    return _build_evicted_tool_message(message, _build_evicted_content(message, replacement_text))
