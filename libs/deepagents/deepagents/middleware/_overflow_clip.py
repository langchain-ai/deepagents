# ruff: noqa: E501
"""Read-side clipping for the summarization-on-overflow fallback path.

When `SummarizationMiddleware`'s `wrap_model_call` catches a
`ContextOverflowError`, it falls through to summarization and *also* invokes
`_clip_overflow_tail` (or its async variant) to shrink the trailing
ToolMessage batch in the preserved suffix. Two per-TM paths:

- `read_file` tool result: reduce the content and append a notice pointing back
    to the original `file_path` argument. Results over `_SLICE_CHARS` keep a head
    preview; shorter ones are replaced by a bare pointer. No new backend write is
    needed because the original file already lives at that path.
- Any other tool result: archive it at
    `/large_tool_results/{tool_call_id}`, then replace the message with a
    `TOO_LARGE_TOOL_MSG` stub. Inline media is uploaded as a binary file and the
    archive becomes a JSON manifest that points to it.

Non-text blocks are dropped from the replacement rather than carried over: this
path runs *because* the request already exceeded the context window, so
re-sending an inline base64 payload would defeat the clip. Generic results
remain recoverable through their manifest; `read_file` results point back to
their original file.

Two independent questions drive the loop, and they must not share a number:

- *Should clipping begin?* `_derive_overflow_clip_threshold_tokens()` plus a raw
    non-text payload floor. Below both, the batch isn't worth rewriting.
- *Is a given replacement worth using?* Whether it measurably shrinks the tail,
    or removes non-text payload the token counter can't see. A replacement that
    would make the tail *larger* -- a 609-char offload stub standing in for a
    4-char result -- is discarded and the original kept.

There is deliberately no "the tail is small enough now, stop" condition. The
overflow already proved the advertised context limit was not respected, so no
available number reliably answers "will the retry fit?"; stopping early on a
guess is what leaves an oversized tail and a second, uncaught overflow. Every
result that can be shrunk is shrunk, and per-result rejection is what protects
small siblings from being needlessly rewritten.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from deepagents.middleware._media import _decode_data_url, _extract_data_url
from deepagents.middleware._message_eviction import (
    _build_offload_replacement,
    _extract_text_from_message,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain.agents.middleware.summarization import ContextSize, TokenCounter

    from deepagents.backends.protocol import BackendProtocol, FileUploadResponse

logger = logging.getLogger(__name__)

_SLICE_CHARS = 4_000
"""Head-slice size for a clipped `read_file` result, and the truncate-vs-omit boundary.

Results longer than this keep this many head chars plus a truncation notice.
Results at or under it have their text dropped entirely in favour of a path
pointer, which still shrinks a collectively oversized batch.
"""

_NON_TEXT_PAYLOAD_ENGAGE_CHARS = 4_000
"""Raw non-text payload that justifies clipping even when the tail measures small.

`count_tokens_approximately` attributes a near-constant ~92 tokens to a media
block regardless of its size, so a multi-megabyte base64 image measures far
below any `keep`-derived threshold and would never engage the clip. Sizing
non-text blocks by their serialized length is the only signal available here
that tracks what the request actually costs the provider.
"""


def _derive_overflow_clip_threshold_tokens(keep: ContextSize, max_input_tokens: int | None) -> int:
    """Derive a token threshold for tail-ToolMessage clipping from `keep`.

    Returns the keep token budget. If `keep` is message-based (no token info),
    falls back to 5_000 -- equivalent to a 20_000-char floor under a `chars / 4`
    approximation.
    """
    kind, value = keep
    if kind == "tokens":
        return int(value)
    if kind == "fraction":
        if max_input_tokens is None:
            return 5_000
        return int(max_input_tokens * value)
    return 5_000


def _find_tail_tool_message_batch(messages: list[AnyMessage]) -> tuple[int, list[ToolMessage]] | None:
    """Return `(start_index, batch)` if `messages` ends with consecutive ToolMessages."""
    if not messages or not isinstance(messages[-1], ToolMessage):
        return None
    i = len(messages) - 1
    while i >= 0 and isinstance(messages[i], ToolMessage):
        i -= 1
    start = i + 1
    return start, [cast("ToolMessage", m) for m in messages[start:]]


def _build_tool_call_index(messages: list[AnyMessage]) -> dict[str, dict[str, Any]]:
    """Map `tool_call_id` -> tool_call dict for all AIMessage tool_calls in `messages`."""
    index: dict[str, dict[str, Any]] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                tcid = tc.get("id")
                if tcid:
                    index[tcid] = cast("dict[str, Any]", tc)
    return index


def _non_text_payload_chars(msg: AnyMessage) -> int:
    """Serialized size of a message's non-text blocks.

    Stands in for the payload cost the token counter under-reports. Uses the
    whole serialized block so inline `base64`/`data`/`url` values are all counted
    without having to enumerate provider-specific field names.
    """
    return sum(len(str(block)) for block in msg.content_blocks if block["type"] != "text")


def _slice_read_file_tm(msg: ToolMessage, original_path: str) -> ToolMessage:
    """Reduce a `read_file` result and append a pointer to the original path.

    `read_file` results don't need a fresh `/large_tool_results/{tcid}` write -- the
    full file is already on the backend at `original_path`, and the agent can
    recover with `read_file(file_path=original_path, offset=N, limit=K)`.
    Results over `_SLICE_CHARS` keep a head preview; results at or under it have
    their text dropped entirely in favour of a compact pointer, which still
    reduces a collectively oversized batch.

    Non-text blocks are dropped rather than carried over, and replaced by a
    pointer to `original_path`, which still holds the file. Each notice describes
    only what actually happened, so a result that was not truncated doesn't claim
    it was.
    """
    content = _extract_text_from_message(msg)
    has_non_text = _non_text_payload_chars(msg) > 0
    truncated = len(content) > _SLICE_CHARS
    omitted = bool(content) and not truncated and not has_non_text
    clipped_content = "" if omitted else content[:_SLICE_CHARS]
    notice = ""
    if truncated:
        notice += (
            f"\n\n[Output was truncated due to context window size limits. "
            f"The full content is at {original_path}. "
            f"Use read_file with offset and limit parameters to retrieve specific portions. "
            f"For example, to read the first 100 lines, call read_file with file_path='{original_path}', offset=0, limit=100.]"
        )
    if omitted:
        notice += (
            f"[Output was omitted due to context window size limits. "
            f"The full content is at {original_path}. "
            f"Use read_file with offset and limit parameters to retrieve specific portions.]"
        )
    if has_non_text:
        notice += (
            f"\n\n[Non-text content (such as image, audio, or binary file data) was removed "
            f"due to context window size limits. The original file is at {original_path}. "
            f"Call read_file with file_path='{original_path}' to view it again.]"
        )
    return msg.model_copy(update={"content": clipped_content + notice})


def _read_file_original_path(msg: ToolMessage, tc_index: dict[str, dict[str, Any]]) -> str | None:
    """Return the `file_path` arg from the matching read_file tool_call, or `None`."""
    tc = tc_index.get(msg.tool_call_id) if msg.tool_call_id else None
    if not tc or tc.get("name") != "read_file":
        return None
    path = tc.get("args", {}).get("file_path")
    return path if isinstance(path, str) and path else None


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A proposed replacement for one tail ToolMessage, plus any write it depends on."""

    message: ToolMessage
    """The replacement to substitute for the original."""

    pending_write: tuple[str, str] | None
    """`(file_path, content)` that must be committed before `message` is used.

    `None` for `read_file` results, whose content is already on the backend at
    the path their notice points to.
    """

    pending_uploads: tuple[tuple[str, bytes], ...] = ()
    """Binary media files that must be uploaded before `pending_write`."""


def _with_media_reference(block: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Replace inline data in a content block with a backend path reference.

    Takes a `Mapping` so a `content_blocks` TypedDict can be passed directly; the
    returned dict is always a fresh copy, so the caller's block is never mutated.
    """
    archived = dict(block)
    if archived.pop("base64", None) is not None:
        archived["url"] = path
        return archived
    url = archived.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        archived["url"] = path
        return archived
    image_url = archived.get("image_url")
    if isinstance(image_url, dict):
        archived["image_url"] = {**image_url, "url": path}
    return archived


_ELIDED_INLINE_DATA = "<inline data omitted -- see the manifest at this path>"
"""Stand-in for an inline payload in the preview embedded in the stub message."""


def _without_inline_data(block: dict[str, Any]) -> dict[str, Any]:
    """Strip inline payloads from a block so it is safe to quote in the stub.

    Only blocks whose data could not be decoded still carry a payload by the time
    the manifest is built -- decodable media has already been swapped for a path
    by `_with_media_reference`. Those undecodable payloads are kept in the
    manifest for recoverability, but the stub goes back into the context window,
    so quoting one there would reinstate the base64 this path exists to remove.
    """
    elided = dict(block)
    if elided.get("base64"):
        elided["base64"] = _ELIDED_INLINE_DATA
    url = elided.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        elided["url"] = _ELIDED_INLINE_DATA
    image_url = elided.get("image_url")
    if isinstance(image_url, dict) and isinstance(image_url.get("url"), str) and image_url["url"].startswith("data:"):
        elided["image_url"] = {**image_url, "url": _ELIDED_INLINE_DATA}
    return elided


def _archive_generic_result(msg: ToolMessage, large_tool_results_prefix: str) -> tuple[str, str, tuple[tuple[str, bytes], ...]]:
    """Serialize a generic tool result, moving inline media to binary files.

    Text-only results retain the historical plain-text archive format. Results
    with non-text blocks use a JSON manifest so every block remains represented.
    Decodable inline media is replaced by a backend path, keeping raw base64 out
    of both the replacement message and the manifest. Non-inline blocks, such as
    remote URLs or provider file references, are preserved verbatim.

    Returns `(archive, preview_source, uploads)`. `preview_source` is the archive
    with any surviving inline payload elided: the stub quotes a preview of it and
    claims that preview describes the file at the offload path, so it has to be
    the same document -- but it must not smuggle base64 back into the context.
    """
    content = _extract_text_from_message(msg)
    if _non_text_payload_chars(msg) == 0:
        return content, content, ()

    uploads: dict[str, bytes] = {}
    archived_blocks: list[dict[str, Any]] = []
    for block in msg.content_blocks:
        data_url = _extract_data_url(block)
        if data_url is None:
            archived_blocks.append(dict(block))
            continue
        decoded = _decode_data_url(data_url)
        if decoded is None:
            # Preserve malformed/unknown inline data in the manifest rather
            # than silently discarding a block that could not be decoded.
            archived_blocks.append(dict(block))
            continue
        raw, ext, _mime = decoded
        key = hashlib.sha256(raw).hexdigest()[:16]
        path = f"{large_tool_results_prefix}/media/{key}.{ext}"
        uploads[path] = raw
        archived_blocks.append(_with_media_reference(block, path))

    archive = json.dumps({"content": archived_blocks}, ensure_ascii=False, indent=2, default=str)
    preview_source = json.dumps({"content": [_without_inline_data(b) for b in archived_blocks]}, ensure_ascii=False, indent=2, default=str)
    return archive, preview_source, tuple(uploads.items())


def _build_candidate(msg: ToolMessage, tc_index: dict[str, dict[str, Any]], large_tool_results_prefix: str) -> _Candidate:
    """Build the replacement for one tail TM without touching the backend.

    Deferring the write lets the caller measure the replacement and discard it
    without leaving an orphaned `/large_tool_results/` file that no message
    points at.
    """
    original_path = _read_file_original_path(msg, tc_index)
    if original_path is not None:
        return _Candidate(message=_slice_read_file_tm(msg, original_path), pending_write=None)
    archive, preview_source, uploads = _archive_generic_result(msg, large_tool_results_prefix)
    # Preview the archive, not the extracted text: the stub tells the agent the
    # preview shows the file at `file_path`, and for a media-only result the
    # extracted text is empty -- promising a preview and delivering none.
    replacement, file_path = _build_offload_replacement(msg, preview_source, large_tool_results_prefix, drop_non_text=True)
    return _Candidate(message=replacement, pending_write=(file_path, archive), pending_uploads=uploads)


def _failed_upload_paths(uploads: tuple[tuple[str, bytes], ...], responses: list[FileUploadResponse]) -> list[str]:
    """Return paths whose upload response is missing or reports an error."""
    return [path for i, (path, _content) in enumerate(uploads) if i >= len(responses) or responses[i].error is not None]


def _commit_candidate(candidate: _Candidate, backend: BackendProtocol) -> list[str]:
    """Commit a candidate's binary uploads and manifest, returning failed paths."""
    if candidate.pending_write is None:
        return []
    if candidate.pending_uploads:
        try:
            responses = backend.upload_files(list(candidate.pending_uploads))
        except Exception:  # noqa: BLE001
            logger.warning("Context-overflow media upload raised unexpectedly", exc_info=True)
            return [path for path, _content in candidate.pending_uploads]
        if failures := _failed_upload_paths(candidate.pending_uploads, responses):
            return failures
    file_path, content = candidate.pending_write
    try:
        result = backend.write(file_path, content)
    except Exception:  # noqa: BLE001
        logger.warning("Context-overflow result archive write raised unexpectedly", exc_info=True)
        return [file_path]
    return [file_path] if result is None or result.error else []


async def _acommit_candidate(candidate: _Candidate, backend: BackendProtocol) -> list[str]:
    """Async variant of `_commit_candidate`."""
    if candidate.pending_write is None:
        return []
    if candidate.pending_uploads:
        try:
            responses = await backend.aupload_files(list(candidate.pending_uploads))
        except Exception:  # noqa: BLE001
            logger.warning("Context-overflow media upload raised unexpectedly", exc_info=True)
            return [path for path, _content in candidate.pending_uploads]
        if failures := _failed_upload_paths(candidate.pending_uploads, responses):
            return failures
    file_path, content = candidate.pending_write
    try:
        result = await backend.awrite(file_path, content)
    except Exception:  # noqa: BLE001
        logger.warning("Context-overflow result archive write raised unexpectedly", exc_info=True)
        return [file_path]
    return [file_path] if result is None or result.error else []


def _accept_candidate(original: ToolMessage, replacement: ToolMessage, candidate_tokens: int, current_tokens: int) -> bool:
    """Whether `replacement` is worth using in place of `original`.

    Accepts when the tail measurably shrinks, or when the replacement removes
    non-text payload -- the case the token counter under-reports, and the reason
    a media-bearing result can look "free" to keep. Rejecting otherwise is what
    keeps a small sibling from being inflated into a larger offload stub.
    """
    if candidate_tokens < current_tokens:
        return True
    return _non_text_payload_chars(original) > _non_text_payload_chars(replacement)


def _prepare_clip(
    preserved_messages: list[AnyMessage],
    *,
    keep: ContextSize,
    max_input_tokens: int | None,
    token_counter: TokenCounter,
) -> tuple[int, list[ToolMessage], int, int] | None:
    """Resolve the tail batch and decide whether clipping should begin.

    Returns `(start, tail, current_tokens, threshold)`, or `None` when there is
    no trailing ToolMessage batch or it isn't worth rewriting. The engage
    decision deliberately consults raw non-text payload as well as the token
    count, because the default counter cannot see inline media.
    """
    found = _find_tail_tool_message_batch(preserved_messages)
    if found is None:
        return None
    start, tail = found
    threshold = _derive_overflow_clip_threshold_tokens(keep, max_input_tokens)
    current_tokens = token_counter(tail)
    payload_chars = sum(_non_text_payload_chars(m) for m in tail)
    if current_tokens < threshold and payload_chars < _NON_TEXT_PAYLOAD_ENGAGE_CHARS:
        return None
    return start, tail, current_tokens, threshold


def _log_clip_outcome(current_tokens: int, threshold: int, write_failures: list[str]) -> None:
    """Surface a clip that fell short, and any backend write that failed.

    Without this, "every candidate was rejected", "every backend write failed",
    and "nothing needed clipping" are indistinguishable from the caller's side --
    it sees the same empty replacement list, then an uncaught `ContextOverflowError`
    from the retried model call with no breadcrumb explaining that the mitigation
    ran and fell short.
    """
    if write_failures:
        logger.warning(
            "Context-overflow tail clip could not offload %d tool result(s); their original content was kept inline. Failed paths: %s",
            len(write_failures),
            ", ".join(write_failures),
        )
    if current_tokens >= threshold:
        logger.warning(
            "Context-overflow tail clip reduced the trailing tool results to %d tokens, still at or above the %d-token budget. The retried model call may exceed the context window.",
            current_tokens,
            threshold,
        )


def _clip_overflow_tail(
    preserved_messages: list[AnyMessage],
    backend: BackendProtocol,
    *,
    keep: ContextSize,
    max_input_tokens: int | None,
    token_counter: TokenCounter,
    large_tool_results_prefix: str,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Shrink the trailing ToolMessage batch, message by message.

    Engages only when `preserved_messages` ends with consecutive ToolMessages
    that either reach `_derive_overflow_clip_threshold_tokens()` or carry at
    least `_NON_TEXT_PAYLOAD_ENGAGE_CHARS` of non-text payload. Every result is
    then considered; a replacement is used only when it shrinks the measured tail
    or removes non-text payload, so small siblings whose offload stub would be
    larger keep their originals. Each clipped non-`read_file` TM is written under
    `large_tool_results_prefix/{tool_call_id}` -- after the accept decision, so a
    rejected candidate leaves no orphaned file.

    Returns `(modified preserved_messages, tail to persist in state)`. The second
    element is the full tail, position for position: clipped messages plus the
    untouched originals of anything not clipped. Replacements carry the original
    ids so the `add_messages` reducer overwrites the originals when the caller
    propagates them via a `Command` update; a replacement whose original had no
    id is given one, since `add_messages` would otherwise append it alongside the
    oversized original instead of replacing it. Both lists are returned unchanged
    when nothing was clipped.
    """
    prepared = _prepare_clip(preserved_messages, keep=keep, max_input_tokens=max_input_tokens, token_counter=token_counter)
    if prepared is None:
        return preserved_messages, []
    start, tail, current_tokens, threshold = prepared
    tc_index = _build_tool_call_index(preserved_messages)
    new_tail: list[AnyMessage] = list(tail)
    write_failures: list[str] = []
    any_clipped = False
    for i, m in enumerate(tail):
        candidate = _build_candidate(m, tc_index, large_tool_results_prefix)
        r = candidate.message if candidate.message.id is not None else candidate.message.model_copy(update={"id": str(uuid.uuid4())})
        candidate_tail = [*new_tail[:i], r, *new_tail[i + 1 :]]
        candidate_tokens = token_counter(candidate_tail)
        if not _accept_candidate(m, r, candidate_tokens, current_tokens):
            continue
        if failures := _commit_candidate(candidate, backend):
            write_failures.extend(failures)
            continue
        new_tail = candidate_tail
        current_tokens = candidate_tokens
        any_clipped = True
    _log_clip_outcome(current_tokens, threshold, write_failures)
    if not any_clipped:
        return preserved_messages, []
    return [*preserved_messages[:start], *new_tail], new_tail


async def _aclip_overflow_tail(
    preserved_messages: list[AnyMessage],
    backend: BackendProtocol,
    *,
    keep: ContextSize,
    max_input_tokens: int | None,
    token_counter: TokenCounter,
    large_tool_results_prefix: str,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Async variant of `_clip_overflow_tail`.

    Identical apart from awaiting `backend.awrite`. The offloads cannot run
    concurrently: each accept decision depends on the tail as left by the
    decisions before it, so `asyncio.gather` would break the feedback loop that
    keeps a small sibling from being inflated.
    """
    prepared = _prepare_clip(preserved_messages, keep=keep, max_input_tokens=max_input_tokens, token_counter=token_counter)
    if prepared is None:
        return preserved_messages, []
    start, tail, current_tokens, threshold = prepared
    tc_index = _build_tool_call_index(preserved_messages)
    new_tail: list[AnyMessage] = list(tail)
    write_failures: list[str] = []
    any_clipped = False
    for i, m in enumerate(tail):
        candidate = _build_candidate(m, tc_index, large_tool_results_prefix)
        r = candidate.message if candidate.message.id is not None else candidate.message.model_copy(update={"id": str(uuid.uuid4())})
        candidate_tail = [*new_tail[:i], r, *new_tail[i + 1 :]]
        candidate_tokens = token_counter(candidate_tail)
        if not _accept_candidate(m, r, candidate_tokens, current_tokens):
            continue
        if failures := await _acommit_candidate(candidate, backend):
            write_failures.extend(failures)
            continue
        new_tail = candidate_tail
        current_tokens = candidate_tokens
        any_clipped = True
    _log_clip_outcome(current_tokens, threshold, write_failures)
    if not any_clipped:
        return preserved_messages, []
    return [*preserved_messages[:start], *new_tail], new_tail
