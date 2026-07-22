"""Read-side clipping for the summarization-on-overflow fallback path.

When `SummarizationMiddleware`'s `wrap_model_call` catches a
`ContextOverflowError`, it falls through to summarization and *also* invokes
`_clip_overflow_tail` (or its async variant) to shrink the trailing
ToolMessage batch in the preserved suffix. Two per-TM paths:

- `read_file` tool result: head-slice the content and append a notice
    pointing back to the original `file_path` argument. No new backend write
    is needed because the original file already lives at that path.
- Any other tool result: full offload to `/large_tool_results/{tool_call_id}`
    via the shared eviction helper, then replace the message with a
    `TOO_LARGE_TOOL_MSG` stub.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from deepagents.middleware._message_eviction import (
    _aoffload_tool_message_content,
    _build_evicted_content,
    _extract_text_from_message,
    _offload_tool_message_content,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.summarization import ContextSize, TokenCounter

    from deepagents.backends.protocol import BackendProtocol

_READ_FILE_CLIP_CHARS = 4_000
"""Head-slice budget for a `read_file` tail ToolMessage.

Results whose text fits within this are left untouched (no truncation notice);
longer results are head-sliced to this many chars with a recovery notice.
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


def _slice_read_file_tm(msg: ToolMessage, original_path: str) -> ToolMessage | None:
    """Head-slice an oversized `read_file` ToolMessage, preserving any media blocks.

    `read_file` results don't need a fresh `/large_tool_results/{tcid}` write -- the
    full file is already on the backend at `original_path`, and the agent can
    recover with `read_file(file_path=original_path, offset=N, limit=K)`. The
    truncation notice mirrors `READ_FILE_TRUNCATION_MSG` in shape so the
    agent encounters a consistent format whether the tool truncated itself
    or the middleware did.

    Returns `None` when the text already fits within `_READ_FILE_CLIP_CHARS`, so
    the caller keeps the original message untouched -- this avoids appending a
    misleading truncation notice to a small result. When the text does overflow,
    only the text is head-sliced; non-text blocks (images/audio/video) are carried
    through via `_build_evicted_content`, matching the offload/eviction path
    instead of silently dropping media.
    """
    content = _extract_text_from_message(msg)
    if len(content) <= _READ_FILE_CLIP_CHARS:
        # Nothing to truncate; leave the message (and any media blocks) as-is.
        return None
    notice = (
        f"\n\n[Output was truncated due to context window size limits. "
        f"The full content is at {original_path}. "
        f"Use read_file with offset and limit parameters to retrieve specific portions. "
        f"For example, to read the first 100 lines, call read_file with file_path='{original_path}', offset=0, limit=100.]"
    )
    sliced_text = content[:_READ_FILE_CLIP_CHARS] + notice
    return msg.model_copy(update={"content": _build_evicted_content(msg, sliced_text)})


def _read_file_original_path(msg: ToolMessage, tc_index: dict[str, dict[str, Any]]) -> str | None:
    """Return the `file_path` arg from the matching read_file tool_call, or `None`."""
    tc = tc_index.get(msg.tool_call_id) if msg.tool_call_id else None
    if not tc or tc.get("name") != "read_file":
        return None
    path = tc.get("args", {}).get("file_path")
    return path if isinstance(path, str) and path else None


def _clip_one_tail_message(
    msg: ToolMessage,
    tc_index: dict[str, dict[str, Any]],
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    """Apply the appropriate per-TM clip: read_file slice vs generic eviction."""
    original_path = _read_file_original_path(msg, tc_index)
    if original_path is not None:
        return _slice_read_file_tm(msg, original_path)
    return _offload_tool_message_content(msg, _extract_text_from_message(msg), backend, large_tool_results_prefix)


async def _aclip_one_tail_message(
    msg: ToolMessage,
    tc_index: dict[str, dict[str, Any]],
    backend: BackendProtocol,
    large_tool_results_prefix: str,
) -> ToolMessage | None:
    """Async variant of `_clip_one_tail_message`."""
    original_path = _read_file_original_path(msg, tc_index)
    if original_path is not None:
        return _slice_read_file_tm(msg, original_path)
    return await _aoffload_tool_message_content(msg, _extract_text_from_message(msg), backend, large_tool_results_prefix)


def _clip_overflow_tail(
    preserved_messages: list[AnyMessage],
    backend: BackendProtocol,
    *,
    keep: ContextSize,
    max_input_tokens: int | None,
    token_counter: TokenCounter,
    large_tool_results_prefix: str,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Offload the trailing ToolMessage batch when it's large enough to matter.

    Engages only when `preserved_messages` ends with consecutive ToolMessages
    whose combined token count reaches `_derive_overflow_clip_threshold_tokens()`.
    Each large TM is written under `large_tool_results/{tool_call_id}` and
    replaced in-place by an offload-pointer ToolMessage.

    Returns `(modified preserved_messages, replacement TMs to persist in
    state)`. Replacements carry the original ids so the `add_messages`
    reducer overwrites the originals when the caller propagates them via
    a `Command` update. The replacements list omits any TM whose backend
    write failed (those keep their originals in both lists).
    """
    found = _find_tail_tool_message_batch(preserved_messages)
    if found is None:
        return preserved_messages, []
    start, tail = found
    if token_counter(tail) < _derive_overflow_clip_threshold_tokens(keep, max_input_tokens):
        return preserved_messages, []
    tc_index = _build_tool_call_index(preserved_messages)
    new_tail: list[AnyMessage] = []
    any_clipped = False
    for m in tail:
        r = _clip_one_tail_message(m, tc_index, backend, large_tool_results_prefix)
        if r is not None:
            if r.id is None:
                r = r.model_copy(update={"id": str(uuid.uuid4())})
            new_tail.append(r)
            any_clipped = True
        else:
            new_tail.append(m)
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
    """Async variant of `_clip_overflow_tail`. Offloads each tail TM concurrently."""
    found = _find_tail_tool_message_batch(preserved_messages)
    if found is None:
        return preserved_messages, []
    start, tail = found
    if token_counter(tail) < _derive_overflow_clip_threshold_tokens(keep, max_input_tokens):
        return preserved_messages, []
    tc_index = _build_tool_call_index(preserved_messages)
    results = await asyncio.gather(*(_aclip_one_tail_message(m, tc_index, backend, large_tool_results_prefix) for m in tail))
    new_tail: list[AnyMessage] = []
    any_clipped = False
    for r, m in zip(results, tail, strict=True):
        if r is not None:
            if r.id is None:
                r = r.model_copy(update={"id": str(uuid.uuid4())})  # noqa: PLW2901
            new_tail.append(r)
            any_clipped = True
        else:
            new_tail.append(m)
    if not any_clipped:
        return preserved_messages, []
    return [*preserved_messages[:start], *new_tail], new_tail
