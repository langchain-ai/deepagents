"""Tests for the summarization-on-overflow tail clipping (`_overflow_clip`)."""

from __future__ import annotations

import base64
import json
import tempfile
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, MessageLikeRepresentation, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import FileUploadResponse, WriteResult
from deepagents.middleware._overflow_clip import (
    _aclip_overflow_tail,
    _clip_overflow_tail,
    _derive_overflow_clip_threshold_tokens,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain.agents.middleware.summarization import TokenCounter

_IMAGE_BYTES = b"hello" + (b"A" * 15_000)
_IMAGE_B64 = base64.b64encode(_IMAGE_BYTES).decode("ascii")
"""A base64 payload big enough that leaking it inline is unmistakable."""

# Mirrors the values `create_deep_agent` actually threads into the clip path, so
# tests that care about production reachability don't silently drift to a
# hand-picked threshold. `("fraction", 0.1)` of 200_000 == a 20_000-token budget.
_PROD_KEEP = ("fraction", 0.1)
_PROD_MAX_INPUT = 200_000


def _backend() -> FilesystemBackend:
    return FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)


class _FailingBackend(FilesystemBackend):
    """Backend whose writes always fail, to exercise the offload failure path."""

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error="simulated write failure")


class _FailingUploadBackend(FilesystemBackend):
    """Backend whose binary uploads always fail."""

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=path, error="simulated upload failure") for path, _content in files]


def _read_file_turn(tool_call_id: str, path: str, content: str | list[str | dict]) -> list[AnyMessage]:
    ai = AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "read_file", "args": {"file_path": path}}])
    tm = ToolMessage(tool_call_id=tool_call_id, name="read_file", content=content)
    return [ai, tm]


def _chars(msgs: Iterable[MessageLikeRepresentation]) -> int:
    return sum(len(str(m.content if isinstance(m, BaseMessage) else m)) for m in msgs)


def _mixed_batch() -> list[AnyMessage]:
    """One oversized `read_file` result plus a tiny result from another tool.

    The `grep` result is 4 chars; its offload stub is ~600, so clipping it would
    *grow* the tail and must be rejected.
    """
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "big", "name": "read_file", "args": {"file_path": "/big.txt"}},
                {"id": "small", "name": "grep", "args": {"pattern": "x"}},
            ],
        ),
        ToolMessage(tool_call_id="big", name="read_file", content="x" * 10_000),
        ToolMessage(tool_call_id="small", name="grep", content="tiny"),
    ]


def _small_generic_batch() -> list[AnyMessage]:
    """Two generic results that overflow only when counted together."""
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "first", "name": "grep", "args": {"pattern": "x"}},
                {"id": "second", "name": "grep", "args": {"pattern": "y"}},
            ],
        ),
        ToolMessage(tool_call_id="first", name="grep", content="x" * 3_000),
        ToolMessage(tool_call_id="second", name="grep", content="y" * 3_000),
    ]


def _clip_full(
    messages: list[AnyMessage],
    counter: TokenCounter,
    keep_tokens: int = 1,
    backend: FilesystemBackend | None = None,
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Run the sync clip, returning both the messages and the tail to persist."""
    return _clip_overflow_tail(
        messages,
        backend if backend is not None else _backend(),
        keep=("tokens", keep_tokens),
        max_input_tokens=1000,
        token_counter=counter,
        large_tool_results_prefix="/large_tool_results",
    )


def _clip(messages: list[AnyMessage], counter: TokenCounter, keep_tokens: int = 1) -> list[AnyMessage]:
    new_messages, _ = _clip_full(messages, counter, keep_tokens)
    return new_messages


# --------------------------------------------------------------------------- #
# Engage decision: "should clipping begin?"
# --------------------------------------------------------------------------- #


def test_derive_threshold_covers_every_context_size_shape() -> None:
    """Each `keep` shape maps to its documented budget (#4954).

    Production never passes `("tokens", N)` -- `create_deep_agent` uses
    `("fraction", 0.1)` and direct construction defaults to `("messages", 20)` --
    so the fraction and fallback branches are the ones that matter.
    """
    assert _derive_overflow_clip_threshold_tokens(("tokens", 1234), None) == 1234
    assert _derive_overflow_clip_threshold_tokens(("fraction", 0.1), 200_000) == 20_000
    assert _derive_overflow_clip_threshold_tokens(("fraction", 0.1), None) == 5_000
    assert _derive_overflow_clip_threshold_tokens(("messages", 20), 200_000) == 5_000


def test_media_only_tail_engages_under_the_real_token_counter() -> None:
    """A media-only tail is clipped with the production counter and config (#4954).

    Regression test for the gate being token-only: `count_tokens_approximately`
    scores a multi-megabyte base64 image at ~92 tokens, far below the 20_000-token
    budget, so a token-only gate never opens and the payload reaches the model
    that just rejected the request. The engage decision consults raw non-text
    payload for exactly this case.
    """
    messages = _read_file_turn("call_1", "/pic.png", [{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}])

    # Sanity-check the premise: the counter really cannot see this payload.
    assert count_tokens_approximately(messages[1:]) < 1_000

    new_messages, _ = _clip_overflow_tail(
        messages,
        _backend(),
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert _IMAGE_B64 not in str(new_messages[-1].content)
    assert "Non-text content" in new_messages[-1].content


def test_media_from_a_non_read_file_tool_is_archived_before_being_stripped() -> None:
    """Inline media from any tool is archived, not just from `read_file` (#4954).

    Screenshot/browser/MCP tools return images too; they route through the
    generic offload path. The replacement must not carry the payload over, but
    its recovery pointer must lead to a manifest that references the uploaded
    image rather than an empty text-only file.
    """
    backend = _backend()
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(
            tool_call_id="shot",
            name="browser_screenshot",
            content=[{"type": "text", "text": "page loaded"}, {"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
        ),
    ]

    new_messages, _ = _clip_overflow_tail(
        messages,
        backend,
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert _IMAGE_B64 not in str(new_messages[-1].content)
    assert "saved in the filesystem" in str(new_messages[-1].content)
    archive = backend.read("/large_tool_results/shot")
    assert archive.file_data is not None
    manifest = json.loads(archive.file_data["content"])
    assert manifest["content"][0] == {"type": "text", "text": "page loaded"}
    media_path = manifest["content"][1]["url"]
    assert manifest["content"][1] == {"type": "image", "mime_type": "image/png", "url": media_path}
    media = backend.read(media_path)
    assert media.file_data is not None
    assert base64.b64decode(media.file_data["content"]) == _IMAGE_BYTES


def test_media_only_generic_result_has_a_nonempty_recovery_manifest() -> None:
    """A media-only screenshot remains recoverable from its offload pointer."""
    backend = _backend()
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(
            tool_call_id="shot",
            name="browser_screenshot",
            content=[{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
        ),
    ]

    new_messages, _ = _clip_overflow_tail(
        messages,
        backend,
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert _IMAGE_B64 not in str(new_messages[-1].content)
    archive = backend.read("/large_tool_results/shot")
    assert archive.file_data is not None
    manifest = json.loads(archive.file_data["content"])
    assert manifest["content"][0]["type"] == "image"
    assert manifest["content"][0]["url"].startswith("/large_tool_results/media/")


def test_non_inline_media_block_is_preserved_in_recovery_manifest() -> None:
    """External media references survive even though no binary upload is needed."""
    backend = _backend()
    block = {"type": "audio", "url": "https://example.com/result.mp3", "mime_type": "audio/mpeg"}
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "audio", "name": "recording", "args": {}}]),
        ToolMessage(tool_call_id="audio", name="recording", content=[block]),
    ]

    new_messages, _ = _clip_full(messages, _chars, backend=backend)

    # The block itself is gone from the message; only the manifest preview
    # mentions the remote URL, as text, which costs ~30 chars rather than a fetch.
    assert all(b["type"] == "text" for b in new_messages[-1].content_blocks)
    archive = backend.read("/large_tool_results/audio")
    assert archive.file_data is not None
    assert json.loads(archive.file_data["content"])["content"] == [block]


def test_media_only_stub_previews_the_manifest_it_points_at() -> None:
    """The stub's preview describes the file at the offload path (#4954).

    The stub states that the preview shows the head and tail of the result at
    `file_path`. Previewing the extracted text instead leaves a media-only result
    promising a preview and delivering an empty one.
    """
    backend = _backend()
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(tool_call_id="shot", name="browser_screenshot", content=[{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}]),
    ]

    new_messages, _ = _clip_full(messages, _chars, backend=backend)

    stub = str(new_messages[-1].content)
    media_path = json.loads(backend.read("/large_tool_results/shot").file_data["content"])["content"][0]["url"]
    assert media_path in stub
    assert '"type": "image"' in stub


def test_undecodable_payload_is_archived_but_kept_out_of_the_stub() -> None:
    """An undecodable inline payload survives in the manifest but not in the context (#4954).

    Such a block is preserved verbatim so nothing is lost, which puts base64 into
    the manifest -- so the stub's preview of that manifest has to elide it, or the
    clip hands the payload straight back to the model it was removed for.
    """
    backend = _backend()
    payload = "!!!not-valid-base64!!!" + ("Z" * 8_000)
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(tool_call_id="shot", name="browser_screenshot", content=[{"type": "image", "base64": payload, "mime_type": "image/png"}]),
    ]

    new_messages, _ = _clip_full(messages, _chars, backend=backend)

    stub = str(new_messages[-1].content)
    assert payload[:40] not in stub
    assert "inline data omitted" in stub
    # Still recoverable: the manifest keeps the payload the preview elided.
    assert payload in backend.read("/large_tool_results/shot").file_data["content"]


# --------------------------------------------------------------------------- #
# Accept decision: "is this replacement worth using?"
# --------------------------------------------------------------------------- #


def test_replacement_that_does_not_shrink_the_tail_is_rejected() -> None:
    """A candidate whose token count ties the original is discarded (#4954).

    The constant counter makes `candidate_tokens == current_tokens` for every
    candidate, pinning the `>=` rejection boundary. It says nothing about notice
    content -- the replacement is built and thrown away before any notice is
    observable.
    """
    messages = _read_file_turn("call_1", "/f.txt", "short")

    new_messages, replacements = _clip_full(messages, lambda _msgs: 10_000)

    assert new_messages is messages
    assert replacements == []
    assert new_messages[-1].content == "short"


def test_sibling_whose_offload_stub_would_grow_the_tail_is_left_untouched() -> None:
    """A tiny sibling keeps its original rather than being inflated (#4954).

    This is what protects small results: the `grep` stub is ~600 chars against a
    4-char original, so replacing it would enlarge the very tail being shrunk.
    """
    messages = _mixed_batch()

    new_messages = _clip(messages, _chars, keep_tokens=5_000)

    assert "Output was truncated" in new_messages[-2].content
    assert new_messages[-1].content == "tiny"


def test_rejected_candidate_leaves_no_orphaned_offload_file() -> None:
    """A rejected generic candidate must not leave a file nothing points at (#4954).

    The offload write is deferred until after the accept decision. Otherwise the
    tiny sibling's content lands in `/large_tool_results/` while its message keeps
    the original content -- and the agent is told to grep that directory.
    """
    backend = _backend()
    messages: list[AnyMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "accepted", "name": "grep", "args": {"pattern": "x"}},
                {"id": "rejected", "name": "grep", "args": {"pattern": "w"}},
            ],
        ),
        # Large enough that its offload stub is a clear win.
        ToolMessage(tool_call_id="accepted", name="grep", content="x" * 60_000),
        # So small that the ~600-char stub would grow the tail.
        ToolMessage(tool_call_id="rejected", name="grep", content="w" * 20),
    ]

    new_messages, _ = _clip_full(messages, _chars, backend=backend)

    assert "saved in the filesystem" in new_messages[-2].content
    assert new_messages[-1].content == "w" * 20
    # The accepted offload was written; the rejected one left nothing behind.
    assert backend.read("/large_tool_results/accepted").error is None
    assert backend.read("/large_tool_results/rejected").error is not None


def test_individually_small_generic_results_are_all_offloaded() -> None:
    """Collectively large generic results are each reduced (#4954).

    There is no "the tail fits now, stop" condition: the overflow already proved
    the advertised limit was not respected, so every result that can shrink does.
    """
    messages = _small_generic_batch()

    new_messages = _clip(messages, _chars, keep_tokens=5_000)

    assert "saved in the filesystem" in new_messages[-2].content
    assert "saved in the filesystem" in new_messages[-1].content
    assert _chars(new_messages[-2:]) < 5_000


def test_individually_small_read_results_use_path_pointer() -> None:
    """Collectively large `read_file` results are each replaced by a path pointer."""
    messages: list[AnyMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "first", "name": "read_file", "args": {"file_path": "/first.txt"}},
                {"id": "second", "name": "read_file", "args": {"file_path": "/second.txt"}},
            ],
        ),
        ToolMessage(tool_call_id="first", name="read_file", content="x" * 3_000),
        ToolMessage(tool_call_id="second", name="read_file", content="y" * 3_000),
    ]

    new_messages = _clip(messages, _chars, keep_tokens=5_000)

    assert "Output was omitted" in new_messages[-2].content
    assert "/first.txt" in new_messages[-2].content
    assert "Output was omitted" in new_messages[-1].content
    assert "/second.txt" in new_messages[-1].content
    # A result that was not truncated must not claim it was.
    assert "Output was truncated" not in new_messages[-2].content
    assert _chars(new_messages[-2:]) < 5_000


# --------------------------------------------------------------------------- #
# `_slice_read_file_tm` notice shapes
# --------------------------------------------------------------------------- #


def test_truncated_result_keeps_its_head_preview() -> None:
    """A truncated `read_file` result retains the head of the file (#4954).

    The notice promises a preview, so one has to be there. Without this, blanking
    the content entirely while still emitting "Output was truncated" goes
    unnoticed.
    """
    body = "HEAD-MARKER\n" + ("m" * 20_000) + "\nTAIL-MARKER"
    messages = _read_file_turn("call_1", "/big.txt", body)

    clipped = _clip(messages, _chars, keep_tokens=5_000)[-1].content

    assert clipped.startswith("HEAD-MARKER\n")
    assert "Output was truncated" in clipped
    preview = clipped.split("[Output was truncated")[0]
    assert "TAIL-MARKER" not in preview
    # Exactly the first `_SLICE_CHARS` of the file, and nothing more.
    assert preview.rstrip("\n") == body[:4_000]


def test_short_text_alongside_media_keeps_the_text() -> None:
    """Text accompanying a dropped media block survives (#4954).

    The omit-the-text shortcut only applies when there is nothing else to keep;
    a caption next to an image is the payload the agent still needs.
    """
    messages = _read_file_turn(
        "call_1",
        "/pic.png",
        [{"type": "text", "text": "CAPTION"}, {"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
    )

    clipped = _clip(messages, _chars, keep_tokens=5_000)[-1].content

    assert "CAPTION" in clipped
    assert "Non-text content" in clipped
    assert "Output was omitted" not in clipped
    assert _IMAGE_B64 not in clipped


def test_truncated_result_with_media_gets_both_notices() -> None:
    """A long text + media `read_file` result reports both reductions (#4954)."""
    messages = _read_file_turn(
        "call_1",
        "/report.pdf",
        [{"type": "text", "text": "T" * 20_000}, {"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
    )

    clipped = _clip(messages, _chars, keep_tokens=5_000)[-1].content

    assert "Output was truncated" in clipped
    assert "Non-text content" in clipped
    assert clipped.startswith("T" * 100)
    assert _IMAGE_B64 not in clipped


# --------------------------------------------------------------------------- #
# Return contract
# --------------------------------------------------------------------------- #


def test_replacements_carry_original_ids_and_mirror_the_tail() -> None:
    """The persisted tail mirrors the clipped tail and reuses original ids (#4954).

    The ids are load-bearing: `add_messages` overwrites by id, so a replacement
    must carry the original's id to replace rather than accompany it.
    """
    messages = _mixed_batch()
    messages[1].id = "orig-big"
    messages[2].id = "orig-small"

    new_messages, replacements = _clip_full(messages, _chars, keep_tokens=5_000)

    assert replacements == new_messages[-2:]
    assert [m.id for m in replacements] == ["orig-big", "orig-small"]


def test_replacement_gets_an_id_when_the_original_had_none() -> None:
    """A replacement for an id-less original is given one (#4954).

    `add_messages` appends a message with no id, which would leave the oversized
    original in state *next to* the clipped stub -- growing the context this path
    exists to shrink.
    """
    messages = _read_file_turn("call_1", "/big.txt", "x" * 20_000)
    assert messages[1].id is None

    _new_messages, replacements = _clip_full(messages, _chars, keep_tokens=5_000)

    assert replacements[0].id is not None


def test_prefix_and_length_are_preserved() -> None:
    """Clipping rewrites only the tail, leaving the AIMessage that owns the tool_calls.

    Dropping the prefix would orphan the ToolMessages from their `tool_calls`,
    which providers reject outright.
    """
    messages = _mixed_batch()

    new_messages = _clip(messages, _chars, keep_tokens=5_000)

    assert len(new_messages) == len(messages)
    assert new_messages[0] is messages[0]


def test_write_failure_keeps_the_original_content() -> None:
    """A failed offload write leaves the original message intact (#4954).

    The message must not be replaced by a stub pointing at a file that was never
    written, or the content becomes unrecoverable.
    """
    messages = _small_generic_batch()

    new_messages, replacements = _clip_full(
        messages, _chars, keep_tokens=5_000, backend=_FailingBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
    )

    assert new_messages is messages
    assert replacements == []
    assert new_messages[-1].content == "y" * 3_000


def test_media_upload_failure_keeps_the_original_content() -> None:
    """A failed binary upload cannot produce a misleading recovery pointer."""
    backend = _FailingUploadBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(
            tool_call_id="shot",
            name="browser_screenshot",
            content=[{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
        ),
    ]

    new_messages, replacements = _clip_overflow_tail(
        messages,
        backend,
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert new_messages is messages
    assert replacements == []
    assert _IMAGE_B64 in str(new_messages[-1].content)
    assert backend.read("/large_tool_results/shot").error is not None


# --------------------------------------------------------------------------- #
# Async parity
# --------------------------------------------------------------------------- #


async def _aclip(messages: list[AnyMessage], counter: TokenCounter, keep_tokens: int = 1) -> list[AnyMessage]:
    new_messages, _ = await _aclip_overflow_tail(
        messages,
        _backend(),
        keep=("tokens", keep_tokens),
        max_input_tokens=1000,
        token_counter=counter,
        large_tool_results_prefix="/large_tool_results",
    )
    return new_messages


async def test_async_clip_matches_sync_clip() -> None:
    """The async path produces the same contents as the sync path on the same input.

    Compares against `_clip_overflow_tail` directly rather than restating literal
    expectations, so drift between the hand-duplicated loops fails here.
    """
    sync_messages = _clip(_mixed_batch(), _chars, keep_tokens=5_000)
    async_messages = await _aclip(_mixed_batch(), _chars, keep_tokens=5_000)

    assert [str(m.content) for m in async_messages] == [str(m.content) for m in sync_messages]


async def test_async_media_result_is_clipped() -> None:
    """The async path also strips inline media (#4954).

    The async loop is a hand-written twin of the sync one; without this the media
    fix could be applied to one and missed in the other.
    """
    messages = _read_file_turn("call_1", "/pic.png", [{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}])

    new_messages, _ = await _aclip_overflow_tail(
        messages,
        _backend(),
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert _IMAGE_B64 not in str(new_messages[-1].content)
    assert "Non-text content" in new_messages[-1].content


async def test_async_generic_media_result_is_archived() -> None:
    """The async generic path uploads media before writing its manifest."""
    backend = _backend()
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(
            tool_call_id="shot",
            name="browser_screenshot",
            content=[{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
        ),
    ]

    new_messages, _ = await _aclip_overflow_tail(
        messages,
        backend,
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert _IMAGE_B64 not in str(new_messages[-1].content)
    archive = backend.read("/large_tool_results/shot")
    assert archive.file_data is not None
    manifest = json.loads(archive.file_data["content"])
    media = backend.read(manifest["content"][0]["url"])
    assert media.file_data is not None
    assert base64.b64decode(media.file_data["content"]) == _IMAGE_BYTES


async def test_async_media_upload_failure_keeps_the_original_content() -> None:
    """The async path also refuses a pointer when binary upload fails."""
    backend = _FailingUploadBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
    messages: list[AnyMessage] = [
        AIMessage(content="", tool_calls=[{"id": "shot", "name": "browser_screenshot", "args": {}}]),
        ToolMessage(
            tool_call_id="shot",
            name="browser_screenshot",
            content=[{"type": "image", "base64": _IMAGE_B64, "mime_type": "image/png"}],
        ),
    ]

    new_messages, replacements = await _aclip_overflow_tail(
        messages,
        backend,
        keep=_PROD_KEEP,
        max_input_tokens=_PROD_MAX_INPUT,
        token_counter=count_tokens_approximately,
        large_tool_results_prefix="/large_tool_results",
    )

    assert new_messages is messages
    assert replacements == []
    assert _IMAGE_B64 in str(new_messages[-1].content)
    assert backend.read("/large_tool_results/shot").error is not None


async def test_async_rejects_a_replacement_that_does_not_shrink_the_tail() -> None:
    """The async path applies the same rejection boundary as the sync path."""
    messages = _read_file_turn("call_1", "/f.txt", "short")

    new_messages = await _aclip(messages, lambda _msgs: 10_000)

    assert new_messages is messages
    assert new_messages[-1].content == "short"


async def test_async_individually_small_generic_results_are_all_offloaded() -> None:
    """The async path also reduces collectively large generic results."""
    new_messages = await _aclip(_small_generic_batch(), _chars, keep_tokens=5_000)

    assert "saved in the filesystem" in new_messages[-2].content
    assert "saved in the filesystem" in new_messages[-1].content
    assert _chars(new_messages[-2:]) < 5_000
