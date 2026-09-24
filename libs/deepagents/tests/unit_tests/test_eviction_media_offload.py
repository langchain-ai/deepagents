"""Offloading inline media out of evicted `HumanMessage`s.

A message that is already a turn old does not need its image re-sent on every
subsequent model call, so `FilesystemMiddleware` uploads that media once and puts
a text pointer in its place. These tests pin the three behaviours that make that
safe: the pointer is plain text (a backend-relative URL is not fetchable by a
provider), a failed upload leaves the block inline rather than dropping it, and
the current turn's media is never touched.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain.agents.middleware.types import AgentState, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware._message_eviction import (
    _aoffload_inline_media_blocks,
    _extract_data_url,
    _inline_media_payload_chars,
    _message_char_size,
    _offload_inline_media_blocks,
)
from deepagents.middleware.filesystem import (
    _EVICTED_MEDIA_KEY,
    FilesystemMiddleware,
    _apply_evicted_media,
)
from tests.unit_tests.middleware.test_summarization_middleware import (
    MockBackend,
    make_mock_model,
    make_mock_runtime,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepagents.backends.protocol import FileUploadResponse

_PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes" * 4
_BIG_PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"big-image-bytes" * 40
"""Large enough that its base64 form alone exceeds a 100-token (400 char) limit."""


def _base64_block(payload: bytes = _PAYLOAD) -> dict[str, Any]:
    """Standard LangChain image block carrying an explicit `base64` field."""
    return {"type": "image", "base64": base64.b64encode(payload).decode(), "mime_type": "image/png"}


def _data_url_block(payload: bytes = _PAYLOAD) -> dict[str, Any]:
    """Image block whose `url` is an inline `data:` URL."""
    return {"type": "image", "url": f"data:image/png;base64,{base64.b64encode(payload).decode()}"}


def _openai_block(payload: bytes = _PAYLOAD) -> dict[str, Any]:
    """OpenAI-style `image_url` block whose inner `url` is a `data:` URL."""
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(payload).decode()}"}}


def _remote_block() -> dict[str, Any]:
    """Image block referencing a remote URL -- nothing to offload."""
    return {"type": "image", "url": "https://example.com/field.png"}


def _media_blocks(message: HumanMessage) -> list[dict[str, Any]]:
    return [block for block in message.content_blocks if block["type"] != "text"]


def _pointer_texts(message: HumanMessage) -> list[str]:
    return [block["text"] for block in message.content_blocks if block["type"] == "text" and "offloaded to the filesystem" in block.get("text", "")]


def _request(messages: list[Any]) -> ModelRequest:
    model = make_mock_model()
    return ModelRequest(
        model=model,
        messages=messages,
        state=cast("AgentState", {"messages": messages}),
        runtime=make_mock_runtime(),
    )


class _FailingUploadBackend(MockBackend):
    """Backend whose media uploads fail, to exercise the fail-safe path."""

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        msg = "simulated upload outage"
        raise RuntimeError(msg)


class TestInlineMediaDetection:
    """`_inline_media_payload_chars` must agree with `_extract_data_url`."""

    @pytest.mark.parametrize("block_factory", [_base64_block, _data_url_block, _openai_block])
    def test_every_inline_shape_is_measured_and_extracted(self, block_factory: Callable[[], dict[str, Any]]) -> None:
        block = block_factory()

        assert _extract_data_url(block) is not None
        assert _inline_media_payload_chars(block) > 0

    def test_remote_url_is_neither_measured_nor_extracted(self) -> None:
        block = _remote_block()

        assert _extract_data_url(block) is None
        assert _inline_media_payload_chars(block) == 0

    def test_text_block_is_neither_measured_nor_extracted(self) -> None:
        block = {"type": "text", "text": "data:not-really-a-url"}

        assert _extract_data_url(block) is None
        assert _inline_media_payload_chars(block) == 0

    def test_message_char_size_adds_media_to_text(self) -> None:
        message = HumanMessage(content=[{"type": "text", "text": "caption"}, _base64_block()])

        text_only = len("caption")
        assert _message_char_size(message) > text_only
        assert _message_char_size(message) == text_only + _inline_media_payload_chars(_base64_block())

    def test_message_char_size_equals_text_length_without_media(self) -> None:
        message = HumanMessage(content="plain caption")

        assert _message_char_size(message) == len("plain caption")


class TestOffloadInlineMediaBlocks:
    """Pointer alignment, dedup, and the fail-safe on upload failure."""

    def test_pointers_align_with_non_text_blocks(self) -> None:
        backend = MockBackend()
        message = HumanMessage(content=[{"type": "text", "text": "caption"}, _base64_block(), _remote_block()])

        pointers = _offload_inline_media_blocks(message, backend, "/conversation_history/media")

        assert pointers is not None
        assert len(pointers) == 2, "one entry per non-text block"
        assert isinstance(pointers[0], str)
        assert pointers[1] is None, "a remote URL is not inline media"

    def test_uploaded_path_is_content_addressed(self) -> None:
        backend = MockBackend()
        message = HumanMessage(content=[_base64_block()])

        pointers = _offload_inline_media_blocks(message, backend, "/conversation_history/media")

        expected_key = hashlib.sha256(_PAYLOAD).hexdigest()[:16]
        assert pointers is not None
        assert f"/conversation_history/media/{expected_key}.png" in pointers[0]

    def test_identical_media_is_uploaded_once(self) -> None:
        backend = MockBackend()
        message = HumanMessage(content=[_base64_block(), _base64_block()])

        pointers = _offload_inline_media_blocks(message, backend, "/conversation_history/media")

        assert pointers is not None
        assert pointers[0] == pointers[1]
        assert len(backend.write_calls) == 1, "content-hash dedup should upload once"

    def test_pointer_is_text_not_a_backend_url_media_block(self) -> None:
        backend = MockBackend()
        message = HumanMessage(content=[_base64_block()])

        pointers = _offload_inline_media_blocks(message, backend, "/conversation_history/media")

        assert pointers is not None
        assert isinstance(pointers[0], str), "a typed media block would carry a url the provider cannot fetch"
        assert "read_file" in pointers[0]

    def test_nothing_to_offload_returns_none(self) -> None:
        backend = MockBackend()
        message = HumanMessage(content=[{"type": "text", "text": "caption"}, _remote_block()])

        assert _offload_inline_media_blocks(message, backend, "/conversation_history/media") is None
        assert backend.write_calls == []

    def test_text_only_message_returns_none(self) -> None:
        backend = MockBackend()

        assert _offload_inline_media_blocks(HumanMessage(content="caption"), backend, "/media") is None

    def test_failed_upload_keeps_every_block_inline(self) -> None:
        message = HumanMessage(content=[_base64_block()])

        pointers = _offload_inline_media_blocks(message, _FailingUploadBackend(), "/conversation_history/media")

        assert pointers is None, "no data may be lost when the backend is unavailable"

    def test_partial_failure_only_replaces_uploaded_blocks(self) -> None:
        class _SecondBlockFails(MockBackend):
            def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                if self.write_calls:
                    msg = "simulated upload outage"
                    raise RuntimeError(msg)
                return super().upload_files(files)

        backend = _SecondBlockFails()
        first, second = _base64_block(b"first-payload"), _base64_block(b"second-payload")
        message = HumanMessage(content=[first, second])

        pointers = _offload_inline_media_blocks(message, backend, "/conversation_history/media")

        assert pointers is not None
        assert isinstance(pointers[0], str)
        assert pointers[1] is None, "a failed block stays inline rather than being dropped"

    @pytest.mark.asyncio
    async def test_async_variant_matches_sync(self) -> None:
        message = HumanMessage(content=[{"type": "text", "text": "caption"}, _base64_block(), _remote_block()])

        sync_pointers = _offload_inline_media_blocks(message, MockBackend(), "/conversation_history/media")
        async_pointers = await _aoffload_inline_media_blocks(message, MockBackend(), "/conversation_history/media")

        assert sync_pointers is not None
        assert async_pointers == sync_pointers


class TestApplyEvictedMedia:
    """The request-only rewrite of a tagged message."""

    def test_replaces_offloaded_blocks_and_keeps_others(self) -> None:
        message = HumanMessage(content=[{"type": "text", "text": "caption"}, _base64_block(), _remote_block()])
        message.additional_kwargs[_EVICTED_MEDIA_KEY] = ["[image offloaded to /x.png; call read_file on that path to view it]", None]

        result = _apply_evicted_media(message)

        blocks = result.content_blocks
        assert blocks[0] == {"type": "text", "text": "caption"}
        assert blocks[1]["type"] == "text", "the offloaded image became a pointer"
        assert "read_file" in blocks[1]["text"]
        assert blocks[2] == _remote_block(), "a remote reference keeps its place"
        assert message.content_blocks[1]["type"] == "image", "content in state is unchanged"

    def test_untagged_message_is_returned_as_is(self) -> None:
        message = HumanMessage(content=[_base64_block()])

        assert _apply_evicted_media(message) is message

    def test_mismatched_pointer_count_leaves_message_intact(self) -> None:
        message = HumanMessage(content=[_base64_block(), _remote_block()])
        message.additional_kwargs[_EVICTED_MEDIA_KEY] = ["only-one-pointer"]

        result = _apply_evicted_media(message)

        assert result is message, "positional mismatch must not rewrite the wrong block"


class TestEvictionPathIntegration:
    """`_evict_and_truncate_messages` end to end, without a graph."""

    def _middleware(self, backend: MockBackend, *, limit: int = 100) -> FilesystemMiddleware:
        return FilesystemMiddleware(backend=backend, human_message_token_limit_before_evict=limit)

    def test_historical_media_is_offloaded_but_current_turn_is_not(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        historical = HumanMessage(content=[{"type": "text", "text": "老图"}, _base64_block(_BIG_PAYLOAD)])
        current = HumanMessage(content=[{"type": "text", "text": "新图"}, _base64_block(_BIG_PAYLOAD)])
        messages = [historical, AIMessage(content="ok"), current]

        result = middleware._evict_and_truncate_messages(_request(messages))

        assert result is not None
        processed, command = result
        assert command is not None, "offloading media must be persisted to state"
        tagged = [msg for msg in processed if msg.additional_kwargs.get(_EVICTED_MEDIA_KEY)]
        assert len(tagged) == 1, "only the historical message is tagged"
        assert _media_blocks(processed[0]) == [], "the historical image was replaced, leaving only text"
        assert _pointer_texts(processed[0]), "the historical image became a read_file pointer"
        assert _media_blocks(processed[2])[0]["type"] == "image", "the current turn keeps its image inline"

    def test_small_historical_media_stays_inline(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        historical = HumanMessage(content=[{"type": "text", "text": "小图"}, _base64_block()])
        messages = [historical, AIMessage(content="ok"), HumanMessage(content="接着看")]

        result = middleware._evict_and_truncate_messages(_request(messages))

        assert backend.write_calls == [], "a small image must not cost a read_file round-trip"
        assert result is None, "nothing to rewrite, so the fast path is taken"

    def test_media_aware_threshold_evicts_a_short_caption(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        caption = {"type": "text", "text": "短caption"}
        message = HumanMessage(content=[caption, _base64_block(_BIG_PAYLOAD)])
        assert len(caption["text"]) < 100 * 4, "text alone is under the threshold"
        assert _message_char_size(message) > 100 * 4, "text plus media is over it"

        result = middleware._evict_and_truncate_messages(_request([message]))

        assert result is not None
        _, command = result
        assert command is not None, "the media-aware measurement must trip the threshold"

    def test_already_tagged_media_is_not_reuploaded(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        tagged = HumanMessage(content=[_base64_block(_BIG_PAYLOAD)])
        tagged.additional_kwargs[_EVICTED_MEDIA_KEY] = ["[image offloaded to /x.png; call read_file on that path to view it]"]
        messages = [tagged, AIMessage(content="ok"), HumanMessage(content="next")]

        result = middleware._evict_and_truncate_messages(_request(messages))

        assert backend.write_calls == [], "an already-offloaded message is not uploaded again"
        assert result is not None
        processed, _ = result
        assert processed[0].content_blocks[0]["type"] == "text", "but it is still rewritten for the request"

    def test_media_offload_shares_the_eviction_switch(self) -> None:
        backend = MockBackend()
        middleware = FilesystemMiddleware(backend=backend, human_message_token_limit_before_evict=None)
        messages = [HumanMessage(content=[_base64_block()]), AIMessage(content="ok"), HumanMessage(content="next")]

        assert middleware._evict_and_truncate_messages(_request(messages)) is None
        assert backend.write_calls == []

    def test_no_processing_needed_takes_the_fast_path(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        messages = [HumanMessage(content="plain"), AIMessage(content="ok"), HumanMessage(content="next")]

        assert middleware._evict_and_truncate_messages(_request(messages)) is None
        assert backend.write_calls == []

    @pytest.mark.asyncio
    async def test_async_eviction_offloads_historical_media(self) -> None:
        backend = MockBackend()
        middleware = self._middleware(backend)
        historical = HumanMessage(content=[{"type": "text", "text": "老图"}, _base64_block(_BIG_PAYLOAD)])
        messages = [historical, AIMessage(content="ok"), HumanMessage(content="next")]

        result = await middleware._aevict_and_truncate_messages(_request(messages))

        assert result is not None
        processed, command = result
        assert command is not None
        assert processed[0].additional_kwargs.get(_EVICTED_MEDIA_KEY) is not None
        assert backend.write_calls, "the async path must upload too"
