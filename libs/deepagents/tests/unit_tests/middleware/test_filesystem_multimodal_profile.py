"""Tests for scrubbing multimodal content blocks unsupported by `model.profile`.

Covers the fix for a `read_file` attachment (e.g. a non-PDF `file` block from a
`.docx`) reaching a model that rejects it and ending the thread with a
non-retryable provider error.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.filesystem import (
    FilesystemMiddleware,
    _infer_model_provider,
    _multimodal_block_supported,
    _scrub_unsupported_multimodal_content,
)


class FakeChatModel:
    """Minimal stand-in for a `BaseChatModel` exposing `profile` and `_get_ls_params`."""

    def __init__(self, profile: dict[str, Any] | None, ls_provider: str | None = None) -> None:
        self.profile = profile
        self._ls_provider = ls_provider

    def _get_ls_params(self) -> dict[str, Any]:
        if self._ls_provider is None:
            return {}
        return {"ls_provider": self._ls_provider}


class NamedOnlyModel:
    """Model with no `_get_ls_params`, so provider must come from the class name."""

    def __init__(self, profile: dict[str, Any] | None) -> None:
        self.profile = profile


class ChatAnthropic(NamedOnlyModel):
    """Stand-in with the real Anthropic chat model's class name."""


def _docx_block() -> dict[str, str]:
    return {
        "type": "file",
        "base64": "ZmFrZQ==",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }


def _pdf_block() -> dict[str, str]:
    return {"type": "file", "base64": "ZmFrZQ==", "mime_type": "application/pdf"}


def _image_block() -> dict[str, str]:
    return {"type": "image", "base64": "ZmFrZQ==", "mime_type": "image/png"}


class TestInferModelProvider:
    def test_prefers_ls_provider_over_class_name(self) -> None:
        model = FakeChatModel(profile={}, ls_provider="google-genai")
        assert _infer_model_provider(model) == "google_genai"

    def test_falls_back_to_class_name(self) -> None:
        assert _infer_model_provider(ChatAnthropic(profile={})) == "anthropic"

    def test_returns_none_for_unrecognized_model(self) -> None:
        assert _infer_model_provider(NamedOnlyModel(profile={})) is None

    def test_tolerates_get_ls_params_raising(self) -> None:
        class Explodes(NamedOnlyModel):
            def _get_ls_params(self) -> dict[str, Any]:
                msg = "no params"
                raise RuntimeError(msg)

        assert _infer_model_provider(Explodes(profile={})) is None


class TestMultimodalBlockSupported:
    def test_image_blocked_when_profile_says_no(self) -> None:
        assert not _multimodal_block_supported(_image_block(), profile={"image_inputs": False}, provider=None, in_tool_message=True)

    def test_image_allowed_when_field_absent(self) -> None:
        assert _multimodal_block_supported(_image_block(), profile={}, provider=None, in_tool_message=True)

    def test_image_blocked_by_tool_message_specific_field(self) -> None:
        assert not _multimodal_block_supported(
            _image_block(),
            profile={"image_inputs": True, "image_tool_message": False},
            provider=None,
            in_tool_message=True,
        )

    def test_image_tool_message_field_ignored_outside_tool_message(self) -> None:
        assert _multimodal_block_supported(
            _image_block(),
            profile={"image_tool_message": False},
            provider=None,
            in_tool_message=False,
        )

    def test_pdf_gated_by_pdf_inputs(self) -> None:
        assert not _multimodal_block_supported(_pdf_block(), profile={"pdf_inputs": False}, provider=None, in_tool_message=True)
        assert _multimodal_block_supported(_pdf_block(), profile={}, provider=None, in_tool_message=True)

    def test_non_pdf_file_requires_tolerant_provider(self) -> None:
        assert not _multimodal_block_supported(_docx_block(), profile={}, provider=None, in_tool_message=True)
        assert not _multimodal_block_supported(_docx_block(), profile={}, provider="anthropic", in_tool_message=True)
        assert _multimodal_block_supported(_docx_block(), profile={}, provider="openai", in_tool_message=True)
        assert _multimodal_block_supported(_docx_block(), profile={}, provider="google_genai", in_tool_message=True)


class TestScrubUnsupportedMultimodalContent:
    def test_noop_when_model_is_none(self) -> None:
        messages = [ToolMessage(content_blocks=[_pdf_block()], name="read_file", tool_call_id="call_1")]
        assert _scrub_unsupported_multimodal_content(messages, None) == messages

    def test_noop_when_model_has_no_profile(self) -> None:
        messages = [ToolMessage(content_blocks=[_pdf_block()], name="read_file", tool_call_id="call_1")]
        model = FakeChatModel(profile=None)
        assert _scrub_unsupported_multimodal_content(messages, model) == messages

    def test_strips_pdf_block_when_profile_disallows(self) -> None:
        message = ToolMessage(
            content_blocks=[_pdf_block()],
            name="read_file",
            tool_call_id="call_1",
            additional_kwargs={"read_file_path": "/doc.pdf"},
        )
        model = FakeChatModel(profile={"pdf_inputs": False})

        scrubbed = _scrub_unsupported_multimodal_content([message], model)

        assert len(scrubbed) == 1
        blocks = scrubbed[0].content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "/doc.pdf" in blocks[0]["text"]
        assert "file" in blocks[0]["text"]

    def test_strips_docx_block_for_intolerant_provider(self) -> None:
        message = ToolMessage(
            content_blocks=[_docx_block()],
            name="read_file",
            tool_call_id="call_1",
            additional_kwargs={"read_file_path": "/report.docx"},
        )
        model = ChatAnthropic(profile={})

        scrubbed = _scrub_unsupported_multimodal_content([message], model)

        blocks = scrubbed[0].content_blocks
        assert blocks[0]["type"] == "text"
        assert "/report.docx" in blocks[0]["text"]

    def test_keeps_docx_block_for_tolerant_provider(self) -> None:
        original_block = _docx_block()
        message = ToolMessage(content_blocks=[original_block], name="read_file", tool_call_id="call_1")
        model = FakeChatModel(profile={}, ls_provider="openai")

        scrubbed = _scrub_unsupported_multimodal_content([message], model)

        assert scrubbed == [message]
        assert scrubbed[0].content_blocks[0] == original_block

    def test_keeps_supported_image_block(self) -> None:
        message = ToolMessage(content_blocks=[_image_block()], name="read_file", tool_call_id="call_1")
        model = FakeChatModel(profile={"image_inputs": True})

        scrubbed = _scrub_unsupported_multimodal_content([message], model)

        assert scrubbed == [message]

    def test_leaves_text_only_messages_untouched(self) -> None:
        messages = [HumanMessage("hello"), ToolMessage(content="plain text", tool_call_id="call_1")]
        model = FakeChatModel(profile={"image_inputs": False, "pdf_inputs": False})

        assert _scrub_unsupported_multimodal_content(messages, model) == messages

    def test_scrubs_synthetic_video_media_human_message(self) -> None:
        message = HumanMessage(
            content=[{"type": "text", "text": "frames"}, _image_block()],
            additional_kwargs={"read_file_media_result": True, "read_file_path": "/clip.mp4"},
        )
        model = FakeChatModel(profile={"image_inputs": False})

        scrubbed = _scrub_unsupported_multimodal_content([message], model)

        blocks = scrubbed[0].content_blocks
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "text"
        assert "/clip.mp4" in blocks[1]["text"]


class TestWrapModelCallIntegration:
    def test_wrap_model_call_scrubs_unsupported_pdf_before_handler(self) -> None:
        """A model that rejects `pdf_inputs` should see a placeholder, not the file bytes."""
        mw = FilesystemMiddleware()
        tool_message = ToolMessage(
            content_blocks=[_pdf_block()],
            name="read_file",
            tool_call_id="call_1",
            additional_kwargs={"read_file_path": "/doc.pdf"},
        )
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "read_file", "args": {"file_path": "/doc.pdf"}}],
        )
        model = FakeChatModel(profile={"pdf_inputs": False})
        request = ModelRequest(model=model, messages=[HumanMessage("read the doc"), ai_msg, tool_message], tools=[])

        captured: list[ModelRequest] = []

        def handler(request: ModelRequest) -> ModelResponse:
            captured.append(request)
            return ModelResponse(result=[AIMessage(content="ok")])

        mw.wrap_model_call(request, handler)

        assert len(captured) == 1
        scrubbed_tool_message = next(m for m in captured[0].messages if isinstance(m, ToolMessage))
        blocks = scrubbed_tool_message.content_blocks
        assert blocks[0]["type"] == "text"
        assert "/doc.pdf" in blocks[0]["text"]

    async def test_awrap_model_call_scrubs_unsupported_pdf_before_handler(self) -> None:
        mw = FilesystemMiddleware()
        tool_message = ToolMessage(
            content_blocks=[_pdf_block()],
            name="read_file",
            tool_call_id="call_1",
            additional_kwargs={"read_file_path": "/doc.pdf"},
        )
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "read_file", "args": {"file_path": "/doc.pdf"}}],
        )
        model = FakeChatModel(profile={"pdf_inputs": False})
        request = ModelRequest(model=model, messages=[HumanMessage("read the doc"), ai_msg, tool_message], tools=[])

        captured: list[ModelRequest] = []

        async def handler(request: ModelRequest) -> ModelResponse:
            captured.append(request)
            return ModelResponse(result=[AIMessage(content="ok")])

        await mw.awrap_model_call(request, handler)

        assert len(captured) == 1
        scrubbed_tool_message = next(m for m in captured[0].messages if isinstance(m, ToolMessage))
        assert scrubbed_tool_message.content_blocks[0]["type"] == "text"
