"""End-to-end tests for scrubbing multimodal content blocks unsupported by `model.profile`.

Tests exercise `FilesystemMiddleware.wrap_model_call`/`awrap_model_call` directly instead
of the private scrubbing helpers, so a refactor of those internals doesn't require
rewriting this file, only what the model actually receives is asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.filesystem import FilesystemMiddleware

if TYPE_CHECKING:
    from langchain_core.messages.content import ContentBlock


class FakeChatModel:
    """Minimal `BaseChatModel` stand-in exposing the two attributes the scrub reads.

    `llm_type` mirrors real values (`ChatOpenAI` -> `"openai-chat"`, `ChatAnthropic` ->
    `"anthropic-chat"`, `ChatGoogleGenerativeAI` -> `"chat-google-generative-ai"`) so tests
    reflect actual provider behavior rather than an arbitrary label.
    """

    def __init__(self, *, profile: dict[str, Any] | None = None, llm_type: str = "fake-chat") -> None:
        self.profile = profile
        self._llm_type = llm_type


def _docx_block() -> ContentBlock:
    return {
        "type": "file",
        "base64": "ZmFrZQ==",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }


def _pdf_block() -> ContentBlock:
    return {"type": "file", "base64": "ZmFrZQ==", "mime_type": "application/pdf"}


def _image_block() -> ContentBlock:
    return {"type": "image", "base64": "ZmFrZQ==", "mime_type": "image/png"}


def _file_id_block() -> ContentBlock:
    return {"type": "file", "file_id": "file_abc123"}


def _url_pdf_block() -> ContentBlock:
    return {"type": "file", "url": "https://example.com/report.pdf"}


def _read_file_tool_message(block: ContentBlock, *, path: str = "/attachment") -> ToolMessage:
    return ToolMessage(
        content_blocks=[block],
        name="read_file",
        tool_call_id="call_1",
        additional_kwargs={"read_file_path": path},
    )


def _wrap_model_call_request(model: FakeChatModel | None, *messages: Any) -> ModelRequest:
    ai_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "read_file", "args": {"file_path": "/attachment"}}],
    )
    return ModelRequest(model=model, messages=[HumanMessage("read it"), ai_msg, *messages], tools=[])


def _run(model: FakeChatModel | None, tool_message: ToolMessage) -> list[ContentBlock]:
    """Send `tool_message` through `wrap_model_call` and return what the handler received."""
    mw = FilesystemMiddleware()
    request = _wrap_model_call_request(model, tool_message)
    captured: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    mw.wrap_model_call(request, handler)
    return next(m for m in captured[0].messages if isinstance(m, ToolMessage)).content_blocks


async def _arun(model: FakeChatModel | None, tool_message: ToolMessage) -> list[ContentBlock]:
    """Async counterpart of `_run`, through `awrap_model_call`."""
    mw = FilesystemMiddleware()
    request = _wrap_model_call_request(model, tool_message)
    captured: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    await mw.awrap_model_call(request, handler)
    return next(m for m in captured[0].messages if isinstance(m, ToolMessage)).content_blocks


def _is_placeholder(blocks: list[ContentBlock], *, path: str = "/attachment") -> bool:
    return len(blocks) == 1 and blocks[0]["type"] == "text" and path in blocks[0]["text"]


class TestNoModelOrProfile:
    def test_pdf_passes_through_when_model_is_none(self) -> None:
        assert _run(None, _read_file_tool_message(_pdf_block())) == [_pdf_block()]

    def test_pdf_passes_through_when_profile_is_none(self) -> None:
        model = FakeChatModel(profile=None, llm_type="anthropic-chat")
        assert _run(model, _read_file_tool_message(_pdf_block())) == [_pdf_block()]


class TestProfileGatedBlocks:
    def test_pdf_stripped_when_profile_disallows(self) -> None:
        model = FakeChatModel(profile={"pdf_inputs": False})
        assert _is_placeholder(_run(model, _read_file_tool_message(_pdf_block())))

    def test_image_stripped_when_profile_disallows(self) -> None:
        model = FakeChatModel(profile={"image_inputs": False})
        assert _is_placeholder(_run(model, _read_file_tool_message(_image_block())))

    def test_image_stripped_by_tool_message_specific_field(self) -> None:
        """A model may allow images generally but reject them specifically in a `ToolMessage`."""
        model = FakeChatModel(profile={"image_inputs": True, "image_tool_message": False})
        assert _is_placeholder(_run(model, _read_file_tool_message(_image_block())))

    def test_image_tool_message_field_does_not_gate_human_message_images(self) -> None:
        """`*_tool_message` fields only apply inside a `ToolMessage`.

        Not, for example, the synthetic `HumanMessage` carrying sampled video frames.
        """
        mw = FilesystemMiddleware()
        model = FakeChatModel(profile={"image_tool_message": False})
        media_message = HumanMessage(
            content=[_image_block()],
            additional_kwargs={"read_file_media_result": True, "read_file_path": "/clip.mp4"},
        )
        request = _wrap_model_call_request(model, media_message)
        captured: list[ModelRequest] = []

        def handler(request: ModelRequest) -> ModelResponse:
            captured.append(request)
            return ModelResponse(result=[AIMessage(content="ok")])

        mw.wrap_model_call(request, handler)

        scrubbed_media = next(m for m in captured[0].messages if isinstance(m, HumanMessage) and m.additional_kwargs.get("read_file_media_result"))
        assert scrubbed_media.content_blocks == [_image_block()]


class TestNonPdfFileProviderGate:
    """Non-PDF `file` blocks (`.docx`, ...) have no `ModelProfile` field yet.

    Support is hard-coded per provider — regardless of whether `profile` itself is present.
    """

    def test_docx_stripped_for_anthropic(self) -> None:
        model = FakeChatModel(profile=None, llm_type="anthropic-chat")
        assert _is_placeholder(_run(model, _read_file_tool_message(_docx_block(), path="/report.docx")), path="/report.docx")

    def test_docx_passes_for_openai(self) -> None:
        model = FakeChatModel(profile=None, llm_type="openai-chat")
        assert _run(model, _read_file_tool_message(_docx_block())) == [_docx_block()]

    def test_docx_passes_for_gemini(self) -> None:
        model = FakeChatModel(profile=None, llm_type="chat-google-generative-ai")
        assert _run(model, _read_file_tool_message(_docx_block())) == [_docx_block()]

    def test_docx_stripped_when_model_is_none(self) -> None:
        assert _is_placeholder(_run(None, _read_file_tool_message(_docx_block())))


class TestFileReferencesPassThrough:
    """`file_id`/`url` references aren't `read_file`'s base64 attachments.

    They should never be scrubbed, even for a provider that doesn't tolerate non-PDF
    base64 uploads.
    """

    def test_file_id_reference_untouched(self) -> None:
        model = FakeChatModel(profile=None, llm_type="anthropic-chat")
        assert _run(model, _read_file_tool_message(_file_id_block())) == [_file_id_block()]

    def test_url_reference_untouched(self) -> None:
        model = FakeChatModel(profile=None, llm_type="anthropic-chat")
        assert _run(model, _read_file_tool_message(_url_pdf_block())) == [_url_pdf_block()]


class TestAsyncPath:
    async def test_awrap_model_call_strips_docx_for_anthropic(self) -> None:
        model = FakeChatModel(profile=None, llm_type="anthropic-chat")
        blocks = await _arun(model, _read_file_tool_message(_docx_block(), path="/report.docx"))
        assert _is_placeholder(blocks, path="/report.docx")
