"""Middleware for filtering requests against the active model profile."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from deepagents.backends.utils import _EXTENSION_TO_FILE_TYPE

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ResponseT
    from langchain.chat_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langchain_core.messages.content import ContentBlock

_MULTIMODAL_BLOCK_TYPES: Final = frozenset(_EXTENSION_TO_FILE_TYPE.values())
_PDF_MIME_TYPE: Final = "application/pdf"
_PROFILE_FIELD_BY_BLOCK_TYPE: Final = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "file": "pdf_inputs",
}
_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE: Final = {
    "image": "image_tool_message",
    "file": "pdf_tool_message",
}

try:
    from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI, ChatOpenAI as _ChatOpenAI
except ImportError:
    _OPENAI_FILE_MODEL_TYPES: tuple[type[Any], ...] = ()
else:
    _OPENAI_FILE_MODEL_TYPES = (_AzureChatOpenAI, _ChatOpenAI)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
except ImportError:
    _GOOGLE_FILE_MODEL_TYPES: tuple[type[Any], ...] = ()
else:
    _GOOGLE_FILE_MODEL_TYPES = (_ChatGoogleGenerativeAI,)


def _model_tolerates_non_pdf_files(model: BaseChatModel | None) -> bool:
    """Return whether the provider class accepts non-PDF `file` blocks."""
    return isinstance(model, _OPENAI_FILE_MODEL_TYPES + _GOOGLE_FILE_MODEL_TYPES)


def _multimodal_block_supported(
    block: ContentBlock,
    *,
    profile: Mapping[str, Any],
    tolerates_non_pdf_files: bool,
    in_tool_message: bool,
) -> bool:
    """Return whether the active model accepts a multimodal content block."""
    block_type = block["type"]
    if block_type == "file" and "base64" not in block:
        return True
    if block_type == "file" and block.get("mime_type") != _PDF_MIME_TYPE:
        return tolerates_non_pdf_files

    field = _PROFILE_FIELD_BY_BLOCK_TYPE.get(block_type)
    if field is None:
        return True
    if in_tool_message:
        tool_field = _TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE.get(block_type)
        if tool_field and profile.get(tool_field) is False:
            return False
    return profile.get(field) is not False


def _unsupported_multimodal_placeholder(block: ContentBlock, message: AnyMessage) -> ContentBlock:
    """Build the text block replacing media the active model cannot accept."""
    mime_type = block.get("mime_type", "unknown")
    path = message.additional_kwargs.get("read_file_path", "the requested file")
    return cast(
        "ContentBlock",
        {
            "type": "text",
            "text": f"[read_file: {path} was not attached because this model does not support {block['type']} content ({mime_type}).]",
        },
    )


def _scrub_message_multimodal_content(
    message: AnyMessage,
    *,
    profile: Mapping[str, Any],
    tolerates_non_pdf_files: bool,
) -> AnyMessage:
    """Replace unsupported blocks in one model-bound message."""
    if not isinstance(message, (ToolMessage, HumanMessage)):
        return message

    in_tool_message = isinstance(message, ToolMessage)
    blocks = message.content_blocks
    new_blocks = [
        block
        if block["type"] not in _MULTIMODAL_BLOCK_TYPES
        or _multimodal_block_supported(
            block,
            profile=profile,
            tolerates_non_pdf_files=tolerates_non_pdf_files,
            in_tool_message=in_tool_message,
        )
        else _unsupported_multimodal_placeholder(block, message)
        for block in blocks
    ]
    if new_blocks == blocks:
        return message
    return message.model_copy(update={"content": new_blocks})


def _scrub_unsupported_multimodal_content(
    messages: list[AnyMessage],
    model: BaseChatModel | None,
) -> list[AnyMessage]:
    """Replace content blocks the active model profile marks unsupported."""
    profile = model.profile if model is not None else None
    if not isinstance(profile, dict):
        profile = {}
    tolerates_non_pdf_files = _model_tolerates_non_pdf_files(model)
    return [
        _scrub_message_multimodal_content(
            message,
            profile=profile,
            tolerates_non_pdf_files=tolerates_non_pdf_files,
        )
        for message in messages
    ]


class _ModelProfileMiddleware(AgentMiddleware[Any, Any, Any]):
    """Filter model-bound media after upstream request overrides are applied."""

    @staticmethod
    def _scrub(request: ModelRequest[Any]) -> ModelRequest[Any]:
        messages = _scrub_unsupported_multimodal_content(list(request.messages), request.model)
        return request.override(messages=messages) if messages != list(request.messages) else request

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Filter media against the active model before invoking it."""
        return handler(self._scrub(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Filter media against the active model before invoking it asynchronously."""
        return await handler(self._scrub(request))
