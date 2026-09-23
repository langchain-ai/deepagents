"""Middleware dropping input content the active model can't accept."""

from typing import TYPE_CHECKING, Any, Final, cast

from langchain_core.messages import AnyMessage
from langchain_core.messages.content import ContentBlock

from deepagents.backends.utils import _OPENAI_FILE_MIME_TYPES
from deepagents.middleware._unsupported_content import _UnsupportedContentMiddleware

try:
    from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI, ChatOpenAI as _ChatOpenAI
except ImportError:
    _OPENAI_FILE_MODEL_TYPES: tuple[type[Any], ...] = ()
else:
    _OPENAI_FILE_MODEL_TYPES = (_AzureChatOpenAI, _ChatOpenAI)

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel

_PDF_MIME_TYPE: Final = "application/pdf"


class UnsupportedContentMiddleware(_UnsupportedContentMiddleware):
    """Scrub multimodal blocks the active model rejects, naming the `read_file` path.

    Extends the base middleware with the two things specific to a deep agent:

    - Non-PDF base64 `file` blocks pass only when their MIME type is supported by
      an OpenAI Responses model, since no `ModelProfile` field describes them yet.
    - The placeholder names the file `read_file` was asked for, so the model can tell
      which attachment went missing.
    """

    def _is_supported(self, block: ContentBlock, *, model: "BaseChatModel", in_tool_message: bool) -> bool:
        """Gate non-PDF `file` blocks on MIME type and the OpenAI endpoint."""
        if block["type"] != "file" or "base64" not in block or block.get("mime_type") == _PDF_MIME_TYPE:
            return super()._is_supported(block, model=model, in_tool_message=in_tool_message)
        return block.get("mime_type") in _OPENAI_FILE_MIME_TYPES and isinstance(model, _OPENAI_FILE_MODEL_TYPES) and bool(model.use_responses_api)

    def _replace(self, block: ContentBlock, message: AnyMessage) -> ContentBlock:
        """Name the `read_file` path in the placeholder the model sees."""
        mime_type = block.get("mime_type", "unknown")
        path = message.additional_kwargs.get("read_file_path", "the requested file")
        return cast(
            "ContentBlock",
            {
                "type": "text",
                "text": f"[read_file: {path} was not attached because this model does not support {block['type']} content ({mime_type}).]",
            },
        )
