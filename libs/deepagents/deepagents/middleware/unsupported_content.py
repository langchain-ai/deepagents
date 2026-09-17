"""Middleware dropping input content the active model can't accept."""

from typing import TYPE_CHECKING, Any, Final, cast

from langchain.agents.middleware.unsupported_content import (
    UnsupportedContentMiddleware as _UnsupportedContentMiddleware,
)
from langchain_core.messages import AnyMessage
from langchain_core.messages.content import ContentBlock

# `ChatOpenAI`, `AzureChatOpenAI`, and `ChatGoogleGenerativeAI` accept non-PDF
# `file` blocks such as `.docx` and `.pptx`. `ModelProfile` only encodes PDF
# support today, so these providers get a hard-coded pass until profiles can
# describe support for other office and document formats.
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

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel

_PDF_MIME_TYPE: Final = "application/pdf"


def _model_tolerates_non_pdf_files(model: "BaseChatModel | None") -> bool:
    """Whether `model` is a provider class known to accept non-PDF `file` blocks."""
    return isinstance(model, _OPENAI_FILE_MODEL_TYPES + _GOOGLE_FILE_MODEL_TYPES)


class UnsupportedContentMiddleware(_UnsupportedContentMiddleware):
    """Scrub multimodal blocks the active model rejects, naming the `read_file` path.

    Extends the base middleware with the two things specific to a deep agent:

    - Non-PDF base64 `file` blocks (what `read_file` emits for a `.docx`) pass only
      for provider classes known to accept them, since no `ModelProfile` field
      describes them yet.
    - The placeholder names the file `read_file` was asked for, so the model can tell
      which attachment went missing.
    """

    def is_supported(self, block: ContentBlock, *, model: "BaseChatModel", in_tool_message: bool) -> bool:
        """Gate non-PDF `file` blocks on the provider class, else defer to the profile."""
        if block["type"] == "file" and "base64" in block and block.get("mime_type") != _PDF_MIME_TYPE:
            return _model_tolerates_non_pdf_files(model)
        return super().is_supported(block, model=model, in_tool_message=in_tool_message)

    def replace(self, block: ContentBlock, message: AnyMessage) -> ContentBlock:
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
