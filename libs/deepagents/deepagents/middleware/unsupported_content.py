"""Middleware dropping input content the active model can't accept."""

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock

from deepagents.backends.utils import _EXTENSION_TO_FILE_TYPE

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

_MULTIMODAL_BLOCK_TYPES: Final = frozenset(_EXTENSION_TO_FILE_TYPE.values())
"""Content block types `read_file` may emit that require multimodal model support.

Derived from `_EXTENSION_TO_FILE_TYPE`'s values (`"text"` never appears there,
since it's `_get_file_type`'s default for unmapped extensions).
"""

_PDF_MIME_TYPE: Final = "application/pdf"

_PROFILE_FIELD_BY_BLOCK_TYPE: Final = {"image": "image_inputs", "audio": "audio_inputs", "video": "video_inputs", "file": "pdf_inputs"}
"""`ModelProfile` field gating each block type. `file` only applies to PDF `mime_type`; other
file types have no field yet and are handled separately via provider class checks."""

_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE: Final = {"image": "image_tool_message", "file": "pdf_tool_message"}
"""Extra `ModelProfile` field that can gate a block type specifically within a `ToolMessage`."""


def _model_tolerates_non_pdf_files(model: "BaseChatModel | None") -> bool:
    """Whether `model` is a provider class known to accept non-PDF `file` blocks."""
    return isinstance(model, _OPENAI_FILE_MODEL_TYPES + _GOOGLE_FILE_MODEL_TYPES)


OnUnsupported = Callable[[ContentBlock, AnyMessage], "ContentBlock | str | None"]
"""Builds the stand-in for an unsupported block. Return `None` to drop it outright."""


class UnsupportedContentMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Replace multimodal input blocks the active model can't accept with a text notice.

    Some providers return a non-retryable 400 when sent a content block they don't
    support (e.g. a `file` block whose `mime_type` isn't `application/pdf`, produced
    when `read_file` reads a `.docx`), which would otherwise end the thread. Swapping
    the unsupported block for a text placeholder keeps the turn alive and tells the
    model the attachment was dropped.

    `create_deep_agent` installs this last, making it the innermost `wrap_model_call`
    layer and so the only one guaranteed to see the model an outer middleware selected
    at runtime. Pass a middleware with the same `name` to `create_deep_agent` to replace
    it, or subclass it to change which blocks are gated (`is_supported`) or what the
    model sees in their place (`replace`).
    """

    def __init__(self, *, on_unsupported: OnUnsupported | None = None) -> None:
        """Initialize `UnsupportedContentMiddleware`.

        Args:
            on_unsupported: Builds the replacement for an unsupported block, receiving
                the block and the message carrying it. Return a content block or a
                string to substitute it, or `None` to drop the block without a trace.
                Defaults to a notice naming the `read_file` path and block type.
        """
        super().__init__()
        self.on_unsupported = on_unsupported

    def is_supported(self, block: ContentBlock, *, model: "BaseChatModel | None", profile: Mapping[str, Any], in_tool_message: bool) -> bool:
        """Check whether `profile` (plus the hard-coded provider exception) accepts `block`.

        Missing `ModelProfile` fields default to supported, since profile coverage is
        incomplete. Only an explicit `False` rejects a block type.

        Args:
            block: The input content block under consideration.
            model: The model the request will reach. Consulted where no profile field
                describes the block yet.
            profile: `model.profile`, or an empty mapping when the model has none.
            in_tool_message: Whether `block` sits in a `ToolMessage`, which some
                providers gate separately from ordinary input.

        Returns:
            `True` unless the block type is explicitly rejected.
        """
        block_type = block["type"]
        if block_type not in _MULTIMODAL_BLOCK_TYPES:
            return True
        if block_type == "file" and "base64" not in block:
            # URL-/file-ID-backed file references are provider-managed and often don't
            # include a `mime_type`, so leave them untouched.
            return True
        if block_type == "file" and block.get("mime_type") != _PDF_MIME_TYPE:
            # Non-PDF base64 `file` blocks (`.docx`, `.pptx`, ...) aren't described
            # by any `ModelProfile` field yet; only the hard-coded tolerant
            # providers pass.
            return _model_tolerates_non_pdf_files(model)

        field = _PROFILE_FIELD_BY_BLOCK_TYPE.get(block_type)
        if field is None:
            return True
        if in_tool_message:
            tool_field = _TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE.get(block_type)
            if tool_field and profile.get(tool_field) is False:
                return False
        return profile.get(field) is not False

    def replace(self, block: ContentBlock, message: AnyMessage) -> ContentBlock | None:
        """Build the stand-in for an unsupported `block`.

        Args:
            block: The block the active model rejects.
            message: The message carrying `block`.

        Returns:
            A replacement content block, or `None` to drop the block.
        """
        if self.on_unsupported is not None:
            replacement = self.on_unsupported(block, message)
            if not isinstance(replacement, str):
                return replacement
        else:
            mime_type = block.get("mime_type", "unknown")
            path = message.additional_kwargs.get("read_file_path", "the requested file")
            replacement = f"[read_file: {path} was not attached because this model does not support {block['type']} content ({mime_type}).]"
        return cast("ContentBlock", {"type": "text", "text": replacement})

    def _filter_message(self, message: AnyMessage, *, model: "BaseChatModel | None", profile: Mapping[str, Any]) -> AnyMessage:
        """Return `message` unchanged, or a copy with unsupported blocks replaced."""
        in_tool_message = isinstance(message, ToolMessage)
        new_blocks: list[ContentBlock] = []
        changed = False
        for block in message.content_blocks:
            if self.is_supported(block, model=model, profile=profile, in_tool_message=in_tool_message):
                new_blocks.append(block)
                continue
            changed = True
            if (replacement := self.replace(block, message)) is not None:
                new_blocks.append(replacement)
        if not changed:
            return message
        return message.model_copy(update={"content": new_blocks})

    def _filter_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return `request`, or an override whose messages the active model accepts.

        A `model` with no `profile` (including `None` `model`, e.g. in tests) is
        treated as an empty profile rather than skipped: `ModelProfile` is often
        absent for models `langchain_anthropic` doesn't have a static entry for
        (e.g. `ChatAnthropic(model="claude-3-5-sonnet-latest")`), and the
        provider-based non-PDF `file` gate doesn't depend on profile data at all —
        skipping the whole scrub in that case would silently leave the exact
        `.docx`-on-Anthropic bug this fixes unfixed for those models. An empty
        profile still defaults every per-field check to "supported."
        """
        model = request.model
        profile = model.profile if model is not None else None
        if not isinstance(profile, dict):
            profile = {}
        messages: list[AnyMessage] = []
        changed = False
        for message in request.messages:
            # String content can only hold text, so it never needs filtering. Skipping
            # it also avoids parsing `content_blocks` for the bulk of a long history.
            if isinstance(message.content, str) or not isinstance(message, (ToolMessage, HumanMessage)):
                messages.append(message)
                continue
            filtered = self._filter_message(message, model=model, profile=profile)
            changed = changed or filtered is not message
            messages.append(filtered)
        return request.override(messages=messages) if changed else request

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Filter unsupported input blocks, then invoke the model.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Filter unsupported input blocks, then invoke the model.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        return await handler(self._filter_request(request))
