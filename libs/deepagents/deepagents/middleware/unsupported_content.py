"""Replace input content blocks the active model can't accept."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import HumanMessage, ToolMessage

from deepagents.backends.utils import _OPENAI_FILE_MIME_TYPES

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage, ContentBlock

try:
    from langchain_openai import AzureChatOpenAI as _AzureChatOpenAI, ChatOpenAI as _ChatOpenAI
except ImportError:
    _OPENAI_FILE_MODEL_TYPES: tuple[type[Any], ...] = ()
else:
    _OPENAI_FILE_MODEL_TYPES = (_AzureChatOpenAI, _ChatOpenAI)

_PROFILE_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_inputs",
    "audio": "audio_inputs",
    "video": "video_inputs",
    "file": "pdf_inputs",
}
"""`ModelProfile` field gating each block type outside of a `ToolMessage`."""

_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_tool_message",
    "file": "pdf_tool_message",
}
"""Additional `ModelProfile` field gating a block type inside a `ToolMessage`."""

_PDF_MIME_TYPE: Final = "application/pdf"


def _profile_accepts(block: ContentBlock, profile: Mapping[str, Any], *, in_tool_message: bool) -> bool:
    """Return whether `profile` accepts `block`.

    Missing `ModelProfile` fields read as supported, since profile coverage is
    incomplete; only an explicit `False` rejects a block.

    Args:
        block: The input content block under consideration.
        profile: The active model's profile, or an empty mapping if it has none.
        in_tool_message: Whether `block` sits in a `ToolMessage`, which some
            providers gate separately from ordinary input.

    Returns:
        `True` unless a profile field explicitly rejects the block.
    """
    block_type = block["type"]
    field = _PROFILE_FIELD_BY_BLOCK_TYPE.get(block_type)
    if field is None:
        return True
    if block_type == "file" and ("base64" not in block or block.get("mime_type") != _PDF_MIME_TYPE):
        # URL- and file-ID-backed references are provider-managed, and no profile
        # field describes non-PDF payloads (`.docx`, `.pptx`, ...).
        return True
    if in_tool_message:
        tool_field = _TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE.get(block_type)
        if tool_field is not None and profile.get(tool_field) is False:
            return False
    return profile.get(field) is not False


# deepagents-specific: behavior `ModelProfile` can't express yet. Outside this
# section and the `_OPENAI_FILE_MIME_TYPES` / `langchain_openai` imports it uses,
# the module has no deepagents dependencies.


def _is_inline_document(block: ContentBlock) -> bool:
    """Return whether `block` is a non-PDF base64 `file`, which no profile field describes."""
    return block["type"] == "file" and "base64" in block and block.get("mime_type") != _PDF_MIME_TYPE


def _openai_responses_accepts(block: ContentBlock, model: BaseChatModel) -> bool:
    """Return whether `model` is an OpenAI Responses model accepting `block`'s MIME type."""
    return block.get("mime_type") in _OPENAI_FILE_MIME_TYPES and isinstance(model, _OPENAI_FILE_MODEL_TYPES) and bool(model.use_responses_api)


def _read_file_placeholder(block: ContentBlock, message: AnyMessage) -> ContentBlock:
    """Build a placeholder naming the `read_file` path, so the model knows what's missing."""
    mime_type = block.get("mime_type", "unknown")
    path = message.additional_kwargs.get("read_file_path", "the requested file")
    return cast(
        "ContentBlock",
        {
            "type": "text",
            "text": f"[read_file: {path} was not attached because this model does not support {block['type']} content ({mime_type}).]",
        },
    )


class UnsupportedContentMiddleware(AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]):
    """Replace multimodal input blocks the active model can't accept with a text notice.

    Without it, a request carrying content the model can't accept (e.g. an image sent
    to a text-only model) fails, and since that content stays in the thread, every
    later request fails too. This middleware replaces such blocks with a text notice
    on every model request. The thread itself keeps the original content, so switching
    to a model that accepts it sends it again.

    Support is read from
    [`model.profile`](https://docs.langchain.com/oss/python/langchain/models#model-profiles).

    Place this middleware last in the `middleware` list, so that if
    `ModelRequest.model` changes, this middleware will apply to the correct one.

    [`create_deep_agent`][deepagents.graph.create_deep_agent] adds it automatically.

    Example:
        ```python
        from deepagents.middleware import FilesystemMiddleware, UnsupportedContentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(model, middleware=[FilesystemMiddleware(), UnsupportedContentMiddleware()])
        ```
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def _is_supported(self, block: ContentBlock, *, model: BaseChatModel, in_tool_message: bool) -> bool:
        """Return whether `model` accepts `block`.

        A model with no profile is read as an empty profile rather than skipped, so
        the checks that don't depend on profile data still run.
        """
        if _is_inline_document(block):
            return _openai_responses_accepts(block, model)
        return _profile_accepts(block, model.profile or {}, in_tool_message=in_tool_message)

    def _replace(self, block: ContentBlock, message: AnyMessage) -> ContentBlock:
        """Build the text block replacing a `block` the active model can't accept."""
        return _read_file_placeholder(block, message)

    def _filter_message(self, message: AnyMessage, *, model: BaseChatModel) -> AnyMessage:
        """Return `message`, or a copy with unsupported blocks replaced."""
        in_tool_message = isinstance(message, ToolMessage)
        blocks = message.content_blocks
        new_blocks = [
            block if self._is_supported(block, model=model, in_tool_message=in_tool_message) else self._replace(block, message) for block in blocks
        ]
        if new_blocks == blocks:
            return message
        return message.model_copy(update={"content": new_blocks})

    def _filter_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Return `request`, or an override whose messages the active model accepts."""
        model = request.model
        messages: list[AnyMessage] = []
        changed = False
        for message in request.messages:
            if not isinstance(message, (HumanMessage, ToolMessage)):
                messages.append(message)
                continue
            filtered = self._filter_message(message, model=model)
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
            request: Model request to execute.
            handler: Callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return handler(self._filter_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Filter unsupported input blocks, then invoke the model.

        Args:
            request: Model request to execute.
            handler: Async callback that executes the model request.

        Returns:
            The result of invoking the handler.
        """
        return await handler(self._filter_request(request))
