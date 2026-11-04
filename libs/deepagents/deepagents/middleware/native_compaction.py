"""Opt-in OpenAI Responses compaction without flattening opaque provider context."""

from __future__ import annotations

from contextvars import ContextVar
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain.agents.factory import _supports_provider_strategy
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.agents.structured_output import AutoStrategy, OutputToolBinding, ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding
from langgraph.types import Command

from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SUMMARIZATION_SESSION_ID_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
    compute_summarization_defaults,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from deepagents.backends.protocol import BackendProtocol

_IDENTITY_KEY = "deepagents_compaction_model"


def _identity(model: BaseChatModel) -> dict[str, str]:
    identity = {key: str(getattr(model, key, "")) for key in ("model_name", "deployment_name", "openai_api_version")}
    identity["adapter"] = f"{type(model).__module__}.{type(model).__qualname__}"
    identity["endpoint"] = sha256(str(getattr(model, "openai_api_base", "")).encode()).hexdigest()
    return identity


def _native_model(model: BaseChatModel) -> BaseChatModel:
    required = ("compact", "acompact", "count_input_tokens", "acount_input_tokens")
    if not all(callable(getattr(model, method, None)) for method in required):
        msg = "OpenAI compaction requires compact/acompact and count_input_tokens/acount_input_tokens; the installed adapter is unsupported."
        raise ValueError(msg)
    if getattr(model, "use_responses_api", False) is not True:
        msg = "OpenAI compaction requires use_responses_api=True on the chat model."
        raise ValueError(msg)
    if getattr(model, "use_previous_response_id", False):
        msg = "OpenAI compaction requires use_previous_response_id=False for stateless continuation."
        raise ValueError(msg)
    return model


class _NativeAdapter(Protocol):
    def compact(self, messages: list[AnyMessage], **kwargs: Any) -> AIMessage: ...

    async def acompact(self, messages: list[AnyMessage], **kwargs: Any) -> AIMessage: ...

    def count_input_tokens(self, messages: list[AnyMessage], **kwargs: Any) -> int: ...

    async def acount_input_tokens(self, messages: list[AnyMessage], **kwargs: Any) -> int: ...


def _request_binding(request: ModelRequest) -> tuple[BaseChatModel, dict[str, Any]]:
    if request.model_settings.get("model", getattr(request.model, "model_name", None)) != getattr(request.model, "model_name", None):
        msg = "OpenAI compaction cannot override the main model through model_settings."
        raise ValueError(msg)
    tools = list(request.tools)
    settings = {**request.model_settings, "store": False, "previous_response_id": None}
    response_format = request.response_format
    if response_format is not None:
        if not isinstance(response_format, (AutoStrategy, ProviderStrategy, ToolStrategy)):
            response_format = AutoStrategy(schema=response_format)
        if isinstance(response_format, AutoStrategy):
            response_format = (
                ProviderStrategy(schema=response_format.schema)
                if _supports_provider_strategy(request.model, tools=tools)
                else ToolStrategy(schema=response_format.schema)
            )
        if isinstance(response_format, ProviderStrategy):
            settings = {**response_format.to_model_kwargs(), **settings}
        else:
            tools.extend(OutputToolBinding.from_schema_spec(spec).tool for spec in response_format.schema_specs)
    tool_choice = "any" if isinstance(response_format, ToolStrategy) else request.tool_choice
    bound = request.model.bind_tools(tools, tool_choice=tool_choice, **settings) if tools or response_format else request.model.bind(**settings)
    if not isinstance(bound, RunnableBinding):
        msg = "OpenAI compaction requires a chat adapter returning RunnableBinding from bind_tools."
        raise TypeError(msg)
    return _native_model(request.model), dict(bound.kwargs)


class OpenAICompactionMiddleware(SummarizationMiddleware):
    """Compact complete conversation prefixes with a compatible Responses adapter."""

    compaction_strategy = "openai"

    def __init__(self, model: BaseChatModel, *, backend: BackendProtocol) -> None:
        """Configure native compaction using the main model's context threshold."""
        _native_model(model)
        defaults = compute_summarization_defaults(model)
        super().__init__(model=model, backend=backend, trigger=defaults["trigger"], keep=("messages", 1), trim_tokens_to_summarize=None)
        self._request: ContextVar[ModelRequest | None] = ContextVar("native_compaction_request", default=None)

    def _validate_context(self, messages: list[AnyMessage], model: BaseChatModel) -> None:
        for message in messages:
            identity = message.response_metadata.get(_IDENTITY_KEY)
            if identity is not None and identity != _identity(model):
                msg = "Cannot switch models or endpoints while OpenAI compacted context is present. Start a new conversation instead."
                raise ValueError(msg)

    def _get_effective_messages(self, request: ModelRequest) -> list[AnyMessage]:
        messages = super()._get_effective_messages(request)
        self._validate_context(messages, request.model)
        return messages

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        pending: set[str] = set()
        cutoff = 0
        for index, message in enumerate(messages):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    call_id = call.get("id")
                    if not call_id or call_id in pending:
                        msg = "Cannot compact missing or duplicate tool call IDs."
                        raise ValueError(msg)
                    pending.add(call_id)
            elif isinstance(message, ToolMessage):
                if message.tool_call_id not in pending:
                    msg = "Cannot compact an incomplete or unmatched tool exchange."
                    raise ValueError(msg)
                pending.remove(message.tool_call_id)
            if not pending:
                cutoff = index + 1
        return cutoff if not pending or self._only_compact_pending(messages[-1], pending) else 0

    @staticmethod
    def _only_compact_pending(last: AnyMessage, pending: set[str]) -> bool:
        if not isinstance(last, AIMessage) or len(last.tool_calls) != 1:
            return False
        call = last.tool_calls[0]
        return call["name"] == "compact_conversation" and pending == {call["id"]}

    def _count_tokens(
        self, messages: list[AnyMessage], system_message: SystemMessage | None = None, tools: list[BaseTool | dict[str, Any]] | None = None
    ) -> int:
        request = self._request.get()
        if request is not None:
            model, kwargs = _request_binding(request)
        else:
            model, kwargs = self.model, {"tools": tools or [], "store": False}
        inputs = [system_message, *messages] if system_message else messages
        return cast("_NativeAdapter", model).count_input_tokens(inputs, **kwargs)

    def count_request_tokens(self, request: ModelRequest) -> int:
        """Count the exact generation payload, including instructions and schemas."""
        token = self._request.set(request)
        try:
            return self._count_tokens(self._get_effective_messages(request), request.system_message, request.tools)
        finally:
            self._request.reset(token)

    async def acount_request_tokens(self, request: ModelRequest) -> int:
        """Count a generation payload without blocking the event loop."""
        model, kwargs = _request_binding(request)
        messages = self._get_effective_messages(request)
        inputs = [request.system_message, *messages] if request.system_message else messages
        return await cast("_NativeAdapter", model).acount_input_tokens(inputs, **kwargs)

    def _compact_input(self, messages: list[AnyMessage]) -> tuple[BaseChatModel, list[AnyMessage], dict[str, Any]]:
        request = self._request.get()
        model, kwargs = _request_binding(request) if request else (self.model, {"store": False})
        self._validate_context(messages, model)
        if self._determine_cutoff_index(messages) != len(messages):
            msg = "Cannot compact before all tool results are complete."
            raise ValueError(msg)
        inputs = [request.system_message, *messages] if request and request.system_message else messages
        return model, inputs, kwargs

    @staticmethod
    def _encode(message: AIMessage, model: BaseChatModel) -> str:
        if not isinstance(message, AIMessage) or message.tool_calls or message.invalid_tool_calls:
            msg = "OpenAI compaction must return opaque AIMessage context without executable tool calls."
            raise ValueError(msg)
        if not any(isinstance(block, dict) and block.get("type") == "compaction" and block.get("encrypted_content") for block in message.content):
            msg = "OpenAI compaction returned no opaque compaction context."
            raise ValueError(msg)
        message = message.model_copy(update={"response_metadata": {**message.response_metadata, _IDENTITY_KEY: _identity(model)}})
        return message.model_dump_json()

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        model, inputs, kwargs = self._compact_input(messages_to_summarize)
        return self._encode(cast("_NativeAdapter", model).compact(inputs, **kwargs), model)

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        model, inputs, kwargs = self._compact_input(messages_to_summarize)
        return self._encode(await cast("_NativeAdapter", model).acompact(inputs, **kwargs), model)

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        del file_path
        return [AIMessage.model_validate_json(summary)]

    def _truncate_args(self, messages: list[AnyMessage], total_tokens: int) -> tuple[list[AnyMessage], bool]:
        del total_tokens
        return messages, False

    def _event(self, request: ModelRequest, summary: str, cutoff: int, file_path: str | None) -> SummarizationEvent:
        return {
            "cutoff_index": self._compute_state_cutoff(request.state.get(SUMMARIZATION_EVENT_KEY), cutoff),
            "summary_message": self._build_new_messages_with_path(summary, file_path)[0],
            "file_path": file_path,
            "created_at_index": len(request.messages),
        }

    def _continuation(self, request: ModelRequest, messages: list[AnyMessage]) -> ModelRequest:
        _native_model(request.model)
        self._validate_context(messages, request.model)
        return request.override(messages=messages, model_settings={**request.model_settings, "store": False, "previous_response_id": None})

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse | ExtendedModelResponse:
        """Apply provider context and atomically checkpoint successful compaction."""
        token = self._request.set(request)
        try:
            messages = self._get_effective_messages(request)
            cutoff = self._determine_cutoff_index(messages)
            if not cutoff or cutoff != len(messages):
                return handler(self._continuation(request, messages))
            count = self._count_tokens(messages, request.system_message, request.tools)
            if not (self._should_summarize(messages, count) or self._over_budget(request, count)):
                return handler(self._continuation(request, messages))
            summary = self._create_summary(messages[:cutoff])
            session = self._get_session_id(request.state)
            path = self._offload_to_backend(self._backend, messages[:cutoff], session)
            event = self._event(request, summary, cutoff, path)
            response = handler(self._continuation(request, [event["summary_message"], *messages[cutoff:]]))
            return ExtendedModelResponse(
                model_response=response, command=Command(update={SUMMARIZATION_EVENT_KEY: event, SUMMARIZATION_SESSION_ID_KEY: session})
            )
        finally:
            self._request.reset(token)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply native compaction using asynchronous provider endpoints."""
        token = self._request.set(request)
        try:
            messages = self._get_effective_messages(request)
            cutoff = self._determine_cutoff_index(messages)
            if not cutoff or cutoff != len(messages):
                return await handler(self._continuation(request, messages))
            count = await self.acount_request_tokens(request)
            if not (self._should_summarize(messages, count) or self._over_budget(request, count)):
                return await handler(self._continuation(request, messages))
            summary = await self._acreate_summary(messages[:cutoff])
            session = self._get_session_id(request.state)
            path = await self._aoffload_to_backend(self._backend, messages[:cutoff], session)
            event = self._event(request, summary, cutoff, path)
            response = await handler(self._continuation(request, [event["summary_message"], *messages[cutoff:]]))
            return ExtendedModelResponse(
                model_response=response, command=Command(update={SUMMARIZATION_EVENT_KEY: event, SUMMARIZATION_SESSION_ID_KEY: session})
            )
        finally:
            self._request.reset(token)
