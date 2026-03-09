"""ChatCodexOAuth - LangChain BaseChatModel wrapper for Codex OAuth."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from deepagents_codex.client import CodexClient
from deepagents_codex.response import parse_chat_response, parse_stream_chunk

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)


def _convert_message(msg: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to OpenAI API format."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AIMessage):
        result: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["args"]
                        if isinstance(tc["args"], str)
                        else json.dumps(tc["args"]),
                    },
                }
                for tc in msg.tool_calls
            ]
        return result
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "content": msg.content,
            "tool_call_id": msg.tool_call_id,
        }
    return {"role": "user", "content": str(msg.content)}


class ChatCodexOAuth(BaseChatModel):
    """Chat model that authenticates via Codex OAuth.

    Uses OAuth Bearer tokens from the Codex auth store instead of API keys.
    Supports streaming, tool calling, and automatic token refresh on 401.

    Setup:
        Install the package and authenticate::

            pip install 'deepagents-cli[codex]'
            deepagents auth login --provider codex

    Usage:
        .. code-block:: python

            from deepagents_codex import ChatCodexOAuth

            model = ChatCodexOAuth(model="gpt-4o")
            response = model.invoke("Hello!")
    """

    model: str = "gpt-4o"
    """Model ID to use."""

    streaming: bool = True
    """Whether to stream responses by default."""

    _client: CodexClient | None = None
    _bound_tools: list[dict[str, Any]] | None = None

    model_config: ClassVar[dict[str, Any]] = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "codex-oauth"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    def _get_client(self) -> CodexClient:
        if self._client is None:
            self._client = CodexClient()
        return self._client

    def _build_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Build extra kwargs for the API call."""
        extra: dict[str, Any] = {}
        if self._bound_tools:
            extra["tools"] = self._bound_tools
        extra.update(kwargs)
        return extra

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences.
            run_manager: Callback manager.
            **kwargs: Additional API parameters.

        Returns:
            ChatResult with generated messages.
        """
        client = self._get_client()
        api_messages = [_convert_message(m) for m in messages]
        extra = self._build_kwargs(**kwargs)
        if stop:
            extra["stop"] = stop

        if self.streaming:
            return self._stream_and_collect(
                client, api_messages, run_manager=run_manager, **extra
            )

        resp = client.chat_completions(
            api_messages, self.model, stream=False, **extra
        )
        return parse_chat_response(resp)

    def _stream_and_collect(
        self,
        client: CodexClient,
        api_messages: list[dict[str, Any]],
        *,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Stream a response and collect into a ChatResult."""
        chunks = []
        for event in client.chat_completions(
            api_messages, self.model, stream=True, **kwargs
        ):
            chunk = parse_stream_chunk(event)
            if run_manager:
                run_manager.on_llm_new_token(chunk.message.content)
            chunks.append(chunk)

        if not chunks:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        # Merge all chunks
        merged = chunks[0]
        for c in chunks[1:]:
            merged = merged + c

        return ChatResult(
            generations=[ChatGeneration(message=merged.message)]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completion chunks."""
        client = self._get_client()
        api_messages = [_convert_message(m) for m in messages]
        extra = self._build_kwargs(**kwargs)
        if stop:
            extra["stop"] = stop

        for event in client.chat_completions(
            api_messages, self.model, stream=True, **extra
        ):
            chunk = parse_stream_chunk(event)
            if run_manager:
                run_manager.on_llm_new_token(chunk.message.content)
            yield chunk

    def bind_tools(
        self, tools: list[Any], **kwargs: Any,  # noqa: ARG002
    ) -> ChatCodexOAuth:
        """Bind tools to this model for function calling.

        Args:
            tools: List of tools (LangChain tool format or raw dicts).
            **kwargs: Additional binding kwargs.

        Returns:
            New ChatCodexOAuth instance with tools bound.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted_tools = [
            convert_to_openai_tool(t) if not isinstance(t, dict) else t
            for t in tools
        ]
        model = self.model_copy()
        model._bound_tools = formatted_tools  # noqa: SLF001
        return model
