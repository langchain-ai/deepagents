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

from deepagents_codex.client import CodexClient
from deepagents_codex.response import collect_response_events, parse_stream_event

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Extract plain text from LangChain message content.

    Handles both string content and list-of-block content
    (e.g. ``[{"type": "text", "text": "..."}]``).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _convert_messages(
    messages: list[BaseMessage],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert LangChain messages to Responses API format.

    Extracts system messages as `instructions` and converts the rest
    to Responses API `input` items.

    Args:
        messages: List of LangChain messages.

    Returns:
        Tuple of (instructions_str, input_items_list).
    """
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            instructions_parts.append(_content_to_text(msg.content))
        elif isinstance(msg, HumanMessage):
            input_items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": _content_to_text(msg.content)},
                    ],
                }
            )
        elif isinstance(msg, AIMessage):
            # P1-fix: prior assistant text uses input_text, not output_text
            if msg.content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "input_text",
                                "text": _content_to_text(msg.content),
                            },
                        ],
                    }
                )
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc["args"]
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["name"],
                            "arguments": args,
                        }
                    )
        elif isinstance(msg, ToolMessage):
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": _content_to_text(msg.content),
                }
            )

    instructions = "\n\n".join(instructions_parts)
    return instructions, input_items


class ChatCodexOAuth(BaseChatModel):
    """Chat model that authenticates via Codex OAuth.

    Uses OAuth Bearer tokens from the Codex auth store instead of API keys.
    Sends requests to the Codex Responses API at
    ``chatgpt.com/backend-api/codex/responses``.

    Setup:
        Install the package and authenticate::

            pip install 'deepagents-cli[codex]'
            deepagents auth login --provider codex

    Usage:
        .. code-block:: python

            from deepagents_codex import ChatCodexOAuth

            model = ChatCodexOAuth(model="gpt-5.1-codex")
            response = model.invoke("Hello!")
    """

    model: str = "gpt-5.4"
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

    def _get_tools(self) -> list[dict[str, Any]] | None:
        """Get bound tools in Responses API format."""
        return self._bound_tools

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using the Codex Responses API.

        Args:
            messages: List of LangChain messages.
            stop: Stop sequences (not supported by Codex API).
            run_manager: Callback manager.
            **kwargs: Additional API parameters.

        Returns:
            ChatResult with generated messages.
        """
        client = self._get_client()
        instructions, input_items = _convert_messages(messages)
        tools = self._get_tools()

        events = []
        for event in client.create_response(
            input_items,
            self.model,
            instructions=instructions,
            tools=tools,
            **kwargs,
        ):
            events.append(event)
            if self.streaming and run_manager:
                chunk = parse_stream_event(event)
                if chunk and chunk.message.content:
                    run_manager.on_llm_new_token(chunk.message.content)

        return collect_response_events(events)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream response chunks from the Codex Responses API."""
        client = self._get_client()
        instructions, input_items = _convert_messages(messages)
        tools = self._get_tools()

        for event in client.create_response(
            input_items,
            self.model,
            instructions=instructions,
            tools=tools,
            **kwargs,
        ):
            chunk = parse_stream_event(event)
            if chunk is not None:
                if run_manager and chunk.message.content:
                    run_manager.on_llm_new_token(chunk.message.content)
                yield chunk

    def bind_tools(
        self,
        tools: list[Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatCodexOAuth:
        """Bind tools to this model for function calling.

        Converts tools to Responses API format (flat ``type/name/parameters``),
        not the nested Chat Completions format.

        Args:
            tools: List of tools (LangChain tool format or raw dicts).
            **kwargs: Additional binding kwargs.

        Returns:
            New ChatCodexOAuth instance with tools bound.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted_tools = []
        for t in tools:
            tool_dict = t if isinstance(t, dict) else convert_to_openai_tool(t)
            # Convert from Chat Completions nested format to Responses flat format
            if "function" in tool_dict and "name" not in tool_dict:
                func = tool_dict["function"]
                tool_dict = {
                    "type": "function",
                    "name": func["name"],
                    **{k: v for k, v in func.items() if k != "name"},
                }
            formatted_tools.append(tool_dict)

        model = self.model_copy()
        model._bound_tools = formatted_tools  # noqa: SLF001
        return model
