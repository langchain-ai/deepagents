"""Fake chat models for testing purposes."""

import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import override


class GenericFakeChatModel(BaseChatModel):
    r"""Generic fake chat model that can be used to test the chat model interface.

    * Chat model should be usable in both sync and async tests
    * Invokes `on_llm_new_token` to allow for testing of callback related code for new
        tokens.
    * Includes configurable logic to break messages into chunks for streaming.
    * Tracks all invoke calls for inspection (messages, kwargs)

    Args:
        messages: An iterator over messages (use `iter()` to convert a list)
        stream_delimiter: How to chunk content when streaming. Options:
            - None (default): Return content in a single chunk (no streaming)
            - A string delimiter (e.g., " "): Split content on this delimiter,
              preserving the delimiter as separate chunks
            - A regex pattern (e.g., r"(\\s)"): Split using the pattern with a capture
              group to preserve delimiters

    Examples:
        # No streaming - single chunk
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello world")]))

        # Stream on whitespace
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Hello world")]),
            stream_delimiter=" "
        )
        # Yields: "Hello", " ", "world"

        # Stream on whitespace (regex) - more flexible
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Hello world")]),
            stream_delimiter=r"(\\s)"
        )
        # Yields: "Hello", " ", "world"

        # Access call history
        model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))
        model.invoke([HumanMessage(content="Hi")])
        print(model.call_history[0]["messages"])
        print(model.call_history[0]["kwargs"])
    """

    messages: Iterator[AIMessage | str]
    """Get an iterator over messages.

    This can be expanded to accept other types like Callables / dicts / strings
    to make the interface more generic if needed.

    !!! note
        if you want to pass a list, you can use `iter` to convert it to an iterator.
    """

    call_history: list[Any] = []  # noqa: RUF012  # Test-only model class

    stream_delimiter: str | None = None
    """Delimiter for chunking content during streaming.

    - None (default): No chunking, returns content in a single chunk
    - String: Split content on this exact string, preserving delimiter as chunks
    - Regex pattern: Use re.split() with the pattern
      (use capture groups to preserve delimiters)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the fake chat model with call tracking."""
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind_tools to return self."""
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate the next message from the iterator."""
        self.call_history.append({"messages": messages, "kwargs": kwargs})

        message = next(self.messages)
        message_ = AIMessage(content=message) if isinstance(message, str) else message
        generation = ChatGeneration(message=message_)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the next message from the iterator in chunks."""
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if not isinstance(chat_result, ChatResult):
            msg = "Expected generate to return ChatResult"
            raise ValueError(msg)  # noqa: TRY004
        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            msg = "Expected generation to return AIMessage"
            raise ValueError(msg)  # noqa: TRY004

        content = message.content
        if content is None:
            content = ""

        if self.stream_delimiter is None:
            content_chunks = cast(list[str], [content])
        else:
            content_chunks = cast(
                list[str],
                [
                    part
                    for part in re.split(self.stream_delimiter, content)
                    if part
                ],
            )

        role = "assistant"
        for token in content_chunks:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(id=message.id, content=token)
            )
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

            if role == "assistant":
                role = ""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "generic-fake-chat-model"
