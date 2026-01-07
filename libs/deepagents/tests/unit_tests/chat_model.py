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
    """Generic fake chat model that can be used to test the chat model interface.

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

    call_history: list[Any] = []

    stream_delimiter: str | None = None
    """Delimiter for chunking content during streaming.

    - None (default): No chunking, returns content in a single chunk
    - String: Split content on this exact string, preserving delimiter as chunks
    - Regex pattern: Use re.split() with the pattern (use capture groups to preserve delimiters)
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
        # Record this call
        self.call_history.append(
            {
                "messages": messages,
                "kwargs": {
                    "stop": stop,
                    "run_manager": run_manager,
                    **kwargs,
                },
            }
        )

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
        chat_result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if not isinstance(chat_result, ChatResult):
            msg = f"Expected generate to return a ChatResult, but got {type(chat_result)} instead."
            raise ValueError(msg)  # noqa: TRY004

        message = chat_result.generations[0].message

        if not isinstance(message, AIMessage):
            msg = f"Expected invoke to return an AIMessage, but got {type(message)} instead."
            raise ValueError(msg)  # noqa: TRY004

        content = message.content
        tool_calls = message.tool_calls if hasattr(message, "tool_calls") else []

        if content:
            if not isinstance(content, str):
                msg = "Expected content to be a string."
                raise ValueError(msg)

            # Chunk content based on stream_delimiter configuration
            if self.stream_delimiter is None:
                # No streaming - return entire content in a single chunk
                content_chunks = [content]
            else:
                # Split content using the delimiter
                # Use re.split to support both string and regex patterns
                content_chunks = cast("list[str]", re.split(self.stream_delimiter, content))
                # Remove empty strings that can result from splitting
                content_chunks = [chunk for chunk in content_chunks if chunk]

            for idx, token in enumerate(content_chunks):
                # Include tool_calls only in the last chunk
                is_last = idx == len(content_chunks) - 1
                chunk_tool_calls = tool_calls if is_last else []

                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=token,
                        id=message.id,
                        tool_calls=chunk_tool_calls,
                    )
                )
                if is_last and isinstance(chunk.message, AIMessageChunk) and not message.additional_kwargs:
                    chunk.message.chunk_position = "last"
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        elif tool_calls:
            # If there's no content but there are tool_calls, yield a single chunk with them
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=message.id,
                    tool_calls=tool_calls,
                    chunk_position="last",
                )
            )
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

        if message.additional_kwargs:
            for key, value in message.additional_kwargs.items():
                # We should further break down the additional kwargs into chunks
                # Special case for function call
                if key == "function_call":
                    for fkey, fvalue in value.items():
                        if isinstance(fvalue, str):
                            # Break function call by `,`
                            fvalue_chunks = cast("list[str]", re.split(r"(,)", fvalue))
                            for fvalue_chunk in fvalue_chunks:
                                chunk = ChatGenerationChunk(
                                    message=AIMessageChunk(
                                        id=message.id,
                                        content="",
                                        additional_kwargs={"function_call": {fkey: fvalue_chunk}},
                                    )
                                )
                                if run_manager:
                                    run_manager.on_llm_new_token(
                                        "",
                                        chunk=chunk,  # No token for function call
                                    )
                                yield chunk
                        else:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(
                                    id=message.id,
                                    content="",
                                    additional_kwargs={"function_call": {fkey: fvalue}},
                                )
                            )
                            if run_manager:
                                run_manager.on_llm_new_token(
                                    "",
                                    chunk=chunk,  # No token for function call
                                )
                            yield chunk
                else:
                    chunk = ChatGenerationChunk(message=AIMessageChunk(id=message.id, content="", additional_kwargs={key: value}))
                    if run_manager:
                        run_manager.on_llm_new_token(
                            "",
                            chunk=chunk,  # No token for function call
                        )
                    yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"
