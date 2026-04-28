from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import override


class GenericFakeChatModel(BaseChatModel):
    messages: Iterator[AIMessage | str]
    call_history: list[Any] = []  # noqa: RUF012  # Test-only model class
    stream_delimiter: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del tools, tool_choice, kwargs
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
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
        chat_result = self._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        message = chat_result.generations[0].message
        if not isinstance(message, AIMessage):
            msg = "Expected generation to return AIMessage"
            raise ValueError(msg)

        content = message.content or ""
        if self.stream_delimiter is None:
            content_chunks = cast("list[str]", [content])
        else:
            content_chunks = cast(
                "list[str]",
                [part for part in re.split(self.stream_delimiter, content) if part],
            )

        for token in content_chunks:
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(id=message.id, content=token)
            )
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"
