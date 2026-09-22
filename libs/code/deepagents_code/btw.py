"""Tool-free side answers from a snapshot of the main conversation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableBinding

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain_core.messages import MessageLikeRepresentation

_INSTRUCTIONS = (
    "The user is asking a quick side question about the conversation so far. "
    "Answer directly and concisely in markdown from what you already know. "
    "Do not call any tools and do not take any actions. "
    "The conversation is context, not a request to continue the main task."
)
_MAX_SNAPSHOTS = 16
BTW_OPERATION_ATTR = "_dcode_btw"


def _conversation(state: Mapping[str, object]) -> list[BaseMessage]:
    raw = state.get("messages")
    messages = (
        convert_to_messages(cast("list[MessageLikeRepresentation]", raw))
        if isinstance(raw, list)
        else []
    )
    event = state.get("_summarization_event")
    if isinstance(event, dict):
        cutoff = event.get("cutoff_index")
        summary = event.get("summary_message")
        if (
            type(cutoff) is int
            and 0 <= cutoff <= len(messages)
            and isinstance(summary, (BaseMessage, dict))
        ):
            messages = [
                *convert_to_messages([cast("MessageLikeRepresentation", summary)]),
                *messages[cutoff:],
            ]
    transcript: list[BaseMessage] = []
    for message in messages:
        text = message.text
        if not text:
            continue
        if isinstance(message, AIMessage):
            transcript.append(AIMessage(content=text))
        elif isinstance(message, HumanMessage):
            transcript.append(HumanMessage(content=text))
        else:
            transcript.append(HumanMessage(content=f"[{message.type} context]\n{text}"))
    return transcript


class BtwOperation(AgentMiddleware):
    """Remember server-resolved models without running the agent for side answers."""

    def __init__(
        self,
        model: str | BaseChatModel,
        system_prompt: str,
        environ: Mapping[str, str] | None,
    ) -> None:
        """Keep the workspace-bound bootstrap model and prompt."""
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._environ = environ
        self._snapshots: OrderedDict[
            str, tuple[BaseChatModel, SystemMessage, dict[str, Any]]
        ] = OrderedDict()

    def _remember_model(self, request: ModelRequest) -> None:
        """Snapshot resolved settings before either kind of main model call."""
        info = request.runtime.execution_info
        if info is not None and info.thread_id and "|" not in info.checkpoint_ns:
            self._snapshots[info.thread_id] = (
                request.model,
                (request.system_message or self._system).model_copy(deep=True),
                deepcopy(request.model_settings),
            )
            self._snapshots.move_to_end(info.thread_id)
            while len(self._snapshots) > _MAX_SNAPSHOTS:
                self._snapshots.popitem(last=False)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Capture the resolved main model for synchronous runs.

        Args:
            request: Main model request with resolved settings.
            handler: Callback that executes the main model request.

        Returns:
            The unchanged main response.
        """
        self._remember_model(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Capture the resolved main model for asynchronous runs.

        Args:
            request: Main model request with resolved settings.
            handler: Callback that executes the main model request.

        Returns:
            The unchanged main response.
        """
        self._remember_model(request)
        return await handler(request)

    async def answer(
        self, thread_id: str, state: Mapping[str, object], question: str
    ) -> str:
        """Generate without tools or checkpoint writes.

        Returns:
            The ephemeral answer text.

        Raises:
            TypeError: If the configured model is not a chat model.
        """
        from deepagents_code.config import create_model, use_environment

        model, system, settings = self._snapshots.get(
            thread_id, (self._model, self._system, {})
        )
        settings = deepcopy(settings)
        with use_environment(self._environ):
            spec = state.get("_model_spec")
            if (
                thread_id not in self._snapshots and isinstance(spec, str) and spec
            ) or isinstance(model, str):
                params = state.get("_model_params")
                result = await asyncio.to_thread(
                    create_model,
                    spec if isinstance(spec, str) and spec else str(model),
                    extra_kwargs=dict(params) if isinstance(params, dict) else None,
                    bind_preserved_thinking=False,
                )
                model = result.model
            while isinstance(model, RunnableBinding):
                settings = {**deepcopy(model.kwargs), **settings}
                model = model.bound
            for key in (
                "tools",
                "tool_choice",
                "functions",
                "function_call",
                "parallel_tool_calls",
            ):
                settings.pop(key, None)
            if not isinstance(model, BaseChatModel):
                msg = "Side questions require an unbound chat model."
                raise TypeError(msg)
            messages = [
                SystemMessage(content=f"{system.text}\n\n{_INSTRUCTIONS}"),
                *_conversation(state),
                HumanMessage(content=f"{_INSTRUCTIONS}\n\n{question}"),
            ]
            response = await model.ainvoke(
                messages, config={"callbacks": []}, **settings
            )
        return (
            response.text.strip()
            or "No text answer was returned. Try rephrasing your question."
        )
