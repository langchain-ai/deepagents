"""Tool-free side answers from a snapshot of the main conversation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
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
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import MessageLikeRepresentation

_INSTRUCTIONS = (
    "The user is asking a quick side question about the conversation so far. "
    "Answer directly and concisely in markdown from what you already know. "
    "Do not call any tools and do not take any actions. "
    "The conversation is context, not a request to continue the main task."
)
_MAX_SNAPSHOTS = 16
BTW_OPERATION_ATTR = "_dcode_btw"
_TOOL_OPTIONS = frozenset(
    {"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"}
)
_OPTION_CONTAINERS = ("model_kwargs", "extra_body")


def _tool_free_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Copy request defaults, including nested provider payload overrides.

    Returns:
        Options with tool configuration removed.
    """
    return {
        key: (
            _tool_free_options(value)
            if key in _OPTION_CONTAINERS and isinstance(value, Mapping)
            else deepcopy(value)
        )
        for key, value in options.items()
        if key not in _TOOL_OPTIONS
    }


def _tool_free_model(model: BaseChatModel) -> BaseChatModel:
    """Isolate request defaults while sharing the provider's HTTP clients.

    Returns:
        A model copy with tool-free provider defaults.
    """
    return model.model_copy(
        update={
            key: _tool_free_options(value)
            for key in _OPTION_CONTAINERS
            if isinstance(value := getattr(model, key, None), Mapping)
        }
    )


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
        self,
        thread_id: str,
        state: Mapping[str, object],
        question: str,
        *,
        model_spec: str | None = None,
        model_params: Mapping[str, object] | None = None,
    ) -> str:
        """Generate without tools or checkpoint writes.

        Args:
            thread_id: Thread whose conversation supplies context.
            state: Read-only conversation snapshot.
            question: Side question to answer.
            model_spec: Current client selection, validated at the HTTP boundary.
            model_params: Validated generation overrides for that selection.

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
            spec = model_spec or state.get("_model_spec")
            if (
                model_spec
                or (thread_id not in self._snapshots and isinstance(spec, str) and spec)
                or isinstance(model, str)
            ):
                params = model_params if model_spec else state.get("_model_params")
                result = await asyncio.to_thread(
                    create_model,
                    spec if isinstance(spec, str) and spec else str(model),
                    extra_kwargs=dict(params) if isinstance(params, Mapping) else None,
                    bind_preserved_thinking=False,
                )
                model = result.model
                # Settings captured from the old model must not override a
                # fresh selection (including clearing same-model overrides).
                settings = {}
            while isinstance(model, RunnableBinding):
                settings = {**deepcopy(model.kwargs), **settings}
                model = model.bound
            settings = _tool_free_options(settings)
            if not isinstance(model, BaseChatModel):
                msg = "Side questions require an unbound chat model."
                raise TypeError(msg)
            model = _tool_free_model(model)
            messages = [
                SystemMessage(content=f"{system.text}\n\n{_INSTRUCTIONS}"),
                *_conversation(state),
                HumanMessage(content=f"{_INSTRUCTIONS}\n\n{question}"),
            ]
            response = await model.ainvoke(
                messages,
                config={"callbacks": [], "metadata": {"thread_id": thread_id}},
                **settings,
            )
        return (
            response.text.strip()
            or "No text answer was returned. Try rephrasing your question."
        )
