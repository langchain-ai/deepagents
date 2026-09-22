"""Tool-free side answers from a snapshot of the main conversation."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from deepagents.middleware.memory import MemoryState
from deepagents.middleware.skills import SkillsState
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
    ToolMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableBinding
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.skills import SkillsMiddleware
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


class _InstructionState(MemoryState, SkillsState):
    """Local copy of the checkpoint channels used to assemble instructions."""


async def _restore_system(
    system: SystemMessage,
    model: BaseChatModel,
    state: Mapping[str, object],
    middleware: Sequence[MemoryMiddleware | SkillsMiddleware],
) -> SystemMessage:
    channels = ("memory_contents", "skills_metadata", "skills_load_errors")
    local = cast(
        "_InstructionState",
        {
            "messages": [],
            **{key: deepcopy(state[key]) for key in channels if key in state},
        },
    )
    runtime = Runtime()
    request = ModelRequest(
        model=model,
        system_message=system,
        state=local,
        messages=[],
        tools=[],
        runtime=runtime,
    )
    for item in middleware:
        update = await item.abefore_agent(local, runtime, {})
        if update:
            local.update(update)
        request = item.modify_request(request)
    return request.system_message or system


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
        if isinstance(message, AIMessage):
            calls = [
                f"[tool call {call['id']}: {call['name']}]\n"
                + json.dumps(call["args"], ensure_ascii=False)
                for call in message.tool_calls
            ]
            text = "\n\n".join(part for part in [text, *calls] if part)
        elif isinstance(message, ToolMessage):
            label = f"{message.tool_call_id}: {message.name or 'tool'}"
            text = f"[tool result {label}]\n{text}"
        if not text:
            continue
        if isinstance(message, AIMessage):
            transcript.append(AIMessage(content=text))
        elif isinstance(message, HumanMessage | ToolMessage):
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
        *,
        instruction_middleware: Sequence[MemoryMiddleware | SkillsMiddleware] = (),
    ) -> None:
        """Keep workspace defaults and the read-only instruction loaders.

        Args:
            model: Workspace bootstrap model.
            system_prompt: Base instructions for a thread without a live snapshot.
            environ: Workspace environment for lazy model resolution.
            instruction_middleware: Main agent memory and skill loaders, in order.
        """
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._environ = environ
        self._instruction_middleware = tuple(instruction_middleware)
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
    ) -> str:
        """Generate without tools or checkpoint writes.

        Use the thread's latest server-resolved model and instructions, falling
        back to checkpoint settings or the workspace's bootstrap model. Restore
        memory and skills through the main agent's loaders when no live snapshot
        exists, without writing their updates back to the conversation.

        Args:
            thread_id: Thread whose conversation supplies context.
            state: Read-only conversation snapshot.
            question: Side question to answer.

        Returns:
            The ephemeral answer text.

        Raises:
            TypeError: If the configured model is not a chat model.
        """
        from deepagents_code.config import create_model, use_environment

        snapshot = self._snapshots.get(thread_id)
        model, system, settings = snapshot or (self._model, self._system, {})
        settings = deepcopy(settings)
        with use_environment(self._environ):
            spec = state.get("_model_spec")
            if (snapshot is None and isinstance(spec, str) and spec) or isinstance(
                model, str
            ):
                params = state.get("_model_params")
                result = await asyncio.to_thread(
                    create_model,
                    spec if isinstance(spec, str) and spec else str(model),
                    extra_kwargs=dict(params) if isinstance(params, Mapping) else None,
                    bind_preserved_thinking=False,
                )
                model = result.model
            while isinstance(model, RunnableBinding):
                settings = {**deepcopy(model.kwargs), **settings}
                model = model.bound
            settings = _tool_free_options(settings)
            if not isinstance(model, BaseChatModel):
                msg = "Side questions require an unbound chat model."
                raise TypeError(msg)
            if snapshot is None:
                system = await _restore_system(
                    system, model, state, self._instruction_middleware
                )
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
