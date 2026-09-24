"""Tool-free side answers from a snapshot of the main conversation."""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from deepagents.backends import StateBackend
from deepagents.middleware.memory import MemoryState
from deepagents.middleware.skills import SkillsState
from deepagents.middleware.summarization import create_summarization_middleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    trim_messages,
)
from langchain_core.runnables import RunnableBinding
from langgraph.runtime import Runtime
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.skills import SkillsMiddleware
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.messages import AnyMessage, MessageLikeRepresentation

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


class _BtwState(AgentState):
    """Effective instructions saved with the main model's successful response."""

    _btw_system_prompt: Annotated[NotRequired[str], PrivateStateAttr]


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


def _conversation(state: Mapping[str, object]) -> list[AnyMessage]:
    """Restore the effective checkpoint messages before read-only compaction.

    Returns:
        The saved summary followed by messages after its cutoff, if present.
    """
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
    return cast("list[AnyMessage]", messages)


def _tool_result_context(message: ToolMessage) -> HumanMessage:
    """Preserve tool-result content without provider-only message metadata.

    Returns:
        A labeled human message with an isolated copy of any content blocks.
    """
    label = f"[tool result {message.tool_call_id}: {message.name or 'tool'}]\n"
    if isinstance(message.content, str):
        return HumanMessage(content=f"{label}{message.content}")
    return HumanMessage(
        content=[{"type": "text", "text": label}, *deepcopy(message.content)]
    )


def _tool_free_transcript(messages: Sequence[BaseMessage]) -> list[AnyMessage]:
    """Render tool exchanges as context after their arguments have been truncated.

    Returns:
        A transcript without executable tool calls or provider-only metadata.
    """
    transcript: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            if message.content:
                transcript.append(HumanMessage(content=deepcopy(message.content)))
            continue
        if isinstance(message, ToolMessage):
            transcript.append(_tool_result_context(message))
            continue
        text = message.text
        if isinstance(message, AIMessage):
            calls = [
                f"[tool call {call['id']}: {call['name']}]\n"
                + json.dumps(call["args"], ensure_ascii=False)
                for call in message.tool_calls
            ]
            text = "\n\n".join(part for part in [text, *calls] if part)
        if not text:
            continue
        if isinstance(message, AIMessage):
            transcript.append(AIMessage(content=text))
        else:
            transcript.append(HumanMessage(content=f"[{message.type} context]\n{text}"))
    return transcript


def _fit_context(
    request: ModelRequest, compaction: SummarizationMiddleware
) -> list[BaseMessage]:
    """Keep the system prompt and latest question while bounding older context.

    Returns:
        Messages within the model's input budget when its limit is known.
    """
    messages: list[BaseMessage] = [
        *([request.system_message] if request.system_message is not None else []),
        *request.messages,
    ]
    budget = compaction._input_budget(request)
    if budget is None or not compaction._over_budget(request):
        return messages
    # Never drop or shorten the user's question to make the request fit.
    compaction._check_reduction(
        request, request.override(messages=request.messages[-1:]), None
    )
    return trim_messages(
        messages,
        max_tokens=budget,
        token_counter=compaction.token_counter,
        strategy="last",
        start_on="human",
        include_system=True,
    )


def _prepare_messages(request: ModelRequest) -> list[BaseMessage]:
    """Reuse SDK compaction policies without running summaries or backend writes.

    Returns:
        A tool-free transcript sized for the resolved model and output settings.
    """
    compaction = create_summarization_middleware(request.model, StateBackend())
    messages, _ = compaction._truncate_args(
        request.messages,
        compaction._count_tokens(request.messages, request.system_message, []),
    )
    return _fit_context(
        request.override(messages=_tool_free_transcript(messages)), compaction
    )


class BtwOperation(AgentMiddleware):
    """Remember server-resolved models without running the agent for side answers."""

    state_schema = _BtwState

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

    def _remember_model(self, request: ModelRequest) -> Command | None:
        """Snapshot resolved settings before either kind of main model call.

        Returns:
            Instructions to checkpoint on success, or `None` for nested calls.
        """
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
            return Command(
                update={
                    "_btw_system_prompt": (request.system_message or self._system).text
                }
            )
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        """Capture the resolved main model for synchronous runs.

        Args:
            request: Main model request with resolved settings.
            handler: Callback that executes the main model request.

        Returns:
            The main response with private instruction metadata to checkpoint.
        """
        command = self._remember_model(request)
        return ExtendedModelResponse(model_response=handler(request), command=command)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ExtendedModelResponse:
        """Capture the resolved main model for asynchronous runs.

        Args:
            request: Main model request with resolved settings.
            handler: Callback that executes the main model request.

        Returns:
            The main response with private instruction metadata to checkpoint.
        """
        command = self._remember_model(request)
        return ExtendedModelResponse(
            model_response=await handler(request), command=command
        )

    async def answer(
        self,
        thread_id: str,
        state: Mapping[str, object],
        question: str,
        *,
        history: Sequence[tuple[str, str]] = (),
    ) -> str:
        """Generate without tools or checkpoint writes.

        Use the thread's latest server-resolved model and instructions, falling
        back to checkpoint settings or the workspace's bootstrap model. Restore
        the checkpointed effective instructions when no live snapshot exists.
        New and legacy threads restore memory and skills through the main
        agent's loaders without writing updates back to the conversation.
        Large older file arguments are truncated using the main agent's policy;
        older context is dropped when needed to fit the side request's budget.
        Instructions and questions that cannot fit raise `ContextOverflowError`.

        Args:
            thread_id: Thread whose conversation supplies context.
            state: Read-only conversation snapshot.
            question: Side question to answer.
            history: Completed question/answer pairs from this side conversation.

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
                saved_system = state.get("_btw_system_prompt")
                if isinstance(saved_system, str):
                    system = SystemMessage(content=saved_system)
                else:
                    system = await _restore_system(
                        system, model, state, self._instruction_middleware
                    )
            model = _tool_free_model(model)
            messages = [
                *_conversation(state),
                *(
                    message
                    for prompt, answer in history
                    for message in (
                        HumanMessage(content=prompt),
                        AIMessage(content=answer),
                    )
                ),
                HumanMessage(content=f"{_INSTRUCTIONS}\n\n{question}"),
            ]
            response = await model.ainvoke(
                _prepare_messages(
                    ModelRequest(
                        model=model,
                        system_message=SystemMessage(
                            content=f"{system.text}\n\n{_INSTRUCTIONS}"
                        ),
                        messages=messages,
                        tools=[],
                        model_settings=settings,
                        runtime=Runtime(),
                    )
                ),
                config={"callbacks": [], "metadata": {"thread_id": thread_id}},
                **settings,
            )
        return (
            response.text.strip()
            or "No text answer was returned. Try rephrasing your question."
        )
