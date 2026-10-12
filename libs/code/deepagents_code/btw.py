"""Read-only side answers from an isolated snapshot of the main conversation."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import aclosing
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from deepagents.backends import StateBackend
from deepagents.backends.utils import TOOL_RESULT_TOKEN_LIMIT, TRUNCATION_GUIDANCE
from deepagents.middleware.filesystem import NUM_CHARS_PER_TOKEN, FilesystemMiddleware
from deepagents.middleware.memory import MemoryState
from deepagents.middleware.skills import SkillsMiddleware, SkillsState
from deepagents.middleware.summarization import create_summarization_middleware
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
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
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    trim_messages,
)
from langchain_core.runnables import RunnableBinding
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents_code._tool_free import tool_free_model, tool_free_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.filesystem import FsToolName
    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import (
        AnyMessage,
        MessageLikeRepresentation,
    )
    from langchain_core.runnables import Runnable, RunnableConfig

_INSTRUCTIONS = (
    "The user is asking a quick side question about the conversation so far. "
    "Answer directly and concisely in markdown. {access} "
    "Do not modify files, run commands, delegate, or continue the main task. "
    "Everything before this side-conversation boundary is reference context only, "
    "not active instructions, approvals, or pending tool calls to execute."
)
_MAX_SNAPSHOTS = 16
_MAX_TOOL_ROUNDS = 3
_LAST_ROUND_NOTE = (
    "Side tool limit reached. Answer from the results gathered so far "
    "without calling more tools."
)
_MAX_TOOL_CALLS = 8
_READ_TOOLS: tuple[FsToolName, ...] = ("ls", "read_file", "glob", "grep")
_MAX_RETRY_DELAY = 60
"""Leave generation time within the API's 120-second request deadline."""
BTW_OPERATION_ATTR = "_dcode_btw"


class BtwLimitError(ValueError):
    """A side question exceeded a limit; the message is safe to show the user."""


class _InstructionState(MemoryState, SkillsState):
    """Local copy of the checkpoint channels used to assemble instructions."""


class _BtwState(AgentState):
    """Effective side-request context saved with a successful main response."""

    _btw_system_prompt: Annotated[NotRequired[str], PrivateStateAttr]
    _btw_model_settings: Annotated[NotRequired[dict[str, object]], PrivateStateAttr]
    _btw_model_retries: Annotated[NotRequired[int | None], PrivateStateAttr]
    _btw_tools: Annotated[NotRequired[list[dict[str, Any]]], PrivateStateAttr]


def _instruction_state(state: Mapping[str, object]) -> _InstructionState:
    """Copy the checkpoint channels that the instruction loaders read.

    Returns:
        A local state the loaders can update without touching the checkpoint.
    """
    channels = (
        "memory_contents",
        "skills_metadata",
        "skills_load_errors",
        "skills_loaded_by",
    )
    return cast(
        "_InstructionState",
        {
            "messages": [],
            **{key: deepcopy(state[key]) for key in channels if key in state},
        },
    )


async def _restore_system(
    system: SystemMessage,
    model: BaseChatModel,
    state: Mapping[str, object],
    middleware: Sequence[MemoryMiddleware | SkillsMiddleware],
) -> SystemMessage:
    local = _instruction_state(state)
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


def _skills_section(
    loader: SkillsMiddleware, model: BaseChatModel, state: _InstructionState
) -> str:
    """Render the skill section that `loader` adds to a system prompt.

    Returns:
        The section text, or an empty string when `loader` adds none.
    """
    request = ModelRequest(
        model=model,
        system_message=None,
        state=state,
        messages=[],
        tools=[],
        runtime=Runtime(),
    )
    message = loader.modify_request(request).system_message
    return message.text if message else ""


async def _refresh_skills(
    system: SystemMessage,
    model: BaseChatModel,
    state: Mapping[str, object],
    middleware: Sequence[MemoryMiddleware | SkillsMiddleware],
) -> SystemMessage:
    """Replace stale skill sections in saved instructions.

    A saved prompt can outlive its skill list. `/reload` restarts the server,
    so the loaders rescan before the next main turn saves a new prompt.

    Returns:
        `system` with each changed skill section replaced.
    """
    text = system.text
    local = _instruction_state(state)
    runtime = Runtime()
    for loader in middleware:
        if not isinstance(loader, SkillsMiddleware):
            continue
        before = _skills_section(loader, model, local)
        if update := await loader.abefore_agent(local, runtime, {}):
            local.update(update)
            text = text.replace(before, _skills_section(loader, model, local), 1)
    return SystemMessage(content=text)


def _retry_budget(model: object) -> int | None:
    """Read the effective budget without persisting the provider model.

    Returns:
        The dcode-owned retry count, if the model carries one.
    """
    from deepagents_code.config import MODEL_RETRIES_ATTR

    while isinstance(model, RunnableBinding):
        model = model.bound
    retries = getattr(model, MODEL_RETRIES_ATTR, None)
    return retries if type(retries) is int and retries >= 0 else None


async def _open_text_stream(
    model: Runnable[LanguageModelInput, AIMessage],
    messages: Sequence[BaseMessage],
    config: RunnableConfig,
    settings: Mapping[str, Any],
) -> tuple[AsyncGenerator[AIMessageChunk, None], AIMessageChunk]:
    """Open one attempt through its first visible fragment, closing failures.

    Returns:
        The owned stream and accumulated response through its first visible text.
    """
    # BaseChatModel.astream is an async generator, annotated as an iterator.
    stream = cast(
        "AsyncGenerator[AIMessageChunk, None]",
        model.astream(messages, config=config, **settings),
    )
    response = AIMessageChunk(content="")
    try:
        async for chunk in stream:
            response += chunk
            if chunk.text:
                return stream, response
    except BaseException:
        await stream.aclose()
        raise
    return stream, response


async def _stream_answer(
    model: Runnable[LanguageModelInput, AIMessage],
    messages: Sequence[BaseMessage],
    config: RunnableConfig,
    settings: Mapping[str, Any],
    on_text: Callable[[str], Awaitable[None]],
    retry_model: BaseChatModel,
) -> AIMessageChunk:
    """Retry before visible output; never replay an answer already being shown.

    Returns:
        The complete response, including streamed tool calls.
    """
    from deepagents_code.model_retry import aretry_model_call

    stream, first = await aretry_model_call(
        retry_model,
        lambda: _open_text_stream(model, messages, config, settings),
        max_total_delay=_MAX_RETRY_DELAY,
    )
    response = first
    async with aclosing(stream):
        if first.text:
            await on_text(first.text)
        async for chunk in stream:
            response += chunk
            if fragment := chunk.text:
                await on_text(fragment)
    return response


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


def _structured_transcript(messages: Sequence[BaseMessage]) -> list[AnyMessage]:
    """Keep complete tool exchanges and close interrupted calls.

    Returns:
        Isolated messages with valid tool-call/result pairs.
    """
    transcript: list[AnyMessage] = []
    pending: dict[str, str] = {}
    for original in messages:
        message = original.model_copy(deep=True)
        if not isinstance(message, ToolMessage):
            transcript.extend(
                ToolMessage(
                    content="Main tool call was still pending at the side snapshot.",
                    tool_call_id=call_id,
                    name=name,
                )
                for call_id, name in pending.items()
            )
            pending.clear()
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                transcript.append(_tool_result_context(message))
                continue
            pending.pop(message.tool_call_id)
        if isinstance(message, AIMessage):
            pending.update(
                {
                    call["id"]: call["name"]
                    for call in message.tool_calls
                    if call["id"] is not None
                }
            )
        transcript.append(cast("AnyMessage", message))
    transcript.extend(
        ToolMessage(
            content="Main tool call was still pending at the side snapshot.",
            tool_call_id=call_id,
            name=name,
        )
        for call_id, name in pending.items()
    )
    return transcript


def _bound_result(message: ToolMessage) -> ToolMessage:
    """Shorten a side tool result the main agent would evict.

    Side tools run outside `FilesystemMiddleware`, so large results would
    otherwise enter the next request in full and overflow its budget.

    Returns:
        `message`, or a copy with its text cut to the main eviction limit.
    """
    limit = NUM_CHARS_PER_TOKEN * TOOL_RESULT_TOKEN_LIMIT
    if not isinstance(message.content, str) or len(message.content) <= limit:
        return message
    content = message.content[: limit - len(TRUNCATION_GUIDANCE)] + TRUNCATION_GUIDANCE
    return message.model_copy(update={"content": content})


def _with_note(message: ToolMessage, note: str) -> ToolMessage:
    """Append `note` to a tool result without dropping content blocks.

    Returns:
        A copy of `message` that ends with `note`.
    """
    if isinstance(message.content, str):
        content: str | list[str | dict] = f"{message.content}\n\n{note}"
    else:
        content = [*message.content, {"type": "text", "text": note}]
    return message.model_copy(update={"content": content})


def _client_tools(tools: Sequence[object]) -> list[dict[str, Any]]:
    """Retain only client-dispatched schemas, never provider-executed tools.

    Returns:
        Isolated client tool definitions.
    """
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            schemas.append(convert_to_openai_tool(tool))
        elif isinstance(tool, dict) and tool.get("type") == "function":
            schemas.append(deepcopy(cast("dict[str, Any]", tool)))
    return schemas


def _instructions(tools: Sequence[str]) -> str:
    """Describe only the tools this side chat can execute.

    Returns:
        Side-chat instructions for the final user message.
    """
    access = (
        f"You may inspect files using {', '.join(tools)}. "
        "All other tools are disabled in this side chat."
        if tools
        else "All tools are disabled in this side chat. Answer from the conversation."
    )
    return _INSTRUCTIONS.format(access=access)


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
    question_index = next(
        (
            index
            for index in range(len(request.messages) - 1, -1, -1)
            if isinstance(request.messages[index], HumanMessage)
        ),
        0,
    )
    compaction._check_reduction(
        request, request.override(messages=request.messages[question_index:]), None
    )
    return trim_messages(
        messages,
        max_tokens=max(0, budget - compaction._count_tokens([], None, request.tools)),
        token_counter=compaction.token_counter,
        strategy="last",
        start_on="human",
        include_system=True,
    )


def _prepare_messages(request: ModelRequest) -> list[AnyMessage]:
    """Reuse SDK compaction policies without running summaries or backend writes.

    Returns:
        A structured transcript sized for the resolved model and output settings.
    """
    request = UnsupportedContentMiddleware()._filter_request(request)
    compaction = create_summarization_middleware(request.model, StateBackend())
    messages, _ = compaction._truncate_args(
        request.messages,
        compaction._count_tokens(
            request.messages, request.system_message, request.tools
        ),
    )
    return _structured_transcript(
        _fit_context(
            request.override(messages=_structured_transcript(messages)), compaction
        )
    )


class BtwOperation(AgentMiddleware):
    """Capture effective main-call context and answer read-only side questions.

    Register after middleware that modifies the model, prompt, or settings,
    including extensions. `ConfigurableModelMiddleware` resolves the model
    early, before memory, skills, and local context enrich the prompt. Those
    downstream request overrides are not returned to it with the response.
    Resume and cost hooks inspect state and responses rather than the complete
    model request, so neither supplies this capture point.

    This separate hook snapshots the resolved request before calling the model,
    allowing `/btw` to use it while the main response is still streaming. Main
    calls also checkpoint effective instructions and generation settings on
    success for use after restart or eviction. `answer` invokes the model without
    running the agent or writing conversation state; when saved instructions are absent,
    it reuses the main agent's memory and skill loaders on a local state copy.
    """

    state_schema = _BtwState

    def __init__(
        self,
        model: str | BaseChatModel,
        system_prompt: str,
        environ: Mapping[str, str] | None,
        *,
        instruction_middleware: Sequence[MemoryMiddleware | SkillsMiddleware] = (),
        profile_overrides: dict[str, object] | None = None,
        backend: BackendProtocol | None = None,
        filesystem_tools: Sequence[FsToolName] | None = None,
    ) -> None:
        """Keep workspace defaults and the read-only instruction loaders.

        Args:
            model: Workspace bootstrap model.
            system_prompt: Base instructions for a thread without a live snapshot.
            environ: Workspace environment for lazy model resolution.
            instruction_middleware: Main agent memory and skill loaders, in order.
            profile_overrides: Session profile fields to retain during reconstruction.
            backend: Main filesystem backend for read-only inspection.
            filesystem_tools: Main filesystem allowlist to respect in side calls.
        """
        allowed = [
            name
            for name in _READ_TOOLS
            if filesystem_tools is None or name in filesystem_tools
        ]
        self._read_tools = (
            FilesystemMiddleware(backend=backend, tools=allowed).tools
            if backend is not None and "read_file" in allowed
            else []
        )
        self._tool_snapshots: dict[str, list[dict[str, Any]]] = {}
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._environ = environ
        self._instruction_middleware = tuple(instruction_middleware)
        self._profile_overrides = deepcopy(profile_overrides)
        self._snapshots: OrderedDict[
            str, tuple[BaseChatModel, SystemMessage, dict[str, Any]]
        ] = OrderedDict()

    def _remember_model(self, request: ModelRequest) -> Command | None:
        """Snapshot resolved settings before either kind of main model call.

        The in-memory snapshot is available immediately, even while the main
        call is pending. Its context checkpoint update is only returned
        with a successful main response.

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
            self._tool_snapshots[info.thread_id] = _client_tools(request.tools)
            self._snapshots.move_to_end(info.thread_id)
            while len(self._snapshots) > _MAX_SNAPSHOTS:
                evicted, _ = self._snapshots.popitem(last=False)
                self._tool_snapshots.pop(evicted, None)
            return Command(
                update={
                    "_btw_system_prompt": (request.system_message or self._system).text,
                    "_btw_model_settings": tool_free_settings(
                        request.model, request.model_settings
                    ),
                    "_btw_model_retries": _retry_budget(request.model),
                    "_btw_tools": deepcopy(self._tool_snapshots[info.thread_id]),
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

    async def _resolve_context(
        self, thread_id: str, state: Mapping[str, object]
    ) -> tuple[BaseChatModel, SystemMessage, dict[str, Any]]:
        """Resolve side-answer settings within the workspace environment.

        Returns:
            Model, instructions, and generation settings with tools removed,
            preferring live snapshots over checkpoint and workspace defaults.

        Raises:
            TypeError: If the configured model is not a chat model.
        """
        from deepagents_code.config import MODEL_RETRIES_ATTR, create_model

        snapshot = self._snapshots.get(thread_id)
        model, system, settings = snapshot or (self._model, self._system, {})
        if snapshot is None:
            saved_settings = state.get("_btw_model_settings")
            if isinstance(saved_settings, Mapping):
                settings = dict(saved_settings)
        settings = deepcopy(settings)
        spec = state.get("_model_spec")
        if (snapshot is None and isinstance(spec, str) and spec) or isinstance(
            model, str
        ):
            params = state.get("_model_params")
            retries = state.get("_btw_model_retries")
            result = await asyncio.to_thread(
                create_model,
                spec if isinstance(spec, str) and spec else str(model),
                extra_kwargs=dict(params) if isinstance(params, Mapping) else None,
                profile_overrides=self._profile_overrides,
                bind_preserved_thinking=False,
                cli_max_retries=retries if type(retries) is int else None,
            )
            model = result.model
        settings = tool_free_settings(model, settings)
        while isinstance(model, RunnableBinding):
            model = model.bound
        if not isinstance(model, BaseChatModel):
            msg = "Side questions require an unbound chat model."
            raise TypeError(msg)
        if snapshot is None:
            saved_system = state.get("_btw_system_prompt")
            if isinstance(saved_system, str):
                system = await _refresh_skills(
                    SystemMessage(content=saved_system),
                    model,
                    state,
                    self._instruction_middleware,
                )
            else:
                system = await _restore_system(
                    system, model, state, self._instruction_middleware
                )
        model = tool_free_model(model)
        if snapshot is None and type(retries := state.get("_btw_model_retries")) is int:
            setattr(model, MODEL_RETRIES_ATTR, retries)
        return model, system, settings

    async def answer(
        self,
        thread_id: str,
        state: Mapping[str, object],
        question: str,
        *,
        history: Sequence[tuple[str, str]] = (),
        on_text: Callable[[str], Awaitable[None]] | None = None,
        read_tools: bool = False,
    ) -> str:
        """Generate with optional read-only tools and no checkpoint writes.

        Use the thread's latest server-resolved model and instructions, falling
        back to checkpoint settings or the workspace's bootstrap model. Restore
        the checkpointed instructions, generation settings, and retry budget when
        no live snapshot exists.
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
            on_text: Optional receiver for text fragments as the model produces them.
            read_tools: Execute the main allowlist's read-only file tools.
                Side calls bypass the main tool middleware, including
                `PreToolUse` hooks, so enable this only when no hook could
                deny a read.

        Returns:
            The ephemeral answer text.

        Raises:
            TypeError: If the configured model is not a chat model.
            BtwLimitError: If the side conversation exceeds its tool budget.
        """  # noqa: DOC502 — raised by _resolve_context
        from deepagents_code.config import use_environment
        from deepagents_code.model_retry import aretry_model_call

        with use_environment(self._environ):
            model, system, settings = await self._resolve_context(thread_id, state)
            schemas = (
                self._tool_snapshots.get(thread_id)
                if thread_id in self._snapshots
                else state.get("_btw_tools")
            )
            tools = (
                _client_tools(schemas)
                if isinstance(schemas, list)
                else _client_tools(self._read_tools)
            )
            advertised = {tool["function"]["name"] for tool in tools}
            executable = [
                tool
                for tool in self._read_tools
                if read_tools and tool.name in advertised
            ]
            messages = [
                *_structured_transcript(_conversation(state)),
                *(
                    message
                    for prompt, answer in history
                    for message in (
                        HumanMessage(content=prompt),
                        AIMessage(content=answer),
                    )
                ),
                HumanMessage(
                    content=f"{_instructions([tool.name for tool in executable])}"
                    f"\n\n{question}"
                ),
            ]
            prepared = _prepare_messages(
                ModelRequest(
                    model=model,
                    system_message=system,
                    messages=messages,
                    tools=[*tools],
                    model_settings=settings,
                    runtime=Runtime(),
                )
            )
            config: RunnableConfig = {
                "callbacks": [],
                "metadata": {"thread_id": thread_id},
            }
            bound = model.bind_tools(tools) if tools else model
            node = ToolNode(executable)
            parts: list[str] = []
            for round_index in range(_MAX_TOOL_ROUNDS + 1):
                if on_text is None:
                    response = await aretry_model_call(
                        model,
                        lambda messages=prepared: bound.ainvoke(
                            messages, config=config, **settings
                        ),
                        max_total_delay=_MAX_RETRY_DELAY,
                    )
                else:
                    response = await _stream_answer(
                        bound, prepared, config, settings, on_text, model
                    )
                if response.text:
                    parts.append(response.text)
                if not isinstance(response, AIMessage) or not response.tool_calls:
                    break
                if round_index == _MAX_TOOL_ROUNDS:
                    if parts:
                        break
                    msg = "Side tool limit reached. Try a narrower question."
                    raise BtwLimitError(msg)
                if len(response.tool_calls) > _MAX_TOOL_CALLS:
                    msg = "Too many side tools requested. Try a narrower question."
                    raise BtwLimitError(msg)
                result = await node.ainvoke(
                    {"messages": [response]}, config=config, runtime=Runtime()
                )
                results = [_bound_result(message) for message in result["messages"]]
                if round_index == _MAX_TOOL_ROUNDS - 1:
                    results[-1] = _with_note(results[-1], _LAST_ROUND_NOTE)
                prepared = _prepare_messages(
                    ModelRequest(
                        model=model,
                        system_message=system,
                        messages=[*prepared[1:], response, *results],
                        tools=[*tools],
                        model_settings=settings,
                        runtime=Runtime(),
                    )
                )
        text = "".join(parts)
        return (
            text.strip() or "No text answer was returned. Try rephrasing your question."
        )
