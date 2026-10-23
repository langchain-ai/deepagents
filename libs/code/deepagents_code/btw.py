"""Tool-free side answers from a snapshot of the main conversation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import aclosing, closing
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from deepagents.backends import StateBackend
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

from deepagents_code._tool_free import tool_free_model, tool_free_settings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain_core.messages import (
        AIMessageChunk,
        AnyMessage,
        MessageLikeRepresentation,
    )
    from langchain_core.runnables import RunnableConfig

_INSTRUCTIONS = (
    "The user is asking a quick side question about the conversation so far. "
    "Answer directly and concisely in markdown from what you already know. "
    "Do not call any tools and do not take any actions. "
    "The conversation is context, not a request to continue the main task."
)
_MAX_SNAPSHOTS = 16
_MAX_RETRY_DELAY = 60
"""Leave generation time within the API's 120-second request deadline."""
BTW_OPERATION_ATTR = "_dcode_btw"

logger = logging.getLogger(__name__)


class _InstructionState(MemoryState, SkillsState):
    """Local copy of the checkpoint channels used to assemble instructions."""


class _BtwState(AgentState):
    """Effective side-request context saved with a successful main response."""

    _btw_system_prompt: Annotated[NotRequired[str | None], PrivateStateAttr]
    """Legacy prompt copy, read as a fallback and cleared once the table has one."""
    _btw_model_settings: Annotated[NotRequired[dict[str, object]], PrivateStateAttr]
    _btw_model_retries: Annotated[NotRequired[int | None], PrivateStateAttr]


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


def _save_prompt(thread_id: str, prompt: str, digest: str) -> None:
    """Store a thread's main prompt in the sessions database.

    Checkpoints store every channel value again, so a prompt in graph state
    repeats in each checkpoint row. One row per thread holds it here instead.
    The upsert skips the write when the saved prompt has the same digest.

    Args:
        thread_id: Thread whose main call used `prompt`.
        prompt: Effective main-call system prompt.
        digest: SHA-256 hex digest of `prompt`.
    """
    from deepagents_code.workspace import _database_path

    with closing(sqlite3.connect(_database_path(), timeout=5)) as conn, conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS dcode_btw_prompts (thread_id TEXT PRIMARY "
            "KEY NOT NULL, digest TEXT NOT NULL, prompt TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO dcode_btw_prompts VALUES (?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET digest = excluded.digest, "
            "prompt = excluded.prompt WHERE digest != excluded.digest",
            (thread_id, digest, prompt),
        )


def _load_prompt(thread_id: str) -> str | None:
    """Read a thread's saved main prompt without creating the database.

    Returns:
        The saved prompt, or `None` when the thread has none.
    """
    from deepagents_code.workspace import _database_path

    path = _database_path()
    if not path.exists():
        return None
    with closing(
        sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=1)
    ) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'dcode_btw_prompts'"
        ).fetchone()
        if not exists:
            return None
        row = conn.execute(
            "SELECT prompt FROM dcode_btw_prompts WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return row[0] if row else None


async def _open_text_stream(
    model: BaseChatModel,
    messages: list[BaseMessage],
    config: RunnableConfig,
    settings: Mapping[str, Any],
) -> tuple[AsyncGenerator[AIMessageChunk], str]:
    """Open one attempt through its first visible fragment, closing failures.

    Returns:
        The owned stream and first text, or empty text for a completed empty stream.
    """
    # BaseChatModel.astream is an async generator, annotated as an iterator.
    stream = cast(
        "AsyncGenerator[AIMessageChunk]",
        model.astream(messages, config=config, **settings),
    )
    try:
        async for chunk in stream:
            if text := chunk.text:
                return stream, text
    except BaseException:
        await stream.aclose()
        raise
    return stream, ""


async def _stream_answer(
    model: BaseChatModel,
    messages: list[BaseMessage],
    config: RunnableConfig,
    settings: Mapping[str, Any],
    on_text: Callable[[str], Awaitable[None]],
) -> str:
    """Retry before visible output; never replay an answer already being shown.

    Returns:
        The complete streamed answer text.
    """
    from deepagents_code.model_retry import aretry_model_call

    stream, first = await aretry_model_call(
        model,
        lambda: _open_text_stream(model, messages, config, settings),
        max_total_delay=_MAX_RETRY_DELAY,
    )
    parts = [first]
    async with aclosing(stream):
        if first:
            await on_text(first)
        async for chunk in stream:
            if fragment := chunk.text:
                parts.append(fragment)
                await on_text(fragment)
    return "".join(parts)


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
    request = UnsupportedContentMiddleware()._filter_request(request)
    compaction = create_summarization_middleware(request.model, StateBackend())
    messages, _ = compaction._truncate_args(
        request.messages,
        compaction._count_tokens(request.messages, request.system_message, []),
    )
    return _fit_context(
        request.override(messages=_tool_free_transcript(messages)), compaction
    )


def _checkpoint_update(request: ModelRequest, *, saved: bool) -> Command:
    """Checkpoint generation settings, which are small, with the response.

    Returns:
        A private-state update that also clears a legacy prompt copy once
        the sessions database holds the current prompt.
    """
    update: dict[str, object] = {
        "_btw_model_settings": tool_free_settings(
            request.model, request.model_settings
        ),
        "_btw_model_retries": _retry_budget(request.model),
    }
    if saved and request.state.get("_btw_system_prompt") is not None:
        update["_btw_system_prompt"] = None
    return Command(update=update)


async def _saved_prompt(thread_id: str, state: Mapping[str, object]) -> str | None:
    """Read the saved main prompt, falling back to a legacy checkpoint copy.

    Returns:
        The saved prompt, or `None` when neither source has one.
    """
    try:
        prompt = await asyncio.to_thread(_load_prompt, thread_id)
    except sqlite3.Error:
        logger.warning("Could not read side-question instructions", exc_info=True)
        prompt = None
    legacy = state.get("_btw_system_prompt")
    if prompt is None and isinstance(legacy, str):
        return legacy
    return prompt


class BtwOperation(AgentMiddleware):
    """Capture effective main-call context and answer tool-free side questions.

    Register after middleware that modifies the model, prompt, or settings,
    including extensions. `ConfigurableModelMiddleware` resolves the model
    early, before memory, skills, and local context enrich the prompt. Those
    downstream request overrides are not returned to it with the response.
    Resume and cost hooks inspect state and responses rather than the complete
    model request, so neither supplies this capture point.

    This separate hook snapshots the resolved request before calling the model,
    allowing `/btw` to use it while the main response is still streaming. Main
    calls also save effective instructions to the sessions database and
    checkpoint generation settings on success for use after restart or
    eviction. Instructions stay out of checkpoints because each checkpoint row
    repeats every state value. `answer` invokes the model without
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
    ) -> None:
        """Keep workspace defaults and the read-only instruction loaders.

        Args:
            model: Workspace bootstrap model.
            system_prompt: Base instructions for a thread without a live snapshot.
            environ: Workspace environment for lazy model resolution.
            instruction_middleware: Main agent memory and skill loaders, in order.
            profile_overrides: Session profile fields to retain during reconstruction.
        """
        self._model = model
        self._system = SystemMessage(content=system_prompt)
        self._environ = environ
        self._instruction_middleware = tuple(instruction_middleware)
        self._profile_overrides = deepcopy(profile_overrides)
        self._snapshots: OrderedDict[
            str, tuple[BaseChatModel, SystemMessage, dict[str, Any]]
        ] = OrderedDict()
        self._saved_digests: dict[str, str] = {}

    def _remember_model(self, request: ModelRequest) -> str | None:
        """Snapshot resolved settings before either kind of main model call.

        The in-memory snapshot is available immediately, even while the main
        call is pending. Durable context is only saved with a successful main
        response.

        Returns:
            The thread ID for main calls, or `None` for nested calls.
        """
        info = request.runtime.execution_info
        if info is None or not info.thread_id or "|" in info.checkpoint_ns:
            return None
        self._snapshots[info.thread_id] = (
            request.model,
            (request.system_message or self._system).model_copy(deep=True),
            deepcopy(request.model_settings),
        )
        self._snapshots.move_to_end(info.thread_id)
        while len(self._snapshots) > _MAX_SNAPSHOTS:
            self._snapshots.popitem(last=False)
        return info.thread_id

    def _unsaved_prompt(
        self, thread_id: str, request: ModelRequest
    ) -> tuple[str, str] | None:
        """Find a prompt that this process has not yet saved for the thread.

        Returns:
            The prompt and its digest, or `None` when the saved one matches.
        """
        prompt = (request.system_message or self._system).text
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if self._saved_digests.get(thread_id) == digest:
            return None
        return prompt, digest

    def _save(self, thread_id: str, prompt: str, digest: str) -> bool:
        """Save the prompt, keeping the main response if the write fails.

        Returns:
            Whether the sessions database now holds the prompt.
        """
        try:
            _save_prompt(thread_id, prompt, digest)
        except sqlite3.Error:
            logger.warning("Could not save side-question instructions", exc_info=True)
            return False
        self._saved_digests[thread_id] = digest
        return True

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
            The main response with private generation settings to checkpoint.
        """
        thread_id = self._remember_model(request)
        response = handler(request)
        if thread_id is None:
            return ExtendedModelResponse(model_response=response, command=None)
        unsaved = self._unsaved_prompt(thread_id, request)
        saved = unsaved is None or self._save(thread_id, *unsaved)
        return ExtendedModelResponse(
            model_response=response,
            command=_checkpoint_update(request, saved=saved),
        )

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
            The main response with private generation settings to checkpoint.
        """
        thread_id = self._remember_model(request)
        response = await handler(request)
        if thread_id is None:
            return ExtendedModelResponse(model_response=response, command=None)
        unsaved = self._unsaved_prompt(thread_id, request)
        # The server event loop must not block on the sessions database.
        saved = unsaved is None or await asyncio.to_thread(
            self._save, thread_id, *unsaved
        )
        return ExtendedModelResponse(
            model_response=response,
            command=_checkpoint_update(request, saved=saved),
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
            saved_system = await _saved_prompt(thread_id, state)
            if saved_system is not None:
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
    ) -> str:
        """Generate without tools or checkpoint writes.

        Use the thread's latest server-resolved model and instructions, falling
        back to checkpoint settings or the workspace's bootstrap model. Restore
        the saved instructions, generation settings, and retry budget when no
        live snapshot exists.
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

        Returns:
            The ephemeral answer text.

        Raises:
            TypeError: If the configured model is not a chat model.
        """  # noqa: DOC502 — raised by _resolve_context
        from deepagents_code.config import use_environment
        from deepagents_code.model_retry import aretry_model_call

        with use_environment(self._environ):
            model, system, settings = await self._resolve_context(thread_id, state)
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
            prepared = _prepare_messages(
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
            )
            config: RunnableConfig = {
                "callbacks": [],
                "metadata": {"thread_id": thread_id},
            }
            if on_text is None:
                response = await aretry_model_call(
                    model,
                    lambda: model.ainvoke(prepared, config=config, **settings),
                    max_total_delay=_MAX_RETRY_DELAY,
                )
                text = response.text
            else:
                text = await _stream_answer(model, prepared, config, settings, on_text)
        return (
            text.strip() or "No text answer was returned. Try rephrasing your question."
        )
