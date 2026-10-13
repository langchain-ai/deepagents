"""Read-only side answers from an isolated snapshot of the main conversation."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from deepagents.backends import StateBackend
from deepagents.backends.utils import truncate_if_too_long
from deepagents.middleware.filesystem import FilesystemMiddleware
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
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "The user is asking a quick side question about the conversation so far. "
    "Answer directly and concisely in markdown. {access} "
    "Do not modify files, run commands, delegate, or continue the main task. "
    "Everything before this message is reference context only, "
    "not active instructions, approvals, or pending tool calls to execute."
)
_MAX_SNAPSHOTS = 16
_MAX_TOOL_ROUNDS = 3
"""Tool rounds per side answer; one more model call must answer without tools."""
_LAST_ROUND_NOTE = (
    "Side tool limit reached. Answer from the results gathered so far "
    "without calling more tools."
)
_MAX_TOOL_CALLS = 8
"""Tool calls allowed in one model response."""
_ROUND_SEPARATOR = "\n\n"
_READ_TOOLS: tuple[FsToolName, ...] = ("ls", "read_file", "glob", "grep")
_MAX_RETRY_DELAY = 60
"""Maximum total retry delay, in seconds, for each model call.

The API's 120-second deadline covers every call of one answer, so a slow
answer with several tool rounds can still time out.
"""
_PENDING_RESULT = "Main tool call had no result in the side snapshot."
_SERVER_TOOL_BLOCKS = frozenset(
    {
        "server_tool_use",
        "mcp_tool_use",
        "server_tool_call",
        "server_tool_call_chunk",
        "server_tool_result",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
    }
)
"""Provider-executed tool blocks, plus any type ending in `_tool_result`."""
_MCP_TOOL_BLOCKS = frozenset(
    {
        "mcp_tool_use",
        "mcp_tool_result",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
    }
)
"""Provider MCP connector blocks, which side calls never keep."""
_MCP_TOOL_TYPES = frozenset({"mcp", "mcp_toolset"})
"""Provider MCP connector tools, which can change remote state."""
_EXTRAS_KEY = "_dcode_extras"
"""Snapshot key for the `BaseTool.extras` that schema conversion drops."""
_OMITTED_SERVER_TOOLS = "[Provider tool output omitted from the side snapshot.]"
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
        The owned stream and the response through its first visible text, or
        the complete response, such as tool calls only, when it has no text.
    """
    # Chat model and binding `astream` are async generators, annotated as iterators.
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
    *,
    prefix: str = "",
) -> AIMessageChunk:
    """Retry before visible output; never replay an answer already being shown.

    `prefix` starts the first visible fragment, such as a round separator.

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
            await on_text(prefix + first.text)
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


def _is_server_tool_block(block: object, *, provider_tools: bool) -> bool:
    """Check whether a content block records a tool the side request lacks.

    Returns:
        `True` for MCP connector blocks, and for other server tool calls and
        results when the side request defines no provider tools.
    """
    kind = block.get("type") if isinstance(block, dict) else None
    if not isinstance(kind, str):
        return False
    if kind in _MCP_TOOL_BLOCKS:
        return True
    return not provider_tools and (
        kind in _SERVER_TOOL_BLOCKS or kind.endswith("_tool_result")
    )


def _without_server_tools(message: AIMessage, *, provider_tools: bool) -> AIMessage:
    """Drop provider-executed tool blocks that the side request cannot define.

    Returns:
        `message`, or a copy without those blocks.
    """
    if not isinstance(message.content, list) or not any(
        _is_server_tool_block(block, provider_tools=provider_tools)
        for block in message.content
    ):
        return message
    content = [
        block
        for block in message.content
        if not _is_server_tool_block(block, provider_tools=provider_tools)
    ]
    if not content and not message.tool_calls:
        content = [{"type": "text", "text": _OMITTED_SERVER_TOOLS}]
    return message.model_copy(update={"content": content})


def _tool_free_transcript(messages: Sequence[BaseMessage]) -> list[AnyMessage]:
    """Render tool calls and results as text for a request that defines no tools.

    Returns:
        Messages without tool blocks; other messages are unchanged.
    """
    transcript: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            transcript.append(_tool_result_context(message))
        elif isinstance(message, AIMessage) and (
            message.tool_calls or message.invalid_tool_calls
        ):
            calls = [
                f"[tool call {call['id']}: {call['name']}]\n"
                + json.dumps(call["args"], ensure_ascii=False)
                for call in message.tool_calls
            ] + [
                f"[tool call {call['id']}: {call['name'] or 'tool'}]\n{call['args']}"
                for call in message.invalid_tool_calls
            ]
            text = "\n\n".join(part for part in [message.text, *calls] if part)
            transcript.append(AIMessage(content=text))
        elif isinstance(message, AIMessage):
            transcript.append(_without_server_tools(message, provider_tools=False))
        else:
            transcript.append(cast("AnyMessage", message))
    return transcript


def _structured_transcript(
    messages: Sequence[BaseMessage], *, provider_tools: bool
) -> list[AnyMessage]:
    """Pair every tool call that has an id with a result.

    Calls without a result get a placeholder result, and results without a
    call become labeled human context. Server tool blocks stay only when
    `provider_tools` says the request defines their tools.

    Returns:
        Messages that providers accept as a tool transcript.
    """
    transcript: list[AnyMessage] = []
    pending: dict[str, str] = {}
    for original in messages:
        message = (
            _without_server_tools(original, provider_tools=provider_tools)
            if isinstance(original, AIMessage)
            else original
        )
        if isinstance(message, ToolMessage):
            if pending.pop(message.tool_call_id, None) is None:
                transcript.append(_tool_result_context(message))
                continue
        else:
            transcript.extend(_placeholders(pending))
            pending.clear()
        if isinstance(message, AIMessage):
            # Providers send unparsable calls too, so they need results as well.
            pending.update(
                {
                    call["id"]: call["name"] or "tool"
                    for call in [*message.tool_calls, *message.invalid_tool_calls]
                    if call["id"] is not None
                }
            )
        transcript.append(cast("AnyMessage", message))
    transcript.extend(_placeholders(pending))
    return transcript


def _placeholders(pending: Mapping[str, str]) -> list[ToolMessage]:
    """Answer main tool calls that have no result in the snapshot.

    Returns:
        One placeholder result per pending call id.
    """
    return [
        ToolMessage(content=_PENDING_RESULT, tool_call_id=call_id, name=name)
        for call_id, name in pending.items()
    ]


def _bound_result(message: ToolMessage) -> ToolMessage:
    """Cap a side tool result at the main eviction threshold.

    Side reads skip the main tool wrapper, so a large result would otherwise
    enter the next request in full and overflow its budget.

    Returns:
        `message`, or a copy with its text cut to the threshold.
    """
    if not isinstance(message.content, str):
        return message
    content = truncate_if_too_long(message.content)
    if content is message.content:
        return message
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


def _is_mcp_tool(tool: object) -> bool:
    """Check whether a tool is a provider MCP connector.

    Returns:
        `True` for MCP connector tools, as dicts or tool definitions.
    """
    extras = getattr(tool, "extras", None)
    if isinstance(extras, dict):
        tool = extras.get("provider_tool_definition")
    return isinstance(tool, dict) and tool.get("type") in _MCP_TOOL_TYPES


def _is_provider_tool(tool: object) -> bool:
    """Check whether the provider, not the side call, executes a tool.

    Returns:
        `True` for provider tool dicts and tools that wrap their definition.
    """
    if isinstance(tool, dict):
        return tool.get("type") != "function"
    extras = getattr(tool, "extras", None)
    return isinstance(extras, dict) and "provider_tool_definition" in extras


def _tool_name(tool: object) -> str | None:
    """Name a tool the way the SDK's tool exclusion does.

    Returns:
        The tool's name, or `None` when it has none.
    """
    name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _client_tools(tools: Sequence[object]) -> list[dict[str, Any]]:
    """Keep the main call's tools as isolated, checkpointable definitions.

    Provider tools stay so the side request matches the main prompt cache,
    except MCP connectors, which could change remote state.

    Returns:
        Tool definitions in the main call's order.
    """
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schemas: list[dict[str, Any]] = []
    for tool in tools:
        if _is_mcp_tool(tool):
            continue
        if isinstance(tool, BaseTool):
            schema = convert_to_openai_tool(tool)
            if tool.extras:
                # Conversion drops extras such as `defer_loading`.
                schema[_EXTRAS_KEY] = deepcopy(tool.extras)
            schemas.append(schema)
        elif isinstance(tool, dict):
            schemas.append(deepcopy(cast("dict[str, Any]", tool)))
    return schemas


def _bindable(
    schema: dict[str, Any], *, stub: bool = False
) -> dict[str, Any] | BaseTool:
    """Rebuild a tool with saved extras so providers bind them as before.

    `stub` forces a tool stub, which prompt caching needs to tag a tool.

    Returns:
        `schema`, or a tool stub that carries its extras.
    """
    from langchain_core.tools import StructuredTool

    extras = schema.get(_EXTRAS_KEY)
    function = schema.get("function")
    if not isinstance(function, dict) or not (stub or isinstance(extras, dict)):
        return schema
    return StructuredTool(
        name=function["name"],
        description=function.get("description", ""),
        args_schema=function.get("parameters", {}),
        func=_unused_tool,
        extras=extras if isinstance(extras, dict) else None,
    )


def _unused_tool(**_: object) -> None:
    """Never runs: side calls execute only the main read tools."""


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
    # Fail rather than drop or shorten the user's question to make it fit.
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
        A transcript that starts with the system prompt, sized for the resolved
        model and output settings. It keeps tool blocks only when the request
        defines tools.
    """
    request = UnsupportedContentMiddleware()._filter_request(request)
    compaction = create_summarization_middleware(request.model, StateBackend())
    messages, _ = compaction._truncate_args(
        request.messages,
        compaction._count_tokens(
            request.messages, request.system_message, request.tools
        ),
    )
    if not request.tools:
        # Providers reject tool blocks in a request that defines no tools.
        flat = _tool_free_transcript(messages)
        return cast(
            "list[AnyMessage]",
            _fit_context(request.override(messages=flat), compaction),
        )
    provider_tools = any(_is_provider_tool(tool) for tool in request.tools)
    structured = _structured_transcript(messages, provider_tools=provider_tools)
    return _structured_transcript(
        _fit_context(request.override(messages=structured), compaction),
        provider_tools=provider_tools,
    )


def _side_messages(
    state: Mapping[str, object],
    history: Sequence[tuple[str, str]],
    question: str,
    tool_names: Sequence[str],
    *,
    provider_tools: bool,
) -> list[AnyMessage]:
    """Place the side history and instructed question after the main transcript.

    Returns:
        Messages for the first side request, before fitting.
    """
    return [
        # Copy once here; later steps copy only the messages they change.
        *_structured_transcript(
            [message.model_copy(deep=True) for message in _conversation(state)],
            provider_tools=provider_tools,
        ),
        *(
            message
            for prompt, answer in history
            for message in (HumanMessage(content=prompt), AIMessage(content=answer))
        ),
        HumanMessage(content=f"{_instructions(tool_names)}\n\n{question}"),
    ]


def _with_prompt_caching(request: ModelRequest) -> ModelRequest:
    """Tag `request` as the SDK's Anthropic prompt caching middleware would.

    Returns:
        The tagged request, or `request` for other providers.
    """
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

    tagged = [request]

    def keep(request: ModelRequest) -> ModelResponse:
        tagged[0] = request
        return ModelResponse(result=[])

    caching = AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    caching.wrap_model_call(request, keep)
    return tagged[0]


@dataclass(frozen=True)
class _SideRequest:
    """Resolved settings shared by every model call of one side answer."""

    model: BaseChatModel
    system: SystemMessage
    settings: dict[str, Any]
    tools: list[dict[str, Any] | BaseTool]
    config: RunnableConfig

    @classmethod
    def cached(
        cls,
        model: BaseChatModel,
        system: SystemMessage,
        settings: dict[str, Any],
        tools: Sequence[dict[str, Any]],
        config: RunnableConfig,
    ) -> _SideRequest:
        """Add the cache breakpoints that the main call's prompt caching adds.

        `BtwOperation` captures the main request before the SDK's prompt
        caching middleware tags it, so an untagged side call reads no cache.

        Returns:
            A request that can read the main call's cached prefix.
        """
        bindable = [_bindable(tool) for tool in tools]
        request = ModelRequest(
            model=model,
            system_message=system,
            messages=[],
            # Caching tags only a tool stub, so the last tool must be one.
            tools=[
                *bindable[:-1],
                *(_bindable(tool, stub=True) for tool in tools[-1:]),
            ],
            model_settings=settings,
            runtime=Runtime(),
        )
        tagged = _with_prompt_caching(request)
        if tagged is request:
            return cls(model, system, settings, bindable, config)
        system = tagged.system_message or system
        return cls(model, system, tagged.model_settings, tagged.tools, config)

    def prepare(self, messages: Sequence[BaseMessage]) -> list[AnyMessage]:
        """Fit `messages` and the system prompt to the model's input budget.

        Returns:
            Messages that start with the system prompt.
        """
        return _prepare_messages(
            ModelRequest(
                model=self.model,
                system_message=self.system,
                messages=cast("list[AnyMessage]", [*messages]),
                tools=[*self.tools],
                model_settings=self.settings,
                runtime=Runtime(),
            )
        )

    async def generate(
        self,
        messages: Sequence[BaseMessage],
        on_text: Callable[[str], Awaitable[None]] | None,
        *,
        prefix: str = "",
    ) -> AIMessage:
        """Call the model once, streaming text to `on_text` when given.

        Returns:
            The complete model response, including any tool calls.
        """
        from deepagents_code.model_retry import aretry_model_call

        bound = self.model.bind_tools(self.tools) if self.tools else self.model
        if on_text is not None:
            return await _stream_answer(
                bound,
                messages,
                self.config,
                self.settings,
                on_text,
                self.model,
                prefix=prefix,
            )
        return await aretry_model_call(
            self.model,
            lambda: bound.ainvoke(messages, config=self.config, **self.settings),
            max_total_delay=_MAX_RETRY_DELAY,
        )


def _blocked_reason(
    response: AIMessage, round_index: int, *, can_run: bool
) -> str | None:
    """Explain why the side call must not run this response's tool calls.

    Returns:
        A message for the user, or `None` when the calls may run.
    """
    if response.invalid_tool_calls:
        # Running only the valid calls would leave the others without results.
        logger.warning(
            "Side model sent unparsable tool calls: %s",
            [call.get("name") for call in response.invalid_tool_calls],
        )
        return "The model sent a tool call that could not be read. Try again."
    if not can_run:
        return (
            "File reads are off for this side question. "
            "Ask about the conversation instead."
        )
    if round_index == _MAX_TOOL_ROUNDS:
        return "Side tool limit reached. Try a narrower question."
    if len(response.tool_calls) > _MAX_TOOL_CALLS:
        return "Too many side tools requested. Try a narrower question."
    return None


async def _append_note(
    note: str,
    parts: list[str],
    on_text: Callable[[str], Awaitable[None]] | None,
) -> None:
    """Tell the user why the answer stops after the model's lead-in text."""
    logger.info("Side tool calls skipped: %s", note)
    text = f"_{note}_"
    parts.append(text)
    if on_text is not None:
        await on_text(_ROUND_SEPARATOR + text)


def _tool_messages(result: object) -> list[BaseMessage]:
    """Collect tool results from either `ToolNode` output shape.

    `ToolNode` returns a list instead of a dict when any tool returns a
    `Command`, as `read_file` does for images and other media.

    Returns:
        Tool result messages, in call order.
    """
    messages: list[BaseMessage] = []
    for output in result if isinstance(result, list) else [result]:
        update = output.update if isinstance(output, Command) else output
        batch = update.get("messages", []) if isinstance(update, Mapping) else update
        if isinstance(batch, list):
            messages.extend(item for item in batch if isinstance(item, BaseMessage))
    return messages


def _round_results(
    messages: Sequence[BaseMessage], round_index: int
) -> list[BaseMessage]:
    """Bound tool results and tell the model when no tool round is left.

    Returns:
        Tool results to add to the next request.
    """
    results: list[BaseMessage] = [
        _bound_result(message) if isinstance(message, ToolMessage) else message
        for message in messages
    ]
    if round_index == _MAX_TOOL_ROUNDS - 1:
        # Media reads follow their tool result with a human message.
        last = next(
            (
                i
                for i in range(len(results) - 1, -1, -1)
                if isinstance(results[i], ToolMessage)
            ),
            None,
        )
        if last is not None:
            results[last] = _with_note(
                cast("ToolMessage", results[last]), _LAST_ROUND_NOTE
            )
    return results


async def _run_rounds(
    request: _SideRequest,
    messages: list[AnyMessage],
    executable: Sequence[BaseTool],
    on_text: Callable[[str], Awaitable[None]] | None,
) -> list[str]:
    """Alternate model calls and read-only tool rounds until the model answers.

    Returns:
        Text from each model call, in order, plus a note when tool calls
        that came with text could not run.

    Raises:
        BtwLimitError: If tool calls cannot run and the model sent no text.
    """
    node = ToolNode(executable)
    parts: list[str] = []
    for round_index in range(_MAX_TOOL_ROUNDS + 1):
        response = await request.generate(
            messages, on_text, prefix=_ROUND_SEPARATOR if parts else ""
        )
        if response.text:
            parts.append(response.text)
        if not (response.tool_calls or response.invalid_tool_calls):
            break
        if reason := _blocked_reason(response, round_index, can_run=bool(executable)):
            if not response.text:
                raise BtwLimitError(reason)
            await _append_note(reason, parts, on_text)
            break
        result = await node.ainvoke(
            {"messages": [response]}, config=request.config, runtime=Runtime()
        )
        # `prepare` adds the system prompt back, so drop its first copy.
        messages = request.prepare(
            [
                *messages[1:],
                response,
                *_round_results(_tool_messages(result), round_index),
            ]
        )
    return parts


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
    It may run read-only filesystem tools directly, outside the agent's tool
    middleware.
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
        excluded_tools: frozenset[str] = frozenset(),
    ) -> None:
        """Keep workspace defaults and the read-only instruction loaders.

        Args:
            model: Workspace bootstrap model.
            system_prompt: Base instructions for a thread without a live snapshot.
            environ: Workspace environment for lazy model resolution.
            instruction_middleware: Main agent memory and skill loaders, in order.
            profile_overrides: Session profile fields to retain during reconstruction.
            backend: Main filesystem backend for read-only inspection, or
                `None` to turn side tools off.
            filesystem_tools: Main filesystem allowlist, or `None` for all read
                tools. Side tools stay off unless `read_file` is allowed.
            excluded_tools: Harness profile tool exclusions. The SDK removes
                these after this middleware captures the main request, so
                the capture removes them too.
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
        self._excluded_tools = excluded_tools
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
            self._tool_snapshots[info.thread_id] = _client_tools(
                [
                    tool
                    for tool in request.tools
                    if _tool_name(tool) not in self._excluded_tools
                ]
            )
            self._snapshots.move_to_end(info.thread_id)
            while len(self._snapshots) > _MAX_SNAPSHOTS:
                evicted, _ = self._snapshots.popitem(last=False)
                self._tool_snapshots.pop(evicted, None)
            update: dict[str, object] = {
                "_btw_system_prompt": (request.system_message or self._system).text,
                "_btw_model_settings": tool_free_settings(
                    request.model, request.model_settings
                ),
                "_btw_model_retries": _retry_budget(request.model),
            }
            tools = self._tool_snapshots[info.thread_id]
            # Tool schemas can be large, so checkpoint them only when they change.
            if (request.state or {}).get("_btw_tools") != tools:
                update["_btw_tools"] = deepcopy(tools)
            return Command(update=update)
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

    def _side_tools(
        self, thread_id: str, state: Mapping[str, object], *, read_tools: bool
    ) -> tuple[list[dict[str, Any]], list[BaseTool]]:
        """Advertise the main call's tools but run only reads it advertised.

        Returns:
            Tool schemas to bind and the read tools this side call may run.
        """
        schemas = (
            self._tool_snapshots.get(thread_id)
            if thread_id in self._snapshots
            else state.get("_btw_tools")
        )
        if isinstance(schemas, list):
            tools = _client_tools(schemas)
        else:
            tools = _client_tools(self._read_tools) if read_tools else []
        advertised = {tool.get("function", {}).get("name") for tool in tools}
        executable = [
            tool for tool in self._read_tools if read_tools and tool.name in advertised
        ]
        return tools, executable

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
        the checkpointed instructions, generation settings, retry budget, and
        tool schemas when no live snapshot exists.
        Bind the main call's tools, except MCP connectors, even though only
        reads run: providers need definitions for the tool calls in the
        transcript, and the same tools keep the main prompt cache valid.
        Provider tools such as web search can still run on the provider.
        New and legacy threads restore memory and skills through the main
        agent's loaders without writing updates back to the conversation.
        Large older file arguments are truncated using the main agent's policy;
        older context is dropped when needed to fit the side request's budget.
        Instructions, the question, and later tool rounds that cannot fit raise
        `ContextOverflowError`.

        Args:
            thread_id: Thread whose conversation supplies context.
            state: Read-only conversation snapshot.
            question: Side question to answer.
            history: Completed question/answer pairs from this side conversation.
            on_text: Optional receiver for text fragments as the model produces them.
            read_tools: Execute the main allowlist's read-only file tools that
                the main call also advertised. Side calls bypass the main tool
                middleware, including tool hooks, so enable this only when no
                hook could deny or audit a read.

        Returns:
            The ephemeral answer text.

        Raises:
            TypeError: If the configured model is not a chat model.
            BtwLimitError: If the model asks for tools that cannot run and
                sends no text: reads are off, a limit is hit, or a tool call
                cannot be parsed.
            ContextOverflowError: If the instructions, question, or a later
                tool round cannot fit the model's context.
        """  # noqa: DOC502 — raised by helpers
        from deepagents_code.config import use_environment

        with use_environment(self._environ):
            model, system, settings = await self._resolve_context(thread_id, state)
            tools, executable = self._side_tools(
                thread_id, state, read_tools=read_tools
            )
            request = _SideRequest.cached(
                model,
                system,
                settings,
                tools,
                {"callbacks": [], "metadata": {"thread_id": thread_id}},
            )
            messages = _side_messages(
                state,
                history,
                question,
                [tool.name for tool in executable],
                provider_tools=any(_is_provider_tool(tool) for tool in request.tools),
            )
            parts = await _run_rounds(
                request, request.prepare(messages), executable, on_text
            )
        text = _ROUND_SEPARATOR.join(parts).strip()
        if not text:
            # A content filter, thinking-only output, or a cut-off stream.
            logger.warning("Side answer for thread %s had no text", thread_id)
            return "No text answer was returned. Try rephrasing your question."
        return text
