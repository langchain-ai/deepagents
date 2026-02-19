"""Middleware that exposes a `compact_conversation` tool to the agent.

Allows the LLM to trigger conversation compaction on its own, using the same
summarization logic as the `/compact` slash command.  The tool updates
`_summarization_event` in state so the SDK's `SummarizationMiddleware`
sees compacted messages on the next model call, without removing messages
from state or requiring a UI refresh.
"""

import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain.tools import ToolRuntime
    from langchain_core.messages import AnyMessage, HumanMessage
    from langchain_core.tools import BaseTool
    from langgraph.types import Command

logger = logging.getLogger(__name__)


def _apply_event(
    messages: "list[AnyMessage]",
    event: "dict[str, Any] | None",
) -> "list[AnyMessage]":
    """Reconstruct effective messages from state messages and a summarization event.

    When a prior summarization event exists, the effective conversation is the
    summary message followed by all messages from `cutoff_index` onward.

    Args:
        messages: Full message list from state.
        event: The `_summarization_event` dict, or `None`.

    Returns:
        The effective message list the model would see.
    """
    if event is None:
        return list(messages)
    result: list[Any] = [event["summary_message"]]
    result.extend(messages[event["cutoff_index"] :])
    return result


def _offload(
    backend: "BackendProtocol",
    messages: "list[AnyMessage]",
    thread_id: str,
    *,
    history_path_prefix: str = "/conversation_history",
) -> str | None:
    """Write messages as markdown to backend storage.

    Follows the same append pattern as `SummarizationMiddleware._offload_to_backend`.

    Args:
        backend: Backend to write to.
        messages: Messages to persist.
        thread_id: Thread identifier for the history file.
        history_path_prefix: Path prefix for the history file.

    Returns:
        The file path where history was stored, or `None` on failure.
    """
    from datetime import UTC, datetime

    from langchain_core.messages import HumanMessage, get_buffer_string

    path = f"{history_path_prefix}/{thread_id}.md"

    # Filter out previous summary messages to avoid redundant storage
    filtered = [
        m
        for m in messages
        if not (
            isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == "summarization"
        )
    ]
    if not filtered:
        return None

    timestamp = datetime.now(UTC).isoformat()
    new_section = f"## Compacted at {timestamp}\n\n{get_buffer_string(filtered)}\n\n"

    existing_content = ""
    try:
        responses = backend.download_files([path])
        resp = responses[0] if responses else None
        if resp and resp.content is not None and resp.error is None:
            existing_content = resp.content.decode("utf-8")
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not read existing history at %s; treating as new file", path
        )

    combined = existing_content + new_section

    try:
        result = (
            backend.edit(path, existing_content, combined)
            if existing_content
            else backend.write(path, combined)
        )
        if result is None or result.error:
            logger.warning("Failed to offload compact history to %s", path)
            return None
    except Exception:
        logger.warning(
            "Exception offloading compact history to %s", path, exc_info=True
        )
        return None

    logger.debug("Offloaded %d messages to %s", len(filtered), path)
    return path


async def _aoffload(
    backend: "BackendProtocol",
    messages: "list[AnyMessage]",
    thread_id: str,
    *,
    history_path_prefix: str = "/conversation_history",
) -> str | None:
    """Async variant of `_offload`.

    Args:
        backend: Backend to write to.
        messages: Messages to persist.
        thread_id: Thread identifier for the history file.
        history_path_prefix: Path prefix for the history file.

    Returns:
        The file path where history was stored, or `None` on failure.
    """
    from datetime import UTC, datetime

    from langchain_core.messages import HumanMessage, get_buffer_string

    path = f"{history_path_prefix}/{thread_id}.md"

    filtered = [
        m
        for m in messages
        if not (
            isinstance(m, HumanMessage)
            and m.additional_kwargs.get("lc_source") == "summarization"
        )
    ]
    if not filtered:
        return None

    timestamp = datetime.now(UTC).isoformat()
    new_section = f"## Compacted at {timestamp}\n\n{get_buffer_string(filtered)}\n\n"

    existing_content = ""
    try:
        responses = await backend.adownload_files([path])
        resp = responses[0] if responses else None
        if resp and resp.content is not None and resp.error is None:
            existing_content = resp.content.decode("utf-8")
    except Exception:  # noqa: BLE001
        logger.debug(
            "Could not read existing history at %s; treating as new file", path
        )

    combined = existing_content + new_section

    try:
        result = (
            await backend.aedit(path, existing_content, combined)
            if existing_content
            else await backend.awrite(path, combined)
        )
        if result is None or result.error:
            logger.warning("Failed to offload compact history to %s", path)
            return None
    except Exception:
        logger.warning(
            "Exception offloading compact history to %s", path, exc_info=True
        )
        return None

    logger.debug("Offloaded %d messages to %s", len(filtered), path)
    return path


def _generate_summary(
    model: Any,  # noqa: ANN401
    messages: "list[AnyMessage]",
) -> str:
    """Generate a summary of messages using the model.

    Args:
        model: Chat model instance.
        messages: Messages to summarize.

    Returns:
        Summary text.
    """
    from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
    from langchain_core.messages import HumanMessage, get_buffer_string

    conversation_text = get_buffer_string(messages)
    prompt = DEFAULT_SUMMARY_PROMPT.format(conversation=conversation_text)
    response = model.invoke([HumanMessage(content=prompt)])
    return _extract_summary_text(response)


async def _agenerate_summary(
    model: Any,  # noqa: ANN401
    messages: "list[AnyMessage]",
) -> str:
    """Async variant of `_generate_summary`.

    Args:
        model: Chat model instance.
        messages: Messages to summarize.

    Returns:
        Summary text.
    """
    from langchain.agents.middleware.summarization import DEFAULT_SUMMARY_PROMPT
    from langchain_core.messages import HumanMessage, get_buffer_string

    conversation_text = get_buffer_string(messages)
    prompt = DEFAULT_SUMMARY_PROMPT.format(conversation=conversation_text)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return _extract_summary_text(response)


def _extract_summary_text(
    response: Any,  # noqa: ANN401
) -> str:
    """Extract text from a model response, handling content block lists.

    Args:
        response: The model response object.

    Returns:
        Extracted summary text.

    Raises:
        RuntimeError: If the response contains no usable text.
    """
    content = response.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)

    if not isinstance(content, str) or not content.strip():
        msg = "Model returned empty summary"
        raise RuntimeError(msg)

    return content.strip()


class CompactToolMiddleware(AgentMiddleware):
    """Middleware that exposes a `compact_conversation` tool.

    The tool triggers conversation compaction by generating a summary of older
    messages and storing the result in `_summarization_event`.  On the next
    model call the SDK's `SummarizationMiddleware` picks up the event and
    presents only the summary + recent messages.

    Args:
        backend: Backend for persisting offloaded conversation history.
        history_path_prefix: Path prefix for history files.
    """

    def __init__(
        self,
        backend: "BackendProtocol",
        *,
        history_path_prefix: str = "/conversation_history",
    ) -> None:
        """Initialize compact tool middleware.

        Args:
            backend: Backend for persisting offloaded conversation history.
            history_path_prefix: Path prefix for history files.
        """
        # Deferred import keeps CLI startup fast
        from deepagents.middleware.summarization import SummarizationState

        self.state_schema = SummarizationState
        self._backend = backend
        self._history_path_prefix = history_path_prefix
        self.tools: list[BaseTool] = [self._create_compact_tool()]

    def _create_compact_tool(self) -> "BaseTool":
        """Create the `compact_conversation` structured tool.

        Returns:
            A `StructuredTool` with both sync and async implementations.
        """
        # NOTE: ToolRuntime must be imported here (not under TYPE_CHECKING)
        # because StructuredTool.from_function resolves type hints at runtime
        # via typing.get_type_hints. The import is deferred to __init__ time
        # (not module load) to keep CLI startup fast.
        from langchain.tools import ToolRuntime
        from langchain_core.tools import StructuredTool
        from langgraph.types import Command as _Command

        backend = self._backend
        history_prefix = self._history_path_prefix

        def sync_compact(
            runtime: ToolRuntime,
        ) -> _Command:
            """Compact the conversation by summarizing older messages.

            Returns:
                A `Command` with state updates.
            """
            return _run_compact(runtime, backend, history_prefix)

        async def async_compact(
            runtime: ToolRuntime,
        ) -> _Command:
            """Compact the conversation by summarizing older messages (async).

            Returns:
                A `Command` with state updates.
            """
            return await _arun_compact(runtime, backend, history_prefix)

        return StructuredTool.from_function(
            name="compact_conversation",
            description=(
                "Compact the conversation by summarizing older messages "
                "into a concise summary. Use this proactively when the "
                "conversation is getting long to free up context window "
                "space. This tool takes no arguments."
            ),
            func=sync_compact,
            coroutine=async_compact,
        )


def _run_compact(
    runtime: "ToolRuntime",
    backend: "BackendProtocol",
    history_prefix: str,
) -> "Command":
    """Synchronous compact implementation.

    Args:
        runtime: The `ToolRuntime` injected by the tool node.
        backend: Backend for persisting conversation history.
        history_prefix: Path prefix for history files.

    Returns:
        A `Command` with `_summarization_event` state update, or a
        `Command` with a "not enough messages" `ToolMessage`.
    """
    from deepagents.middleware.summarization import (
        SummarizationEvent,
        _compute_summarization_defaults,  # noqa: PLC2701
    )
    from langchain.agents.middleware.summarization import (
        SummarizationMiddleware as LCSummarizationMiddleware,
    )
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from deepagents_cli.config import create_model

    messages = runtime.state.get("messages", [])
    event = runtime.state.get("_summarization_event")
    effective = _apply_event(messages, event)

    result = create_model()
    model = result.model

    defaults = _compute_summarization_defaults(model)
    mw = LCSummarizationMiddleware(model=model, keep=defaults["keep"])
    cutoff = mw._determine_cutoff_index(effective)

    if cutoff == 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Nothing to compact yet"
                            " \u2014 conversation is within"
                            " the token budget."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    to_summarize, _to_keep = mw._partition_messages(effective, cutoff)

    # Extract thread_id from runtime config
    thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

    file_path = _offload(
        backend,
        to_summarize,
        thread_id,
        history_path_prefix=history_prefix,
    )

    summary = _generate_summary(model, to_summarize)

    summary_msg = _build_summary_message(summary, file_path)

    # Calculate state cutoff index accounting for prior events
    state_cutoff = event["cutoff_index"] + cutoff - 1 if event is not None else cutoff

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }

    return Command(
        update={
            "_summarization_event": new_event,
            "messages": [
                ToolMessage(
                    content=(
                        "Conversation compacted. "
                        f"Summarized {len(to_summarize)} "
                        "messages into a concise summary."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


async def _arun_compact(
    runtime: "ToolRuntime",
    backend: "BackendProtocol",
    history_prefix: str,
) -> "Command":
    """Asynchronous compact implementation.

    Args:
        runtime: The `ToolRuntime` injected by the tool node.
        backend: Backend for persisting conversation history.
        history_prefix: Path prefix for history files.

    Returns:
        A `Command` with `_summarization_event` state update, or a
        `Command` with a "not enough messages" `ToolMessage`.
    """
    from deepagents.middleware.summarization import (
        SummarizationEvent,
        _compute_summarization_defaults,  # noqa: PLC2701
    )
    from langchain.agents.middleware.summarization import (
        SummarizationMiddleware as LCSummarizationMiddleware,
    )
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from deepagents_cli.config import create_model

    messages = runtime.state.get("messages", [])
    event = runtime.state.get("_summarization_event")
    effective = _apply_event(messages, event)

    result = create_model()
    model = result.model

    defaults = _compute_summarization_defaults(model)
    mw = LCSummarizationMiddleware(model=model, keep=defaults["keep"])
    cutoff = mw._determine_cutoff_index(effective)

    if cutoff == 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            "Nothing to compact yet"
                            " \u2014 conversation is within"
                            " the token budget."
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    to_summarize, _to_keep = mw._partition_messages(effective, cutoff)

    thread_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

    file_path = await _aoffload(
        backend,
        to_summarize,
        thread_id,
        history_path_prefix=history_prefix,
    )

    summary = await _agenerate_summary(model, to_summarize)

    summary_msg = _build_summary_message(summary, file_path)

    state_cutoff = event["cutoff_index"] + cutoff - 1 if event is not None else cutoff

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }

    return Command(
        update={
            "_summarization_event": new_event,
            "messages": [
                ToolMessage(
                    content=(
                        "Conversation compacted. "
                        f"Summarized {len(to_summarize)} "
                        "messages into a concise summary."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def _build_summary_message(
    summary: str,
    file_path: str | None,
) -> "HumanMessage":
    """Build the summary `HumanMessage` matching `SummarizationMiddleware` format.

    Args:
        summary: The generated summary text.
        file_path: Where history was offloaded, or `None`.

    Returns:
        A `HumanMessage` tagged with `lc_source='summarization'`.
    """
    from langchain_core.messages import HumanMessage

    if file_path is not None:
        content = (
            "You are in the middle of a conversation "
            "that has been summarized.\n\n"
            "The full conversation history has been "
            f"saved to {file_path} "
            "should you need to refer back to it "
            "for details.\n\n"
            "A condensed summary follows:\n\n"
            f"<summary>\n{summary}\n</summary>"
        )
    else:
        content = "Here is a summary of the conversation to date:\n\n" + summary

    return HumanMessage(
        content=content,
        additional_kwargs={"lc_source": "summarization"},
    )
