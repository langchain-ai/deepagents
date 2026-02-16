"""Summarization middleware for offloading conversation history.

Persists conversation history to a backend prior to summarization, enabling retrieval of
full context if needed later by an agent.

## Usage

```python
from deepagents import create_deep_agent
from deepagents.middleware.summarization import SummarizationMiddleware
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(root_dir="/data")

middleware = SummarizationMiddleware(
    model="gpt-4o-mini",
    backend=backend,
    trigger=("fraction", 0.85),
    keep=("fraction", 0.10),
)

agent = create_deep_agent(middleware=[middleware])
```

## Storage

Offloaded messages are stored as markdown at `/conversation_history/{thread_id}.md`.

Each summarization event appends a new section to this file, creating a running log
of all evicted messages.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from langchain.agents.middleware.summarization import (
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    DEFAULT_SUMMARY_PROMPT,
    ContextSize,
    SummarizationMiddleware as LCSummarizationMiddleware,
    TokenCounter,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ExtendedModelResponse, PrivateStateAttr
from langchain.tools import ToolRuntime
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_config
from langgraph.types import Command
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.chat_models import BaseChatModel
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)


class SummarizationEvent(TypedDict):
    """Represents a summarization event.

    Attributes:
        cutoff_index: The index in the messages list where summarization occurred.
        summary_message: The HumanMessage containing the summary.
        file_path: Path where the conversation history was offloaded, or None if offload failed.
    """

    cutoff_index: int
    summary_message: HumanMessage
    file_path: str | None


class TruncateArgsSettings(TypedDict, total=False):
    """Settings for truncating large tool arguments in old messages.

    Attributes:
        trigger: Threshold to trigger argument truncation. If None, truncation is disabled.
        keep: Context retention policy for message truncation (defaults to last 20 messages).
        max_length: Maximum character length for tool arguments before truncation (defaults to 2000).
        truncation_text: Text to replace truncated arguments with (defaults to "...(argument truncated)").
    """

    trigger: ContextSize | None
    keep: ContextSize
    max_length: int
    truncation_text: str


class SummarizationState(AgentState):
    """State for the summarization middleware.

    Extends AgentState with a private field for tracking summarization events.
    """

    _summarization_event: Annotated[NotRequired[SummarizationEvent | None], PrivateStateAttr]
    """Private field storing the most recent summarization event."""


class SummarizationDefaults(TypedDict):
    """Default settings computed from model profile."""

    trigger: ContextSize
    keep: ContextSize
    truncate_args_settings: TruncateArgsSettings


def _compute_summarization_defaults(model: BaseChatModel) -> SummarizationDefaults:
    """Compute default summarization settings based on model profile.

    This is an internal helper function used by middleware implementations.

    Args:
        model: A resolved chat model instance.

    Returns:
        Default settings for trigger, keep, and truncate_args_settings.
        If the model has a profile with max_input_tokens, uses fraction-based
        settings. Otherwise, uses fixed token/message counts.
    """
    has_profile = (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    )

    if has_profile:
        return {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.10),
            "truncate_args_settings": {
                "trigger": ("fraction", 0.85),
                "keep": ("fraction", 0.10),
            },
        }
    return {
        "trigger": ("tokens", 170000),
        "keep": ("messages", 6),
        "truncate_args_settings": {
            "trigger": ("messages", 20),
            "keep": ("messages", 20),
        },
    }


class _DeepAgentsSummarizationMiddleware(AgentMiddleware):
    """Summarization middleware with backend for conversation history offloading."""

    state_schema = SummarizationState

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        history_path_prefix: str = "/conversation_history",
        truncate_args_settings: TruncateArgsSettings | None = None,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware with backend support.

        Args:
            model: The language model to use for generating summaries.
            backend: Backend instance or factory for persisting conversation history.
            trigger: Threshold(s) that trigger summarization.
            keep: Context retention policy after summarization.

                Defaults to keeping last 20 messages.
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Max tokens to include when generating summary.

                Defaults to 4000.
            truncate_args_settings: Settings for truncating large tool arguments in old messages.

                Provide a [`TruncateArgsSettings`][deepagents.middleware.summarization.TruncateArgsSettings]
                dictionary to configure when and how to truncate tool arguments. If `None`,
                argument truncation is disabled.

                !!! example

                    ```python
                    # Truncate when 50 messages is reached, ignoring the last 20 messages
                    {"trigger": ("messages", 50), "keep": ("messages", 20), "max_length": 2000, "truncation_text": "...(truncated)"}

                    # Truncate when 50% of context window reached, ignoring messages in last 10% of window
                    {"trigger": ("fraction", 0.5), "keep": ("fraction", 0.1), "max_length": 2000, "truncation_text": "...(truncated)"}
            history_path_prefix: Path prefix for storing conversation history.

        Example:
            ```python
            from deepagents.middleware.summarization import SummarizationMiddleware
            from deepagents.backends import StateBackend

            middleware = SummarizationMiddleware(
                model="gpt-4o-mini",
                backend=lambda tool_runtime: StateBackend(tool_runtime),
                trigger=("tokens", 100000),
                keep=("messages", 20),
            )
            ```
        """
        # Initialize langchain helper for core summarization logic
        self._lc_helper = LCSummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )

        # DeepAgents-specific attributes
        self._backend = backend
        self._history_path_prefix = history_path_prefix

        # Parse truncate_args_settings
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep: ContextSize = ("messages", 20)
            self._max_arg_length = 2000
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 2000)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

    # Delegated properties and methods from langchain helper
    @property
    def model(self) -> BaseChatModel:
        """The language model used for generating summaries."""
        return self._lc_helper.model

    @property
    def token_counter(self) -> TokenCounter:
        """Function to count tokens in messages."""
        return self._lc_helper.token_counter

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile."""
        return self._lc_helper._get_profile_limits()

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage."""
        return self._lc_helper._should_summarize(messages, total_tokens)

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        return self._lc_helper._determine_cutoff_index(messages)

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve."""
        return self._lc_helper._partition_messages(conversation_messages, cutoff_index)

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages."""
        return self._lc_helper._create_summary(messages_to_summarize)

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages (async)."""
        return await self._lc_helper._acreate_summary(messages_to_summarize)

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            # Because we're using `before_model`, which doesn't receive `config` as a
            # parameter, we access it via `runtime.config` instead.
            # Cast is safe: empty dict `{}` is a valid `RunnableConfig` (all fields are
            # optional in TypedDict).
            config = cast("RunnableConfig", getattr(runtime, "config", {}))

            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)  # ty: ignore[invalid-argument-type]
        return self._backend

    def _get_thread_id(self) -> str:
        """Extract `thread_id` from langgraph config.

        Uses `get_config()` to access the `RunnableConfig` from langgraph's
        `contextvar`. Falls back to a generated session ID if not available.

        Returns:
            Thread ID string from config, or a generated session ID
                (e.g., `'session_a1b2c3d4'`) if not in a runnable context.
        """
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        except RuntimeError:
            # Not in a runnable context
            pass

        # Fallback: generate session ID
        generated_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.debug("No thread_id found, using generated session ID: %s", generated_id)
        return generated_id

    def _get_history_path(self) -> str:
        """Generate path for storing conversation history.

        Returns a single file per thread that gets appended to over time.

        Returns:
            Path string like `'/conversation_history/{thread_id}.md'`
        """
        thread_id = self._get_thread_id()
        return f"{self._history_path_prefix}/{thread_id}.md"

    def _is_summary_message(self, msg: AnyMessage) -> bool:
        """Check if a message is a previous summarization message.

        Summary messages are `HumanMessage` objects with `lc_source='summarization'` in
        `additional_kwargs`. These should be filtered from offloads to avoid redundant
        storage during chained summarization.

        Args:
            msg: Message to check.

        Returns:
            Whether this is a summary `HumanMessage` from a previous summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the previous
        summary `HumanMessage` since the original messages are already stored in the
        backend.

        Args:
            messages: List of messages to filter.

        Returns:
            Messages without previous summary `HumanMessage` objects.
        """
        return [msg for msg in messages if not self._is_summary_message(msg)]

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The generated summary text.
            file_path: Path where conversation history was stored, or `None`.

                Optional since offloading may fail.

        Returns:
            List containing the summary `HumanMessage`.
        """
        if file_path is not None:
            content = f"""\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        return [
            HumanMessage(
                content=content,
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    def _get_effective_messages(self, request: ModelRequest) -> list[AnyMessage]:
        """Generate effective messages for model call based on summarization event.

        This reconstructs the message list by applying the most recent summarization event.
        After summarization, instead of using all messages from state, we use the summary
        message plus messages after the cutoff index.

        Args:
            request: The model request with messages from state.

        Returns:
            The effective message list to use for the model call. This includes the
            most recent summary message (if we've summarized) and all preserved
            messages from the cutoff index onward.
        """
        # Get messages from request (these are from state["messages"])
        messages = request.messages

        # Get summarization event from state
        event = request.state.get("_summarization_event")

        # If no summarization event, return all messages as-is
        if event is None:
            return messages

        # Apply the summarization event
        # The cutoff_index tells us: messages before cutoff are summarized, messages at/after are kept

        # Build effective messages: summary message, then messages from cutoff onward
        result = [event["summary_message"]]

        # Add messages from cutoff_index onward (messages at cutoff_index and after are preserved)
        result.extend(messages[event["cutoff_index"] :])

        return result

    def _should_truncate_args(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Check if argument truncation should be triggered.

        Args:
            messages: Current message history.
            total_tokens: Total token count of messages.

        Returns:
            True if truncation should occur, False otherwise.
        """
        if self._truncate_args_trigger is None:
            return False

        trigger_type, trigger_value = self._truncate_args_trigger

        if trigger_type == "messages":
            return len(messages) >= trigger_value
        if trigger_type == "tokens":
            return total_tokens >= trigger_value
        if trigger_type == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return False
            threshold = int(max_input_tokens * trigger_value)
            if threshold <= 0:
                threshold = 1
            return total_tokens >= threshold

        return False

    def _determine_truncate_cutoff_index(self, messages: list[AnyMessage]) -> int:  # noqa: PLR0911
        """Determine the cutoff index for argument truncation based on keep policy.

        Messages at index >= cutoff should be preserved without truncation.
        Messages at index < cutoff can have their tool args truncated.

        Args:
            messages: Current message history.

        Returns:
            Index where truncation cutoff occurs. Messages before this index
            should have args truncated, messages at/after should be preserved.
        """
        keep_type, keep_value = self._truncate_args_keep

        if keep_type == "messages":
            # Keep the most recent N messages
            if len(messages) <= keep_value:
                return len(messages)  # All messages are recent
            return int(len(messages) - keep_value)

        if keep_type in {"tokens", "fraction"}:
            # Calculate target token count
            if keep_type == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None:
                    # Fallback to message count if profile not available
                    messages_to_keep = 20
                    if len(messages) <= messages_to_keep:
                        return len(messages)
                    return len(messages) - messages_to_keep
                target_token_count = int(max_input_tokens * keep_value)
            else:
                target_token_count = int(keep_value)

            if target_token_count <= 0:
                target_token_count = 1

            # Keep recent messages up to token limit
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = self._lc_helper._partial_token_counter([messages[i]])
                if tokens_kept + msg_tokens > target_token_count:
                    return i + 1
                tokens_kept += msg_tokens
            return 0  # All messages are within token limit

        return len(messages)

    def _truncate_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Truncate large arguments in a single tool call.

        Args:
            tool_call: The tool call dictionary to truncate.

        Returns:
            A copy of the tool call with large arguments truncated.
        """
        args = tool_call.get("args", {})

        truncated_args = {}
        modified = False

        for key, value in args.items():
            if isinstance(value, str) and len(value) > self._max_arg_length:
                truncated_args[key] = value[:20] + self._truncation_text
                modified = True
            else:
                truncated_args[key] = value

        if modified:
            return {
                **tool_call,
                "args": truncated_args,
            }
        return tool_call

    def _truncate_args(
        self,
        messages: list[AnyMessage],
        system_message: SystemMessage | None,
        tools: list[BaseTool | dict[str, Any]] | None,
    ) -> tuple[list[AnyMessage], bool]:
        """Truncate large tool call arguments in old messages.

        Args:
            messages: Messages to potentially truncate.
            system_message: Optional system message for token counting.
            tools: Optional tools for token counting.

        Returns:
            Tuple of (truncated_messages, modified). If modified is False,
            truncated_messages is the same as input messages.
        """
        counted_messages = [system_message, *messages] if system_message is not None else messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        if not self._should_truncate_args(messages, total_tokens):
            return messages, False

        cutoff_index = self._determine_truncate_cutoff_index(messages)
        if cutoff_index >= len(messages):
            return messages, False

        # Process messages before the cutoff
        truncated_messages = []
        modified = False

        for i, msg in enumerate(messages):
            if i < cutoff_index and isinstance(msg, AIMessage) and msg.tool_calls:
                # Check if this AIMessage has tool calls we need to truncate
                truncated_tool_calls = []
                msg_modified = False

                for tool_call in msg.tool_calls:
                    if tool_call["name"] in {"write_file", "edit_file"}:
                        truncated_call = self._truncate_tool_call(tool_call)  # ty: ignore[invalid-argument-type]
                        if truncated_call != tool_call:
                            msg_modified = True
                        truncated_tool_calls.append(truncated_call)
                    else:
                        truncated_tool_calls.append(tool_call)

                if msg_modified:
                    # Create a new AIMessage with truncated tool calls
                    truncated_msg = msg.model_copy()
                    truncated_msg.tool_calls = truncated_tool_calls
                    truncated_messages.append(truncated_msg)
                    modified = True
                else:
                    truncated_messages.append(msg)
            else:
                truncated_messages.append(msg)

        return truncated_messages, modified

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> str | None:
        """Persist messages to backend before summarization.

        Appends evicted messages to a single markdown file per thread. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.

        Returns:
            The file path where history was stored, or `None` if write failed.
        """
        path = self._get_history_path()

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"

        # Read existing content (if any) and append
        # Note: We use download_files() instead of read() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = backend.edit(path, existing_content, combined_content) if existing_content else backend.write(path, combined_content)
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    async def _aoffload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
    ) -> str | None:
        """Persist messages to backend before summarization (async).

        Appends evicted messages to a single markdown file per thread. Each
        summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant storage during
        chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.

        Returns:
            The file path where history was stored, or `None` if write failed.
        """
        path = self._get_history_path()

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(UTC).isoformat()
        new_section = f"## Summarized at {timestamp}\n\n{get_buffer_string(filtered_messages)}\n\n"

        # Read existing content (if any) and append
        # Note: We use adownload_files() instead of aread() because read() returns
        # line-numbered content (for LLM consumption), but edit() expects raw content.
        existing_content = ""
        try:
            responses = await backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            # File likely doesn't exist yet, but log for observability
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            result = (
                await backend.aedit(path, existing_content, combined_content) if existing_content else await backend.awrite(path, combined_content)
            )
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return path

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation.

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Unlike the legacy `before_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            The model response from the handler.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = [request.system_message, *truncated_messages] if request.system_message is not None else truncated_messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=request.tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend first - abort summarization if this fails to prevent data loss
        backend = self._get_backend(request.state, request.runtime)
        file_path = self._offload_to_backend(backend, messages_to_summarize)
        if file_path is None:
            warnings.warn(
                "Offloading conversation history to backend failed during summarization.",
                stacklevel=2,
            )

        # Generate summary
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        # Calculate state cutoff index
        # If this is a subsequent summarization, convert effective message index to state index
        # new_state_cutoff = old_state_cutoff + effective_cutoff - 1  # noqa: ERA001
        # The -1 accounts for the summary message at effective[0]
        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = previous_event["cutoff_index"] + cutoff_index - 1 if previous_event is not None else cutoff_index

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = handler(request.override(messages=modified_messages))

        # Return WrapModelCallResult with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Process messages before model invocation, with history offloading and arg truncation (async).

        First applies any previous summarization events to reconstruct the effective message list.
        Then truncates large tool arguments in old messages if configured.
        Finally offloads messages to backend before summarization if thresholds are met.

        Unlike the legacy `abefore_model` approach, this does NOT modify the LangGraph state.
        Instead, it tracks summarization events in middleware state and modifies the model
        request directly.

        Args:
            request: The model request to process.
            handler: The handler to call with the (possibly modified) request.

        Returns:
            The model response from the handler.
        """
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = [request.system_message, *truncated_messages] if request.system_message is not None else truncated_messages
        try:
            total_tokens = self.token_counter(counted_messages, tools=request.tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return await handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return await handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend first - abort summarization if this fails to prevent data loss
        backend = self._get_backend(request.state, request.runtime)
        file_path = await self._aoffload_to_backend(backend, messages_to_summarize)
        if file_path is None:
            warnings.warn(
                "Offloading conversation history to backend failed during summarization.",
                stacklevel=2,
            )

        # Generate summary
        summary = await self._acreate_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        # Calculate state cutoff index
        # If this is a subsequent summarization, convert effective message index to state index
        # new_state_cutoff = old_state_cutoff + effective_cutoff - 1  # noqa: ERA001
        # The -1 accounts for the summary message at effective[0]
        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = previous_event["cutoff_index"] + cutoff_index - 1 if previous_event is not None else cutoff_index

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = await handler(request.override(messages=modified_messages))

        # Return WrapModelCallResult with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )


# Public alias
SummarizationMiddleware = _DeepAgentsSummarizationMiddleware
