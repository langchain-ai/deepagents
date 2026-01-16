"""Summarization middleware with backend support for offloading conversation history.

This module provides a `SummarizationMiddleware` class that extends langchain's base
`SummarizationMiddleware` (imported as `BaseSummarizationMiddleware`) to persist
conversation history to a backend before summarization, enabling retrieval of full
context when needed by an agent.

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

## Storage Format

Offloaded messages are stored as markdown at `/conversation_history/{thread_id}.md`.

Each summarization event appends a new section to this file, creating a running log
of all evicted messages.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict

from langchain.agents.middleware.summarization import (
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    DEFAULT_SUMMARY_PROMPT,
    ContextSize,
    SummarizationMiddleware as BaseSummarizationMiddleware,
    TokenCounter,
    count_tokens_approximately,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, get_buffer_string
from langgraph.config import get_config
from langgraph.types import Overwrite
from typing_extensions import override

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState
    from langchain.chat_models import BaseChatModel
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)


class TruncateArgsSettings(TypedDict, total=False):
    """Settings for truncating large tool arguments in old messages.

    Attributes:
        trigger: Threshold to trigger argument truncation. If None, truncation is disabled.
        keep: Context retention policy for message truncation (defaults to last 20 messages).
        max_length: Maximum character length for tool arguments before truncation (defaults to 100).
        truncation_text: Text to replace truncated arguments with (defaults to "...(argument truncated)").
    """

    trigger: ContextSize | None
    keep: ContextSize
    max_length: int
    truncation_text: str


class SummarizationMiddleware(BaseSummarizationMiddleware):
    """Summarization middleware with backend support for conversation history offloading.

    Args:
        model: The language model to use for generating summaries.
        backend: Backend instance or factory for persisting conversation history.

            If `None`, no offloading occurs (like base `SummarizationMiddleware`).
        trigger: Threshold(s) that trigger summarization.
        keep: Context retention policy after summarization.
        token_counter: Function to count tokens in messages.
        summary_prompt: Prompt template for generating summaries.
        trim_tokens_to_summarize: Max tokens to include when generating summary.
        history_path_prefix: Path prefix for storing conversation history.

            Defaults to `'/conversation_history'`.

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

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES | None = None,
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
            trigger: One or more thresholds that trigger summarization.

                Provide a single
                [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple or a list of tuples, in which case summarization runs when any
                threshold is met.

                !!! example

                    ```python
                    # Trigger summarization when 50 messages is reached
                    ("messages", 50)

                    # Trigger summarization when 3000 tokens is reached
                    ("tokens", 3000)

                    # Trigger summarization either when 80% of model's max input tokens
                    # is reached or when 100 messages is reached (whichever comes first)
                    [("fraction", 0.8), ("messages", 100)]
                    ```

                    See [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                    for more details.
            keep: Context retention policy applied after summarization.

                Provide a [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple to specify how much history to preserve.

                Defaults to keeping the most recent `20` messages.

                Does not support multiple values like `trigger`.

                !!! example

                    ```python
                    # Keep the most recent 20 messages
                    ("messages", 20)

                    # Keep the most recent 3000 tokens
                    ("tokens", 3000)

                    # Keep the most recent 30% of the model's max input tokens
                    ("fraction", 0.3)
                    ```
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
                the summarization call.

                Pass `None` to skip trimming entirely.
            truncate_args_settings: Settings for truncating large tool arguments in old messages.

                Provide a [`TruncateArgsSettings`][deepagents.middleware.summarization.TruncateArgsSettings]
                dictionary to configure when and how to truncate tool arguments. If `None`,
                argument truncation is disabled.

                !!! example

                    ```python
                    # Truncate when 50 messages is reached, keep last 20 messages
                    {"trigger": ("messages", 50), "keep": ("messages", 20), "max_length": 100, "truncation_text": "...(truncated)"}

                    # Truncate when 50% of tokens reached, keep 10% of tokens
                    {"trigger": ("fraction", 0.5), "keep": ("fraction", 0.1), "max_length": 100, "truncation_text": "...(truncated)"}
                    ```
        """
        super().__init__(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
            **deprecated_kwargs,
        )
        self._backend = backend
        self._history_path_prefix = history_path_prefix

        # Parse truncate_args_settings
        if truncate_args_settings is None:
            self._truncate_args_trigger = None
            self._truncate_args_keep = ("messages", 20)
            self._max_arg_length = 100
            self._truncation_text = "...(argument truncated)"
        else:
            self._truncate_args_trigger = truncate_args_settings.get("trigger")
            self._truncate_args_keep = truncate_args_settings.get("keep", ("messages", 20))
            self._max_arg_length = truncate_args_settings.get("max_length", 100)
            self._truncation_text = truncate_args_settings.get("truncation_text", "...(argument truncated)")

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol | None:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.

        Returns:
            Resolved backend instance, or None if no backend configured.
        """
        if self._backend is None:
            return None

        if callable(self._backend):
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=getattr(runtime, "config", {}),
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    def _get_thread_id(self) -> str:
        """Extract `thread_id` from langgraph config.

        Uses `get_config()` to access the `RunnableConfig` from langgraph's
        contextvar. Falls back to a generated session ID if not available.

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
            `True` if this is a summary message from a previous summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the
        previous summary `HumanMessage` since the original messages are already
        stored in the backend.

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

        Returns:
            List containing the summary `HumanMessage`.
        """
        if file_path is not None:
            content = f"""You are in the middle of a conversation that has been summarized.

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

    def _format_messages_as_markdown(self, messages: list[AnyMessage]) -> str:
        """Format messages as human-readable text with role prefixes.

        Uses `get_buffer_string` to produce a compact, LLM-friendly representation of
        the conversation history with role prefixes (e.g., `'Human:'`, `'AI:'`).

        Args:
            messages: List of messages to format.

        Returns:
            Plain text string with messages prefixed by role.
        """
        return get_buffer_string(messages)

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

    def _determine_truncate_cutoff_index(self, messages: list[AnyMessage]) -> int:
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
            return len(messages) - keep_value

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
                msg_tokens = self.token_counter([messages[i]])
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

    def _truncate_args(self, messages: list[AnyMessage]) -> tuple[list[AnyMessage], bool]:
        """Truncate large tool call arguments in old messages.

        Args:
            messages: Messages to potentially truncate.

        Returns:
            Tuple of (truncated_messages, modified). If modified is False,
            truncated_messages is the same as input messages.
        """
        total_tokens = self.token_counter(messages)
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
                        truncated_call = self._truncate_tool_call(tool_call)
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

        Previous summary messages are filtered out to avoid redundant storage
        during chained summarization events.

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
        new_section = f"## Summarized at {timestamp}\n\n{self._format_messages_as_markdown(filtered_messages)}\n\n"

        # Read existing content (if any) and append
        existing_content = ""
        try:
            content = backend.read(path)
            # backend.read returns a string, empty or error string if file doesn't exist
            if content and not content.startswith("Error:"):
                existing_content = content
        except Exception:  # noqa: BLE001, S110
            # File doesn't exist yet, that's fine
            pass

        combined_content = existing_content + new_section

        try:
            result = backend.write(path, combined_content)
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
            # Don't fail summarization if offloading fails - this is optional functionality
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

        Previous summary messages are filtered out to avoid redundant storage
        during chained summarization events.

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
        new_section = f"## Summarized at {timestamp}\n\n{self._format_messages_as_markdown(filtered_messages)}\n\n"

        # Read existing content (if any) and append
        existing_content = ""
        try:
            content = await backend.aread(path)
            # backend.aread returns a string, empty or error string if file doesn't exist
            if content and not content.startswith("Error:"):
                existing_content = content
        except Exception:  # noqa: BLE001, S110
            # File doesn't exist yet, that's fine
            pass

        combined_content = existing_content + new_section

        try:
            result = await backend.awrite(path, combined_content)
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
            # Don't fail summarization if offloading fails - this is optional functionality
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

    @override
    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, with history offloading and arg truncation.

        First truncates large tool arguments in old messages if configured.
        Then offloads messages to backend before summarization if thresholds are met.
        The summary message includes a reference to the file path where the
        full conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with truncated/summarized messages if processing was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        # Step 1: Truncate args if configured
        truncated_messages, args_were_truncated = self._truncate_args(messages)

        # Step 2: Check if summarization should happen
        total_tokens = self.token_counter(truncated_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If only truncation happened (no summarization)
        if args_were_truncated and not should_summarize:
            return {"messages": Overwrite(truncated_messages)}

        # If no truncation and no summarization
        if not should_summarize:
            return None

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # If truncation happened but we can't summarize, still return truncated messages
            if args_were_truncated:
                return {"messages": Overwrite(truncated_messages)}
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend first to get the file path for the summary message
        file_path: str | None = None
        backend = self._get_backend(state, runtime)
        if backend is not None:
            file_path = self._offload_to_backend(backend, messages_to_summarize)

        # Generate summary
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference if available
        new_messages = self._build_new_messages_with_path(summary, file_path)

        return {
            "messages": Overwrite(
                [
                    *new_messages,
                    *preserved_messages,
                ]
            )
        }

    @override
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, with history offloading and arg truncation (async).

        First truncates large tool arguments in old messages if configured.
        Then offloads messages to backend before summarization if thresholds are met.
        The summary message includes a reference to the file path where the
        full conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with truncated/summarized messages if processing was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        # Step 1: Truncate args if configured
        truncated_messages, args_were_truncated = self._truncate_args(messages)

        # Step 2: Check if summarization should happen
        total_tokens = self.token_counter(truncated_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If only truncation happened (no summarization)
        if args_were_truncated and not should_summarize:
            return {"messages": Overwrite(truncated_messages)}

        # If no truncation and no summarization
        if not should_summarize:
            return None

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # If truncation happened but we can't summarize, still return truncated messages
            if args_were_truncated:
                return {"messages": Overwrite(truncated_messages)}
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(truncated_messages, cutoff_index)

        # Offload to backend first to get the file path for the summary message
        file_path: str | None = None
        backend = self._get_backend(state, runtime)
        if backend is not None:
            file_path = await self._aoffload_to_backend(backend, messages_to_summarize)

        # Generate summary
        summary = await self._acreate_summary(messages_to_summarize)

        # Build summary message with file path reference if available
        new_messages = self._build_new_messages_with_path(summary, file_path)

        return {
            "messages": Overwrite(
                [
                    *new_messages,
                    *preserved_messages,
                ]
            )
        }
