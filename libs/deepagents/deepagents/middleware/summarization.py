"""Summarization middleware with backend support for offloading conversation history.

This module extends `SummarizationMiddleware` to persist conversation history to a
backend before summarization, enabling retrieval of full context when needed by an agent.

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

Offloaded messages are stored as JSON at:

```txt
/conversation_history/{thread_id}/{timestamp}.json
```

Each file contains:
- `messages`: List of serialized messages that were summarized
- `messages_text`: Human-readable text representation of the messages
- `message_count`: Number of messages stored
- `thread_id`: The conversation thread identifier
- `timestamp`: ISO 8601 timestamp of when summarization occurred
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.summarization import (
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    DEFAULT_SUMMARY_PROMPT,
    ContextSize,
    TokenCounter,
    count_tokens_approximately,
)
from langchain.agents.middleware.summarization import (
    SummarizationMiddleware as BaseSummarizationMiddleware,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import get_buffer_string
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing_extensions import override

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState
    from langchain.chat_models import BaseChatModel
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)


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
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware with backend support."""
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

    def _get_thread_id(self, runtime: Runtime) -> str:
        """Extract `thread_id` from runtime config if available.

        Args:
            runtime: Runtime context.

        Returns:
            Thread ID string, or a generated UUID if not available.
        """
        # Try to get config from runtime
        config = getattr(runtime, "config", None)
        if config is not None:
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id")
            if thread_id is not None:
                return str(thread_id)

        # Fall back to generated ID - log this since it affects file organization
        generated_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.debug("No thread_id in runtime config, using generated session ID: %s", generated_id)
        return generated_id

    def _get_history_path(self, runtime: Runtime) -> str:
        """Generate path for storing conversation history.

        Args:
            runtime: Runtime context.

        Returns:
            Path string like `'/conversation_history/{thread_id}/{timestamp}.json'`
        """
        thread_id = self._get_thread_id(runtime)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")
        return f"{self._history_path_prefix}/{thread_id}/{timestamp}.json"

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

    def _serialize_messages(self, messages: list[AnyMessage]) -> str:
        """Serialize messages for storage.

        Args:
            messages: List of messages to serialize.

        Returns:
            JSON string containing serialized messages.
        """
        serialized = []
        for msg in messages:
            try:
                # Use model_dump if available (Pydantic v2)
                if hasattr(msg, "model_dump"):
                    serialized.append(msg.model_dump())
                else:
                    # Fallback to dict() for older versions
                    serialized.append(msg.dict())
            except (TypeError, ValueError, AttributeError) as e:
                # Last resort: use string representation (may lose metadata like tool calls)
                logger.warning(
                    "Message serialization failed for %s, using minimal representation: %s",
                    type(msg).__name__,
                    e,
                )
                serialized.append(
                    {
                        "type": type(msg).__name__,
                        "content": str(msg.content),
                    }
                )
        return json.dumps(serialized, indent=2, default=str)

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        runtime: Runtime,
    ) -> str | None:
        """Persist messages to backend before summarization.

        Previous summary messages are filtered out to avoid redundant storage
        during chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            runtime: Runtime context.

        Returns:
            The file path where history was stored, or `None` if write failed.
        """
        path = self._get_history_path(runtime)
        thread_id = self._get_thread_id(runtime)

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "thread_id": thread_id,
            "message_count": len(filtered_messages),
            "messages": json.loads(self._serialize_messages(filtered_messages)),
        }

        try:
            result = backend.write(path, json.dumps(payload, indent=2, default=str))
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (thread: %s, %d messages): %s",
                    path,
                    thread_id,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            # Don't fail summarization if offloading fails - this is optional functionality
            logger.warning(
                "Exception offloading conversation history to %s (thread: %s, %d messages): %s: %s",
                path,
                thread_id,
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
        runtime: Runtime,
    ) -> str | None:
        """Persist messages to backend before summarization (async).

        Previous summary messages are filtered out to avoid redundant storage
        during chained summarization events.

        Args:
            backend: Backend to write to.
            messages: Messages being summarized.
            runtime: Runtime context.

        Returns:
            The file path where history was stored, or `None` if write failed.
        """
        path = self._get_history_path(runtime)
        thread_id = self._get_thread_id(runtime)

        # Filter out previous summary messages to avoid redundant storage
        filtered_messages = self._filter_summary_messages(messages)

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "thread_id": thread_id,
            "message_count": len(filtered_messages),
            "messages": json.loads(self._serialize_messages(filtered_messages)),
        }

        try:
            result = await backend.awrite(path, json.dumps(payload, indent=2, default=str))
            if result is None or result.error:
                error_msg = result.error if result else "backend returned None"
                logger.warning(
                    "Failed to offload conversation history to %s (thread: %s, %d messages): %s",
                    path,
                    thread_id,
                    len(filtered_messages),
                    error_msg,
                )
                return None
        except Exception as e:  # noqa: BLE001
            # Don't fail summarization if offloading fails - this is optional functionality
            logger.warning(
                "Exception offloading conversation history to %s (thread: %s, %d messages): %s: %s",
                path,
                thread_id,
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
        """Process messages before model invocation, with history offloading.

        Overrides parent to offload messages to backend before summarization.
        The summary message includes a reference to the file path where the
        full conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with summarized messages if summarization was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # Offload to backend first to get the file path for the summary message
        file_path: str | None = None
        backend = self._get_backend(state, runtime)
        if backend is not None:
            file_path = self._offload_to_backend(backend, messages_to_summarize, runtime)

        # Generate summary
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference if available
        new_messages = self._build_new_messages_with_path(summary, file_path)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Process messages before model invocation, with history offloading (async).

        Overrides parent to offload messages to backend before summarization.
        The summary message includes a reference to the file path where the
        full conversation history was stored.

        Args:
            state: The agent state.
            runtime: The runtime environment.

        Returns:
            Updated state with summarized messages if summarization was performed.
        """
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        # Offload to backend first to get the file path for the summary message
        file_path: str | None = None
        backend = self._get_backend(state, runtime)
        if backend is not None:
            file_path = await self._aoffload_to_backend(backend, messages_to_summarize, runtime)

        # Generate summary
        summary = await self._acreate_summary(messages_to_summarize)

        # Build summary message with file path reference if available
        new_messages = self._build_new_messages_with_path(summary, file_path)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }
