"""Unit tests for `SummarizationMiddleware` with backend offloading."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents.backends.protocol import BackendProtocol, EditResult, FileDownloadResponse, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


def make_conversation_messages(
    num_old: int = 6,
    num_recent: int = 3,
    *,
    include_previous_summary: bool = False,
) -> list:
    """Create a realistic conversation message sequence.

    Args:
        num_old: Number of "old" messages that will be summarized
        num_recent: Number of "recent" messages to preserve
        include_previous_summary: If `True`, start with a summary `HumanMessage`
            containing placeholder text.

    Returns:
        List of messages simulating a conversation
    """
    messages: list[BaseMessage] = []

    if include_previous_summary:
        messages.append(
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nPrevious summary content...",
                additional_kwargs={"lc_source": "summarization"},
                id="summary-msg-0",
            )
        )

    # Add old messages (to be summarized)
    for i in range(num_old):
        if i % 3 == 0:
            messages.append(HumanMessage(content=f"User message {i}", id=f"human-{i}"))
        elif i % 3 == 1:
            messages.append(
                AIMessage(
                    content=f"AI response {i}",
                    id=f"ai-{i}",
                    tool_calls=[{"id": f"tool-call-{i}", "name": "test_tool", "args": {}}],
                )
            )
        else:
            messages.append(
                ToolMessage(
                    content=f"Tool result {i}",
                    tool_call_id=f"tool-call-{i - 1}",
                    id=f"tool-{i}",
                )
            )

    # Add recent messages (to be preserved)
    for i in range(num_recent):
        idx = num_old + i
        messages.append(HumanMessage(content=f"Recent message {idx}", id=f"recent-{idx}"))

    return messages


class MockBackend(BackendProtocol):
    """A mock backend that records read/write calls and can simulate failures."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        error_message: str | None = None,
        existing_content: str | None = None,
        download_raises: bool = False,
        write_raises: bool = False,
    ) -> None:
        """Initialize the mock backend.

        Args:
            should_fail: If `True`, write operations will simulate a failure.
            error_message: The error message to return on failure.
            existing_content: Initialize the backend with existing content for reads.
            download_raises: If `True`, `download_files` will raise an exception.
            write_raises: If `True`, `write`/`edit` will raise an exception.
        """
        self.write_calls: list[tuple[str, str]] = []
        self.edit_calls: list[tuple[str, str, str]] = []
        self.read_calls: list[str] = []
        self.download_files_calls: list[list[str]] = []
        self.should_fail = should_fail
        self.error_message = error_message
        self.existing_content = existing_content
        self.download_raises = download_raises
        self.write_raises = write_raises

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        self.read_calls.append(path)
        if self.existing_content is not None:
            return self.existing_content
        return ""

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(path, offset, limit)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files - returns raw content as bytes."""
        self.download_files_calls.append(paths)
        if self.download_raises:
            msg = "Mock download_files exception"
            raise RuntimeError(msg)
        responses = []
        for path in paths:
            if self.existing_content is not None:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=self.existing_content.encode("utf-8"),
                        error=None,
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",
                    )
                )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        if self.download_raises:
            msg = "Mock adownload_files exception"
            raise RuntimeError(msg)
        return self.download_files(paths)

    def write(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        if self.write_raises:
            msg = "Mock write exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return WriteResult(error=self.error_message or "Mock write failure")
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        if self.write_raises:
            msg = "Mock awrite exception"
            raise RuntimeError(msg)
        return self.write(path, content)

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Edit a file by replacing string occurrences."""
        self.edit_calls.append((path, old_string, new_string))
        if self.write_raises:
            msg = "Mock edit exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return EditResult(error=self.error_message or "Mock edit failure")
        return EditResult(path=path, occurrences=1)

    async def aedit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Async version of edit."""
        if self.write_raises:
            msg = "Mock aedit exception"
            raise RuntimeError(msg)
        return self.edit(path, old_string, new_string, replace_all)


def make_mock_runtime() -> MagicMock:
    """Create a mock `Runtime`.

    Note: `Runtime` does not have a `config` attribute. Config is accessed
    via `get_config()` from langgraph's contextvar. Use `mock_get_config()`
    to control thread_id in tests.
    """
    runtime = MagicMock()
    runtime.context = {}
    runtime.stream_writer = MagicMock()
    runtime.store = None
    # Explicitly don't set runtime.config - it doesn't exist on real Runtime
    del runtime.config
    return runtime


@contextmanager
def mock_get_config(thread_id: str | None = "test-thread-123") -> Generator[None, None, None]:
    """Context manager to mock `get_config()` with a specific `thread_id`.

    Args:
        thread_id: The `thread_id` to return, or `None` to simulate missing config.

    Yields:
        `None` - use as a context manager around test code.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id is not None else {"configurable": {}}

    with patch("deepagents.middleware.summarization.get_config", return_value=config):
        yield


def make_mock_model(summary_response: str = "This is a test summary.") -> MagicMock:
    """Create a mock LLM model for summarization.

    Args:
        summary_response: The text to return as the summary for testing purposes.
    """
    model = MagicMock()
    model.invoke.return_value = MagicMock(text=summary_response)
    model._llm_type = "test-model"
    model.profile = {"max_input_tokens": 100000}
    model._get_ls_params.return_value = {"ls_provider": "test"}
    return model


def make_model_request(
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> ModelRequest:
    """Create a ModelRequest from a state dict.

    Args:
        state: The agent state containing messages.
        runtime: The runtime object.

    Returns:
        A ModelRequest suitable for calling wrap_model_call.
    """
    mock_model = make_mock_model()
    return ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=None,
        tools=[],
        runtime=runtime,
        state=state,
    )


def call_wrap_model_call(
    middleware: SummarizationMiddleware,
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> tuple["ModelResponse | ExtendedModelResponse", ModelRequest | None]:
    """Helper to call wrap_model_call and capture what was passed to handler.

    Args:
        middleware: The middleware instance to test.
        state: The agent state.
        runtime: The runtime object.

    Returns:
        Tuple of (result, modified_request) where:
        - result is the return value from wrap_model_call
        - modified_request is the request passed to the handler (or None if handler wasn't called)
    """
    request = make_model_request(state, runtime)
    captured_request = None

    def handler(req: ModelRequest) -> "ModelResponse":
        nonlocal captured_request
        captured_request = req
        # Return a mock response
        return AIMessage(content="Mock response")

    result = middleware.wrap_model_call(request, handler)
    return result, captured_request


async def call_awrap_model_call(
    middleware: SummarizationMiddleware,
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> tuple["ModelResponse | ExtendedModelResponse", ModelRequest | None]:
    """Helper to call awrap_model_call and capture what was passed to handler (async version).

    Args:
        middleware: The middleware instance to test.
        state: The agent state.
        runtime: The runtime object.

    Returns:
        Tuple of (result, modified_request) where:
        - result is the return value from awrap_model_call
        - modified_request is the request passed to the handler (or None if handler wasn't called)
    """
    request = make_model_request(state, runtime)
    captured_request = None

    async def handler(req: ModelRequest) -> "ModelResponse":
        nonlocal captured_request
        captured_request = req
        # Return a mock response
        return AIMessage(content="Mock response")

    result = await middleware.awrap_model_call(request, handler)
    return result, captured_request


# -----------------------------------------------------------------------------


class TestSummarizationMiddlewareInit:
    """Tests for middleware initialization."""

    def test_init_with_backend(self) -> None:
        """Test initialization with a backend instance."""
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert middleware._backend is backend
        assert middleware._history_path_prefix == "/conversation_history"

    def test_init_with_backend_factory(self) -> None:
        """Test initialization with a backend factory function."""
        backend = MockBackend()
        factory = lambda _rt: backend  # noqa: E731

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert callable(middleware._backend)


class TestOffloadingBasic:
    """Tests for basic offloading behavior."""

    def test_offload_writes_to_backend(self) -> None:
        """Test that summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should have triggered summarization
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert "_summarization_event" in result.command.update
        assert len(backend.write_calls) == 1

        path, content = backend.write_calls[0]
        assert path == "/conversation_history/test-thread-123.md"

        assert "## Summarized at" in content
        assert "Human:" in content or "AI:" in content

    def test_offload_appends_to_existing_content(self) -> None:
        """Test that second summarization appends to existing file."""
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        assert len(backend.edit_calls) == 1
        _, old_string, new_string = backend.edit_calls[0]

        # old_string should be the existing content
        assert old_string == existing

        # new_string (combined content) should contain both old and new sections
        assert "## Summarized at 2024-01-01T00:00:00Z" in new_string
        expected_section_count = 2  # One existing + one new summarization section
        assert new_string.count("## Summarized at") == expected_section_count

    def test_typical_tool_heavy_conversation(self) -> None:
        """Test with a realistic tool-heavy conversation pattern.

        Simulates:

        ```txt
        HumanMessage -> AIMessage(tool_calls) -> ToolMessage -> ToolMessage ->
        ToolMessage -> AIMessage -> HumanMessage -> AIMessage -> ToolMessage (trigger)
        ```
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 8),
            keep=("messages", 3),
        )

        messages = [
            HumanMessage(content="Search for Python tutorials", id="h1"),
            AIMessage(
                content="I'll search for Python tutorials.",
                id="a1",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "python"}}],
            ),
            ToolMessage(content="Result 1: Python basics", tool_call_id="tc1", id="t1"),
            ToolMessage(content="Result 2: Advanced Python", tool_call_id="tc1", id="t2"),
            ToolMessage(content="Result 3: Python projects", tool_call_id="tc1", id="t3"),
            AIMessage(content="Here are some Python tutorials I found...", id="a2"),
            HumanMessage(content="Show me the first one", id="h2"),
            AIMessage(
                content="Let me get that for you.",
                id="a3",
                tool_calls=[{"id": "tc2", "name": "fetch", "args": {"url": "..."}}],
            ),
            ToolMessage(content="Tutorial content...", tool_call_id="tc2", id="t4"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert len(backend.write_calls) == 1

        _, content = backend.write_calls[0]

        # Should have markdown content with summarized messages
        assert "## Summarized at" in content
        assert "Search for Python tutorials" in content

    def test_second_summarization_after_first(self) -> None:
        """Test a second summarization event after an initial one.

        Ensures the chained summarization correctly handles the existing summary message.
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization
        messages = [
            # Previous summary from first summarization
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None

        _, content = backend.write_calls[0]

        # The previous summary message (marked with lc_source: "summarization") should NOT
        # be offloaded—it's a synthetic message, and the original messages it summarized
        # are already stored in the backend file from the first summarization
        assert "First summary" not in content, "Previous summary should be filtered from offload"
        # But the new conversation messages should be offloaded
        assert "New question 1" in content

    def test_filters_previous_summary_messages(self) -> None:
        """Test that previous summary `HumanMessage` objects are NOT included in offload.

        When a second summarization occurs, the previous summary message should be
        filtered out since we already have the original messages stored.
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # Create messages that include a previous summary
        messages = make_conversation_messages(
            num_old=6,
            num_recent=2,
            include_previous_summary=True,
        )
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        _, content = backend.write_calls[0]

        # Check that the offloaded content doesn't include "Previous summary content"
        # (which is the content of the summary message added by include_previous_summary)
        assert "Previous summary content" not in content, "Previous summary message should be filtered from offload"


class TestSummaryMessageFormat:
    """Tests for the summary message format with file path reference."""

    def test_summary_includes_file_path(self) -> None:
        """Test that summary message includes the file path reference."""
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Test summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # Get the summary message (first in modified messages list)
        summary_msg = modified_request.messages[0]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/test-thread.md" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Test summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

    def test_summary_has_lc_source_marker(self) -> None:
        """Test that summary message has `lc_source=summarization` marker."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert modified_request is not None
        summary_msg = modified_request.messages[0]

        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"

    def test_summarization_aborts_on_backend_failure(self) -> None:
        """Test that summarization warns when backend write fails but still summarizes."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model(summary_response="Unused summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    def test_summary_includes_file_path_after_second_summarization(self) -> None:
        """Test that summary message includes file path reference after multiple summarizations.

        This ensures the path reference is present even when a previous summary message
        exists in the conversation (i.e., chained summarization).
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization - starts with a summary message
        messages = [
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary that trigger second summarization
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="multi-summarize-thread"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # The summary message should be the first message
        summary_msg = modified_request.messages[0]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/multi-summarize-thread.md" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Second summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

        # Should have lc_source marker
        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"


class TestNoSummarizationTriggered:
    """Tests for when summarization threshold is not met."""

    def test_no_offload_when_below_threshold(self) -> None:
        """Test that no offload occurs when message count is below trigger."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization)
        assert not isinstance(result, ExtendedModelResponse)

        # No writes should have occurred
        assert len(backend.write_calls) == 0


def test_system_message_counts_for_trigger_only() -> None:
    """System message should affect token trigger but not be sent in messages."""
    backend = MockBackend()
    seen_system = {"counted": False}

    def token_counter(messages: list[BaseMessage]) -> int:
        if any(isinstance(msg, SystemMessage) for msg in messages):
            seen_system["counted"] = True
        return len(messages)

    middleware = SummarizationMiddleware(
        model=make_mock_model(),
        backend=backend,
        trigger=("tokens", 3),
        keep=("messages", 1),
        token_counter=token_counter,
    )

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = make_model_request(state, runtime).override(system_message=SystemMessage(content="sys"))

    captured_request = None

    def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    with mock_get_config():
        result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, ExtendedModelResponse)
    assert seen_system["counted"] is True
    assert captured_request is not None
    assert captured_request.system_message is not None
    assert all(not isinstance(msg, SystemMessage) for msg in captured_request.messages)
    assert len(backend.write_calls) == 1


@pytest.mark.anyio
async def test_async_tools_passed_to_token_counter_for_summarization() -> None:
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))
    seen = {"tools": False, "system": False}

    def token_counter(messages: list[BaseMessage], *, tools: list[dict[str, Any]] | None = None) -> int:
        if tools:
            seen["tools"] = True
        if any(isinstance(msg, SystemMessage) for msg in messages):
            seen["system"] = True
        return 3 if seen["system"] else 2

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("tokens", 3),
        keep=("messages", 1),
        token_counter=token_counter,
    )

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=SystemMessage(content="sys"),
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        runtime=runtime,
        state=state,
    )

    captured_request = None

    async def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    with mock_get_config():
        result = await middleware.awrap_model_call(request, handler)

    assert isinstance(result, ExtendedModelResponse)
    assert seen["tools"]
    assert seen["system"] is True
    assert captured_request is not None
    assert all(not isinstance(msg, SystemMessage) for msg in captured_request.messages)


@pytest.mark.anyio
async def test_async_system_message_counts_for_truncate_trigger() -> None:
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

    def token_counter(messages: list[BaseMessage], *, tools: list[dict[str, Any]] | None = None) -> int:
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            msg = "system message not included"
            raise AssertionError(msg)
        assert tools is not None
        return 3

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 1),
        truncate_args_settings={
            "trigger": ("tokens", 3),
            "keep": ("messages", 1),
            "max_length": 40,
            "truncation_text": "...(argument truncated)",
        },
        token_counter=token_counter,
    )

    long_content = "x" * 100
    messages = [
        AIMessage(
            content="write",
            tool_calls=[{"id": "call-1", "name": "write_file", "args": {"content": long_content}}],
        ),
        HumanMessage(content="next"),
    ]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=SystemMessage(content="sys"),
        tools=[],
        runtime=runtime,
        state=state,
    )

    captured_request = None

    async def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    result = await middleware.awrap_model_call(request, handler)

    assert not isinstance(result, ExtendedModelResponse)
    assert captured_request is not None
    truncated_call = captured_request.messages[0].tool_calls[0]
    assert truncated_call["args"]["content"] == "x" * 20 + "...(argument truncated)"


class TestBackendFailureHandling:
    """Tests for backend failure handling - summarization aborts to prevent data loss."""

    def test_summarization_aborts_on_write_failure(self) -> None:
        """Test that summarization warns when backend write fails but still summarizes."""
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Test that summarization warns when backend raises exception but still summarizes."""
        backend = MagicMock()
        backend.download_files.return_value = []
        backend.write.side_effect = Exception("Network error")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None


class TestThreadIdExtraction:
    """Tests for thread ID extraction via `get_config()`."""

    def test_thread_id_from_config(self) -> None:
        """Test that `thread_id` is correctly extracted from `get_config()`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="custom-thread-456"):
            call_wrap_model_call(middleware, state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/conversation_history/custom-thread-456.md"

    def test_fallback_thread_id_when_missing(self) -> None:
        """Test that a fallback ID is generated when `thread_id` is not in config."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id=None):
            call_wrap_model_call(middleware, state, runtime)

        path, _ = backend.write_calls[0]

        # Should have a generated session ID in the path
        assert "session_" in path
        assert path.endswith(".md")


class TestAsyncBehavior:
    """Tests for async version of `before_model`."""

    @pytest.mark.anyio
    async def test_async_offload_writes_to_backend(self) -> None:
        """Test that async summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()
        # Mock the async create summary
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_aborts_on_failure(self) -> None:
        """Test that async summarization warns on backend failure but still summarizes."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None


class TestBackendFactoryInvocation:
    """Tests for backend factory invocation during summarization."""

    def test_backend_factory_invoked_during_summarization(self) -> None:
        """Test that backend factory is called with `ToolRuntime` during summarization."""
        backend = MockBackend()
        factory_called_with: list = []

        def factory(tool_runtime: object) -> MockBackend:
            factory_called_with.append(tool_runtime)
            return backend

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        # Factory should have been called once
        assert len(factory_called_with) == 1
        # Backend should have received write call
        assert len(backend.write_calls) == 1


class TestCustomHistoryPathPrefix:
    """Tests for custom `history_path_prefix` configuration."""

    def test_custom_history_path_prefix(self) -> None:
        """Test that custom `history_path_prefix` is used in file paths."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
            history_path_prefix="/custom/path",
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            call_wrap_model_call(middleware, state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/custom/path/test-thread.md"


class TestMarkdownFormatting:
    """Tests for markdown message formatting using `get_buffer_string`."""

    def test_markdown_format_includes_message_content(self) -> None:
        """Test that markdown format includes message content."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None

        # Verify the offloaded content is markdown formatted
        _, content = backend.write_calls[0]

        # Should contain human-readable message prefixes
        assert "Human:" in content or "AI:" in content
        # Should contain the actual message content
        assert "User message" in content


class TestDownloadFilesException:
    """Tests for exception handling when download_files raises."""

    def test_summarization_continues_on_download_files_exception(self) -> None:
        """Test that summarization continues when download_files raises an exception."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        # download_files was called (and raised)
        assert len(backend.download_files_calls) == 1
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_summarization_continues_on_download_files_exception(self) -> None:
        """Test that async summarization continues when adownload_files raises."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1


class TestWriteEditException:
    """Tests for exception handling when `write`/`edit` raises - summarization aborts."""

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Test that summarization warns when `write` raises an exception but still summarizes.

        Covers lines 314-322: Exception handler for write in _offload_to_backend.
        """
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_write_exception(self) -> None:
        """Test that async summarization warns when `awrite` raises but still summarizes.

        Covers lines 387-395: Exception handler for awrite in _aoffload_to_backend.
        """
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    def test_summarization_aborts_on_edit_exception(self) -> None:
        """Test that summarization warns when `edit` raises an exception but still summarizes (existing content).

        Covers lines 314-322: Exception handler for edit in _offload_to_backend.
        """
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_edit_exception(self) -> None:
        """Test that async summarization warns when `aedit` raises but still summarizes (existing content).

        Covers lines 387-395: Exception handler for aedit in _aoffload_to_backend.
        """
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None


class TestCutoffIndexEdgeCases:
    """Tests for edge cases where `cutoff_index <= 0`."""

    def test_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that no summarization occurs when `cutoff_index` is `0`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization) because cutoff_index would be 0 or negative
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should occur
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_not_triggered(self) -> None:
        """Test that async `abefore_model` returns `None` when summarization not triggered."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization)
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should have occurred
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that async `abefore_model` returns `None` when `cutoff_index <= 0`."""
        backend = MockBackend()
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization) because cutoff_index would be 0 or negative
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should occur
        assert len(backend.write_calls) == 0


# -----------------------------------------------------------------------------
# Argument truncation tests
# -----------------------------------------------------------------------------


def test_no_truncation_when_trigger_is_none() -> None:
    """Test that no truncation occurs when truncate_args_settings is None."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings=None,  # Truncation disabled
    )

    # Create messages with large tool calls
    large_content = "x" * 200
    messages = [
        HumanMessage(content="Write a file", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should return ModelResponse (no truncation, no summarization)
    assert not isinstance(result, ExtendedModelResponse)


def test_truncate_old_write_file_tool_call() -> None:
    """Test that old write_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message with write_file tool call (will be cleaned)
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    # Check that the old tool call was cleaned
    first_ai_msg = cleaned_messages[0]
    assert isinstance(first_ai_msg, AIMessage)
    assert len(first_ai_msg.tool_calls) == 1
    assert first_ai_msg.tool_calls[0]["name"] == "write_file"
    # Content should be first 20 chars + truncation text
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_old_edit_file_tool_call() -> None:
    """Test that old edit_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_old_string = "a" * 100
    large_new_string = "b" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "edit_file",
                    "args": {
                        "file_path": "/test.py",
                        "old_string": large_old_string,
                        "new_string": large_new_string,
                    },
                }
            ],
        ),
        ToolMessage(content="File edited", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["name"] == "edit_file"
    assert first_ai_msg.tool_calls[0]["args"]["old_string"] == "a" * 20 + "...(argument truncated)"
    assert first_ai_msg.tool_calls[0]["args"]["new_string"] == "b" * 20 + "...(argument truncated)"


def test_truncate_ignores_other_tool_calls() -> None:
    """Test that tool calls other than write_file and edit_file are not affected."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should return AIMessage since read_file is not cleaned (no truncation)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify read_file args are unchanged
    first_msg = modified_request.messages[0]
    assert isinstance(first_msg, AIMessage)
    assert first_msg.tool_calls[0]["args"]["content"] == large_content


def test_truncate_respects_recent_messages() -> None:
    """Test that recent messages are not cleaned."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 4),  # Keep last 4 messages
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a1"),
        # Recent message with write_file (should NOT be cleaned - it's in the last 4)
        AIMessage(
            content="",
            id="a2",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # No truncation should happen since the tool call is in the keep window (last 4 messages)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify the write_file content is unchanged
    write_file_msg = modified_request.messages[2]
    assert isinstance(write_file_msg, AIMessage)
    assert write_file_msg.tool_calls[0]["args"]["content"] == large_content


def test_truncate_with_token_keep_policy() -> None:
    """Test truncation with token-based keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()

    # Custom token counter that returns predictable counts
    def simple_token_counter(msgs: list) -> int:
        return len(msgs) * 100  # 100 tokens per message

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("tokens", 250),  # Keep ~2-3 messages
            "max_length": 100,
        },
        token_counter=simple_token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    # First message should be cleaned since it's outside the token window
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_with_fraction_trigger_and_keep() -> None:
    """Test truncation with fraction-based trigger and keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.profile = {"max_input_tokens": 1000}

    # Custom token counter: 200 tokens per message
    def token_counter(msgs: list) -> int:
        return len(msgs) * 200

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold for summarization
        truncate_args_settings={
            "trigger": ("fraction", 0.5),  # Trigger at 50% of 1000 = 500 tokens
            "keep": ("fraction", 0.2),  # Keep 20% of 1000 = 200 tokens (~1 message)
            "max_length": 100,
        },
        token_counter=token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Message 1", id="h1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should trigger truncation: 3 messages * 200 = 600 tokens > 500 threshold
    # Should keep only ~200 tokens (1 message) from the end
    # So first 2 messages should be in truncation zone
    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_before_summarization() -> None:
    """Test that truncation happens before summarization."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Test summary")

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 10),  # Trigger summarization
        keep=("messages", 2),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 3),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message that will be cleaned and summarized
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ] + [HumanMessage(content=f"Message {i}", id=f"h{i}") for i in range(10)]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    with mock_get_config(thread_id="test-thread"):
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert modified_request is not None

    # Should have triggered both truncation and summarization
    # Backend should have received a write call for offloading
    assert len(backend.write_calls) == 1

    # Result should contain summary message
    new_messages = modified_request.messages
    assert any("summary" in str(msg.content).lower() for msg in new_messages)


def test_truncate_without_summarization() -> None:
    """Test that truncation can happen independently of summarization."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # No backend write (no summarization)
    assert len(backend.write_calls) == 0

    # But truncation should have happened
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_preserves_small_arguments() -> None:
    """Test that small arguments are not truncated even in old messages."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    small_content = "short"

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": small_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # No modification should happen since content is small
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify the content is unchanged
    first_msg = modified_request.messages[0]
    assert isinstance(first_msg, AIMessage)
    assert first_msg.tool_calls[0]["args"]["content"] == small_content


def test_truncate_mixed_tool_calls() -> None:
    """Test that only write_file and edit_file are cleaned in a message with multiple tool calls."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt"},
                },
                {
                    "id": "tc2",
                    "name": "write_file",
                    "args": {"file_path": "/output.txt", "content": large_content},
                },
                {
                    "id": "tc3",
                    "name": "shell",
                    "args": {"command": "ls -la"},
                },
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        ToolMessage(content="File written", tool_call_id="tc2", id="t2"),
        ToolMessage(content="Output", tool_call_id="tc3", id="t3"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert len(first_ai_msg.tool_calls) == 3

    # read_file should be unchanged
    assert first_ai_msg.tool_calls[0]["name"] == "read_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.txt"

    # write_file should be cleaned
    assert first_ai_msg.tool_calls[1]["name"] == "write_file"
    assert first_ai_msg.tool_calls[1]["args"]["content"] == "x" * 20 + "...(argument truncated)"

    # shell should be unchanged
    assert first_ai_msg.tool_calls[2]["name"] == "shell"
    assert first_ai_msg.tool_calls[2]["args"]["command"] == "ls -la"


def test_truncate_custom_truncation_text() -> None:
    """Test that custom truncation text is used."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
            "truncation_text": "[TRUNCATED]",
        },
    )

    large_content = "y" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "y" * 20 + "[TRUNCATED]"


@pytest.mark.anyio
async def test_truncate_async_works() -> None:
    """Test that async argument truncation works correctly."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = await call_awrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


# -----------------------------------------------------------------------------
# Chained summarization cutoff index tests
# -----------------------------------------------------------------------------


def test_chained_summarization_cutoff_index() -> None:
    """Test that state_cutoff_index is computed correctly across three chained summarizations.

    The formula is:
        state_cutoff = old_state_cutoff + effective_cutoff - 1

    The -1 accounts for the synthetic summary message at effective[0] which does not
    correspond to any state message.

    Setup: trigger=("messages", 5), keep=("messages", 2).

    Round 1 (no previous event):
        State: [S0..S7] (8 messages), cutoff = 8 - 2 = 6.
        Preserved: [S6, S7]. Event: cutoff_index=6.

    Round 2 (previous cutoff=6):
        State: [S0..S13] (14 messages).
        effective = [summary_1, S6..S13] (9 messages), effective cutoff = 9 - 2 = 7.
        state_cutoff = 6 + 7 - 1 = 12. Preserved: [S12, S13].

    Round 3 (previous cutoff=12):
        State: [S0..S19] (20 messages).
        effective = [summary_2, S12..S19] (9 messages), effective cutoff = 9 - 2 = 7.
        state_cutoff = 12 + 7 - 1 = 18. Preserved: [S18, S19].
    """
    backend = MockBackend()
    mock_model = make_mock_model()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )
    runtime = make_mock_runtime()

    def make_state_messages(n: int) -> list:
        return [HumanMessage(content=f"S{i}", id=f"s{i}") if i % 2 == 0 else AIMessage(content=f"S{i}", id=f"s{i}") for i in range(n)]

    def offloaded_labels(write_call_content: str) -> list[str]:
        """Extract S-labels from backend write content (e.g. "Human: S0" -> "S0")."""
        return [word for word in write_call_content.split() if word.startswith("S") and word[1:].isdigit()]

    # --- Round 1: first summarization, no previous event ---
    state = cast("AgentState[Any]", {"messages": make_state_messages(8)})
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_1 = result.command.update["_summarization_event"]
    assert event_1["cutoff_index"] == 6
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S6", "S7"]
    _, content = backend.write_calls[0]
    assert offloaded_labels(content) == ["S0", "S1", "S2", "S3", "S4", "S5"]

    # --- Round 2: second summarization, feed back event from round 1 ---
    state = cast(
        "AgentState[Any]",
        {"messages": make_state_messages(14), "_summarization_event": event_1},
    )
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_2 = result.command.update["_summarization_event"]
    assert event_2["cutoff_index"] == 12
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S12", "S13"]
    _, content = backend.write_calls[1]
    assert offloaded_labels(content) == ["S6", "S7", "S8", "S9", "S10", "S11"]

    # --- Round 3: third summarization, feed back event from round 2 ---
    state = cast(
        "AgentState[Any]",
        {"messages": make_state_messages(20), "_summarization_event": event_2},
    )
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_3 = result.command.update["_summarization_event"]
    assert event_3["cutoff_index"] == 18
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S18", "S19"]
    _, content = backend.write_calls[2]
    assert offloaded_labels(content) == ["S12", "S13", "S14", "S15", "S16", "S17"]


# -----------------------------------------------------------------------------
# ContextOverflowError fallback tests
# -----------------------------------------------------------------------------


def test_context_overflow_triggers_summarization() -> None:
    """Test that ContextOverflowError triggers fallback to summarization."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Fallback summary")

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold - won't trigger normally
        keep=("messages", 2),
    )

    messages = make_conversation_messages(num_old=6, num_recent=2)
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    # Create a handler that raises ContextOverflowError on first call
    call_count = {"count": 0}

    def handler_with_overflow(_req: ModelRequest) -> "ModelResponse":
        call_count["count"] += 1
        if call_count["count"] == 1:
            # First call with unsummarized messages throws overflow
            raise ContextOverflowError
        # Second call with summarized messages succeeds
        return AIMessage(content="Success after summarization")

    request = make_model_request(state, runtime)

    with mock_get_config():
        result = middleware.wrap_model_call(request, handler_with_overflow)

    # Should have triggered summarization as fallback
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert result.command.update["_summarization_event"]

    # Should have called handler twice (once failed, once succeeded)
    assert call_count["count"] == 2

    # Backend should have offloaded messages
    assert len(backend.write_calls) == 1


@pytest.mark.anyio
async def test_async_context_overflow_triggers_summarization() -> None:
    """Test that ContextOverflowError triggers fallback to summarization (async)."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Fallback summary")
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async fallback summary"))

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold - won't trigger normally
        keep=("messages", 2),
    )

    messages = make_conversation_messages(num_old=6, num_recent=2)
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    # Create a handler that raises ContextOverflowError on first call
    call_count = {"count": 0}

    async def handler_with_overflow(_req: ModelRequest) -> "ModelResponse":
        call_count["count"] += 1
        if call_count["count"] == 1:
            # First call with unsummarized messages throws overflow
            raise ContextOverflowError
        # Second call with summarized messages succeeds
        return AIMessage(content="Success after summarization")

    request = make_model_request(state, runtime)

    with mock_get_config():
        result = await middleware.awrap_model_call(request, handler_with_overflow)

    # Should have triggered summarization as fallback
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update

    # Should have called handler twice (once failed, once succeeded)
    assert call_count["count"] == 2

    # Backend should have offloaded messages
    assert len(backend.write_calls) == 1


def test_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""

    def token_counter(messages: list[BaseMessage], **_kwargs: Any) -> int:
        return len(messages) * 200

    # Create a mock model with profile
    mock_model = make_mock_model()
    mock_model.profile = {"max_input_tokens": 1000}

    backend = MockBackend()

    # Test 1: Don't engage summarization when below threshold
    # total_tokens = 4 * 200 = 800, threshold = 0.81 * 1000 = 810
    # 800 < 810, so no summarization
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.81),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    messages: list[BaseMessage] = [
        HumanMessage(content="Message 1", id="h1"),
        AIMessage(content="Message 2", id="a1"),
        HumanMessage(content="Message 3", id="h2"),
        AIMessage(content="Message 4", id="a2"),
    ]

    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should not trigger summarization
    assert not isinstance(result, ExtendedModelResponse)
    assert len(backend.write_calls) == 0

    # Test 2: Engage summarization when at threshold
    # total_tokens = 4 * 200 = 800, threshold = 0.80 * 1000 = 800
    # 800 >= 800, so summarization triggers
    backend = MockBackend()  # Reset backend
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should trigger summarization
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update
    assert len(backend.write_calls) == 1

    # Check the modified messages
    assert modified_request is not None
    summary_message = modified_request.messages[0]
    assert isinstance(summary_message, HumanMessage)
    assert "summarized" in summary_message.content.lower()
    assert "<summary>" in summary_message.content

    # Should preserve last 2 messages (keep=0.5 * 1000 = 500 tokens, 500/200 = 2.5 messages)
    preserved_messages = modified_request.messages[1:]
    assert len(preserved_messages) == 2
    assert [msg.content for msg in preserved_messages] == ["Message 3", "Message 4"]

    # Test 3: With keep=("fraction", 0.6), preserve more messages
    # target tokens = 0.6 * 1000 = 600, 600/200 = 3 messages
    backend = MockBackend()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.6),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    assert modified_request is not None
    preserved_messages = modified_request.messages[1:]
    assert len(preserved_messages) == 3
    assert [msg.content for msg in preserved_messages] == ["Message 2", "Message 3", "Message 4"]

    # Test 4: With keep=("fraction", 0.8), keep everything (no summarization needed)
    # target tokens = 0.8 * 1000 = 800, which equals total tokens, so keep all
    backend = MockBackend()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.8),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should not trigger summarization since we'd keep everything anyway
    assert not isinstance(result, ExtendedModelResponse)
    assert len(backend.write_calls) == 0


def test_usage_metadata_trigger() -> None:
    """Test that usage_metadata from AI messages can trigger summarization.

    This tests advanced triggering based on `usage_metadata` from AI messages,
    particularly for models like Anthropic that report token usage in response metadata.
    """
    backend = MockBackend()
    mock_model = make_mock_model()
    # Mock the model to appear as Anthropic - need to match ls_provider
    mock_model._llm_type = "anthropic-chat"
    mock_model._get_ls_params.return_value = {"ls_provider": "anthropic"}

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("tokens", 10_000),
        keep=("messages", 4),
    )

    messages: list[BaseMessage] = [
        HumanMessage(content="msg1", id="h1"),
        AIMessage(
            content="msg2",
            id="a1",
            tool_calls=[{"name": "tool", "args": {}, "id": "call1"}],
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 5000,
                "output_tokens": 1000,
                "total_tokens": 6000,
            },
        ),
        ToolMessage(content="result", tool_call_id="call1", id="t1"),
        AIMessage(
            content="msg3",
            id="a2",
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 6100,
                "output_tokens": 900,
                "total_tokens": 7000,
            },
        ),
        HumanMessage(content="msg4", id="h2"),
        AIMessage(
            content="msg5",
            id="a3",
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 7500,
                "output_tokens": 2501,
                "total_tokens": 10_001,
            },
        ),
    ]

    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should trigger summarization because usage_metadata shows we exceeded 10k tokens
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update
    assert len(backend.write_calls) == 1
