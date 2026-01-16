"""Unit tests for `SummarizationMiddleware` with backend offloading."""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from deepagents.backends.protocol import BackendProtocol, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware

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

    Returns:
        List of messages simulating a conversation
    """
    messages: list[BaseMessage] = []

    # Optionally include a previous summary message
    if include_previous_summary:
        messages.append(
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nPrevious summary content...",
                additional_kwargs={"lc_source": "summarization"},
                id="summary-msg-0",
            )
        )

    # Add old messages (will be summarized)
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

    # Add recent messages (will be preserved)
    for i in range(num_recent):
        idx = num_old + i
        messages.append(HumanMessage(content=f"Recent message {idx}", id=f"recent-{idx}"))

    return messages


class MockBackend(BackendProtocol):
    """Mock backend that records read/write calls and can simulate failures."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        error_message: str | None = None,
        existing_content: str | None = None,
    ) -> None:
        self.write_calls: list[tuple[str, str]] = []
        self.read_calls: list[str] = []
        self.should_fail = should_fail
        self.error_message = error_message
        self.existing_content = existing_content

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> str:  # noqa: ARG002
        self.read_calls.append(path)
        if self.existing_content is not None:
            return self.existing_content
        return ""

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(path, offset, limit)

    def write(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        if self.should_fail:
            return WriteResult(error=self.error_message or "Mock write failure")
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        return self.write(path, content)


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
def mock_get_config(thread_id: str | None = "test-thread-123"):
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
    """Create a mock LLM model for summarization."""
    model = MagicMock()
    model.invoke.return_value = MagicMock(text=summary_response)
    model._llm_type = "test-model"
    model.profile = {"max_input_tokens": 100000}
    model._get_ls_params.return_value = {"ls_provider": "test"}
    return model


# -----------------------------------------------------------------------------
# Basic functionality tests
# -----------------------------------------------------------------------------


class TestNoBackendConfigured:
    """Tests for behavior when no backend is configured."""

    def test_no_offload_without_backend(self) -> None:
        """Test that no offloading occurs when backend is `None`."""
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        # Should still return summarization result
        result = middleware.before_model(state, runtime)

        assert result is not None
        assert "messages" in result

    def test_summarization_works_without_backend(self) -> None:
        """Test that summarization still works correctly without a backend."""
        mock_model = make_mock_model(summary_response="Summary without backend")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=None,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        # Should have summary message
        assert result is not None
        new_messages = result["messages"]
        # First is RemoveMessage, second is the summary HumanMessage
        summary_msg = new_messages[1]
        assert "Summary without backend" in summary_msg.content


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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread-123"):
            result = middleware.before_model(state, runtime)

        # Should have triggered summarization
        assert result is not None

        # Backend should have received one write call
        assert len(backend.write_calls) == 1

        path, content = backend.write_calls[0]

        # Path should be single markdown file per thread
        assert path == "/conversation_history/test-thread-123.md"

        # Content should be markdown with timestamp header
        assert "## Summarized at" in content
        # Content should include conversation messages
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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

        _, content = backend.write_calls[0]

        # Should contain both old and new sections
        assert "## Summarized at 2024-01-01T00:00:00Z" in content
        assert content.count("## Summarized at") == 2  # Two summarization sections


class TestRealisticScenarios:
    """More realistic test scenarios with typical message patterns."""

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

        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        assert result is not None
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

        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        assert result is not None

        _, content = backend.write_calls[0]

        # The previous summary should NOT be in the offloaded messages
        assert "First summary" not in content, "Previous summary should be filtered from offload"
        # But the new questions should be there
        assert "New question 1" in content


class TestChainedSummarization:
    """Tests for handling multiple summarization events (chained summarization)."""

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
            include_previous_summary=True,  # Include previous summary message
        )
        state = {"messages": messages}
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            result = middleware.before_model(state, runtime)

        # Get the summary message (second in list, after RemoveMessage)
        summary_msg = result["messages"][1]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/test-thread.md" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Test summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

    def test_summary_without_backend_has_simple_format(self) -> None:
        """Test that summary without backend uses simple format (no file path)."""
        mock_model = make_mock_model(summary_response="Simple summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=None,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        summary_msg = result["messages"][1]

        # Should NOT have file path reference
        assert "full conversation history has been saved to" not in summary_msg.content

        # Should have simple format
        assert "Here is a summary of the conversation to date:" in summary_msg.content
        assert "Simple summary" in summary_msg.content

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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        summary_msg = result["messages"][1]

        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"

    def test_summary_fallback_on_backend_failure(self) -> None:
        """Test that summary uses simple format when backend write fails."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model(summary_response="Fallback summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        summary_msg = result["messages"][1]

        # Should fall back to simple format since write failed
        assert "full conversation history has been saved to" not in summary_msg.content
        assert "Here is a summary of the conversation to date:" in summary_msg.content


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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)

        # Should return None (no summarization)
        assert result is None

        # No writes should have occurred
        assert len(backend.write_calls) == 0


class TestBackendFailureHandling:
    """Tests for graceful handling of backend failures."""

    def test_summarization_continues_on_write_failure(self) -> None:
        """Test that summarization completes even if backend write fails."""
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        # Should not raise, should return result
        result = middleware.before_model(state, runtime)

        assert result is not None
        assert "messages" in result

    def test_summarization_continues_on_write_exception(self) -> None:
        """Test that summarization completes even if backend raises exception."""
        backend = MagicMock()
        backend.write.side_effect = Exception("Network error")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = {"messages": messages}
        runtime = make_mock_runtime()

        # Should not raise
        result = middleware.before_model(state, runtime)

        assert result is not None


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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="custom-thread-456"):
            middleware.before_model(state, runtime)

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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        with mock_get_config(thread_id=None):
            middleware.before_model(state, runtime)

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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_continues_on_failure(self) -> None:
        """Test that async summarization completes even on backend failure."""
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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = await middleware.abefore_model(state, runtime)

        assert result is not None


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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        middleware.before_model(state, runtime)

        # Factory should have been called once
        assert len(factory_called_with) == 1
        # Backend should have received write call
        assert len(backend.write_calls) == 1


class TestMarkdownFormatting:
    """Tests for markdown message formatting using get_buffer_string."""

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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        result = middleware.before_model(state, runtime)
        assert result is not None

        # Verify the offloaded content is markdown formatted
        _, content = backend.write_calls[0]

        # Should contain human-readable message prefixes
        assert "Human:" in content or "AI:" in content
        # Should contain the actual message content
        assert "User message" in content


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
        state = {"messages": messages}
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            middleware.before_model(state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/custom/path/test-thread.md"
