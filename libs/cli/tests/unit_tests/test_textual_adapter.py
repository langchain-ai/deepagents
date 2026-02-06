"""Unit tests for textual_adapter functions."""

from asyncio import Future

from deepagents_cli.textual_adapter import (
    TextualUIAdapter,
    _build_stream_config,
    _is_summarization_chunk,
)


async def _mock_mount(widget: object) -> None:
    """Mock mount function for tests."""


def _mock_approval() -> Future[object]:
    """Mock approval function for tests."""
    future: Future[object] = Future()
    return future


def _noop_status(_: str) -> None:
    """No-op status callback for tests."""


class TestTextualUIAdapterInit:
    """Tests for `TextualUIAdapter` initialization."""

    def test_set_spinner_callback_stored(self) -> None:
        """Verify `set_spinner` callback is properly stored."""

        async def mock_spinner(status: str | None) -> None:
            pass

        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=mock_spinner,
        )
        assert adapter._set_spinner is mock_spinner

    def test_set_spinner_defaults_to_none(self) -> None:
        """Verify `set_spinner` is optional and defaults to `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._set_spinner is None

    def test_current_tool_messages_initialized_empty(self) -> None:
        """Verify `_current_tool_messages` is initialized as empty dict."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._current_tool_messages == {}

    def test_token_tracker_initialized_none(self) -> None:
        """Verify `_token_tracker` is initialized as `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._token_tracker is None

    def test_set_token_tracker(self) -> None:
        """Verify `set_token_tracker` stores the tracker."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        mock_tracker = object()
        adapter.set_token_tracker(mock_tracker)
        assert adapter._token_tracker is mock_tracker


class TestBuildStreamConfig:
    """Tests for `_build_stream_config` metadata construction."""

    def test_assistant_fields_present(self) -> None:
        """Assistant-specific metadata should be present when `assistant_id` is set."""
        config = _build_stream_config("t-456", assistant_id="my-agent")
        assert config["metadata"]["assistant_id"] == "my-agent"
        assert config["metadata"]["agent_name"] == "my-agent"
        assert "updated_at" in config["metadata"]

    def test_no_assistant_fields_when_none(self) -> None:
        """Assistant-specific fields should be absent when `assistant_id` is `None`."""
        config = _build_stream_config("t-789", assistant_id=None)
        assert config["metadata"] == {}

    def test_configurable_thread_id(self) -> None:
        """``configurable.thread_id`` should match the provided thread ID."""
        config = _build_stream_config("t-abc", assistant_id=None)
        assert config["configurable"]["thread_id"] == "t-abc"


class TestIsSummarizationChunk:
    """Tests for `_is_summarization_chunk` detection."""

    def test_returns_true_for_summarization_source(self) -> None:
        """Should return `True` when `lc_source` is `'summarization'`."""
        metadata = {"lc_source": "summarization"}
        assert _is_summarization_chunk(metadata) is True

    def test_returns_false_for_none_metadata(self) -> None:
        """Should return `False` when `metadata` is `None`."""
        assert _is_summarization_chunk(None) is False
        assert _is_summarization_chunk({}) is False

    def test_returns_false_for_none_lc_source(self) -> None:
        """Should return `False` when `lc_source` is not `'summarization'`."""
        metadata_none = {"lc_source": None}
        assert _is_summarization_chunk(metadata_none) is False

        metadata_other = {"lc_source": "other"}
        assert _is_summarization_chunk(metadata_other) is False

        metadata_missing = {"other_key": "value"}
        assert _is_summarization_chunk(metadata_missing) is False
