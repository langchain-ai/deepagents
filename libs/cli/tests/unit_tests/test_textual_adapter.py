"""Unit tests for textual_adapter functions."""

import asyncio
from asyncio import Future
from collections.abc import AsyncIterator, Generator
from datetime import datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import ValidationError
from rich.console import Console

from deepagents_cli import textual_adapter
from deepagents_cli.textual_adapter import (
    ModelStats,
    SessionStats,
    TextualUIAdapter,
    _build_interrupted_ai_message,
    _handle_interrupt_cleanup,
    _is_summarization_chunk,
    execute_task_textual,
    format_token_count,
    print_usage_table,
)
from deepagents_cli.widgets.messages import SummarizationMessage


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

    def test_token_callbacks_initialized_none(self) -> None:
        """Verify token callbacks are initialized as `None`."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )
        assert adapter._on_tokens_update is None
        assert adapter._on_tokens_hide is None
        assert adapter._on_tokens_show is None

    def test_set_token_callbacks(self) -> None:
        """Verify token callbacks can be assigned."""
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        def update_cb(count: int, *, approximate: bool = False) -> None:
            pass

        def hide_cb() -> None:
            pass

        def show_cb(*, approximate: bool = False) -> None:
            pass

        adapter._on_tokens_update = update_cb
        adapter._on_tokens_hide = hide_cb
        adapter._on_tokens_show = show_cb
        assert adapter._on_tokens_update is update_cb
        assert adapter._on_tokens_hide is hide_cb
        assert adapter._on_tokens_show is show_cb

    def test_finalize_pending_tools_with_error_marks_and_clears(self) -> None:
        """Pending tool widgets should be marked error and then cleared."""
        set_active = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_active_message=set_active,
        )

        tool_1 = MagicMock()
        tool_2 = MagicMock()
        adapter._current_tool_messages = {"a": tool_1, "b": tool_2}

        adapter.finalize_pending_tools_with_error("Agent error: boom")

        tool_1.set_error.assert_called_once_with("Agent error: boom")
        tool_2.set_error.assert_called_once_with("Agent error: boom")
        assert adapter._current_tool_messages == {}
        set_active.assert_called_once_with(None)


class TestInterruptCleanup:
    """Tests for interrupt cleanup token handling."""

    async def test_tool_only_interrupt_marks_tokens_approximate(self) -> None:
        """Tool-only interrupted turns should keep the stale-token marker."""
        mounted: list[object] = []

        async def mount_message(widget: object) -> None:
            mounted.append(widget)
            await asyncio.sleep(0)

        set_spinner = AsyncMock()
        set_active = MagicMock()
        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=set_spinner,
            set_active_message=set_active,
        )

        tool_widget = MagicMock()
        tool_widget._tool_name = "read_file"
        tool_widget._args = {"path": "notes.txt"}
        adapter._current_tool_messages = {"call-1": tool_widget}

        show_calls: list[bool] = []

        def show_cb(*, approximate: bool = False) -> None:
            show_calls.append(approximate)

        adapter._on_tokens_show = show_cb

        agent = SimpleNamespace(aupdate_state=AsyncMock())
        turn_stats = SessionStats()
        config = {"configurable": {"thread_id": "t-1"}}

        with patch("deepagents_cli.textual_adapter.time.monotonic", return_value=101.0):
            await _handle_interrupt_cleanup(
                adapter=adapter,
                agent=agent,
                config=config,  # type: ignore[arg-type]
                pending_text_by_namespace={},
                captured_input_tokens=0,
                captured_output_tokens=0,
                turn_stats=turn_stats,
                start_time=100.0,
            )

        assert mounted
        assert show_calls == [True]
        assert turn_stats.wall_time_seconds == 1.0
        set_active.assert_called_once_with(None)
        set_spinner.assert_awaited_once_with(None)
        tool_widget.set_rejected.assert_called_once_with()
        assert adapter._current_tool_messages == {}

        interrupted_payload = agent.aupdate_state.await_args_list[0].args[1]
        interrupted_msg = interrupted_payload["messages"][0]
        assert interrupted_msg.tool_calls[0]["id"] == "call-1"
        assert interrupted_msg.tool_calls[0]["name"] == "read_file"


class TestBuildStreamConfig:
    """Tests for `_build_stream_config` metadata construction."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        textual_adapter._git_branch_cache.clear()

    def test_assistant_fields_present(self) -> None:
        """Assistant-specific metadata should be present when `assistant_id` is set."""
        config = _build_stream_config("t-456", assistant_id="my-agent")
        assert config["metadata"]["assistant_id"] == "my-agent"
        assert config["metadata"]["agent_name"] == "my-agent"
        assert "updated_at" in config["metadata"]
        assert "cwd" in config["metadata"]

    def test_updated_at_is_valid_iso_timestamp(self) -> None:
        """`updated_at` should be a valid timezone-aware ISO 8601 timestamp."""
        config = _build_stream_config("t-456", assistant_id="my-agent")
        raw = config["metadata"]["updated_at"]
        assert isinstance(raw, str)
        parsed = datetime.fromisoformat(raw)
        assert parsed.tzinfo is not None

    def test_no_assistant_fields_when_none(self) -> None:
        """Assistant-specific fields should be absent when `assistant_id` is `None`."""
        config = _build_stream_config("t-789", assistant_id=None)
        metadata = config["metadata"]
        assert "assistant_id" not in metadata
        assert "agent_name" not in metadata
        assert "updated_at" not in metadata
        assert "cwd" in metadata

    def test_no_assistant_fields_when_empty_string(self) -> None:
        """Empty-string `assistant_id` should be treated as absent."""
        config = _build_stream_config("t-000", assistant_id="")
        metadata = config["metadata"]
        assert "assistant_id" not in metadata
        assert "agent_name" not in metadata
        assert "updated_at" not in metadata
        assert "cwd" in metadata

    def test_git_branch_included_when_available(self) -> None:
        """Git branch should be included in metadata when in a git repo."""
        with patch(
            "deepagents_cli.textual_adapter._get_git_branch",
            return_value="feature-branch",
        ):
            config = _build_stream_config("t-git", assistant_id="agent")
        assert config["metadata"]["git_branch"] == "feature-branch"

    def test_git_branch_absent_when_not_in_repo(self) -> None:
        """Git branch should be absent when not in a git repo."""
        with patch(
            "deepagents_cli.textual_adapter._get_git_branch",
            return_value=None,
        ):
            config = _build_stream_config("t-nogit", assistant_id="agent")
        assert "git_branch" not in config["metadata"]

    def test_configurable_thread_id(self) -> None:
        """`configurable.thread_id` should match the provided thread ID."""
        config = _build_stream_config("t-abc", assistant_id=None)
        assert config["configurable"]["thread_id"] == "t-abc"

    def test_no_model_keys_in_configurable(self) -> None:
        """Model/model_params should not be in configurable."""
        config = _build_stream_config("t-no-model", assistant_id=None)
        assert "model" not in config["configurable"]
        assert "model_params" not in config["configurable"]

    def test_versions_contains_cli_version(self) -> None:
        """CLI version should always be present in metadata.versions."""
        from deepagents_cli._version import __version__

        config = build_stream_config("t-ver", assistant_id=None)
        assert config["metadata"]["versions"]["deepagents-cli"] == __version__

    def test_versions_contains_sdk_version_when_installed(self) -> None:
        """SDK version should be in versions when deepagents is installed."""
        with patch(
            "importlib.metadata.version",
            return_value="0.5.0",
        ):
            config = build_stream_config("t-sdk", assistant_id=None)
        assert config["metadata"]["versions"]["deepagents"] == "0.5.0"

    def test_versions_omits_sdk_when_not_installed(self) -> None:
        """SDK version key should be absent when deepagents is not installed."""
        from importlib.metadata import PackageNotFoundError

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("deepagents"),
        ):
            config = build_stream_config("t-nosdk", assistant_id=None)
        assert "deepagents" not in config["metadata"]["versions"]
        from deepagents_cli._version import __version__

        assert config["metadata"]["versions"]["deepagents-cli"] == __version__

    def test_user_id_included_when_set(self) -> None:
        """DEEPAGENTS_CLI_USER_ID should appear in metadata when set."""
        with patch.dict("os.environ", {"DEEPAGENTS_CLI_USER_ID": "mason"}):
            config = build_stream_config("t-uid", assistant_id=None)
        assert config["metadata"]["user_id"] == "mason"

    def test_user_id_absent_when_unset(self) -> None:
        """user_id should be absent from metadata when env var is not set."""
        with patch.dict("os.environ", {"DEEPAGENTS_CLI_USER_ID": ""}):
            config = build_stream_config("t-nouid", assistant_id=None)
        assert "user_id" not in config["metadata"]


class TestGetGitBranch:
    """Tests for `_get_git_branch` caching."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        textual_adapter._git_branch_cache.clear()

    def test_reuses_cached_branch_for_same_working_directory(self) -> None:
        """Repeated lookups in one repo should only spawn `git` once."""
        result = MagicMock(returncode=0, stdout="feature-branch\n")

        with (
            patch(
                "deepagents_cli.textual_adapter.Path.cwd",
                return_value=Path("/tmp/repo"),
            ),
            patch("subprocess.run", return_value=result) as mock_run,
        ):
            assert textual_adapter._get_git_branch() == "feature-branch"
            assert textual_adapter._get_git_branch() == "feature-branch"

        assert mock_run.call_count == 1


class TestGetGitBranchOSError:
    """Tests for _get_git_branch when Path.cwd() raises OSError."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        textual_adapter._git_branch_cache.clear()

    def test_returns_none_on_cwd_oserror(self) -> None:
        """_get_git_branch should return None when cwd is inaccessible."""
        with patch(
            "deepagents_cli.textual_adapter.Path.cwd",
            side_effect=OSError("deleted"),
        ):
            assert textual_adapter._get_git_branch() is None


class TestBuildStreamConfigOSError:
    """Tests for _build_stream_config when Path.cwd() raises OSError."""

    def setup_method(self) -> None:
        """Clear the git-branch cache between tests."""
        textual_adapter._git_branch_cache.clear()

    def test_cwd_absent_on_oserror(self) -> None:
        """Cwd should be absent from metadata when Path.cwd() raises."""
        with patch(
            "deepagents_cli.textual_adapter.Path.cwd",
            side_effect=OSError("deleted"),
        ):
            config = _build_stream_config("t-err", assistant_id="agent")
        assert "cwd" not in config["metadata"]


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

    def test_returns_false_for_unrelated_metadata(self) -> None:
        """Should return `False` when only unrelated keys are present."""
        assert _is_summarization_chunk({"langgraph_node": "model"}) is False
        assert _is_summarization_chunk({"langgraph_node": None}) is False


class _FakeAgent:
    """Minimal async stream agent used for adapter execution tests."""

    def __init__(self, chunks: list[tuple]) -> None:
        self._chunks = chunks

    async def astream(self, *_: Any, **__: Any) -> AsyncIterator[tuple[Any, ...]]:
        """Yield preconfigured stream chunks."""
        for chunk in self._chunks:
            yield chunk


class _SequencedAgent:
    """Agent test double that returns a different stream per call."""

    def __init__(self, streams_by_call: list[list[tuple[Any, ...]]]) -> None:
        self._streams_by_call = streams_by_call
        self.stream_inputs: list[dict | Command] = []

    async def astream(
        self,
        stream_input: dict | Command,
        *_: Any,
        **__: Any,
    ) -> AsyncIterator[tuple[Any, ...]]:
        """Yield chunks for this invocation and record stream inputs."""
        self.stream_inputs.append(stream_input)
        chunks = self._streams_by_call.pop(0) if self._streams_by_call else []
        for chunk in chunks:
            yield chunk


def _ask_user_interrupt_chunk(payload: dict[str, Any]) -> tuple[Any, ...]:
    """Build an updates-stream chunk containing one ask_user interrupt."""
    interrupt = SimpleNamespace(id="interrupt-1", value=payload)
    return ((), "updates", {"__interrupt__": [interrupt]})


class TestExecuteTaskTextualSummarizationFeedback:
    """Tests for summarization spinner and notification feedback."""

    async def test_spinner_transitions_for_summarization_stream(self) -> None:
        """Spinner should move Thinking -> Summarizing -> Thinking."""
        statuses: list[str | None] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def mount_message(_widget: object) -> None:
            await asyncio.sleep(0)

        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
            ((), "messages", (HumanMessage(content="regular chunk"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert statuses[0] == "Thinking"
        assert "Summarizing" in statuses
        assert statuses[-1] == "Thinking"

    async def test_mounts_summarization_notification_on_regular_chunk(self) -> None:
        """Notification should render when regular chunks resume after summarization."""
        statuses: list[str | None] = []
        mounted_widgets: list[object] = []

        async def record_spinner(status: str | None) -> None:
            await asyncio.sleep(0)
            statuses.append(status)

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted_widgets.append(widget)

        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
            # Regular chunk from the actual model — signals summarization ended.
            ((), "messages", (HumanMessage(content="regular"), {})),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert any(
            isinstance(widget, SummarizationMessage) for widget in mounted_widgets
        )

    async def test_mounts_notification_when_stream_ends_mid_summarization(self) -> None:
        """Notification should still render if stream exhausts during summarization."""
        mounted_widgets: list[object] = []

        async def record_spinner(_status: str | None) -> None:
            await asyncio.sleep(0)

        async def mount_message(widget: object) -> None:
            await asyncio.sleep(0)
            mounted_widgets.append(widget)

        # Only summarization chunks, no regular chunks follow.
        chunks = [
            (
                (),
                "messages",
                (AIMessage(content="summary chunk"), {"lc_source": "summarization"}),
            ),
        ]

        adapter = TextualUIAdapter(
            mount_message=mount_message,
            update_status=_noop_status,
            request_approval=_mock_approval,
            set_spinner=record_spinner,
        )

        await execute_task_textual(
            user_input="hello",
            agent=_FakeAgent(chunks),
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert any(
            isinstance(widget, SummarizationMessage) for widget in mounted_widgets
        )


class TestExecuteTaskTextualAskUser:
    """Tests for ask_user interrupt handling in the Textual adapter."""

    async def test_request_ask_user_returning_none_is_reported_as_error(self) -> None:
        """A `None` callback result should resume with explicit error status."""

        async def request_ask_user(
            _questions: list[Any],
        ) -> asyncio.Future[object] | None:
            await asyncio.sleep(0)
            return None

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        assert len(agent.stream_inputs) >= 2
        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "ask_user callback returned no response"
        assert ask_user_resume["answers"] == [""]

    async def test_request_ask_user_mount_error_is_not_treated_as_cancel(self) -> None:
        """UI mount failures should resume with explicit error status."""

        async def request_ask_user(
            _questions: list[Any],
        ) -> asyncio.Future[object] | None:
            await asyncio.sleep(0)
            msg = "boom"
            raise RuntimeError(msg)

        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=request_ask_user,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "failed to display ask_user prompt"
        assert ask_user_resume["answers"] == [""]

    async def test_request_ask_user_missing_callback_is_reported_as_error(self) -> None:
        """ask_user interrupts without a UI callback should resume with error."""
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            "questions": [{"question": "Name?", "type": "text"}],
                            "tool_call_id": "tool-1",
                        }
                    )
                ],
                [],
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
            request_ask_user=None,
        )

        await execute_task_textual(
            user_input="hello",
            agent=agent,
            assistant_id="assistant",
            session_state=SimpleNamespace(thread_id="thread-1", auto_approve=False),
            adapter=adapter,
        )

        resume_cmd = agent.stream_inputs[1]
        assert isinstance(resume_cmd, Command)
        resume_payload = cast("dict[str, dict[str, Any]]", resume_cmd.resume)
        ask_user_resume = resume_payload["interrupt-1"]
        assert ask_user_resume["status"] == "error"
        assert ask_user_resume["error"] == "ask_user not supported by this UI"
        assert ask_user_resume["answers"] == [""]

    async def test_invalid_ask_user_interrupt_payload_raises_validation_error(
        self,
    ) -> None:
        """Missing required ask_user keys should fail validation at ingestion."""
        agent = _SequencedAgent(
            streams_by_call=[
                [
                    _ask_user_interrupt_chunk(
                        {
                            "type": "ask_user",
                            # Missing required keys: `questions` and `tool_call_id`.
                        }
                    )
                ]
            ]
        )
        adapter = TextualUIAdapter(
            mount_message=_mock_mount,
            update_status=_noop_status,
            request_approval=_mock_approval,
        )

        with pytest.raises(ValidationError):
            await execute_task_textual(
                user_input="hello",
                agent=agent,
                assistant_id="assistant",
                session_state=SimpleNamespace(
                    thread_id="thread-1",
                    auto_approve=False,
                ),
                adapter=adapter,
            )


# ---------------------------------------------------------------------------
# Helpers for dict-iteration safety tests
# ---------------------------------------------------------------------------


def _make_tool_widget(name: str = "tool", args: dict | None = None) -> MagicMock:
    """Create a MagicMock that mimics a ToolCallMessage widget."""
    widget = MagicMock()
    widget._tool_name = name
    widget._args = args or {}
    return widget


class _MutatingItemsDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.items()` deletes another key mid-iteration.

    This deterministically reproduces the `RuntimeError: dictionary
    changed size during iteration` that occurs when async tool-result
    callbacks mutate `_current_tool_messages` while the HITL approval
    loop is iterating over it.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def items(self) -> Generator[tuple[str, Any], None, None]:  # type: ignore[override]
        """Yield items while mutating the dict mid-iteration."""
        it = iter(dict.items(self))
        first = next(it)
        # Remove a *different* key while iteration is in progress.
        remaining = [k for k in self if k != first[0]]
        if remaining:
            del self[remaining[0]]
        yield first
        yield from it


class _MutatingValuesDict(dict):  # noqa: FURB189  # must subclass dict to override C-level iteration
    """Dict whose `.values()` deletes a key mid-iteration.

    We intentionally subclass `dict` (not `UserDict`) because we
    need to override the C-level iteration that triggers the error.
    """

    def values(self) -> Generator[Any, None, None]:  # type: ignore[override]
        """Yield values while mutating the dict mid-iteration."""
        it = iter(dict.values(self))
        first = next(it)
        # Remove the first key to trigger size-change error.
        first_key = next(iter(self))
        del self[first_key]
        yield first
        yield from it


class TestDictIterationSafety:
    """Regression tests for #956.

    Parallel tool calls can modify `adapter._current_tool_messages`
    while another coroutine iterates over it, raising
    `RuntimeError: dictionary changed size during iteration`.

    The fix wraps every iteration with `list()` so a snapshot is
    taken before the loop body runs.  These tests prove the fix is
    necessary and sufficient.
    """

    # -- Test A: bare iteration over a mutating dict raises ----

    def test_items_iteration_fails_without_list(self) -> None:
        """Iterating .items() on a concurrently-mutated dict raises."""
        d = _MutatingItemsDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.items():
                pass

    def test_values_iteration_fails_without_list(self) -> None:
        """Iterating .values() on a concurrently-mutated dict raises."""
        d = _MutatingValuesDict(
            {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(3)}
        )
        with pytest.raises(RuntimeError, match="changed size"):
            for _ in d.values():
                pass

    # -- Test B: list() snapshot protects iteration ----

    def test_items_iteration_safe_with_list(self) -> None:
        """`list(d.items())` snapshots before mutation can occur."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        for key, _val in list(d.items()):
            collected.append(key)
            d.pop(key, None)  # mutate during loop body
        assert len(collected) == 5
        assert len(d) == 0

    def test_values_iteration_safe_with_list(self) -> None:
        """`list(d.values())` snapshots before mutation."""
        d: dict = {f"id_{i}": _make_tool_widget(f"t{i}") for i in range(5)}
        collected = []
        keys = list(d.keys())
        for val in list(d.values()):
            collected.append(val)
            if keys:
                d.pop(keys.pop(0), None)
        assert len(collected) == 5

    # -- Test C: _build_interrupted_ai_message uses list() ----

    def test_build_interrupted_ai_message_safe(self) -> None:
        """_build_interrupted_ai_message correctly builds an AIMessage.

        Verifies the function reconstructs tool calls and content from
        the provided widget dict. The `list()` snapshot inside the
        production code protects against external async mutation at
        `await` boundaries, which cannot be deterministically simulated
        in a synchronous unit test.
        """
        widgets = {
            f"id_{i}": _make_tool_widget(f"tool_{i}", {"k": i}) for i in range(4)
        }
        pending_text: dict[tuple, str] = {(): "hello"}
        result = _build_interrupted_ai_message(pending_text, widgets)
        assert result is not None
        assert result.content == "hello"
        assert len(result.tool_calls) == 4
        names = {tc["name"] for tc in result.tool_calls}
        assert names == {"tool_0", "tool_1", "tool_2", "tool_3"}

    def test_build_interrupted_ai_message_empty(self) -> None:
        """Returns None when there is no text and no tool calls."""
        result = _build_interrupted_ai_message({}, {})
        assert result is None


# ---------------------------------------------------------------------------
# SessionStats tests
# ---------------------------------------------------------------------------


class TestSessionStats:
    """Tests for `SessionStats` recording and merging."""

    def test_record_request_named_model(self) -> None:
        """record_request updates totals and per_model for a named model."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)

        assert stats.request_count == 1
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50
        assert "gpt-4" in stats.per_model
        assert stats.per_model["gpt-4"].request_count == 1
        assert stats.per_model["gpt-4"].input_tokens == 100
        assert stats.per_model["gpt-4"].output_tokens == 50

    def test_record_request_empty_model(self) -> None:
        """record_request with empty model skips per_model entry."""
        stats = SessionStats()
        stats.record_request("", 200, 80)

        assert stats.request_count == 1
        assert stats.input_tokens == 200
        assert stats.output_tokens == 80
        assert stats.per_model == {}

    def test_record_request_multiple_models(self) -> None:
        """Multiple models create separate per_model entries."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)

        assert stats.request_count == 2
        assert stats.input_tokens == 300
        assert stats.output_tokens == 130
        assert len(stats.per_model) == 2
        assert stats.per_model["gpt-4"].request_count == 1
        assert stats.per_model["claude-opus-4-6"].request_count == 1

    def test_merge(self) -> None:
        """merge() folds another SessionStats into self."""
        a = SessionStats(
            request_count=1, input_tokens=100, output_tokens=50, wall_time_seconds=1.0
        )
        a.per_model["gpt-4"] = ModelStats(
            request_count=1, input_tokens=100, output_tokens=50
        )

        b = SessionStats(
            request_count=2, input_tokens=300, output_tokens=120, wall_time_seconds=2.5
        )
        b.per_model["claude-opus-4-6"] = ModelStats(
            request_count=2, input_tokens=300, output_tokens=120
        )

        a.merge(b)

        assert a.request_count == 3
        assert a.input_tokens == 400
        assert a.output_tokens == 170
        assert a.wall_time_seconds == pytest.approx(3.5)
        assert len(a.per_model) == 2
        assert a.per_model["claude-opus-4-6"].request_count == 2

    def test_merge_overlapping_models(self) -> None:
        """merge() combines per_model entries for the same model."""
        a = SessionStats()
        a.record_request("gpt-4", 100, 50)

        b = SessionStats()
        b.record_request("gpt-4", 200, 80)

        a.merge(b)

        assert a.request_count == 2
        assert a.input_tokens == 300
        assert a.output_tokens == 130
        assert a.per_model["gpt-4"].request_count == 2
        assert a.per_model["gpt-4"].input_tokens == 300
        assert a.per_model["gpt-4"].output_tokens == 130


# ---------------------------------------------------------------------------
# format_token_count tests
# ---------------------------------------------------------------------------


class TestFormatTokenCount:
    """Tests for `format_token_count` shared formatter."""

    def test_small_count(self) -> None:
        assert format_token_count(500) == "500"

    def test_thousands(self) -> None:
        assert format_token_count(12_500) == "12.5K"

    def test_millions(self) -> None:
        assert format_token_count(1_200_000) == "1.2M"

    def test_exact_thousand(self) -> None:
        assert format_token_count(1000) == "1.0K"

    def test_zero(self) -> None:
        assert format_token_count(0) == "0"


# ---------------------------------------------------------------------------
# print_usage_table tests
# ---------------------------------------------------------------------------


class TestPrintUsageTable:
    """Tests for `print_usage_table` output."""

    def test_no_model_called_skips_unknown_row(self) -> None:
        """When no model was called, the table should not show 'unknown'."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=1.5, console=console)
        output = buf.getvalue()
        assert "unknown" not in output
        assert "Usage Stats" not in output
        assert "Agent active" in output

    def test_single_model_shows_name(self) -> None:
        """Single-model session should display the model name."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "unknown" not in output

    def test_multi_model_shows_all_names_and_total(self) -> None:
        """Multi-model session should show each model and a Total row."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        stats.record_request("claude-opus-4-6", 200, 80)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=2.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "claude-opus-4-6" in output
        assert "Total" in output
        assert "unknown" not in output

    def test_tokens_with_no_wall_time_omits_timing_line(self) -> None:
        """Token table should print but timing line should be absent."""
        stats = SessionStats()
        stats.record_request("gpt-4", 100, 50)
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.0, console=console)
        output = buf.getvalue()
        assert "gpt-4" in output
        assert "Agent active" not in output

    def test_no_requests_no_time_prints_nothing(self) -> None:
        """Empty stats with negligible wall time should print nothing."""
        stats = SessionStats()
        buf = StringIO()
        console = Console(file=buf, force_terminal=True)
        print_usage_table(stats, wall_time=0.01, console=console)
        output = buf.getvalue()
        assert output.strip() == ""
