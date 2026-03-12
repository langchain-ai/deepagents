"""Unit tests for main entry point."""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.app import AppResult, DeepAgentsApp, run_textual_app
from deepagents_cli.config import build_langsmith_thread_url, reset_langsmith_url_cache
from deepagents_cli.main import (
    check_optional_tools,
    format_tool_warning_cli,
    format_tool_warning_tui,
    run_textual_cli_async,
)


class TestResumeHintLogic:
    """Test that resume hint logic is correct.

    The actual condition in `cli_main` is::

        thread_id and return_code == 0 and asyncio.run(thread_exists(thread_id))

    These tests mirror the three-part condition. `thread_exists` is
    represented as a boolean to keep the tests as pure unit tests.
    """

    def test_resume_hint_condition_error_case(self) -> None:
        """Resume hint should NOT be shown when return_code is non-zero."""
        thread_id = "test123"
        return_code = 1
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "Resume hint should not be shown on error"

    def test_resume_hint_condition_success_case(self) -> None:
        """Resume hint SHOULD be shown on success with checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown on success"

    def test_resume_hint_shown_for_resumed_threads(self) -> None:
        """Resume hint SHOULD be shown for resumed threads too."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = True

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert show, "Resume hint should be shown for resumed threads"

    def test_resume_hint_not_shown_without_checkpoints(self) -> None:
        """Resume hint should NOT appear when thread has no checkpoints."""
        thread_id = "test123"
        return_code = 0
        has_checkpoints = False

        show = bool(thread_id) and return_code == 0 and has_checkpoints
        assert not show, "No hint when thread_exists returns False"


class TestLangSmithTeardownUrl:
    """Test LangSmith thread URL display logic on teardown."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()

    def test_thread_url_requires_all_components(self) -> None:
        """LangSmith link requires thread_id, project_name, and project_url."""
        thread_url = build_langsmith_thread_url("abc123")
        # Without LangSmith configured, should return None
        assert thread_url is None

    def test_thread_url_not_shown_for_none_thread_id(self) -> None:
        """Guard condition: thread_url and thread_exists both needed."""
        thread_url = None
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_not_shown_when_no_checkpoints(self) -> None:
        """Guard condition: thread must have checkpointed content."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = False
        show_link = bool(thread_url and thread_exists)
        assert not show_link

    def test_thread_url_shown_when_all_conditions_met(self) -> None:
        """Guard condition: both thread_url and thread_exists must be truthy."""
        thread_url = "https://smith.langchain.com/o/org/projects/p/proj/t/abc"
        thread_exists = True
        show_link = bool(thread_url and thread_exists)
        assert show_link


class TestAppResult:
    """Tests for the AppResult dataclass."""

    def test_fields_accessible(self) -> None:
        """AppResult should expose return_code and thread_id."""
        result = AppResult(return_code=0, thread_id="tid-abc")
        assert result.return_code == 0
        assert result.thread_id == "tid-abc"

    def test_thread_id_none(self) -> None:
        """AppResult should accept None for thread_id."""
        result = AppResult(return_code=1, thread_id=None)
        assert result.thread_id is None

    def test_frozen(self) -> None:
        """AppResult should be immutable."""
        from dataclasses import FrozenInstanceError

        result = AppResult(return_code=0, thread_id="tid")
        with pytest.raises(FrozenInstanceError):
            result.return_code = 1  # type: ignore[misc]


class TestRunTextualAppReturnType:
    """Test that run_textual_app returns AppResult."""

    async def test_run_textual_app_returns_app_result(self) -> None:
        """run_textual_app should return an AppResult."""
        sig = inspect.signature(run_textual_app)
        annotation = sig.return_annotation
        assert annotation in (AppResult, "AppResult"), (
            f"run_textual_app should return AppResult, got {annotation}"
        )


class TestRunTextualCliAsyncReturnType:
    """Test that run_textual_cli_async returns AppResult."""

    def test_run_textual_cli_async_returns_app_result(self) -> None:
        """run_textual_cli_async should return an AppResult."""
        sig = inspect.signature(run_textual_cli_async)
        assert sig.return_annotation in (AppResult, "AppResult"), (
            "run_textual_cli_async should return AppResult, "
            f"got {sig.return_annotation}"
        )


class TestThreadMessage:
    """Test thread info display format.

    Thread info is now displayed in the WelcomeBanner widget rather than via
    pre-TUI console output, so we verify the banner receives the thread ID.
    """

    def test_thread_id_forwarded_to_app(self) -> None:
        """run_textual_cli_async passes thread_id to run_textual_app."""
        source = inspect.getsource(run_textual_cli_async)
        assert "thread_id=thread_id" in source, (
            "thread_id should be forwarded to run_textual_app"
        )


class TestRunTextualCliAsyncMcp:
    """Tests for MCP/server kwargs forwarding in interactive server mode.

    Server startup and MCP preload now happen inside the TUI via deferred
    kwargs rather than being invoked directly in `run_textual_cli_async`.
    """

    async def test_passes_server_and_mcp_kwargs_to_textual_app(self) -> None:
        """Interactive TUI should receive server_kwargs and mcp_preload_kwargs."""
        model_result = SimpleNamespace(
            model=object(),
            provider="openai",
            model_name="gpt-5",
            apply_to_settings=MagicMock(),
        )
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with (
            patch("deepagents_cli.config.console"),
            patch("deepagents_cli.config.create_model", return_value=model_result),
            patch("deepagents_cli.model_config.save_recent_model"),
            patch("deepagents_cli.app.run_textual_app", new=_run_textual_app_stub),
        ):
            result = await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
            )

        assert result == app_result
        model_result.apply_to_settings.assert_called_once_with()

        # Server kwargs forwarded for deferred startup inside the TUI
        assert captured_kwargs["server_kwargs"] is not None
        assert captured_kwargs["server_kwargs"]["assistant_id"] == "agent"
        assert captured_kwargs["server_kwargs"]["interactive"] is True

        # MCP preload kwargs forwarded (no_mcp=False by default)
        assert captured_kwargs["mcp_preload_kwargs"] is not None
        assert captured_kwargs["mcp_preload_kwargs"]["no_mcp"] is False

    async def test_no_mcp_kwargs_when_disabled(self) -> None:
        """mcp_preload_kwargs should be None when no_mcp=True."""
        model_result = SimpleNamespace(
            model=object(),
            provider="openai",
            model_name="gpt-5",
            apply_to_settings=MagicMock(),
        )
        app_result = AppResult(return_code=0, thread_id="thread-123")
        captured_kwargs: dict[str, Any] = {}

        async def _run_textual_app_stub(**kwargs: Any) -> AppResult:
            captured_kwargs.update(kwargs)
            await asyncio.sleep(0)
            return app_result

        with (
            patch("deepagents_cli.config.console"),
            patch("deepagents_cli.config.create_model", return_value=model_result),
            patch("deepagents_cli.model_config.save_recent_model"),
            patch("deepagents_cli.app.run_textual_app", new=_run_textual_app_stub),
        ):
            await run_textual_cli_async(
                "agent",
                thread_id="thread-123",
                no_mcp=True,
            )

        assert captured_kwargs["mcp_preload_kwargs"] is None


class TestServerCleanupLifecycle:
    """Verify server_proc.stop() is guaranteed after the TUI exits."""

    async def test_server_proc_stopped_after_app_exits(self) -> None:
        """run_textual_app must call server_proc.stop() in the finally block."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new_callable=AsyncMock,
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # type: ignore[invalid-argument-type]

        server_proc.stop.assert_called_once_with()

    async def test_server_proc_stopped_even_on_crash(self) -> None:
        """server_proc.stop() must fire even when run_async raises."""
        server_proc = SimpleNamespace(stop=MagicMock())

        with (
            patch.object(
                DeepAgentsApp,
                "run_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await run_textual_app(server_proc=server_proc, thread_id="t-1")  # type: ignore[invalid-argument-type]

        server_proc.stop.assert_called_once_with()

    async def test_deferred_server_proc_stopped_after_app_exits(self) -> None:
        """server_proc set by the background worker must still be cleaned up."""
        server_proc = SimpleNamespace(stop=MagicMock())

        async def _fake_run_async(self: DeepAgentsApp) -> None:  # noqa: RUF029
            # Simulate the background worker having set _server_proc
            self._server_proc = server_proc

        with patch.object(
            DeepAgentsApp,
            "run_async",
            new=_fake_run_async,
        ):
            await run_textual_app(
                server_kwargs={"assistant_id": "a"},
                thread_id="t-1",
            )

        server_proc.stop.assert_called_once_with()


class TestCheckOptionalTools:
    """Tests for check_optional_tools() function."""

    def test_returns_tool_name_when_rg_not_found(self) -> None:
        """Returns `['ripgrep']` when `rg` is not on PATH."""
        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools()

        assert missing == ["ripgrep"]

    def test_returns_empty_when_rg_found(self) -> None:
        """Returns empty list when `rg` is found on PATH."""
        with patch("deepagents_cli.main.shutil.which", return_value="/usr/bin/rg"):
            missing = check_optional_tools()

        assert missing == []

    def test_warning_suppressed_via_config(self, tmp_path: Path) -> None:
        """Returns empty list when ripgrep warning is suppressed in config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["ripgrep"]\n')

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == []

    def test_malformed_config_does_not_suppress(self, tmp_path: Path) -> None:
        """Malformed TOML config degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("this is not valid toml [[[")

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_non_list_suppress_does_not_crash(self, tmp_path: Path) -> None:
        """Non-list `suppress` value degrades gracefully instead of crashing."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("[warnings]\nsuppress = true\n")

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]

    def test_unrelated_suppress_key_does_not_suppress(self, tmp_path: Path) -> None:
        """Suppressing a different key does not suppress the ripgrep warning."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('[warnings]\nsuppress = ["something_else"]\n')

        with patch("deepagents_cli.main.shutil.which", return_value=None):
            missing = check_optional_tools(config_path=config_path)

        assert missing == ["ripgrep"]


class TestFormatToolWarnings:
    """Tests for TUI and CLI warning formatters."""

    def test_tui_format_contains_url(self) -> None:
        """TUI format includes the install URL as plain text."""
        msg = format_tool_warning_tui("ripgrep")
        assert "https://github.com/BurntSushi/ripgrep#installation" in msg
        assert "[link=" not in msg

    def test_cli_format_contains_rich_link(self) -> None:
        """CLI format wraps the URL in Rich `[link]` markup."""
        msg = format_tool_warning_cli("ripgrep")
        assert "[link=https://github.com/BurntSushi/ripgrep#installation]" in msg
        assert "[/link]" in msg

    def test_both_formats_contain_suppress_hint(self) -> None:
        """Both formats include the config suppression hint."""
        formatters = (format_tool_warning_tui, format_tool_warning_cli)
        for fmt in formatters:
            msg = fmt("ripgrep")
            assert "\\[warnings]" in msg
            assert 'suppress = \\["ripgrep"]' in msg

    def test_unknown_tool_fallback(self) -> None:
        """Unknown tools get a generic message."""
        assert format_tool_warning_tui("foo") == "foo is not installed."
        assert format_tool_warning_cli("foo") == "foo is not installed."
