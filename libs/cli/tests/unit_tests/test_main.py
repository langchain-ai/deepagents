"""Unit tests for main entry point."""

import inspect

import pytest

from deepagents_cli.app import run_textual_app
from deepagents_cli.config import build_langsmith_thread_url, reset_langsmith_url_cache
from deepagents_cli.main import run_textual_cli_async


class TestResumeHintLogic:
    """Test that resume hint logic is correct."""

    def test_resume_hint_condition_error_case(self) -> None:
        """Resume hint should NOT be shown when return_code is non-zero.

        This tests the condition: thread_id and not is_resumed and return_code == 0
        """
        # Simulating the condition from main.py
        thread_id = "test123"
        is_resumed = False
        return_code = 1  # Error case

        show_resume_hint = thread_id and not is_resumed and return_code == 0
        assert not show_resume_hint, "Resume hint should not be shown on error"

    def test_resume_hint_condition_success_case(self) -> None:
        """Resume hint SHOULD be shown when return_code is 0 (success)."""
        thread_id = "test123"
        is_resumed = False
        return_code = 0  # Success case

        show_resume_hint = thread_id and not is_resumed and return_code == 0
        assert show_resume_hint, "Resume hint should be shown on success"

    def test_resume_hint_not_shown_for_resumed_threads(self) -> None:
        """Resume hint should NOT be shown for resumed threads."""
        thread_id = "test123"
        is_resumed = True  # Resumed session
        return_code = 0

        show_resume_hint = thread_id and not is_resumed and return_code == 0
        assert not show_resume_hint, "Resume hint not shown for resumed threads"


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


class TestRunTextualAppReturnType:
    """Test that run_textual_app returns proper return code."""

    @pytest.mark.asyncio
    async def test_run_textual_app_returns_int(self) -> None:
        """run_textual_app should return an integer return code."""
        # Verify the function signature returns int
        sig = inspect.signature(run_textual_app)
        # Handle both 'int' string and int type (forward refs)
        annotation = sig.return_annotation
        assert annotation in (int, "int"), (
            f"run_textual_app should return int, got {annotation}"
        )


class TestRunTextualCliAsyncReturnType:
    """Test that run_textual_cli_async returns proper return code."""

    def test_run_textual_cli_async_returns_int(self) -> None:
        """run_textual_cli_async should return an integer return code."""
        # Verify the function signature returns int
        sig = inspect.signature(run_textual_cli_async)
        assert sig.return_annotation in (int, "int"), (
            f"run_textual_cli_async should return int, got {sig.return_annotation}"
        )


class TestThreadMessage:
    """Test thread info display format."""

    def test_new_session_message_format(self) -> None:
        """New session message should say 'Starting with thread:' not 'Thread:'."""
        # This tests that the format is correct by checking the source
        source = inspect.getsource(run_textual_cli_async)
        assert "Starting with thread:" in source, (
            "New session should show 'Starting with thread:' message"
        )
        # Should not have the old format (Thread: without Starting)
        # Note: "Resuming thread:" is still valid for resumed sessions
        lines = [
            line
            for line in source.split("\n")
            if "Thread:" in line and "Resuming" not in line and "Starting" not in line
        ]
        assert len(lines) == 0, f"Should not have old 'Thread:' format. Found: {lines}"
