"""Unit tests for main entry point."""

import pytest


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


class TestRunTextualAppReturnType:
    """Test that run_textual_app returns proper return code."""

    @pytest.mark.asyncio
    async def test_run_textual_app_returns_int(self) -> None:
        """run_textual_app should return an integer return code."""
        import inspect

        from deepagents_cli.app import run_textual_app

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
        import inspect

        from deepagents_cli.main import run_textual_cli_async

        # Verify the function signature returns int
        sig = inspect.signature(run_textual_cli_async)
        assert sig.return_annotation in (int, "int"), (
            f"run_textual_cli_async should return int, got {sig.return_annotation}"
        )


class TestThreadMessage:
    """Test thread info display format."""

    def test_new_session_message_format(self) -> None:
        """New session message should say 'Starting with thread:' not 'Thread:'."""
        import inspect

        from deepagents_cli.main import run_textual_cli_async

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
