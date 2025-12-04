"""Integration test for CLI with auto-approve mode.

This module implements benchmarking for simple tasks using the DeepAgents CLI; e.g.,
"write a poem to a file", "create multiple files", etc.

The agent runs on auto-approve mode, meaning it can perform actions without
user confirmation.

Note on testing approach:
- We use StringIO to capture console output, which is the recommended
  approach according to Rich's documentation for unit/integration tests.
- The capture() context manager is an alternative, but StringIO provides
  better control and is simpler for testing purposes.
- We patch console instances in both main and config modules to ensure
  all output is captured in the test.
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from deepagents_cli import config as config_module
from deepagents_cli import main as main_module
from deepagents_cli.agent import create_cli_agent
from deepagents_cli.config import SessionState, create_model
from deepagents_cli.main import simple_cli


@asynccontextmanager
async def run_cli_task(task: str, tmp_path: Path) -> AsyncIterator[tuple[Path, str]]:
    """Context manager to run a CLI task with auto-approve and capture output.

    Args:
        task: The task string to give to the agent
        tmp_path: Temporary directory for the test

    Yields:
        tuple: (working_directory: Path, console_output: str)
    """
    original_dir = Path.cwd()
    os.chdir(tmp_path)

    # Capture console output
    # Using StringIO is the recommended approach for testing (per Rich docs)
    output = StringIO()
    captured_console = Console(
        file=output,
        force_terminal=False,  # Disable ANSI codes for simpler assertions
        width=120,  # Fixed width for predictable output
        color_system=None,  # Explicitly disable colors for testing
        legacy_windows=False,  # Modern behavior
    )

    try:
        # Mock the prompt session to provide input and exit
        # Use patch.object() to fail immediately if attributes don't exist
        with patch.object(main_module, "create_prompt_session") as mock_prompt:
            mock_session = AsyncMock()
            mock_session.prompt_async.side_effect = [
                task,  # User input
                EOFError(),  # Exit after task
            ]
            mock_prompt.return_value = mock_session

            # Mock console to capture output
            # Use patch.object() to fail immediately if attributes don't exist
            with (
                patch.object(main_module, "console", captured_console),
                patch.object(config_module, "console", captured_console),
            ):
                # Import after patching
                from deepagents_cli.agent import create_cli_agent
                from deepagents_cli.config import create_model

                # Create real agent with real model (will use env var or fail gracefully)
                model = create_model()
                agent, backend = create_cli_agent(
                    model=model,
                    assistant_id="test_agent",
                    tools=[],
                    sandbox=None,
                    sandbox_type=None,
                )

                # Create session state with auto-approve
                session_state = SessionState(auto_approve=True)

                # Run the CLI
                await simple_cli(
                    agent=agent,
                    assistant_id="test_agent",
                    session_state=session_state,
                    baseline_tokens=0,
                    backend=backend,
                    sandbox_type=None,
                    setup_script_path=None,
                )

            # Verify that our mocks were actually used (ensures patching worked)
            mock_prompt.assert_called_once()
            assert mock_session.prompt_async.call_count >= 1, (
                "prompt_async should have been called at least once"
            )

        # Yield the directory and captured output
        yield tmp_path, output.getvalue()

    finally:
        os.chdir(original_dir)


@asynccontextmanager
async def run_agent_task_with_hitl(task: str, tmp_path: Path) -> AsyncIterator:
    """Context manager to run an agent task with HIL and stream events.

    Args:
        task: The task string to give to the agent
        tmp_path: Temporary directory for the test

    Yields:
        AsyncGenerator: Stream of events from the agent
    """
    original_dir = Path.cwd()
    os.chdir(tmp_path)

    try:
        # Create agent with HIL enabled (no auto-approve)
        model = create_model()
        checkpointer = MemorySaver()
        agent, _backend = create_cli_agent(
            model=model,
            assistant_id="test_agent",
            tools=[],
            sandbox=None,
            sandbox_type=None,
        )
        agent.checkpointer = checkpointer

        # Create config with thread_id for checkpointing
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Yield the stream generator for the test to consume
        yield agent.astream(
            {"messages": [{"role": "user", "content": task}]},
            config=config,
            stream_mode="values",
        )

    finally:
        os.chdir(original_dir)


class TestSimpleTasks:
    """A collection of simple task benchmarks for the deepagents-cli."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)  # Agent can take 60-120 seconds
    async def test_write_hello_to_a_file(self, tmp_path: Path) -> None:
        """Test agents to write 'hello' to a file."""
        async with run_cli_task("write hello to file foo.md", tmp_path) as (
            work_dir,
            console_output,
        ):
            # Verify the file was created
            output_file = work_dir / "foo.md"
            assert output_file.exists(), f"foo.md should have been created in {work_dir}"

            content = output_file.read_text()
            assert "hello" in content.lower(), f"File should contain 'hello', but got: {content!r}"

            # Verify console output shows auto-approve mode
            # Print output for debugging if assertion fails
            assert "Auto-approve" in console_output or "⚡" in console_output, (
                f"Expected auto-approve indicator in output.\nConsole output:\n{console_output}"
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_cli_auto_approve_multiple_operations(self, tmp_path: Path) -> None:
        """Test agent to create multiple files with auto-approve."""
        task = "create files test1.txt and test2.txt with content 'test file'"

        async with run_cli_task(task, tmp_path) as (work_dir, console_output):
            # Verify both files were created
            test1 = work_dir / "test1.txt"
            test2 = work_dir / "test2.txt"

            # At least one file should be created (agent might interpret task differently)
            created_files = [f for f in [test1, test2] if f.exists()]
            assert len(created_files) > 0, (
                f"Expected at least one test file to be created in {work_dir}.\n"
                f"Files in directory: {list(work_dir.iterdir())}"
            )

            # Verify console output captured the interaction
            assert len(console_output) > 0, "Console output should not be empty"


class TestAgentBehavior:
    """A collection of tests for agent behavior (non-CLI level)."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_run_command_calls_shell_tool(self, tmp_path: Path) -> None:
        """Test that 'run make format' calls shell tool with 'make format' command.

        This test verifies that when a user says "run make format", the agent
        correctly interprets this as a shell command and calls the shell tool
        with just "make format" (not including the word "run").

        The test stops at the interrupt (HITL approval point) before the shell
        tool is actually executed, to verify the correct command is being passed.
        """
        # Mock the settings to use a fresh filesystem in tmp_path
        from deepagents_cli.config import Settings

        mock_settings = Settings.from_environment(start_path=tmp_path)

        # Patch settings in all modules that import it
        patches = [
            patch("deepagents_cli.config.settings", mock_settings),
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.file_ops.settings", mock_settings),
            patch("deepagents_cli.tools.settings", mock_settings),
            patch("deepagents_cli.token_utils.settings", mock_settings),
        ]

        # Apply all patches using ExitStack for cleaner nesting
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)

            async with run_agent_task_with_hitl("run make format", tmp_path) as stream:
                # Stream events and capture the final result
                events = []
                result = {}
                async for event in stream:
                    events.append(event)
                    result = event

                # Verify that we captured events
                assert len(events) > 0, "Expected to receive events from agent stream"

                # Verify that an interrupt occurred (shell tool requires approval)
                assert "__interrupt__" in result, "Expected shell tool to trigger HITL interrupt"
                assert result["__interrupt__"] is not None

                # Extract interrupt information
                interrupts = result["__interrupt__"]
                assert len(interrupts) > 0, "Expected at least one interrupt"

                interrupt_value = interrupts[0].value
                action_requests = interrupt_value.get("action_requests", [])

                # Verify that a shell tool call is present
                shell_calls = [req for req in action_requests if req.get("name") == "shell"]
                assert len(shell_calls) > 0, "Expected at least one shell tool call"

                # Verify the shell command is "make format" (not "run make format")
                shell_call = shell_calls[0]
                command = shell_call.get("args", {}).get("command", "")
                assert command == "make format", (
                    f"Expected shell command to be 'make format', got: {command}"
                )
