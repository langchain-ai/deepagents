"""End-to-end unit tests for deepagents-cli with fake LLM models."""

import uuid
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool

from deepagents_cli.agent import create_cli_agent


@tool(description="Sample tool")
def sample_tool(sample_input: str) -> str:
    """A sample tool that returns the input string."""
    return sample_input


class FixedGenericFakeChatModel(GenericFakeChatModel):
    """Fixed version of GenericFakeChatModel that properly handles bind_tools."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Override bind_tools to return self."""
        return self


@contextmanager
def mock_settings(tmp_path: Path, assistant_id: str = "test-agent") -> Generator[Path, None, None]:
    """Context manager for patching CLI settings with temporary directories.

    Args:
        tmp_path: Temporary directory path (typically from pytest's tmp_path fixture)
        assistant_id: Agent identifier for directory setup

    Yields:
        The agent directory path
    """
    # Setup directory structure
    agent_dir = tmp_path / "agents" / assistant_id
    agent_dir.mkdir(parents=True)
    agent_md = agent_dir / "agent.md"
    agent_md.write_text("# Test Agent\nTest agent instructions.")

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True)

    # Patch settings
    with patch("deepagents_cli.agent.settings") as mock_settings_obj:
        mock_settings_obj.user_deepagents_dir = tmp_path / "agents"
        mock_settings_obj.ensure_agent_dir.return_value = agent_dir
        mock_settings_obj.ensure_user_skills_dir.return_value = skills_dir
        mock_settings_obj.get_project_skills_dir.return_value = None

        # Mock methods that get called during agent execution to return real Path objects
        # This prevents MagicMock objects from being stored in state (which would fail serialization)
        def get_user_agent_md_path(agent_id: str) -> Path:
            return tmp_path / "agents" / agent_id / "agent.md"

        def get_agent_dir(agent_id: str) -> Path:
            return tmp_path / "agents" / agent_id

        mock_settings_obj.get_user_agent_md_path = get_user_agent_md_path
        mock_settings_obj.get_project_agent_md_path.return_value = None
        mock_settings_obj.get_agent_dir = get_agent_dir
        mock_settings_obj.project_root = None

        yield agent_dir


class TestDeepAgentsCLIEndToEnd:
    """Test suite for end-to-end deepagents-cli functionality with fake LLM."""

    def test_cli_agent_with_fake_llm_basic(self, tmp_path: Path) -> None:
        """Test basic CLI agent functionality with a fake LLM model.

        This test verifies that a CLI agent can be created and invoked with
        a fake LLM model that returns predefined responses.
        """
        with mock_settings(tmp_path):
            # Create a fake model that returns predefined messages
            model = FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="I'll help you with that.",
                            tool_calls=[
                                {
                                    "name": "write_todos",
                                    "args": {"todos": []},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="Task completed successfully!",
                        ),
                    ]
                )
            )

            # Create a CLI agent with the fake model
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[],
            )

            # Invoke the agent with a simple message
            result = agent.invoke(
                {"messages": [HumanMessage(content="Hello, agent!")]},
                {"configurable": {"thread_id": str(uuid.uuid4())}},
            )

            # Verify the agent executed correctly
            assert "messages" in result
            assert len(result["messages"]) > 0

            # Verify we got AI responses
            ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
            assert len(ai_messages) > 0

            # Verify the final AI message contains our expected content
            final_ai_message = ai_messages[-1]
            assert "Task completed successfully!" in final_ai_message.content

    def test_cli_agent_with_fake_llm_with_tools(self, tmp_path: Path) -> None:
        """Test CLI agent with tools using a fake LLM model.

        This test verifies that a CLI agent can handle tool calls correctly
        when using a fake LLM model.
        """
        with mock_settings(tmp_path):
            # Create a fake model that calls sample_tool
            model = FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "sample_tool",
                                    "args": {"sample_input": "test input"},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="I called the sample_tool with 'test input'.",
                        ),
                    ]
                )
            )

            # Create a CLI agent with the fake model and sample_tool
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[sample_tool],
            )

            # Invoke the agent
            result = agent.invoke(
                {"messages": [HumanMessage(content="Use the sample tool")]},
                {"configurable": {"thread_id": "test-thread-2"}},
            )

            # Verify the agent executed correctly
            assert "messages" in result

            # Verify tool was called
            tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
            assert len(tool_messages) > 0

            # Verify the tool message contains our expected input
            assert any("test input" in msg.content for msg in tool_messages)

    def test_cli_agent_with_fake_llm_filesystem_tool(self, tmp_path: Path) -> None:
        """Test CLI agent with filesystem tools using a fake LLM model.

        This test verifies that a CLI agent can use the built-in filesystem
        tools (ls, read_file, etc.) with a fake LLM model.
        """
        with mock_settings(tmp_path):
            # Create a test file to list
            test_file = tmp_path / "test.txt"
            test_file.write_text("test content")

            # Create a fake model that uses filesystem tools
            model = FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "ls",
                                    "args": {"path": str(tmp_path)},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="I've listed the files in the directory.",
                        ),
                    ]
                )
            )

            # Create a CLI agent with the fake model
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[],
            )

            # Invoke the agent
            result = agent.invoke(
                {"messages": [HumanMessage(content="List files")]},
                {"configurable": {"thread_id": "test-thread-3"}},
            )

            # Verify the agent executed correctly
            assert "messages" in result

            # Verify ls tool was called
            tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
            assert len(tool_messages) > 0

    def test_cli_agent_with_fake_llm_multiple_tool_calls(self, tmp_path: Path) -> None:
        """Test CLI agent with multiple tool calls using a fake LLM model.

        This test verifies that a CLI agent can handle multiple sequential
        tool calls with a fake LLM model.
        """
        with mock_settings(tmp_path):
            # Create a fake model that makes multiple tool calls
            model = FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "sample_tool",
                                    "args": {"sample_input": "first call"},
                                    "id": "call_1",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "sample_tool",
                                    "args": {"sample_input": "second call"},
                                    "id": "call_2",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(
                            content="I completed both tool calls successfully.",
                        ),
                    ]
                )
            )

            # Create a CLI agent with the fake model and sample_tool
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[sample_tool],
            )

            # Invoke the agent
            result = agent.invoke(
                {"messages": [HumanMessage(content="Use sample tool twice")]},
                {"configurable": {"thread_id": "test-thread-4"}},
            )

            # Verify the agent executed correctly
            assert "messages" in result

            # Verify multiple tool calls occurred
            tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
            assert len(tool_messages) >= 2

            # Verify both inputs were used
            tool_contents = [msg.content for msg in tool_messages]
            assert any("first call" in content for content in tool_contents)
            assert any("second call" in content for content in tool_contents)

    def test_cli_agent_backend_setup(self, tmp_path: Path) -> None:
        """Test that CLI agent creates the correct backend setup.

        This test verifies that the backend is properly configured with
        a CompositeBackend containing a FilesystemBackend.
        """
        with mock_settings(tmp_path):
            # Create a simple fake model
            model = FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(content="Done."),
                    ]
                )
            )

            # Create a CLI agent
            agent, backend = create_cli_agent(
                model=model,
                assistant_id="test-agent",
                tools=[],
            )

            assert isinstance(backend, CompositeBackend)
            assert isinstance(backend.default, FilesystemBackend)
