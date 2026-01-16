"""End-to-end unit tests for deepagents with fake LLM models."""

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.store.memory import InMemoryStore

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.graph import create_deep_agent
from tests.utils import assert_all_deepagent_qualities


class SystemMessageCapturingMiddleware(AgentMiddleware):
    """Middleware that captures the system message for testing purposes."""

    def __init__(self) -> None:
        self.captured_system_messages: list = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if request.system_message is not None:
            self.captured_system_messages.append(request.system_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if request.system_message is not None:
            self.captured_system_messages.append(request.system_message)
        return await handler(request)


@tool(description="Sample tool")
def sample_tool(sample_input: str) -> str:
    """A sample tool that returns the input string."""
    return sample_input


def make_runtime(tid: str = "tc") -> ToolRuntime:
    """Create a ToolRuntime for testing."""
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tid,
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def create_filesystem_backend_virtual(tmp_path: Path) -> BackendProtocol:
    """Create a FilesystemBackend in virtual mode."""
    return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)


def create_state_backend(tmp_path: Path) -> BackendProtocol:  # noqa: ARG001
    """Create a StateBackend."""
    return StateBackend(make_runtime())


def create_store_backend(tmp_path: Path) -> BackendProtocol:  # noqa: ARG001
    """Create a StoreBackend."""
    return StoreBackend(make_runtime())


# Backend factories for parametrization
BACKEND_FACTORIES = [
    pytest.param(create_filesystem_backend_virtual, id="filesystem_virtual"),
    pytest.param(create_state_backend, id="state"),
    pytest.param(create_store_backend, id="store"),
]


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


class TestDeepAgentEndToEnd:
    """Test suite for end-to-end deepagent functionality with fake LLM."""

    def test_deep_agent_with_fake_llm_basic(self) -> None:
        """Test basic deepagent functionality with a fake LLM model.

        This test verifies that a deepagent can be created and invoked with
        a fake LLM model that returns predefined responses.
        """
        # Create a fake model that returns predefined messages
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="I'll use the sample_tool to process your request.",
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

        # Create a deep agent with the fake model
        agent = create_deep_agent(model=model)

        # Invoke the agent with a simple message
        result = agent.invoke({"messages": [HumanMessage(content="Hello, agent!")]})

        # Verify the agent executed correctly
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Verify we got AI responses
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        assert len(ai_messages) > 0

        # Verify the final AI message contains our expected content
        final_ai_message = ai_messages[-1]
        assert "Task completed successfully!" in final_ai_message.content

    def test_deep_agent_with_fake_llm_with_tools(self) -> None:
        """Test deepagent with tools using a fake LLM model.

        This test verifies that a deepagent can handle tool calls correctly
        when using a fake LLM model.
        """
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

        # Create a deep agent with the fake model and sample_tool
        agent = create_deep_agent(model=model, tools=[sample_tool])

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="Use the sample tool")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify tool was called
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        # Verify the tool message contains our expected input
        assert any("test input" in msg.content for msg in tool_messages)

    def test_deep_agent_with_fake_llm_filesystem_tool(self) -> None:
        """Test deepagent with filesystem tools using a fake LLM model.

        This test verifies that a deepagent can use the built-in filesystem
        tools (ls, read_file, etc.) with a fake LLM model.
        """
        # Create a fake model that uses filesystem tools
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ls",
                                "args": {"path": "."},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've listed the files in the current directory.",
                    ),
                ]
            )
        )

        # Create a deep agent with the fake model
        agent = create_deep_agent(model=model)

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="List files")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify ls tool was called
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

    def test_deep_agent_with_fake_llm_multiple_tool_calls(self) -> None:
        """Test deepagent with multiple tool calls using a fake LLM model.

        This test verifies that a deepagent can handle multiple sequential
        tool calls with a fake LLM model.
        """
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

        # Create a deep agent with the fake model and sample_tool
        agent = create_deep_agent(model=model, tools=[sample_tool])

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content="Use sample tool twice")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Verify multiple tool calls occurred
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) >= 2

        # Verify both inputs were used
        tool_contents = [msg.content for msg in tool_messages]
        assert any("first call" in content for content in tool_contents)
        assert any("second call" in content for content in tool_contents)

    def test_deep_agent_with_string_model_name(self) -> None:
        """Test that create_deep_agent handles string model names correctly.

        This test verifies that when a model name is passed as a string,
        it is properly initialized using init_chat_model instead of
        causing an AttributeError when accessing the profile attribute.
        """
        # Mock init_chat_model to return a fake model
        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="Response from string-initialized model.",
                    )
                ]
            )
        )

        with patch("deepagents.graph.init_chat_model", return_value=fake_model):
            # This should not raise AttributeError: 'str' object has no attribute 'profile'
            agent = create_deep_agent(model="claude-sonnet-4-5-20250929", tools=[sample_tool])

            # Verify agent was created successfully
            assert agent is not None

            # Invoke the agent to ensure it works
            result = agent.invoke({"messages": [HumanMessage(content="Test message")]})

            # Verify the agent executed correctly
            assert "messages" in result
            assert len(result["messages"]) > 0

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_truncate_lines(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test line count limiting in read_file tool with very long lines."""
        # Create a file with a very long line (18,000 chars) that will be split into continuation lines
        # With MAX_LINE_LENGTH=5000, this becomes line 2, 2.1, 2.2, 2.3 (4 output lines for 1 logical line)
        very_long_line = "x" * 18000  # 18,000 characters -> will split into 4 continuation lines (5k each)

        # Add some normal lines before and after
        lines = [
            "short line 0",
            very_long_line,  # This becomes lines 2, 2.1, 2.2, 2.3 (4 output lines)
            "short line 2",
            "short line 3",
            "short line 4",
        ]
        content = "\n".join(lines)

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/my_file"
        res = backend.write(file_path, content)
        if isinstance(backend, StateBackend):
            backend.runtime.state["files"].update(res.files_update)

        # Create a fake model that calls read_file with limit=3
        # This should return: line 1 (short line 0), line 2 (first chunk of very_long_line), line 2.1 (second chunk)
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": file_path, "limit": 3},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the file successfully.",
                    ),
                ]
            )
        )

        # Create agent with backend
        agent = create_deep_agent(model=model, backend=backend)

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content=f"Read {file_path}")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Get the tool message containing the file content
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        file_content = tool_messages[0].content

        # Should have the first short line
        assert "short line 0" in file_content

        # Should have the beginning of the very long line (line 2 with continuation)
        assert "xxx" in file_content  # The very long line should be present

        # Should NOT have the later short lines because the limit cuts off after 3 output lines
        # (line 1, line 2, line 2.1)
        assert "short line 2" not in file_content
        assert "short line 3" not in file_content
        assert "short line 4" not in file_content

        # Count actual lines in the output (excluding empty lines from formatting)
        output_lines = [line for line in file_content.split("\n") if line.strip()]
        # Should be at most 3 lines (the limit we specified)
        # This includes continuation lines as separate lines
        assert len(output_lines) <= 3

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_read_empty_file(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test reading an empty file through the agent."""
        # Create backend and write empty file
        backend = backend_factory(tmp_path)

        file_path = "/my_file"
        res = backend.write(file_path, "")
        if isinstance(backend, StateBackend):
            backend.runtime.state["files"].update(res.files_update)

        # Create a fake model that calls read_file
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": file_path},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the empty file.",
                    ),
                ]
            )
        )

        # Create agent with backend
        agent = create_deep_agent(model=model, backend=backend)

        # Invoke the agent
        result = agent.invoke({"messages": [HumanMessage(content=f"Read {file_path}")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Get the tool message containing the file content
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        file_content = tool_messages[0].content

        # Empty file should return empty or minimal content
        # (Backend might add warnings or format)
        assert isinstance(file_content, str)

    def test_deep_agent_with_system_message(self) -> None:
        """Test that create_deep_agent accepts a SystemMessage for system_prompt."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello! How can I help you today?"),
                ]
            )
        )
        capturing_middleware = SystemMessageCapturingMiddleware()
        system_msg = SystemMessage(
            content=[
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": "Always be polite."},
            ]
        )
        agent = create_deep_agent(model=model, system_prompt=system_msg, middleware=[capturing_middleware])
        assert_all_deepagent_qualities(agent)

        agent.invoke({"messages": [HumanMessage(content="Hello")]})

        content = str(capturing_middleware.captured_system_messages[0].content)
        assert "You are a helpful assistant." in content
        assert "Always be polite." in content
        assert "you have access to a number of standard tools" in content

    def test_deep_agent_with_system_message_string_content(self) -> None:
        """Test that create_deep_agent accepts a SystemMessage with string content."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(content="Hello! I'm your research assistant."),
                ]
            )
        )
        capturing_middleware = SystemMessageCapturingMiddleware()
        system_msg = SystemMessage(content="You are a helpful research assistant.")
        agent = create_deep_agent(model=model, system_prompt=system_msg, middleware=[capturing_middleware])
        assert_all_deepagent_qualities(agent)

        agent.invoke({"messages": [HumanMessage(content="Hello")]})

        content = str(capturing_middleware.captured_system_messages[0].content)
        assert "You are a helpful research assistant." in content
        assert "you have access to a number of standard tools" in content
