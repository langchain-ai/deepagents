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
from deepagents.backends.utils import TOOL_RESULT_TOKEN_LIMIT
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import NUM_CHARS_PER_TOKEN
from tests.utils import SampleMiddlewareWithTools, SampleMiddlewareWithToolsAndState, assert_all_deepagent_qualities


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
        assert "You are a Deep Agent" in content

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
        assert "You are a Deep Agent" in content

    def test_deep_agent_two_turns_no_initial_files(self) -> None:
        """Test deepagent with two conversation turns without specifying files on invoke.

        This test reproduces the edge case from issue #731 where the files state
        can become corrupted (turning into a list) during the second message in a
        conversation when files are not explicitly provided in the initial invoke.
        """
        # Create a model that handles both turns
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    # Turn 1: write a file
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/test.txt", "content": "Hello World"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've created the file.",
                    ),
                    # Turn 2: glob files
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "glob",
                                "args": {"pattern": "*.txt"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've listed the files.",
                    ),
                ]
            )
        )

        # Create agent - IMPORTANT: don't specify files in initial state
        agent = create_deep_agent(model=model)

        # First invoke - no files key in input
        result1 = agent.invoke({"messages": [HumanMessage(content="Create a test file")]})

        # Verify first turn succeeded
        assert "messages" in result1
        tool_messages = [msg for msg in result1["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        # Second invoke using same agent instance - this is where the bug might occur
        # Continue from previous state but don't pass files key
        result2 = agent.invoke(
            {
                "messages": result1["messages"] + [HumanMessage(content="List all text files")],
                # Explicitly not providing "files" key to test state initialization
            }
        )

        # Verify second turn succeeded without AttributeError
        assert "messages" in result2
        tool_messages2 = [msg for msg in result2["messages"] if msg.type == "tool"]
        assert len(tool_messages2) > 0

        # The glob tool should not crash with "AttributeError: 'list' object has no attribute 'items'"
        # Check that we got a valid response (not an error)
        glob_result = tool_messages2[-1].content
        assert isinstance(glob_result, str)
        # Should either find files or return "No files found", not crash
        assert "AttributeError" not in glob_result

    def test_deep_agent_two_turns_state_backend_edge_case(self) -> None:
        """Test StateBackend with two turns to reproduce potential state corruption.

        This test specifically targets the edge case where the files state might
        become corrupted into a list instead of a dict, causing AttributeError
        when tools try to access .items().
        """
        # Create a StateBackend with an explicitly initialized runtime
        runtime = make_runtime()

        # IMPORTANT: Initialize files as empty dict, not missing
        # This is key to potentially trigger the reducer issue
        runtime.state["files"] = {}

        backend = StateBackend(runtime)

        # Create a model that writes then globs
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    # Turn 1: write a file
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/test.txt", "content": "Test content"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="File created."),
                    # Turn 2: glob for files
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "glob",
                                "args": {"pattern": "*.txt"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Files listed."),
                ]
            )
        )

        # Create agent with StateBackend
        agent = create_deep_agent(model=model, backend=backend)

        # First turn
        result1 = agent.invoke({"messages": [HumanMessage(content="Create a file")]})
        assert "messages" in result1

        # Verify files state is still a dict after first turn
        if "files" in result1:
            assert isinstance(result1["files"], dict), f"files is {type(result1['files'])}, expected dict"

        # Second turn - this should trigger glob on the files state
        result2 = agent.invoke(
            {
                "messages": result1["messages"] + [HumanMessage(content="List files")],
            }
        )
        assert "messages" in result2

        # Verify glob succeeded without AttributeError
        tool_messages = [msg for msg in result2["messages"] if msg.type == "tool"]
        glob_messages = [msg for msg in tool_messages if "glob" in msg.name or "*.txt" in str(msg)]

        if glob_messages:
            # Check that glob result doesn't contain error
            glob_result = glob_messages[-1].content
            assert "AttributeError" not in glob_result
            assert "'list' object has no attribute 'items'" not in glob_result

    @pytest.mark.asyncio
    async def test_deep_agent_two_turns_no_initial_files_async(self) -> None:
        """Async version: Test deepagent with two conversation turns without specifying files.

        This async test reproduces the edge case from issue #731 where the files state
        can become corrupted during async execution.
        """
        # Create a model that handles both turns
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    # Turn 1: write a file
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/test.txt", "content": "Hello World"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've created the file.",
                    ),
                    # Turn 2: glob files
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "glob",
                                "args": {"pattern": "*.txt"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've listed the files.",
                    ),
                ]
            )
        )

        # Create agent - IMPORTANT: don't specify files in initial state
        agent = create_deep_agent(model=model)

        # First invoke - no files key in input
        result1 = await agent.ainvoke({"messages": [HumanMessage(content="Create a test file")]})

        # Verify first turn succeeded
        assert "messages" in result1
        tool_messages = [msg for msg in result1["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        # Second invoke using same agent instance - this is where the bug might occur
        # Continue from previous state but don't pass files key
        result2 = await agent.ainvoke(
            {
                "messages": result1["messages"] + [HumanMessage(content="List all text files")],
                # Explicitly not providing "files" key to test state initialization
            }
        )

        # Verify second turn succeeded without AttributeError
        assert "messages" in result2
        tool_messages2 = [msg for msg in result2["messages"] if msg.type == "tool"]
        assert len(tool_messages2) > 0

        # The glob tool should not crash with "AttributeError: 'list' object has no attribute 'items'"
        # Check that we got a valid response (not an error)
        glob_result = tool_messages2[-1].content
        assert isinstance(glob_result, str)
        # Should either find files or return "No files found", not crash
        assert "AttributeError" not in glob_result

    @pytest.mark.asyncio
    async def test_deep_agent_two_turns_state_backend_edge_case_async(self) -> None:
        """Async version: Test StateBackend with two turns to reproduce potential state corruption.

        This async test specifically targets the edge case where concurrent async operations
        might cause the files state to become corrupted into a list instead of a dict.
        """
        # Create a StateBackend with an explicitly initialized runtime
        runtime = make_runtime()

        # IMPORTANT: Initialize files as empty dict, not missing
        # This is key to potentially trigger the reducer issue
        runtime.state["files"] = {}

        backend = StateBackend(runtime)

        # Create a model that writes then globs
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    # Turn 1: write a file
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/test.txt", "content": "Test content"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="File created."),
                    # Turn 2: glob for files
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "glob",
                                "args": {"pattern": "*.txt"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Files listed."),
                ]
            )
        )

        # Create agent with StateBackend
        agent = create_deep_agent(model=model, backend=backend)

        # First turn (async)
        result1 = await agent.ainvoke({"messages": [HumanMessage(content="Create a file")]})
        assert "messages" in result1

        # Verify files state is still a dict after first turn
        if "files" in result1:
            assert isinstance(result1["files"], dict), f"files is {type(result1['files'])}, expected dict"

        # Second turn (async) - this should trigger glob on the files state
        result2 = await agent.ainvoke(
            {
                "messages": result1["messages"] + [HumanMessage(content="List files")],
            }
        )
        assert "messages" in result2

        # Verify glob succeeded without AttributeError
        tool_messages = [msg for msg in result2["messages"] if msg.type == "tool"]
        glob_messages = [msg for msg in tool_messages if "glob" in msg.name or "*.txt" in str(msg)]

        if glob_messages:
            # Check that glob result doesn't contain error
            glob_result = glob_messages[-1].content
            assert "AttributeError" not in glob_result
            assert "'list' object has no attribute 'items'" not in glob_result

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_read_file_truncation(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test that read_file truncates large files and provides pagination guidance."""
        # Create a file with content that exceeds the truncation threshold
        # Default token_limit_before_evict is 20000, so threshold is 4 * 20000 = 80000 chars
        large_content = "x" * 85000  # 85k chars exceeds the 80k threshold

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/large_file.txt"
        res = backend.write(file_path, large_content)
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
                                "args": {"file_path": file_path, "offset": 0, "limit": 100},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the file.",
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

        # Verify truncation occurred
        assert "Output was truncated due to size limits" in file_content
        assert "reformatting" in file_content.lower() or "reformat" in file_content.lower()

        # Verify the content stays under threshold (including truncation message)
        assert len(file_content) <= 80000

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_read_file_no_truncation_small_file(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test that read_file does NOT truncate small files."""
        # Create a small file that doesn't exceed the truncation threshold
        small_content = "Hello, world!\n" * 100  # Much smaller than 80k chars

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/small_file.txt"
        res = backend.write(file_path, small_content)
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
                        content="I've read the file.",
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

        # Verify NO truncation occurred
        assert "Output was truncated" not in file_content
        assert "Hello, world!" in file_content

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_read_file_truncation_with_offset(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test that read_file truncation message includes correct offset for pagination."""
        # Create a large file with many lines (each line is 500 chars + newline)
        # 500 lines total, we'll read lines 50-250 (200 lines)
        # 200 lines * ~510 chars (including formatting) = ~102,000 chars, exceeds 80k threshold
        large_content = "\n".join(["y" * 500 for _ in range(500)])

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/large_file_offset.txt"
        res = backend.write(file_path, large_content)
        if isinstance(backend, StateBackend):
            backend.runtime.state["files"].update(res.files_update)

        # Create a fake model that calls read_file with a non-zero offset
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": file_path, "offset": 50, "limit": 200},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the file.",
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

        # Verify truncation occurred
        assert "Output was truncated due to size limits" in file_content
        assert "reformatting" in file_content.lower() or "reformat" in file_content.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    async def test_deep_agent_read_file_truncation_async(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test that read_file truncates large files in async mode."""
        # Create a large file
        large_content = "z" * 85000

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/large_file_async.txt"
        res = await backend.awrite(file_path, large_content)
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
                                "args": {"file_path": file_path, "offset": 0, "limit": 100},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the file.",
                    ),
                ]
            )
        )

        # Create agent with backend
        agent = create_deep_agent(model=model, backend=backend)

        # Invoke the agent (async)
        result = await agent.ainvoke({"messages": [HumanMessage(content=f"Read {file_path}")]})

        # Verify the agent executed correctly
        assert "messages" in result

        # Get the tool message containing the file content
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        file_content = tool_messages[0].content

        # Verify truncation occurred
        assert "Output was truncated due to size limits" in file_content
        assert "reformatting" in file_content.lower() or "reformat" in file_content.lower()

        # Verify the content is actually truncated
        assert len(file_content) < 85000

    @pytest.mark.parametrize("backend_factory", BACKEND_FACTORIES)
    def test_deep_agent_read_file_single_long_line_behavior(self, tmp_path: Path, backend_factory: Callable[[Path], BackendProtocol]) -> None:
        """Test the behavior with a single very long line.

        When a file has a single very long line (e.g., 85,000 chars), it gets split
        into continuation markers (1, 1.1, 1.2, etc.) by format_content_with_line_numbers.

        The current behavior:
        - offset works on logical lines (before formatting)
        - limit applies to formatted output lines (after continuation markers)
        - This allows pagination through long lines by increasing limit
        - Limitation: cannot use offset to skip within a long line

        This test verifies:
        1. A single long line with limit=1 returns only the first chunk (respects limit on formatted lines)
        2. Size-based truncation applies if the formatted output exceeds threshold
        """
        # Create a file with a SINGLE very long line (no newlines)
        # This will be split into ~17 continuation chunks (85000 / 5000)
        single_long_line = "x" * 85000

        # Create backend and write file
        backend = backend_factory(tmp_path)

        file_path = "/single_long_line.txt"
        res = backend.write(file_path, single_long_line)
        if isinstance(backend, StateBackend):
            backend.runtime.state["files"].update(res.files_update)

        # Create a fake model that calls read_file with limit=1
        # This should return just 1 formatted line (the first chunk of the long line)
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": file_path, "offset": 0, "limit": 1},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="I've read the file.",
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

        # Verify behavior: with limit=1, we get only the first formatted line
        # (not all continuation markers)
        assert len(file_content) < 10000  # Only got first chunk (~5000 chars)
        assert len(file_content.splitlines()) == 1  # Only 1 formatted line
        assert "1.1" not in file_content  # No continuation markers (would need higher limit)

        # To get more of the line, the model would need to increase limit, not offset
        # E.g., read_file(offset=0, limit=20) would get first 20 formatted lines

    def test_read_large_single_line_file_returns_reasonable_size(self) -> None:
        """Test that read_file doesn't return excessive chars for a single-line file.

        When tool results are evicted via str(dict), they become a single line.
        read_file chunks this into 100 lines x 5000 chars = 500K chars - potential token overflow.
        This test verifies that the truncation logic prevents such overflow.
        """
        max_reasonable_chars = TOOL_RESULT_TOKEN_LIMIT * NUM_CHARS_PER_TOKEN  # 80,000 chars

        # str(dict) produces no newlines—exactly how evicted tool results are serialized
        large_dict = {"records": [{"id": i, "data": "x" * 100} for i in range(4000)]}
        large_content = str(large_dict)
        assert "\n" not in large_content

        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {
                                    "file_path": "/large_tool_results/evicted_data",
                                    "content": large_content,
                                },
                                "id": "call_write",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/large_tool_results/evicted_data"},
                                "id": "call_read",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(content="Done reading the file."),
                ]
            )
        )

        agent = create_deep_agent(model=fake_model)
        result = agent.invoke({"messages": [HumanMessage(content="Write and read a large file")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        read_file_response = tool_messages[-1]

        # Verify truncation occurred and result stays under threshold
        assert "Output was truncated due to size limits" in read_file_response.content, "Expected truncation message for large single-line file"
        assert len(read_file_response.content) <= max_reasonable_chars, (
            f"read_file returned {len(read_file_response.content):,} chars. "
            f"Expected <= {max_reasonable_chars:,} chars (TOOL_RESULT_TOKEN_LIMIT * 4). "
            f"A single-line file should not cause token overflow."
        )


class TestDeepAgentStructure:
    """Test basic deep agent structure without making network calls."""

    def test_base_deep_agent(self) -> None:
        """Verifies that a basic deep agent can be created with default settings."""
        agent = create_deep_agent()
        assert_all_deepagent_qualities(agent)

    def test_deep_agent_with_tool(self) -> None:
        """Verifies that a deep agent can be created with tools and the tools are properly bound."""
        agent = create_deep_agent(tools=[sample_tool])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name

    def test_deep_agent_with_middleware_with_tool(self) -> None:
        """Verifies that middleware can inject tools into a deep agent."""
        agent = create_deep_agent(middleware=[SampleMiddlewareWithTools()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name

    def test_deep_agent_with_middleware_with_tool_and_state(self) -> None:
        """Verifies that middleware can inject both tools and extended state channels."""
        agent = create_deep_agent(middleware=[SampleMiddlewareWithToolsAndState()])
        assert_all_deepagent_qualities(agent)
        assert "sample_tool" in agent.nodes["tools"].bound._tools_by_name
        assert "sample_input" in agent.stream_channels
