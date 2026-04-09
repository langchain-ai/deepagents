"""End-to-end unit tests for deepagents with fake LLM models."""

import base64
import json
import warnings
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import TOOL_RESULT_TOKEN_LIMIT, create_file_data
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import NUM_CHARS_PER_TOKEN
from deepagents.middleware.subagents import SubAgent  # noqa: TC001
from deepagents.middleware.summarization import create_summarization_tool_middleware
from deepagents.permissions import FilesystemPermission, ToolPermission
from tests.unit_tests.chat_model import GenericFakeChatModel as FakeChatModelWithHistory
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


@pytest.fixture(params=["filesystem_virtual", "state", "store"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> BackendProtocol:
    """Create a backend for parametrized tests."""
    if request.param == "filesystem_virtual":
        return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    if request.param == "state":
        return StateBackend()
    return StoreBackend(store=InMemoryStore())


def prepopulate_file(backend: BackendProtocol, file_path: str, content: str) -> dict[str, Any] | None:
    """Write a file to the backend, returning starter ``files`` for StateBackend.

    For external backends (filesystem, store) the write happens immediately.
    For StateBackend, the file data is returned as a dict that should be
    passed via ``agent.invoke({"files": ...})`` since StateBackend requires
    a LangGraph execution context.
    """
    if isinstance(backend, StateBackend):
        return {file_path: {**create_file_data(content)}}
    backend.write(file_path, content)
    return None


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

    def test_main_agent_streaming_metadata_includes_tags_and_config_metadata(self) -> None:
        """Test main-agent-only streaming metadata on `messages` mode.

        Verifies streamed model chunks from the main agent include:
        1. `ls_integration`
        2. Config `tags`
        3. Config `metadata` entries
        """
        agent = create_deep_agent(
            model=FixedGenericFakeChatModel(
                messages=iter([AIMessage(content="MAIN_AGENT_RESPONSE")]),
            ),
            name="supervisor",
        )

        first_metadata: dict | None = None

        for stream_mode, data in agent.stream(
            {"messages": [HumanMessage(content="Do something directly")]},
            stream_mode=["messages", "updates"],
            config={
                "configurable": {"thread_id": "test_main_stream"},
                "tags": ["main-tag", "session-456"],
                "metadata": {"request_id": "req-main-123", "tenant": "acme-main"},
            },
        ):
            if stream_mode != "messages":
                continue
            _message_chunk, first_metadata = data
            break

        assert first_metadata is not None
        assert first_metadata.get("ls_integration") == "langchain_chat_model"
        assert first_metadata.get("tags") == ["main-tag", "session-456"]
        assert first_metadata.get("request_id") == "req-main-123"
        assert first_metadata.get("tenant") == "acme-main"

    def test_tool_runtime_config_includes_default_graph_metadata(self) -> None:
        """Test tool runtime config includes the defaults from `create_deep_agent`."""
        captured_config: dict[str, Any] | None = None

        @tool
        def foo(runtime: ToolRuntime) -> str:
            """Capture runtime config."""
            nonlocal captured_config
            captured_config = runtime.config
            return "foo-result"

        agent = create_deep_agent(
            model=FixedGenericFakeChatModel(
                messages=iter(
                    [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "foo",
                                    "args": {},
                                    "id": "call_foo",
                                    "type": "tool_call",
                                }
                            ],
                        ),
                        AIMessage(content="Done."),
                    ]
                )
            ),
            tools=[foo],
            name="supervisor",
        )

        agent.invoke(
            {"messages": [HumanMessage(content="Call foo")]},
            config={
                "configurable": {"thread_id": "test_tool_runtime_metadata"},
                "tags": ["tool-tag", "tool-session-456"],
            },
        )

        assert captured_config is not None
        assert captured_config["recursion_limit"] == agent.config["recursion_limit"]
        assert captured_config["tags"] == ["tool-tag", "tool-session-456"]
        assert captured_config["metadata"]["ls_integration"] == "deepagents"
        assert captured_config["metadata"]["lc_agent_name"] == "supervisor"
        assert "deepagents" in captured_config["metadata"]["versions"]

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
        """Test that create_deep_agent resolves string model names correctly.

        This test verifies that when a model name is passed as a string,
        it is properly resolved to a chat model instead of
        causing an AttributeError when accessing the profile attribute.
        """
        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="Response from string-initialized model.",
                    )
                ]
            )
        )

        with patch("deepagents.graph.resolve_model", return_value=fake_model):
            agent = create_deep_agent(model="claude-sonnet-4-6", tools=[sample_tool])
            assert agent is not None

            result = agent.invoke({"messages": [HumanMessage(content="Test message")]})
            assert "messages" in result
            assert len(result["messages"]) > 0

    def test_deep_agent_truncate_lines(self, tmp_path: Path, backend: BackendProtocol) -> None:
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

        file_path = "/my_file"
        starter_files = prepopulate_file(backend, file_path, content)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

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

    def test_deep_agent_read_empty_file(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test reading an empty file through the agent."""
        # Create backend and write empty file

        file_path = "/my_file"
        starter_files = prepopulate_file(backend, file_path, "")

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

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
        backend = StateBackend()

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

    async def test_deep_agent_two_turns_state_backend_edge_case_async(self) -> None:
        """Async version: Test StateBackend with two turns to reproduce potential state corruption.

        This async test specifically targets the edge case where concurrent async operations
        might cause the files state to become corrupted into a list instead of a dict.
        """
        backend = StateBackend()

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

    def test_deep_agent_read_file_truncation(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test that read_file truncates large files and provides pagination guidance."""
        # Create a file with content that exceeds the truncation threshold
        # Default token_limit_before_evict is 20000, so threshold is 4 * 20000 = 80000 chars
        large_content = "x" * 85000  # 85k chars exceeds the 80k threshold

        # Create backend and write file

        file_path = "/large_file.txt"
        starter_files = prepopulate_file(backend, file_path, large_content)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

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

    def test_deep_agent_read_file_no_truncation_small_file(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test that read_file does NOT truncate small files."""
        # Create a small file that doesn't exceed the truncation threshold
        small_content = "Hello, world!\n" * 100  # Much smaller than 80k chars

        # Create backend and write file

        file_path = "/small_file.txt"
        starter_files = prepopulate_file(backend, file_path, small_content)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

        # Verify the agent executed correctly
        assert "messages" in result

        # Get the tool message containing the file content
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        file_content = tool_messages[0].content

        # Verify NO truncation occurred
        assert "Output was truncated" not in file_content
        assert "Hello, world!" in file_content

    def test_deep_agent_read_file_truncation_with_offset(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test that read_file truncation message includes correct offset for pagination."""
        # Create a large file with many lines (each line is 500 chars + newline)
        # 500 lines total, we'll read lines 50-250 (200 lines)
        # 200 lines * ~510 chars (including formatting) = ~102,000 chars, exceeds 80k threshold
        large_content = "\n".join(["y" * 500 for _ in range(500)])

        # Create backend and write file

        file_path = "/large_file_offset.txt"
        starter_files = prepopulate_file(backend, file_path, large_content)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

        # Verify the agent executed correctly
        assert "messages" in result

        # Get the tool message containing the file content
        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) > 0

        file_content = tool_messages[0].content

        # Verify truncation occurred
        assert "Output was truncated due to size limits" in file_content
        assert "reformatting" in file_content.lower() or "reformat" in file_content.lower()

    async def test_deep_agent_read_file_truncation_async(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test that read_file truncates large files in async mode."""
        # Create a large file
        large_content = "z" * 85000

        # Create backend and write file

        file_path = "/large_file_async.txt"
        starter_files = prepopulate_file(backend, file_path, large_content)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = await agent.ainvoke(invoke_input)

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

    def test_deep_agent_read_file_single_long_line_behavior(self, tmp_path: Path, backend: BackendProtocol) -> None:
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

        file_path = "/single_long_line.txt"
        starter_files = prepopulate_file(backend, file_path, single_long_line)

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
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content=f"Read {file_path}")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

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

    def test_deep_agent_read_file_invalid_args_returns_tool_message(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test invalid read_file arguments still produce a ToolMessage."""
        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"foo": "/missing.txt", "does_not_exist": True},
                                "id": "call_invalid_read",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Handled the invalid read_file call."),
                ]
            )
        )

        agent = create_deep_agent(model=fake_model, backend=backend)
        result = agent.invoke({"messages": [HumanMessage(content="Try reading a file with invalid args")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        tool_message = tool_messages[0]
        assert tool_message.tool_call_id == "call_invalid_read"
        assert tool_message.status == "error"
        assert "Error invoking tool 'read_file' with kwargs " in tool_message.content

    def test_deep_agent_read_image_file(self, tmp_path: Path, backend: BackendProtocol) -> None:
        """Test that reading an image returns a ToolMessage with content blocks."""
        img_bytes = b"\x89PNG\r\n\x1a\n fake image data"

        starter_files = None
        if isinstance(backend, FilesystemBackend):
            (tmp_path / "photo.png").write_bytes(img_bytes)
        else:
            encoded = base64.b64encode(img_bytes).decode("ascii")
            starter_files = prepopulate_file(backend, "/photo.png", encoded)

        fake_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/photo.png"},
                                "id": "call_img",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Here is the image."),
                ]
            )
        )

        agent = create_deep_agent(model=fake_model, backend=backend)
        invoke_input: dict[str, Any] = {"messages": [HumanMessage(content="Read the image")]}
        if starter_files:
            invoke_input["files"] = starter_files
        result = agent.invoke(invoke_input)

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        tm = tool_messages[0]
        assert isinstance(tm.content, list)
        assert len(tm.content) == 1
        assert tm.content[0]["type"] == "image"
        assert tm.content[0]["mime_type"] == "image/png"
        assert "base64" in tm.content[0]


class TestDeepAgentPermissionsEndToEnd:
    """End-to-end tests for create_deep_agent with FilesystemPermission and ToolPermission."""

    def test_filesystem_permission_deny_write_blocks_write_file(self) -> None:
        """FilesystemPermission deny write blocks write_file and returns error message."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/secrets/key.txt", "content": "data"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Write to secrets")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "permission denied" in tool_messages[0].content
        assert "write" in tool_messages[0].content

    def test_filesystem_permission_deny_read_blocks_read_file(self) -> None:
        """FilesystemPermission deny read blocks read_file and returns error message."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/secrets/key.txt"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Read secrets")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "permission denied" in tool_messages[0].content
        assert "read" in tool_messages[0].content

    def test_filesystem_permission_allow_unrelated_path(self) -> None:
        """FilesystemPermission deny on one path does not block unrelated paths."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/workspace/file.txt", "content": "hello"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Write to workspace")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "permission denied" not in tool_messages[0].content

    def test_tool_permission_deny_blocks_tool(self) -> None:
        """ToolPermission deny blocks the matching tool and returns an error ToolMessage."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "hello"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run sample_tool")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content

    def test_tool_permission_allow_with_arg_pattern(self) -> None:
        """ToolPermission allow with argument pattern permits matching calls."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "allowed value"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", args={"sample_input": "allowed *"}),
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run sample_tool")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" not in tool_messages[0].content
        assert "allowed value" in tool_messages[0].content

    def test_tool_permission_deny_non_matching_arg(self) -> None:
        """ToolPermission deny falls through when argument pattern doesn't match, blocking the call."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "forbidden value"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", args={"sample_input": "allowed *"}),
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run sample_tool")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content

    def test_combined_filesystem_and_tool_permissions(self) -> None:
        """FilesystemPermission and ToolPermission can coexist; both are enforced."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/restricted/data.txt", "content": "data"},
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
                                "args": {"sample_input": "test"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/restricted/**"], mode="deny"),
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run both tools")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 2
        assert "permission denied" in tool_messages[0].content
        assert "denied" in tool_messages[1].content

    def test_no_permissions_allows_all(self) -> None:
        """With no permissions specified, all tool calls proceed normally."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "unrestricted"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(model=model, tools=[sample_tool])
        result = agent.invoke({"messages": [HumanMessage(content="Run sample_tool")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" not in tool_messages[0].content
        assert "unrestricted" in tool_messages[0].content

    def test_glob_post_filters_denied_paths(self) -> None:
        """FilesystemPermission post-filters glob results, removing denied paths."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "glob",
                                "args": {"pattern": "**/*.txt", "path": "/"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Glob all txt files")],
                "files": {
                    "/public/a.txt": {**create_file_data("pub")},
                    "/secrets/b.txt": {**create_file_data("priv")},
                },
            }
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "/secrets/b.txt" not in tool_messages[0].content
        assert "/public/a.txt" in tool_messages[0].content

    def test_grep_post_filters_denied_paths(self) -> None:
        """FilesystemPermission post-filters grep results, removing denied paths."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "grep",
                                "args": {"pattern": "keyword"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Search for keyword")],
                "files": {
                    "/public/a.txt": {**create_file_data("keyword here")},
                    "/secrets/b.txt": {**create_file_data("keyword there")},
                },
            }
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "/secrets/b.txt" not in tool_messages[0].content
        assert "/public/a.txt" in tool_messages[0].content

    def test_tool_permission_denies_filesystem_tool_by_name(self) -> None:
        """ToolPermission can deny a filesystem tool (read_file) by name via PermissionMiddleware."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/workspace/file.txt"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                ToolPermission(name="read_file", mode="deny"),
            ],
        )
        result = agent.invoke(
            {
                "messages": [HumanMessage(content="Read a file")],
                "files": {
                    "/workspace/file.txt": {**create_file_data("hello")},
                },
            }
        )

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content

    def test_tool_permission_arg_glob_on_execute_pattern(self) -> None:
        """ToolPermission argument glob matching works end-to-end with allowlist + catch-all deny."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "pytest tests/unit"},
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
                                "args": {"sample_input": "make build"},
                                "id": "call_2",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "rm -rf /"},
                                "id": "call_3",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", args={"sample_input": "pytest *"}),
                ToolPermission(name="sample_tool", args={"sample_input": "make *"}),
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Run commands")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 3
        assert "denied" not in tool_messages[0].content
        assert "pytest tests/unit" in tool_messages[0].content
        assert "denied" not in tool_messages[1].content
        assert "make build" in tool_messages[1].content
        assert "denied" in tool_messages[2].content

    async def test_filesystem_permission_deny_write_async(self) -> None:
        """(async) FilesystemPermission deny write blocks write_file."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/secrets/key.txt", "content": "data"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="Write to secrets")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "permission denied" in tool_messages[0].content

    async def test_tool_permission_deny_async(self) -> None:
        """(async) ToolPermission deny blocks the matching tool."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "hello"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="Run sample_tool")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        assert len(tool_messages) == 1
        assert "denied" in tool_messages[0].content


class TestSubAgentPermissionsEndToEnd:
    """End-to-end tests for subagent permission inheritance and override.

    The `task` ToolMessage seen by the parent contains the subagent's *final AIMessage*
    content. To verify a permission blocked a tool inside the subagent, the subagent
    model's final response must reflect the denial (as a real model would), which is then
    what the parent receives. For override/success cases we additionally verify the file
    state to confirm the tool actually executed.
    """

    def _parent_model_calling_subagent(self, subagent_type: str = "worker") -> FixedGenericFakeChatModel:
        return FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Do the work",
                                    "subagent_type": subagent_type,
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

    def test_subagent_inherits_parent_filesystem_permission(self) -> None:
        """Subagent without its own permissions inherits the parent's FilesystemPermission."""
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/secrets/key.txt", "content": "data"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Subagent's final response echoes what the tool returned (the denial)
                    AIMessage(content="Error: permission denied for write on /secrets/key.txt"),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
        }

        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        # The parent sees the subagent's last AIMessage which reflects the denial
        assert "permission denied" in task_result.content
        # And the file must not have been written
        assert "/secrets/key.txt" not in result.get("files", {})

    def test_subagent_inherits_parent_tool_permission(self) -> None:
        """Subagent without its own permissions inherits the parent's ToolPermission."""
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "hello"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Error: tool use denied by permission"),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
            "tools": [sample_tool],
        }

        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "denied" in task_result.content

    def test_subagent_own_permissions_override_parent(self) -> None:
        """Subagent with its own permissions replaces parent rules entirely.

        Parent denies `sample_tool`. The subagent specifies empty permissions
        (no rules → permissive default), so the tool is allowed inside the subagent.
        """
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "hello"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Tool returned: hello"),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
            "tools": [sample_tool],
            "permissions": [],  # empty → no rules → permissive, overrides parent deny
        }

        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "denied" not in task_result.content
        assert "hello" in task_result.content

    def test_subagent_own_permissions_stricter_than_parent(self) -> None:
        """Subagent can have its own, more restrictive permissions than the parent."""
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/workspace/out.txt", "content": "data"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Error: permission denied for write on /workspace/out.txt"),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A read-only worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
            "permissions": [
                FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
            ],
        }

        # Parent has no write restrictions — the subagent's own rules are more restrictive
        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "permission denied" in task_result.content
        assert "/workspace/out.txt" not in result.get("files", {})

    def test_general_purpose_subagent_inherits_parent_permissions(self) -> None:
        """The auto-added general-purpose subagent inherits the parent's permissions."""
        parent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {
                                    "description": "Write a file",
                                    "subagent_type": "general-purpose",
                                },
                                "id": "call_task",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )
        gp_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/secrets/key.txt", "content": "data"},
                                "id": "gp_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Error: permission denied for write on /secrets/key.txt"),
                ]
            )
        )

        gp_subagent: SubAgent = {
            "name": "general-purpose",
            "description": "General purpose subagent.",
            "system_prompt": "Help with tasks.",
            "model": gp_model,
            # No permissions key → inherits from parent
        }

        agent = create_deep_agent(
            model=parent_model,
            subagents=[gp_subagent],
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Ask GP to write")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "permission denied" in task_result.content
        assert "/secrets/key.txt" not in result.get("files", {})

    def test_subagent_own_permissions_allow_what_parent_restricts(self) -> None:
        """Subagent with its own (permissive) rules can write where parent restricts.

        The parent denies writes to /secrets. The subagent has its own empty
        permissions so the deny doesn't apply to it, and the write succeeds.
        """
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/workspace/out.txt", "content": "result"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="File written successfully."),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
            "permissions": [],  # no rules → all allowed, overrides parent
        }

        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            ],
        )
        result = agent.invoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "permission denied" not in task_result.content
        assert "written successfully" in task_result.content
        assert "/workspace/out.txt" in result.get("files", {})

    async def test_subagent_inherits_parent_permissions_async(self) -> None:
        """(async) Subagent without its own permissions inherits the parent's ToolPermission."""
        subagent_model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "sample_tool",
                                "args": {"sample_input": "hello"},
                                "id": "sub_call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Error: tool use denied by permission"),
                ]
            )
        )

        worker: SubAgent = {
            "name": "worker",
            "description": "A worker subagent.",
            "system_prompt": "Do work.",
            "model": subagent_model,
            "tools": [sample_tool],
        }

        agent = create_deep_agent(
            model=self._parent_model_calling_subagent(),
            subagents=[worker],
            tools=[sample_tool],
            permissions=[
                ToolPermission(name="sample_tool", mode="deny"),
            ],
        )
        result = await agent.ainvoke({"messages": [HumanMessage(content="Delegate the work")]})

        tool_messages = [msg for msg in result["messages"] if msg.type == "tool"]
        task_result = next(m for m in tool_messages if m.name == "task")
        assert "denied" in task_result.content


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


class TestLargeHumanMessageEviction:
    """Test that oversized HumanMessages are evicted to the filesystem."""

    def test_large_human_message_evicted_before_model_call(self) -> None:
        """An oversized HumanMessage is evicted and tagged with ``lc_evicted_to``.

        The agent receives a HumanMessage (no id) whose text content exceeds
        the eviction threshold. The filesystem middleware's ``wrap_model_call``
        should write the full content to the backend and tag the message in
        state via ``lc_evicted_to``, while preserving the original content.
        The model should see a truncated preview, not the full content.
        """
        threshold = 50_000
        large_content = "x" * (NUM_CHARS_PER_TOKEN * threshold + 1)

        fake_model = FakeChatModelWithHistory(messages=iter([AIMessage(content="Got it.")]))

        agent = create_deep_agent(model=fake_model)
        result = agent.invoke({"messages": [HumanMessage(content=large_content)]})

        human_messages = [msg for msg in result["messages"] if isinstance(msg, HumanMessage)]
        assert len(human_messages) == 1
        msg = human_messages[0]

        assert msg.content == large_content
        evicted_to = msg.additional_kwargs.get("lc_evicted_to")
        assert evicted_to is not None
        assert evicted_to.startswith("/conversation_history/")

        files = result.get("files", {})
        assert evicted_to in files, f"Evicted file {evicted_to} not found in state"
        assert files[evicted_to]["content"] == large_content

        assert len(fake_model.call_history) == 1
        model_messages = fake_model.call_history[0]["messages"]
        model_human = [m for m in model_messages if isinstance(m, HumanMessage)]
        assert len(model_human) == 1
        assert len(model_human[0].content) < len(large_content)
        assert "/conversation_history/" in model_human[0].content

    def test_multi_turn_eviction(self) -> None:
        """Tagged messages are truncated on subsequent turns.

        Simulates a multi-turn conversation with two oversized HumanMessages
        separated by a normal-sized message. On each model call, all
        previously-tagged messages should be truncated in the model request.
        """
        threshold = 50_000
        large_content_1 = "a" * (NUM_CHARS_PER_TOKEN * threshold + 1)
        large_content_2 = "b" * (NUM_CHARS_PER_TOKEN * threshold + 1)
        short_content = "short message"

        fake_model = FakeChatModelWithHistory(
            messages=iter(
                [
                    AIMessage(content="Response 1"),
                    AIMessage(content="Response 2"),
                    AIMessage(content="Response 3"),
                ]
            )
        )

        agent = create_deep_agent(model=fake_model, checkpointer=InMemorySaver())

        config = {"configurable": {"thread_id": "test-eviction"}}
        _ = agent.invoke({"messages": [HumanMessage(content=large_content_1)]}, config)
        _ = agent.invoke({"messages": [HumanMessage(content=short_content)]}, config)
        result_3 = agent.invoke({"messages": [HumanMessage(content=large_content_2)]}, config)

        assert len(fake_model.call_history) == 3

        call_1_messages = fake_model.call_history[0]["messages"]
        call_1_human = [m for m in call_1_messages if isinstance(m, HumanMessage)]
        assert len(call_1_human) == 1
        assert len(call_1_human[0].content) < len(large_content_1)

        call_2_messages = fake_model.call_history[1]["messages"]
        call_2_human = [m for m in call_2_messages if isinstance(m, HumanMessage)]
        assert len(call_2_human) == 2
        assert len(call_2_human[0].content) < len(large_content_1)
        assert call_2_human[1].content == short_content

        call_3_messages = fake_model.call_history[2]["messages"]
        call_3_human = [m for m in call_3_messages if isinstance(m, HumanMessage)]
        assert len(call_3_human) == 3
        assert len(call_3_human[0].content) < len(large_content_1)
        assert call_3_human[1].content == short_content
        assert len(call_3_human[2].content) < len(large_content_2)

        final_human = [m for m in result_3["messages"] if isinstance(m, HumanMessage)]
        tagged = [m for m in final_human if m.additional_kwargs.get("lc_evicted_to")]
        assert len(tagged) == 2
        for m in tagged:
            assert m.content in (large_content_1, large_content_2)

        files = result_3.get("files", {})
        for m in tagged:
            evicted_to = m.additional_kwargs["lc_evicted_to"]
            assert evicted_to in files, f"Evicted file {evicted_to} not found in state"
            assert files[evicted_to]["content"] == m.content


class TestSummarizationOffloadToState:
    """Test that SummarizationMiddleware offloads conversation history to StateBackend."""

    def test_offloaded_file_persisted_in_state(self) -> None:
        """Summarization should write the offloaded history to state via files_update.

        Uses ``create_deep_agent`` with default ``StateBackend`` so that
        ``backend.write`` returns a ``files_update`` dict. The ``Command``
        produced by ``wrap_model_call`` must propagate that dict so the file
        is persisted in graph state under the ``files`` channel.
        """
        fake_model = FakeChatModelWithHistory(
            messages=iter(
                [
                    AIMessage(content="summary goes here"),
                    AIMessage(content="response"),
                ]
            )
        )
        fake_model.profile = {"max_input_tokens": 200_000}

        agent = create_deep_agent(
            model=fake_model,
            checkpointer=InMemorySaver(),
        )

        text_10_000_tokens = "x" * 10_000 * NUM_CHARS_PER_TOKEN
        text_50_000_tokens = "x" * 50_000 * NUM_CHARS_PER_TOKEN
        input_messages = [
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),  # 60,000 tokens
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),  # 120,000 tokens
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),  # 180,000 tokens (summarizes)
            HumanMessage(content="query"),
        ]

        config = {"configurable": {"thread_id": "summarization-state-test"}}
        result = agent.invoke({"messages": input_messages}, config)

        assert len(result["messages"]) == 8  # 7 inputs + response
        assert result["messages"][-1].content == "response"

        # two calls: one to summarize, one for response
        assert len(fake_model.call_history) == 2

        # summarization call
        summarization_messages = fake_model.call_history[0]["messages"]
        assert any("Messages to summarize:" in m.content for m in summarization_messages if hasattr(m, "content"))

        # model call on reduced context
        summarized_messages = fake_model.call_history[1]["messages"]
        assert len(summarized_messages) < len(input_messages)
        summary_message = next(m for m in summarized_messages if isinstance(m, HumanMessage))
        assert "summary goes here" in summary_message.content

        # Verify conversation history was offloaded to state
        state = agent.get_state(config)
        files = state.values.get("files", {})
        conversation_history_files = {k: v for k, v in files.items() if k.startswith("/conversation_history/")}
        assert conversation_history_files, "Offloaded conversation history file not found in state"


class TestCompactConversationTool:
    """Test that the compact_conversation tool triggers summarization."""

    def test_compact_conversation_tool_invocation(self) -> None:
        """Agent invokes compact_conversation and conversation is compacted.

        Uses ``create_summarization_tool_middleware`` with a fake model that
        has a profile so fraction-based defaults are used. Input messages
        carry ``usage_metadata`` and ``response_metadata`` so the eligibility
        gate passes. The summarization model (same fake instance) returns a
        summary, and the agent model emits the tool call then a final response.
        """
        summary_model = FakeChatModelWithHistory(messages=iter([AIMessage(content="summary of earlier conversation")]))
        summary_model.profile = {"max_input_tokens": 200_000}

        provider = summary_model._get_ls_params()["ls_provider"]

        agent_model = FakeChatModelWithHistory(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "compact_conversation",
                                "args": {},
                                "id": "call_compact",
                                "type": "tool_call",
                            }
                        ],
                        usage_metadata={"input_tokens": 150_000, "output_tokens": 100, "total_tokens": 150_100},
                        response_metadata={"model_provider": provider},
                    ),
                    AIMessage(content="Done, conversation compacted."),
                ]
            )
        )

        agent = create_deep_agent(
            model=agent_model,
            middleware=[
                create_summarization_tool_middleware(summary_model, StateBackend),
            ],
            checkpointer=InMemorySaver(),
        )

        text_10k = "x" * 10_000 * NUM_CHARS_PER_TOKEN
        text_50k = "x" * 50_000 * NUM_CHARS_PER_TOKEN

        input_messages: list = [
            HumanMessage(content=text_10k),
            AIMessage(
                content=text_50k,
                usage_metadata={"input_tokens": 60_000, "output_tokens": 30_000, "total_tokens": 90_000},
                response_metadata={"model_provider": provider},
            ),
            HumanMessage(content=text_10k),
            AIMessage(
                content=text_50k,
                usage_metadata={"input_tokens": 120_000, "output_tokens": 30_000, "total_tokens": 150_000},
                response_metadata={"model_provider": provider},
            ),
            HumanMessage(content="please compact"),
        ]

        config = {"configurable": {"thread_id": "compact-tool-test"}}
        result = agent.invoke({"messages": input_messages}, config)

        assert result["messages"][-1].content == "Done, conversation compacted."

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        compact_msgs = [m for m in tool_messages if m.tool_call_id == "call_compact"]
        assert len(compact_msgs) == 1
        assert "compacted" in compact_msgs[0].content.lower() or "summarized" in compact_msgs[0].content.lower()
        assert "conversation that has been summarized" in agent_model.call_history[1]["messages"][1].content


class TestStateBackendConfigKeys:
    """Test that StateBackend reads/writes via CONFIG_KEY_READ/CONFIG_KEY_SEND."""

    def test_write_then_read(self) -> None:
        """Write a file via tool call, then read it back in the same turn."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/hello.txt", "content": "hello world"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/hello.txt"},
                                "id": "call_r",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(model=model)
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        # write should succeed
        assert any("Updated file" in m.content for m in tool_msgs)
        # read should return the content
        assert any("hello world" in m.content for m in tool_msgs)

    def test_write_then_edit_then_read(self) -> None:
        """Write, edit, then read a file across tool calls."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/doc.txt", "content": "foo bar baz"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "edit_file",
                                "args": {
                                    "file_path": "/doc.txt",
                                    "old_string": "bar",
                                    "new_string": "qux",
                                },
                                "id": "call_e",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/doc.txt"},
                                "id": "call_r",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(model=model)
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        # The read should reflect the edit
        read_msg = next(m for m in tool_msgs if m.tool_call_id == "call_r")
        assert "foo qux baz" in read_msg.content

    def test_write_then_ls(self) -> None:
        """Write a file, then ls to verify it appears."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/data/notes.md", "content": "# Notes"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "ls",
                                "args": {"path": "/data/"},
                                "id": "call_ls",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(model=model)
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        ls_msg = next(m for m in tool_msgs if m.tool_call_id == "call_ls")
        assert "/data/notes.md" in ls_msg.content

    def test_backward_compat_lambda_factory(self) -> None:
        """The old ``lambda rt: StateBackend(rt)`` factory pattern still works."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/compat.txt", "content": "works"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/compat.txt"},
                                "id": "call_r",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = create_deep_agent(
                model=model,
                backend=StateBackend,
            )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        assert any("works" in m.content for m in tool_msgs)

    def test_state_backend_runtime_deprecation(self) -> None:
        """Passing runtime to StateBackend emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StateBackend("ignored_runtime_value")

        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "runtime" in str(deprecations[0].message).lower()

    def test_store_backend_runtime_deprecation(self) -> None:
        """Passing runtime to StoreBackend emits a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            StoreBackend("ignored_runtime_value")

        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "runtime" in str(deprecations[0].message).lower()

    def test_store_backend_explicit_store_works(self) -> None:
        """StoreBackend(store=my_store) works without a graph context."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/note.txt", "content": "stored"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/note.txt"},
                                "id": "call_r",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        mem_store = InMemoryStore()
        agent = create_deep_agent(
            model=model,
            backend=StoreBackend(store=mem_store),
            store=mem_store,
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        assert any("stored" in m.content for m in tool_msgs)

    def test_store_backend_get_store_fallback(self) -> None:
        """StoreBackend() without explicit store works inside a graph via get_store()."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write_file",
                                "args": {"file_path": "/auto.txt", "content": "via get_store"},
                                "id": "call_w",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/auto.txt"},
                                "id": "call_r",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            backend=StoreBackend(),  # No explicit store
            store=InMemoryStore(),  # Passed to graph — get_store() picks it up
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]})

        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        assert any("via get_store" in m.content for m in tool_msgs)

    def test_backend_factory_deprecation(self) -> None:
        """Passing a callable factory as backend emits a DeprecationWarning."""
        model = FixedGenericFakeChatModel(messages=iter([AIMessage(content="hi")]))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = create_deep_agent(
                model=model,
                backend=StateBackend,
            )
            agent.invoke({"messages": [HumanMessage(content="go")]})

        dep_msgs = [x for x in w if issubclass(x.category, DeprecationWarning)]
        factory_warnings = [x for x in dep_msgs if "callable" in str(x.message).lower() or "factory" in str(x.message).lower()]
        assert len(factory_warnings) >= 1


class TestArtifactsRoot:
    """Test that artifacts_root on CompositeBackend parameterizes internal paths."""

    def test_deep_agent_artifacts_root_system_prompt_and_eviction(self) -> None:
        """Custom artifacts_root flows through to system prompt and eviction paths."""

        @tool(description="Returns a very large string")
        def big_tool() -> str:
            """Return a large string to trigger eviction."""
            return "x" * 500_000

        store_backend = StoreBackend(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )

        capturing_middleware = SystemMessageCapturingMiddleware()

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "big_tool",
                                "args": {},
                                "id": "call_big",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[big_tool],
            backend=backend,
            middleware=[capturing_middleware],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Call the big tool")]})

        # Verify system prompt references the custom artifacts_root
        system_content = str(capturing_middleware.captured_system_messages[0].content)
        assert "/workspace/large_tool_results/" in system_content
        assert "/large_tool_results/<tool_call_id>" not in system_content or "/workspace/large_tool_results/<tool_call_id>" in system_content

        # Verify the evicted tool result was written under the custom prefix
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        evicted_msg = next(m for m in tool_messages if m.tool_call_id == "call_big")
        assert "/workspace/large_tool_results/" in evicted_msg.content
        [resp] = backend.download_files(["/workspace/large_tool_results/call_big"])
        assert resp.error is None
        assert resp.content is not None
        assert b"x" * 100 in resp.content

    def test_deep_agent_artifacts_root_eviction_then_read(self) -> None:
        """Evicted tool result under custom artifacts_root can be read back."""
        large_content = "z" * 500_000

        @tool(description="Returns a very large string")
        def big_tool() -> str:
            """Return a large string to trigger eviction."""
            return large_content

        store_backend = StoreBackend(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "big_tool",
                                "args": {},
                                "id": "call_big",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_file",
                                "args": {"file_path": "/workspace/large_tool_results/call_big"},
                                "id": "call_read",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            tools=[big_tool],
            backend=backend,
        )

        result = agent.invoke({"messages": [HumanMessage(content="Call the big tool then read it")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        read_msg = next(m for m in tool_messages if m.tool_call_id == "call_read")
        assert "z" * 100 in read_msg.content

    def test_deep_agent_artifacts_root_conversation_history_offload(self) -> None:
        """Summarization offloads conversation history under the custom artifacts_root."""
        store_backend = StoreBackend(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )

        fake_model = FakeChatModelWithHistory(
            messages=iter(
                [
                    AIMessage(content="summary goes here"),
                    AIMessage(content="response"),
                ]
            )
        )
        fake_model.profile = {"max_input_tokens": 200_000}

        agent = create_deep_agent(
            model=fake_model,
            backend=backend,
            checkpointer=InMemorySaver(),
        )

        text_10_000_tokens = "x" * 10_000 * NUM_CHARS_PER_TOKEN
        text_50_000_tokens = "x" * 50_000 * NUM_CHARS_PER_TOKEN
        input_messages = [
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),
            HumanMessage(content=text_10_000_tokens),
            AIMessage(content=text_50_000_tokens),
            HumanMessage(content="query"),
        ]

        config = {"configurable": {"thread_id": "artifacts-root-summarization-test"}}
        result = agent.invoke({"messages": input_messages}, config)

        assert result["messages"][-1].content == "response"

        # Verify conversation history was offloaded under /workspace/conversation_history/
        ls_result = backend.ls("/workspace/conversation_history/")
        assert ls_result.entries, "Expected conversation history offloaded under /workspace/conversation_history/"

        # Verify nothing was written to the default /conversation_history/ path
        default_ls = backend.ls("/conversation_history/")
        assert not default_ls.entries, "No files should be written to /conversation_history/ when artifacts_root is set"

    def test_create_deep_agent_no_composite_backend(self) -> None:
        """create_deep_agent with a non-composite backend defaults artifacts_root to '/'."""
        backend = StateBackend()
        capturing_middleware = SystemMessageCapturingMiddleware()
        agent = create_deep_agent(
            model=FakeChatModelWithHistory(messages=iter([AIMessage(content="done")])),
            backend=backend,
            middleware=[capturing_middleware],
        )
        agent.invoke({"messages": [HumanMessage(content="Hi")]})
        system_content = str(capturing_middleware.captured_system_messages[0].content)
        assert "/large_tool_results/" in system_content

    def test_create_deep_agent_composite_backend_default_artifacts_root(self) -> None:
        """create_deep_agent with CompositeBackend without artifacts_root defaults to '/'."""
        backend = CompositeBackend(default=StateBackend(), routes={})
        assert backend.artifacts_root == "/"

        capturing_middleware = SystemMessageCapturingMiddleware()
        agent = create_deep_agent(
            model=FakeChatModelWithHistory(messages=iter([AIMessage(content="done")])),
            backend=backend,
            middleware=[capturing_middleware],
        )
        agent.invoke({"messages": [HumanMessage(content="Hi")]})
        system_content = str(capturing_middleware.captured_system_messages[0].content)
        assert "/large_tool_results/" in system_content

    def test_human_message_eviction_uses_artifacts_root(self) -> None:
        """Oversized HumanMessage is evicted under the custom artifacts_root."""
        store_backend = StoreBackend(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )

        threshold = 50_000
        large_content = "x" * (NUM_CHARS_PER_TOKEN * threshold + 1)

        model = FakeChatModelWithHistory(messages=iter([AIMessage(content="Got it.")]))

        agent = create_deep_agent(model=model, backend=backend, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "eviction-artifacts-root"}}
        result = agent.invoke({"messages": [HumanMessage(content=large_content)]}, config)

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        evicted_to = human_messages[0].additional_kwargs.get("lc_evicted_to")
        assert evicted_to is not None
        assert evicted_to.startswith("/workspace/conversation_history/")

        default_ls = backend.ls("/conversation_history/")
        assert not default_ls.entries

    async def test_async_human_message_eviction_uses_artifacts_root(self) -> None:
        """Async: oversized HumanMessage is evicted under the custom artifacts_root."""
        store_backend = StoreBackend(store=InMemoryStore(), namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )

        threshold = 50_000
        large_content = "x" * (NUM_CHARS_PER_TOKEN * threshold + 1)

        model = FakeChatModelWithHistory(messages=iter([AIMessage(content="Got it.")]))

        agent = create_deep_agent(model=model, backend=backend, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "async-eviction-artifacts-root"}}
        result = await agent.ainvoke({"messages": [HumanMessage(content=large_content)]}, config)

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        evicted_to = human_messages[0].additional_kwargs.get("lc_evicted_to")
        assert evicted_to is not None
        assert evicted_to.startswith("/workspace/conversation_history/")

        default_ls = backend.ls("/conversation_history/")
        assert not default_ls.entries


class TestAsyncSubagentEndToEnd:
    """End-to-end tests for async (non-blocking) subagent tools.

    These tests wire up ``create_deep_agent`` with ``AsyncSubAgent`` specs and
    mock the LangGraph SDK client to verify the full agent loop: model emits a
    tool call → tool executes against the (mocked) remote server → state is
    updated with ``async_tasks`` → model sees the result and responds.
    """

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_async_task(self, mock_get_client: MagicMock) -> None:
        """Agent launches an async subagent and receives a task ID."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_001"}
        mock_client.runs.create.return_value = {"run_id": "run_001"}
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Research quantum computing",
                                    "subagent_type": "researcher",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="I've launched the researcher. Task ID: thread_001"),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches topics in depth.",
                    "graph_id": "research_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Research quantum computing")]})

        assert "messages" in result
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert any("thread_001" in m.content for m in tool_messages)

        async_tasks = result.get("async_tasks", {})
        assert "thread_001" in async_tasks
        task = async_tasks["thread_001"]
        assert task["agent_name"] == "researcher"
        assert task["status"] == "running"
        assert task["run_id"] == "run_001"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_then_check_completed_task(self, mock_get_client: MagicMock) -> None:
        """Agent starts a task, then checks it after completion."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_002"}
        mock_client.runs.create.return_value = {"run_id": "run_002"}
        mock_client.runs.get.return_value = {"run_id": "run_002", "status": "success"}
        mock_client.threads.get.return_value = {
            "values": {
                "messages": [
                    {"role": "assistant", "content": "Quantum computing uses qubits."},
                ]
            }
        }
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Research quantum computing",
                                    "subagent_type": "researcher",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "check_async_task",
                                "args": {"task_id": "thread_002"},
                                "id": "call_check",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="The research is complete: Quantum computing uses qubits."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches topics.",
                    "graph_id": "research_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Research quantum computing")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        check_msg = next(m for m in tool_messages if m.tool_call_id == "call_check")
        parsed = json.loads(check_msg.content)
        assert parsed["status"] == "success"
        assert "qubits" in parsed["result"]

        async_tasks = result.get("async_tasks", {})
        assert async_tasks["thread_002"]["status"] == "success"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_then_cancel_task(self, mock_get_client: MagicMock) -> None:
        """Agent starts a task, then cancels it."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_003"}
        mock_client.runs.create.return_value = {"run_id": "run_003"}
        mock_client.runs.cancel.return_value = None
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Long running analysis",
                                    "subagent_type": "analyst",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "cancel_async_task",
                                "args": {"task_id": "thread_003"},
                                "id": "call_cancel",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="The analysis task has been cancelled."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "analyst",
                    "description": "Performs data analysis.",
                    "graph_id": "analysis_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Cancel the analysis")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        cancel_msg = next(m for m in tool_messages if m.tool_call_id == "call_cancel")
        assert "Cancelled" in cancel_msg.content

        async_tasks = result.get("async_tasks", {})
        assert async_tasks["thread_003"]["status"] == "cancelled"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_then_update_task(self, mock_get_client: MagicMock) -> None:
        """Agent starts a task, then sends an update message."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_004"}
        mock_client.runs.create.side_effect = [
            {"run_id": "run_004"},
            {"run_id": "run_004_updated"},
        ]
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Write a report on AI trends",
                                    "subagent_type": "writer",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "update_async_task",
                                "args": {
                                    "task_id": "thread_004",
                                    "message": "Focus specifically on LLM trends",
                                },
                                "id": "call_update",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="I've updated the writer with new instructions."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "writer",
                    "description": "Writes reports.",
                    "graph_id": "writer_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Update the report")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        update_msg = next(m for m in tool_messages if m.tool_call_id == "call_update")
        assert "Updated" in update_msg.content

        async_tasks = result.get("async_tasks", {})
        task = async_tasks["thread_004"]
        assert task["run_id"] == "run_004_updated"
        assert task["status"] == "running"

        mock_client.runs.create.assert_any_call(
            thread_id="thread_004",
            assistant_id="writer_graph",
            input={"messages": [{"role": "user", "content": "Focus specifically on LLM trends"}]},
            multitask_strategy="interrupt",
        )

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_list_async_tasks(self, mock_get_client: MagicMock) -> None:
        """Agent starts two tasks, then lists them."""
        mock_client = MagicMock()
        mock_client.threads.create.side_effect = [
            {"thread_id": "thread_A"},
            {"thread_id": "thread_B"},
        ]
        mock_client.runs.create.side_effect = [
            {"run_id": "run_A"},
            {"run_id": "run_B"},
        ]
        mock_client.runs.get.side_effect = [
            {"run_id": "run_A", "status": "running"},
            {"run_id": "run_B", "status": "success"},
        ]
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Task A",
                                    "subagent_type": "worker",
                                },
                                "id": "call_start_a",
                                "type": "tool_call",
                            },
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Task B",
                                    "subagent_type": "worker",
                                },
                                "id": "call_start_b",
                                "type": "tool_call",
                            },
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "list_async_tasks",
                                "args": {},
                                "id": "call_list",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Here are the task statuses."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "worker",
                    "description": "General worker.",
                    "graph_id": "worker_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Start two tasks and list them")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        list_msg = next(m for m in tool_messages if m.tool_call_id == "call_list")
        assert "2 tracked task(s)" in list_msg.content
        assert "thread_A" in list_msg.content
        assert "thread_B" in list_msg.content

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_invalid_subagent_type(self, mock_get_client: MagicMock) -> None:
        """Agent tries to start a subagent with an invalid type and gets an error."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Do something",
                                    "subagent_type": "nonexistent",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="That subagent type doesn't exist."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches topics.",
                    "graph_id": "research_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Use a nonexistent agent")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "Unknown async subagent type" in tool_messages[0].content
        assert "`researcher`" in tool_messages[0].content

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_nonexistent_task(self, mock_get_client: MagicMock) -> None:
        """Agent tries to check a task that doesn't exist."""
        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "check_async_task",
                                "args": {"task_id": "nonexistent_thread"},
                                "id": "call_check",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="That task doesn't exist."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Researches topics.",
                    "graph_id": "research_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Check a nonexistent task")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "No tracked task found" in tool_messages[0].content

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_async_subagents_coexist_with_sync_subagents(self, mock_get_client: MagicMock) -> None:
        """Async and sync subagent tools are both available on the same agent."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_async"}
        mock_client.runs.create.return_value = {"run_id": "run_async"}
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Remote research",
                                    "subagent_type": "remote-researcher",
                                },
                                "id": "call_async",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Launched the remote researcher."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "remote-researcher",
                    "description": "Researches remotely.",
                    "graph_id": "research_graph",
                    "url": "http://localhost:8123",
                },
            ],
        )

        tool_names = set(agent.nodes["tools"].bound._tools_by_name.keys())
        assert "task" in tool_names
        assert "start_async_task" in tool_names
        assert "check_async_task" in tool_names
        assert "update_async_task" in tool_names
        assert "cancel_async_task" in tool_names
        assert "list_async_tasks" in tool_names

        result = agent.invoke({"messages": [HumanMessage(content="Do remote research")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert any("thread_async" in m.content for m in tool_messages)

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_start_then_check_errored_task(self, mock_get_client: MagicMock) -> None:
        """Agent starts a task that errors, then checks it and sees the error."""
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_err"}
        mock_client.runs.create.return_value = {"run_id": "run_err"}
        mock_client.runs.get.return_value = {
            "run_id": "run_err",
            "status": "error",
            "error": "Out of memory",
        }
        mock_get_client.return_value = mock_client

        model = FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "start_async_task",
                                "args": {
                                    "description": "Heavy computation",
                                    "subagent_type": "compute",
                                },
                                "id": "call_start",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "check_async_task",
                                "args": {"task_id": "thread_err"},
                                "id": "call_check",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="The task failed with an out of memory error."),
                ]
            )
        )

        agent = create_deep_agent(
            model=model,
            subagents=[
                {
                    "name": "compute",
                    "description": "Runs heavy computations.",
                    "graph_id": "compute_graph",
                    "url": "http://localhost:8123",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="Run heavy computation")]})

        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        check_msg = next(m for m in tool_messages if m.tool_call_id == "call_check")
        parsed = json.loads(check_msg.content)
        assert parsed["status"] == "error"
        assert "Out of memory" in parsed["error"]

        async_tasks = result.get("async_tasks", {})
        assert async_tasks["thread_err"]["status"] == "error"
