import pytest
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.graph.message import add_messages

from deepagents.middleware.filesystem import (
    FILESYSTEM_SYSTEM_PROMPT,
    FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT,
    FileData,
    FilesystemMiddleware,
    FilesystemState,
    _create_file_data,
    _update_file_data,
)
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import DEFAULT_GENERAL_PURPOSE_DESCRIPTION, TASK_SYSTEM_PROMPT, TASK_TOOL_DESCRIPTION, SubAgentMiddleware


class TestAddMiddleware:
    def test_filesystem_middleware(self):
        middleware = [FilesystemMiddleware()]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools

    def test_subagent_middleware(self):
        middleware = [SubAgentMiddleware(default_tools=[], subagents=[], default_model="claude-sonnet-4-20250514")]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "task" in agent.nodes["tools"].bound._tools_by_name.keys()

    def test_multiple_middleware(self):
        middleware = [FilesystemMiddleware(), SubAgentMiddleware(default_tools=[], subagents=[], default_model="claude-sonnet-4-20250514")]
        agent = create_agent(model="claude-sonnet-4-20250514", middleware=middleware, tools=[])
        assert "files" in agent.stream_channels
        agent_tools = agent.nodes["tools"].bound._tools_by_name.keys()
        assert "ls" in agent_tools
        assert "read_file" in agent_tools
        assert "write_file" in agent_tools
        assert "edit_file" in agent_tools
        assert "task" in agent_tools


class TestFilesystemMiddleware:
    def test_init_local(self):
        middleware = FilesystemMiddleware(long_term_memory=False)
        assert middleware.long_term_memory is False
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
        assert len(middleware.tools) == 4

    def test_init_longterm(self):
        middleware = FilesystemMiddleware(long_term_memory=True)
        assert middleware.long_term_memory is True
        assert middleware.system_prompt == (FILESYSTEM_SYSTEM_PROMPT + FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT)
        assert len(middleware.tools) == 4

    def test_init_custom_system_prompt_shortterm(self):
        middleware = FilesystemMiddleware(long_term_memory=False, system_prompt="Custom system prompt")
        assert middleware.long_term_memory is False
        assert middleware.system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 4

    def test_init_custom_system_prompt_longterm(self):
        middleware = FilesystemMiddleware(long_term_memory=True, system_prompt="Custom system prompt")
        assert middleware.long_term_memory is True
        assert middleware.system_prompt == "Custom system prompt"
        assert len(middleware.tools) == 4

    def test_init_custom_tool_descriptions_shortterm(self):
        middleware = FilesystemMiddleware(long_term_memory=False, custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert middleware.long_term_memory is False
        assert middleware.system_prompt == FILESYSTEM_SYSTEM_PROMPT
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_init_custom_tool_descriptions_longterm(self):
        middleware = FilesystemMiddleware(long_term_memory=True, custom_tool_descriptions={"ls": "Custom ls tool description"})
        assert middleware.long_term_memory is True
        assert middleware.system_prompt == (FILESYSTEM_SYSTEM_PROMPT + FILESYSTEM_SYSTEM_PROMPT_LONGTERM_SUPPLEMENT)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        assert ls_tool.description == "Custom ls tool description"

    def test_ls_shortterm(self):
        state = FilesystemState(
            messages=[],
            files={
                "test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware(long_term_memory=False)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke(
            {"runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={})}
        )
        assert result == ["test.txt", "test2.txt"]

    def test_ls_shortterm_with_path(self):
        state = FilesystemState(
            messages=[],
            files={
                "/test.txt": FileData(
                    content=["Hello world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/test2.txt": FileData(
                    content=["Goodbye world"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/charmander.txt": FileData(
                    content=["Ember"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
                "/pokemon/water/squirtle.txt": FileData(
                    content=["Water"],
                    modified_at="2021-01-01",
                    created_at="2021-01-01",
                ),
            },
        )
        middleware = FilesystemMiddleware(long_term_memory=False)
        ls_tool = next(tool for tool in middleware.tools if tool.name == "ls")
        result = ls_tool.invoke(
            {
                "path": "pokemon/",
                "runtime": ToolRuntime(state=state, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={}),
            }
        )
        assert "/pokemon/test2.txt" in result
        assert "/pokemon/charmander.txt" in result

    def test_create_file_data_splits_long_lines(self):
        long_line = "a" * 3500
        short_line = "short line"
        content = f"{short_line}\n{long_line}"

        file_data = _create_file_data(content)

        for line in file_data["content"]:
            assert len(line) <= 2000

        assert len(file_data["content"]) == 3
        assert file_data["content"][0] == short_line
        assert file_data["content"][1] == "a" * 2000
        assert file_data["content"][2] == "a" * 1500

    def test_update_file_data_splits_long_lines(self):
        initial_file_data = _create_file_data("initial content")

        long_line = "b" * 5000
        short_line = "another short line"
        new_content = f"{short_line}\n{long_line}"

        updated_file_data = _update_file_data(initial_file_data, new_content)

        for line in updated_file_data["content"]:
            assert len(line) <= 2000

        assert len(updated_file_data["content"]) == 4
        assert updated_file_data["content"][0] == short_line
        assert updated_file_data["content"][1] == "b" * 2000
        assert updated_file_data["content"][2] == "b" * 2000
        assert updated_file_data["content"][3] == "b" * 1000

        assert updated_file_data["created_at"] == initial_file_data["created_at"]


@pytest.mark.requires("langchain_openai")
class TestSubagentMiddleware:
    """Test the SubagentMiddleware class."""

    def test_subagent_middleware_init(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
        )
        assert middleware is not None
        assert middleware.system_prompt is TASK_SYSTEM_PROMPT
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"
        expected_desc = TASK_TOOL_DESCRIPTION.format(available_agents=f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")
        assert middleware.tools[0].description == expected_desc

    def test_default_subagent_with_tools(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
        )
        assert middleware is not None
        assert middleware.system_prompt == TASK_SYSTEM_PROMPT

    def test_default_subagent_custom_system_prompt(self):
        middleware = SubAgentMiddleware(
            default_model="gpt-4o-mini",
            default_tools=[],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        assert middleware.system_prompt == "Use the task tool to call a subagent."


class TestPatchToolCallsMiddleware:
    def test_first_message(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 3
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1].type == "system"
        assert state_update["messages"][1].content == "You are a helpful assistant."
        assert state_update["messages"][2].type == "human"
        assert state_update["messages"][2].content == "Hello, how are you?"
        assert state_update["messages"][2].id == "2"

    def test_missing_tool_call(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]

    def test_no_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            ToolMessage(content="I have no events for that date.", tool_call_id="123", id="4"),
            HumanMessage(content="What is the weather in Tokyo?", id="5"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 6
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1:] == input_messages
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 5
        assert updated_messages == input_messages

    def test_two_missing_tool_calls(self) -> None:
        input_messages = [
            SystemMessage(content="You are a helpful assistant.", id="1"),
            HumanMessage(content="Hello, how are you?", id="2"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="123", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="3",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="4"),
            AIMessage(
                content="I'm doing well, thank you!",
                tool_calls=[ToolCall(id="456", name="get_events_for_days", args={"date_str": "2025-01-01"})],
                id="5",
            ),
            HumanMessage(content="What is the weather in Tokyo?", id="6"),
        ]
        middleware = PatchToolCallsMiddleware()
        state_update = middleware.before_agent({"messages": input_messages}, None)
        assert state_update is not None
        assert len(state_update["messages"]) == 9
        assert state_update["messages"][0].type == "remove"
        assert state_update["messages"][1] == input_messages[0]
        assert state_update["messages"][2] == input_messages[1]
        assert state_update["messages"][3] == input_messages[2]
        assert state_update["messages"][4].type == "tool"
        assert state_update["messages"][4].tool_call_id == "123"
        assert state_update["messages"][4].name == "get_events_for_days"
        assert state_update["messages"][5] == input_messages[3]
        assert state_update["messages"][6] == input_messages[4]
        assert state_update["messages"][7].type == "tool"
        assert state_update["messages"][7].tool_call_id == "456"
        assert state_update["messages"][7].name == "get_events_for_days"
        assert state_update["messages"][8] == input_messages[5]
        updated_messages = add_messages(input_messages, state_update["messages"])
        assert len(updated_messages) == 8
        assert updated_messages[0] == input_messages[0]
        assert updated_messages[1] == input_messages[1]
        assert updated_messages[2] == input_messages[2]
        assert updated_messages[3].type == "tool"
        assert updated_messages[3].tool_call_id == "123"
        assert updated_messages[3].name == "get_events_for_days"
        assert updated_messages[4] == input_messages[3]
        assert updated_messages[5] == input_messages[4]
        assert updated_messages[6].type == "tool"
        assert updated_messages[6].tool_call_id == "456"
        assert updated_messages[6].name == "get_events_for_days"
        assert updated_messages[7] == input_messages[5]
