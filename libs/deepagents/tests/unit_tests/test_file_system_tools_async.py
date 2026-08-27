"""Async unit tests for file system tools path validation.

This module contains async versions of the path validation error handling tests.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


async def test_path_traversal_returns_error_message_async() -> None:
    """Verify that path traversal attempts return error messages instead of crashing."""
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "./question/..",
                                "old_string": "test",
                                "new_string": "replaced",
                            },
                            "id": "call_path_traversal",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see there was an error with the path."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    # This should NOT raise an exception - it should return an error message
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Edit a file with bad path")]},
        config={"configurable": {"thread_id": "test_path_traversal_async"}},
    )

    # Find the ToolMessage in the result
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    # The tool message should contain an error about path traversal
    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ./question/.."


async def test_windows_absolute_path_returns_error_message_async() -> None:
    """Verify that Windows absolute paths return error messages instead of crashing."""
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {
                                "file_path": "C:\\Users\\test\\file.txt",
                            },
                            "id": "call_windows_path",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see there was an error with the path."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Read a file with Windows path")]},
        config={"configurable": {"thread_id": "test_windows_path_async"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    expected_error = (
        "Error: Windows absolute paths are not supported: C:\\Users\\test\\file.txt. "
        "Please use virtual paths starting with / (e.g., /workspace/file.txt)"
    )
    assert error_message == expected_error


async def test_tilde_path_returns_error_message_async() -> None:
    """Verify that tilde paths return error messages instead of crashing."""
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "~/secret.txt",
                                "content": "secret data",
                            },
                            "id": "call_tilde_path",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see there was an error with the path."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Write a file with tilde path")]},
        config={"configurable": {"thread_id": "test_tilde_path_async"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ~/secret.txt"


async def test_ls_with_invalid_path_returns_error_message_async() -> None:
    """Verify that ls tool with invalid path returns error message instead of crashing."""
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ls",
                            "args": {
                                "path": "../../../etc",
                            },
                            "id": "call_ls_invalid",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see there was an error with the path."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="List directory with invalid path")]},
        config={"configurable": {"thread_id": "test_ls_invalid_path_async"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ../../../etc"


async def test_ls_with_limit_truncates_entries_async() -> None:
    """Verify that async ls tool with limit truncates entries and appends a truncation note."""
    from deepagents.backends.state import StateBackend

    backend = StateBackend()
    for i in range(1, 6):
        backend.write(f"/project/file{i}.txt", "content")

    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ls",
                            "args": {
                                "path": "/project",
                                "limit": 2,
                            },
                            "id": "call_ls_limit_async",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see the truncated listing."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        filesystem_backend=backend,
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="List directory with limit async")]},
        config={"configurable": {"thread_id": "test_ls_limit_async"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    content = tool_messages[0].content
    assert tool_messages[0].status == "success"
    assert "[Showing 2 of 5 entries" in content


async def test_ls_with_invalid_limit_returns_error_async() -> None:
    """Verify that async ls tool rejects non-positive limit values with a clear error."""
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ls",
                            "args": {
                                "path": "/project",
                                "limit": -1,
                            },
                            "id": "call_ls_limit_neg_async",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I see there was an error."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="List directory with negative limit")]},
        config={"configurable": {"thread_id": "test_ls_limit_neg_async"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1

    error_message = tool_messages[0].content
    assert tool_messages[0].status == "error"
    assert "Error: limit must be > 0, got -1" == error_message
