"""Unit tests for FilesystemMiddleware before_model hook."""

from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.filesystem import FilesystemMiddleware


def test_no_messages() -> None:
    """Test that middleware returns None when there are no messages."""
    middleware = FilesystemMiddleware()
    state = {"messages": []}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is None


def test_not_enough_ai_messages() -> None:
    """Test that middleware returns None when there aren't enough AI messages to trigger cleaning."""
    middleware = FilesystemMiddleware(old_messages_length=20)

    # Only 5 AI messages, not enough to trigger cleaning
    messages = []
    for i in range(5):
        messages.append(HumanMessage(content=f"Request {i}"))
        messages.append(AIMessage(content=f"Response {i}"))

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is None


def test_clean_old_write_file_tool_call() -> None:
    """Test that old write_file tool calls with large arguments get truncated."""
    middleware = FilesystemMiddleware(old_messages_length=3, max_write_edit_arg_length=100)

    large_content = "x" * 200  # Large content that should be truncated

    messages = [
        # Old AI message with write_file tool call (will be cleaned)
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="call_1"),
        # Add enough AI messages to push the first one past the threshold
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Request 3"),
        AIMessage(content="Response 3"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is not None
    assert "messages" in result
    cleaned_messages = result["messages"].value

    # Check that the old tool call was cleaned
    first_ai_msg = cleaned_messages[0]
    assert isinstance(first_ai_msg, AIMessage)
    assert len(first_ai_msg.tool_calls) == 1
    assert first_ai_msg.tool_calls[0]["name"] == "write_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.txt"
    # Content should be first 20 chars + truncation text
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_clean_old_edit_file_tool_call() -> None:
    """Test that old edit_file tool calls with large arguments get truncated."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=50)

    large_old_string = "a" * 100
    large_new_string = "b" * 100

    messages = [
        # Old AI message with edit_file tool call (will be cleaned)
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "edit_file",
                    "args": {
                        "file_path": "/test.py",
                        "old_string": large_old_string,
                        "new_string": large_new_string,
                    },
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File edited", tool_call_id="call_1"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is not None
    cleaned_messages = result["messages"].value

    # Check that both old_string and new_string were truncated
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["name"] == "edit_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.py"
    assert first_ai_msg.tool_calls[0]["args"]["old_string"] == "a" * 20 + "...(argument truncated)"
    assert first_ai_msg.tool_calls[0]["args"]["new_string"] == "b" * 20 + "...(argument truncated)"


def test_recent_messages_not_cleaned() -> None:
    """Test that recent AI messages are not cleaned."""
    middleware = FilesystemMiddleware(old_messages_length=5, max_write_edit_arg_length=100)

    large_content = "x" * 200

    messages = [
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
        HumanMessage(content="Request 3"),
        # Recent AI message with write_file (should NOT be cleaned)
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="call_1"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    # No cleaning should happen since the tool call is recent
    assert result is None


def test_other_tool_calls_not_affected() -> None:
    """Test that tool calls other than write_file and edit_file are not affected."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=50)

    large_content = "x" * 200

    messages = [
        # Old AI message with read_file tool call (should NOT be cleaned)
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File content", tool_call_id="call_1"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    # No cleaning should happen since it's not write_file or edit_file
    assert result is None


def test_mixed_tool_calls() -> None:
    """Test that only write_file and edit_file are cleaned in a message with multiple tool calls."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=50)

    large_content = "x" * 200

    messages = [
        # Old AI message with multiple tool calls
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": "/test.txt"},
                    "id": "call_1",
                },
                {
                    "name": "write_file",
                    "args": {"file_path": "/output.txt", "content": large_content},
                    "id": "call_2",
                },
                {
                    "name": "shell",
                    "args": {"command": "ls -la"},
                    "id": "call_3",
                },
            ],
        ),
        ToolMessage(content="File content", tool_call_id="call_1"),
        ToolMessage(content="File written", tool_call_id="call_2"),
        ToolMessage(content="Output", tool_call_id="call_3"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is not None
    cleaned_messages = result["messages"].value

    first_ai_msg = cleaned_messages[0]
    assert len(first_ai_msg.tool_calls) == 3  # noqa: PLR2004

    # read_file should be unchanged
    assert first_ai_msg.tool_calls[0]["name"] == "read_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.txt"
    assert "content" not in first_ai_msg.tool_calls[0]["args"]

    # write_file should be cleaned
    assert first_ai_msg.tool_calls[1]["name"] == "write_file"
    assert first_ai_msg.tool_calls[1]["args"]["content"] == "x" * 20 + "...(argument truncated)"

    # shell should be unchanged
    assert first_ai_msg.tool_calls[2]["name"] == "shell"
    assert first_ai_msg.tool_calls[2]["args"]["command"] == "ls -la"


def test_custom_truncation_text() -> None:
    """Test that custom truncation text is used."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=50, truncation_text="[TRUNCATED]")

    large_content = "y" * 100

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="call_1"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    assert result is not None
    cleaned_messages = result["messages"].value

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "y" * 20 + "[TRUNCATED]"


def test_small_arguments_not_truncated() -> None:
    """Test that small arguments are not truncated even in old messages."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=100)

    small_content = "short"

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": small_content},
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="call_1"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    # No modification should happen since content is small
    assert result is None


def test_non_string_arguments_not_affected() -> None:
    """Test that non-string arguments are not truncated."""
    middleware = FilesystemMiddleware(old_messages_length=2, max_write_edit_arg_length=50)

    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {
                        "file_path": "/test.txt",
                        "content": "some content",
                        "mode": 123,  # Non-string argument
                        "options": {"key": "value"},  # Dict argument
                    },
                    "id": "call_1",
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="call_1"),
        HumanMessage(content="Request 1"),
        AIMessage(content="Response 1"),
        HumanMessage(content="Request 2"),
        AIMessage(content="Response 2"),
    ]

    state = {"messages": messages}
    runtime = Mock()

    result = middleware.before_model(state, runtime)

    # No modification since no string arguments exceed max_write_edit_arg_length
    assert result is None
