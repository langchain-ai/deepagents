"""End to end unit tests that verify that the deepagents can use file system tools.

At the moment these tests are written against the state backend, but we will need
to extend them to other backends as well.
"""

from functools import partial

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.backends.state import StateBackend
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_parallel_write_file_calls_trigger_list_reducer(file_format: str) -> None:
    """Verify that parallel write_file calls correctly update file state.

    This test ensures that when an agent's model issues multiple `write_file`
    tool calls in parallel, the `_file_data_reducer` correctly handles the
    list of file updates and merges them into the final state.
    It guards against regressions of the `TypeError` that occurred when the
    reducer received a list instead of a dictionary.
    """
    # Fake model will issue two write_file tool calls in a single turn
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/test1.txt", "content": "hello"},
                            "id": "call_write_file_1",
                            "type": "tool_call",
                        },
                        {
                            "name": "write_file",
                            "args": {"file_path": "/test2.txt", "content": "world"},
                            "id": "call_write_file_2",
                            "type": "tool_call",
                        },
                    ],
                ),
                # Final acknowledgment message
                AIMessage(content="I have written the files."),
            ]
        )
    )

    # Create a deep agent with the fake model and a memory saver
    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
        backend=partial(StateBackend, file_format=file_format),
    )

    # Invoke the agent, which will trigger the parallel tool calls
    result = agent.invoke(
        {"messages": [HumanMessage(content="Write two files")]},
        config={"configurable": {"thread_id": "test_thread_parallel_writes"}},
    )

    # Verify that both files exist in the final state
    assert "/test1.txt" in result["files"], "File /test1.txt should exist in the final state"
    assert "/test2.txt" in result["files"], "File /test2.txt should exist in the final state"

    # Verify the content of the files
    expected_hello = ["hello"] if file_format == "v1" else "hello"
    expected_world = ["world"] if file_format == "v1" else "world"
    assert result["files"]["/test1.txt"]["content"] == expected_hello
    assert result["files"]["/test2.txt"]["content"] == expected_world


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_edit_file_single_replacement(file_format: str) -> None:
    """Verify that edit_file correctly replaces a single occurrence of a string."""
    # Fake model will write a file, then edit it
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/code.py", "content": "def hello():\n    print('hello world')"},
                            "id": "call_write_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/code.py",
                                "old_string": "hello world",
                                "new_string": "hello universe",
                            },
                            "id": "call_edit_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I have edited the file."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
        backend=partial(StateBackend, file_format=file_format),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit the file")]},
        config={"configurable": {"thread_id": "test_thread_edit"}},
    )

    # Verify the file was edited correctly
    assert "/code.py" in result["files"], "File /code.py should exist"
    full_content = result["files"]["/code.py"]["content"]
    if file_format == "v1":
        assert isinstance(full_content, list)
        text = "\n".join(full_content)
    else:
        assert isinstance(full_content, str)
        text = full_content
    assert "hello universe" in text, f"Content should be updated, got: {text}"
    assert "hello world" not in text, "Old content should be replaced"


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_edit_file_replace_all(file_format: str) -> None:
    """Verify that edit_file with replace_all replaces all occurrences of a string."""
    # Fake model will write a file with repeated content, then edit all occurrences
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/data.txt",
                                "content": "foo bar foo baz foo",
                            },
                            "id": "call_write_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/data.txt",
                                "old_string": "foo",
                                "new_string": "qux",
                                "replace_all": True,
                            },
                            "id": "call_edit_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I have edited all occurrences."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
        backend=partial(StateBackend, file_format=file_format),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit all occurrences")]},
        config={"configurable": {"thread_id": "test_thread_edit_all"}},
    )

    # Verify all occurrences were replaced
    assert "/data.txt" in result["files"], "File /data.txt should exist"
    content = result["files"]["/data.txt"]["content"]
    expected = ["qux bar qux baz qux"] if file_format == "v1" else "qux bar qux baz qux"
    assert content == expected, "All occurrences of 'foo' should be replaced with 'qux'"


def test_edit_file_nonexistent_file() -> None:
    """Verify that edit_file returns an error when attempting to edit a nonexistent file."""
    # Fake model will attempt to edit a file that doesn't exist
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/nonexistent.txt",
                                "old_string": "hello",
                                "new_string": "goodbye",
                            },
                            "id": "call_edit_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I tried to edit the file."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit nonexistent file")]},
        config={"configurable": {"thread_id": "test_thread_edit_nonexistent"}},
    )

    # Verify the error message in the ToolMessage
    tool_message = result["messages"][-2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Error: File '/nonexistent.txt' not found"

    # Verify the file doesn't exist in state
    assert "/nonexistent.txt" not in result.get("files", {}), "Nonexistent file should not be in state"


def test_edit_file_string_not_found() -> None:
    """Verify that edit_file returns an error when the old_string is not found in the file."""
    # Fake model will write a file, then attempt to edit with a string that doesn't exist
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"file_path": "/test.txt", "content": "hello world"},
                            "id": "call_write_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/test.txt",
                                "old_string": "goodbye",
                                "new_string": "farewell",
                            },
                            "id": "call_edit_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I tried to edit the file."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit with non-existent string")]},
        config={"configurable": {"thread_id": "test_thread_edit_not_found"}},
    )

    tool_message = result["messages"][-2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Error: String not found in file: 'goodbye'"


def test_grep_finds_written_file() -> None:
    """Verify that grep can find content in a file that was written."""
    # Fake model will write files with specific content, then grep for it
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/project/main.py",
                                "content": "import os\nimport sys\n\ndef main():\n    print('Hello World')",
                            },
                            "id": "call_write_1",
                            "type": "tool_call",
                        },
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/project/utils.py",
                                "content": "def helper():\n    return 42",
                            },
                            "id": "call_write_2",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "grep",
                            "args": {
                                "pattern": "import",
                                "output_mode": "content",
                            },
                            "id": "call_grep_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="Found the imports."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Write files and search")]},
        config={"configurable": {"thread_id": "test_thread_grep"}},
    )

    # Verify files were created
    assert "/project/main.py" in result["files"], "File /project/main.py should exist"
    assert "/project/utils.py" in result["files"], "File /project/utils.py should exist"

    # Verify grep found the pattern in messages
    grep_message = result["messages"][-2]
    assert isinstance(grep_message, ToolMessage)
    assert "import" in grep_message.content.lower(), "Grep should find 'import' in the files"
    assert "/project/main.py" in grep_message.content, "Grep should reference the file containing 'import'"


# Our reducers do not handle parallel edits in StateBackend.
# These will also not work correctly for other backends due to race conditions.
# Even sandbox/file system backend could get into some edge cases (e.g., if the edits are overlapping)
# Generally best to instruct the LLM to avoid parallel edits of the same file likely.
@pytest.mark.xfail(reason="We should add after_model middleware to fail parallel edits of the same file.")
def test_parallel_edit_file_calls() -> None:
    """Verify that parallel edit_file calls correctly update file state."""
    # Fake model will write a file, then issue multiple edit_file calls in parallel
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/multi.txt",
                                "content": "line one\nline two\nline three",
                            },
                            "id": "call_write_1",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/multi.txt",
                                "old_string": "one",
                                "new_string": "1",
                            },
                            "id": "call_edit_1",
                            "type": "tool_call",
                        },
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/multi.txt",
                                "old_string": "two",
                                "new_string": "2",
                            },
                            "id": "call_edit_2",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="I have edited the file in parallel."),
            ]
        )
    )

    agent = create_deep_agent(
        model=fake_model,
        checkpointer=InMemorySaver(),
    )

    _ = agent.invoke(
        {"messages": [HumanMessage(content="Edit file in parallel")]},
        config={"configurable": {"thread_id": "test_thread_parallel_edits"}},
    )
    assert False, "Finish implementing correct behavior to add a ToolMessage with error if parallel edits to the same file are attempted."  # noqa: PT015, B011


def test_path_traversal_returns_error_message() -> None:
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
    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit a file with bad path")]},
        config={"configurable": {"thread_id": "test_path_traversal"}},
    )

    # Find the ToolMessage in the result
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    # The tool message should contain an error about path traversal
    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ./question/.."


def test_windows_absolute_path_returns_error_message() -> None:
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

    result = agent.invoke(
        {"messages": [HumanMessage(content="Read a file with Windows path")]},
        config={"configurable": {"thread_id": "test_windows_path"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    expected_error = (
        "Error: Windows absolute paths are not supported: C:\\Users\\test\\file.txt. "
        "Please use virtual paths starting with / (e.g., /workspace/file.txt)"
    )
    assert error_message == expected_error


def test_tilde_path_returns_error_message() -> None:
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

    result = agent.invoke(
        {"messages": [HumanMessage(content="Write a file with tilde path")]},
        config={"configurable": {"thread_id": "test_tilde_path"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ~/secret.txt"


def test_ls_with_invalid_path_returns_error_message() -> None:
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

    result = agent.invoke(
        {"messages": [HumanMessage(content="List directory with invalid path")]},
        config={"configurable": {"thread_id": "test_ls_invalid_path"}},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) >= 1, "Expected at least one ToolMessage"

    error_message = tool_messages[0].content
    assert error_message == "Error: Path traversal not allowed: ../../../etc"


def test_delete_file_not_registered_by_default() -> None:
    """delete_file should not be in the tool list unless opted in."""
    from deepagents.middleware.filesystem import FilesystemMiddleware

    mw = FilesystemMiddleware()
    tool_names = {t.name for t in mw.tools}
    assert "delete_file" not in tool_names


def test_delete_file_registered_when_opted_in() -> None:
    """delete_file should appear when passed via additional_tools."""
    from deepagents.middleware.filesystem import FilesystemMiddleware

    mw = FilesystemMiddleware(additional_tools={"delete_file"})
    tool_names = {t.name for t in mw.tools}
    assert "delete_file" in tool_names


def _capture_filesystem_system_prompt(*, include_delete_file_tool: bool) -> str:
    """Invoke FilesystemMiddleware.wrap_model_call and return the injected system prompt text."""
    from langchain.agents.middleware.types import ModelRequest

    from deepagents.middleware.filesystem import FilesystemMiddleware

    tools: list = [t for t in FilesystemMiddleware().tools if t.name != "delete_file"]
    if include_delete_file_tool:
        delete_tool = FilesystemMiddleware(additional_tools={"delete_file"})._create_delete_file_tool()
        tools.append(delete_tool)

    mw = FilesystemMiddleware(
        additional_tools={"delete_file"} if include_delete_file_tool else None,
    )

    request = ModelRequest(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="hi")])),
        messages=[HumanMessage(content="hi")],
        system_message=None,
        tools=tools,
        state={"messages": [HumanMessage(content="hi")]},  # type: ignore[typeddict-unknown-key]
    )

    captured: dict = {}

    def handler(req):  # noqa: ANN001, ANN202
        captured["system_message"] = req.system_message
        return AIMessage(content="ok")

    mw.wrap_model_call(request, handler)

    msg = captured["system_message"]
    assert msg is not None
    content = msg.content
    return content if isinstance(content, str) else "\n".join(
        str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
    )


def test_system_prompt_omits_delete_file_when_tool_absent() -> None:
    """Prompt should not mention delete_file when the tool is not registered."""
    prompt = _capture_filesystem_system_prompt(include_delete_file_tool=False)
    assert "delete_file" not in prompt


def test_system_prompt_includes_delete_file_when_tool_present() -> None:
    """Prompt should advertise delete_file in the header and bullet list when registered."""
    prompt = _capture_filesystem_system_prompt(include_delete_file_tool=True)
    assert "`delete_file`" in prompt
    assert "- delete_file: delete a file from the filesystem" in prompt
