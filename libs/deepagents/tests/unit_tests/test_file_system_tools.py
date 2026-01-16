"""End to end unit tests that verify that the deepagents can use file system tools.

At the moment these tests are written against the state backend, but we will need
to extend them to other backends as well.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tests.unit_tests.chat_model import GenericFakeChatModel

from deepagents.graph import create_deep_agent


def test_parallel_write_file_calls_trigger_list_reducer() -> None:
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
    assert result["files"]["/test1.txt"]["content"] == ["hello"], "Content of /test1.txt should be 'hello'"
    assert result["files"]["/test2.txt"]["content"] == ["world"], "Content of /test2.txt should be 'world'"


def test_edit_file_single_replacement() -> None:
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
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit the file")]},
        config={"configurable": {"thread_id": "test_thread_edit"}},
    )

    # Verify the file was edited correctly
    assert "/code.py" in result["files"], "File /code.py should exist"
    # Content is stored as a list of lines
    content_list = result["files"]["/code.py"]["content"]
    full_content = "\n".join(content_list)
    assert "hello universe" in full_content, f"Content should be updated, got: {content_list}"
    assert "hello world" not in full_content, "Old content should be replaced"


def test_edit_file_replace_all() -> None:
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
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit all occurrences")]},
        config={"configurable": {"thread_id": "test_thread_edit_all"}},
    )

    # Verify all occurrences were replaced
    assert "/data.txt" in result["files"], "File /data.txt should exist"
    content = result["files"]["/data.txt"]["content"][0]
    assert content == "qux bar qux baz qux", "All occurrences of 'foo' should be replaced with 'qux'"


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

    # Verify the file content was not changed
    assert "/test.txt" in result["files"], "File should exist"
    assert result["files"]["/test.txt"]["content"][0] == "hello world", "Content should remain unchanged"


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

    result = agent.invoke(
        {"messages": [HumanMessage(content="Edit file in parallel")]},
        config={"configurable": {"thread_id": "test_thread_parallel_edits"}},
    )

    # Verify both edits were applied
    assert "/multi.txt" in result["files"], "File should exist"
    # Content is stored as a list of lines, join them to get full content
    content_list = result["files"]["/multi.txt"]["content"]
    full_content = "\n".join(content_list)
    # Note: Due to parallel execution, both edits should be applied to the original content
    assert "line 1" in full_content or "line one" in full_content, f"First edit should be present or original, got: {content_list}"
    assert "line 2" in full_content or "line two" in full_content, f"Second edit should be present or original, got: {content_list}"
