from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import (
    FileData,
    _file_data_reducer,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_parallel_write_file_calls_trigger_list_reducer():
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


def test_file_data_reducer_handles_list():
    """Verify _file_data_reducer can handle a list of updates."""
    # Simulate initial state where left is a list of dicts
    left = [
        {"/file1.txt": FileData(content=["one"], created_at="", modified_at="")},
        {"/file2.txt": FileData(content=["two"], created_at="", modified_at="")},
    ]
    right = {"/file3.txt": FileData(content=["three"], created_at="", modified_at="")}

    # The reducer should merge the list and the new dict
    result = _file_data_reducer(left, right)

    assert result == {
        "/file1.txt": FileData(content=["one"], created_at="", modified_at=""),
        "/file2.txt": FileData(content=["two"], created_at="", modified_at=""),
        "/file3.txt": FileData(content=["three"], created_at="", modified_at=""),
    }

    # Also test with deletions within the list-based state
    right_with_deletion = {
        "/file1.txt": None,
        "/file4.txt": FileData(content=["four"], created_at="", modified_at=""),
    }
    result_with_deletion = _file_data_reducer(left, right_with_deletion)
    assert result_with_deletion == {
        "/file2.txt": FileData(content=["two"], created_at="", modified_at=""),
        "/file4.txt": FileData(content=["four"], created_at="", modified_at=""),
    }
