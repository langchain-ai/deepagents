"""Test for large single-line file bug.

When tool results are evicted via str(dict), they become a single line.
read_file chunks this into 100 lines x 5000 chars = 500K chars - token overflow.
"""

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.utils import TOOL_RESULT_TOKEN_LIMIT
from deepagents.graph import create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel

MAX_REASONABLE_CHARS = TOOL_RESULT_TOKEN_LIMIT * 4  # 80,000 chars, same as truncate_if_too_long


class TestLargeSingleLineFile:
    """Tests for reading large single-line files."""

    def test_read_large_single_line_file_returns_reasonable_size(self) -> None:
        """read_file should not return 500K chars for a single-line file."""
        # str(dict) produces no newlines—exactly how evicted tool results are serialized
        large_dict = {"records": [{"id": i, "data": "x" * 100} for i in range(4000)]}
        large_content = str(large_dict)
        assert "\n" not in large_content

        fake_model = GenericFakeChatModel(
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

        assert len(read_file_response.content) <= MAX_REASONABLE_CHARS, (
            f"read_file returned {len(read_file_response.content):,} chars. "
            f"Expected <= {MAX_REASONABLE_CHARS:,} chars (TOOL_RESULT_TOKEN_LIMIT * 4). "
            f"A single-line file should not cause token overflow."
        )
