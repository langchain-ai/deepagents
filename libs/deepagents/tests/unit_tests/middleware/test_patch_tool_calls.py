"""Unit tests for PatchToolCallsMiddleware.

This module tests that the middleware correctly patches dangling tool calls
and avoids unnecessary state modifications when no patching is needed.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware


class TestPatchToolCallsMiddlewareNoPatchingNeeded:
    """Tests for scenarios where no patching is needed (should return None)."""

    def test_returns_none_when_messages_is_empty(self) -> None:
        """Test that empty messages list returns None."""
        middleware = PatchToolCallsMiddleware()
        result = middleware.before_agent({"messages": []}, None)  # type: ignore

        assert result is None

    def test_returns_none_when_messages_is_none(self) -> None:
        """Test that None messages returns None."""
        middleware = PatchToolCallsMiddleware()
        result = middleware.before_agent({"messages": None}, None)  # type: ignore

        assert result is None

    def test_returns_none_when_no_ai_messages_with_tool_calls(self) -> None:
        """Test that messages without tool calls return None."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is None

    def test_returns_none_when_all_tool_calls_have_responses(self) -> None:
        """Test that complete tool call chains return None."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            HumanMessage(content="Read a file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "read_file",
                        "args": {"path": "/test.txt"},
                    }
                ],
            ),
            ToolMessage(
                content="File contents here",
                name="read_file",
                tool_call_id="call_123",
            ),
            AIMessage(content="Here's the file content"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is None

    def test_returns_none_when_ai_message_has_empty_tool_calls(self) -> None:
        """Test that AI messages with empty tool_calls list return None."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            AIMessage(content="No tools", tool_calls=[]),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is None

    def test_returns_none_when_ai_message_has_no_tool_calls_attribute(self) -> None:
        """Test that AI messages without tool_calls attribute return None."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            AIMessage(content="Just a regular response"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is None


class TestPatchToolCallsMiddlewareDanglingToolCalls:
    """Tests for scenarios where dangling tool calls need patching."""

    def test_patches_single_dangling_tool_call(self) -> None:
        """Test that a single dangling tool call gets a synthetic ToolMessage."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            HumanMessage(content="Read a file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "read_file",
                        "args": {"path": "/test.txt"},
                    }
                ],
            ),
            HumanMessage(content="Never mind"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is not None
        assert "messages" in result

        # Unwrap Overwrite to get the actual messages
        patched_messages = result["messages"].value

        # Should have 3 original + 1 synthetic ToolMessage
        assert len(patched_messages) == 4

        # Find the synthetic ToolMessage
        tool_messages = [m for m in patched_messages if m.type == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "call_123"
        assert "cancelled" in tool_messages[0].content
        assert tool_messages[0].name == "read_file"

    def test_patches_multiple_dangling_tool_calls_in_single_ai_message(self) -> None:
        """Test that multiple dangling tool calls in one AI message get patched."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            HumanMessage(content="Do multiple things"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "tool_a", "args": {}},
                    {"id": "call_2", "name": "tool_b", "args": {}},
                ],
            ),
            # Both tool calls are dangling
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is not None
        patched_messages = result["messages"].value

        # Should have 2 original + 2 synthetic ToolMessages
        assert len(patched_messages) == 4

        tool_messages = [m for m in patched_messages if m.type == "tool"]
        assert len(tool_messages) == 2
        tool_call_ids = {m.tool_call_id for m in tool_messages}
        assert tool_call_ids == {"call_1", "call_2"}

    def test_patches_multiple_ai_messages_with_dangling_tool_calls(self) -> None:
        """Test that multiple AI messages with dangling tool calls all get patched."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "tool_a", "args": {}}],
            ),
            HumanMessage(content="msg1"),
            AIMessage(
                content="",
                tool_calls=[{"id": "call_2", "name": "tool_b", "args": {}}],
            ),
            HumanMessage(content="msg2"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is not None
        patched_messages = result["messages"].value

        # Should have 4 original + 2 synthetic ToolMessages
        assert len(patched_messages) == 6

        tool_messages = [m for m in patched_messages if m.type == "tool"]
        assert len(tool_messages) == 2

    def test_only_patches_dangling_tool_calls_not_ones_with_responses(self) -> None:
        """Test that only dangling tool calls are patched, not complete ones."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            HumanMessage(content="Do two things"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "read_file", "args": {}},
                    {"id": "call_2", "name": "write_file", "args": {}},
                ],
            ),
            ToolMessage(
                content="File written successfully",
                name="write_file",
                tool_call_id="call_2",
            ),
            HumanMessage(content="Thanks"),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is not None
        patched_messages = result["messages"].value

        # Should have 4 original + 1 synthetic ToolMessage for call_1
        assert len(patched_messages) == 5

        # Find synthetic ToolMessage (contains "cancelled")
        synthetic_messages = [
            m for m in patched_messages if m.type == "tool" and "cancelled" in m.content
        ]
        assert len(synthetic_messages) == 1
        assert synthetic_messages[0].tool_call_id == "call_1"

        # Verify original ToolMessage for call_2 is still there
        original_messages = [
            m
            for m in patched_messages
            if m.type == "tool" and m.content == "File written successfully"
        ]
        assert len(original_messages) == 1
        assert original_messages[0].tool_call_id == "call_2"

    def test_synthetic_tool_message_includes_tool_name_and_id(self) -> None:
        """Test that synthetic ToolMessage includes tool name and call ID in content."""
        middleware = PatchToolCallsMiddleware()
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "unique_call_id_xyz", "name": "special_tool", "args": {}},
                ],
            ),
        ]

        result = middleware.before_agent({"messages": messages}, None)  # type: ignore

        assert result is not None
        patched_messages = result["messages"].value

        tool_message = [m for m in patched_messages if m.type == "tool"][0]
        assert "special_tool" in tool_message.content
        assert "unique_call_id_xyz" in tool_message.content
        assert tool_message.name == "special_tool"
