"""Tests for message store and serialization."""

import pytest

from deepagents_cli.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_cli.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    SystemMessage,
    ToolCallMessage,
    UserMessage,
)


class TestMessageData:
    """Tests for MessageData serialization."""

    def test_user_message_roundtrip(self):
        """Test UserMessage serialization and deserialization."""
        original = UserMessage("Hello, world!", id="test-user-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.USER
        assert data.content == "Hello, world!"
        assert data.id == "test-user-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, UserMessage)
        assert restored._content == "Hello, world!"
        assert restored.id == "test-user-1"

    def test_assistant_message_roundtrip(self):
        """Test AssistantMessage serialization and deserialization."""
        original = AssistantMessage("# Hello\n\nThis is **markdown**.", id="test-asst-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.ASSISTANT
        assert data.content == "# Hello\n\nThis is **markdown**."
        assert data.id == "test-asst-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, AssistantMessage)
        assert restored._content == "# Hello\n\nThis is **markdown**."
        assert restored.id == "test-asst-1"

    def test_tool_message_roundtrip(self):
        """Test ToolCallMessage serialization and deserialization."""
        original = ToolCallMessage(
            tool_name="read_file",
            args={"path": "/test/file.txt"},
            id="test-tool-1",
        )
        # Simulate tool completion
        original._status = "success"
        original._output = "File contents here"
        original._expanded = True

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.TOOL
        assert data.tool_name == "read_file"
        assert data.tool_args == {"path": "/test/file.txt"}
        assert data.tool_status == ToolStatus.SUCCESS
        assert data.tool_output == "File contents here"
        assert data.tool_expanded is True

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, ToolCallMessage)
        assert restored._tool_name == "read_file"
        assert restored._args == {"path": "/test/file.txt"}
        # Deferred state should be set
        assert restored._deferred_status == ToolStatus.SUCCESS
        assert restored._deferred_output == "File contents here"
        assert restored._deferred_expanded is True

    def test_error_message_roundtrip(self):
        """Test ErrorMessage serialization and deserialization."""
        original = ErrorMessage("Something went wrong!", id="test-error-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.ERROR
        assert data.content == "Something went wrong!"
        assert data.id == "test-error-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, ErrorMessage)
        assert restored._content == "Something went wrong!"
        assert restored.id == "test-error-1"

    def test_system_message_roundtrip(self):
        """Test SystemMessage serialization and deserialization."""
        original = SystemMessage("Session started", id="test-sys-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.SYSTEM
        assert data.content == "Session started"
        assert data.id == "test-sys-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, SystemMessage)
        assert restored._content == "Session started"
        assert restored.id == "test-sys-1"

    def test_message_data_defaults(self):
        """Test MessageData default values."""
        data = MessageData(type=MessageType.USER, content="test")

        assert data.id.startswith("msg-")
        assert data.timestamp > 0
        assert data.tool_name is None
        assert data.is_streaming is False
        assert data.height_hint is None


class TestMessageStore:
    """Tests for MessageStore window management."""

    def test_append_and_count(self):
        """Test appending messages and counting."""
        store = MessageStore()
        assert store.total_count == 0
        assert store.visible_count == 0

        store.append(MessageData(type=MessageType.USER, content="msg1"))
        assert store.total_count == 1
        assert store.visible_count == 1

        store.append(MessageData(type=MessageType.ASSISTANT, content="msg2"))
        assert store.total_count == 2
        assert store.visible_count == 2

    def test_window_exceeded(self):
        """Test window size detection."""
        store = MessageStore()
        store.WINDOW_SIZE = 5  # Small for testing

        for i in range(5):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        assert not store.window_exceeded()

        store.append(MessageData(type=MessageType.USER, content="msg5"))
        assert store.window_exceeded()

    def test_prune_messages(self):
        """Test pruning oldest messages."""
        store = MessageStore()
        store.WINDOW_SIZE = 5

        for i in range(7):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        assert store.visible_count == 7
        assert store.window_exceeded()

        # Get messages to prune
        to_prune = store.get_messages_to_prune()
        assert len(to_prune) == 2  # 7 - 5 = 2
        assert to_prune[0].id == "id-0"
        assert to_prune[1].id == "id-1"

        # Mark as pruned
        store.mark_pruned([msg.id for msg in to_prune])
        assert store.visible_count == 5
        assert store._visible_start == 2

    def test_active_message_not_pruned(self):
        """Test that active streaming message is never pruned."""
        store = MessageStore()
        store.WINDOW_SIZE = 3

        for i in range(5):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        # Set first message as active (streaming)
        store.set_active_message("id-0")

        to_prune = store.get_messages_to_prune()
        # Should skip id-0 since it's active
        pruned_ids = [msg.id for msg in to_prune]
        assert "id-0" not in pruned_ids
        assert "id-1" in pruned_ids

    def test_hydrate_messages(self):
        """Test hydrating messages above visible window."""
        store = MessageStore()
        store.HYDRATE_BUFFER = 3

        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        # Simulate having pruned first 5 messages
        store._visible_start = 5
        assert store.has_messages_above

        # Get messages to hydrate
        to_hydrate = store.get_messages_to_hydrate()
        assert len(to_hydrate) == 3  # HYDRATE_BUFFER
        assert to_hydrate[0].id == "id-2"
        assert to_hydrate[1].id == "id-3"
        assert to_hydrate[2].id == "id-4"

        # Mark as hydrated
        store.mark_hydrated(3)
        assert store._visible_start == 2

    def test_clear(self):
        """Test clearing the store."""
        store = MessageStore()

        for i in range(5):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        store.set_active_message("some-id")
        store._visible_start = 2

        store.clear()
        assert store.total_count == 0
        assert store.visible_count == 0
        assert store._active_message_id is None
        assert store._visible_start == 0
        assert store._visible_end == 0

    def test_get_message_by_id(self):
        """Test finding message by ID."""
        store = MessageStore()

        msg = MessageData(type=MessageType.USER, content="test", id="find-me")
        store.append(msg)
        store.append(MessageData(type=MessageType.USER, content="other"))

        found = store.get_message("find-me")
        assert found is not None
        assert found.content == "test"

        not_found = store.get_message("nonexistent")
        assert not_found is None

    def test_update_message(self):
        """Test updating message data."""
        store = MessageStore()

        store.append(MessageData(type=MessageType.USER, content="original", id="update-me"))

        result = store.update_message("update-me", content="updated")
        assert result is True

        msg = store.get_message("update-me")
        assert msg.content == "updated"

        # Update nonexistent
        result = store.update_message("nonexistent", content="fail")
        assert result is False

    def test_should_hydrate_above(self):
        """Test hydration trigger based on scroll position."""
        store = MessageStore()

        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        # No messages above - shouldn't hydrate
        assert not store.should_hydrate_above(scroll_position=0, viewport_height=100)

        # Simulate pruned messages
        store._visible_start = 5
        assert store.has_messages_above

        # Near top - should hydrate
        assert store.should_hydrate_above(scroll_position=50, viewport_height=100)

        # Far from top - shouldn't hydrate
        assert not store.should_hydrate_above(scroll_position=500, viewport_height=100)

    def test_visible_range(self):
        """Test getting visible range."""
        store = MessageStore()

        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        store._visible_start = 3
        store._visible_end = 8

        start, end = store.get_visible_range()
        assert start == 3
        assert end == 8

    def test_get_visible_messages(self):
        """Test getting visible message list."""
        store = MessageStore()

        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        store._visible_start = 3
        store._visible_end = 6

        visible = store.get_visible_messages()
        assert len(visible) == 3
        assert visible[0].id == "id-3"
        assert visible[1].id == "id-4"
        assert visible[2].id == "id-5"


class TestVirtualizationFlow:
    """Tests for the complete virtualization flow."""

    def test_full_prune_hydrate_cycle(self):
        """Test a complete cycle of adding, pruning, and hydrating messages."""
        store = MessageStore()
        store.WINDOW_SIZE = 5
        store.HYDRATE_BUFFER = 2

        # Add 10 messages
        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        # Initially all are visible
        assert store.total_count == 10
        assert store.visible_count == 10
        assert store._visible_start == 0
        assert store._visible_end == 10

        # Prune to window size
        to_prune = store.get_messages_to_prune()
        assert len(to_prune) == 5  # 10 - 5
        store.mark_pruned([msg.id for msg in to_prune])

        assert store.visible_count == 5
        assert store._visible_start == 5
        assert store.has_messages_above
        assert not store.has_messages_below

        # Hydrate 2 messages
        to_hydrate = store.get_messages_to_hydrate(2)
        assert len(to_hydrate) == 2
        assert to_hydrate[0].id == "id-3"
        assert to_hydrate[1].id == "id-4"

        store.mark_hydrated(2)
        assert store._visible_start == 3
        assert store.visible_count == 7

        # Hydrate more
        to_hydrate = store.get_messages_to_hydrate(10)  # Request more than available
        assert len(to_hydrate) == 3  # Only 3 left (id-0, id-1, id-2)
        store.mark_hydrated(3)

        assert store._visible_start == 0
        assert not store.has_messages_above

    def test_tool_message_state_preservation(self):
        """Test that tool message state is preserved through serialization."""
        # Create a tool message with various states
        original = ToolCallMessage(
            tool_name="bash",
            args={"command": "ls -la"},
            id="tool-1",
        )
        original._status = "success"
        original._output = "file1.txt\nfile2.txt\nfile3.txt"
        original._expanded = True

        # Serialize
        data = MessageData.from_widget(original)

        # Verify data
        assert data.tool_name == "bash"
        assert data.tool_args == {"command": "ls -la"}
        assert data.tool_status == ToolStatus.SUCCESS
        assert data.tool_output == "file1.txt\nfile2.txt\nfile3.txt"
        assert data.tool_expanded is True

        # Deserialize
        restored = data.to_widget()

        # Verify deferred state
        assert restored._deferred_status == ToolStatus.SUCCESS
        assert restored._deferred_output == "file1.txt\nfile2.txt\nfile3.txt"
        assert restored._deferred_expanded is True

    def test_streaming_message_protection(self):
        """Test that streaming (active) messages are never pruned."""
        store = MessageStore()
        store.WINDOW_SIZE = 3

        # Add messages
        for i in range(5):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}"))

        # Mark first message as active (simulating streaming)
        store.set_active_message("id-0")
        assert store.is_active("id-0")

        # Try to prune
        to_prune = store.get_messages_to_prune()

        # id-0 should not be in the prune list
        pruned_ids = [msg.id for msg in to_prune]
        assert "id-0" not in pruned_ids

        # But we should still prune id-1
        assert "id-1" in pruned_ids

        # Clear active and verify
        store.set_active_message(None)
        assert not store.is_active("id-0")

    def test_message_update_syncs_data(self):
        """Test that updating message data syncs properly."""
        store = MessageStore()

        # Add assistant message
        msg = MessageData(
            type=MessageType.ASSISTANT,
            content="Initial content",
            id="asst-1",
            is_streaming=True,
        )
        store.append(msg)

        # Update content (simulating streaming)
        store.update_message("asst-1", content="Updated content", is_streaming=False)

        # Verify update
        retrieved = store.get_message("asst-1")
        assert retrieved.content == "Updated content"
        assert retrieved.is_streaming is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
