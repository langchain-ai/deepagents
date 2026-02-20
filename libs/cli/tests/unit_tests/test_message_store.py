"""Tests for message store and serialization."""

import pytest
from textual.widgets import Static

from deepagents_cli.widgets.message_store import (
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    ErrorMessage,
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
        original = AssistantMessage(
            "# Hello\n\nThis is **markdown**.", id="test-asst-1"
        )

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

    def test_app_message_roundtrip(self):
        """Test AppMessage serialization and deserialization."""
        original = AppMessage("Session started", id="test-app-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.APP
        assert data.content == "Session started"
        assert data.id == "test-app-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, AppMessage)
        assert restored._content == "Session started"
        assert restored.id == "test-app-1"

    def test_diff_message_roundtrip(self):
        """Test DiffMessage serialization and deserialization."""
        diff_content = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"
        original = DiffMessage(diff_content, file_path="src/file.py", id="test-diff-1")

        # Serialize
        data = MessageData.from_widget(original)
        assert data.type == MessageType.DIFF
        assert data.content == diff_content
        assert data.diff_file_path == "src/file.py"
        assert data.id == "test-diff-1"

        # Deserialize
        restored = data.to_widget()
        assert isinstance(restored, DiffMessage)
        assert restored._diff_content == diff_content
        assert restored._file_path == "src/file.py"
        assert restored.id == "test-diff-1"

    def test_message_data_defaults(self):
        """Test MessageData default values."""
        data = MessageData(type=MessageType.USER, content="test")

        assert data.id.startswith("msg-")
        assert data.timestamp > 0
        assert data.tool_name is None
        assert data.is_streaming is False
        assert data.height_hint is None

    def test_tool_message_requires_tool_name(self):
        """Test that TOOL messages must have a tool_name."""
        with pytest.raises(ValueError, match="TOOL messages must have a tool_name"):
            MessageData(type=MessageType.TOOL, content="")

    def test_unknown_widget_serializes_as_app(self):
        """Test that unknown widget types fall back to APP MessageData."""
        unknown = Static("hello", id="unk-1")
        data = MessageData.from_widget(unknown)

        assert data.type == MessageType.APP
        assert "Unknown widget" in data.content
        assert data.id == "unk-1"


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
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

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

    def test_active_message_at_start_blocks_all_pruning(self):
        """Test that active message at window start prevents any pruning.

        When the active (streaming) message is the first visible message,
        `get_messages_to_prune` breaks immediately to keep the window
        contiguous — no messages can be pruned.
        """
        store = MessageStore()
        store.WINDOW_SIZE = 3

        for i in range(5):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

        # Set first message as active (streaming)
        store.set_active_message("id-0")

        to_prune = store.get_messages_to_prune()
        # Active at position 0 -> break immediately -> nothing pruned
        assert len(to_prune) == 0

    def test_active_message_in_middle_prunes_up_to_it(self):
        """Test that pruning stops at the active message to keep window contiguous.

        Messages before the active message are prunable, but the active
        message and everything after it are kept.
        """
        store = MessageStore()
        store.WINDOW_SIZE = 3

        for i in range(7):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

        # Set message in the middle as active
        store.set_active_message("id-2")

        to_prune = store.get_messages_to_prune()
        # Can prune id-0 and id-1, then break at id-2
        assert len(to_prune) == 2
        pruned_ids = [msg.id for msg in to_prune]
        assert pruned_ids == ["id-0", "id-1"]
        assert "id-2" not in pruned_ids

    def test_hydrate_messages(self):
        """Test hydrating messages above visible window."""
        store = MessageStore()
        store.HYDRATE_BUFFER = 3

        for i in range(10):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

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

        store.append(
            MessageData(type=MessageType.USER, content="original", id="update-me")
        )

        result = store.update_message("update-me", content="updated")
        assert result is True

        msg = store.get_message("update-me")
        assert msg is not None
        assert msg.content == "updated"

        # Update nonexistent
        result = store.update_message("nonexistent", content="fail")
        assert result is False

    def test_update_message_rejects_unknown_fields(self):
        """Test that updating protected or unknown fields raises ValueError."""
        store = MessageStore()
        store.append(
            MessageData(type=MessageType.USER, content="test", id="protected-1")
        )

        with pytest.raises(ValueError, match="Cannot update unknown or protected"):
            store.update_message("protected-1", id="new-id")

        with pytest.raises(ValueError, match="Cannot update unknown or protected"):
            store.update_message("protected-1", type=MessageType.ERROR)

        with pytest.raises(ValueError, match="Cannot update unknown or protected"):
            store.update_message("protected-1", nonexistent_field="value")

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

    def test_should_prune_below(self):
        """Test prune-below trigger based on scroll position and distance."""
        store = MessageStore()
        store.WINDOW_SIZE = 5

        for i in range(10):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        # Within window size -> no pruning needed
        store2 = MessageStore()
        store2.WINDOW_SIZE = 20
        for i in range(10):
            store2.append(MessageData(type=MessageType.USER, content=f"msg{i}"))
        assert not store2.should_prune_below(
            scroll_position=0, viewport_height=100, content_height=1000
        )

        # Exceeds window, user far from bottom -> should prune
        assert store.should_prune_below(
            scroll_position=0, viewport_height=100, content_height=1000
        )

        # Exceeds window, user near bottom -> should not prune
        assert not store.should_prune_below(
            scroll_position=800, viewport_height=100, content_height=1000
        )

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
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

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
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

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
        assert isinstance(restored, ToolCallMessage)

        # Verify deferred state
        assert restored._deferred_status == ToolStatus.SUCCESS
        assert restored._deferred_output == "file1.txt\nfile2.txt\nfile3.txt"
        assert restored._deferred_expanded is True

    def test_streaming_message_protection(self):
        """Test that streaming (active) messages are never pruned.

        With break-at-active behavior, when the active message is at position
        0, no messages can be pruned at all.
        """
        store = MessageStore()
        store.WINDOW_SIZE = 3

        # Add messages
        for i in range(5):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

        # Mark first message as active (simulating streaming)
        store.set_active_message("id-0")
        assert store.is_active("id-0")

        # Try to prune — active at start means nothing can be pruned
        to_prune = store.get_messages_to_prune()
        assert len(to_prune) == 0

        # Clear active and verify
        store.set_active_message(None)
        assert not store.is_active("id-0")

        # Now pruning should work normally
        to_prune = store.get_messages_to_prune()
        assert len(to_prune) == 2  # 5 - 3 = 2
        assert to_prune[0].id == "id-0"
        assert to_prune[1].id == "id-1"

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
        assert retrieved is not None
        assert retrieved.content == "Updated content"
        assert retrieved.is_streaming is False


class TestBulkLoad:
    """Tests for MessageStore.bulk_load."""

    def test_bulk_load_under_window_size(self):
        """All messages should be visible when count <= WINDOW_SIZE."""
        store = MessageStore()
        store.WINDOW_SIZE = 50

        data = [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(10)
        ]
        archived, visible = store.bulk_load(data)

        assert len(archived) == 0
        assert len(visible) == 10
        assert store.total_count == 10
        assert store.visible_count == 10
        assert store._visible_start == 0
        assert store._visible_end == 10

    def test_bulk_load_over_window_size(self):
        """Only the tail WINDOW_SIZE messages should be visible."""
        store = MessageStore()
        store.WINDOW_SIZE = 5

        data = [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(20)
        ]
        archived, visible = store.bulk_load(data)

        assert len(archived) == 15
        assert len(visible) == 5
        assert store.total_count == 20
        assert store.visible_count == 5
        assert store._visible_start == 15
        assert store._visible_end == 20
        assert visible[0].id == "id-15"
        assert visible[-1].id == "id-19"

    def test_bulk_load_exact_window_size(self):
        """Edge case: count == WINDOW_SIZE means all visible, none archived."""
        store = MessageStore()
        store.WINDOW_SIZE = 10

        data = [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(10)
        ]
        archived, visible = store.bulk_load(data)

        assert len(archived) == 0
        assert len(visible) == 10
        assert store._visible_start == 0
        assert store._visible_end == 10

    def test_bulk_load_then_hydrate(self):
        """Archived messages should be accessible via get_messages_to_hydrate."""
        store = MessageStore()
        store.WINDOW_SIZE = 5
        store.HYDRATE_BUFFER = 3

        data = [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(20)
        ]
        store.bulk_load(data)

        assert store.has_messages_above
        to_hydrate = store.get_messages_to_hydrate()
        assert len(to_hydrate) == 3
        assert to_hydrate[0].id == "id-12"
        assert to_hydrate[1].id == "id-13"
        assert to_hydrate[2].id == "id-14"

    def test_bulk_load_empty(self):
        """Bulk loading an empty list should be a no-op."""
        store = MessageStore()
        archived, visible = store.bulk_load([])

        assert len(archived) == 0
        assert len(visible) == 0
        assert store.total_count == 0

    def test_bulk_load_preserves_existing_messages(self):
        """Bulk load should extend, not replace, existing messages."""
        store = MessageStore()
        store.WINDOW_SIZE = 5

        store.append(MessageData(type=MessageType.USER, content="pre", id="pre-0"))
        data = [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(6)
        ]
        archived, _visible = store.bulk_load(data)

        assert store.total_count == 7
        assert store.visible_count == 5
        assert store._visible_start == 2
        assert archived[0].id == "pre-0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
