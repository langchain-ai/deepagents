"""Tests for message store and serialization."""

import pytest
from textual.content import Content

from deepagents_code.tui.widgets.message_store import (
    DEFAULT_HEIGHT_HINT,
    MIN_HEIGHT_HINT,
    MessageData,
    MessageStore,
    MessageType,
)
from deepagents_code.tui.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    UserMessage,
)


class TestMessageData:
    """Tests for MessageData serialization."""

    def test_assistant_message_defaults_to_agent_output(self):
        """A plain assistant message is agent output, not client output."""
        data = MessageData.from_widget(AssistantMessage("hi", id="asst-plain"))

        assert data.assistant_local_only is False

    def test_error_message_content_body_roundtrip(self):
        """`Content` bodies serialize as plain text; link spans drop on resume."""
        from textual.style import Style as TStyle

        url = "https://example.com/docs"
        body = Content.assemble("see ", (url, TStyle(link=url)))
        original = ErrorMessage(body, id="test-error-content")

        data = MessageData.from_widget(original)
        assert data.type == MessageType.ERROR
        # `data.content` must be a plain `str` (not `Content`) for storage.
        assert isinstance(data.content, str)
        assert data.content == f"see {url}"

        restored = data.to_widget()
        assert isinstance(restored, ErrorMessage)
        # Restored widget renders without crashing (regression guard for the
        # `str(widget._content)` cast in `MessageData.from_widget`).
        assert restored.render().plain == f"Error: see {url}"

    def test_default_ids_use_full_uuid_hex(self):
        """Auto-generated IDs use the full 128-bit hex, not a truncated prefix.

        A wider ID keeps widget IDs unique across large histories and long
        sessions; a collision raises `DuplicateIds` when the widget mounts.
        """
        ids = {
            MessageData(type=MessageType.USER, content="test").id for _ in range(1000)
        }
        # 1000 distinct IDs, each a full uuid4 hex suffix.
        assert len(ids) == 1000
        for message_id in ids:
            assert message_id.startswith("msg-")
            suffix = message_id.removeprefix("msg-")
            assert len(suffix) == 32
            # A full uuid4 hex suffix is valid hexadecimal.
            int(suffix, 16)

    def test_from_widget_fallback_id_uses_full_uuid_hex(self):
        """`from_widget` synthesizes a full-hex ID when the widget has none.

        A widget mounted without an explicit ID must still get a collision-safe
        128-bit identifier, matching the `MessageData` default.
        """
        # Constructed without an `id` kwarg, so `widget.id` is None and the
        # fallback in `from_widget` must synthesize one.
        widget = UserMessage("no id")

        data = MessageData.from_widget(widget)

        assert data.id.startswith("msg-")
        suffix = data.id.removeprefix("msg-")
        assert len(suffix) == 32
        int(suffix, 16)


class TestMessageStore:
    """Tests for MessageStore window management."""

    def test_append_preserves_hidden_tail(self):
        """Appending while scrolled up should keep newer messages hidden."""
        store = MessageStore()
        for i in range(6):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 1
        store._visible_end = 3

        store.append(MessageData(type=MessageType.USER, content="new", id="id-new"))

        assert store.total_count == 7
        assert store.get_visible_range() == (1, 3)
        assert store.has_messages_below
        assert [msg.id for msg in store.get_messages_to_hydrate_below(10)] == [
            "id-3",
            "id-4",
            "id-5",
            "id-new",
        ]

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

    def test_should_hydrate_above_uses_top_spacer_boundary(self):
        """Hydration starts before the viewport reaches the mounted window."""
        store = MessageStore()
        for i in range(30):
            store.append(MessageData(type=MessageType.USER, content=f"msg{i}"))

        assert not store.should_hydrate_above(
            scroll_position=0,
            viewport_height=10,
            top_spacer_bottom=0,
        )

        store._visible_start = 20
        top_spacer_bottom = store.range_height(0, store._visible_start)
        assert store.should_hydrate_above(
            scroll_position=top_spacer_bottom + 70,
            viewport_height=10,
            top_spacer_bottom=top_spacer_bottom,
        )
        assert not store.should_hydrate_above(
            scroll_position=top_spacer_bottom + 90,
            viewport_height=10,
            top_spacer_bottom=top_spacer_bottom,
        )

    def test_should_hydrate_below_uses_bottom_spacer_top(self):
        """Hydration should start near mounted rows, not virtual transcript end."""
        store = MessageStore()
        for i in range(400):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 200
        store._visible_end = 300

        bottom_spacer_top = store.range_height(0, store.get_visible_range()[1])
        assert store.should_hydrate_below(
            scroll_position=bottom_spacer_top - 150,
            viewport_height=100,
            bottom_spacer_top=bottom_spacer_top,
        )
        assert not store.should_hydrate_below(
            scroll_position=bottom_spacer_top - 1000,
            viewport_height=100,
            bottom_spacer_top=bottom_spacer_top,
        )

    def test_should_hydrate_below_at_bottom_edge(self):
        """Reaching the scroll edge must hydrate even if the distance check misses.

        Estimated spacer heights can drift from the real DOM layout, leaving the
        viewport parked at `max_scroll` while the spacer-distance heuristic sits
        exactly at (or just past) its threshold. Without the edge guarantee the
        archived tail is stranded — the flaky-hydration failure this reproduces.
        """
        store = MessageStore()
        for i in range(100):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 16
        store._visible_end = 19

        bottom_spacer_top = store.range_height(0, store.get_visible_range()[1])
        # Geometry where distance_from_bottom_spacer == threshold, so the plain
        # heuristic returns False forever.
        scroll_position = 10.0
        viewport_height = 3
        assert not store.should_hydrate_below(
            scroll_position=scroll_position,
            viewport_height=viewport_height,
            bottom_spacer_top=bottom_spacer_top,
        )
        # Same geometry, but scrolled to the edge: hydration must run.
        assert store.should_hydrate_below(
            scroll_position=scroll_position,
            viewport_height=viewport_height,
            bottom_spacer_top=bottom_spacer_top,
            max_scroll=scroll_position,
        )

    def test_should_hydrate_below_at_edge_with_no_messages_below(self):
        """The bottom edge never hydrates when the window already ends the store."""
        store = MessageStore()
        for i in range(10):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 0
        store._visible_end = 10

        assert not store.has_messages_below
        assert not store.should_hydrate_below(
            scroll_position=5.0,
            viewport_height=3,
            bottom_spacer_top=store.range_height(0, 10),
            max_scroll=5.0,
        )


class TestVirtualizationFlow:
    """Tests for the complete virtualization flow."""

    def test_height_hints_drive_range_estimates(self):
        """Height hints should drive the range-height estimates spacers use."""
        store = MessageStore()
        for i in range(5):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        # Unmeasured messages fall back to DEFAULT_HEIGHT_HINT.
        assert store.range_height(0, 2) == 2 * DEFAULT_HEIGHT_HINT
        assert store.estimate_height(store._messages[0]) == DEFAULT_HEIGHT_HINT

        store.set_height_hint("id-0", 3)
        store.set_height_hint("id-1", 7)

        # id-0=3, id-1=7 measured; id-2 still the default → 3 + 7 + 5 == 15.
        assert store.range_height(0, 3) == 10 + DEFAULT_HEIGHT_HINT
        assert store.estimate_height(store._messages[0]) == 3

    def test_set_height_hint_clamps_and_update_message_rejects(self):
        """height_hint has a single clamping write path (set_height_hint)."""
        store = MessageStore()
        store.append(MessageData(type=MessageType.USER, content="msg", id="id-1"))

        # set_height_hint clamps to the floor rather than storing 0/negatives.
        assert store.set_height_hint("id-1", 0)
        clamped = store.get_message("id-1")
        assert clamped is not None
        assert clamped.height_hint == MIN_HEIGHT_HINT

        # The generic update path must not smuggle an unclamped height through.
        with pytest.raises(ValueError, match="height_hint"):
            store.update_message("id-1", height_hint=-4)

    def test_height_hints_scale_and_clear(self):
        """Width reflow can scale or clear cached height hints."""
        store = MessageStore()
        store.append(MessageData(type=MessageType.USER, content="msg", id="id-1"))
        store.set_height_hint("id-1", 10)

        store.invalidate_height_hints(scale=0.5)
        msg = store.get_message("id-1")
        assert msg is not None
        assert msg.height_hint == 5

        store.invalidate_height_hints()
        assert msg.height_hint is None

    def test_protected_messages_block_top_and_bottom_pruning(self):
        """Live messages should not be pruned from either edge."""
        store = MessageStore()
        store.WINDOW_SIZE = 3
        for i in range(6):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )

        # A protected message at the front blocks top pruning entirely.
        store.protect_message("id-0")
        assert store.get_messages_to_prune() == []
        store.unprotect_message("id-0")

        # Once released, the unprotected front messages prune normally.
        assert [m.id for m in store.get_messages_to_prune()] == ["id-0", "id-1", "id-2"]

        # A protected newest message blocks bottom pruning; releasing it lets
        # the newest rows prune.
        store.protect_message("id-5")
        assert store.get_messages_to_prune_below() == []
        store.unprotect_message("id-5")
        assert [m.id for m in store.get_messages_to_prune_below()] == [
            "id-3",
            "id-4",
            "id-5",
        ]

    def test_protection_reasons_are_independent(self):
        """Independent protection sources must not clobber each other."""
        store = MessageStore()
        store.append(MessageData(type=MessageType.USER, content="msg", id="id-1"))

        # Protect for two reasons: a live tool and the active stream.
        store.protect_message("id-1")  # default _LIVE_REASON
        store.set_active_message("id-1")  # _ACTIVE_REASON
        assert store.is_protected("id-1")

        # Releasing the live-tool reason must leave active protection intact.
        store.unprotect_message("id-1")
        assert store.is_protected("id-1")
        assert store.is_active("id-1")

        # Swapping the active message away releases only the active reason.
        store.set_active_message(None)
        assert not store.is_protected("id-1")
        assert not store.is_active("id-1")

    def test_active_swap_preserves_live_tool_protection(self):
        """Changing the active message must not unprotect a still-live tool."""
        store = MessageStore()
        for msg_id in ("tool-1", "asst-1", "asst-2"):
            store.append(MessageData(type=MessageType.USER, content=msg_id, id=msg_id))

        store.protect_message("tool-1")  # live tool
        store.set_active_message("asst-1")
        # A new streaming message takes over; the live tool stays protected.
        store.set_active_message("asst-2")
        assert store.is_protected("tool-1")
        assert not store.is_protected("asst-1")
        assert store.is_protected("asst-2")


class TestBulkLoad:
    """Tests for MessageStore.bulk_load."""

    @staticmethod
    def _rows(count: int) -> list[MessageData]:
        """Build `count` distinctly-identified rows for bulk_load."""
        return [
            MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            for i in range(count)
        ]

    def test_bulk_load_uses_bounded_initial_tail(self):
        """Resume mounts `INITIAL_WINDOW_SIZE` rows unless the window is smaller."""
        initial = MessageStore.INITIAL_WINDOW_SIZE
        data = self._rows(initial + 20)

        store = MessageStore()
        archived, visible = store.bulk_load(data)
        assert len(visible) == initial
        assert len(archived) == 20
        assert visible[0].id == "id-20"

        store = MessageStore()
        store.WINDOW_SIZE = initial - 18
        archived, visible = store.bulk_load(data)
        assert len(visible) == store.WINDOW_SIZE
        assert len(archived) == len(data) - store.WINDOW_SIZE
        assert visible[0].id == f"id-{len(archived)}"

    def test_bulk_load_just_over_initial_window_size(self):
        """One row above the boundary archives exactly one row."""
        store = MessageStore()
        archived, visible = store.bulk_load(
            self._rows(MessageStore.INITIAL_WINDOW_SIZE + 1)
        )
        assert len(archived) == 1
        assert len(visible) == MessageStore.INITIAL_WINDOW_SIZE
        assert visible[0].id == "id-1"


class TestMessageStoreIndex:
    """Tests for the _index dict that backs O(1) lookups."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
