"""Tests for message store and serialization."""

import logging

import pytest
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Static

from deepagents_code.diff_utils import DiffStats
from deepagents_code.tui.widgets import diff as diff_module
from deepagents_code.tui.widgets.message_store import (
    DEFAULT_HEIGHT_HINT,
    MIN_HEIGHT_HINT,
    MessageData,
    MessageStore,
    MessageType,
    ToolStatus,
)
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    ErrorMessage,
    LazyToolGroupSummary,
    ReasoningMessage,
    RubricResultMessage,
    SkillMessage,
    SummarizationMessage,
    ToolCallMessage,
    UserMessage,
)


def _rendered_text(widget: Widget) -> str:
    """Return a composed child's plain text, ignoring styles."""
    rendered = widget.render()
    return rendered.plain if isinstance(rendered, Content) else str(rendered)


class TestMessageData:
    """Tests for MessageData serialization."""

    def test_user_message_roundtrip_preserves_expansion(self):
        """An expanded long prompt stays expanded across virtualization."""
        original = UserMessage("A" * 12_000, id="test-user-long")
        original._expanded = True

        data = MessageData.from_widget(original)
        assert data.user_expanded is True

        restored = data.to_widget()
        assert isinstance(restored, UserMessage)
        assert restored._deferred_expanded is True

    def test_user_message_roundtrip_preserves_detect_mode(self):
        """`detect_mode=False` survives, so a literal leading slash stays literal."""
        original = UserMessage("/not/a/command", id="test-user-path", detect_mode=False)

        data = MessageData.from_widget(original)
        assert data.user_detect_mode is False

        restored = data.to_widget()
        assert isinstance(restored, UserMessage)
        assert restored._detect_mode is False

    def test_user_message_roundtrip_defaults_to_collapsed(self):
        """A prompt the user never expanded rehydrates collapsed."""
        original = UserMessage("B" * 12_000, id="test-user-collapsed")

        restored = MessageData.from_widget(original).to_widget()
        assert isinstance(restored, UserMessage)
        assert restored._deferred_expanded is False

    def test_assistant_message_defaults_to_agent_output(self):
        """A plain assistant message is agent output, not client output."""
        data = MessageData.from_widget(AssistantMessage("hi", id="asst-plain"))

        assert data.assistant_local_only is False

    def test_local_only_assistant_message_roundtrip(self):
        """`local_only` survives serialization and rehydration.

        `!` shell output renders through `AssistantMessage`, and callers asking
        whether the agent did anything in a thread rely on this flag. Losing it
        on a virtualization round trip would make shell output read as a turn.
        """
        original = AssistantMessage(
            "```text\nREADME.md\n```", id="asst-shell-1", local_only=True
        )

        data = MessageData.from_widget(original)
        assert data.type == MessageType.ASSISTANT
        assert data.assistant_local_only is True

        restored = data.to_widget()
        assert isinstance(restored, AssistantMessage)
        assert restored._local_only is True
        # A second round trip must not lose the flag either.
        assert MessageData.from_widget(restored).assistant_local_only is True

    async def test_lazy_tool_group_mounts_details_only_when_expanded(self) -> None:
        """Collapsed restored groups keep their tool widget trees out of the DOM."""
        data = MessageData(
            type=MessageType.TOOL_GROUP,
            content="",
            tool_group_messages=[
                MessageData(
                    type=MessageType.TOOL,
                    content="",
                    tool_name="read_file",
                    tool_status=ToolStatus.SUCCESS,
                    tool_output="contents",
                ),
                MessageData(
                    type=MessageType.TOOL,
                    content="",
                    tool_name="grep",
                    tool_status=ToolStatus.SUCCESS,
                    tool_output="match",
                ),
            ],
        )
        restored = data.to_widget()
        assert isinstance(restored, LazyToolGroupSummary)

        class _App(App[None]):
            def compose(self) -> ComposeResult:
                yield restored

        async with _App().run_test():
            assert not restored.query(ToolCallMessage)
            await restored._set_expanded(True)
            assert len(restored.query(ToolCallMessage)) == 2
            await restored._set_expanded(False)
            assert not restored.query(ToolCallMessage)

    def test_rejected_supersession_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The guard protects an invariant, so tripping it must leave a trace.

        Two name sources decide one outcome — the adapter gates on
        `record.tool_name`, the widget on its own. A divergence mounts an
        empty-bodied diff reading "no changes" *beside* a row that stayed visible,
        and a silent return leaves nothing to debug that from.
        """
        widget = ToolCallMessage("shell")

        with caplog.at_level(logging.WARNING):
            widget.mark_superseded_by_diff()

        assert widget._diff_superseded is False
        assert any(
            "may be superseded" in record.getMessage() for record in caplog.records
        ), caplog.text

    def test_error_message_content_body_roundtrip(self):
        """`Content` bodies serialize as plain text; link spans drop on resume."""
        from textual.content import Content
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

    def test_app_message_markdown_roundtrip(self):
        """Markdown AppMessages must survive dehydrate/rehydrate with their flag.

        Regression guard: dropping `is_markdown` from either `from_widget`
        or `to_widget` would silently downgrade rehydrated `/version` extras
        tables to plain-text rendering.
        """
        from textual.content import Content

        markdown_source = (
            "### Installed optional dependencies\n"
            "\n"
            "| Extra | Package | Version |\n"
            "| --- | --- | --- |\n"
            "| anthropic | langchain-anthropic | 1.4.0 |\n"
        )
        original = AppMessage(markdown_source, markdown=True, id="test-app-md-1")

        data = MessageData.from_widget(original)
        assert data.type == MessageType.APP
        assert data.content == markdown_source
        assert data.is_markdown is True

        restored = data.to_widget()
        assert isinstance(restored, AppMessage)
        assert restored._is_markdown is True
        # Markdown renders to selectable `Content` (not a raw Rich renderable)
        # so the rehydrated extras table stays copyable.
        rendered = restored.render()
        assert isinstance(rendered, Content)
        assert "langchain-anthropic" in rendered.plain

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

    def test_diff_message_roundtrip_preserves_highlighting_inputs(self) -> None:
        """Virtualized diffs retain only the needed lexer prefixes and true counts.

        `shown` with real counts, because `untrusted_before` leaves `stats`
        unset — `FileOperationRecord.diff_stats` documents that pairing as
        impossible, and a test that builds it stops describing what the code
        produces.
        """
        original = DiffMessage(
            "@@ -1 +1 @@\n-a\n+b",
            "example.py",
            tool_name="edit_file",
            before="a\nunused before\n",
            after="b\nunused after\n",
            stats=DiffStats(additions=200, deletions=200),
            id="test-diff-highlight",
        )

        restored = MessageData.from_widget(original).to_widget()

        assert isinstance(restored, DiffMessage)
        assert (restored._before, restored._after) == ("a", "b")
        assert restored._stats == DiffStats(additions=200, deletions=200)
        assert restored._outcome == "shown"
        # Not only the privates: a rehydration bug preserving all four while
        # breaking composition would pass on the assertions above alone.
        assert any("+200" in _rendered_text(child) for child in restored.compose())

    def test_an_untrusted_diff_roundtrips_without_counts(self) -> None:
        """The outcome and its suppressed body have to survive separately.

        Split from the highlighting round-trip above so each asserts a state the
        tracker can actually produce: this one carries no `stats`, because a
        count taken against a stand-in pre-image would be fiction.
        """
        original = DiffMessage(
            "@@ -1 +1 @@\n-a\n+b",
            "example.py",
            tool_name="edit_file",
            before="a\n",
            after="b\n",
            outcome="untrusted_before",
            id="test-diff-untrusted",
        )

        restored = MessageData.from_widget(original).to_widget()

        assert isinstance(restored, DiffMessage)
        assert restored._outcome == "untrusted_before"
        assert restored._stats is None
        assert any(
            "prior contents could not be read" in _rendered_text(child)
            for child in restored.compose()
        )

    def test_a_suppressed_caveat_stays_suppressed_after_rehydration(self) -> None:
        """The decision depends on what else was mounted, which the store cannot see.

        Without persisting it, a diff whose tool row carries the caveat comes
        back printing the same sentence a second time.
        """
        original = DiffMessage(
            "@@ -1 +1 @@\n-a\n+b",
            "example.py",
            tool_name="edit_file",
            outcome="untrusted_before",
            show_caveat=False,
            id="test-diff-no-caveat",
        )

        restored = MessageData.from_widget(original).to_widget()

        assert isinstance(restored, DiffMessage)
        assert restored.renders_caveat is False
        texts = [_rendered_text(child) for child in restored.compose()]
        assert all("prior contents could not be read" not in text for text in texts)
        # The body stays suppressed regardless — only the sentence was hidden.
        assert all("+b" not in text for text in texts)


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

    def test_hydrate_below_advances_visible_end(self):
        """Hydrating below should mount the next block and advance the tail."""
        store = MessageStore()
        store.WINDOW_SIZE = 3
        for i in range(10):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 0
        store._visible_end = 3

        to_hydrate = store.get_messages_to_hydrate_below(2)
        assert [m.id for m in to_hydrate] == ["id-3", "id-4"]

        store.mark_hydrated_below(len(to_hydrate))
        assert store.get_visible_range() == (0, 5)

        # Nothing left below once the tail is reached.
        store.mark_hydrated_below(store.total_count)
        assert store.get_visible_range() == (0, 10)
        assert not store.has_messages_below
        assert store.get_messages_to_hydrate_below() == []

    def test_prune_below_returns_newest_and_marks_visible_end(self):
        """Bottom pruning removes the newest rows and rewinds _visible_end."""
        store = MessageStore()
        store.WINDOW_SIZE = 3
        for i in range(6):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 0
        store._visible_end = 6

        to_prune = store.get_messages_to_prune_below()  # back to WINDOW_SIZE
        assert [m.id for m in to_prune] == ["id-3", "id-4", "id-5"]

        store.mark_pruned_below([m.id for m in to_prune])
        assert store.get_visible_range() == (0, 3)
        assert store.has_messages_below

    def test_mark_pruned_below_only_rewinds_contiguous_tail(self):
        """A gap at the tail must not over-rewind _visible_end."""
        store = MessageStore()
        for i in range(6):
            store.append(
                MessageData(type=MessageType.USER, content=f"msg{i}", id=f"id-{i}")
            )
        store._visible_start = 0
        store._visible_end = 6

        # The newest row (id-5) was NOT removed from the DOM; only inner rows
        # were. mark_pruned_below must stop at the still-mounted tail.
        store.mark_pruned_below(["id-3", "id-4"])
        assert store.get_visible_range() == (0, 6)


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

    def test_bulk_load_at_initial_window_size(self):
        """Exactly `INITIAL_WINDOW_SIZE` rows should all mount, none archived.

        The boundary belongs to the `INITIAL_WINDOW_SIZE` arm of the `min()`,
        which the other bulk_load tests never reach: they shrink `WINDOW_SIZE`
        below it, so `WINDOW_SIZE` wins there.
        """
        store = MessageStore()
        archived, visible = store.bulk_load(
            self._rows(MessageStore.INITIAL_WINDOW_SIZE)
        )
        assert archived == []
        assert len(visible) == MessageStore.INITIAL_WINDOW_SIZE
        assert store._visible_start == 0

    def test_bulk_load_just_under_initial_window_size(self):
        """One row below the boundary stays fully mounted."""
        store = MessageStore()
        archived, visible = store.bulk_load(
            self._rows(MessageStore.INITIAL_WINDOW_SIZE - 1)
        )
        assert archived == []
        assert len(visible) == MessageStore.INITIAL_WINDOW_SIZE - 1
        assert store._visible_start == 0

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


def test_display_caveat_survives_the_store_roundtrip() -> None:
    """A rehydrated caveated row must still refuse to fold.

    The flag cannot be re-derived from `tool_output` without matching the
    caveat's prose, so it is persisted. Losing it means a scrolled-away write
    whose contents could not be read comes back folded into a summary that says
    only `▸ Wrote 1 file`.
    """
    tool = ToolCallMessage("write_file", {"file_path": "a.py"})
    tool.set_success("could not be shown\n\nWrote file")
    tool._mark_display_caveat()

    restored = MessageData.from_widget(tool).to_widget()

    assert isinstance(restored, ToolCallMessage)
    assert restored.has_display_caveat is True


def test_reasoning_roundtrip_preserves_content_and_expansion() -> None:
    widget = ReasoningMessage("[bold]plain[/bold]")
    widget._expanded = False

    restored = MessageData.from_widget(widget).to_widget()

    assert isinstance(restored, ReasoningMessage)
    assert restored._content == "[bold]plain[/bold]"
    assert restored._deferred_expanded is False
    assert restored._streaming is False
