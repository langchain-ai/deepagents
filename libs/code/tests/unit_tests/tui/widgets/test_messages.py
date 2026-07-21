"""Unit tests for message widgets markup safety."""

import asyncio
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.utils import (
    MAX_LINE_LENGTH,
    format_content_with_line_numbers,
)
from rich.style import Style
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.content import Content
from textual.widgets import Markdown, Static

from deepagents_code import theme
from deepagents_code.formatting import format_duration
from deepagents_code.input import INPUT_HIGHLIGHT_PATTERN
from deepagents_code.tool_display import (
    EXECUTE_HEADER_MAX_LENGTH,
    JS_EVAL_HEADER_MAX_LENGTH,
)
from deepagents_code.tui.widgets.message_store import MessageData
from deepagents_code.tui.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    ErrorMessage,
    QueuedUserMessage,
    RubricResultMessage,
    SkillMessage,
    SummarizationMessage,
    ToolCallMessage,
    UserMessage,
    _MutedRichMarkdown,
    _strip_frontmatter,
    _strip_prompt_prefix,
    _strip_success_exit_line,
)

# Content that previously caused MarkupError crashes
MARKUP_INJECTION_CASES = [
    "[foo] bar [baz]",
    "}, [/* deps */]);",
    "array[0] = value[1]",
    "[bold]not markup[/bold]",
    "[/dim]",
    "const x = arr[i];",
    "[unclosed bracket",
    "nested [[brackets]]",
]


class TestUserMessageMarkupSafety:
    """Test UserMessage handles content with brackets safely."""

    @pytest.mark.parametrize("content", MARKUP_INJECTION_CASES)
    def test_user_message_no_markup_error(self, content: str) -> None:
        """UserMessage should not raise MarkupError on bracket content."""
        msg = UserMessage(content)
        assert msg._content == content

    def test_user_message_preserves_content_exactly(self) -> None:
        """UserMessage should preserve user content without modification."""
        content = "[bold]test[/bold] with [brackets]"
        msg = UserMessage(content)
        assert msg._content == content


class TestErrorMessageMarkupSafety:
    """Test ErrorMessage handles content with brackets safely."""

    @pytest.mark.parametrize("content", MARKUP_INJECTION_CASES)
    def test_error_message_no_markup_error(self, content: str) -> None:
        """ErrorMessage should not raise MarkupError on bracket content."""
        # Instantiation should not raise - this is the key test
        ErrorMessage(content)

    def test_error_message_instantiates(self) -> None:
        """ErrorMessage should instantiate with bracket content."""
        error = "Failed: array[0] is undefined"
        msg = ErrorMessage(error)
        assert msg is not None

    def test_error_message_has_prefix_and_body(self) -> None:
        """ErrorMessage content should have `'Error: '` prefix followed by the body."""
        msg = ErrorMessage("something broke")
        rendered = msg.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "Error: something broke"

    def test_error_message_accepts_content_with_link_span(self) -> None:
        """Pre-built `Content` with `link` spans passes through to render output."""
        from textual.style import Style as TStyle

        url = "https://docs.langchain.com/oss/python/deepagents/code/providers"
        body = Content.assemble(
            "see ",
            (url, TStyle(underline=True, link=url)),
        )
        rendered = ErrorMessage(body).render()
        assert isinstance(rendered, Content)
        links = [
            getattr(span.style, "link", None)
            for span in rendered.spans
            if getattr(span.style, "link", None)
        ]
        assert links == [url]
        assert rendered.plain == f"Error: see {url}"

    def test_error_message_click_on_link_opens_url(self) -> None:
        """Click on a `link`-styled span should route through `open_style_link`."""
        from types import SimpleNamespace

        msg = ErrorMessage("see https://example.com")
        event = SimpleNamespace(
            style=SimpleNamespace(link="https://example.com"),
            app=SimpleNamespace(notify=MagicMock()),
            stop=MagicMock(),
        )
        with patch(
            "deepagents_code.tui.widgets.messages.open_style_link"
        ) as mock_open_link:
            msg.on_click(event)  # ty: ignore

        mock_open_link.assert_called_once_with(event)

    def test_error_message_click_off_link_no_ops(self) -> None:
        """Click outside a link span should not perform timestamp side effects."""
        from types import SimpleNamespace

        msg = ErrorMessage("plain error, no URL")
        event = SimpleNamespace(
            style=SimpleNamespace(link=None),
            app=SimpleNamespace(notify=MagicMock()),
            stop=MagicMock(),
        )
        with patch(
            "deepagents_code.tui.widgets.messages.open_style_link"
        ) as mock_open_link:
            msg.on_click(event)  # ty: ignore

        mock_open_link.assert_not_called()


class TestAppMessageMarkupSafety:
    """Test AppMessage handles content with brackets safely."""

    @pytest.mark.parametrize("content", MARKUP_INJECTION_CASES)
    def test_app_message_no_markup_error(self, content: str) -> None:
        """AppMessage should not raise MarkupError on bracket content."""
        # Instantiation should not raise - this is the key test
        AppMessage(content)

    def test_app_message_instantiates(self) -> None:
        """AppMessage should instantiate with bracket content."""
        content = "Status: processing items[0-10]"
        msg = AppMessage(content)
        assert msg is not None

    def test_app_message_str_gets_dim_italic(self) -> None:
        """String input should be rendered as dim italic `Content`."""
        msg = AppMessage("hello")
        rendered = msg._Static__content  # ty: ignore
        assert isinstance(rendered, Content)
        assert rendered.plain == "hello"

    def test_app_message_content_passthrough(self) -> None:
        """Pre-styled `Content` should pass through unchanged."""
        pre = Content.styled("styled", "bold cyan")
        msg = AppMessage(pre)
        rendered = msg._Static__content  # ty: ignore
        assert rendered is pre

    def test_app_message_markdown_renders_selectable_content(self) -> None:
        """`markdown=True` should render selectable `Content`, not a `RichVisual`.

        Textual text-selection only works over `Content`/`Text` visuals, so
        markdown must resolve to `Content` for its text to be copyable.
        """
        msg = AppMessage("### heading", markdown=True)
        rendered = msg.render()
        assert isinstance(rendered, Content)
        assert "heading" in rendered.plain

    def test_app_message_markdown_requires_string(self) -> None:
        """`markdown=True` with non-string input should raise `TypeError`."""
        pre = Content.styled("styled", "bold")
        with pytest.raises(TypeError):
            AppMessage(pre, markdown=True)


class TestMutedRichMarkdown:
    """Tests for the muted markdown theme wrapper."""

    _DOC = (
        "### Installed optional dependencies\n"
        "\n"
        "| Extra | Package | Version |\n"
        "| --- | --- | --- |\n"
        "| anthropic | langchain-anthropic | 1.4.1 |\n"
    )

    @staticmethod
    def _render(renderable: object) -> str:
        import io

        from rich.console import Console

        console = Console(
            file=io.StringIO(),
            force_terminal=True,
            color_system="truecolor",
            width=80,
            legacy_windows=False,
        )
        console.print(renderable)
        return console.file.getvalue()  # ty: ignore

    def test_strips_heading_and_table_colors(self) -> None:
        """Muted wrapper should drop magenta/cyan from headings and tables."""
        muted = self._render(_MutedRichMarkdown(self._DOC))

        # Some Rich versions paint headings/tables magenta/cyan by default.
        # The wrapper should not emit those hues regardless of Rich's baseline.
        assert "\x1b[35m" not in muted
        assert ";35m" not in muted
        assert "\x1b[36m" not in muted
        assert ";36m" not in muted

    def test_applies_dim_to_body_and_headings(self) -> None:
        """Muted wrapper should layer `dim` onto body, headings, and tables."""
        muted = self._render(_MutedRichMarkdown(self._DOC))

        # `dim` is ANSI code 2. Heading should be bold+dim ("1;2"),
        # plain cells should be dim ("2m"), and both must be present.
        assert "\x1b[1;2m" in muted
        assert "\x1b[2m" in muted

    def test_render_failure_falls_back_to_plain_source(self) -> None:
        """A crash inside Rich markdown rendering must not escape.

        If the themed render path raises, the wrapper should emit the raw
        source so the chat view stays up; the full stream would otherwise
        tear down when Textual asks the widget for content.
        """
        wrapped = _MutedRichMarkdown("# heading\n\nbody")
        # Force the inner Markdown renderable to raise when consumed.
        wrapped._markdown = MagicMock()
        wrapped._markdown.__rich_console__ = MagicMock(side_effect=RuntimeError("boom"))

        rendered = self._render(wrapped)
        assert "body" in rendered


class TestAssistantMessageMarkdownRendering:
    """Tests for assistant markdown render lifecycle."""

    async def test_write_initial_content_uses_full_markdown_update(self) -> None:
        """Preloaded assistant messages should not keep stream state alive."""
        msg = AssistantMessage("```python\nprint('hello')\n```")
        markdown = MagicMock()
        markdown.update = AsyncMock()
        msg._markdown = markdown

        await msg.write_initial_content()

        markdown.update.assert_awaited_once_with("```python\nprint('hello')\n```")
        assert msg._stream is None

    async def test_stop_stream_rerenders_complete_markdown(self) -> None:
        """Completed streams should get a full parse after incremental updates."""
        msg = AssistantMessage()
        markdown = MagicMock()
        markdown.update = AsyncMock()
        stream = MagicMock()
        stream.stop = AsyncMock()
        msg._markdown = markdown
        msg._stream = stream
        msg._content = "```python\nprint('wrapped text')\n```"

        await msg.stop_stream()

        stream.stop.assert_awaited_once_with()
        markdown.update.assert_awaited_once_with(
            "```python\nprint('wrapped text')\n```"
        )
        assert msg._stream is None

    async def test_set_content_replaces_stream_with_single_update(self) -> None:
        """Replacing content should cancel the stream and update exactly once."""
        msg = AssistantMessage()
        markdown = MagicMock()
        markdown.update = AsyncMock()
        stream = MagicMock()
        stream.stop = AsyncMock()
        msg._markdown = markdown
        msg._stream = stream
        msg._content = "old streamed content"

        await msg.set_content("```python\nnew content\n```")

        stream.stop.assert_awaited_once_with()
        markdown.update.assert_awaited_once_with("```python\nnew content\n```")
        assert msg._stream is None
        assert msg._content == "```python\nnew content\n```"


class _AssistantMessageApp(App[None]):
    """Minimal app that mounts an AssistantMessage for runtime tests."""

    def compose(self) -> ComposeResult:
        widget = AssistantMessage()
        widget.id = "assistant"
        yield widget


class TestAssistantMessageLinkPointer:
    """Tests for the pointer cursor shown when hovering markdown links."""

    @staticmethod
    def _move_event(
        *, link: str | None = None, meta: dict | None = None
    ) -> SimpleNamespace:
        """Build a minimal mouse-move-like event exposing the hovered style.

        The handlers only read `event.style.link` and `event.style.meta`, so a
        namespace is enough; the assertions run against the real `Markdown`
        widget mounted by `_AssistantMessageApp`.
        """
        return SimpleNamespace(style=SimpleNamespace(link=link, meta=meta or {}))

    async def test_hovering_markdown_link_sets_pointer_cursor(self) -> None:
        """A markdown `@click=link(...)` span switches the real pointer to pointer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)

            msg.on_mouse_move(self._move_event(meta={"@click": "link('https://x')"}))  # ty: ignore

            assert msg._markdown is not None
            assert msg._markdown.styles.pointer == "pointer"

    async def test_hovering_osc8_link_sets_pointer_cursor(self) -> None:
        """An OSC 8 `Style(link=...)` span also switches the pointer to pointer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)

            msg.on_mouse_move(self._move_event(link="https://example.com"))  # ty: ignore

            assert msg._markdown is not None
            assert msg._markdown.styles.pointer == "pointer"

    async def test_hovering_text_sets_text_pointer(self) -> None:
        """Plain markdown text keeps the text pointer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)

            msg.on_mouse_move(self._move_event())  # ty: ignore

            assert msg._markdown is not None
            assert msg._markdown.styles.pointer == "text"

    async def test_leave_resets_pointer(self) -> None:
        """Leaving the message resets the pointer to text after a link hover."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            msg.on_mouse_move(self._move_event(link="https://example.com"))  # ty: ignore

            msg.on_leave()

            assert msg._markdown is not None
            assert msg._markdown.styles.pointer == "text"

    async def test_markdown_open_links_is_disabled(self) -> None:
        """The app handles Markdown links so it can show URL-opened toasts."""
        async with _AssistantMessageApp().run_test() as pilot:
            markdown = pilot.app.query_one("#assistant-content", Markdown)

            assert markdown._open_links is False

    async def test_markdown_link_clicked_uses_checked_toast_helper(self) -> None:
        """Clicked Markdown links should use the checked browser/toast helper."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            event = SimpleNamespace(href="https://example.com/docs", stop=MagicMock())

            with patch(
                "deepagents_code.tui.widgets.messages.open_checked_url_async",
                new=AsyncMock(return_value=True),
            ) as mock_open:
                await msg.on_markdown_link_clicked(event)  # ty: ignore

            event.stop.assert_called_once()
            mock_open.assert_awaited_once_with(
                "https://example.com/docs",
                app=pilot.app,
                notify_on_success=True,
            )

    async def test_markdown_link_clicked_blocks_suspicious_url(self) -> None:
        """Markdown links should apply the same URL safety check as style links."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            event = SimpleNamespace(
                href="https://example.com/\u200b[admin]",
                stop=MagicMock(),
            )

            with (
                patch.object(pilot.app, "notify") as notify,
                patch(
                    "deepagents_code.tui.widgets._links.webbrowser.open"
                ) as mock_open,
            ):
                await msg.on_markdown_link_clicked(event)  # ty: ignore

            event.stop.assert_called_once()
            mock_open.assert_not_called()
            notify.assert_called_once()
            args, kwargs = notify.call_args
            assert "Blocked suspicious URL" in args[0]
            assert "https://example.com/[admin]" in args[0]
            assert kwargs["severity"] == "warning"
            assert kwargs["markup"] is False

    def test_mouse_move_before_mount_is_noop(self) -> None:
        """Hovering before mount (no markdown widget yet) must not raise."""
        msg = AssistantMessage()
        assert msg._markdown is None

        msg.on_mouse_move(self._move_event(link="https://example.com"))  # ty: ignore

    def test_leave_before_mount_is_noop(self) -> None:
        """Leaving before mount (no markdown widget yet) must not raise."""
        msg = AssistantMessage()
        assert msg._markdown is None

        msg.on_leave()


class TestAssistantMessageStreamCoalescing:
    """Tests for the throttled streaming flush that keeps input responsive."""

    async def test_append_buffers_until_flush(self) -> None:
        """Tokens accumulate in `_content` but defer the markdown write."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            stream = MagicMock()
            stream.write = AsyncMock()
            msg._stream = stream

            await msg.append_content("hello ")
            await msg.append_content("world")

            # No immediate write — tokens are buffered for the timer.
            stream.write.assert_not_awaited()
            assert msg._content == "hello world"
            assert msg._pending_append == "hello world"
            assert msg._flush_timer is not None

    async def test_timer_flushes_coalesced_text_once(self) -> None:
        """The throttled timer writes buffered tokens as a single fragment."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            stream = MagicMock()
            stream.write = AsyncMock()
            msg._stream = stream

            await msg.append_content("foo")
            await msg.append_content("bar")
            await asyncio.sleep(msg._STREAM_FLUSH_INTERVAL * 2)
            await pilot.pause()

            stream.write.assert_awaited_once_with("foobar")
            assert msg._pending_append == ""

    async def test_stop_stream_flushes_and_cancels_timer(self) -> None:
        """Stopping the stream drains buffered text and clears the timer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            markdown = MagicMock()
            markdown.update = AsyncMock()
            stream = MagicMock()
            stream.write = AsyncMock()
            stream.stop = AsyncMock()
            msg._markdown = markdown
            msg._stream = stream

            await msg.append_content("partial")
            await msg.stop_stream()

            stream.write.assert_awaited_once_with("partial")
            stream.stop.assert_awaited_once_with()
            assert msg._flush_timer is None
            assert msg._pending_append == ""

    async def test_set_content_drains_and_cancels_active_timer(self) -> None:
        """`set_content` cancels a live flush timer and drops the buffer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            markdown = MagicMock()
            markdown.update = AsyncMock()
            stream = MagicMock()
            stream.write = AsyncMock()
            stream.stop = AsyncMock()
            msg._markdown = markdown
            msg._stream = stream

            await msg.append_content("buffered")
            assert msg._flush_timer is not None

            await msg.set_content("replacement")
            # Give a stale timer the chance to fire if it was not cancelled.
            await asyncio.sleep(msg._STREAM_FLUSH_INTERVAL * 2)
            await pilot.pause()

            assert msg._flush_timer is None
            assert msg._pending_append == ""
            # Buffered token must not bleed into the replacement render.
            stream.write.assert_not_awaited()
            markdown.update.assert_awaited_once_with("replacement")

    async def test_timer_created_once_across_appends(self) -> None:
        """Repeated appends reuse a single flush timer rather than spawning many."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            stream = MagicMock()
            stream.write = AsyncMock()
            msg._stream = stream

            await msg.append_content("a")
            timer = msg._flush_timer
            assert timer is not None

            await msg.append_content("b")
            await msg.append_content("c")

            assert msg._flush_timer is timer

    async def test_flush_drains_successive_batches(self) -> None:
        """Each flush writes the latest batch; an empty buffer is a no-op."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            stream = MagicMock()
            stream.write = AsyncMock()
            msg._stream = stream

            await msg.append_content("first")
            await msg._flush_pending_append()
            stream.write.assert_awaited_once_with("first")

            # Idle tick with nothing buffered must not write again.
            await msg._flush_pending_append()
            assert stream.write.await_count == 1

            await msg.append_content("second")
            await msg._flush_pending_append()
            assert stream.write.await_count == 2
            stream.write.assert_awaited_with("second")

    async def test_append_empty_text_is_noop(self) -> None:
        """Empty tokens neither buffer text nor arm the flush timer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)

            await msg.append_content("")

            assert msg._flush_timer is None
            assert msg._pending_append == ""
            assert msg._content == ""

    async def test_flush_restores_buffer_when_write_fails(self) -> None:
        """A failed write keeps the buffer for retry and never escapes the timer."""
        async with _AssistantMessageApp().run_test() as pilot:
            msg = pilot.app.query_one("#assistant", AssistantMessage)
            stream = MagicMock()
            stream.write = AsyncMock(side_effect=RuntimeError("render boom"))
            msg._stream = stream

            await msg.append_content("kept")
            # Must not raise: an escaping exception here would crash the app
            # via the Textual timer's exception handler.
            await msg._flush_pending_append()

            stream.write.assert_awaited_once_with("kept")
            assert msg._pending_append == "kept"

            # Text arriving after the failure queues behind the retried fragment.
            await msg.append_content(" more")
            assert msg._pending_append == "kept more"


class TestSummarizationMessage:
    """Tests for summarization notification widget."""

    def test_summarization_message_instantiates(self) -> None:
        """SummarizationMessage should instantiate with default content."""
        msg = SummarizationMessage()
        assert msg is not None

    def test_summarization_message_is_app_message(self) -> None:
        """SummarizationMessage should be treated like an AppMessage."""
        msg = SummarizationMessage()
        assert isinstance(msg, AppMessage)

    def test_summarization_message_str_input(self) -> None:
        """String input should be rendered as bold cyan `Content`."""
        msg = SummarizationMessage("custom text")
        rendered = msg._Static__content  # ty: ignore
        assert isinstance(rendered, Content)
        assert rendered.plain == "custom text"

    def test_summarization_message_content_passthrough(self) -> None:
        """Pre-styled `Content` should pass through unchanged."""
        pre = Content.styled("pre-styled", "bold cyan")
        msg = SummarizationMessage(pre)
        rendered = msg._Static__content  # ty: ignore
        assert rendered is pre


class TestToolCallMessageMarkupSafety:
    """Test ToolCallMessage handles output with brackets safely."""

    @pytest.mark.parametrize("output", MARKUP_INJECTION_CASES)
    def test_tool_output_no_markup_error(self, output: str) -> None:
        """ToolCallMessage should not raise MarkupError on bracket output."""
        msg = ToolCallMessage("test_tool", {"arg": "value"})
        msg._output = output
        assert msg._output == output

    def test_tool_call_with_bracket_args(self) -> None:
        """ToolCallMessage should handle args containing brackets."""
        args = {"code": "arr[0] = val[1]", "file": "test.py"}
        msg = ToolCallMessage("write_file", args)
        assert msg._args == args

    def test_tool_header_escapes_markup_in_label(self) -> None:
        """Task description widget should safely render bracket content."""
        msg = ToolCallMessage(
            "task",
            {"description": "Search for closing tag [/dim] mismatches"},
        )

        # Header shows subagent type; description is a separate dim widget.
        widgets = list(msg.compose())
        # Second widget is the task description line (Static with dim style).
        # Content.styled() produces a Content object stored on the Static.
        content = widgets[1]._Static__content  # ty: ignore
        assert "[/dim]" in content.plain

    def test_tool_args_line_escapes_markup_values(self) -> None:
        """Inline args line should escape bracket content in argument values."""
        msg = ToolCallMessage(
            "custom_tool",
            {"pattern": "[foo]", "note": "raw [/dim] text"},
        )

        widgets = list(msg.compose())
        args_widget = widgets[1]
        content = args_widget._Static__content  # ty: ignore
        assert isinstance(content, Content)
        assert "[foo]" in content.plain
        assert "[/dim]" in content.plain

    def test_ask_user_args_are_collapsed_by_default(self) -> None:
        """`ask_user` should show compact header without inline raw args."""
        msg = ToolCallMessage(
            "ask_user",
            {
                "questions": [
                    {
                        "question": 'Your prompt is just "hi" - what should I build?',
                        "type": "text",
                    }
                    for _ in range(4)
                ]
            },
        )

        widgets = list(msg.compose())
        visible = []
        for widget in widgets[:3]:
            content = widget._Static__content  # ty: ignore
            visible.append(content.plain if isinstance(content, Content) else content)
        visible_plain = "\n".join(visible)

        assert "ask_user(4 questions)" in visible_plain
        assert "Your prompt is just" not in visible_plain
        assert msg.has_expandable_args is True


class TestDiffMessageCredentialRedaction:
    """`DiffMessage` must not render the contents of credential files."""

    @staticmethod
    def _texts(widget: DiffMessage) -> list[str]:
        texts: list[str] = []
        for child in widget.compose():
            rendered = child.render()
            texts.append(
                rendered.plain if isinstance(rendered, Content) else str(rendered)
            )
        return texts

    def test_env_file_diff_is_hidden(self) -> None:
        diff = "@@ -1 +1 @@\n-API_KEY=old\n+API_KEY=supersecret"
        texts = self._texts(DiffMessage(diff, file_path=".env"))
        assert any("may contain credentials" in text for text in texts)
        assert all("supersecret" not in text for text in texts)

    def test_regular_file_diff_is_rendered(self) -> None:
        diff = "@@ -1 +1 @@\n-print('a')\n+print('b')"
        texts = self._texts(DiffMessage(diff, file_path="main.py"))
        assert all("may contain credentials" not in text for text in texts)
        assert any("print('b')" in text for text in texts)

    def test_empty_file_path_renders_diff(self) -> None:
        """An unknown (empty) path renders normally rather than falsely hiding.

        Callers always populate `file_path`, so a blank path means "unknown",
        not "credential"; it must not surface the redaction notice.
        """
        diff = "@@ -1 +1 @@\n-print('a')\n+print('b')"
        texts = self._texts(DiffMessage(diff, file_path=""))
        assert all("may contain credentials" not in text for text in texts)
        assert any("print('b')" in text for text in texts)


class TestToolCallMessageDuration:
    """Tests for the post-run duration shown on long-running tool calls."""

    async def test_execute_shows_took_after_success(self) -> None:
        """`execute` keeps its status row and reports how long it ran."""
        app = _tool_msg_app("execute", {"command": "sleep 1"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_running()
            app.msg._start_time -= 5  # ty: ignore
            app.msg.set_success("done")
            await pilot.pause()

            status = app.msg._status_widget
            assert status is not None
            assert status.display is True
            content = status._Static__content  # ty: ignore
            assert isinstance(content, Content)
            assert content.plain == "Took 5s"

    async def test_execute_shows_fractional_seconds(self) -> None:
        """Sub-minute `execute` runs report tenths — `elapsed` is a float.

        The running spinner truncates to whole seconds, but `set_success`
        passes the raw float to `format_duration`, so a regression that
        truncated `elapsed` to `int` would be caught here.
        """
        app = _tool_msg_app("execute", {"command": "true"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_running()
            app.msg._start_time -= 0.3  # ty: ignore
            app.msg.set_success("done")
            await pilot.pause()

            status = app.msg._status_widget
            assert status is not None
            content = status._Static__content  # ty: ignore
            assert isinstance(content, Content)
            assert content.plain == "Took 0.3s"

    async def test_task_shows_took_after_success(self) -> None:
        """`task` subagent calls keep their status row and report how long they ran."""
        app = _tool_msg_app("task", {"description": "investigate the bug"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_running()
            app.msg._start_time -= 5  # ty: ignore
            app.msg.set_success("done")
            await pilot.pause()

            status = app.msg._status_widget
            assert status is not None
            assert status.display is True
            content = status._Static__content  # ty: ignore
            assert isinstance(content, Content)
            assert content.plain == "Took 5s"

    async def test_task_took_duration_survives_rehydration(self) -> None:
        """A virtualized task row restores its completed duration."""
        app = _tool_msg_app("task", {"description": "investigate the bug"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_running()
            app.msg._start_time -= 5  # ty: ignore
            app.msg.set_success("done")
            data = MessageData.from_widget(app.msg)
            assert data.tool_duration == pytest.approx(5, abs=0.1)

        restored = data.to_widget()
        assert isinstance(restored, ToolCallMessage)
        rehydrated_app = _tool_msg_app("task")
        rehydrated_app.msg = restored
        async with rehydrated_app.run_test() as pilot:
            await pilot.pause()

            status = restored._status_widget
            assert status is not None
            assert status.display is True
            content = status._Static__content  # ty: ignore
            assert isinstance(content, Content)
            assert content.plain == "Took 5s"

    async def test_execute_without_run_falls_back_to_success_status(self) -> None:
        """`execute` success with no recorded start time hides the row.

        Without a prior `set_running`, `_start_time` is `None`, so the
        `elapsed is not None` guard must route to `_show_success_status`
        (which hides the row here because output is present) rather than
        computing a duration from `None` and crashing.
        """
        app = _tool_msg_app("execute", {"command": "true"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success("done")
            await pilot.pause()

            status = app.msg._status_widget
            assert status is not None
            assert status.display is False

    async def test_non_execute_hides_status_on_success(self) -> None:
        """Non-`execute` tools hide the status row and never show a duration."""
        app = _tool_msg_app("read_file", {"file_path": "a.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_running()
            app.msg.set_success("contents")
            await pilot.pause()

            status = app.msg._status_widget
            assert status is not None
            assert status.display is False
            content = status._Static__content  # ty: ignore
            assert "Took" not in getattr(content, "plain", str(content))


class TestToolCallMessageTerminalStateGuards:
    """A rejected/skipped row must not flip to success/error on a resumed turn."""

    async def test_set_success_noop_on_rejected_row(self) -> None:
        """A resumed synthetic success ToolMessage keeps a rejected row rejected.

        After a reasoned reject the turn can resume and stream a synthetic
        ToolMessage for the rejected tool; `set_success` must be ignored so the
        row keeps its terminal rejected state instead of flipping.
        """
        app = _tool_msg_app("execute", {"command": "rm -rf /"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_rejected()
            assert app.msg._status == "rejected"
            app.msg.set_success("done")
            await pilot.pause()
            assert app.msg._status == "rejected"
            assert app.msg.is_success is False

    async def test_set_error_noop_on_rejected_row(self) -> None:
        """A resumed synthetic error ToolMessage keeps a rejected row rejected."""
        app = _tool_msg_app("execute", {"command": "rm -rf /"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_rejected()
            app.msg.set_error("boom")
            await pilot.pause()
            assert app.msg._status == "rejected"

    async def test_set_success_noop_on_skipped_row(self) -> None:
        """A skipped row (sibling rejection) stays skipped, not flipped to success.

        The guard names both `rejected` and `skipped`; a tool skipped because a
        sibling was rejected can still receive a synthetic success ToolMessage on
        the resumed turn, which must be ignored.
        """
        app = _tool_msg_app("execute", {"command": "ls"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_skipped()
            assert app.msg._status == "skipped"
            app.msg.set_success("done")
            await pilot.pause()
            assert app.msg._status == "skipped"
            assert app.msg.is_success is False

    async def test_set_error_noop_on_skipped_row(self) -> None:
        """A skipped row keeps its terminal state instead of flipping to error."""
        app = _tool_msg_app("execute", {"command": "ls"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_skipped()
            app.msg.set_error("boom")
            await pilot.pause()
            assert app.msg._status == "skipped"


class TestToolCallMessageArgs:
    """The public `args` accessor must not expose internal widget state."""

    def test_args_returns_shallow_copy(self) -> None:
        """Rebinding top-level keys of the returned dict must not affect `_args`.

        Hook payloads are built directly from `tool_msg.args`, so the copy is a
        load-bearing safety contract: a consumer that reassigns its payload's
        top-level keys must not corrupt the widget's stored arguments by
        reference. The copy is shallow — nested mutable values are shared (see
        `test_args_nested_values_are_shared`) — which is sufficient because the
        only consumer serializes the payload rather than deep-mutating it.
        """
        msg = ToolCallMessage("write_file", {"file_path": "a.py", "content": "x"})
        returned = msg.args
        returned["file_path"] = "hacked.py"
        returned["injected"] = True
        assert msg.args == {"file_path": "a.py", "content": "x"}
        assert msg._args == {"file_path": "a.py", "content": "x"}

    def test_args_nested_values_are_shared(self) -> None:
        """The copy is shallow: nested mutables are shared, not deep-copied.

        Pins the documented boundary of the `args` accessor so a future reader
        does not mistake the shallow copy for a deep one.
        """
        msg = ToolCallMessage("edit_file", {"edits": [{"old": "a"}]})
        returned = msg.args
        returned["edits"][0]["old"] = "mutated"
        assert msg._args["edits"][0]["old"] == "mutated"


class TestToolCallMessageTodos:
    """Tests for `write_todos` output formatting."""

    def test_todo_preview_truncates_long_content(self) -> None:
        """Collapsed todo preview should keep the compact character limit."""
        long = "Implement " + "very detailed authentication flow " * 4
        msg = ToolCallMessage("write_todos")

        result = msg._format_todos_output(
            repr([{"content": long, "status": "in_progress"}]),
            is_preview=True,
        )

        assert result.content.plain.endswith("...")
        assert long not in result.content.plain
        assert result.truncation == "full todo text"

    async def test_todo_collapsed_short_output_uses_preview_formatting(self) -> None:
        """Collapsed todos should truncate even when raw output fits generically."""
        from textual.app import App, ComposeResult

        long = "Implement " + "very detailed authentication flow " * 3
        assert len(long) > 70
        output = repr([{"content": long, "status": "pending"}])
        assert len(output) < ToolCallMessage._PREVIEW_CHARS

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("write_todos")

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._preview_widget is not None
            assert app.msg._hint_widget is not None
            content = app.msg._preview_widget._Static__content  # ty: ignore
            assert isinstance(content, Content)
            assert "..." in content.plain
            assert long not in content.plain
            assert app.msg._hint_widget.display is True

    async def test_todo_short_fully_visible_output_does_not_expand(self) -> None:
        """Clicking fully visible todo output should not show a collapse hint."""
        from textual.app import App, ComposeResult

        output = repr([{"content": "Write tests", "status": "pending"}])

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("write_todos")

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is False

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._hint_widget.display is False

    def test_todo_expanded_shows_full_wrapped_content(self) -> None:
        """Expanded todo output should wrap long content without truncating."""
        long = (
            "Implement the new authentication flow using OAuth2 with PKCE for "
            "the CLI login command and preserve readable todo output"
        )
        msg = ToolCallMessage("write_todos")

        result = msg._format_todos_output(
            repr([{"content": long, "status": "in_progress"}]),
            is_preview=False,
        )
        plain = result.content.plain

        # Continuation lines hang-indent to the width of the status label, which
        # starts flush at the gutter (the formatter emits no leading pad).
        from deepagents_code.config import get_glyphs

        indent = "\n" + " " * len(f"{get_glyphs().circle_filled} active ")
        assert "..." not in plain
        assert long.replace(" ", "") == plain.split("active ", 1)[1].replace(
            indent,
            "",
        ).replace(" ", "")
        assert indent in plain

    def test_todo_rows_start_flush_at_gutter(self) -> None:
        """No formatted todo line carries a hardcoded leading pad.

        Covers the status rows (which begin with the status glyph) as well as
        the stats header, which is emitted flush at the gutter too.
        """
        msg = ToolCallMessage("write_todos")

        result = msg._format_todos_output(
            repr(
                [
                    {"content": "a", "status": "completed"},
                    {"content": "b", "status": "in_progress"},
                    {"content": "c", "status": "pending"},
                ]
            ),
            is_preview=False,
        )
        lines = result.content.plain.split("\n")

        assert lines
        assert all(not line.startswith(" ") for line in lines)

    def test_todo_empty_state_is_flush(self) -> None:
        """The empty-list placeholder sits flush at the gutter, no leading pad."""
        msg = ToolCallMessage("write_todos")

        result = msg._format_todos_output(repr([]), is_preview=False)

        assert result.content.plain == "No todos"

    def test_todo_expanded_continuation_aligns_content_column(self) -> None:
        """Wrapped continuation lines should align under the todo text."""
        long = "Write integration tests for " + "token refresh revocation " * 4
        msg = ToolCallMessage("write_todos")

        result = msg._format_todos_output(
            repr([{"content": long, "status": "pending"}]),
            is_preview=False,
        )
        lines = result.content.plain.splitlines()
        todo_start = next(
            index for index, line in enumerate(lines) if "todo   " in line
        )

        # Continuation aligns under the todo text, i.e. the status-label width.
        # Assert the exact leading-whitespace width, not just a prefix, so a pad
        # reintroduced only on wrapped lines (wider than the label) is caught.
        from deepagents_code.config import get_glyphs

        indent = " " * len(f"{get_glyphs().circle_empty} todo   ")
        assert len(lines) > todo_start + 1
        continuation = lines[todo_start + 1]
        assert len(continuation) - len(continuation.lstrip(" ")) == len(indent)


class _ToolMsgApp(App[None]):
    """Single-`ToolCallMessage` Textual app for pilot-driven tests."""

    def __init__(self, tool_name: str, args: dict | None = None) -> None:
        super().__init__()
        self.msg = ToolCallMessage(tool_name, args)

    def compose(self) -> ComposeResult:
        yield self.msg


def _tool_msg_app(tool_name: str, args: dict | None = None) -> _ToolMsgApp:
    """Build a single-`ToolCallMessage` Textual app for pilot-driven tests.

    Args:
        tool_name: Tool name the message represents.
        args: Optional tool-call arguments.

    Returns:
        An unmounted `App` exposing the message as `app.msg`.
    """
    return _ToolMsgApp(tool_name, args)


class TestToolCallMessageOutputGutter:
    """The output glyph lives in a fixed gutter so wrapped lines stay aligned."""

    async def test_glyph_in_gutter_not_baked_into_content(self) -> None:
        """The output marker renders in its own gutter column, not in content.

        Regression: when a single long output line soft-wraps, the wrapped
        remainder must not fall under the glyph. Keeping the glyph in a fixed
        gutter (instead of baked into the first content line) lets the content
        widget own a single hanging indent for every wrapped line.
        """
        from deepagents_code.config import get_glyphs

        # Two logical lines; the first is long enough to soft-wrap in a terminal.
        output = (
            "[stderr] fatal: ambiguous argument 'main..branch': unknown revision "
            "or path not in the working tree.\n[stderr] Use '--' to separate paths."
        )

        app = _tool_msg_app("execute", {"command": "git diff main..branch"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            glyph = get_glyphs().output_prefix
            assert app.msg._preview_widget is not None
            content = app.msg._preview_widget._Static__content  # ty: ignore

            # Content is bare: no glyph, and no hand-rolled hanging indent on
            # any logical line (alignment is owned by the gutter layout).
            assert glyph not in content.plain
            assert all(not line.startswith(" ") for line in content.plain.split("\n"))

            # The glyph renders exactly once, in the gutter beside the content.
            assert app.msg._preview_row is not None
            assert app.msg._preview_row.display is True
            gutters = app.msg._preview_row.query(".tool-output-gutter")
            assert len(gutters) == 1
            gutter_content = gutters.first()._Static__content  # ty: ignore
            assert gutter_content == glyph

    async def test_collapsed_preview_preserves_uniform_leading_indent(self) -> None:
        """Collapsed preview keeps line 0's indent so indented rows align.

        Regression: the preview branch pre-stripped the output, lstripping the
        first line only while continuation lines kept their indent. Uniformly
        indented output (e.g. `git branch -r`, which prefixes every branch with
        two spaces) then rendered with line 0 flush and the rest indented. The
        formatter must preserve the shared leading indent across all rows.
        """
        # Mirror `git branch -r`: every row indented by two spaces, > preview
        # line budget so the collapsed preview is shown.
        output = "\n".join(f"  origin/branch-{i}" for i in range(8))

        app = _tool_msg_app("execute", {"command": "git branch -r"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._preview_widget is not None
            assert app.msg._expanded is False
            content = app.msg._preview_widget._Static__content  # ty: ignore

            preview_lines = content.plain.split("\n")
            # Every visible row — including the first — keeps git's two-space
            # indent, so they share a left edge beside the glyph gutter.
            assert preview_lines
            assert all(line.startswith("  origin/") for line in preview_lines)


class TestToolCallMessageSearchOutput:
    """Tests for grep/glob result formatting in `_format_search_output`."""

    def test_glob_list_output_has_no_hardcoded_indent(self) -> None:
        """Glob (list) results must not carry a hardcoded leading indent.

        Alignment is owned by the output gutter layout; the formatter emits
        bare paths so results aren't double-indented under the output marker.
        """
        msg = ToolCallMessage("glob", {"pattern": "**/*.py"})
        result = msg._format_search_output(
            "['/tmp/zzz_a.py', '/tmp/zzz_b.py']", is_preview=False
        )
        lines = result.content.plain.split("\n")
        assert lines
        assert all(not line.startswith(" ") for line in lines)

    def test_grep_line_output_has_no_hardcoded_indent(self) -> None:
        """Grep (line-based) results must not carry a hardcoded leading indent.

        This is a distinct branch from the glob list path: `ast.literal_eval`
        fails for grep output, so it falls through to line-based formatting.
        """
        msg = ToolCallMessage("grep", {"pattern": "x"})
        result = msg._format_search_output(
            "file.py:1:match one\nfile.py:2:match two", is_preview=False
        )
        assert result.content.plain.split("\n") == [
            "file.py:1:match one",
            "file.py:2:match two",
        ]

    def test_grep_preview_truncates_long_single_line(self) -> None:
        """Grep previews should cap long single-line output by characters."""
        msg = ToolCallMessage("grep", {"pattern": "x"})
        output = "file.py:1:" + "x" * ToolCallMessage._PREVIEW_CHARS

        result = msg._format_search_output(output, is_preview=True)

        # The visible slice is exactly the leading char budget of the input,
        # not just any string of the right length.
        assert result.content.plain == output[: ToolCallMessage._PREVIEW_CHARS]
        assert len(result.content.plain) == ToolCallMessage._PREVIEW_CHARS
        assert result.truncation is not None
        assert result.truncation.endswith("more chars")

    def test_grep_preview_truncates_long_multiline_by_chars(self) -> None:
        """Grep previews should cap long multi-line output by characters."""
        msg = ToolCallMessage("grep", {"pattern": "x"})
        # Two wide lines, each under the budget but together over it, so both
        # become rows (no hidden line) and the second is char-sliced — forcing
        # the char hint over the line hint. Width derives from the budget.
        char_run = ToolCallMessage._PREVIEW_CHARS // 2
        lines = [f"file.py:{index}:" + "x" * char_run for index in range(2)]

        result = msg._format_search_output("\n".join(lines), is_preview=True)

        assert len(result.content.plain) == ToolCallMessage._PREVIEW_CHARS
        assert result.truncation is not None
        assert result.truncation.endswith("more chars")

    def test_glob_preview_truncates_long_paths_by_chars(self) -> None:
        """Glob previews cap wide path lists by characters with a file hint."""
        msg = ToolCallMessage("glob", {"pattern": "**/*.py"})
        # Two paths that each fit under the budget but together overflow it, so
        # both become rows (no hidden line) and the second is char-sliced —
        # forcing the char hint rather than the file-count hint.
        long_path = "/tmp/" + "z" * (ToolCallMessage._PREVIEW_CHARS // 2) + ".py"
        output = repr([long_path, long_path])

        result = msg._format_search_output(output, is_preview=True)

        assert len(result.content.plain) == ToolCallMessage._PREVIEW_CHARS
        assert result.truncation is not None
        assert result.truncation.endswith("more chars")

    def test_grep_preview_truncates_by_line_count(self) -> None:
        """Grep previews over the line cap report hidden lines, not chars."""
        msg = ToolCallMessage("grep", {"pattern": "x"})
        output = "\n".join(f"file.py:{index}:hit" for index in range(8))

        result = msg._format_search_output(output, is_preview=True)

        # 8 short lines, preview cap is 5 → 3 hidden, counted as lines.
        assert result.truncation == "3 more lines"

    def test_glob_preview_truncates_by_file_count(self) -> None:
        """Glob previews over the line cap report hidden files, not lines."""
        msg = ToolCallMessage("glob", {"pattern": "**/*.py"})
        paths = [f"/tmp/result_{index}.py" for index in range(8)]

        result = msg._format_search_output(repr(paths), is_preview=True)

        # The "files" unit is what distinguishes the glob path from grep.
        assert result.truncation == "3 more files"

    def test_grep_preview_prefers_line_count_when_both_caps_hit(self) -> None:
        """When both caps trip, the hidden-line count wins over chars."""
        msg = ToolCallMessage("grep", {"pattern": "x"})
        output = "\n".join(f"file.py:{index}:" + "y" * 100 for index in range(10))

        result = msg._format_search_output(output, is_preview=True)

        assert result.truncation is not None
        assert result.truncation.endswith("more lines")

    def test_search_full_output_is_untruncated(self) -> None:
        """Non-preview formatting returns every row with no truncation hint."""
        msg = ToolCallMessage("grep", {"pattern": "x"})
        lines = [f"file.py:{index}:" + "z" * 200 for index in range(10)]

        result = msg._format_search_output("\n".join(lines), is_preview=False)

        assert result.truncation is None
        assert result.content.plain.split("\n") == lines


class TestToolCallMessageLsOutput:
    """Tests for `ls` directory-listing formatting in `_format_ls_output`."""

    def test_ls_output_has_no_hardcoded_indent(self) -> None:
        """Ls entries sit flush under the output gutter, like grep/glob.

        Alignment is owned by the output gutter layout; the formatter emits
        bare names so results aren't double-indented under the output marker.
        Directories keep their trailing slash. Every styled file-type branch
        (python, config, dir, plain) is exercised so none reintroduces a pad.
        """
        msg = ToolCallMessage("ls", {"path": "/tmp"})
        result = msg._format_ls_output(
            "['/tmp/SKILL.md', '/tmp/scripts', '/tmp/init.py', '/tmp/config.json']",
            is_preview=False,
        )
        lines = result.content.plain.split("\n")
        assert lines == ["SKILL.md", "scripts/", "init.py", "config.json"]
        assert all(not line.startswith(" ") for line in lines)


class TestToolCallMessageEditFileOutput:
    """edit_file hides its success result line but still surfaces errors."""

    def test_edit_file_success_preview_renders_no_lines(self) -> None:
        """A successful edit preview stays hidden; the status glyph speaks for it."""
        msg = ToolCallMessage("edit_file", {"file_path": "/tmp/f.py"})
        msg._status = "success"

        result = msg._format_edit_file_output(
            "Successfully replaced 1 instance(s) of the string in '/tmp/f.py'",
            is_preview=True,
        )

        assert result.content.plain == ""
        assert result.truncation is None

    def test_edit_file_success_full_renders_original_output(self) -> None:
        """A successful edit's full output remains recoverable."""
        msg = ToolCallMessage("edit_file", {"file_path": "/tmp/f.py"})
        msg._status = "success"
        output = "Successfully replaced 1 instance(s) of the string in '/tmp/f.py'"

        result = msg._format_edit_file_output(output, is_preview=False)

        assert result.content.plain == output
        assert result.truncation is None

    def test_edit_file_error_still_renders(self) -> None:
        """Errors must still render so failures stay visible."""
        msg = ToolCallMessage("edit_file", {"file_path": "/tmp/f.py"})
        msg._status = "error"

        result = msg._format_edit_file_output(
            "Error: String not found in file", is_preview=False
        )

        assert "String not found in file" in result.content.plain

    async def test_edit_file_success_expands_to_original_output(self) -> None:
        """End to end: a successful edit_file hides preview but expands to output."""
        output = "Successfully replaced 2 instance(s) of the string in '/tmp/f.py'"
        app = _tool_msg_app("edit_file", {"file_path": "/tmp/f.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._preview_row is not None
            assert app.msg._full_row is not None
            assert app.msg._hint_widget is not None
            assert app.msg._preview_row.display is False
            assert app.msg._full_row.display is False
            assert app.msg._hint_widget.display is True
            assert app.msg._has_expandable_output() is True

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._preview_row.display is False
            assert app.msg._full_row.display is True
            full = app.msg._full_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert full.plain == output


class TestToolCallMessageSuccessStatus:
    """A successful call with no output shows a "Success!" status marker."""

    async def test_success_without_output_shows_success_status(self) -> None:
        """edit_file (no visible output) shows the success marker instead of hiding."""
        from deepagents_code.config import get_glyphs

        app = _tool_msg_app("edit_file", {"file_path": "/tmp/f.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(
                "Successfully replaced 1 instance(s) of the string in '/tmp/f.py'"
            )
            await pilot.pause()

            assert app.msg._status_widget is not None
            assert app.msg._status_widget.display is True
            assert app.msg._status_widget.has_class("success")
            content = app.msg._status_widget._Static__content  # ty: ignore
            assert get_glyphs().checkmark in content.plain
            assert "Success!" in content.plain

    async def test_success_with_output_keeps_status_hidden(self) -> None:
        """A tool whose output speaks for itself keeps the status hidden."""
        app = _tool_msg_app("read_file", {"file_path": "/tmp/f.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success("line one\nline two")
            await pilot.pause()

            assert app.msg._status_widget is not None
            assert app.msg._status_widget.display is False
            assert not app.msg._status_widget.has_class("success")


class TestToolCallMessageExpandHint:
    """Tests for the preview/expand hint on collapsed tool output."""

    async def test_long_single_line_search_output_collapses_and_expands(self) -> None:
        """Long single-line grep/glob output collapses by default and expands.

        grep/glob collapse their body entirely (the header names the pattern),
        so even long output shows a count-free expand hint instead of a
        truncated preview; expanding reveals the full untruncated content.
        """
        from textual.app import App, ComposeResult

        output = "Invalid glob pattern: " + "a" * ToolCallMessage._PREVIEW_CHARS
        assert "\n" not in output
        assert len(output) > ToolCallMessage._PREVIEW_CHARS

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("glob", {"pattern": "**/*.py"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is True
            assert app.msg._has_expandable_output() is True
            # Preview is collapsed away; a count-free expand affordance is shown.
            assert app.msg._preview_row is not None
            assert app.msg._preview_row.display is False
            hint = app.msg._hint_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert "expand" in hint.plain
            assert "more" not in hint.plain

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._hint_widget.display is True
            full = app.msg._full_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert full.plain == output

    @pytest.mark.parametrize(
        ("tool", "error"),
        [
            ("glob", "Error: glob timed out after 20.0s. Try a narrower path."),
            ("grep", "Error: invalid regex: unterminated character class."),
        ],
    )
    async def test_short_error_force_expanded_has_no_collapse_hint(
        self, tool: str, error: str
    ) -> None:
        """A short force-expanded error must not show a collapse affordance.

        `set_error` force-expands so the full error is always visible. When the
        error is short enough that the collapsed form would be identical, there
        is nothing to collapse — so no hint, and toggling is a no-op. grep and
        glob share the collapse-by-default branch, so both must honor this.
        """
        assert "\n" not in error
        assert len(error) < ToolCallMessage._PREVIEW_CHARS

        app = _tool_msg_app(tool, {"pattern": "**/*.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_error(error)
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is False
            assert app.msg._has_expandable_output() is False

            app.msg.toggle_output()
            await pilot.pause()

            # Nothing to collapse — stays expanded with the hint hidden.
            assert app.msg._expanded is True
            assert app.msg._hint_widget.display is False

    async def test_multiline_error_force_expanded_offers_collapse(self) -> None:
        """A long force-expanded error should still offer a collapse affordance.

        The positive counterpart to the short-error case: a multi-line error
        exceeds the line threshold and the formatter truncates it, so a smaller
        collapsed form exists and the collapse hint must appear.
        """
        error = "\n".join(f"line {index} of the traceback" for index in range(10))
        assert error.count("\n") + 1 > ToolCallMessage._PREVIEW_LINES

        app = _tool_msg_app("glob", {"pattern": "**/*.py"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_error(error)
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is True
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "collapse" in hint.plain

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._hint_widget.display is True
            collapsed = app.msg._hint_widget._Static__content
            assert "expand" in collapsed.plain

    async def test_long_grep_output_collapses_and_expands(self) -> None:
        """A multi-line grep result collapses its preview then expands on toggle."""
        output = "\n".join(f"file.py:{index}:hit {index}" for index in range(8))
        assert output.count("\n") + 1 > ToolCallMessage._PREVIEW_LINES

        app = _tool_msg_app("grep", {"pattern": "hit"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._has_expandable_output() is True
            assert app.msg._preview_row is not None
            assert app.msg._full_widget is not None
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is True
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in hint.plain
            # The preview is collapsed away entirely rather than truncated.
            assert app.msg._preview_row.display is False

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is True
            full = app.msg._full_widget._Static__content
            assert "hit 7" in full.plain
            collapsed = app.msg._hint_widget._Static__content
            assert "collapse" in collapsed.plain

    async def test_short_non_todo_output_renders_full_without_hint(self) -> None:
        """Short non-todo output uses non-preview formatting and shows no hint.

        Guards the merged collapsed branch: `is_preview` must stay `False` for
        a non-`write_todos` tool below the size threshold, so the full content
        is shown rather than a truncated preview.
        """
        # Five lines: under `_PREVIEW_LINES` (6) but over the shell formatter's
        # own four-line preview cap, so a stray `is_preview=True` would truncate.
        output = "\n".join(f"line {index}" for index in range(5))
        assert output.count("\n") + 1 < ToolCallMessage._PREVIEW_LINES

        app = _tool_msg_app("execute", {"command": "echo hi"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._has_expandable_output() is False
            assert app.msg._preview_widget is not None
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is False
            preview = app.msg._preview_widget._Static__content  # ty: ignore
            assert "line 0" in preview.plain
            assert "line 4" in preview.plain

    async def test_read_file_collapses_preview_by_default(self) -> None:
        """`read_file` hides its content preview by default but stays expandable.

        The file path is already shown in the header, so echoing the contents
        inline is noise. The collapsed view shows an expand hint instead of the
        preview, and expanding reveals the full content.
        """
        # Short output that any other tool would render fully inline.
        output = "\n".join(f"line {index}" for index in range(3))

        app = _tool_msg_app("read_file", {"path": "/tmp/x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._preview_row is not None
            assert app.msg._hint_widget is not None
            # Preview is collapsed away; an expand affordance is shown instead.
            assert app.msg._preview_row.display is False
            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget.display is True
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in hint.plain

            # Expanding reveals the full content.
            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is True
            assert app.msg._full_row is not None
            assert app.msg._full_row.display is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore
            assert "line 0" in full.plain
            assert "line 2" in full.plain

    async def test_large_read_file_collapses_preview_regardless_of_size(
        self,
    ) -> None:
        """Large `read_file` output collapses with a count-free hint and round-trips.

        The short-output case can't prove the "collapse regardless of size"
        invariant — short output wouldn't preview-truncate for any tool. This
        uses output well over `_PREVIEW_LINES`, so a normal tool would render a
        truncated preview with an "N more lines" hint. `read_file` instead hides
        the preview entirely and shows a count-free expand affordance, then
        toggles cleanly back to collapsed.
        """
        line_count = ToolCallMessage._PREVIEW_LINES * 5
        output = "\n".join(f"line {index}" for index in range(line_count))

        app = _tool_msg_app("read_file", {"path": "/tmp/big"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._preview_row is not None
            assert app.msg._full_row is not None
            assert app.msg._hint_widget is not None
            # Preview stays hidden even though the size would normally truncate.
            assert app.msg._preview_row.display is False
            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget.display is True
            # The hint is count-free — no "N more lines" prefix other tools show.
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in hint.plain
            assert "more" not in hint.plain

            # Expanding reveals the full content — including the last line — and
            # offers a collapse affordance.
            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is True
            assert app.msg._full_row.display is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore
            assert f"line {line_count - 1}" in full.plain
            assert app.msg._hint_widget.display is True
            collapse_hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "collapse" in collapse_hint.plain

            # Toggling again re-collapses back to the count-free expand hint.
            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is False
            assert app.msg._preview_row.display is False
            assert app.msg._full_row.display is False
            recollapsed = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in recollapsed.plain
            assert "more" not in recollapsed.plain

    async def test_read_file_click_toggles_output(self) -> None:
        """Clicking a collapsed `read_file` expands it via `has_expandable_output`.

        The public `has_expandable_output` property drives the click / Ctrl+O
        routing in `on_click`; `read_file` must report as expandable there so a
        click reveals the content instead of falling through to the args block.
        """
        output = "\n".join(f"line {index}" for index in range(3))

        app = _tool_msg_app("read_file", {"path": "/tmp/x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg.has_expandable_output is True
            assert app.msg._expanded is False

            event = MagicMock()
            app.msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            assert app.msg._expanded is True

    async def test_short_read_file_error_force_expanded_has_no_collapse_hint(
        self,
    ) -> None:
        """Short `read_file` errors stay visible and non-collapsible."""
        error = "Permission denied"

        app = _tool_msg_app("read_file", {"path": "/etc/passwd"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_error(error)
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is False
            assert app.msg._has_expandable_output() is False
            assert app.msg._full_row is not None
            assert app.msg._full_row.display is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore
            assert error in full.plain

            app.msg.toggle_output()
            await pilot.pause()

            assert app.msg._expanded is True
            assert app.msg._hint_widget.display is False
            assert app.msg._full_row.display is True

    @pytest.mark.parametrize(
        ("tool", "output", "expected"),
        [
            ("grep", "file.py:1:hit one\nfile.py:2:hit two", "hit one"),
            ("glob", "['a.py', 'b.py']", "a.py"),
        ],
    )
    async def test_search_collapses_preview_by_default(
        self, tool: str, output: str, expected: str
    ) -> None:
        """`grep`/`glob` hide their result preview by default but stay expandable.

        The search pattern is already shown in the header, so echoing the matches
        inline is noise. The collapsed view shows an expand hint instead of the
        preview, and expanding reveals the full content.
        """
        app = _tool_msg_app(tool, {"pattern": "x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._preview_row is not None
            assert app.msg._hint_widget is not None
            # Preview is collapsed away; an expand affordance is shown instead.
            assert app.msg._preview_row.display is False
            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget.display is True
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in hint.plain

            # Expanding reveals the full content.
            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is True
            assert app.msg._full_row is not None
            assert app.msg._full_row.display is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore
            assert expected in full.plain

    @pytest.mark.parametrize(
        "tool",
        ["grep", "glob"],
    )
    async def test_large_search_collapses_preview_regardless_of_size(
        self, tool: str
    ) -> None:
        """Large `grep`/`glob` output collapses with a count-free hint.

        Output well over `_PREVIEW_LINES` would normally render a truncated
        preview with an "N more lines/files" hint. grep/glob instead hide the
        preview entirely and show a count-free expand affordance.
        """
        line_count = ToolCallMessage._PREVIEW_LINES * 5
        if tool == "glob":
            output = repr([f"/tmp/result_{index}.py" for index in range(line_count)])
        else:
            output = "\n".join(f"file.py:{index}:hit" for index in range(line_count))

        app = _tool_msg_app(tool, {"pattern": "x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._expanded is False
            assert app.msg._preview_row is not None
            assert app.msg._hint_widget is not None
            # Preview stays hidden even though the size would normally truncate.
            assert app.msg._preview_row.display is False
            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget.display is True
            # The hint is count-free — no "N more" prefix other tools show.
            hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "expand" in hint.plain
            assert "more" not in hint.plain

            # Expanding reveals the full content and offers a collapse affordance.
            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore
            assert f"{line_count - 1}" in full.plain
            collapse_hint = app.msg._hint_widget._Static__content  # ty: ignore
            assert "collapse" in collapse_hint.plain

    async def test_search_click_toggles_output(self) -> None:
        """Clicking a collapsed `grep` expands it via `has_expandable_output`."""
        output = "file.py:1:hit one\nfile.py:2:hit two"

        app = _tool_msg_app("grep", {"pattern": "x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg.has_expandable_output is True
            assert app.msg._expanded is False

            event = MagicMock()
            app.msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            assert app.msg._expanded is True


class TestToolCallMessageEmptyResult:
    """Empty file-op results render nothing instead of an empty box."""

    @pytest.mark.parametrize(
        ("tool", "output"),
        [
            ("glob", "[]"),
            ("grep", "[]"),
            ("ls", "[]"),
            ("glob", "   "),
            # `read_file` has its own collapse branch in `_update_output_display`
            # that sits *below* the shared empty guard; a whitespace-only read
            # must still be suppressed by the guard rather than reaching that
            # branch and rendering an empty box with a bogus expand hint.
            ("read_file", "   "),
        ],
    )
    async def test_empty_serialized_result_hides_output(
        self, tool: str, output: str
    ) -> None:
        """A non-empty raw string that formats to nothing must not render a box.

        `[]` is a synthetic stand-in for output that is a non-empty raw string
        yet formats to no visible content. It is not what the tools actually
        emit for an empty result — real grep/glob return "No matches found" /
        "No files found" (non-empty), which render inline (see
        `test_search_no_result_message_renders_without_expand_hint`). The raw
        output here is truthy, so the early empty guard doesn't fire; without the
        formatted-emptiness check the preview row would render as an empty box
        with a misleading expand affordance. The whitespace-only case ("   ")
        exercises the same check now that the collapsed branch's own empty guard
        is gone.
        """
        app = _tool_msg_app(tool)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._preview_row is not None
            assert app.msg._full_row is not None
            assert app.msg._hint_widget is not None
            assert app.msg._preview_row.display is False
            assert app.msg._full_row.display is False
            assert app.msg._hint_widget.display is False
            assert app.msg._has_expandable_output() is False

    @pytest.mark.parametrize(
        ("tool", "output"),
        [
            ("grep", "No matches found"),
            ("glob", "No files found"),
        ],
    )
    async def test_search_no_result_message_renders_without_expand_hint(
        self, tool: str, output: str
    ) -> None:
        """Search no-result messages stay visible and are not expandable."""
        app = _tool_msg_app(tool)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success(output)
            await pilot.pause()

            assert app.msg._preview_row is not None
            assert app.msg._preview_widget is not None
            assert app.msg._full_row is not None
            assert app.msg._hint_widget is not None
            assert app.msg._preview_row.display is True
            assert app.msg._full_row.display is False
            assert app.msg._hint_widget.display is False
            assert app.msg._has_expandable_output() is False
            preview = app.msg._preview_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert preview.plain == output

    async def test_non_empty_serialized_result_still_renders(self) -> None:
        """A populated result must still render — the guard can't false-positive.

        glob collapses its body by default, so "renders" here means it stays
        expandable (not hidden by the empty guard) and the content is reachable
        once expanded, rather than shown inline.
        """
        app = _tool_msg_app("glob")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_success("['a.py', 'b.py']")
            await pilot.pause()

            assert app.msg._has_expandable_output() is True
            assert app.msg._hint_widget is not None
            assert app.msg._hint_widget.display is True

            app.msg.toggle_output()
            await pilot.pause()
            assert app.msg._expanded is True
            assert app.msg._full_widget is not None
            full = app.msg._full_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert "a.py" in full.plain

    async def test_error_body_is_not_hidden(self) -> None:
        """A real (non-empty) error body must stay visible.

        The emptiness guard runs regardless of status, so this pins the
        invariant that a human-readable error — which always formats non-empty —
        is shown in full rather than collapsed away.
        """
        app = _tool_msg_app("grep", {"pattern": "x"})
        async with app.run_test() as pilot:
            await pilot.pause()
            app.msg.set_error("grep: invalid pattern")
            await pilot.pause()

            assert app.msg._full_row is not None
            assert app.msg._full_widget is not None
            assert app.msg._full_row.display is True
            full = app.msg._full_widget._Static__content  # ty: ignore[unresolved-attribute]
            assert "invalid pattern" in full.plain


class TestToolCallMessageExpandableArgs:
    """Tests for the `ask_user` expandable-arguments toggle."""

    def test_has_expandable_args_false_for_non_ask_user(self) -> None:
        """Only `ask_user` should expose expandable args."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/x"})
        assert msg.has_expandable_args is False

    def test_has_expandable_args_false_for_ask_user_without_args(self) -> None:
        """Empty args dict should not be expandable."""
        msg = ToolCallMessage("ask_user", {})
        assert msg.has_expandable_args is False

    def test_tool_name_property_exposes_underlying_name(self) -> None:
        """Public `tool_name` property should mirror the constructor arg."""
        msg = ToolCallMessage("ask_user", {"questions": []})
        assert msg.tool_name == "ask_user"

    def test_toggle_args_no_op_before_mount(self) -> None:
        """Calling `toggle_args` before mount should not flip state."""
        msg = ToolCallMessage("ask_user", {"questions": [{"question": "?"}]})
        # Without `on_mount`, widget refs are None — `_update_args_display`
        # short-circuits and the expanded flag should not be flipped either,
        # since the user can't possibly see the result.
        msg.toggle_args()
        assert msg._args_expanded is True  # state flips
        # but rendering is a no-op:
        assert msg._args_widget is None

    async def test_toggle_args_swaps_display_state(self) -> None:
        """`toggle_args` should flip the args widget's display after mount."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage(
                    "ask_user",
                    {"questions": [{"question": "Name?", "type": "text"}]},
                )

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg

            # Initial state: hint visible, full args hidden.
            assert msg._args_widget is not None
            assert msg._args_hint_widget is not None
            assert msg._args_widget.display is False
            assert msg._args_hint_widget.display is True

            msg.toggle_args()
            await pilot.pause()
            assert msg._args_expanded is True
            assert msg._args_widget.display is True

            msg.toggle_args()
            await pilot.pause()
            assert msg._args_expanded is False
            assert msg._args_widget.display is False

    async def test_on_click_routes_ask_user_to_toggle_args(self) -> None:
        """Clicking an `ask_user` row (no output) should expand args."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage(
                    "ask_user",
                    {"questions": [{"question": "?"}]},
                )

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            event = MagicMock()
            msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            assert msg._args_expanded is True

    async def test_toggle_output_does_not_fall_through_to_args(self) -> None:
        """`toggle_output` is strictly about output; args stay collapsed."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage(
                    "ask_user",
                    {"questions": [{"question": "?"}]},
                )

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            msg.toggle_output()
            await pilot.pause()
            assert msg._args_expanded is False

    async def test_js_eval_click_toggles_code_when_result_unexpandable(self) -> None:
        """After a short `js_eval` result, clicking must toggle the code block.

        Regression: once eval returned, `_output` was set and `on_click`
        unconditionally routed to `toggle_output`. A short, unexpandable result
        made that a no-op, so the collapsible code block could never open.
        """
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage(
                    "js_eval",
                    {"code": "const x = 1;\nx + 1"},  # multi-line -> expandable
                )

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            # Eval returns a short, unexpandable result.
            msg.set_success("<result>2</result>")
            await pilot.pause()
            assert msg.has_output is True
            assert msg.has_expandable_output is False
            assert msg.has_expandable_args is True

            event = MagicMock()
            msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            # Falls through to the code block instead of no-op output toggle.
            assert msg._args_expanded is True

    async def test_js_eval_click_prefers_expandable_output(self) -> None:
        """When the result *is* expandable, clicking still toggles output."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage(
                    "js_eval",
                    {"code": "const x = 1;\nx + 1"},
                )

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            # A long multi-line stdout makes the output expandable.
            body = "\n".join(str(i) for i in range(50))
            msg.set_success(f"<stdout>\n{body}\n</stdout>\n<result>done</result>")
            await pilot.pause()
            assert msg.has_expandable_output is True

            event = MagicMock()
            msg.on_click(event)
            await pilot.pause()
            assert msg._expanded is True
            assert msg._args_expanded is False


class TestToolCallMessageTaskDescription:
    """Tests for the expandable, truncated `task` description line."""

    def test_short_description_not_expandable(self) -> None:
        """A description that fits is shown in full with no expand affordance."""
        msg = ToolCallMessage("task", {"description": "investigate the bug"})
        assert msg.has_expandable_task_desc is False

    def test_long_description_is_expandable(self) -> None:
        """A description longer than the limit becomes expandable."""
        long_desc = "x" * (ToolCallMessage._TASK_DESC_MAX_LENGTH + 1)
        msg = ToolCallMessage("task", {"description": long_desc})
        assert msg.has_expandable_task_desc is True

    def test_description_at_limit_not_expandable(self) -> None:
        """The threshold is strict `>`: a description of exactly the limit fits.

        Guards against a `>`-to-`>=` regression (or an off-by-one in the slice)
        that every `MAX + 1` test would still pass.
        """
        for length in (
            ToolCallMessage._TASK_DESC_MAX_LENGTH,
            ToolCallMessage._TASK_DESC_MAX_LENGTH - 1,
        ):
            msg = ToolCallMessage("task", {"description": "x" * length})
            assert msg.has_expandable_task_desc is False

    def test_non_task_not_expandable(self) -> None:
        """Only `task` rows expose an expandable description."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/x"})
        assert msg.has_expandable_task_desc is False

    def test_non_string_description_not_expandable(self) -> None:
        """A non-string `description` is coerced to empty, never raising.

        `has_expandable_task_desc` calls `len()` on the description, so dropping
        the `isinstance` guard would raise `TypeError` on these inputs.
        """
        for bad in (123, None, {"nested": "dict"}, ["list"]):
            msg = ToolCallMessage("task", {"description": bad})
            assert msg.has_expandable_task_desc is False

    def test_output_hint_drops_ctrl_o_when_description_expandable(self) -> None:
        """The output hint advertises click-only once the description owns Ctrl+O."""
        long_desc = "x" * (ToolCallMessage._TASK_DESC_MAX_LENGTH + 1)
        msg = ToolCallMessage("task", {"description": long_desc})
        assert msg.has_expandable_task_desc is True
        assert msg._output_hint_keys() == "click"

        short = ToolCallMessage("task", {"description": "short"})
        assert short.has_expandable_task_desc is False
        assert short._output_hint_keys() == "click or Ctrl+O"

    async def test_short_description_shows_widget_hides_hint(self) -> None:
        """A short but present description renders, with no expand hint."""
        app = _tool_msg_app("task", {"description": "investigate the bug"})
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            assert msg._task_desc_widget is not None
            assert msg._task_desc_hint_widget is not None
            assert msg._task_desc_widget.display is True
            assert msg._task_desc_hint_widget.display is False

    async def test_toggle_task_desc_swaps_display_state(self) -> None:
        """`toggle_task_desc` should reveal the full description then re-hide it."""
        from deepagents_code.config import get_glyphs

        long_desc = "word " * 60  # well over the truncation limit

        app = _tool_msg_app("task", {"description": long_desc})
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg

            assert msg._task_desc_widget is not None
            assert msg._task_desc_hint_widget is not None
            # Collapsed: hint reads "expand", description truncated to the limit
            # with a trailing ellipsis glyph.
            assert msg._task_desc_hint_widget.display is True
            hint = msg._task_desc_hint_widget._Static__content  # ty: ignore
            assert hint.plain == "click or Ctrl+O to expand"
            collapsed = msg._task_desc_widget._Static__content  # ty: ignore
            ellipsis = get_glyphs().ellipsis
            assert collapsed.plain.endswith(ellipsis)
            body = collapsed.plain[: -len(ellipsis)]
            assert len(body) <= ToolCallMessage._TASK_DESC_MAX_LENGTH
            assert long_desc.startswith(body)

            msg.toggle_task_desc()
            await pilot.pause()
            assert msg._task_desc_expanded is True
            expanded = msg._task_desc_widget._Static__content  # ty: ignore
            assert expanded.plain == long_desc
            hint = msg._task_desc_hint_widget._Static__content  # ty: ignore
            assert hint.plain == "click or Ctrl+O to collapse"

            msg.toggle_task_desc()
            await pilot.pause()
            assert msg._task_desc_expanded is False

    async def test_click_on_description_toggles_task_desc(self) -> None:
        """Clicking a truncated `task` row should expand its description."""
        app = _tool_msg_app("task", {"description": "word " * 60})
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            event = MagicMock()
            event.widget = msg._task_desc_widget
            msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            assert msg._task_desc_expanded is True

    async def test_click_on_header_toggles_task_desc(self) -> None:
        """Clicking the header of a truncated `task` row expands the description."""
        app = _tool_msg_app("task", {"description": "word " * 60})
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            event = MagicMock()
            event.widget = msg._header_widget
            msg.on_click(event)
            await pilot.pause()
            assert msg._task_desc_expanded is True

    async def test_click_on_output_toggles_output_not_description(self) -> None:
        """A click on the output region toggles output, leaving the desc alone.

        The load-bearing precedence rule: even when the description is
        expandable, a click that lands on the output routes to the output.
        """
        app = _tool_msg_app("task", {"description": "word " * 60})
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            msg.set_success("line\n" * 200)  # long, expandable output
            await pilot.pause()
            assert msg.has_expandable_task_desc is True
            assert msg.has_expandable_output is True

            event = MagicMock()
            event.widget = msg._preview_widget
            msg.on_click(event)
            await pilot.pause()
            assert msg._expanded is True
            assert msg._task_desc_expanded is False


class TestToolCallMessageExecuteCommandExpand:
    """Tests for the collapsible full-command block on `execute` tool calls."""

    def test_long_command_is_expandable(self) -> None:
        """A command too long for the header offers a collapsible block."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH
        msg = ToolCallMessage("execute", {"command": long_cmd})
        assert msg.has_expandable_args is True

    def test_short_command_not_expandable(self) -> None:
        """A command that fits in the header has nothing to expand."""
        short_cmd = "x" * (EXECUTE_HEADER_MAX_LENGTH - 1)
        msg = ToolCallMessage("execute", {"command": short_cmd})
        assert msg.has_expandable_args is False

    def test_missing_command_not_expandable(self) -> None:
        """An execute call without a command string is not expandable."""
        assert ToolCallMessage("execute", {}).has_expandable_args is False

    def test_command_detail_is_plain_and_left_aligned(self) -> None:
        """The command is plain `Content`, left-aligned, with blank padding."""
        command = "cd /tmp && \\\n  make build\nmake test"
        msg = ToolCallMessage("execute", {"command": command})
        detail = msg._format_command_detail()

        assert isinstance(detail, Content)
        assert not detail.spans
        assert detail.plain.split("\n") == [
            "",
            "cd /tmp && \\",
            "  make build",
            "make test",
            "",
        ]

    def test_command_detail_marks_hidden_unicode(self) -> None:
        """Hidden controls in the expanded command render as visible markers."""
        msg = ToolCallMessage("execute", {"command": "echo safe\n#\u202e hidden"})
        detail = msg._format_command_detail()

        assert "\u202e" not in detail.plain
        assert "<U+202E RIGHT-TO-LEFT OVERRIDE>" in detail.plain

    async def test_click_on_header_toggles_command(self) -> None:
        """Clicking the command header expands the collapsible command block."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": long_cmd})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            # Long stdout makes the output expandable too, so a generic click
            # would prefer output; a header click must still reach the command.
            msg.set_success("\n".join(str(i) for i in range(50)))
            await pilot.pause()
            assert msg.has_expandable_output is True

            event = MagicMock()
            event.widget = msg._header_widget
            msg.on_click(event)
            await pilot.pause()
            event.stop.assert_called_once()
            assert msg._args_expanded is True
            assert msg._expanded is False

    async def test_generic_click_still_prefers_output(self) -> None:
        """A click outside the header region toggles output, not the command."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": long_cmd})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            msg.set_success("\n".join(str(i) for i in range(50)))
            await pilot.pause()
            assert msg.has_expandable_output is True

            event = MagicMock()  # mock widget is outside the args region
            msg.on_click(event)
            await pilot.pause()
            assert msg._expanded is True
            assert msg._args_expanded is False

    @staticmethod
    def _harness(msg: ToolCallMessage) -> App[None]:
        """Build a minimal single-widget app hosting `msg`."""

        class _Harness(App[None]):
            def compose(self) -> ComposeResult:
                yield msg

        return _Harness()

    async def test_click_targets_args_region_walks_parents(self) -> None:
        """A click on a descendant of the header still routes to the command.

        A real Textual click reports the rendered-text node *inside* the
        `Static`, not the `Static` itself, so the region check must walk up the
        `.parent` chain rather than compare identities directly.
        """
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Direct hit, one hop, and two hops all resolve to the header.
            one_hop = SimpleNamespace(parent=msg._header_widget)
            two_hops = SimpleNamespace(parent=one_hop)
            assert msg._click_targets_args_region(msg._header_widget) is True
            assert msg._click_targets_args_region(one_hop) is True
            assert msg._click_targets_args_region(two_hops) is True
            # A detached node and `self` never match.
            assert msg._click_targets_args_region(SimpleNamespace(parent=None)) is False
            assert msg._click_targets_args_region(msg) is False

    async def test_click_on_args_hint_toggles_command(self) -> None:
        """Clicking the args-hint line expands the command, not the output."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH
        msg = ToolCallMessage("execute", {"command": long_cmd})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            msg.set_success("\n".join(str(i) for i in range(50)))
            await pilot.pause()
            assert msg.has_expandable_output is True

            event = MagicMock()
            event.widget = msg._args_hint_widget
            msg.on_click(event)
            await pilot.pause()
            assert msg._args_expanded is True
            assert msg._expanded is False

    async def test_click_on_expanded_command_collapses(self) -> None:
        """Clicking the expanded command block collapses it again."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH
        msg = ToolCallMessage("execute", {"command": long_cmd})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            msg.toggle_args()
            await pilot.pause()
            assert msg._args_expanded is True

            event = MagicMock()
            event.widget = msg._args_widget
            msg.on_click(event)
            await pilot.pause()
            assert msg._args_expanded is False

    async def test_expanded_command_uses_command_noun_and_detail(self) -> None:
        """Expanding wires the `command` noun and `_format_command_detail`."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH
        msg = ToolCallMessage("execute", {"command": long_cmd})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            msg.toggle_args()
            await pilot.pause()

            hint = msg._args_hint_widget._Static__content  # ty: ignore
            body = msg._args_widget._Static__content  # ty: ignore
            assert "hide command" in hint.plain
            assert body.plain == msg._format_command_detail().plain

    async def test_output_hint_omits_ctrl_o_when_command_expandable(self) -> None:
        """With an expandable command, Ctrl+O owns it, so the output hint drops it."""
        long_cmd = "echo " + "x" * EXECUTE_HEADER_MAX_LENGTH
        msg = ToolCallMessage("execute", {"command": long_cmd})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            msg.set_success("\n".join(str(i) for i in range(50)))
            await pilot.pause()
            assert msg.has_expandable_args is True

            hint = msg._hint_widget._Static__content  # ty: ignore
            assert "click to expand" in hint.plain
            assert "Ctrl+O" not in hint.plain

    async def test_output_hint_keeps_ctrl_o_without_expandable_command(self) -> None:
        """A short command leaves Ctrl+O on the output, so the hint keeps it."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        app = self._harness(msg)
        async with app.run_test() as pilot:
            await pilot.pause()
            msg.set_success("\n".join(str(i) for i in range(50)))
            await pilot.pause()
            assert msg.has_expandable_args is False

            hint = msg._hint_widget._Static__content  # ty: ignore
            assert "Ctrl+O" in hint.plain


class TestToolCallMessageShellCommand:
    """Test ToolCallMessage shows full shell command for errors.

    When a shell command fails, users need to see the full command to debug.
    The header is truncated for display, but the full command should be
    included in the error output for visibility.
    """

    def test_shell_error_includes_full_command(self) -> None:
        """Error output should include the full command that was executed."""
        long_cmd = "pip install " + " ".join(f"package{i}" for i in range(50))
        assert len(long_cmd) > 120  # Exceeds truncation limit

        msg = ToolCallMessage("execute", {"command": long_cmd})
        msg.set_error("Command not found: pip")

        # The error output should include the full command
        assert long_cmd in msg._output

    def test_shell_error_command_prefix(self) -> None:
        """Error output should have shell prompt prefix."""
        cmd = "echo hello"
        msg = ToolCallMessage("execute", {"command": cmd})
        msg.set_error("Permission denied")

        # Output should have shell prompt prefix
        assert msg._output.startswith("$ ")
        assert cmd in msg._output

    def test_non_shell_error_unchanged(self) -> None:
        """Non-shell tools should not have command prepended."""
        msg = ToolCallMessage("read_file", {"path": "/etc/passwd"})
        error = "Permission denied"
        msg.set_error(error)

        assert msg._output == error
        assert not msg._output.startswith("$ ")

    def test_shell_error_with_none_command(self) -> None:
        """Shell tool with None command should fall back to error-only output."""
        msg = ToolCallMessage("execute", {"command": None})
        error = "Some error"
        msg.set_error(error)

        assert "$ None" not in msg._output
        assert msg._output == error

    def test_shell_error_with_empty_command(self) -> None:
        """Shell tool with empty command should fall back to error-only output."""
        msg = ToolCallMessage("execute", {"command": ""})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error
        assert not msg._output.startswith("$ ")

    def test_shell_error_with_whitespace_command(self) -> None:
        """Shell tool with whitespace command should fall back to error-only output."""
        msg = ToolCallMessage("execute", {"command": "   "})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error

    def test_shell_error_with_no_command_key(self) -> None:
        """Shell tool with no command key should fall back to error-only output."""
        msg = ToolCallMessage("execute", {"other_arg": "value"})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error
        assert not msg._output.startswith("$ ")

    def test_format_shell_output_styles_only_first_line_dim(self) -> None:
        """Shell output formatting should only style the first command line in dim."""
        msg = ToolCallMessage("execute", {"command": "echo test"})
        output = "$ echo test\ntest output\n$ not a command"
        result = msg._format_shell_output(output, is_preview=False)

        assert isinstance(result.content, Content)
        lines = result.content.split("\n")
        # First line (the command) should be styled dim
        assert lines[0].plain == "$ echo test"
        assert "dim" in lines[0].markup
        # Subsequent lines should NOT be dim
        assert lines[2].plain == "$ not a command"
        assert "dim" not in lines[2].markup

    def test_format_shell_output_preview_truncates_long_single_line(self) -> None:
        """Preview should char-truncate single-line output past the budget."""
        msg = ToolCallMessage("execute", {"command": "gh api graphql"})
        # One huge JSON-like line, well past _PREVIEW_CHARS (400).
        output = "x" * 5000
        result = msg._format_shell_output(output, is_preview=True)

        assert result.truncation is not None
        assert "more chars" in result.truncation
        assert len(result.content.plain) <= msg._PREVIEW_CHARS

    def test_format_shell_output_preview_short_no_truncation(self) -> None:
        """Short shell output should not report any truncation in preview."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        output = "$ echo hi\nhi"
        result = msg._format_shell_output(output, is_preview=True)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_shell_output_preview_cumulative_chars_exceed_budget(self) -> None:
        """Many small lines whose total exceeds the budget should char-truncate.

        Char budget is hit, but some lines weren't even attempted — hidden line
        count is the more useful signal than hidden char count.
        """
        msg = ToolCallMessage("execute", {"command": "noisy"})
        # 4 lines of 200 chars => 800 + 3 separators, well past 400.
        output = "\n".join("x" * 200 for _ in range(4))
        result = msg._format_shell_output(output, is_preview=True)

        assert result.truncation is not None
        assert "more lines" in result.truncation
        # Rendered content stays under budget.
        assert len(result.content.plain) <= msg._PREVIEW_CHARS

    def test_format_shell_output_preview_preserves_dim_when_first_line_clipped(
        self,
    ) -> None:
        """Char-clipping line 0 must keep the `$ ` prefix dim styling."""
        msg = ToolCallMessage("execute", {"command": "echo"})
        output = "$ " + ("x" * 5000)
        result = msg._format_shell_output(output, is_preview=True)

        first_line = result.content.split("\n")[0]
        assert first_line.plain.startswith("$ ")
        assert "dim" in first_line.markup

    def test_format_shell_output_full_never_truncates(self) -> None:
        """`is_preview=False` must render full output regardless of size."""
        msg = ToolCallMessage("execute", {"command": "big"})
        output = "x" * 5000
        result = msg._format_shell_output(output, is_preview=False)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_output_preserves_first_line_leading_indent(self) -> None:
        """`_format_output` must keep the first line's own leading indentation.

        A bare `strip()` lstrips only the first line while continuation lines
        keep their indent, so uniformly indented command output (e.g.
        `git branch -r`, which prefixes every branch with two spaces) renders
        misaligned. All rows should retain their leading spaces.
        """
        msg = ToolCallMessage("execute", {"command": "git branch -r"})
        # Mirror `git branch -r`: every row indented by two spaces, trailing \n.
        output = "  origin/HEAD -> origin/main\n  origin/main\n  origin/dev\n"
        result = msg._format_output(output, is_preview=False)

        lines = result.content.plain.split("\n")
        assert lines == [
            "  origin/HEAD -> origin/main",
            "  origin/main",
            "  origin/dev",
        ]
        # Every line shares the same leading indent, so they align beside the
        # fixed glyph gutter.
        assert all(line.startswith("  ") for line in lines)

    def test_format_output_still_trims_leading_blank_lines(self) -> None:
        """Leading blank lines are trimmed while first-line indent survives."""
        msg = ToolCallMessage("execute", {"command": "noop"})
        result = msg._format_output("\n\n  indented\n", is_preview=False)

        assert result.content.plain == "  indented"


class TestToolCallMessageJsEvalOutput:
    """Tests for `_format_js_eval_output`.

    The `js_eval` REPL tool returns an XML-ish envelope
    (`<stdout>`, `<result>`, `<error>`) with `&`, `<`, `>` escaped. The
    formatter unwraps that into labeled, styled sections instead of dumping the
    raw blob.
    """

    def test_format_single_scalar_result_renders_inline(self) -> None:
        """A lone short scalar result renders inline as `result: value`."""
        msg = ToolCallMessage("js_eval", {"code": "1 + 1"})
        result = msg._format_output("<result>2</result>", is_preview=False)

        assert result.content.plain == "result: 2"
        assert result.truncation is None

    def test_format_empty_string_result_stays_empty(self) -> None:
        """An empty string result must not be rewritten as `undefined`."""
        msg = ToolCallMessage("js_eval", {"code": "''"})
        result = msg._format_output("<result></result>", is_preview=False)

        assert result.content.plain == "result: "

    def test_format_newline_only_result_preserves_body(self) -> None:
        """A newline-only string result remains a real value in block form."""
        msg = ToolCallMessage("js_eval", {"code": "'\\n'"})
        result = msg._format_output("<result>\n</result>", is_preview=False)

        assert result.content.plain.split("\n") == ["result", "  ", "  "]

    def test_format_multiline_result_uses_block(self) -> None:
        """A multi-line result keeps the labeled-block layout."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        result = msg._format_output("<result>line1\nline2</result>", is_preview=False)

        assert result.content.plain.split("\n") == ["result", "  line1", "  line2"]

    def test_format_long_scalar_result_uses_block(self) -> None:
        """A long single-line result is not collapsed inline."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        body = "x" * (msg._JS_EVAL_INLINE_RESULT_MAX + 1)
        result = msg._format_output(f"<result>{body}</result>", is_preview=False)

        assert result.content.plain.split("\n") == ["result", f"  {body}"]

    def test_format_stdout_and_result(self) -> None:
        """Stdout and result both render as separate labeled sections."""
        msg = ToolCallMessage("js_eval", {"code": "console.log('hi'); 42"})
        output = "<stdout>\nhi\n</stdout>\n<result>42</result>"
        result = msg._format_output(output, is_preview=False)

        # stdout present -> result is not collapsed inline.
        lines = result.content.plain.split("\n")
        assert lines == ["stdout", "  hi", "result", "  42"]

    def test_format_unescapes_xml_entities(self) -> None:
        """Escaped `<`, `>`, `&` in the body are restored for display."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        output = "<result>&lt;div&gt; &amp;&amp; true</result>"
        result = msg._format_output(output, is_preview=False)

        # Single short scalar -> inline form.
        assert result.content.plain == "result: <div> && true"

    def test_format_error_block_includes_type(self) -> None:
        """An error block surfaces the error type in its label."""
        msg = ToolCallMessage("js_eval", {"code": "boom()"})
        output = '<error type="ReferenceError">boom is not defined</error>'
        result = msg._format_output(output, is_preview=False)

        lines = result.content.plain.split("\n")
        assert lines == ["error (ReferenceError)", "  boom is not defined"]

    def test_format_handle_result_labeled(self) -> None:
        """A `kind`-tagged result is labeled as a handle."""
        msg = ToolCallMessage("js_eval", {"code": "() => 1"})
        output = '<result kind="handle">[Function] arity=0</result>'
        result = msg._format_output(output, is_preview=False)

        lines = result.content.plain.split("\n")
        assert lines == ["result (handle)", "  [Function] arity=0"]

    def test_format_preview_truncates_long_output(self) -> None:
        """Preview mode caps lines and reports the count of hidden lines."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        body = "\n".join(str(i) for i in range(50))
        output = f"<stdout>\n{body}\n</stdout>\n<result>done</result>"
        result = msg._format_output(output, is_preview=True)

        shown = len(result.content.plain.split("\n"))
        assert shown <= msg._PREVIEW_LINES
        # Full render is the stdout label + 50 stdout lines + result label + 1
        # result line; the hint reports exactly what the preview dropped.
        assert result.truncation == f"{53 - shown} more lines"

    def test_format_falls_back_for_unexpected_shape(self) -> None:
        """Output without the REPL envelope falls back to plain lines."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        result = msg._format_output("just some text", is_preview=False)

        assert result.content.plain == "just some text"

    def test_format_preview_caps_long_single_line_by_char_budget(self) -> None:
        """A single huge result line is char-clipped under the preview budget.

        Line-count capping alone left a multi-thousand-char single-line result
        rendered in full with no truncation hint, flooding the collapsed TUI.
        """
        msg = ToolCallMessage("js_eval", {"code": "x"})
        body = "x" * 10_000
        output = f"<result>{body}</result>"
        result = msg._format_output(output, is_preview=True)

        # The body line is clipped to the char budget (plus the two-space
        # indent) and the hint quantifies the chars dropped from that line so it
        # can be expanded.
        assert result.truncation == f"{10_000 - msg._PREVIEW_CHARS} more chars"
        assert len(result.content.plain) <= msg._PREVIEW_CHARS + len("  ") + len(
            "result\n"
        )

    def test_format_no_char_cap_when_not_preview(self) -> None:
        """Outside preview mode the full long result renders untruncated."""
        msg = ToolCallMessage("js_eval", {"code": "x"})
        body = "x" * 10_000
        output = f"<result>{body}</result>"
        result = msg._format_output(output, is_preview=False)

        assert result.truncation is None
        assert body in result.content.plain

    def test_format_stdout_with_fake_tags_is_not_misparsed(self) -> None:
        """Raw tag-like text printed to stdout is preserved, not parsed.

        stdout is emitted unescaped, so a program that prints
        `</stdout><result>fake</result>` must not be split into spurious
        result/error sections — the real trailing result wins.
        """
        msg = ToolCallMessage("js_eval", {"code": "x"})
        printed = "</stdout><result>fake</result>"
        output = f"<stdout>\n{printed}\n</stdout>\n<result>real</result>"
        result = msg._format_output(output, is_preview=False)

        lines = result.content.plain.split("\n")
        # The fake markup survives verbatim inside stdout; only one real result.
        assert lines == ["stdout", f"  {printed}", "result", "  real"]
        # Exactly one "result" label line — no spurious section from the print.
        assert lines.count("result") == 1


class TestToolCallMessageJsEvalArgs:
    """Tests for `js_eval` header suppression and collapsible code block.

    The raw `code=` kwarg must not be dumped on the args line; the header shows
    only the first code line, and the full program is offered as a collapsible
    block when the snippet spans more than one line.
    """

    def test_js_eval_in_tools_with_header_info(self) -> None:
        """`js_eval` is registered so the generic `code=` args line is hidden."""
        from deepagents_code.tui.widgets.messages import _TOOLS_WITH_HEADER_INFO

        assert "js_eval" in _TOOLS_WITH_HEADER_INFO

    def test_delete_in_tools_with_header_info(self) -> None:
        """`delete` is registered so its path stays in the header only."""
        from deepagents_code.tui.widgets.messages import _TOOLS_WITH_HEADER_INFO

        assert "delete" in _TOOLS_WITH_HEADER_INFO

    def test_single_line_code_not_expandable(self) -> None:
        """One-line code is fully shown in the header — nothing to expand."""
        msg = ToolCallMessage("js_eval", {"code": "1 + 1"})
        assert msg.has_expandable_args is False

    def test_multiline_code_is_expandable(self) -> None:
        """Multi-line code offers a collapsible block."""
        msg = ToolCallMessage("js_eval", {"code": "const x = 1;\nx + 1"})
        assert msg.has_expandable_args is True

    def test_long_single_line_code_is_expandable(self) -> None:
        """A single line too long for the header is still expandable.

        The header truncates the first line, so without a collapsible block a
        long one-liner (e.g. minified JS) would be unrecoverable in the TUI.
        """
        long_line = "x".ljust(JS_EVAL_HEADER_MAX_LENGTH + 1, "y")
        msg = ToolCallMessage("js_eval", {"code": long_line})
        assert msg.has_expandable_args is True

    def test_short_single_line_code_not_expandable(self) -> None:
        """A single line that fits in the header has nothing to expand."""
        short_line = "x" * (JS_EVAL_HEADER_MAX_LENGTH - 1)
        msg = ToolCallMessage("js_eval", {"code": short_line})
        assert msg.has_expandable_args is False

    def test_code_detail_is_plain_and_left_aligned(self) -> None:
        """The code is plain `Content`, left-aligned, with blank padding lines."""
        code = "const x = 1;\n  nested();\nx + 1"
        msg = ToolCallMessage("js_eval", {"code": code})
        detail = msg._format_code_detail()

        from textual.content import Content

        assert isinstance(detail, Content)
        # Blank padding lines top and bottom; code's own indentation is
        # preserved and no extra indent is injected.
        assert detail.plain.split("\n") == [
            "",
            "const x = 1;",
            "  nested();",
            "x + 1",
            "",
        ]

    def test_code_detail_is_unstyled(self) -> None:
        """No syntax highlighting: the rendered code carries no style spans."""
        msg = ToolCallMessage("js_eval", {"code": "const x = 1;\nx + 1"})
        detail = msg._format_code_detail()

        assert not detail.spans

    def test_code_detail_strips_surrounding_blank_lines(self) -> None:
        """Code's own surrounding blanks are trimmed (padding lines remain)."""
        msg = ToolCallMessage("js_eval", {"code": "\n\nconst x = 1;\n\n"})
        detail = msg._format_code_detail()

        # One blank padding line top and bottom, around the trimmed code.
        assert detail.plain == "\nconst x = 1;\n"

    def test_code_detail_marks_hidden_unicode(self) -> None:
        """Hidden controls in expanded code are rendered as visible markers."""
        msg = ToolCallMessage("js_eval", {"code": "const safe = 1;\n//\u202e hidden"})
        detail = msg._format_code_detail()

        assert "\u202e" not in detail.plain
        assert "<U+202E RIGHT-TO-LEFT OVERRIDE>" in detail.plain


class TestToolCallMessageFileOutput:
    """Tests for `_format_file_output` char-budget handling.

    Files with very long lines (minified HTML/JS/CSS) used to overflow the
    preview because only line count was capped. Preview now caps both, and
    prefers line counts over char counts in the truncation hint when both
    were hit.
    """

    def test_format_file_output_preview_truncates_long_single_line(self) -> None:
        """A single huge line must be char-clipped under the preview budget.

        Single-line input: no lines hidden, so the hint reports remaining chars.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/big.html"})
        output = "x" * 5000
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation == f"{5000 - msg._PREVIEW_CHARS} more chars"
        assert len(result.content.plain) <= msg._PREVIEW_CHARS

    def test_format_file_output_preview_cumulative_chars_exceed_budget(self) -> None:
        """Within the 4-line cap, total chars past budget prefers `more lines`.

        Some lines weren't even attempted — line count is more useful than
        char count when the line cap also kicked in.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/big.html"})
        # 4 x 200-char lines: line 0 fits (200), line 1 clips (199), lines 2-3
        # are never attempted, so 2 lines are hidden.
        output = "\n".join("x" * 200 for _ in range(4))
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation == "2 more lines"
        assert len(result.content.plain) <= msg._PREVIEW_CHARS

    def test_format_file_output_preview_line_truncation_when_under_char_budget(
        self,
    ) -> None:
        """Many short lines should report `more lines` truncation."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "\n".join(f"line {i}" for i in range(20))
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation == "16 more lines"

    def test_format_file_output_preview_short_no_truncation(self) -> None:
        """Short file content should render fully with no truncation hint."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "hello\nworld"
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_file_output_full_never_truncates(self) -> None:
        """`is_preview=False` must render full output regardless of size."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/big.html"})
        output = "x" * 5000
        result = msg._format_file_output(output, is_preview=False)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_file_output_preview_exact_budget_boundary(self) -> None:
        """A single line that exactly fills the budget should not truncate."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "x" * msg._PREVIEW_CHARS
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_file_output_preview_trailing_newline_at_budget(self) -> None:
        r"""Trailing newline at exact budget shouldn't produce a phantom hint.

        File content fits in the budget exactly; the trailing `\n` is a
        text-file convention, not real hidden content.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "x" * msg._PREVIEW_CHARS + "\n"
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation is None

    def test_format_file_output_preview_trailing_newline_short_file(self) -> None:
        r"""Short file ending in `\n` should not report a phantom extra line."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "hello\nworld\n"
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation is None
        assert result.content.plain == "hello\nworld"

    def test_format_file_output_preview_empty_output(self) -> None:
        """Empty output should produce empty content with no truncation hint."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/empty"})
        result = msg._format_file_output("", is_preview=True)

        assert result.truncation is None
        assert result.content.plain == ""

    def test_format_file_output_preview_exactly_four_short_lines(self) -> None:
        """Exactly 4 short lines should render fully with no truncation."""
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "\n".join(f"line {i}" for i in range(4))
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation is None
        assert result.content.plain == output

    def test_format_file_output_preview_budget_hit_on_separator(self) -> None:
        """Separator-cost path must trigger truncation when line 0 fills budget.

        When line 0 exactly fills the budget, the next line's separator
        triggers the `remaining <= 0` branch (distinct from the
        `len(line) > remaining` branch). Line count should be reported since
        lines were hidden.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "x" * msg._PREVIEW_CHARS + "\nsecond\nthird"
        result = msg._format_file_output(output, is_preview=True)

        assert result.truncation == "2 more lines"

    def test_format_output_compacts_legacy_cat_n_gutter(self) -> None:
        r"""Legacy `cat -n` gutters are tightened, all rows aligned to one column.

        deepagents versions predating #4561 emitted `f"{line_num:6d}\t{line}"` —
        a 6-wide right-justified number plus a tab — which renders far from the
        line numbers and (when the first row's padding was stripped) misaligned.
        Such output may still surface from cached or persisted transcripts. The
        TUI recomputes a compact gutter: numbers right-justified to the widest
        number present, two spaces, then the original source indentation.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        # cat -n style: 6-wide right-justified number + tab + source line.
        output = '     1\t"""doc"""\n     2\t\n     3\t    indented'
        result = msg._format_output(output, is_preview=False)

        # No tab, no 6-wide pad: `{num}  ` gutter, then the original source
        # indentation (the 4 spaces on line 3) preserved verbatim.
        assert result.content.plain == '1  """doc"""\n2  \n3      indented'

    def test_format_output_leaves_current_two_space_gutter_intact(self) -> None:
        r"""The current SDK gutter is already compact, so compaction is a no-op.

        `read_file` now emits `f"{marker:>width}  {line}"` — a right-justified
        marker, two spaces, then source. Re-justifying to the widest marker
        reproduces the same string, and crucially a tab-indented source line
        keeps its leading tab (the two-space separator, not the source tab, is
        consumed). Regression guard: the old tab-splitting gutter dropped that
        indentation, re-introducing the ambiguity this format fixes.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        # Two-digit max marker => width 2; line 2 is tab-indented source.
        output = " 9  def build_config():\n10  \treturn {}"
        result = msg._format_output(output, is_preview=False)

        assert result.content.plain == " 9  def build_config():\n10  \treturn {}"

    def test_compact_line_gutter_right_justifies_to_widest_number(self) -> None:
        r"""Multi-digit line numbers set a uniform, right-justified gutter."""
        # Lines 9 and 10: single- vs double-digit numbers must align right.
        output = "     9\tnine\n    10\tten"
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == " 9  nine\n10  ten"

    def test_compact_line_gutter_handles_continuation_markers(self) -> None:
        r"""`N.M` wrapped-line markers are gutters and drive the column width.

        Long lines are chunked by the SDK with decimal continuation markers
        (`f"{line_num}.{chunk_idx}"`). The marker's width (e.g. `1.1` = 3)
        must set the right-justified column like any other line number.
        """
        output = "     1\tfirst\n   1.1\twrapped"
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == "  1  first\n1.1  wrapped"

    def test_compact_line_gutter_preserves_source_tabs_legacy(self) -> None:
        r"""Only the gutter tab is consumed; a legacy row's source tab stays put.

        Tab-indented source means a tab immediately after the gutter tab. The
        gutter regex consumes only the separator, so the source tab survives.
        """
        output = "     1\t\tdef foo():"
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == "1  \tdef foo():"

    def test_compact_line_gutter_preserves_source_tabs_current(self) -> None:
        r"""A current-format row's leading source tab (and a blank row) survive.

        The current gutter separator is two spaces, so a tab-indented source
        line reads as `"N  \tsource"`. Only the two spaces are consumed; the
        source tab must remain. This is the regression guard for the tab-
        splitting gutter that used to drop it. A blank source line (`"N  "` —
        marker, separator, empty source) round-trips unchanged too; the
        `_compact_line_gutter` return preserves its trailing separator (the
        display path may later strip trailing space, but the compactor does not).
        """
        output = "1  def foo():\n2  \treturn 1\n3  "
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == "1  def foo():\n2  \treturn 1\n3  "

    def test_compact_line_gutter_parses_real_producer_output(self) -> None:
        r"""Round-trip guard against producer/consumer separator drift.

        Feeds real `format_content_with_line_numbers` output (the authoritative
        producer, in the deepagents package) through the TUI parser. If the
        producer separator ever changes without this parser following, the exact
        assertion fails in CI instead of the gutter silently failing to compact
        (or, on a widened separator, leaking a phantom space into every source
        line). Line 2 is tab-indented source — the ambiguity this format fixes.
        """
        output = format_content_with_line_numbers(
            ["def f():", "\treturn 1"], start_line=1
        )
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == "1  def f():\n2  \treturn 1"

    def test_compact_line_gutter_round_trips_continuation_and_padding(self) -> None:
        r"""Real-producer round-trip exercising continuation + multi-digit padding.

        The base round-trip test feeds short, unpadded lines. This one forces a
        wrapped line (an `N.M` continuation marker) and a two-digit line number,
        so the continuation marker drives the column width and the shorter
        markers get leading-space padding — all through the real producer, not a
        hand-built string. Line 3 is tab-indented source, which must survive.
        Extends drift protection to the continuation and padding paths, not just
        the base separator.
        """
        long_line = "x" * (MAX_LINE_LENGTH + 5)  # forces an `N.1` continuation
        output = format_content_with_line_numbers(
            ["short", long_line, "\treturn 1"], start_line=9
        )
        compacted = ToolCallMessage._compact_line_gutter(output)

        lines = compacted.split("\n")
        assert lines[0] == "   9  short"  # width 4, driven by the "10.1" marker
        assert lines[2].startswith("10.1  ")
        assert lines[-1] == "  11  \treturn 1"  # tab-indented source survives

    def test_compact_line_gutter_preserves_double_spaced_source(self) -> None:
        r"""Only the first separator is consumed; the rest of the source is verbatim.

        A source line whose own text starts with digits and two spaces
        (`"42  meaning"`) must survive intact: the regex captures the leading
        gutter marker and emits everything after the first two-space separator
        untouched, including the embedded double space. Guards the `(.*)` capture
        against a future separator group that might reprocess or collapse
        interior spacing (e.g. widening `(?:  |\t)` to `\s+`).
        """
        # width 2 (max marker "10"); row 5's source begins with digits + 2 spaces.
        output = "5  42  meaning\n10  ok"
        compacted = ToolCallMessage._compact_line_gutter(output)

        assert compacted == " 5  42  meaning\n10  ok"

    def test_compact_line_gutter_passes_through_non_numbered(self) -> None:
        """Output without a gutter is returned unchanged."""
        output = "plain text\nno line numbers here"
        assert ToolCallMessage._compact_line_gutter(output) == output

    def test_compact_line_gutter_rejects_malformed_number_heads(self) -> None:
        r"""Heads that aren't a bare `N`/`N.M` are treated as source, not gutter.

        Guards against corrupting data whose first column merely resembles a
        number (leading/trailing dot, multiple dots) — in both the legacy tab
        and current two-space forms, since a malformed head fails the marker
        pattern before the separator alternation is ever reached.
        """
        # Leading dot, trailing dot, and multi-dot heads must all pass through,
        # whichever separator follows. All lines are non-gutter, so nothing
        # matches and the whole output is returned verbatim.
        output = (
            "   .5\tweird\n   5.\talso\n 1.2.3\tnope\n"
            ".5  two-space\n5.  two-space\n1.2.3  two-space"
        )
        assert ToolCallMessage._compact_line_gutter(output) == output

    def test_compact_line_gutter_preview_truncates_with_compacted_gutters(
        self,
    ) -> None:
        """Compaction runs before truncation: previews show compact gutters.

        The char budget and `more lines` hint operate on the already-compacted
        string, so a long cat -n file previews with tight gutters and a
        line-count hint.
        """
        msg = ToolCallMessage("read_file", {"path": "/tmp/a.py"})
        output = "\n".join(f"{i:6d}\tline {i}" for i in range(1, 21))
        result = msg._format_file_output(output, is_preview=True)

        rendered = result.content.plain.split("\n")
        assert rendered[0] == " 1  line 1"  # width 2 (max line number is 20)
        assert result.truncation == "16 more lines"

    def test_compact_line_gutter_empty_output(self) -> None:
        """Empty output has no gutter lines and is returned unchanged."""
        assert ToolCallMessage._compact_line_gutter("") == ""


class TestToolCallMessageAwaitingApproval:
    """Tests for `set_awaiting_approval` / `clear_awaiting_approval`."""

    def test_set_awaiting_approval_hides_widget(self) -> None:
        """`set_awaiting_approval` should mark the widget as hidden."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        assert msg._awaiting_approval is False
        msg.set_awaiting_approval()
        assert msg._awaiting_approval is True
        assert msg.display is False

    def test_clear_awaiting_approval_restores_widget(self) -> None:
        """`clear_awaiting_approval` should restore visibility."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        msg.set_awaiting_approval()
        msg.clear_awaiting_approval()
        assert msg._awaiting_approval is False
        assert msg.display is True

    def test_clear_awaiting_approval_no_op_when_not_set(self) -> None:
        """Clearing before setting should not touch widget visibility."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        msg.clear_awaiting_approval()
        assert msg._awaiting_approval is False

    async def test_awaiting_approval_round_trip_in_mounted_widget(self) -> None:
        """Mounted widget should hide on set, reappear on clear."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": "echo hi"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            assert msg.display is True
            msg.set_awaiting_approval()
            await pilot.pause()
            assert msg.display is False
            msg.clear_awaiting_approval()
            await pilot.pause()
            assert msg.display is True


class TestToolCallMessageRunningSpinner:
    """Tests for `set_running` / `pause_running` spinner state."""

    async def test_set_running_shows_status_widget(self) -> None:
        """`set_running` should reveal the status row and start the timer."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("grep", {"pattern": "foo"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            assert msg._status_widget is not None
            # Pending tools hide the status row until they run.
            assert msg._status_widget.display is False

            msg.set_running()
            await pilot.pause()
            assert msg._status == "running"
            assert msg._status_widget.display is True
            assert msg._animation_timer is not None

    async def test_running_timer_hidden_before_threshold(self) -> None:
        """The elapsed counter stays hidden until the threshold elapses."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("grep", {"pattern": "foo"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            assert msg._status_widget is not None

            msg.set_running()
            await pilot.pause()

            threshold = msg._RUNNING_TIMER_THRESHOLD_SECS

            # `_update_running_animation` recomputes `int(time() - _start_time)`,
            # so each offset below lands on a whole second with >0.99s of slack
            # (the truncated sub-second delta between the two `time()` reads
            # would need a full-second stall to flip) — the assertions are
            # deterministic, not timing-dependent.

            # Just under the threshold: status ends at "Running..." with no
            # trailing elapsed counter. We assert on the suffix rather than
            # exact equality or an `"(" in ...` search because the leading
            # spinner frame may itself contain parens on ASCII terminals.
            msg._start_time = time() - (threshold - 1)
            msg._update_running_animation()
            await pilot.pause()
            assert str(msg._status_widget.render()).endswith("Running...")

            # Exactly at the threshold: the elapsed counter appears.
            msg._start_time = time() - threshold
            msg._update_running_animation()
            await pilot.pause()
            assert str(msg._status_widget.render()).endswith(
                f"Running... ({format_duration(threshold)})"
            )

            # Well past the threshold: the counter keeps updating (guards
            # against a `>=`-to-`==` regression that would show the timer only
            # on the exact threshold second and then hide it again).
            beyond = threshold + 5
            msg._start_time = time() - beyond
            msg._update_running_animation()
            await pilot.pause()
            assert str(msg._status_widget.render()).endswith(
                f"Running... ({format_duration(beyond)})"
            )

    async def test_pause_running_hides_status_and_stops_timer(self) -> None:
        """`pause_running` should revert a running tool to its pending look."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("grep", {"pattern": "foo"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            msg.set_running()
            await pilot.pause()

            msg.pause_running()
            await pilot.pause()
            assert msg._status == "pending"
            assert msg._start_time is None
            assert msg._animation_timer is None
            assert msg._status_widget is not None
            assert msg._status_widget.display is False

    async def test_pause_running_no_op_when_not_running(self) -> None:
        """Pausing a pending tool should leave its status untouched."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("grep", {"pattern": "foo"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            assert msg._status == "pending"
            msg.pause_running()
            await pilot.pause()
            assert msg._status == "pending"
            assert msg._status_widget is not None
            assert msg._status_widget.display is False

    async def test_set_running_resumes_after_pause(self) -> None:
        """A paused tool should be resumable via `set_running` (HITL approve)."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("write_file", {"file_path": "a.txt"})

            def compose(self) -> ComposeResult:
                yield self.msg

        app = _Harness()
        async with app.run_test() as pilot:
            await pilot.pause()
            msg = app.msg
            msg.set_running()
            await pilot.pause()
            msg.pause_running()
            await pilot.pause()
            assert msg._status == "pending"

            msg.set_running()
            await pilot.pause()
            assert msg._status == "running"
            assert msg._start_time is not None
            assert msg._animation_timer is not None
            assert msg._status_widget is not None
            assert msg._status_widget.display is True


class TestToolCallMessageRejectReason:
    """Tests for surfacing a user-supplied HITL reject reason."""

    async def test_set_rejected_with_reason_renders_line(self) -> None:
        """`set_rejected(reason=...)` should display the reason beneath the status."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": "echo hi"})

            def compose(self) -> ComposeResult:
                yield self.msg

        async with _Harness().run_test() as pilot:
            await pilot.pause()
            app = pilot.app
            assert isinstance(app, _Harness)
            msg = app.msg
            msg.set_rejected(reason="please dry-run first")
            await pilot.pause()
            assert msg._reject_reason == "please dry-run first"
            assert msg._reject_reason_widget is not None
            assert msg._reject_reason_widget.display is True

    async def test_set_rejected_without_reason_hides_line(self) -> None:
        """`set_rejected()` with no reason keeps the reason line hidden."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": "echo hi"})

            def compose(self) -> ComposeResult:
                yield self.msg

        async with _Harness().run_test() as pilot:
            await pilot.pause()
            app = pilot.app
            assert isinstance(app, _Harness)
            msg = app.msg
            msg.set_rejected()
            await pilot.pause()
            assert msg._reject_reason is None
            assert msg._reject_reason_widget is not None
            assert msg._reject_reason_widget.display is False

    async def test_blank_reason_does_not_set_attribute(self) -> None:
        """Whitespace-only reasons are treated as no reason."""
        from textual.app import App, ComposeResult

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": "echo hi"})

            def compose(self) -> ComposeResult:
                yield self.msg

        async with _Harness().run_test() as pilot:
            await pilot.pause()
            app = pilot.app
            assert isinstance(app, _Harness)
            msg = app.msg
            msg.set_rejected(reason="   ")
            await pilot.pause()
            assert msg._reject_reason is None

    async def test_reason_with_markup_brackets_renders_safely(self) -> None:
        """User-controlled reasons must round-trip through Rich markup unscathed.

        `from_markup` with `$reason` substitution should escape any literal
        bracket sequences so the reason line never throws a MarkupError.
        """
        from textual.app import App, ComposeResult

        hostile = "[bold red]boom[/bold red] [/dim] $x"

        class _Harness(App[None]):
            def __init__(self) -> None:
                super().__init__()
                self.msg = ToolCallMessage("execute", {"command": "echo hi"})

            def compose(self) -> ComposeResult:
                yield self.msg

        async with _Harness().run_test() as pilot:
            await pilot.pause()
            app = pilot.app
            assert isinstance(app, _Harness)
            msg = app.msg
            msg.set_rejected(reason=hostile)
            await pilot.pause()
            assert msg._reject_reason == hostile
            assert msg._reject_reason_widget is not None
            assert msg._reject_reason_widget.display is True
            rendered = str(msg._reject_reason_widget.render())
            assert "boom" in rendered
            assert "$x" in rendered


class TestUserMessageHighlighting:
    """Test UserMessage highlighting of `@mentions` and `/commands`."""

    def test_at_mention_highlighted(self) -> None:
        """`@file` mentions should be styled in the output."""
        content = "look at @README.md please"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "@README.md"

    def test_slash_command_highlighted_at_start(self) -> None:
        """Slash commands at start should be detected."""
        content = "/help me with something"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "/help"
        assert matches[0].start() == 0

    def test_slash_command_not_matched_mid_text(self) -> None:
        """Slash in middle of text should not match as command due to ^ anchor."""
        content = "check the /usr/bin path"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        # The ^ anchor means /usr doesn't match when not at start of string
        assert len(matches) == 0

    def test_multiple_at_mentions(self) -> None:
        """Multiple `@mentions` should all be detected."""
        content = "compare @file1.py with @file2.py"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 2
        assert matches[0].group() == "@file1.py"
        assert matches[1].group() == "@file2.py"

    def test_at_mention_with_path(self) -> None:
        """`@mentions` with paths should be fully captured."""
        content = "read @src/utils/helpers.py"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "@src/utils/helpers.py"

    def test_no_matches_in_plain_text(self) -> None:
        """Plain text without `@` or `/` should have no matches."""
        content = "just some normal text here"
        matches = list(INPUT_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 0


def _render_content(widget: UserMessage | QueuedUserMessage) -> Content:
    """Extract the `Content` object from a message widget's render method."""
    result = widget.render()
    assert isinstance(result, Content)
    return result


class TestUserMessageModeRendering:
    """Test `UserMessage` renders mode-specific prefix indicators and colors.

    Without an active Textual app, `get_theme_colors` falls back to
    `DARK_COLORS`, so assertions check for hex values from that palette.
    """

    def test_shell_prefix_renders_dollar_indicator(self) -> None:
        """`UserMessage('!ls')` should render with `'$ '` prefix and shell body."""
        content = _render_content(UserMessage("!ls"))
        assert content.plain == "$ ls"
        first_span = content._spans[0]
        assert theme.DARK_COLORS.mode_bash in str(first_span.style)

    def test_incognito_shell_prefix_renders_dollar_indicator(self) -> None:
        """`UserMessage('!!ls')` should strip the full incognito prefix."""
        content = _render_content(UserMessage("!!ls"))
        assert content.plain == "$ ls"
        first_span = content._spans[0]
        assert theme.DARK_COLORS.mode_incognito in str(first_span.style)

    def test_command_prefix_renders_slash_indicator(self) -> None:
        """`UserMessage('/help')` should render with `'/ '` prefix and body."""
        content = _render_content(UserMessage("/help"))
        assert content.plain == "/ help"
        first_span = content._spans[0]
        assert theme.DARK_COLORS.mode_command in str(first_span.style)

    def test_normal_message_renders_angle_bracket(self) -> None:
        """`UserMessage('hello')` should render with `'> '` prefix."""
        content = _render_content(UserMessage("hello"))
        assert content.plain == "> hello"
        first_span = content._spans[0]
        assert theme.DARK_COLORS.primary in str(first_span.style)

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`UserMessage('')` should not crash and should render `'> '` prefix."""
        content = _render_content(UserMessage(""))
        assert content.plain == "> "


class TestModeColorsDrift:
    """Ensure `_mode_color` handles every mode in `MODE_PREFIXES`."""

    def test_mode_color_returns_non_primary_for_all_modes(self) -> None:
        from deepagents_code.config import MODE_PREFIXES
        from deepagents_code.tui.widgets.messages import _mode_color

        primary = _mode_color(None)
        for mode in MODE_PREFIXES:
            color = _mode_color(mode)
            assert color != primary, (
                f"_mode_color({mode!r}) returned primary; add a branch for this mode"
            )


class TestQueuedUserMessageModeRendering:
    """Test `QueuedUserMessage` renders mode-specific prefix indicators (dimmed)."""

    def test_shell_prefix_renders_dimmed_dollar(self) -> None:
        """`QueuedUserMessage('!ls')` should render dimmed `'$ '` prefix."""
        content = _render_content(QueuedUserMessage("!ls"))
        assert content.plain == "$ ls"

    def test_incognito_shell_prefix_renders_dimmed_dollar(self) -> None:
        """`QueuedUserMessage('!!ls')` should strip the full incognito prefix."""
        content = _render_content(QueuedUserMessage("!!ls"))
        assert content.plain == "$ ls"

    def test_command_prefix_renders_dimmed_slash(self) -> None:
        """`QueuedUserMessage('/help')` should render dimmed `'/ '` prefix."""
        content = _render_content(QueuedUserMessage("/help"))
        assert content.plain == "/ help"

    def test_normal_message_renders_dimmed_angle_bracket(self) -> None:
        """`QueuedUserMessage('hello')` should render dimmed `'> '` prefix."""
        content = _render_content(QueuedUserMessage("hello"))
        assert content.plain == "> hello"

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`QueuedUserMessage('')` should not crash and should render `'> '`."""
        content = _render_content(QueuedUserMessage(""))
        assert content.plain == "> "


class TestStripPromptPrefix:
    """Unit tests for `_strip_prompt_prefix` selection trimming."""

    def test_passes_through_none(self) -> None:
        """A `None` result (no extractable text) stays `None`."""
        from textual.selection import SELECT_ALL

        assert _strip_prompt_prefix(None, SELECT_ALL) is None

    def test_select_all_drops_prefix(self) -> None:
        """Select-all (`Selection(None, None)`) trims the two-column prefix."""
        from textual.selection import SELECT_ALL

        assert _strip_prompt_prefix(("> hello", "\n"), SELECT_ALL) == (
            "hello",
            "\n",
        )

    def test_selection_from_row_zero_drops_prefix(self) -> None:
        """A row-0 selection starting at column 0 trims the prefix."""
        from textual.geometry import Offset
        from textual.selection import Selection

        selection = Selection(Offset(0, 0), Offset(7, 0))
        assert _strip_prompt_prefix(("> hello", "\n"), selection) == ("hello", "\n")

    def test_partial_prefix_selection_trims_remaining_glyph(self) -> None:
        """Starting inside the prefix trims only the still-included columns."""
        from textual.geometry import Offset
        from textual.selection import Selection

        selection = Selection(Offset(1, 0), Offset(7, 0))
        assert _strip_prompt_prefix((" hello", "\n"), selection) == ("hello", "\n")

    def test_selection_starting_in_body_is_untouched(self) -> None:
        """A selection beginning past the prefix keeps the body verbatim."""
        from textual.geometry import Offset
        from textual.selection import Selection

        selection = Selection(Offset(4, 0), Offset(7, 0))
        assert _strip_prompt_prefix(("llo", "\n"), selection) == ("llo", "\n")

    def test_selection_starting_below_row_zero_is_untouched(self) -> None:
        """Selections that begin on later rows carry no prefix to strip."""
        from textual.geometry import Offset
        from textual.selection import Selection

        selection = Selection(Offset(0, 1), Offset(5, 1))
        assert _strip_prompt_prefix(("world", "\n"), selection) == ("world", "\n")


class _SelectionApp(App[None]):
    """Mount user-message widgets so `get_selection` has an active app."""

    def compose(self) -> ComposeResult:
        yield UserMessage("hello world", id="user")
        yield UserMessage("!ls", id="shell-user")
        yield QueuedUserMessage("hi there", id="queued")
        yield QueuedUserMessage("!pwd", id="shell-queued")


class TestUserMessageGetSelection:
    """Triple-click / select-all should copy the body, not the prefix glyph."""

    async def test_user_message_select_all_excludes_prefix(self) -> None:
        from textual.selection import SELECT_ALL

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#user", UserMessage)
            result = widget.get_selection(SELECT_ALL)
            assert result is not None
            assert result[0] == "hello world"

    async def test_queued_message_select_all_excludes_prefix(self) -> None:
        from textual.selection import SELECT_ALL

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#queued", QueuedUserMessage)
            result = widget.get_selection(SELECT_ALL)
            assert result is not None
            assert result[0] == "hi there"

    async def test_body_selection_preserved(self) -> None:
        from textual.geometry import Offset
        from textual.selection import Selection

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#user", UserMessage)
            selection = Selection(Offset(8, 0), Offset(13, 0))
            result = widget.get_selection(selection)
            assert result is not None
            assert result[0] == "world"

    async def test_user_message_select_all_starts_after_prompt_prefix(self) -> None:
        from textual.geometry import Offset
        from textual.selection import Selection

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#user", UserMessage)
            widget.text_select_all()
            assert pilot.app.screen.selections[widget] == Selection(Offset(2, 0), None)

    async def test_shell_user_select_all_starts_after_prompt_prefix(self) -> None:
        from textual.geometry import Offset
        from textual.selection import Selection

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#shell-user", UserMessage)
            widget.text_select_all()
            assert pilot.app.screen.selections[widget] == Selection(Offset(2, 0), None)
            result = widget.get_selection(pilot.app.screen.selections[widget])
            assert result is not None
            assert result[0] == "ls"

    async def test_queued_message_select_all_starts_after_prompt_prefix(self) -> None:
        from textual.geometry import Offset
        from textual.selection import Selection

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#queued", QueuedUserMessage)
            widget.text_select_all()
            assert pilot.app.screen.selections[widget] == Selection(Offset(2, 0), None)

    async def test_shell_queued_select_all_starts_after_prompt_prefix(self) -> None:
        from textual.geometry import Offset
        from textual.selection import Selection

        async with _SelectionApp().run_test() as pilot:
            widget = pilot.app.query_one("#shell-queued", QueuedUserMessage)
            widget.text_select_all()
            assert pilot.app.screen.selections[widget] == Selection(Offset(2, 0), None)
            result = widget.get_selection(pilot.app.screen.selections[widget])
            assert result is not None
            assert result[0] == "pwd"


class _MarkdownAppMessageApp(App[None]):
    """Mount a markdown `AppMessage` so selection has an active app + layout."""

    _MARKDOWN = (
        "### Core dependencies\n"
        "\n"
        "| Package | Version |\n"
        "| --- | --- |\n"
        "| langchain | 1.2.3 |\n"
        "| langgraph | not installed |\n"
    )

    def compose(self) -> ComposeResult:
        yield AppMessage(self._MARKDOWN, markdown=True, id="md")


class TestAppMessageMarkdownSelectable:
    """Markdown `AppMessage` output must be selectable and copyable.

    Regression guard: rendering markdown as a raw Rich renderable produces a
    `RichVisual`, which Textual cannot select or copy. The text must resolve to
    `Content` so `/version` tables and incognito shell output stay copyable.
    """

    async def test_markdown_renders_content_visual(self) -> None:
        from textual.content import Content

        async with _MarkdownAppMessageApp().run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            assert isinstance(widget._render(), Content)

    async def test_markdown_select_all_copies_table_text(self) -> None:
        from textual.selection import SELECT_ALL

        async with _MarkdownAppMessageApp().run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            result = widget.get_selection(SELECT_ALL)
            assert result is not None
            selected = result[0]
            assert "Core dependencies" in selected
            assert "langchain" in selected
            assert "not installed" in selected

    async def test_markdown_selection_has_no_trailing_padding(self) -> None:
        from textual.selection import SELECT_ALL

        async with _MarkdownAppMessageApp().run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            result = widget.get_selection(SELECT_ALL)
            assert result is not None
            assert not any(line != line.rstrip() for line in result[0].splitlines())

    async def test_markdown_caches_content_at_same_width(self) -> None:
        """A second render at an unchanged width reuses the cached `Content`."""
        async with _MarkdownAppMessageApp().run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            first = widget.render()
            second = widget.render()
            assert first is second

    async def test_markdown_reflows_on_resize(self) -> None:
        """Shrinking the terminal re-lays-out markdown to the new width.

        Guards the width-keyed cache invalidation (`_markdown_cache[0] != width`):
        a regression that dropped the width key would keep serving the stale,
        wider `Content`.
        """
        markdown = "This is a fairly long paragraph of prose " * 6
        app = _MarkdownAppMessageApp()
        app._MARKDOWN = markdown
        async with app.run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            wide = widget.render()
            wide_cache = widget._markdown_cache
            assert wide_cache is not None
            wide_key = wide_cache[0]

            await pilot.resize_terminal(40, 24)
            await pilot.pause()
            narrow = widget.render()
            narrow_cache = widget._markdown_cache
            assert narrow_cache is not None
            narrow_key = narrow_cache[0]

            assert narrow is not wide
            assert narrow_key < wide_key
            wide_max = max(len(line) for line in wide.plain.splitlines())
            narrow_max = max(len(line) for line in narrow.plain.splitlines())
            assert narrow_max < wide_max
            assert narrow_max <= narrow_key

    async def test_markdown_content_has_style_spans(self) -> None:
        """Styled markdown keeps its spans so emphasis survives to selection."""
        async with _MarkdownAppMessageApp().run_test(size=(80, 24)) as pilot:
            widget = pilot.app.query_one("#md", AppMessage)
            assert widget.render().spans


class TestMarkdownToContent:
    """Direct unit tests for `_markdown_to_content` edge cases."""

    def test_empty_markdown_yields_empty_content(self) -> None:
        from deepagents_code.tui.widgets.messages import _markdown_to_content

        assert not _markdown_to_content("", 40).plain

    def test_whitespace_only_markdown_yields_empty_content(self) -> None:
        from deepagents_code.tui.widgets.messages import _markdown_to_content

        assert not _markdown_to_content("   \n   \n", 40).plain

    def test_trailing_blank_lines_are_trimmed(self) -> None:
        from deepagents_code.tui.widgets.messages import _markdown_to_content

        content = _markdown_to_content("# Title\n\n\n", 40)
        assert "Title" in content.plain
        # Block-level trim: no empty trailing lines left in the joined content.
        assert content.plain == content.plain.rstrip("\n ")

    def test_style_conversion_failure_keeps_text(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failing style conversion drops the span but keeps text, warns once."""
        import logging

        from textual.style import Style

        from deepagents_code.tui.widgets import messages as messages_module
        from deepagents_code.tui.widgets.messages import _markdown_to_content

        def _boom(_style: object) -> Style:
            msg = "unconvertible"
            raise ValueError(msg)

        monkeypatch.setattr(Style, "from_rich_style", staticmethod(_boom))
        monkeypatch.setattr(
            messages_module, "_markdown_style_conversion_warned", [False]
        )

        with caplog.at_level(logging.WARNING, logger=messages_module.__name__):
            content = _markdown_to_content("### heading", 40)

        assert "heading" in content.plain
        assert not content.spans
        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestAppMessageAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`AppMessage` should disable Textual's `auto_links`."""
        assert AppMessage.auto_links is False


_WEBBROWSER_OPEN = "deepagents_code.tui.widgets._links.webbrowser.open"


class TestAppMessageOnClickOpensLink:
    """Tests for `AppMessage.on_click` opening style-embedded hyperlinks."""

    def test_click_on_link_opens_browser(self) -> None:
        """Clicking a styled link should call `webbrowser.open`."""
        msg = AppMessage("test")
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            msg.on_click(event)

        mock_open.assert_called_once_with("https://example.com")
        event.stop.assert_called_once()

    def test_click_without_link_is_noop(self) -> None:
        """Clicking on non-link text should not open the browser."""
        msg = AppMessage("test")
        event = MagicMock()
        event.style = Style()

        with patch(_WEBBROWSER_OPEN) as mock_open:
            msg.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()

    def test_click_with_browser_error_is_graceful(self) -> None:
        """Browser failure should not crash the widget."""
        msg = AppMessage("test")
        event = MagicMock()
        event.style = Style(link="https://example.com")

        with patch(_WEBBROWSER_OPEN, side_effect=OSError("no display")):
            msg.on_click(event)  # should not raise

        event.stop.assert_not_called()

    def test_click_on_suspicious_url_is_blocked(self) -> None:
        """Suspicious Unicode URL should not be opened."""
        msg = AppMessage("test")
        event = MagicMock()
        event.style = Style(link="https://аpple.com")

        with patch(_WEBBROWSER_OPEN) as mock_open:
            msg.on_click(event)

        mock_open.assert_not_called()
        event.stop.assert_not_called()


class TestMountMessageIdSync:
    """Tests for widget id sync in `_mount_message`."""

    def test_widget_id_assigned_from_message_data(self) -> None:
        """Widget with no id should get the MessageData id after from_widget."""
        from deepagents_code.tui.widgets.message_store import MessageData

        widget = UserMessage("hello")
        assert widget.id is None

        data = MessageData.from_widget(widget)
        # Simulate what _mount_message does
        if not widget.id:
            widget.id = data.id

        assert widget.id == data.id
        assert widget.id is not None

    def test_widget_with_existing_id_is_preserved(self) -> None:
        """Widget with an explicit id should keep it."""
        from deepagents_code.tui.widgets.message_store import MessageData

        widget = UserMessage("hello", id="my-custom-id")
        data = MessageData.from_widget(widget)

        if not widget.id:
            widget.id = data.id

        assert widget.id == "my-custom-id"


class TestGenericPreviewTruncation:
    """Tests for generic MCP tool preview truncation fallback."""

    def _make_msg(self, tool_name: str = "mcp_custom_tool") -> ToolCallMessage:
        """Create a ToolCallMessage with the given tool name."""
        return ToolCallMessage(tool_name, {})

    def test_unknown_tool_many_lines_truncated_in_preview(self) -> None:
        """Unknown tool output exceeding line limit should be truncated."""
        msg = self._make_msg()
        output = "\n".join(f"line {i}" for i in range(10))
        result = msg._format_output(output, is_preview=True)
        assert result.truncation is not None
        assert "more lines" in result.truncation

    def test_unknown_tool_long_single_line_truncated_in_preview(self) -> None:
        """Unknown tool output exceeding char limit should be char-truncated."""
        msg = self._make_msg()
        output = "x" * 500
        result = msg._format_output(output, is_preview=True)
        assert result.truncation is not None
        assert "100 more chars" in result.truncation
        assert len(result.content.plain) == 400

    def test_unknown_tool_short_output_no_truncation(self) -> None:
        """Short output from unknown tool should pass through untruncated."""
        msg = self._make_msg()
        output = "short output"
        result = msg._format_output(output, is_preview=True)
        assert result.truncation is None
        assert result.content.plain == "short output"

    def test_unknown_tool_exact_preview_lines_not_truncated(self) -> None:
        """Output with exactly _PREVIEW_LINES lines should NOT be line-truncated."""
        msg = self._make_msg()
        output = "\n".join(f"line {i}" for i in range(msg._PREVIEW_LINES))
        result = msg._format_output(output, is_preview=True)
        # Boundary: exactly at limit should pass through without line truncation
        truncation = result.truncation or ""
        assert result.truncation is None or "more lines" not in truncation

    def test_unknown_tool_full_output_no_truncation(self) -> None:
        """Non-preview mode should return full output regardless of length."""
        msg = self._make_msg()
        output = "x" * 500
        result = msg._format_output(output, is_preview=False)
        assert result.truncation is None
        assert result.content.plain == output


class TestStripFrontmatter:
    """Test _strip_frontmatter helper."""

    def test_strips_yaml_frontmatter(self) -> None:
        text = "---\nname: test\ndescription: A test\n---\n\n# Body\nContent"
        assert _strip_frontmatter(text) == "# Body\nContent"

    def test_no_frontmatter_unchanged(self) -> None:
        text = "# No frontmatter\nJust content"
        assert _strip_frontmatter(text) == text

    def test_unclosed_frontmatter_unchanged(self) -> None:
        text = "---\nname: test\nno closing marker"
        assert _strip_frontmatter(text) == text

    def test_empty_string(self) -> None:
        assert _strip_frontmatter("") == ""

    def test_leading_whitespace_before_frontmatter(self) -> None:
        text = "\n  ---\nname: test\n---\n\nBody"
        assert _strip_frontmatter(text) == "Body"

    def test_frontmatter_only(self) -> None:
        text = "---\nname: test\n---\n"
        assert _strip_frontmatter(text) == ""


class TestSkillMessageMarkupSafety:
    """Test SkillMessage handles content with brackets safely."""

    @pytest.mark.parametrize("content", MARKUP_INJECTION_CASES)
    def test_skill_message_no_markup_error(self, content: str) -> None:
        """SkillMessage should not raise on bracket content."""
        msg = SkillMessage(
            skill_name="test",
            description=content,
            body=content,
            args=content,
        )
        # Construction should not raise; compose() needs a running app
        # (Markdown widget) so we verify fields instead.
        assert msg._description == content
        assert msg._args == content

    def test_skill_message_stores_fields(self) -> None:
        msg = SkillMessage(
            skill_name="web-research",
            description="Research topics",
            source="user",
            body="# Instructions\nDo stuff",
            args="find quantum",
        )
        assert msg._skill_name == "web-research"
        assert msg._description == "Research topics"
        assert msg._source == "user"
        assert msg._body == "# Instructions\nDo stuff"
        assert msg._args == "find quantum"
        assert msg._expanded is False

    def test_skill_message_strips_frontmatter(self) -> None:
        """Body with frontmatter should have it stripped for display."""
        body = "---\nname: test\ndescription: A test\n---\n\n# Real content"
        msg = SkillMessage(skill_name="test", body=body)
        assert msg._stripped_body == "# Real content"
        # Raw body preserved for serialization
        assert msg._body == body

    def test_skill_message_no_args_skips_field(self) -> None:
        """When no args are provided, internal state should reflect that."""
        msg = SkillMessage(skill_name="test", args="")
        assert msg._args == ""
        assert msg._description == ""

    def test_skill_message_with_description_and_args(self) -> None:
        msg = SkillMessage(
            skill_name="test",
            description="A test skill",
            args="do something",
        )
        assert msg._description == "A test skill"
        assert msg._args == "do something"

    def test_skill_message_toggle_state(self) -> None:
        msg = SkillMessage(skill_name="test", body="some body")
        assert msg._expanded is False
        msg._expanded = True
        assert msg._expanded is True


class TestStripSuccessExitLine:
    """Test _strip_success_exit_line helper."""

    def test_strips_success_trailer(self) -> None:
        text = "hello world\n[Command succeeded with exit code 0]"
        assert _strip_success_exit_line(text) == "hello world"

    def test_strips_success_trailer_with_trailing_whitespace(self) -> None:
        text = "output\n[Command succeeded with exit code 0]  \n"
        assert _strip_success_exit_line(text) == "output"

    def test_preserves_failed_exit_code(self) -> None:
        text = "error\n[Command failed with exit code 1]"
        assert _strip_success_exit_line(text) == text

    def test_preserves_non_zero_success_code(self) -> None:
        """Only exit code 0 is stripped; other codes are untouched."""
        text = "output\n[Command succeeded with exit code 2]"
        assert _strip_success_exit_line(text) == text

    def test_empty_string(self) -> None:
        assert _strip_success_exit_line("") == ""

    def test_no_trailer(self) -> None:
        text = "just some output"
        assert _strip_success_exit_line(text) == text

    def test_only_trailer(self) -> None:
        text = "[Command succeeded with exit code 0]"
        assert _strip_success_exit_line(text) == ""

    def test_preserves_mid_string_trailer(self) -> None:
        """Trailer not at end of string should be left intact."""
        text = "before\n[Command succeeded with exit code 0]\nafter"
        assert _strip_success_exit_line(text) == text

    def test_set_success_strips_trailer(self) -> None:
        """Integration: set_success should strip the exit code 0 line."""
        msg = ToolCallMessage("execute", {"command": "echo hi"})
        msg.set_success("hi\n[Command succeeded with exit code 0]")
        assert msg._output == "hi"


class TestUserMessageCancelled:
    """`set_cancelled` dims a prompt whose turn was interrupted."""

    async def test_set_cancelled_adds_dim_class(self) -> None:
        """`set_cancelled` adds the `-cancelled` class that dims the prompt."""

        class _Harness(App[None]):
            def compose(self) -> ComposeResult:
                yield UserMessage("hello")

        app = _Harness()
        async with app.run_test() as pilot:
            msg = app.query_one(UserMessage)
            assert not msg.has_class("-cancelled")
            msg.set_cancelled()
            await pilot.pause()
            assert msg.has_class("-cancelled")


class TestSummarizeToolGroup:
    """Tests for the tool-group summary phrasing."""

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            (["execute"], "Ran 1 shell command"),
            (
                ["read_file", "read_file", "execute", "execute", "execute"],
                "Read 2 files, ran 3 shell commands",
            ),
            (["grep"], "Searched for 1 pattern"),
            (["grep", "glob", "glob"], "Searched for 3 patterns"),
            (["read_file"], "Read 1 file"),
            (["web_search", "web_search"], "Searched the web 2 times"),
            (["web_search"], "Searched the web"),
            (["write_todos"], "Updated todos"),
            (["task", "task"], "Ran 2 agents"),
            (
                ["edit_file", "write_file", "read_file"],
                "Edited 1 file, wrote 1 file, read 1 file",
            ),
            (["mystery", "mystery"], "Ran 2 mystery calls"),
        ],
    )
    def test_summary_phrasing(self, names: list[str], expected: str) -> None:
        """The summary aggregates by category and lowercases trailing verbs."""
        from deepagents_code.tui.widgets.messages import summarize_tool_group

        assert summarize_tool_group(names) == expected

    def test_empty_group_has_fallback(self) -> None:
        """An empty tool list yields a generic fallback rather than crashing."""
        from deepagents_code.tui.widgets.messages import summarize_tool_group

        assert summarize_tool_group([]) == "Ran tools"


class _ToolGroupApp(App[None]):
    """Minimal app mounting two completed tools plus a group summary."""

    def compose(self) -> ComposeResult:
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        t1 = ToolCallMessage("read_file", {"file_path": "a.py"})
        t1.id = "t1"
        t2 = ToolCallMessage("execute", {"command": "ls"})
        t2.id = "t2"
        summary = ToolGroupSummary(tools=[t1, t2], collapsible=[t1, t2])
        summary.id = "summary"
        yield summary
        yield t1
        yield t2


class TestToolGroupSummary:
    """Runtime collapse/expand behavior for the group summary widget."""

    async def test_collapsed_hides_members_and_renders_summary(self) -> None:
        """On mount the summary collapses its members and shows the count line."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _ToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)
            t2 = pilot.app.query_one("#t2", ToolCallMessage)

            assert summary._collapsed is True
            assert t1.display is False
            assert t2.display is False
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Read 1 file, ran 1 shell command" in rendered.plain

    async def test_toggle_expands_and_recollapses_members(self) -> None:
        """Toggling flips member visibility and the disclosure glyph."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _ToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)
            t2 = pilot.app.query_one("#t2", ToolCallMessage)

            summary.toggle()
            await pilot.pause()
            assert summary._collapsed is False
            assert t1.display is True
            assert t2.display is True

            summary.toggle()
            await pilot.pause()
            assert summary._collapsed is True
            assert t1.display is False
            assert t2.display is False

    async def test_has_attached_members_tracks_removal(self) -> None:
        """`has_attached_members` flips to False once members are removed."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _ToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            assert summary.has_attached_members is True

            await pilot.app.query_one("#t1", ToolCallMessage).remove()
            await pilot.app.query_one("#t2", ToolCallMessage).remove()
            await pilot.pause()
            assert summary.has_attached_members is False


class TestSummarizeToolGroupPresentTense:
    """Present-tense phrasing used while a step's tools are still running."""

    def test_present_tense(self) -> None:
        from deepagents_code.tui.widgets.messages import summarize_tool_group

        assert (
            summarize_tool_group(["execute"], tense="present")
            == "Running 1 shell command"
        )
        assert (
            summarize_tool_group(["read_file", "read_file", "grep"], tense="present")
            == "Reading 2 files, searching for 1 pattern"
        )


class _LiveToolGroupApp(App[None]):
    """Minimal app with an empty live group and two tools to add to it."""

    def compose(self) -> ComposeResult:
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        summary = ToolGroupSummary(live=True)
        summary.id = "summary"
        t1 = ToolCallMessage("execute", {"command": "ls"})
        t1.id = "t1"
        t2 = ToolCallMessage("read_file", {"file_path": "a.py"})
        t2.id = "t2"
        yield summary
        yield t1
        yield t2


class _LiveToolGroupSameCategoryApp(App[None]):
    """Live group with two tools of the same category (both shell commands)."""

    def compose(self) -> ComposeResult:
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        summary = ToolGroupSummary(live=True)
        summary.id = "summary"
        t1 = ToolCallMessage("execute", {"command": "ls"})
        t1.id = "t1"
        t2 = ToolCallMessage("execute", {"command": "pwd"})
        t2.id = "t2"
        yield summary
        yield t1
        yield t2


class TestLiveToolGroupSummary:
    """Eager/live group: collapsed from the start, running -> ran transition."""

    async def test_present_tense_while_running_then_past_on_close(self) -> None:
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)

            # add_member renders synchronously; avoid pilot.pause() while the
            # live spinner timer is running (it keeps the app from going idle).
            summary.add_member(t1)
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Running 1 shell command" in rendered.plain
            assert t1.display is False  # collapsed from the start

            t1.set_success("done")
            summary.close()  # stops the spinner timer, flips to past tense

            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Ran 1 shell command" in rendered.plain
            assert t1.display is False
            # Survives the idle tick after close — guards against the summary's
            # state attributes colliding with Textual's MessagePump internals
            # (e.g. `_closed`), which would silently prune the widget.
            await pilot.pause()
            assert summary.is_attached
            assert bool(pilot.app.query(ToolGroupSummary))

    async def test_live_line_counts_only_running_members(self) -> None:
        """Finished tools drop out of the live line while others still run."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            done = pilot.app.query_one("#t1", ToolCallMessage)  # execute
            running = pilot.app.query_one("#t2", ToolCallMessage)  # read_file

            summary.add_member(done)
            summary.add_member(running)
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Running 1 shell command, reading 1 file" in rendered.plain

            # The shell command finishes but the read is still in flight: the
            # live line must stop advertising the completed command.
            done.set_success("done")
            summary._render_line()
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Reading 1 file" in rendered.plain
            assert "shell command" not in rendered.plain

    async def test_live_line_decrements_same_category_count(self) -> None:
        """One of two shell commands finishing drops the live count 2 -> 1."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupSameCategoryApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            done = pilot.app.query_one("#t1", ToolCallMessage)
            running = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(done)
            summary.add_member(running)
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Running 2 shell commands" in rendered.plain

            # One command finishes; the surviving pending tuple shrinks from
            # ("execute", "execute") to ("execute",), which must invalidate the
            # cached line even though the category (and membership) is unchanged.
            done.set_success("done")
            summary._render_line()
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Running 1 shell command" in rendered.plain
            assert "2 shell commands" not in rendered.plain

    async def test_live_line_relayouts_only_when_summary_changes(self) -> None:
        """A shorter pending summary recalculates height on the next tick."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            done = pilot.app.query_one("#t1", ToolCallMessage)
            running = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(done)
            summary.add_member(running)
            summary._stop_timer()
            done.set_success("done")

            with patch.object(summary, "update", wraps=summary.update) as update:
                summary._tick()
                assert update.call_args.kwargs["layout"] is True

                update.reset_mock()
                summary._tick()
                assert update.call_args.kwargs["layout"] is False

    async def test_pending_member_is_revealed_for_approval(self) -> None:
        """Only unfinished calls leave the collapsed group before approval."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            completed = pilot.app.query_one("#t1", ToolCallMessage)
            pending = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(completed)
            summary.add_member(pending)
            completed.set_success("done")
            summary.reveal_pending()
            await pilot.pause()

            assert completed.display is False
            assert completed.has_class("-grouped")
            assert pending.display is True
            assert not pending.has_class("-grouped")
            assert summary._tools == [completed]
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Ran 1 shell command" in rendered.plain

    async def test_suppressed_pending_member_stays_hidden(self) -> None:
        """A command mirrored in the approval menu is detached without duplication."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            pending = pilot.app.query_one("#t1", ToolCallMessage)

            summary.add_member(pending)
            pending.set_awaiting_approval()
            summary.reveal_pending()
            await pilot.pause()

            assert pending.display is False
            assert not pending.has_class("-grouped")
            assert not summary.is_attached

            pending.clear_awaiting_approval()
            assert pending.display is True

    async def test_failed_member_is_evicted_on_close(self) -> None:
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)
            t2 = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(t1)
            summary.add_member(t2)
            t1.set_error("boom")
            t2.set_success("ok")
            summary.close()
            await pilot.pause()

            # The errored tool is un-folded; the successful one stays collapsed.
            assert t1.display is True
            assert not t1.has_class("-grouped")
            assert t2.display is False
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Read 1 file" in rendered.plain

    async def test_close_waits_for_pending_member_terminal_status(self) -> None:
        """A stream boundary must not report a still-pending tool as having run."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            shell = pilot.app.query_one("#t1", ToolCallMessage)
            read = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(shell)
            summary.add_member(read)
            summary.close()

            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Running 1 shell command, reading 1 file" in rendered.plain
            assert summary._finalized is False

            shell.set_error("authorization classifier unavailable")
            read.set_success("ok")
            summary._tick()
            await pilot.pause()

            assert summary._finalized is True
            assert shell.display is True
            assert not shell.has_class("-grouped")
            assert read.display is False
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Read 1 file" in rendered.plain
            assert "shell command" not in rendered.plain

    async def test_open_group_accepts_member_after_current_members_settle(self) -> None:
        """Settled members leave the live line without finalizing the group."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            shell = pilot.app.query_one("#t1", ToolCallMessage)
            read = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(shell)
            shell.set_success("ok")
            summary._tick()

            assert summary._finalized is False
            assert summary._timer is None

            summary.add_member(read)

            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Reading 1 file" in rendered.plain
            assert "shell command" not in rendered.plain
            assert summary._timer is not None

            read.set_error("boom")
            summary._tick()
            await pilot.pause()

            assert summary._tools == [shell]
            assert read.display is True
            assert not read.has_class("-grouped")
            assert summary._finalized is False
            assert summary._timer is None

            summary.close()
            assert summary._finalized is True
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Ran 1 shell command" in rendered.plain

    async def test_reveal_pending_finalizes_closed_settled_members(self) -> None:
        """Approval finalizes retained successes after pending calls leave."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            completed = pilot.app.query_one("#t1", ToolCallMessage)
            pending = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(completed)
            summary.add_member(pending)
            completed.set_success("ok")
            summary.close()

            assert summary._finalized is False
            assert summary._timer is not None

            summary.reveal_pending()
            await pilot.pause()

            assert summary._tools == [completed]
            assert summary._finalized is True
            assert summary._timer is None
            assert pending.display is True
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Ran 1 shell command" in rendered.plain

    async def test_rejected_member_is_evicted_on_close(self) -> None:
        """A rejected tool stays visible, mirroring the errored-tool path."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)
            t2 = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(t1)
            summary.add_member(t2)
            t1.set_rejected(reason="not now")
            t2.set_success("ok")
            summary.close()
            await pilot.pause()

            assert t1.display is True
            assert not t1.has_class("-grouped")
            assert t2.display is False
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Read 1 file" in rendered.plain

    async def test_skipped_member_is_evicted_and_uncounted_on_close(self) -> None:
        """A skipped tool stays visible and is left out of the summary count.

        Regression: `skipped` once fell through `is_success`/`is_failed`/
        `is_pending`, so a skipped tool stayed folded and inflated the count
        (e.g. "Ran 1 shell command" for a command that never executed).
        """
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)  # execute
            t2 = pilot.app.query_one("#t2", ToolCallMessage)  # read_file

            summary.add_member(t1)
            summary.add_member(t2)
            t1.set_skipped()
            t2.set_success("ok")
            summary.close()
            await pilot.pause()

            # The skipped tool is un-folded and no longer part of the group.
            assert t1.display is True
            assert not t1.has_class("-grouped")
            assert t2.display is False
            rendered = summary.render()
            assert isinstance(rendered, Content)
            assert "Read 1 file" in rendered.plain
            # The skipped execute must not be summarized as if it had run.
            assert "shell command" not in rendered.plain

    async def test_all_failed_members_remove_summary_on_close(self) -> None:
        """When every member fails, the empty summary removes itself."""
        from deepagents_code.tui.widgets.messages import ToolGroupSummary

        async with _LiveToolGroupApp().run_test() as pilot:
            summary = pilot.app.query_one("#summary", ToolGroupSummary)
            t1 = pilot.app.query_one("#t1", ToolCallMessage)
            t2 = pilot.app.query_one("#t2", ToolCallMessage)

            summary.add_member(t1)
            summary.add_member(t2)
            t1.set_error("boom")
            t2.set_rejected(reason="no")
            summary.close()
            await pilot.pause()

            # Nothing left to summarize: the summary detaches, both tools show.
            assert not summary.is_attached
            assert not pilot.app.query(ToolGroupSummary)
            assert t1.display is True
            assert t2.display is True


class TestUserMessageTruncation:
    """Test head+tail truncation of very long user messages at render time."""

    def test_short_message_not_truncated(self) -> None:
        """Messages under the threshold should render in full."""
        from deepagents_code.tui.widgets.messages import _truncate_for_display

        text = "short message"
        assert _truncate_for_display(text) == text

    def test_long_message_truncated_with_elision(self) -> None:
        """Messages over 10k chars should get head+tail+elision marker."""
        from deepagents_code.tui.widgets.messages import _truncate_for_display

        text = "A" * 12_000
        result = _truncate_for_display(text)
        assert "… +" in result
        assert " lines …" in result
        # Head and tail are preserved
        assert result.startswith("A" * 2500)
        assert result.endswith("A" * 2500)

    def test_truncation_counts_hidden_lines(self) -> None:
        """The elision marker should report the correct hidden line count."""
        from deepagents_code.tui.widgets.messages import _truncate_for_display

        lines = [f"line {i:04d} " + "x" * 20 for i in range(600)]
        text = "\n".join(lines)
        assert len(text) > 10_000
        result = _truncate_for_display(text)
        assert "… +" in result
        assert " lines …" in result

    def test_full_content_preserved_in_widget(self) -> None:
        """The widget should store full content even when display is truncated."""
        big = "B" * 12_000
        msg = UserMessage(big)
        assert msg._content == big
        assert len(msg._content) == 12_000

    def test_message_at_boundary_not_truncated(self) -> None:
        """Messages exactly at the threshold should not be truncated."""
        from deepagents_code.tui.widgets.messages import _truncate_for_display

        text = "C" * 10_000
        assert _truncate_for_display(text) == text

    async def test_selection_returns_full_content_not_truncated(self) -> None:
        """Selecting a truncated message should return the full original text."""
        from textual.geometry import Offset
        from textual.selection import Selection

        big = "X" * 12_000
        msg = UserMessage(big)

        class _TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield msg

        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            # Select the entire message body (skip prefix glyph)
            selection = Selection(Offset(2, 0), None)
            result = msg.get_selection(selection)
            assert result is not None
            text, _ending = result
            assert text == big
            assert "…" not in text

    def test_truncation_reports_exact_hidden_newline_count(self) -> None:
        """The elision marker reports the exact number of hidden newlines."""
        from deepagents_code.tui.widgets.messages import _truncate_for_display

        text = "H" * 6000 + "\n" * 50 + "T" * 6000
        result = _truncate_for_display(text)
        assert "… +50 lines …" in result

    async def test_partial_selection_uses_visible_render(self) -> None:
        """A partial selection defers to the on-screen (truncated) render.

        Select-all extracts from the full content, but a partial selection must
        stay aligned with what is visible, so it delegates to the base widget.
        """
        from textual.geometry import Offset
        from textual.selection import Selection
        from textual.widgets import Static

        big = "X" * 12_000
        msg = UserMessage(big)

        class _TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield msg

        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            partial = Selection(Offset(2, 0), Offset(6, 0))
            expected = _strip_prompt_prefix(Static.get_selection(msg, partial), partial)
            result = msg.get_selection(partial)
            # Delegates to the base (truncated) extraction, so it must not
            # return the full 12k body the way full-content extraction would.
            assert result == expected
            assert result is not None
            assert result[0] != big

    def test_queued_message_render_truncates(self) -> None:
        """QueuedUserMessage render truncates long content with an elision marker."""
        content = _render_content(QueuedUserMessage("Q" * 12_000))
        assert content.plain.startswith("> ")
        assert "… +" in content.plain
        assert len(content.plain) < 12_000

    async def test_queued_selection_returns_full_content(self) -> None:
        """Select-all on a truncated QueuedUserMessage returns the full text."""
        from textual.geometry import Offset
        from textual.selection import Selection

        big = "Y" * 12_000
        msg = QueuedUserMessage(big)

        class _TestApp(App[None]):
            def compose(self) -> ComposeResult:
                yield msg

        app = _TestApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            result = msg.get_selection(Selection(Offset(2, 0), None))
            assert result is not None
            text, _ending = result
            assert text == big
            assert "…" not in text


class _RubricResultApp(App[None]):
    """Minimal app mounting a rubric result."""

    def compose(self) -> ComposeResult:
        yield RubricResultMessage(
            "Acceptance criteria not yet satisfied",
            "Explanation\n" + "complete detail " * 200,
            id="rubric-result",
        )


class _DeferredExpandedRubricApp(App[None]):
    """Mounts a rubric result whose expansion was restored from virtualization."""

    def compose(self) -> ComposeResult:
        widget = RubricResultMessage(
            "Acceptance criteria not yet satisfied",
            "Explanation\ndetail",
            id="rubric-deferred",
        )
        widget._deferred_expanded = True
        yield widget


class _RecordingRubricApp(App[None]):
    """Records `ExpansionChanged` messages posted by a mounted rubric result."""

    def __init__(self) -> None:
        super().__init__()
        self.expansions: list[bool] = []

    def compose(self) -> ComposeResult:
        yield RubricResultMessage(
            "Acceptance criteria not yet satisfied",
            "Explanation\ndetail",
            id="rubric-events",
        )

    def on_rubric_result_message_expansion_changed(
        self,
        event: RubricResultMessage.ExpansionChanged,
    ) -> None:
        self.expansions.append(event.expanded)


class TestRubricResultMessage:
    """Grader details stay complete, collapsed, and scrollable."""

    async def test_details_expand_without_truncation(self) -> None:
        """The compact default should reveal the full plain-text result on demand."""
        app = _RubricResultApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            message = app.query_one("#rubric-result", RubricResultMessage)
            details_scroll = message.query_one(
                ".rubric-result-details-scroll",
                VerticalScroll,
            )
            details = message.query_one(".rubric-result-details", Static)

            assert message._expanded is False
            assert details_scroll.display is False
            assert str(details.content) == message._details
            assert "complete detail " * 200 in str(details.content)
            assert details_scroll.styles.max_height is not None
            assert details_scroll.styles.max_height.cells == 16

            await pilot.click(".rubric-result-hint")
            await pilot.pause()

            assert message._expanded is True
            assert details_scroll.display is True
            assert str(details.content) == message._details

    async def test_deferred_expansion_is_restored_on_mount(self) -> None:
        """A rehydrated widget must reopen its details, not just remember the flag."""
        app = _DeferredExpandedRubricApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            message = app.query_one("#rubric-deferred", RubricResultMessage)
            details_scroll = message.query_one(
                ".rubric-result-details-scroll",
                VerticalScroll,
            )

            assert message._expanded is True
            assert details_scroll.display is True

    async def test_toggle_posts_expansion_changed(self) -> None:
        """Toggling an attached widget must publish state for virtualization."""
        app = _RecordingRubricApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            message = app.query_one("#rubric-events", RubricResultMessage)
            # Mounting collapsed (deferred False) must not emit a spurious event.
            assert app.expansions == []

            message.toggle_details()
            await pilot.pause()
            message.toggle_details()
            await pilot.pause()

            assert app.expansions == [True, False]
