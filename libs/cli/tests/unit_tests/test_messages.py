"""Unit tests for message widgets markup safety."""

from unittest.mock import MagicMock, patch

import pytest
from rich.style import Style
from textual.content import Content

from deepagents_cli.config import COLORS
from deepagents_cli.input import INPUT_HIGHLIGHT_PATTERN
from deepagents_cli.widgets.messages import (
    AppMessage,
    AssistantMessage,
    DiffMessage,
    ErrorMessage,
    QueuedUserMessage,
    SummarizationMessage,
    ToolCallMessage,
    UserMessage,
    _show_timestamp_toast,
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
        rendered = msg._Static__content  # type: ignore[attr-defined]
        assert isinstance(rendered, Content)
        assert rendered.plain == "Error: something broke"


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
        rendered = msg._Static__content  # type: ignore[attr-defined]
        assert isinstance(rendered, Content)
        assert rendered.plain == "hello"

    def test_app_message_content_passthrough(self) -> None:
        """Pre-styled `Content` should pass through unchanged."""
        pre = Content.styled("styled", "bold cyan")
        msg = AppMessage(pre)
        rendered = msg._Static__content  # type: ignore[attr-defined]
        assert rendered is pre


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
        rendered = msg._Static__content  # type: ignore[attr-defined]
        assert isinstance(rendered, Content)
        assert rendered.plain == "custom text"

    def test_summarization_message_content_passthrough(self) -> None:
        """Pre-styled `Content` should pass through unchanged."""
        pre = Content.styled("pre-styled", "bold cyan")
        msg = SummarizationMessage(pre)
        rendered = msg._Static__content  # type: ignore[attr-defined]
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
        """Tool header should safely render label content with markup-like chars."""
        msg = ToolCallMessage(
            "task",
            {"description": "Search for closing tag [/dim] mismatches"},
        )

        # `task` has no inline args widget, so this validates the header markup.
        header = next(iter(msg.compose()))
        content = header._Static__content
        assert isinstance(content, Content)
        assert "[/dim]" in content.plain

    def test_tool_args_line_escapes_markup_values(self) -> None:
        """Inline args line should escape bracket content in argument values."""
        msg = ToolCallMessage(
            "custom_tool",
            {"pattern": "[foo]", "note": "raw [/dim] text"},
        )

        widgets = list(msg.compose())
        args_widget = widgets[1]
        content = args_widget._Static__content  # type: ignore[attr-defined]
        assert isinstance(content, Content)
        assert "[foo]" in content.plain
        assert "[/dim]" in content.plain


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

        msg = ToolCallMessage("shell", {"command": long_cmd})
        msg.set_error("Command not found: pip")

        # The error output should include the full command
        assert long_cmd in msg._output

    def test_shell_error_command_prefix(self) -> None:
        """Error output should have shell prompt prefix."""
        cmd = "echo hello"
        msg = ToolCallMessage("shell", {"command": cmd})
        msg.set_error("Permission denied")

        # Output should have shell prompt prefix
        assert msg._output.startswith("$ ")
        assert cmd in msg._output

    def test_bash_error_includes_full_command(self) -> None:
        """Error output should include full command for bash tool too."""
        cmd = "make build"
        msg = ToolCallMessage("bash", {"command": cmd})
        msg.set_error("make: *** No rule to make target")

        assert msg._output.startswith("$ ")
        assert cmd in msg._output

    def test_execute_error_includes_full_command(self) -> None:
        """Error output should include full command for execute tool too."""
        cmd = "docker build ."
        msg = ToolCallMessage("execute", {"command": cmd})
        msg.set_error("Cannot connect to Docker daemon")

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
        msg = ToolCallMessage("shell", {"command": None})
        error = "Some error"
        msg.set_error(error)

        assert "$ None" not in msg._output
        assert msg._output == error

    def test_shell_error_with_empty_command(self) -> None:
        """Shell tool with empty command should fall back to error-only output."""
        msg = ToolCallMessage("shell", {"command": ""})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error
        assert not msg._output.startswith("$ ")

    def test_shell_error_with_whitespace_command(self) -> None:
        """Shell tool with whitespace command should fall back to error-only output."""
        msg = ToolCallMessage("shell", {"command": "   "})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error

    def test_shell_error_with_no_command_key(self) -> None:
        """Shell tool with no command key should fall back to error-only output."""
        msg = ToolCallMessage("shell", {"other_arg": "value"})
        error = "Some error"
        msg.set_error(error)

        assert msg._output == error
        assert not msg._output.startswith("$ ")

    def test_format_shell_output_styles_only_first_line_dim(self) -> None:
        """Shell output formatting should only style the first command line in dim."""
        msg = ToolCallMessage("shell", {"command": "echo test"})
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


def _compose_content(widget: UserMessage | QueuedUserMessage) -> Content:
    """Extract the `Content` object from a message widget's first yielded Static."""
    statics = list(widget.compose())
    assert statics, "compose() yielded no widgets"
    result = statics[0]._Static__content  # type: ignore[attr-defined]
    assert isinstance(result, Content)
    return result


class TestUserMessageModeRendering:
    """Test `UserMessage` renders mode-specific prefix indicators and colors."""

    def test_shell_prefix_renders_dollar_indicator(self) -> None:
        """`UserMessage('!ls')` should render with `'$ '` prefix and shell body."""
        content = _compose_content(UserMessage("!ls"))
        assert content.plain == "$ ls"
        first_span = content._spans[0]
        assert COLORS["mode_shell"] in str(first_span.style)

    def test_command_prefix_renders_slash_indicator(self) -> None:
        """`UserMessage('/help')` should render with `'/ '` prefix and body."""
        content = _compose_content(UserMessage("/help"))
        assert content.plain == "/ help"
        first_span = content._spans[0]
        assert COLORS["mode_command"] in str(first_span.style)

    def test_normal_message_renders_angle_bracket(self) -> None:
        """`UserMessage('hello')` should render with `'> '` prefix."""
        content = _compose_content(UserMessage("hello"))
        assert content.plain == "> hello"
        first_span = content._spans[0]
        assert COLORS["primary"] in str(first_span.style)

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`UserMessage('')` should not crash and should render `'> '` prefix."""
        content = _compose_content(UserMessage(""))
        assert content.plain == "> "


class TestQueuedUserMessageModeRendering:
    """Test `QueuedUserMessage` renders mode-specific prefix indicators (dimmed)."""

    def test_shell_prefix_renders_dimmed_dollar(self) -> None:
        """`QueuedUserMessage('!ls')` should render dimmed `'$ '` prefix."""
        content = _compose_content(QueuedUserMessage("!ls"))
        assert content.plain == "$ ls"

    def test_command_prefix_renders_dimmed_slash(self) -> None:
        """`QueuedUserMessage('/help')` should render dimmed `'/ '` prefix."""
        content = _compose_content(QueuedUserMessage("/help"))
        assert content.plain == "/ help"

    def test_normal_message_renders_dimmed_angle_bracket(self) -> None:
        """`QueuedUserMessage('hello')` should render dimmed `'> '` prefix."""
        content = _compose_content(QueuedUserMessage("hello"))
        assert content.plain == "> hello"

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`QueuedUserMessage('')` should not crash and should render `'> '`."""
        content = _compose_content(QueuedUserMessage(""))
        assert content.plain == "> "


class TestAppMessageAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`AppMessage` should disable Textual's `auto_links`."""
        assert AppMessage.auto_links is False


_WEBBROWSER_OPEN = "deepagents_cli.widgets._links.webbrowser.open"


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


# ---------------------------------------------------------------------------
# Timestamp toast tests
# ---------------------------------------------------------------------------

_MSG_STORE_PATH = "deepagents_cli.widgets.messages"


class TestShowTimestampToast:
    """Tests for `_show_timestamp_toast` helper."""

    def test_noop_when_widget_not_mounted(self) -> None:
        """Should not raise when widget has no app."""
        widget = MagicMock(spec=["app", "id"])
        # Simulate unmounted widget: .app property raises
        type(widget).app = property(
            lambda _: (_ for _ in ()).throw(RuntimeError("no app"))
        )
        widget.id = "msg-abc"
        _show_timestamp_toast(widget)  # should not raise

    def test_noop_when_widget_id_is_none(self) -> None:
        """Should return early when widget.id is None."""
        widget = MagicMock()
        widget.id = None
        widget.app = MagicMock()
        _show_timestamp_toast(widget)
        widget.app.notify.assert_not_called()

    def test_noop_when_message_not_in_store(self) -> None:
        """Should return early when message is not found in the store."""
        widget = MagicMock()
        widget.id = "msg-missing"
        widget.app._message_store.get_message.return_value = None
        _show_timestamp_toast(widget)
        widget.app.notify.assert_not_called()

    def test_shows_toast_with_formatted_timestamp(self) -> None:
        """Should call notify with a human-readable timestamp."""
        from deepagents_cli.widgets.message_store import MessageData, MessageType

        data = MessageData(
            type=MessageType.USER,
            content="hello",
            id="msg-test123",
            timestamp=1709744055.0,  # 2024-03-06 17:14:15 UTC
        )
        widget = MagicMock()
        widget.id = "msg-test123"
        widget.app._message_store.get_message.return_value = data

        _show_timestamp_toast(widget)

        widget.app.notify.assert_called_once()
        call_args = widget.app.notify.call_args
        label = call_args[0][0]
        # Should contain month abbreviation and time components
        assert "Mar" in label
        assert ":" in label
        assert call_args[1]["timeout"] == 3


class TestTimestampClickMixin:
    """Tests for `_TimestampClickMixin` on message widgets."""

    @pytest.mark.parametrize(
        "cls",
        [UserMessage, AssistantMessage, DiffMessage, ErrorMessage],
        ids=["UserMessage", "AssistantMessage", "DiffMessage", "ErrorMessage"],
    )
    def test_mixin_classes_have_on_click(self, cls: type) -> None:
        """Mixin widget classes should have an `on_click` handler."""
        assert hasattr(cls, "on_click")

    def test_tool_call_message_click_without_output_shows_toast(self) -> None:
        """ToolCallMessage click with no output should show timestamp toast."""
        msg = ToolCallMessage("test_tool", {})
        msg._output = ""
        event = MagicMock()

        with patch(f"{_MSG_STORE_PATH}._show_timestamp_toast") as mock_toast:
            msg.on_click(event)

        event.stop.assert_called_once()
        mock_toast.assert_called_once_with(msg)

    def test_tool_call_message_click_with_output_toggles(self) -> None:
        """ToolCallMessage click with output should toggle, not toast."""
        msg = ToolCallMessage("test_tool", {})
        msg._output = "some output"
        event = MagicMock()

        with (
            patch.object(msg, "toggle_output") as mock_toggle,
            patch(f"{_MSG_STORE_PATH}._show_timestamp_toast") as mock_toast,
        ):
            msg.on_click(event)

        event.stop.assert_called_once()
        mock_toggle.assert_called_once()
        mock_toast.assert_not_called()

    def test_app_message_click_shows_toast_alongside_link(self) -> None:
        """AppMessage click should open link and show toast."""
        msg = AppMessage("test")
        event = MagicMock()
        event.style = Style()

        with (
            patch(_WEBBROWSER_OPEN),
            patch(f"{_MSG_STORE_PATH}._show_timestamp_toast") as mock_toast,
        ):
            msg.on_click(event)

        mock_toast.assert_called_once_with(msg)


class TestMountMessageIdSync:
    """Tests for widget id sync in `_mount_message`."""

    def test_widget_id_assigned_from_message_data(self) -> None:
        """Widget with no id should get the MessageData id after from_widget."""
        from deepagents_cli.widgets.message_store import MessageData

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
        from deepagents_cli.widgets.message_store import MessageData

        widget = UserMessage("hello", id="my-custom-id")
        data = MessageData.from_widget(widget)

        if not widget.id:
            widget.id = data.id

        assert widget.id == "my-custom-id"
