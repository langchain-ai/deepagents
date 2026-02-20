"""Unit tests for message widgets markup safety."""

from unittest.mock import MagicMock, patch

import pytest
from rich.markup import render
from rich.style import Style
from rich.text import Text

from deepagents_cli.config import COLORS
from deepagents_cli.input import INPUT_HIGHLIGHT_PATTERN
from deepagents_cli.widgets.messages import (
    AppMessage,
    ErrorMessage,
    QueuedUserMessage,
    ToolCallMessage,
    UserMessage,
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
        """Tool header should escape tool label content before Rich parsing."""
        msg = ToolCallMessage(
            "task",
            {"description": "Search for closing tag [/dim] mismatches"},
        )

        # `task` has no inline args widget, so this validates the header markup.
        header = next(iter(msg.compose()))
        content = header._Static__content  # type: ignore[attr-defined]
        assert isinstance(content, str)
        rendered = render(content)
        assert "[/dim]" in rendered.plain

    def test_tool_args_line_escapes_markup_values(self) -> None:
        """Inline args line should escape bracket content in argument values."""
        msg = ToolCallMessage(
            "custom_tool",
            {"pattern": "[foo]", "note": "raw [/dim] text"},
        )

        widgets = list(msg.compose())
        args_widget = widgets[1]
        content = args_widget._Static__content  # type: ignore[attr-defined]
        assert isinstance(content, str)
        rendered = render(content)
        assert "[foo]" in rendered.plain
        assert "[/dim]" in rendered.plain


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
        # Include a line that looks like a command prompt in the output
        output = "$ echo test\ntest output\n$ not a command"
        result = msg._format_shell_output(output, is_preview=False)

        # First line (the command) should be wrapped in [dim] markup
        assert "[dim]$ echo test[/dim]" in result.content
        # Subsequent lines starting with $ should NOT be dimmed
        assert "$ not a command" in result.content
        assert "[dim]$ not a command" not in result.content


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


def _compose_text(widget: UserMessage | QueuedUserMessage) -> Text:
    """Extract the Rich `Text` object from a message widget's first yielded Static."""
    statics = list(widget.compose())
    assert statics, "compose() yielded no widgets"
    content = statics[0]._Static__content  # type: ignore[attr-defined]
    assert isinstance(content, Text)
    return content


class TestUserMessageModeRendering:
    """Test `UserMessage` renders mode-specific prefix indicators and colors."""

    def test_bash_prefix_renders_bang_indicator(self) -> None:
        """`UserMessage('!ls')` should render with `'! '` prefix and bash body."""
        text = _compose_text(UserMessage("!ls"))
        assert text.plain == "! ls"
        first_span = text._spans[0]
        assert COLORS["mode_bash"] in str(first_span.style)

    def test_command_prefix_renders_slash_indicator(self) -> None:
        """`UserMessage('/help')` should render with `'/ '` prefix and body."""
        text = _compose_text(UserMessage("/help"))
        assert text.plain == "/ help"
        first_span = text._spans[0]
        assert COLORS["mode_command"] in str(first_span.style)

    def test_normal_message_renders_angle_bracket(self) -> None:
        """`UserMessage('hello')` should render with `'> '` prefix."""
        text = _compose_text(UserMessage("hello"))
        assert text.plain == "> hello"
        first_span = text._spans[0]
        assert COLORS["primary"] in str(first_span.style)

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`UserMessage('')` should not crash and should render `'> '` prefix."""
        text = _compose_text(UserMessage(""))
        assert text.plain == "> "


class TestQueuedUserMessageModeRendering:
    """Test `QueuedUserMessage` renders mode-specific prefix indicators (dimmed)."""

    def test_bash_prefix_renders_dimmed_bang(self) -> None:
        """`QueuedUserMessage('!ls')` should render dimmed `'! '` prefix."""
        text = _compose_text(QueuedUserMessage("!ls"))
        assert text.plain == "! ls"

    def test_command_prefix_renders_dimmed_slash(self) -> None:
        """`QueuedUserMessage('/help')` should render dimmed `'/ '` prefix."""
        text = _compose_text(QueuedUserMessage("/help"))
        assert text.plain == "/ help"

    def test_normal_message_renders_dimmed_angle_bracket(self) -> None:
        """`QueuedUserMessage('hello')` should render dimmed `'> '` prefix."""
        text = _compose_text(QueuedUserMessage("hello"))
        assert text.plain == "> hello"

    def test_empty_content_renders_angle_bracket(self) -> None:
        """`QueuedUserMessage('')` should not crash and should render `'> '`."""
        text = _compose_text(QueuedUserMessage(""))
        assert text.plain == "> "


class TestAppMessageAutoLinksDisabled:
    """Tests that `auto_links` is disabled to prevent hover flicker."""

    def test_auto_links_is_false(self) -> None:
        """`AppMessage` should disable Textual's `auto_links`."""
        assert AppMessage.auto_links is False


_WEBBROWSER_OPEN = "deepagents_cli.widgets._links.webbrowser.open"


class TestAppMessageOnClickOpensLink:
    """Tests for `AppMessage.on_click` opening Rich-style hyperlinks."""

    def test_click_on_link_opens_browser(self) -> None:
        """Clicking a Rich link should call `webbrowser.open`."""
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
