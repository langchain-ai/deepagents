"""Unit tests for message widgets markup safety."""

import pytest

from deepagents_cli.widgets.messages import (
    ErrorMessage,
    SystemMessage,
    ToolCallMessage,
    UserMessage,
)

# Content that previously caused MarkupError crashes
MARKUP_INJECTION_CASES = [
    "[foo] bar [baz]",
    "}, [/* deps */]);",
    "array[0] = value[1]",
    "[bold]not markup[/bold]",
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


class TestSystemMessageMarkupSafety:
    """Test SystemMessage handles content with brackets safely."""

    @pytest.mark.parametrize("content", MARKUP_INJECTION_CASES)
    def test_system_message_no_markup_error(self, content: str) -> None:
        """SystemMessage should not raise MarkupError on bracket content."""
        # Instantiation should not raise - this is the key test
        SystemMessage(content)

    def test_system_message_instantiates(self) -> None:
        """SystemMessage should instantiate with bracket content."""
        content = "Status: processing items[0-10]"
        msg = SystemMessage(content)
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


class TestUserMessageHighlighting:
    """Test UserMessage highlighting of `@mentions` and `/commands`."""

    def test_at_mention_highlighted(self) -> None:
        """`@file` mentions should be styled in the output."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "look at @README.md please"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "@README.md"

    def test_slash_command_highlighted_at_start(self) -> None:
        """Slash commands at start should be detected."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "/help me with something"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "/help"
        assert matches[0].start() == 0

    def test_slash_command_not_matched_mid_text(self) -> None:
        """Slash in middle of text should not match as command due to ^ anchor."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "check the /usr/bin path"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        # The ^ anchor means /usr doesn't match when not at start of string
        assert len(matches) == 0

    def test_multiple_at_mentions(self) -> None:
        """Multiple `@mentions` should all be detected."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "compare @file1.py with @file2.py"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 2
        assert matches[0].group() == "@file1.py"
        assert matches[1].group() == "@file2.py"

    def test_at_mention_with_path(self) -> None:
        """`@mentions` with paths should be fully captured."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "read @src/utils/helpers.py"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group() == "@src/utils/helpers.py"

    def test_no_matches_in_plain_text(self) -> None:
        """Plain text without `@` or `/` should have no matches."""
        from deepagents_cli.widgets.messages import _USER_HIGHLIGHT_PATTERN

        content = "just some normal text here"
        matches = list(_USER_HIGHLIGHT_PATTERN.finditer(content))
        assert len(matches) == 0
