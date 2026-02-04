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
