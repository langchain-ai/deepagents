"""Tests for middleware utility functions."""

from langchain_core.messages import SystemMessage

from deepagents.middleware._utils import append_to_system_message


class TestAppendToSystemMessage:
    """Tests for append_to_system_message."""

    def test_none_system_message_returns_string_content(self) -> None:
        result = append_to_system_message(None, "hello")
        assert isinstance(result, SystemMessage)
        assert result.content == "hello"
        assert isinstance(result.content, str)

    def test_string_content_stays_string(self) -> None:
        msg = SystemMessage(content="existing prompt")
        result = append_to_system_message(msg, "new text")
        assert isinstance(result.content, str)
        assert result.content == "existing prompt\n\nnew text"

    def test_string_content_multiple_appends_stay_string(self) -> None:
        msg = SystemMessage(content="base")
        msg = append_to_system_message(msg, "second")
        msg = append_to_system_message(msg, "third")
        assert isinstance(msg.content, str)
        assert msg.content == "base\n\nsecond\n\nthird"

    def test_list_content_stays_list(self) -> None:
        msg = SystemMessage(
            content=[
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                {"type": "text", "text": "describe this image"},
            ]
        )
        result = append_to_system_message(msg, "additional instructions")
        assert isinstance(result.content, list)
        # New text appended as last block
        assert result.content[-1] == {"type": "text", "text": "\n\nadditional instructions"}

    def test_content_blocks_content_stays_list(self) -> None:
        msg = SystemMessage(
            content_blocks=[
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ]
        )
        result = append_to_system_message(msg, "third")
        # content_blocks input produces list content; verify the original is type list
        assert isinstance(result.content, list)
