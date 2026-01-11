"""Unit tests for deepagents.graph module."""

from langchain_core.messages import SystemMessage

from deepagents.graph import BASE_AGENT_PROMPT, _merge_system_prompt


class TestMergeSystemPrompt:
    """Test suite for _merge_system_prompt function."""

    def test_none_input(self) -> None:
        """Returns SystemMessage with just BASE_AGENT_PROMPT when input is None."""
        result = _merge_system_prompt(None)
        assert isinstance(result, SystemMessage)
        assert result.content == BASE_AGENT_PROMPT

    def test_string_input(self) -> None:
        """Wraps string input in SystemMessage and appends BASE_AGENT_PROMPT."""
        result = _merge_system_prompt("Custom prompt")
        assert isinstance(result, SystemMessage)
        assert result.content == f"Custom prompt\n\n{BASE_AGENT_PROMPT}"

    def test_system_message_with_string_content(self) -> None:
        """Handles SystemMessage with string content."""
        input_msg = SystemMessage(content="Custom prompt")
        result = _merge_system_prompt(input_msg)
        assert isinstance(result, SystemMessage)
        assert result.content == f"Custom prompt\n\n{BASE_AGENT_PROMPT}"

    def test_system_message_with_list_content(self) -> None:
        """Preserves block structure when content is a list."""
        blocks = [
            {"type": "text", "text": "First block"},
            {"type": "text", "text": "Second block"},
        ]
        input_msg = SystemMessage(content=blocks)
        result = _merge_system_prompt(input_msg)
        assert isinstance(result, SystemMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 3
        assert result.content[0] == {"type": "text", "text": "First block"}
        assert result.content[1] == {"type": "text", "text": "Second block"}
        assert result.content[2] == {"type": "text", "text": BASE_AGENT_PROMPT}

    def test_preserves_metadata(self) -> None:
        """Preserves SystemMessage metadata like name and additional_kwargs."""
        input_msg = SystemMessage(
            content="Custom prompt",
            name="test_name",
            additional_kwargs={"custom_key": "custom_value"},
        )
        result = _merge_system_prompt(input_msg)
        assert result.name == "test_name"
        assert result.additional_kwargs == {"custom_key": "custom_value"}

    def test_preserves_metadata_with_list_content(self) -> None:
        """Preserves metadata when content is a list."""
        blocks = [{"type": "text", "text": "Block"}]
        input_msg = SystemMessage(
            content=blocks,
            name="test_name",
            additional_kwargs={"key": "value"},
        )
        result = _merge_system_prompt(input_msg)
        assert result.name == "test_name"
        assert result.additional_kwargs == {"key": "value"}
        assert isinstance(result.content, list)
