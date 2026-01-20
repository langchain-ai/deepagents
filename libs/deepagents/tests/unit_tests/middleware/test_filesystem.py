"""Unit tests for filesystem middleware helper functions."""

import pytest
from langchain_core.tools import StructuredTool

from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import _get_filesystem_tools, _truncate_lines


def test_no_truncation_when_below_limit() -> None:
    """Test that lines shorter than max_line_length are not modified."""
    text = "short line\nanother short line"
    result = _truncate_lines(text, max_line_length=100)
    assert result == text


def test_truncation_without_suffix() -> None:
    """Test basic truncation without suffix."""
    text = "a" * 100
    result = _truncate_lines(text, max_line_length=10)
    assert result == "a" * 10


def test_truncation_with_suffix() -> None:
    """Test truncation with suffix appended."""
    text = "a" * 100
    result = _truncate_lines(text, max_line_length=10, suffix="...")
    assert result == "a" * 7 + "..."
    assert len(result) == 10  # noqa: PLR2004


def test_truncation_preserves_newlines() -> None:
    """Test that newlines are preserved after truncation."""
    text = "a" * 100 + "\n" + "b" * 100 + "\n"
    result = _truncate_lines(text, max_line_length=10)
    lines = result.splitlines(keepends=True)
    assert lines[0] == "a" * 10 + "\n"
    assert lines[1] == "b" * 10 + "\n"


def test_truncation_multiline_mixed_lengths() -> None:
    """Test mixed line lengths with some needing truncation."""
    text = "short\n" + "a" * 100 + "\nmedium line"
    result = _truncate_lines(text, max_line_length=20)
    lines = result.splitlines(keepends=True)
    assert lines[0] == "short\n"
    assert lines[1] == "a" * 20 + "\n"
    assert lines[2] == "medium line"


def test_truncation_empty_string() -> None:
    """Test that empty string is handled correctly."""
    result = _truncate_lines("", max_line_length=10)
    assert result == ""


def test_truncation_max_line_length_zero() -> None:
    """Test edge case where max_line_length is 0."""
    text = "some text"
    result = _truncate_lines(text, max_line_length=0)
    assert result == ""


def test_truncation_max_line_length_zero_with_suffix() -> None:
    """Test edge case where max_line_length is 0 with suffix."""
    text = "some text"
    result = _truncate_lines(text, max_line_length=0, suffix="...")
    # When max_line_length=0 and suffix exists, cutoff=max(0, 0-3)=0
    assert result == "..."


def test_truncation_negative_max_line_length_raises_error() -> None:
    """Test that negative max_line_length raises ValueError."""
    with pytest.raises(ValueError, match="max_line_length must be non-negative"):
        _truncate_lines("text", max_line_length=-1)


def test_truncation_suffix_longer_than_max_length() -> None:
    """Test behavior when suffix is longer than max_line_length."""
    text = "a" * 100
    result = _truncate_lines(text, max_line_length=2, suffix="...")
    # cutoff = max(0, 2-3) = 0, so we get just the suffix
    assert result == "..."


def test_truncation_preserves_different_newline_types() -> None:
    """Test that different newline types are preserved."""
    text = "line1\nline2\r\nline3"
    result = _truncate_lines(text, max_line_length=100)
    assert result == text
    assert "\r\n" in result


class TestFilesystemToolSchemas:
    """Test that filesystem tool JSON schemas have types and descriptions for all args."""

    def test_all_filesystem_tools_have_arg_descriptions(self) -> None:
        """Verify all filesystem tool args have type and description in JSON schema.

        Uses tool_call_schema.model_json_schema() which is the schema passed to the LLM.
        """
        # Create a mock backend - we just need to generate the tools
        backend = StateBackend(None)  # type: ignore[arg-type]
        tools = _get_filesystem_tools(backend)

        # Expected tools and their user-facing args (excludes `runtime` which is internal)
        expected_tools = {
            "ls": ["path"],
            "read_file": ["file_path", "offset", "limit"],
            "write_file": ["file_path", "content"],
            "edit_file": ["file_path", "old_string", "new_string", "replace_all"],
            "glob": ["pattern", "path"],
            "grep": ["pattern", "path", "glob", "output_mode"],
            "execute": ["command"],
        }

        tool_map = {tool.name: tool for tool in tools}

        for tool_name, expected_args in expected_tools.items():
            assert tool_name in tool_map, f"Tool '{tool_name}' not found in filesystem tools"
            tool = tool_map[tool_name]

            # Get the JSON schema that's passed to the LLM
            schema = tool.tool_call_schema.model_json_schema()
            properties = schema.get("properties", {})

            for arg_name in expected_args:
                assert arg_name in properties, f"Arg '{arg_name}' not found in schema for tool '{tool_name}'"
                arg_schema = properties[arg_name]

                # Check type is present
                has_type = "type" in arg_schema or "anyOf" in arg_schema or "$ref" in arg_schema
                assert has_type, f"Arg '{arg_name}' in tool '{tool_name}' is missing type in JSON schema"

                # Check description is present
                assert "description" in arg_schema, (
                    f"Arg '{arg_name}' in tool '{tool_name}' is missing description in JSON schema. "
                    f"Add an Annotated type hint with a description string."
                )

    def test_sync_async_schema_parity(self) -> None:
        """Verify sync and async functions produce identical JSON schemas.

        This ensures that the sync_* and async_* function pairs would generate
        the same tool schema if used independently.
        """
        backend = StateBackend(None)  # type: ignore[arg-type]
        tools = _get_filesystem_tools(backend)

        for tool in tools:
            # Create temporary tools from sync and async functions independently
            sync_tool = StructuredTool.from_function(func=tool.func, name=f"{tool.name}_sync")
            async_tool = StructuredTool.from_function(coroutine=tool.coroutine, name=f"{tool.name}_async")

            sync_schema = sync_tool.tool_call_schema.model_json_schema()
            async_schema = async_tool.tool_call_schema.model_json_schema()

            # Remove fields that differ by design (title from name, description from docstring)
            for schema in (sync_schema, async_schema):
                schema.pop("title", None)
                schema.pop("description", None)

            assert sync_schema == async_schema, (
                f"Tool '{tool.name}' has mismatched JSON schemas between sync and async functions.\n"
                f"Sync schema: {sync_schema}\n"
                f"Async schema: {async_schema}"
            )
