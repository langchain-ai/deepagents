"""Unit tests for filesystem middleware helper functions."""

import pytest

from deepagents.middleware.filesystem import _truncate_lines


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
