"""Tests for input parsing functions (no mocks needed)."""

from pathlib import Path

import pytest

from deepagents_cli.input import parse_file_mentions


class TestParseFileMentions:
    """Tests for parse_file_mentions function."""

    def test_no_mentions(self) -> None:
        """Test text with no @ mentions."""
        text, files = parse_file_mentions("hello world")
        assert text == "hello world"
        assert files == []

    def test_single_file_mention(self, tmp_path: Path) -> None:
        """Test single @file mention."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        text, files = parse_file_mentions(f"Read @{test_file}")
        assert text == f"Read @{test_file}"
        assert len(files) == 1
        assert files[0] == test_file.resolve()

    def test_multiple_file_mentions(self, tmp_path: Path) -> None:
        """Test multiple @file mentions."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        text, files = parse_file_mentions(f"Read @{file1} and @{file2}")
        assert len(files) == 2
        assert file1.resolve() in files
        assert file2.resolve() in files

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test @mention of nonexistent file."""
        fake_file = tmp_path / "nonexistent.txt"
        text, files = parse_file_mentions(f"Read @{fake_file}")
        assert files == []  # Nonexistent files are filtered out

    def test_file_with_spaces_escaped(self, tmp_path: Path) -> None:
        """Test file with escaped spaces."""
        file_with_space = tmp_path / "my file.txt"
        file_with_space.write_text("content")

        # Use escaped space
        escaped_path = str(file_with_space).replace(" ", r"\ ")
        text, files = parse_file_mentions(f"Read @{escaped_path}")
        assert len(files) == 1
        assert files[0].name == "my file.txt"

    def test_relative_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test relative path resolution."""
        # Change to tmp_path
        monkeypatch.chdir(tmp_path)

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Use relative path
        text, files = parse_file_mentions("Read @test.txt")
        assert len(files) == 1
        assert files[0].name == "test.txt"

    def test_absolute_path(self, tmp_path: Path) -> None:
        """Test absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        text, files = parse_file_mentions(f"Read @{test_file.absolute()}")
        assert len(files) == 1
        assert files[0] == test_file.resolve()

    def test_expanduser_tilde(self) -> None:
        """Test ~ expansion in paths uses user's home directory."""
        # Create a test file in actual home directory (if accessible)
        # This is a lightweight test - just verify the function handles ~
        # We can't easily mock expanduser, so we just test it doesn't crash
        text, files = parse_file_mentions("Read @~/nonexistent_test_file_12345.txt")
        # File won't exist, so should return empty list
        assert files == []
        # But text should be preserved
        assert text == "Read @~/nonexistent_test_file_12345.txt"

    def test_directory_not_included(self, tmp_path: Path) -> None:
        """Test that directories are not included (only files)."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        text, files = parse_file_mentions(f"Read @{test_dir}")
        assert files == []  # Directories should not be included

    def test_multiple_at_symbols(self, tmp_path: Path) -> None:
        """Test text with multiple @ symbols (not all file mentions)."""
        test_file = tmp_path / "file.txt"
        test_file.write_text("content")

        text, files = parse_file_mentions(f"Email me@example.com and read @{test_file}")
        # Should parse both @ mentions, but only the valid file is included
        assert len(files) == 1
        assert files[0].name == "file.txt"

    def test_at_symbol_at_end(self) -> None:
        """Test @ symbol at end of text."""
        text, files = parse_file_mentions("Send to user@")
        assert files == []

    def test_empty_string(self) -> None:
        """Test empty string input."""
        text, files = parse_file_mentions("")
        assert text == ""
        assert files == []

    def test_text_preserved(self, tmp_path: Path) -> None:
        """Test that original text is returned unchanged."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        original = f"Please read @{test_file} carefully"
        text, files = parse_file_mentions(original)
        assert text == original  # Text should be unchanged

    def test_special_characters_in_filename(self, tmp_path: Path) -> None:
        """Test file with special characters."""
        special_file = tmp_path / "file-name_123.txt"
        special_file.write_text("content")

        text, files = parse_file_mentions(f"Read @{special_file}")
        assert len(files) == 1
        assert files[0].name == "file-name_123.txt"
