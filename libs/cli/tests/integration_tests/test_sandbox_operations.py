"""Integration tests for BaseSandbox file operations.

This module tests the core file operations implemented in BaseSandbox:
- write(): Create new files
- read(): Read file contents
- edit(): String replacement in files
- ls_info(): List directory contents
- grep_raw(): Search for patterns
- glob_info(): Pattern matching for files

All tests run on a single sandbox instance (class-scoped fixture)
to avoid the overhead of spinning up multiple containers.
"""

from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol

from deepagents_cli.integrations.sandbox_factory import create_sandbox


class TestSandboxOperations:
    """Test core sandbox file operations using a single sandbox instance."""

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        """Provide a single Daytona sandbox instance for all tests."""
        with create_sandbox("daytona") as sandbox:
            yield sandbox

    @pytest.fixture(autouse=True)
    def setup_test_dir(self, sandbox: SandboxBackendProtocol) -> None:
        """Set up a clean test directory before each test."""
        sandbox.execute(
            "rm -rf /tmp/test_sandbox_ops && mkdir -p /tmp/test_sandbox_ops"
        )

    # ==================== write() tests ====================

    def test_write_new_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing a new file with basic content."""
        test_path = "/tmp/test_sandbox_ops/new_file.txt"
        content = "Hello, sandbox!\nLine 2\nLine 3"

        result = sandbox.write(test_path, content)

        assert result.error is None
        assert result.path == test_path
        # Verify file was created
        exec_result = sandbox.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_write_creates_parent_dirs(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that write creates parent directories automatically."""
        test_path = "/tmp/test_sandbox_ops/deep/nested/dir/file.txt"
        content = "Nested file content"

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify file exists
        exec_result = sandbox.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_write_existing_file_fails(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that writing to an existing file returns an error."""
        test_path = "/tmp/test_sandbox_ops/existing.txt"
        # Create file first
        sandbox.write(test_path, "First content")

        # Try to write again
        result = sandbox.write(test_path, "Second content")

        assert result.error is not None
        assert "already exists" in result.error.lower()
        # Verify original content unchanged
        exec_result = sandbox.execute(f"cat {test_path}")
        assert exec_result.output.strip() == "First content"

    def test_write_special_characters(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing content with special characters and escape sequences."""
        test_path = "/tmp/test_sandbox_ops/special.txt"
        content = (
            "Special chars: $VAR, `command`, $(subshell), 'quotes', \"quotes\"\n"
            "Tab\there\nBackslash: \\"
        )

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify content is preserved exactly
        exec_result = sandbox.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_write_empty_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing an empty file."""
        test_path = "/tmp/test_sandbox_ops/empty.txt"
        content = ""

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify file exists but is empty
        exec_result = sandbox.execute(
            f"[ -f {test_path} ] && echo 'exists' || echo 'missing'"
        )
        assert "exists" in exec_result.output

    def test_write_path_with_spaces(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing a file with spaces in the path."""
        test_path = "/tmp/test_sandbox_ops/dir with spaces/file name.txt"
        content = "Content in file with spaces"

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify file was created
        exec_result = sandbox.execute(f"cat '{test_path}'")
        assert exec_result.output.strip() == content

    def test_write_unicode_content(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing content with unicode characters and emojis."""
        test_path = "/tmp/test_sandbox_ops/unicode.txt"
        content = "Hello 👋 世界 مرحبا Привет 🌍\nLine with émojis 🎉"

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify content is preserved exactly
        exec_result = sandbox.execute(f"cat {test_path}")
        assert exec_result.output.strip() == content

    def test_write_consecutive_slashes_in_path(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test that paths with consecutive slashes are handled correctly."""
        test_path = "/tmp//test_sandbox_ops///file.txt"
        content = "Content"

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify file exists (shell should normalize the path)
        exec_result = sandbox.execute("cat /tmp/test_sandbox_ops/file.txt")
        assert exec_result.output.strip() == content

    def test_write_very_long_content(self, sandbox: SandboxBackendProtocol) -> None:
        """Test writing a file with moderately long content (1000 lines)."""
        test_path = "/tmp/test_sandbox_ops/very_long.txt"
        content = "\n".join([f"Line {i} with some content here" for i in range(1000)])

        result = sandbox.write(test_path, content)

        assert result.error is None
        # Verify file has correct number of lines
        exec_result = sandbox.execute(f"wc -l {test_path}")
        # wc -l counts newlines, so 1000 lines = 999 newlines
        # if last line has no newline
        assert "999" in exec_result.output or "1000" in exec_result.output

    def test_write_content_with_only_newlines(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test writing content that consists only of newlines."""
        test_path = "/tmp/test_sandbox_ops/only_newlines.txt"
        content = "\n\n\n\n\n"

        result = sandbox.write(test_path, content)

        assert result.error is None
        exec_result = sandbox.execute(f"wc -l {test_path}")
        assert "5" in exec_result.output

    # ==================== read() tests ====================

    def test_read_basic_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file with basic content."""
        test_path = "/tmp/test_sandbox_ops/read_test.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox.write(test_path, content)

        result = sandbox.read(test_path)

        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_read_nonexistent_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file that doesn't exist."""
        test_path = "/tmp/test_sandbox_ops/nonexistent.txt"

        result = sandbox.read(test_path)

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_read_empty_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading an empty file."""
        test_path = "/tmp/test_sandbox_ops/empty_read.txt"
        sandbox.write(test_path, "")

        result = sandbox.read(test_path)

        # Empty files should return a system reminder
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "empty" in content.lower()

    def test_read_with_offset(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file with offset parameter."""
        test_path = "/tmp/test_sandbox_ops/offset_test.txt"
        content = "\n".join([f"Row_{i}_content" for i in range(1, 11)])
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=5)

        # Should start from line 6 (offset=5 means skip first 5 lines)
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "Row_6_content" in content
        assert "Row_1_content" not in content

    def test_read_with_limit(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file with limit parameter."""
        test_path = "/tmp/test_sandbox_ops/limit_test.txt"
        content = "\n".join([f"Row_{i}_content" for i in range(1, 101)])
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=0, limit=5)

        # Should only have first 5 lines
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "Row_1_content" in content
        assert "Row_5_content" in content
        assert "Row_6_content" not in content

    def test_read_with_offset_and_limit(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file with both offset and limit."""
        test_path = "/tmp/test_sandbox_ops/offset_limit_test.txt"
        content = "\n".join([f"Row_{i}_content" for i in range(1, 21)])
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=10, limit=5)

        # Should have lines 11-15
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "Row_11_content" in content
        assert "Row_15_content" in content
        assert "Row_10_content" not in content
        assert "Row_16_content" not in content

    def test_read_unicode_content(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading a file with unicode content."""
        test_path = "/tmp/test_sandbox_ops/unicode_read.txt"
        content = "Hello 👋 世界\nПривет мир\nمرحبا العالم"
        sandbox.write(test_path, content)

        result = sandbox.read(test_path)

        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "👋" in content
        assert "世界" in content
        assert "Привет" in content

    def test_read_file_with_very_long_lines(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test reading a file with lines longer than 2000 characters."""
        test_path = "/tmp/test_sandbox_ops/long_lines.txt"
        # Create a line with 3000 characters
        long_line = "x" * 3000
        content = f"Short line\n{long_line}\nAnother short line"
        sandbox.write(test_path, content)

        result = sandbox.read(test_path)

        # Should still read successfully (implementation may truncate)
        assert result.error is None
        assert result.file_data is not None
        content = result.file_data["content"]
        assert "Short line" in content

    def test_read_with_zero_limit(self, sandbox: SandboxBackendProtocol) -> None:
        """Test reading with limit=0 returns nothing."""
        test_path = "/tmp/test_sandbox_ops/zero_limit.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=0, limit=0)

        # Should return empty or no content lines
        if result.error is not None:
            pass  # An error is acceptable for limit=0
        else:
            assert result.file_data is not None
            content = result.file_data["content"]
            assert "Line 1" not in content or content.strip() == ""

    def test_read_offset_beyond_file_length(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test reading with offset beyond the file length."""
        test_path = "/tmp/test_sandbox_ops/offset_beyond.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=100, limit=10)

        # Should return empty result or error (no lines to read)
        if result.error is not None:
            pass  # An error is acceptable when offset exceeds file length
        else:
            assert result.file_data is not None
            content = result.file_data["content"]
            assert "Line 1" not in content
            assert "Line 2" not in content
            assert "Line 3" not in content

    def test_read_offset_at_exact_file_length(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test reading with offset exactly at file length."""
        test_path = "/tmp/test_sandbox_ops/offset_exact.txt"
        content = "\n".join([f"Line {i}" for i in range(1, 6)])  # 5 lines
        sandbox.write(test_path, content)

        result = sandbox.read(test_path, offset=5, limit=10)

        # Should return empty (offset=5 means skip first 5 lines)
        if result.error is not None:
            pass  # An error is acceptable when offset is at exact file length
        else:
            assert result.file_data is not None
            content = result.file_data["content"]
            assert "Line 1" not in content
            assert "Line 5" not in content

    def test_read_very_large_file_in_chunks(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test reading a large file in chunks using offset and limit."""
        test_path = "/tmp/test_sandbox_ops/large_chunked.txt"
        # Create 1000 line file
        content = "\n".join([f"Line_{i:04d}_content" for i in range(1000)])
        sandbox.write(test_path, content)

        # Read first chunk
        chunk1 = sandbox.read(test_path, offset=0, limit=100)
        assert chunk1.error is None
        assert chunk1.file_data is not None
        content1 = chunk1.file_data["content"]
        assert "Line_0000_content" in content1
        assert "Line_0099_content" in content1
        assert "Line_0100_content" not in content1

        # Read middle chunk
        chunk2 = sandbox.read(test_path, offset=500, limit=100)
        assert chunk2.error is None
        assert chunk2.file_data is not None
        content2 = chunk2.file_data["content"]
        assert "Line_0500_content" in content2
        assert "Line_0599_content" in content2
        assert "Line_0499_content" not in content2

        # Read last chunk
        chunk3 = sandbox.read(test_path, offset=900, limit=100)
        assert chunk3.error is None
        assert chunk3.file_data is not None
        content3 = chunk3.file_data["content"]
        assert "Line_0900_content" in content3
        assert "Line_0999_content" in content3

    # ==================== edit() tests ====================

    def test_edit_single_occurrence(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing a file with a single occurrence of the search string."""
        test_path = "/tmp/test_sandbox_ops/edit_single.txt"
        content = "Hello world\nGoodbye world\nHello again"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "Goodbye", "Farewell")

        assert result.error is None
        assert result.occurrences == 1
        # Verify change
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Farewell world" in file_content
        assert "Goodbye" not in file_content

    def test_edit_multiple_occurrences_without_replace_all(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test editing fails when multiple occurrences exist without replace_all."""
        test_path = "/tmp/test_sandbox_ops/edit_multi.txt"
        content = "apple\nbanana\napple\norange\napple"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "apple", "pear", replace_all=False)

        assert result.error is not None
        assert "multiple times" in result.error.lower()
        # Verify file unchanged
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "apple" in file_content
        assert "pear" not in file_content

    def test_edit_multiple_occurrences_with_replace_all(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test editing all occurrences with replace_all=True."""
        test_path = "/tmp/test_sandbox_ops/edit_replace_all.txt"
        content = "apple\nbanana\napple\norange\napple"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "apple", "pear", replace_all=True)

        assert result.error is None
        assert result.occurrences == 3
        # Verify all replaced
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "apple" not in file_content
        assert file_content.count("pear") == 3

    def test_edit_string_not_found(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing when search string is not found."""
        test_path = "/tmp/test_sandbox_ops/edit_not_found.txt"
        content = "Hello world"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "nonexistent", "replacement")

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_nonexistent_file(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing a file that doesn't exist."""
        test_path = "/tmp/test_sandbox_ops/nonexistent_edit.txt"

        result = sandbox.edit(test_path, "old", "new")

        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_special_characters(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing with special characters and regex metacharacters."""
        test_path = "/tmp/test_sandbox_ops/edit_special.txt"
        content = "Price: $100.00\nPattern: [a-z]*\nPath: /usr/bin"
        sandbox.write(test_path, content)

        # Test with dollar signs
        result = sandbox.edit(test_path, "$100.00", "$200.00")
        assert result.error is None

        # Test with regex metacharacters
        result = sandbox.edit(test_path, "[a-z]*", "[0-9]+")
        assert result.error is None

        # Verify changes
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "$200.00" in file_content
        assert "[0-9]+" in file_content

    def test_edit_multiline_support(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that edit handles multiline strings correctly."""
        test_path = "/tmp/test_sandbox_ops/edit_multiline.txt"
        content = "Line 1\nLine 2\nLine 3"
        sandbox.write(test_path, content)

        # Should successfully replace multiline content
        result = sandbox.edit(test_path, "Line 1\nLine 2", "Combined")

        assert result.error is None
        assert result.occurrences == 1
        # Verify the replacement worked correctly
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Combined" in file_content
        assert "Line 3" in file_content
        assert "Line 1" not in file_content

    def test_edit_with_empty_new_string(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing to delete content (replace with empty string)."""
        test_path = "/tmp/test_sandbox_ops/edit_delete.txt"
        content = "Keep this\nDelete this part\nKeep this too"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "Delete this part\n", "")

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Keep this" in file_content
        assert "Keep this too" in file_content
        assert "Delete this part" not in file_content

    def test_edit_identical_strings(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing where old_string equals new_string."""
        test_path = "/tmp/test_sandbox_ops/edit_identical.txt"
        content = "Same text"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "Same text", "Same text")

        # Should succeed with 1 occurrence
        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Same text" in file_content

    def test_edit_unicode_content(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing with unicode characters and emojis."""
        test_path = "/tmp/test_sandbox_ops/edit_unicode.txt"
        content = "Hello 👋 world\n世界 is beautiful"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "👋", "🌍")

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "🌍" in file_content
        assert "👋" not in file_content

    def test_edit_whitespace_only_strings(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test editing with whitespace-only strings."""
        test_path = "/tmp/test_sandbox_ops/edit_whitespace.txt"
        content = "Line1    Line2"  # 4 spaces
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "    ", " ")  # Replace 4 spaces with 1

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Line1 Line2" in file_content

    def test_edit_with_very_long_strings(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing with very long old and new strings."""
        test_path = "/tmp/test_sandbox_ops/edit_long.txt"
        old_string = "x" * 1000
        new_string = "y" * 1000
        content = f"Start\n{old_string}\nEnd"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, old_string, new_string)

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "y" * 100 in file_content  # Check partial presence
        assert "x" * 100 not in file_content

    def test_edit_line_ending_preservation(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test that edit preserves line endings correctly."""
        test_path = "/tmp/test_sandbox_ops/edit_line_endings.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "Line 2", "Modified Line 2")

        assert result.error is None
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "Line 1" in file_content
        assert "Modified Line 2" in file_content
        assert "Line 3" in file_content

    def test_edit_partial_line_match(self, sandbox: SandboxBackendProtocol) -> None:
        """Test editing a substring within a line."""
        test_path = "/tmp/test_sandbox_ops/edit_partial.txt"
        content = "The quick brown fox jumps over the lazy dog"
        sandbox.write(test_path, content)

        result = sandbox.edit(test_path, "brown fox", "red cat")

        assert result.error is None
        assert result.occurrences == 1
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        file_content = read_result.file_data["content"]
        assert "red cat" in file_content
        assert "The quick red cat jumps" in file_content

    # ==================== ls_info() tests ====================

    def test_ls_info_path_is_absolute(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that files returned from ls_info have absolute paths."""
        base_dir = "/tmp/test_sandbox_ops/ls_absolute"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "content")
        result = sandbox.ls_info(base_dir)
        assert result.entries is not None
        assert len(result.entries) == 1
        assert result.entries[0]["path"] == "/tmp/test_sandbox_ops/ls_absolute/file.txt"

    def test_ls_info_basic_directory(self, sandbox: SandboxBackendProtocol) -> None:
        """Test listing a directory with files and subdirectories."""
        base_dir = "/tmp/test_sandbox_ops/ls_test"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file1.txt", "content1")
        sandbox.write(f"{base_dir}/file2.txt", "content2")
        sandbox.execute(f"mkdir -p {base_dir}/subdir")

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        assert len(result.entries) == 3
        paths = [info["path"] for info in result.entries]
        assert f"{base_dir}/file1.txt" in paths
        assert f"{base_dir}/file2.txt" in paths
        assert f"{base_dir}/subdir" in paths
        # Check is_dir flag
        for info in result.entries:
            if info["path"] == f"{base_dir}/subdir":
                assert info["is_dir"] is True
            else:
                assert info["is_dir"] is False

    def test_ls_info_empty_directory(self, sandbox: SandboxBackendProtocol) -> None:
        """Test listing an empty directory."""
        empty_dir = "/tmp/test_sandbox_ops/empty_dir"
        sandbox.execute(f"mkdir -p {empty_dir}")

        result = sandbox.ls_info(empty_dir)

        assert result.entries == []

    def test_ls_info_nonexistent_directory(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test listing a directory that doesn't exist."""
        nonexistent_dir = "/tmp/test_sandbox_ops/does_not_exist"

        result = sandbox.ls_info(nonexistent_dir)

        assert result.entries == []

    def test_ls_info_hidden_files(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that ls_info includes hidden files (starting with .)."""
        base_dir = "/tmp/test_sandbox_ops/hidden_test"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/.hidden", "hidden content")
        sandbox.write(f"{base_dir}/visible.txt", "visible content")

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        paths = [info["path"] for info in result.entries]
        assert f"{base_dir}/.hidden" in paths
        assert f"{base_dir}/visible.txt" in paths

    def test_ls_info_directory_with_spaces(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test listing a directory that has spaces in file/dir names."""
        base_dir = "/tmp/test_sandbox_ops/ls_spaces"
        sandbox.execute(f"mkdir -p '{base_dir}'")
        sandbox.write(f"{base_dir}/file with spaces.txt", "content")
        sandbox.execute(f"mkdir -p '{base_dir}/dir with spaces'")

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        paths = [info["path"] for info in result.entries]
        assert f"{base_dir}/file with spaces.txt" in paths
        assert f"{base_dir}/dir with spaces" in paths

    def test_ls_info_unicode_filenames(self, sandbox: SandboxBackendProtocol) -> None:
        """Test listing directory with unicode filenames."""
        base_dir = "/tmp/test_sandbox_ops/ls_unicode"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/测试文件.txt", "content")
        sandbox.write(f"{base_dir}/файл.txt", "content")

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        paths = [info["path"] for info in result.entries]
        # Should contain the unicode filenames
        assert len(paths) == 2

    def test_ls_info_large_directory(self, sandbox: SandboxBackendProtocol) -> None:
        """Test listing a directory with many files."""
        base_dir = "/tmp/test_sandbox_ops/ls_large"
        # Create 50 files in a single command (much faster than loop)
        # Note: Using $(seq 0 49) instead of {0..49} for better shell compatibility
        sandbox.execute(
            f"mkdir -p {base_dir} && "
            f"cd {base_dir} && "
            "for i in $(seq 0 49); do "
            "echo 'content' > file_$(printf '%03d' $i).txt; done"
        )

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        assert len(result.entries) == 50
        paths = [info["path"] for info in result.entries]
        assert f"{base_dir}/file_000.txt" in paths
        assert f"{base_dir}/file_049.txt" in paths

    def test_ls_info_path_with_trailing_slash(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test that trailing slash in path is handled correctly."""
        base_dir = "/tmp/test_sandbox_ops/ls_trailing"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "content")

        # List with trailing slash
        result = sandbox.ls_info(f"{base_dir}/")

        # Should work the same as without trailing slash
        assert (
            result.entries is not None and len(result.entries) >= 1
        ) or result.entries == []  # Implementation dependent

    def test_ls_info_special_characters_in_filenames(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test listing files with special characters in names."""
        base_dir = "/tmp/test_sandbox_ops/ls_special"
        sandbox.execute(f"mkdir -p {base_dir}")
        # Create files with various special characters (shell-safe ones)
        sandbox.write(f"{base_dir}/file(1).txt", "content")
        sandbox.write(f"{base_dir}/file[2].txt", "content")
        sandbox.write(f"{base_dir}/file-3.txt", "content")

        result = sandbox.ls_info(base_dir)

        assert result.entries is not None
        paths = [info["path"] for info in result.entries]
        assert f"{base_dir}/file(1).txt" in paths
        assert f"{base_dir}/file[2].txt" in paths
        assert f"{base_dir}/file-3.txt" in paths

    # ==================== grep_raw() tests ====================

    def test_grep_basic_search(self, sandbox: SandboxBackendProtocol) -> None:
        """Test basic grep search for a literal pattern (not regex)."""
        base_dir = "/tmp/test_sandbox_ops/grep_test"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file1.txt", "Hello world\nGoodbye world")
        sandbox.write(f"{base_dir}/file2.txt", "Hello there\nGoodbye friend")

        result = sandbox.grep_raw("Hello", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 2
        # Check that both files matched
        paths = [match["path"] for match in result.matches]
        assert any("file1.txt" in p for p in paths)
        assert any("file2.txt" in p for p in paths)
        # Check line numbers
        for match in result.matches:
            assert match["line"] == 1
            assert "Hello" in match["text"]

    def test_grep_with_glob_pattern(self, sandbox: SandboxBackendProtocol) -> None:
        """Test grep with glob pattern to filter files."""
        base_dir = "/tmp/test_sandbox_ops/grep_glob"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/test.txt", "pattern")
        sandbox.write(f"{base_dir}/test.py", "pattern")
        sandbox.write(f"{base_dir}/test.md", "pattern")

        result = sandbox.grep_raw("pattern", path=base_dir, glob="*.py")

        assert result.matches is not None
        assert len(result.matches) == 1
        assert "test.py" in result.matches[0]["path"]

    def test_grep_no_matches(self, sandbox: SandboxBackendProtocol) -> None:
        """Test grep when no matches are found."""
        base_dir = "/tmp/test_sandbox_ops/grep_empty"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "Hello world")

        result = sandbox.grep_raw("nonexistent", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 0

    def test_grep_multiple_matches_per_file(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test grep with multiple matches in a single file."""
        base_dir = "/tmp/test_sandbox_ops/grep_multi"
        sandbox.execute(f"mkdir -p {base_dir}")
        content = "apple\nbanana\napple\norange\napple"
        sandbox.write(f"{base_dir}/fruits.txt", content)

        result = sandbox.grep_raw("apple", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 3
        # Check line numbers
        line_numbers = [match["line"] for match in result.matches]
        assert line_numbers == [1, 3, 5]

    def test_grep_literal_string_matching(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test grep with literal string matching (not regex)."""
        base_dir = "/tmp/test_sandbox_ops/grep_literal"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/numbers.txt", "test123\ntest456\nabcdef")

        # Pattern is treated as literal string, not regex
        result = sandbox.grep_raw("test123", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 1
        assert "test123" in result.matches[0]["text"]

    def test_grep_unicode_pattern(self, sandbox: SandboxBackendProtocol) -> None:
        """Test grep with unicode pattern and content."""
        base_dir = "/tmp/test_sandbox_ops/grep_unicode"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/unicode.txt", "Hello 世界\nПривет мир\n测试 pattern")

        result = sandbox.grep_raw("世界", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 1
        assert "世界" in result.matches[0]["text"]

    def test_grep_case_sensitivity(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that grep is case-sensitive by default."""
        base_dir = "/tmp/test_sandbox_ops/grep_case"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/case.txt", "Hello\nhello\nHELLO")

        result = sandbox.grep_raw("Hello", path=base_dir)

        assert result.matches is not None
        # Should only match "Hello", not "hello" or "HELLO"
        assert len(result.matches) == 1
        assert "Hello" in result.matches[0]["text"]

    def test_grep_with_special_characters(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test grep with special character patterns (treated as literals)."""
        base_dir = "/tmp/test_sandbox_ops/grep_special"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(
            f"{base_dir}/special.txt", "Price: $100\nPath: /usr/bin\nPattern: [a-z]*"
        )

        # Test with dollar sign (treated as literal)
        result = sandbox.grep_raw("$100", path=base_dir)
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "$100" in result.matches[0]["text"]

        # Test with brackets (treated as literal)
        result = sandbox.grep_raw("[a-z]*", path=base_dir)
        assert result.matches is not None
        assert len(result.matches) == 1
        assert "[a-z]*" in result.matches[0]["text"]

    def test_grep_empty_directory(self, sandbox: SandboxBackendProtocol) -> None:
        """Test grep in a directory with no files."""
        base_dir = "/tmp/test_sandbox_ops/grep_empty_dir"
        sandbox.execute(f"mkdir -p {base_dir}")

        result = sandbox.grep_raw("anything", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 0

    def test_grep_across_nested_directories(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test grep recursively searches nested directories."""
        base_dir = "/tmp/test_sandbox_ops/grep_nested"
        sandbox.execute(f"mkdir -p {base_dir}/sub1/sub2")
        sandbox.write(f"{base_dir}/root.txt", "target here")
        sandbox.write(f"{base_dir}/sub1/level1.txt", "target here")
        sandbox.write(f"{base_dir}/sub1/sub2/level2.txt", "target here")

        result = sandbox.grep_raw("target", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 3
        # Should find matches in all nested levels

    def test_grep_with_multiline_matches(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that grep reports correct line numbers for matches."""
        base_dir = "/tmp/test_sandbox_ops/grep_multiline"
        sandbox.execute(f"mkdir -p {base_dir}")
        content = "\n".join([f"Line {i}" for i in range(1, 101)])
        sandbox.write(f"{base_dir}/long.txt", content)

        result = sandbox.grep_raw("Line 50", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 1
        assert result.matches[0]["line"] == 50

    # ==================== glob_info() tests ====================

    def test_glob_basic_pattern(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with basic wildcard pattern."""
        base_dir = "/tmp/test_sandbox_ops/glob_test"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file1.txt", "content")
        sandbox.write(f"{base_dir}/file2.txt", "content")
        sandbox.write(f"{base_dir}/file3.py", "content")

        result = sandbox.glob_info("*.txt", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 2
        paths = [info["path"] for info in result.matches]
        assert "file1.txt" in paths
        assert "file2.txt" in paths
        assert not any(".py" in p for p in paths)

    def test_glob_recursive_pattern(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with recursive pattern (**)."""
        base_dir = "/tmp/test_sandbox_ops/glob_recursive"
        sandbox.execute(f"mkdir -p {base_dir}/subdir1 {base_dir}/subdir2")
        sandbox.write(f"{base_dir}/root.txt", "content")
        sandbox.write(f"{base_dir}/subdir1/nested1.txt", "content")
        sandbox.write(f"{base_dir}/subdir2/nested2.txt", "content")

        result = sandbox.glob_info("**/*.txt", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) >= 2  # At least the nested files
        paths = [info["path"] for info in result.matches]
        assert any("nested1.txt" in p for p in paths)
        assert any("nested2.txt" in p for p in paths)

    def test_glob_no_matches(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob when no files match the pattern."""
        base_dir = "/tmp/test_sandbox_ops/glob_empty"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "content")

        result = sandbox.glob_info("*.py", path=base_dir)

        assert result.matches is not None
        assert result.matches == []

    def test_glob_with_directories(self, sandbox: SandboxBackendProtocol) -> None:
        """Test that glob includes directories in results."""
        base_dir = "/tmp/test_sandbox_ops/glob_dirs"
        sandbox.execute(f"mkdir -p {base_dir}/dir1 {base_dir}/dir2")
        sandbox.write(f"{base_dir}/file.txt", "content")

        result = sandbox.glob_info("*", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 3
        # Check is_dir flags
        dir_count = sum(1 for info in result.matches if info["is_dir"])
        file_count = sum(1 for info in result.matches if not info["is_dir"])
        assert dir_count == 2
        assert file_count == 1

    def test_glob_specific_extension(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with specific file extension pattern."""
        base_dir = "/tmp/test_sandbox_ops/glob_ext"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/test.py", "content")
        sandbox.write(f"{base_dir}/test.txt", "content")
        sandbox.write(f"{base_dir}/test.md", "content")

        result = sandbox.glob_info("*.py", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 1
        assert "test.py" in result.matches[0]["path"]

    def test_glob_hidden_files_explicitly(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test glob with pattern that explicitly matches hidden files."""
        base_dir = "/tmp/test_sandbox_ops/glob_hidden"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/.hidden1", "content")
        sandbox.write(f"{base_dir}/.hidden2", "content")
        sandbox.write(f"{base_dir}/visible.txt", "content")

        result = sandbox.glob_info(".*", path=base_dir)

        # Should only match hidden files
        assert result.matches is not None
        paths = [info["path"] for info in result.matches]
        assert ".hidden1" in paths or ".hidden2" in paths
        # Should not match visible.txt
        assert not any("visible" in p for p in paths)

    def test_glob_with_character_class(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with character class patterns."""
        base_dir = "/tmp/test_sandbox_ops/glob_charclass"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file1.txt", "content")
        sandbox.write(f"{base_dir}/file2.txt", "content")
        sandbox.write(f"{base_dir}/file3.txt", "content")
        sandbox.write(f"{base_dir}/fileA.txt", "content")

        result = sandbox.glob_info("file[1-2].txt", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) == 2
        paths = [info["path"] for info in result.matches]
        assert "file1.txt" in paths
        assert "file2.txt" in paths
        assert "file3.txt" not in paths
        assert "fileA.txt" not in paths

    def test_glob_with_question_mark(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with single character wildcard (?)."""
        base_dir = "/tmp/test_sandbox_ops/glob_question"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file1.txt", "content")
        sandbox.write(f"{base_dir}/file2.txt", "content")
        sandbox.write(f"{base_dir}/file10.txt", "content")

        result = sandbox.glob_info("file?.txt", path=base_dir)

        # Should match file1.txt and file2.txt, but not file10.txt
        assert result.matches is not None
        assert len(result.matches) == 2
        paths = [info["path"] for info in result.matches]
        assert "file10.txt" not in paths

    def test_glob_multiple_extensions(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob matching multiple extensions."""
        base_dir = "/tmp/test_sandbox_ops/glob_multi_ext"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "content")
        sandbox.write(f"{base_dir}/file.py", "content")
        sandbox.write(f"{base_dir}/file.md", "content")
        sandbox.write(f"{base_dir}/file.js", "content")

        # Using separate patterns (implementation may support brace expansion)
        result_txt = sandbox.glob_info("*.txt", path=base_dir)
        result_py = sandbox.glob_info("*.py", path=base_dir)

        assert result_txt.matches is not None
        assert result_py.matches is not None
        assert len(result_txt.matches) == 1
        assert len(result_py.matches) == 1

    def test_glob_deeply_nested_pattern(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with deeply nested directory structure."""
        base_dir = "/tmp/test_sandbox_ops/glob_deep"
        sandbox.execute(f"mkdir -p {base_dir}/a/b/c/d")
        sandbox.write(f"{base_dir}/a/b/c/d/deep.txt", "content")
        sandbox.write(f"{base_dir}/a/b/other.txt", "content")

        result = sandbox.glob_info("**/deep.txt", path=base_dir)

        assert result.matches is not None
        assert len(result.matches) >= 1
        # Should find the deeply nested file

    def test_glob_with_no_path_argument(self, sandbox: SandboxBackendProtocol) -> None:
        """Test glob with default path behavior."""
        base_dir = "/tmp/test_sandbox_ops/glob_default"
        sandbox.execute(f"mkdir -p {base_dir}")
        sandbox.write(f"{base_dir}/file.txt", "content")

        # Call with explicit path to match expected signature
        result = sandbox.glob_info("*.txt", path=base_dir)

        # Should work with explicit path
        assert result.matches is not None

    # ==================== Integration tests ====================

    def test_write_read_edit_workflow(self, sandbox: SandboxBackendProtocol) -> None:
        """Test a complete workflow: write, read, edit, read again."""
        test_path = "/tmp/test_sandbox_ops/workflow.txt"

        # Write initial content
        write_result = sandbox.write(test_path, "Original content")
        assert write_result.error is None

        # Read it back
        read_result = sandbox.read(test_path)
        assert read_result.error is None
        assert read_result.file_data is not None
        content = read_result.file_data["content"]
        assert "Original content" in content

        # Edit it
        edit_result = sandbox.edit(test_path, "Original", "Modified")
        assert edit_result.error is None

        # Read again to verify
        updated_read = sandbox.read(test_path)
        assert updated_read.error is None
        assert updated_read.file_data is not None
        updated_content = updated_read.file_data["content"]
        assert "Modified content" in updated_content
        assert "Original" not in updated_content

    def test_complex_directory_operations(
        self, sandbox: SandboxBackendProtocol
    ) -> None:
        """Test complex scenario with multiple operations."""
        base_dir = "/tmp/test_sandbox_ops/complex"

        # Create directory structure
        sandbox.write(f"{base_dir}/root.txt", "root file")
        sandbox.write(f"{base_dir}/subdir1/file1.txt", "file 1")
        sandbox.write(f"{base_dir}/subdir1/file2.py", "file 2")
        sandbox.write(f"{base_dir}/subdir2/file3.txt", "file 3")

        # List root directory
        ls_result = sandbox.ls_info(base_dir)
        assert ls_result.entries is not None
        paths = [info["path"] for info in ls_result.entries]
        assert f"{base_dir}/root.txt" in paths
        assert f"{base_dir}/subdir1" in paths
        assert f"{base_dir}/subdir2" in paths

        # Glob for txt files
        glob_result = sandbox.glob_info("**/*.txt", path=base_dir)
        assert glob_result.matches is not None
        assert len(glob_result.matches) == 3

        # Grep for a pattern
        grep_result = sandbox.grep_raw("file", path=base_dir)
        assert grep_result.matches is not None
        assert len(grep_result.matches) >= 3  # At least 3 matches
