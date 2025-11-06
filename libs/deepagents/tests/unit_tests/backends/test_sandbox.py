"""Unit tests for BaseSandbox implementation.

Tests the sandbox protocol using a local shell implementation in a temporary directory.
All operations are scoped to temp directories for safety.
"""

import subprocess
import tempfile
from typing import Generator

import pytest

from deepagents.backends.protocol import EditResult, ExecuteResponse, WriteResult
from deepagents.backends.sandbox import BaseSandbox


class LocalShellSandbox(BaseSandbox):
    """Concrete sandbox implementation using local shell for testing.

    All operations are scoped to the working_dir for safety.
    """

    def __init__(self, working_dir: str):
        self.working_dir = working_dir

    def execute(
        self,
        command: str,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute command in working directory using subprocess."""
        try:
            result = subprocess.run(
                command,
                # Please note that shell=True is intentional here!
                # This is a suite of tests used to be able to test sandbox
                # backends in general.
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ExecuteResponse(
                output=result.stdout + result.stderr,
                exit_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output="Command timed out",
                exit_code=-1,
                truncated=True,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Execution error: {str(e)}",
                exit_code=-1,
            )


class TestLocalSandbox:
    """Standard test suite for sandbox implementations.

    Tests the SandboxBackendProtocol using only the standard protocol methods.
    Can be reused for different sandbox implementations.
    """

    @pytest.fixture
    def sandbox(self) -> Generator[BaseSandbox, None, None]:
        """Create a LocalShellSandbox in a temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sandbox = LocalShellSandbox(working_dir=temp_dir)
            yield sandbox

    def test_basic_write_and_read(self, sandbox: BaseSandbox):
        """Test basic file write and read operations."""
        # Write a simple file
        result = sandbox.write("test.txt", "Hello, World!\n")
        assert isinstance(result, WriteResult)
        assert result.error is None
        assert result.path == "test.txt"

        # Read it back
        content = sandbox.read("test.txt")
        assert "Hello, World!" in content
        assert "     1\t" in content  # Check line number format

    def test_write_existing_file_fails(self, sandbox: BaseSandbox):
        """Test that writing to an existing file returns an error."""
        # Create file
        sandbox.write("existing.txt", "original")

        # Try to write again
        result = sandbox.write("existing.txt", "new content")
        assert result.error is not None
        assert "exists" in result.error.lower()

    def test_edit_simple_replacement(self, sandbox: BaseSandbox):
        """Test basic string replacement with edit."""
        # Create file
        sandbox.write("edit.txt", "foo bar baz")

        # Edit it
        result = sandbox.edit("edit.txt", "bar", "qux")
        assert isinstance(result, EditResult)
        assert result.error is None
        assert result.occurrences == 1

        # Verify edit
        content = sandbox.read("edit.txt")
        assert "qux" in content
        assert "bar" not in content or content.count("bar") == 0  # bar should be replaced

    def test_edit_with_single_quotes(self, sandbox: BaseSandbox):
        """Test edit with single quotes in strings."""
        content = "She said 'hello' to me"
        sandbox.write("quotes.txt", content)

        result = sandbox.edit("quotes.txt", "'hello'", "'goodbye'")
        assert result.error is None
        assert result.occurrences == 1

        output = sandbox.read("quotes.txt")
        assert "'goodbye'" in output

    def test_edit_with_double_quotes(self, sandbox: BaseSandbox):
        """Test edit with double quotes in strings."""
        content = 'He said "hello" to me'
        sandbox.write("dquotes.txt", content)

        result = sandbox.edit("dquotes.txt", '"hello"', '"goodbye"')
        assert result.error is None
        assert result.occurrences == 1

        output = sandbox.read("dquotes.txt")
        assert '"goodbye"' in output

    def test_edit_with_mixed_quotes(self, sandbox: BaseSandbox):
        """Test edit with mixed single and double quotes."""
        content = """She said "I'm fine" """
        sandbox.write("mixed.txt", content)

        result = sandbox.edit("mixed.txt", "I'm", "I am")
        assert result.error is None
        assert result.occurrences == 1

        output = sandbox.read("mixed.txt")
        assert "I am" in output

    def test_edit_with_special_chars(self, sandbox: BaseSandbox):
        """Test edit with special shell characters."""
        content = "Cost: $100 & more!"
        sandbox.write("special.txt", content)

        result = sandbox.edit("special.txt", "$100", "$200")
        assert result.error is None
        assert result.occurrences == 1

        output = sandbox.read("special.txt")
        assert "$200" in output

    def test_edit_multiple_occurrences_without_replace_all(self, sandbox: BaseSandbox):
        """Test that multiple occurrences without replace_all returns error."""
        sandbox.write("multi.txt", "foo foo foo")

        result = sandbox.edit("multi.txt", "foo", "bar", replace_all=False)
        assert result.error is not None
        assert "multiple" in result.error.lower()

    def test_edit_multiple_occurrences_with_replace_all(self, sandbox: BaseSandbox):
        """Test replace_all flag replaces all occurrences."""
        sandbox.write("multi.txt", "foo foo foo")

        result = sandbox.edit("multi.txt", "foo", "bar", replace_all=True)
        assert result.error is None
        assert result.occurrences == 3

        content = sandbox.read("multi.txt")
        assert content.count("bar") == 3
        assert "foo" not in content or content.count("foo") == 0

    def test_edit_nonexistent_file(self, sandbox: BaseSandbox):
        """Test editing a nonexistent file returns error."""
        result = sandbox.edit("nonexistent.txt", "old", "new")
        assert result.error is not None

    def test_edit_string_not_found(self, sandbox: BaseSandbox):
        """Test editing with string that doesn't exist returns error."""
        sandbox.write("test.txt", "hello world")

        result = sandbox.edit("test.txt", "nonexistent", "new")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_read_nonexistent_file(self, sandbox: BaseSandbox):
        """Test reading a nonexistent file returns error."""
        content = sandbox.read("nonexistent.txt")
        assert "Error" in content or "not found" in content.lower()

    def test_read_with_offset_and_limit(self, sandbox: BaseSandbox):
        """Test reading file with offset and limit."""
        lines = "\n".join([f"Line {i}" for i in range(1, 101)])
        sandbox.write("long.txt", lines)

        # Read first 10 lines
        content = sandbox.read("long.txt", offset=0, limit=10)
        assert "Line 1" in content
        assert "Line 10" in content
        assert "Line 11" not in content

        # Read lines 50-60
        content = sandbox.read("long.txt", offset=49, limit=10)
        assert "Line 50" in content
        assert "Line 59" in content
        assert "Line 49" not in content
        assert "Line 60" not in content

    def test_file_with_whitespace_in_name(self, sandbox: BaseSandbox):
        """Test operations on files with spaces in names."""
        filename = "file with spaces.txt"
        content = "content here"

        # Write
        result = sandbox.write(filename, content)
        assert result.error is None

        # Read
        output = sandbox.read(filename)
        assert "content here" in output

        # Edit
        edit_result = sandbox.edit(filename, "content", "new content")
        assert edit_result.error is None

        # Verify
        output = sandbox.read(filename)
        assert "new content" in output

    def test_file_with_special_chars_in_name(self, sandbox: BaseSandbox):
        """Test operations on files with special characters in names."""
        # Test with safe special characters
        filename = "test-file_v2.0.txt"
        content = "test content"

        result = sandbox.write(filename, content)
        assert result.error is None

        output = sandbox.read(filename)
        assert "test content" in output

    def test_write_creates_parent_directories(self, sandbox: BaseSandbox):
        """Test that write creates parent directories."""
        result = sandbox.write("subdir/nested/file.txt", "content")
        assert result.error is None

        content = sandbox.read("subdir/nested/file.txt")
        assert "content" in content

    def test_grep_basic_search(self, sandbox: BaseSandbox):
        """Test basic grep functionality."""
        sandbox.write("file1.txt", "hello world\nfoo bar")
        sandbox.write("file2.txt", "hello universe\nbaz qux")

        matches = sandbox.grep_raw("hello")
        assert isinstance(matches, list)
        assert len(matches) >= 2
        assert any("file1.txt" in m["path"] for m in matches)
        assert any("file2.txt" in m["path"] for m in matches)

    def test_grep_with_glob_filter(self, sandbox: BaseSandbox):
        """Test grep with glob pattern filter."""
        sandbox.write("test.py", "def hello(): pass")
        sandbox.write("test.txt", "hello world")
        sandbox.write("test.md", "# hello")

        # Search only .py files
        matches = sandbox.grep_raw("hello", glob="*.py")
        assert isinstance(matches, list)
        assert len(matches) == 1
        assert "test.py" in matches[0]["path"]

    def test_grep_returns_line_numbers(self, sandbox: BaseSandbox):
        """Test that grep returns line numbers."""
        content = "line 1\nline 2 target\nline 3\nline 4 target"
        sandbox.write("test.txt", content)

        matches = sandbox.grep_raw("target")
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert matches[0]["line"] == 2
        assert matches[1]["line"] == 4

    def test_grep_no_matches(self, sandbox: BaseSandbox):
        """Test grep with no matches returns empty list."""
        sandbox.write("test.txt", "hello world")

        matches = sandbox.grep_raw("nonexistent")
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_glob_basic_pattern(self, sandbox: BaseSandbox):
        """Test basic glob pattern matching."""
        sandbox.write("file1.txt", "a")
        sandbox.write("file2.py", "b")
        sandbox.write("file3.txt", "c")

        matches = sandbox.glob_info("*.txt")
        assert len(matches) >= 2
        paths = [m["path"] for m in matches]
        assert any("file1.txt" in p for p in paths)
        assert any("file3.txt" in p for p in paths)
        assert not any("file2.py" in p for p in paths)

    def test_glob_recursive_pattern(self, sandbox: BaseSandbox):
        """Test recursive glob pattern."""
        sandbox.write("root.txt", "a")
        sandbox.write("subdir/nested.txt", "b")
        sandbox.write("subdir/deep/file.txt", "c")

        matches = sandbox.glob_info("**/*.txt")
        assert len(matches) >= 3

    def test_glob_no_matches(self, sandbox: BaseSandbox):
        """Test glob with no matches returns empty list."""
        sandbox.write("test.py", "code")

        matches = sandbox.glob_info("*.txt")
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_ls_info_basic(self, sandbox: BaseSandbox):
        """Test basic ls_info functionality."""
        sandbox.write("file1.txt", "a")
        sandbox.write("file2.txt", "b")
        sandbox.write("subdir/nested.txt", "c")

        # List current directory
        infos = sandbox.ls_info(".")
        assert len(infos) >= 2  # At least file1.txt, file2.txt, and subdir

        paths = [info["path"] for info in infos]
        # Check that files are listed
        assert any("file1.txt" in p for p in paths)
        assert any("file2.txt" in p for p in paths)

    def test_multiline_content(self, sandbox: BaseSandbox):
        """Test working with multiline file content."""
        content = """Line 1
Line 2
Line 3
Line 4"""
        sandbox.write("multiline.txt", content)

        # Read and verify
        output = sandbox.read("multiline.txt")
        assert "Line 1" in output
        assert "Line 4" in output

        # Edit multiline
        result = sandbox.edit("multiline.txt", "Line 2", "Modified Line 2")
        assert result.error is None

        output = sandbox.read("multiline.txt")
        assert "Modified Line 2" in output

    def test_empty_file(self, sandbox: BaseSandbox):
        """Test operations on empty files."""
        sandbox.write("empty.txt", "")

        content = sandbox.read("empty.txt")
        # Should handle empty file gracefully
        assert isinstance(content, str)

    def test_unicode_content(self, sandbox: BaseSandbox):
        """Test handling of unicode characters."""
        content = "Hello 世界 🌍 Привет"
        sandbox.write("unicode.txt", content)

        output = sandbox.read("unicode.txt")
        assert "世界" in output
        assert "🌍" in output
        assert "Привет" in output

    def test_newlines_in_edit(self, sandbox: BaseSandbox):
        """Test edit with newlines in replacement strings."""
        sandbox.write("newlines.txt", "foo")

        result = sandbox.edit("newlines.txt", "foo", "bar\nbaz")
        assert result.error is None

        output = sandbox.read("newlines.txt")
        assert "bar" in output
        assert "baz" in output

    def test_execute_basic_command(self, sandbox: BaseSandbox):
        """Test basic command execution."""
        result = sandbox.execute("echo 'hello'")
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_execute_command_with_exit_code(self, sandbox: BaseSandbox):
        """Test that non-zero exit codes are captured."""
        result = sandbox.execute("exit 42")
        assert result.exit_code == 42

    def test_execute_creates_files(self, sandbox: BaseSandbox):
        """Test that executed commands can create files."""
        sandbox.execute("echo 'test content' > executed.txt")

        content = sandbox.read("executed.txt")
        assert "test content" in content
