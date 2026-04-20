from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import DeleteResult, EditResult, ReadResult, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware


def write_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def test_filesystem_backend_normal_mode(tmp_path: Path):
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.py"
    write_file(f1, "hello fs")
    write_file(f2, "print('x')\nhello")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    # ls_info absolute path - should only list files in root, not subdirectories
    infos = be.ls(str(root)).entries
    assert infos is not None
    paths = {i["path"] for i in infos}
    assert str(f1) in paths  # File in root should be listed
    assert str(f2) not in paths  # File in subdirectory should NOT be listed
    assert (str(root / "dir") + "/") in paths  # Directory should be listed

    # read, edit, write
    read_result = be.read(str(f1))
    assert isinstance(read_result, ReadResult) and read_result.file_data is not None
    assert "hello fs" in read_result.file_data["content"]
    msg = be.edit(str(f1), "fs", "filesystem", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1
    msg2 = be.write(str(root / "new.txt"), "new content")
    assert isinstance(msg2, WriteResult) and msg2.error is None and msg2.path.endswith("new.txt")

    # grep
    matches = be.grep("hello", path=str(root)).matches
    assert matches is not None and any(m["path"].endswith("a.txt") for m in matches)

    # glob
    g = be.glob("*.py", path=str(root)).matches
    assert any(i["path"] == str(f2) for i in g)


def test_filesystem_backend_virtual_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.md"
    write_file(f1, "hello virtual")
    write_file(f2, "content")

    monkeypatch.setattr(FilesystemBackend, "_ripgrep_search", lambda *_args, **_kwargs: None)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # ls_info from virtual root - should only list files in root, not subdirectories
    infos = be.ls("/").entries
    assert infos is not None
    paths = {i["path"] for i in infos}
    assert "/a.txt" in paths  # File in root should be listed
    assert "/dir/b.md" not in paths  # File in subdirectory should NOT be listed
    assert "/dir/" in paths  # Directory should be listed

    # read and edit via virtual path
    read_result = be.read("/a.txt")
    assert isinstance(read_result, ReadResult) and read_result.file_data is not None
    assert "hello virtual" in read_result.file_data["content"]
    msg = be.edit("/a.txt", "virtual", "virt", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1

    # write new file via virtual path
    msg2 = be.write("/new.txt", "x")
    assert isinstance(msg2, WriteResult) and msg2.error is None
    assert (root / "new.txt").exists()

    # grep limited to path
    matches = be.grep("virt", path="/").matches
    assert matches is not None and any(m["path"] == "/a.txt" for m in matches)

    # glob
    g = be.glob("**/*.md", path="/").matches
    assert any(i["path"] == "/dir/b.md" for i in g)

    # literal search should work with special regex chars like "[" and "("
    result_bracket = be.grep("[", path="/")
    assert result_bracket.matches is not None  # Should not error, returns empty list or matches

    # path traversal blocked
    with pytest.raises(ValueError, match="traversal"):
        be.read("/../a.txt")


def test_filesystem_backend_ls_nested_directories(tmp_path: Path):
    root = tmp_path

    files = {
        root / "config.json": "config",
        root / "src" / "main.py": "code",
        root / "src" / "utils" / "helper.py": "utils code",
        root / "src" / "utils" / "common.py": "common utils",
        root / "docs" / "readme.md": "documentation",
        root / "docs" / "api" / "reference.md": "api docs",
    }

    for path, content in files.items():
        write_file(path, content)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    root_listing = be.ls("/").entries
    assert root_listing is not None
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths

    src_listing = be.ls("/src/").entries
    assert src_listing is not None
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = be.ls("/src/utils/").entries
    assert utils_listing is not None
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = be.ls("/nonexistent/")
    assert empty_listing.entries == []


def test_filesystem_backend_ls_normal_mode_nested(tmp_path: Path):
    """Test ls_info with nested directories in normal (non-virtual) mode."""
    root = tmp_path

    files = {
        root / "file1.txt": "content1",
        root / "subdir" / "file2.txt": "content2",
        root / "subdir" / "nested" / "file3.txt": "content3",
    }

    for path, content in files.items():
        write_file(path, content)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    root_listing = be.ls(str(root)).entries
    assert root_listing is not None
    root_paths = [fi["path"] for fi in root_listing]

    assert str(root / "file1.txt") in root_paths
    assert str(root / "subdir") + "/" in root_paths
    assert str(root / "subdir" / "file2.txt") not in root_paths

    subdir_listing = be.ls(str(root / "subdir")).entries
    assert subdir_listing is not None
    subdir_paths = [fi["path"] for fi in subdir_listing]
    assert str(root / "subdir" / "file2.txt") in subdir_paths
    assert str(root / "subdir" / "nested") + "/" in subdir_paths
    assert str(root / "subdir" / "nested" / "file3.txt") not in subdir_paths


def test_filesystem_backend_ls_trailing_slash(tmp_path: Path):
    """Test ls_info edge cases for filesystem backend."""
    root = tmp_path

    files = {
        root / "file.txt": "content",
        root / "dir" / "nested.txt": "nested",
    }

    for path, content in files.items():
        write_file(path, content)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    listing_with_slash = be.ls("/").entries
    assert listing_with_slash is not None
    assert len(listing_with_slash) > 0

    listing = be.ls("/").entries
    assert listing is not None
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    listing1 = be.ls("/dir/").entries
    listing2 = be.ls("/dir").entries
    assert listing1 is not None
    assert listing2 is not None
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]

    empty = be.ls("/nonexistent/")
    assert empty.entries == []


def test_filesystem_backend_read_non_utf8_file(tmp_path: Path):
    """FilesystemBackend.read should return an error result, not raise, for non-UTF-8 text files."""
    root = tmp_path
    # Write a file with GBK-encoded bytes that are invalid UTF-8 (e.g. 0x87)
    gbk_file = root / "chinese.txt"
    gbk_file.write_bytes("中文内容".encode("gbk"))

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)
    result = be.read(str(gbk_file))

    assert isinstance(result, ReadResult)
    assert result.error is not None
    assert "chinese.txt" in result.error


def test_filesystem_backend_intercept_large_tool_result(tmp_path: Path):
    """Test that FilesystemBackend properly handles large tool result interception."""
    root = tmp_path
    rt = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="test_fs",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=str(root), virtual_mode=True), tool_token_limit_before_evict=1000)

    large_content = "f" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_fs_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_fs_123" in result.content
    saved_file = root / "large_tool_results" / "test_fs_123"
    assert saved_file.exists()
    assert saved_file.read_text() == large_content


def test_filesystem_upload_single_file(tmp_path: Path):
    """Test uploading a single binary file."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    test_path = "/test_upload.bin"
    test_content = b"Hello, Binary World!"

    responses = be.upload_files([(test_path, test_content)])

    assert len(responses) == 1
    assert responses[0].path == test_path
    assert responses[0].error is None

    # Verify file exists and content matches
    uploaded_file = root / "test_upload.bin"
    assert uploaded_file.exists()
    assert uploaded_file.read_bytes() == test_content


def test_filesystem_upload_multiple_files(tmp_path: Path):
    """Test uploading multiple files in one call."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    files = [
        ("/file1.bin", b"Content 1"),
        ("/file2.bin", b"Content 2"),
        ("/subdir/file3.bin", b"Content 3"),
    ]

    responses = be.upload_files(files)

    assert len(responses) == 3
    for i, (path, _content) in enumerate(files):
        assert responses[i].path == path
        assert responses[i].error is None

    # Verify all files created
    assert (root / "file1.bin").read_bytes() == b"Content 1"
    assert (root / "file2.bin").read_bytes() == b"Content 2"
    assert (root / "subdir" / "file3.bin").read_bytes() == b"Content 3"


def test_filesystem_download_single_file(tmp_path: Path):
    """Test downloading a single file."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create a file manually
    test_file = root / "test_download.bin"
    test_content = b"Download me!"
    test_file.write_bytes(test_content)

    responses = be.download_files(["/test_download.bin"])

    assert len(responses) == 1
    assert responses[0].path == "/test_download.bin"
    assert responses[0].content == test_content
    assert responses[0].error is None


def test_filesystem_download_multiple_files(tmp_path: Path):
    """Test downloading multiple files in one call."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create several files
    files = {
        root / "file1.txt": b"File 1",
        root / "file2.txt": b"File 2",
        root / "subdir" / "file3.txt": b"File 3",
    }

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    paths = ["/file1.txt", "/file2.txt", "/subdir/file3.txt"]
    responses = be.download_files(paths)

    assert len(responses) == 3
    assert responses[0].path == "/file1.txt"
    assert responses[0].content == b"File 1"
    assert responses[0].error is None

    assert responses[1].path == "/file2.txt"
    assert responses[1].content == b"File 2"
    assert responses[1].error is None

    assert responses[2].path == "/subdir/file3.txt"
    assert responses[2].content == b"File 3"
    assert responses[2].error is None


def test_filesystem_upload_download_roundtrip(tmp_path: Path):
    """Test upload followed by download for data integrity."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test with binary content including special bytes
    test_path = "/roundtrip.bin"
    test_content = bytes(range(256))  # All possible byte values

    # Upload
    upload_responses = be.upload_files([(test_path, test_content)])
    assert upload_responses[0].error is None

    # Download
    download_responses = be.download_files([test_path])
    assert download_responses[0].error is None
    assert download_responses[0].content == test_content


def test_filesystem_download_errors(tmp_path: Path):
    """Test download error handling."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test file_not_found
    responses = be.download_files(["/nonexistent.txt"])
    assert len(responses) == 1
    assert responses[0].path == "/nonexistent.txt"
    assert responses[0].content is None
    assert responses[0].error == "file_not_found"

    # Test is_directory
    (root / "testdir").mkdir()
    responses = be.download_files(["/testdir"])
    assert responses[0].error == "is_directory"
    assert responses[0].content is None

    # Test invalid_path (path traversal)
    responses = be.download_files(["/../etc/passwd"])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None


def test_filesystem_upload_errors(tmp_path: Path):
    """Test upload error handling."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test invalid_path (path traversal)
    responses = be.upload_files([("/../bad/path.txt", b"content")])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"


def test_filesystem_partial_success_upload(tmp_path: Path):
    """Test partial success in batch upload."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    files = [
        ("/valid1.txt", b"Valid content 1"),
        ("/../invalid.txt", b"Invalid path"),  # Path traversal
        ("/valid2.txt", b"Valid content 2"),
    ]

    responses = be.upload_files(files)

    assert len(responses) == 3
    # First file should succeed
    assert responses[0].error is None
    assert (root / "valid1.txt").exists()

    # Second file should fail
    assert responses[1].error == "invalid_path"

    # Third file should still succeed (partial success)
    assert responses[2].error is None
    assert (root / "valid2.txt").exists()


def test_filesystem_partial_success_download(tmp_path: Path):
    """Test partial success in batch download."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create one valid file
    valid_file = root / "exists.txt"
    valid_content = b"I exist!"
    valid_file.write_bytes(valid_content)

    paths = ["/exists.txt", "/doesnotexist.txt", "/../invalid"]
    responses = be.download_files(paths)

    assert len(responses) == 3

    # First should succeed
    assert responses[0].error is None
    assert responses[0].content == valid_content

    # Second should fail with file_not_found
    assert responses[1].error == "file_not_found"
    assert responses[1].content is None

    # Third should fail with invalid_path
    assert responses[2].error == "invalid_path"
    assert responses[2].content is None


def test_filesystem_upload_to_existing_directory_path(tmp_path: Path):
    """Test uploading to a path where the target is an existing directory.

    This simulates trying to overwrite a directory with a file, which should
    produce an error. For example, if /mydir/ exists as a directory, trying
    to upload a file to /mydir should fail.
    """
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create a directory
    (root / "existing_dir").mkdir()

    # Try to upload a file with the same name as the directory
    # Note: on Unix systems, this will likely succeed but create a different inode
    # The behavior depends on the OS and filesystem. Let's just verify we get a response.
    responses = be.upload_files([("/existing_dir", b"file content")])

    assert len(responses) == 1
    assert responses[0].path == "/existing_dir"
    # Depending on OS behavior, this might succeed or fail
    # We're just documenting the behavior exists


def test_filesystem_upload_parent_is_file(tmp_path: Path):
    """Test uploading to a path where a parent component is a file, not a directory.

    For example, if /somefile.txt exists as a file, trying to upload to
    /somefile.txt/child.txt should fail because somefile.txt is not a directory.
    """
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create a file
    parent_file = root / "parent.txt"
    parent_file.write_text("I am a file, not a directory")

    # Try to upload a file as if parent.txt were a directory
    responses = be.upload_files([("/parent.txt/child.txt", b"child content")])

    assert len(responses) == 1
    assert responses[0].path == "/parent.txt/child.txt"
    # This should produce some kind of error since parent.txt is a file
    assert responses[0].error is not None


def test_filesystem_download_directory_as_file(tmp_path: Path):
    """Test that downloading a directory returns is_directory error.

    This is already tested in test_filesystem_download_errors but we add
    an explicit test case to make it clear this is a supported error scenario.
    """
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create a directory
    (root / "mydir").mkdir()

    # Try to download the directory as if it were a file
    responses = be.download_files(["/mydir"])

    assert len(responses) == 1
    assert responses[0].path == "/mydir"
    assert responses[0].content is None
    assert responses[0].error == "is_directory"


@pytest.mark.parametrize(
    ("pattern", "expected_file"),
    [
        ("def __init__(", "test1.py"),  # Parentheses (not regex grouping)
        ("str | int", "test2.py"),  # Pipe (not regex OR)
        ("[a-z]", "test3.py"),  # Brackets (not character class)
        ("(.*)", "test3.py"),  # Multiple special chars
        ("$19.99", "test4.txt"),  # Dot and $ (not "any character")
        ("user@example", "test4.txt"),  # @ character (literal)
    ],
)
def test_grep_literal_search_with_special_chars(tmp_path: Path, pattern: str, expected_file: str) -> None:
    """Test that grep treats patterns as literal strings, not regex.

    Tests with both ripgrep (if available) and Python fallback.
    """
    root = tmp_path

    # Create test files with special regex characters
    (root / "test1.py").write_text("def __init__(self, arg):\n    pass")
    (root / "test2.py").write_text("@overload\ndef func(x: str | int):\n    return x")
    (root / "test3.py").write_text("pattern = r'[a-z]+'\nregex_chars = '(.*)'")
    (root / "test4.txt").write_text("Price: $19.99\nEmail: user@example.com")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test literal search with the pattern (uses ripgrep if available, otherwise Python fallback)
    matches = be.grep(pattern, path="/").matches
    assert matches is not None
    assert any(expected_file in m["path"] for m in matches), f"Pattern '{pattern}' not found in {expected_file}"


class TestToVirtualPath:
    """Tests for FilesystemBackend._to_virtual_path."""

    def test_returns_forward_slash_relative_path(self, tmp_path: Path):
        """Nested path is returned as forward-slash virtual path."""
        (tmp_path / "src").mkdir()
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be._to_virtual_path(tmp_path / "src" / "file.py")
        assert result == "/src/file.py"

    def test_cwd_itself_returns_slash_dot(self, tmp_path: Path):
        """Cwd path returns `/.` since `Path('.').as_posix()` is `'.'`."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be._to_virtual_path(tmp_path)
        assert result == "/."

    def test_outside_cwd_raises_value_error(self, tmp_path: Path):
        """Path outside cwd raises ValueError."""
        sub = tmp_path / "sub"
        sub.mkdir()
        be = FilesystemBackend(root_dir=str(sub), virtual_mode=True)
        with pytest.raises(ValueError, match="is not in the subpath of"):
            be._to_virtual_path(tmp_path / "outside.txt")


class TestWindowsPathHandling:
    """Tests that virtual-mode paths always use forward slashes."""

    @pytest.fixture
    def backend(self, tmp_path: Path):
        """Create a backend with nested directories."""
        (tmp_path / "src" / "utils").mkdir(parents=True)
        (tmp_path / "src" / "main.py").write_text("print('main')")
        (tmp_path / "src" / "utils" / "helper.py").write_text("def help(): pass")
        return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    def test_ls_paths(self, backend):
        """Ls should return forward-slash paths."""
        infos = backend.ls("/src").entries
        assert infos is not None
        for info in infos:
            assert "\\" not in info["path"], f"Backslash in ls path: {info['path']}"

    def test_glob_paths(self, backend):
        """Glob should return forward-slash paths."""
        result = backend.glob("**/*.py", path="/")
        assert result.matches is not None
        for info in result.matches:
            assert "\\" not in info["path"], f"Backslash in glob path: {info['path']}"

    def test_grep_paths(self, backend):
        """Grep should return forward-slash paths."""
        matches = backend.grep("def", path="/").matches
        assert matches is not None
        for m in matches:
            assert "\\" not in m["path"], f"Backslash in grep path: {m['path']}"

    def test_deeply_nested_path(self, tmp_path: Path):
        """Deeply nested paths should still use forward slashes."""
        deep = tmp_path / "a" / "b" / "c" / "d"
        deep.mkdir(parents=True)
        (deep / "file.txt").write_text("content")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        infos = be.ls("/a/b/c/d").entries
        assert infos is not None
        for info in infos:
            assert "\\" not in info["path"], f"Backslash in deep path: {info['path']}"


class TestEditCrlfNormalization:
    """Tests for CRLF normalization in edit(). See #2247."""

    def test_edit_normalizes_crlf_in_old_string(self, tmp_path: Path):
        """edit() should succeed when old_string contains CRLF but file has LF.

        Addresses a bug where download_files() returns raw bytes (binary
        mode) that may contain CRLF, the caller decodes them and passes
        to edit(), but edit() reads the file in text mode (LF-normalized).
        """
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        content = "line1\nline2\nline3\n"
        be.write("/test.txt", content)

        result = be.edit("/test.txt", "line1\r\nline2\r\n", "replaced\n")
        assert result.error is None
        assert result.occurrences == 1
        assert (tmp_path / "test.txt").read_text() == "replaced\nline3\n"

    def test_edit_normalizes_crlf_in_new_string(self, tmp_path: Path):
        """edit() should normalize CRLF in new_string too."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        be.write("/test.txt", "hello world\n")

        result = be.edit("/test.txt", "hello", "goodbye\r\n")
        assert result.error is None
        raw = (tmp_path / "test.txt").read_bytes()
        assert b"\r" not in raw

    def test_edit_crlf_with_replace_all(self, tmp_path: Path):
        """edit() should normalize CRLF when replace_all=True."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        be.write("/test.txt", "foo\nbar\nfoo\n")

        result = be.edit("/test.txt", "foo\r\n", "baz\n", replace_all=True)
        assert result.error is None
        assert result.occurrences == 2
        assert (tmp_path / "test.txt").read_text() == "baz\nbar\nbaz\n"

    def test_edit_with_download_roundtrip_crlf(self, tmp_path: Path):
        """Simulate a download-then-edit flow where downloaded content has CRLF.

        1. write() creates a file
        2. Simulate download_files() returning CRLF bytes (binary-mode read)
        3. edit() with the CRLF-decoded content as old_string should succeed
        """
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        original = "## Summary\n\nHuman: hello\nAI: hi\n\n"
        be.write("/history.md", original)

        crlf_content = original.replace("\n", "\r\n")

        appended = "## Summary 2\n\nHuman: next\nAI: ok\n\n"
        combined = crlf_content + appended

        result = be.edit("/history.md", crlf_content, combined)
        assert result.error is None
        assert result.occurrences == 1

        final = (tmp_path / "history.md").read_text()
        assert "## Summary 2" in final
        assert "Human: next" in final


def test_filesystem_backend_delete_normal_mode(tmp_path: Path):
    root = tmp_path
    f1 = root / "to_delete.txt"
    write_file(f1, "delete me")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    result = be.delete(str(f1))
    assert isinstance(result, DeleteResult)
    assert result.error is None
    assert result.path == str(f1)
    assert not f1.exists()


def test_filesystem_backend_delete_virtual_mode(tmp_path: Path):
    root = tmp_path
    f1 = root / "to_delete.txt"
    write_file(f1, "delete me")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    result = be.delete("/to_delete.txt")
    assert isinstance(result, DeleteResult)
    assert result.error is None
    assert result.path == "/to_delete.txt"
    assert not f1.exists()


def test_filesystem_backend_delete_nonexistent(tmp_path: Path):
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.delete(str(tmp_path / "nonexistent.txt"))
    assert isinstance(result, DeleteResult)
    assert result.error is not None
    assert "not found" in result.error


def test_filesystem_backend_delete_directory(tmp_path: Path):
    d = tmp_path / "mydir"
    d.mkdir()

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.delete(str(d))
    assert isinstance(result, DeleteResult)
    assert result.error is not None
    assert "directory" in result.error.lower()
