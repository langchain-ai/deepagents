import base64
import io
import json
import logging
import shutil
import threading
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents._api.deprecation import LangChainDeprecationWarning
from deepagents.backends import filesystem as fs_module
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import DeleteResult, EditResult, ReadResult, WriteResult
from deepagents.middleware.filesystem import GLOB_TIMEOUT, FilesystemMiddleware


def write_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def make_symlink_loop(path: Path) -> None:
    try:
        path.symlink_to(path)
    except (NotImplementedError, OSError):
        pytest.skip("platform does not support symlinks")


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


def test_filesystem_backend_glob_default_matches_backend_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    in_root = root / "dir" / "inside.py"
    outside_root = tmp_path / "outside.py"
    write_file(in_root, "print('inside')")
    write_file(outside_root, "print('outside')")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    omitted = be.glob("**/*.py").matches or []
    explicit_root = be.glob("**/*.py", path="/").matches or []

    omitted_paths = {info["path"] for info in omitted}
    explicit_root_paths = {info["path"] for info in explicit_root}
    assert omitted_paths == explicit_root_paths
    assert str(in_root) in omitted_paths
    assert str(outside_root) not in omitted_paths


def test_filesystem_backend_glob_matches_hidden_paths(tmp_path: Path) -> None:
    root = tmp_path
    write_file(root / ".env", "TOKEN=value")
    write_file(root / ".hidden.py", "print('hidden')")
    write_file(root / ".github" / "workflows" / "ci.yml", "name: ci")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    root_matches = {info["path"] for info in be.glob("*", path="/").matches or []}
    py_matches = {info["path"] for info in be.glob("*.py", path="/").matches or []}
    yml_matches = {info["path"] for info in be.glob("**/*.yml", path="/").matches or []}

    assert "/.env" in root_matches
    assert "/.hidden.py" in py_matches
    assert "/.github/workflows/ci.yml" in yml_matches


def test_filesystem_backend_virtual_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.md"
    write_file(f1, "hello virtual")
    write_file(f2, "content")

    monkeypatch.setattr(FilesystemBackend, "_ripgrep_search", lambda *_args, **_kwargs: (None, False))

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
    assert empty_listing.entries is None
    assert empty_listing.error == "Path '/nonexistent/': path_not_found"


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
    assert empty.entries is None
    assert empty.error == "Path '/nonexistent/': path_not_found"


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


def test_filesystem_backend_reads_mkv_as_binary(tmp_path: Path) -> None:
    """Local `.mkv` reads must be routed as binary before UTF-8 decoding."""
    target = tmp_path / "clip.mkv"
    raw = b"\x80\x81mkv bytes"
    target.write_bytes(raw)

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    result = be.read(str(target))

    assert isinstance(result, ReadResult)
    assert result.error is None
    assert result.file_data == {
        "content": base64.standard_b64encode(raw).decode("ascii"),
        "encoding": "base64",
    }


def test_filesystem_backend_rejects_oversized_video_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local video reads fail before loading oversized files into memory."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"abcd")
    monkeypatch.setattr(fs_module, "MAX_VIDEO_INPUT_BYTES", 3)

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    result = be.read(str(target))

    assert isinstance(result, ReadResult)
    assert result.error == "Video file exceeds maximum input size of 3 bytes"


def test_filesystem_backend_rejects_oversized_mkv_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local `.mkv` reads use the video size guard before loading bytes."""
    target = tmp_path / "clip.mkv"
    target.write_bytes(b"abcd")
    monkeypatch.setattr(fs_module, "MAX_VIDEO_INPUT_BYTES", 3)

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    result = be.read(str(target))

    assert isinstance(result, ReadResult)
    assert result.error == "Video file exceeds maximum input size of 3 bytes"


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


def test_grep_ripgrep_glob_with_directory_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for #2732.

    ripgrep `--glob` patterns with a directory component (e.g. `docs/*.md`)
    must still match when the process cwd differs from the search root.
    """
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")

    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("hello world\n")
    (root / "notes.md").write_text("hello world\n")

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    matches = be.grep("hello", path=str(root), glob="docs/*.md").matches
    assert matches is not None
    matched_paths = [m["path"] for m in matches]
    assert any(p.endswith("docs/guide.md") for p in matched_paths), f"expected docs/guide.md in {matched_paths}"
    assert not any(p.endswith("notes.md") for p in matched_paths), f"glob should have excluded notes.md but matched {matched_paths}"


def test_grep_ripgrep_glob_virtual_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for #2732, virtual mode variant.

    Exercises the relative-path re-anchoring through `_to_virtual_path` so a
    regression in path handling can't silently drop results.
    """
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")

    root = tmp_path / "project"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text("hello world\n")
    (root / "notes.md").write_text("hello world\n")

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    matches = be.grep("hello", path="/", glob="docs/*.md").matches
    assert matches is not None
    matched_paths = [m["path"] for m in matches]
    assert any(p == "/docs/guide.md" for p in matched_paths), f"expected /docs/guide.md in {matched_paths}"
    assert not any("notes" in p for p in matched_paths), f"glob should have excluded notes.md but matched {matched_paths}"


def test_grep_on_single_file_path(tmp_path: Path) -> None:
    """Regression test: grep with `path` pointing at a single file must not crash.

    Before #2732's fix, ripgrep was given the file path directly. Naively
    threading `cwd=base_full` would raise NotADirectoryError for file paths.
    """
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")

    target = tmp_path / "single.txt"
    target.write_text("hello single\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("hello", path=str(target))
    assert result.error is None, f"unexpected error: {result.error}"
    assert result.matches is not None
    assert any(m["path"].endswith("single.txt") for m in result.matches)


def test_grep_preserves_symlink_path_in_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for #2732: result paths must keep symlink form.

    Pre-fix, ripgrep emitted absolute paths exactly as it crawled them — no
    `.resolve()` was applied — so users saw the symlinked path they searched
    under. The fix must preserve that behavior.
    """
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")

    real = tmp_path / "real"
    real.mkdir()
    (real / "target.txt").write_text("hello symlink\n")

    root = tmp_path / "project"
    root.mkdir()
    link = root / "via_link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("platform does not support directory symlinks")

    monkeypatch.chdir(tmp_path)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)
    result = be.grep("hello", path=str(link))
    assert result.error is None
    assert result.matches is not None
    matched = [m["path"] for m in result.matches]
    assert matched, "expected at least one match"
    for p in matched:
        assert str(link) in p, f"symlink form lost; got {p}"
        assert str(real) not in p, f"path was resolved through the symlink; got {p}"


def test_grep_containment_check_blocks_escaping_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: ripgrep results via symlinks that escape the root must be filtered.

    A directory symlink inside `root` that points outside `root` gives ripgrep
    access to files beyond the intended search boundary. The containment check
    must drop those results so they never surface to callers.
    """
    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("hello secret\n")

    root = tmp_path / "root"
    root.mkdir()
    escape = root / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("platform does not support directory symlinks")

    monkeypatch.chdir(tmp_path)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)
    result = be.grep("hello", path=str(root))
    matched = [m["path"] for m in (result.matches or [])]
    assert not any("secret" in p for p in matched), f"containment check failed — escaping symlink leaked result: {matched}"


_RG_MISSING_PREFIX = "ripgrep ('rg') not found on PATH"


@pytest.fixture
def _isolate_rg_cache() -> Iterator[None]:
    """Clear the process-wide `rg` resolver cache around each test that touches it."""
    fs_module._resolve_ripgrep_path.cache_clear()
    yield
    fs_module._resolve_ripgrep_path.cache_clear()


class _BlockingStdout:
    """Text stream that yields queued lines then blocks until the proc is killed.

    Mirrors how ripgrep's stdout behaves under `subprocess.Popen(text=True)`:
    iterating yields one JSON frame per line, and once ripgrep is terminated the
    pipe closes so iteration ends. The `killed` event stands in for that close,
    letting the timeout watchdog end the loop without real timing races.
    """

    def __init__(self, lines: list[str], killed: threading.Event) -> None:
        self._lines = iter(lines)
        self._killed = killed

    def __iter__(self) -> "_BlockingStdout":
        return self

    def __next__(self) -> str:
        if self._killed.is_set():
            raise StopIteration
        try:
            return next(self._lines)
        except StopIteration:
            # No more queued frames: emulate a stream that stays open until the
            # process is killed (e.g. by the timeout watchdog).
            self._killed.wait()
            raise

    def close(self) -> None:
        pass


class _FakePopen:
    """Minimal `subprocess.Popen` stand-in for the streaming ripgrep path."""

    def __init__(self, stdout_lines: list[str] | None = None, stderr: str = "", returncode: int = 0, *, block: bool = False) -> None:
        self._killed = threading.Event()
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        lines = stdout_lines or []
        if block:
            self.stdout: object = _BlockingStdout(lines, self._killed)
        else:
            self.stdout = io.StringIO("".join(lines))
            self._killed.set()  # non-blocking: nothing to wait on
        self.stderr = io.StringIO(stderr)

    def terminate(self) -> None:
        self.terminated = True
        self._killed.set()

    def kill(self) -> None:
        self.killed = True
        # A real kill yields a negative (signal) return code.
        self.returncode = -9
        self._killed.set()

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_resolve_ripgrep_logs_once_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Missing `rg` should log an `INFO` message exactly once per process.

    Operators investigating slow searches need at least one signal that the
    Python fallback is in play.
    """
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: None)

    (tmp_path / "a.txt").write_text("hello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    with caplog.at_level(logging.INFO, logger=fs_module.logger.name):
        be.grep("hello", path=str(tmp_path))
        be.grep("hello", path=str(tmp_path))  # second call must not re-log

    matching = [r for r in caplog.records if r.levelno == logging.INFO and r.getMessage().startswith(_RG_MISSING_PREFIX)]
    assert len(matching) == 1, f"expected exactly one INFO log, got {len(matching)}: {[r.getMessage() for r in matching]}"


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_resolve_ripgrep_uses_resolved_path_in_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached absolute path is exec'd verbatim, alongside the expected flags."""
    fake_rg = "/opt/fake/bin/rg"
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: fake_rg)

    captured: dict[str, list[str]] = {}

    def fake_popen(cmd: list[str], **_kwargs: object) -> _FakePopen:
        captured["cmd"] = cmd
        return _FakePopen()

    monkeypatch.setattr(fs_module.subprocess, "Popen", fake_popen)

    (tmp_path / "a.txt").write_text("hello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    be.grep("hello", path=str(tmp_path))

    cmd = captured["cmd"]
    assert cmd[0] == fake_rg
    assert "--json" in cmd
    assert "-F" in cmd
    assert "hello" in cmd  # pattern made it through


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_timeout_logs_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """When the streaming watchdog kills ripgrep with no output, we warn and fall back."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    # Tiny budget so the watchdog timer fires almost immediately; the fake
    # stdout blocks until that kill closes the stream (no timing race).
    monkeypatch.setattr(fs_module, "DEFAULT_GREP_TIMEOUT", 0.05)

    def timeout_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        return _FakePopen(stdout_lines=[], block=True)

    monkeypatch.setattr(fs_module.subprocess, "Popen", timeout_popen)

    (tmp_path / "a.txt").write_text("hello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    with caplog.at_level(logging.WARNING, logger=fs_module.logger.name):
        result = be.grep("hello", path=str(tmp_path))

    assert any("timed out" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
    # Python fallback still ran, so the actual match should come through.
    assert result.matches and any(m["path"].endswith("a.txt") for m in result.matches)
    # No partial ripgrep output was captured, so this is a clean fallback, not a truncation.
    assert result.truncated is False


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_timeout_returns_partial_results_when_output_captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ripgrep is killed after streaming matches, those partial matches are returned flagged as truncated."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(fs_module, "DEFAULT_GREP_TIMEOUT", 0.05)
    (tmp_path / "a.txt").write_text("hello\n")
    frame = json.dumps({"type": "match", "data": {"path": {"text": "a.txt"}, "lines": {"text": "hello\n"}, "line_number": 1}})

    def timeout_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        # One frame streams through, then the stream blocks until the watchdog
        # kills the process (mirroring a search that outran its time budget).
        return _FakePopen(stdout_lines=[frame + "\n"], block=True)

    monkeypatch.setattr(fs_module.subprocess, "Popen", timeout_popen)
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("hello", path=str(tmp_path))

    assert result.truncated is True
    assert result.error is None
    assert result.matches and any(m["path"].endswith("a.txt") and m["text"] == "hello" for m in result.matches)


def test_glob_times_out_and_flags_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`glob` returns whatever it gathered with `truncated=True` once its wall-clock budget elapses."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    # A negative budget puts the deadline in the past so the first iteration trips it.
    monkeypatch.setattr(fs_module, "_DEFAULT_GLOB_TIMEOUT", -1)

    result = be.glob("*.py")

    assert result.truncated is True
    assert result.error is None


def test_glob_zero_match_still_honors_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern that matches nothing still checks the deadline while walking (not only on matches)."""
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "a.txt").write_text("x")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    # A negative budget puts the deadline in the past so the walk trips it even
    # though the pattern never matches anything.
    monkeypatch.setattr(fs_module, "_DEFAULT_GLOB_TIMEOUT", -1)

    result = be.glob("__missing__")

    assert result.truncated is True
    assert result.error is None
    assert result.matches == []


def test_glob_returns_matches_gathered_before_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the deadline trips mid-walk, `glob` returns the matches found so far, not an empty list."""
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    monkeypatch.setattr(fs_module, "_DEFAULT_GLOB_TIMEOUT", 2)
    # First reading sets the deadline (0 + 2 = 2); the walk then processes two
    # entries (ticks 1, 2) before the third (tick 3) trips the deadline.
    ticks = iter([0, 1, 2, 3, 4, 5, 6])
    monkeypatch.setattr(fs_module.time, "monotonic", lambda: next(ticks))

    result = be.glob("*.py")

    assert result.truncated is True
    assert result.error is None
    # Partial, non-empty payload: some matches landed before the deadline.
    assert result.matches is not None
    assert 0 < len(result.matches) < 3


def test_glob_mid_iteration_oserror_is_error_not_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mid-walk failure surfaces as an error, distinct from a benign timeout truncation."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    _install_flaky_rglob(monkeypatch, OSError("simulated mid-walk failure"))

    result = be.glob("*")

    assert result.error is not None
    assert "aborted partway" in result.error
    assert result.truncated is False


def test_glob_backend_budget_below_middleware_deadline() -> None:
    """The backend glob budget must stay below the middleware's outer deadline so partial results win first."""
    assert fs_module._DEFAULT_GLOB_TIMEOUT < GLOB_TIMEOUT


def test_glob_supports_brace_expansion(tmp_path: Path) -> None:
    """Glob enables brace expansion via `wcmatch`, diverging from stdlib `rglob` (which is literal)."""
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.md").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    matches = {info["path"] for info in be.glob("*.{py,md}", path="/").matches or []}

    assert matches == {"/a.py", "/b.md"}


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_exec_race_logs_warning_and_clears_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """`FileNotFoundError` at exec (post-`which` race) warns with detail and re-probes next call."""
    which_calls = {"n": 0}

    def counting_which(_name: str) -> str | None:
        which_calls["n"] += 1
        return "/usr/bin/rg"

    monkeypatch.setattr(fs_module.shutil, "which", counting_which)

    def missing_popen(cmd: list[str], **_kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(fs_module.subprocess, "Popen", missing_popen)

    (tmp_path / "a.txt").write_text("hello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    with caplog.at_level(logging.WARNING, logger=fs_module.logger.name):
        be.grep("hello", path=str(tmp_path))
        be.grep("hello", path=str(tmp_path))

    failure_msgs = [r.getMessage() for r in caplog.records if "ripgrep subprocess failed" in r.getMessage()]
    assert failure_msgs
    assert "FileNotFoundError" in failure_msgs[0]
    assert "No such file or directory" in failure_msgs[0]  # str(e) carried through
    assert which_calls["n"] >= 2, "cache should have been cleared so `which` re-runs"


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_nonzero_returncode_falls_back_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hard ripgrep error (rc=2) must not be silently parsed as 'no matches'.

    Without this guard, malformed globs or unreadable directories produce an
    empty result set and the agent confidently reports no matches.
    """
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")

    def erroring_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        return _FakePopen(stdout_lines=[], stderr="rg: error parsing glob 'docs/[': unclosed character class", returncode=2)

    monkeypatch.setattr(fs_module.subprocess, "Popen", erroring_popen)

    (tmp_path / "a.txt").write_text("hello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    with caplog.at_level(logging.WARNING, logger=fs_module.logger.name):
        result = be.grep("hello", path=str(tmp_path))

    msgs = [r.getMessage() for r in caplog.records if "ripgrep exited 2" in r.getMessage()]
    assert msgs, [r.getMessage() for r in caplog.records]
    assert "error parsing glob" in msgs[0]
    # Python fallback ran and still returned the real match.
    assert result.matches and any(m["path"].endswith("a.txt") for m in result.matches)


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_nonzero_returncode_discards_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hard ripgrep error after a match reruns the complete Python search."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    frame = _rg_match_frame("a.txt", 1, "hello\n")

    def erroring_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        return _FakePopen(stdout_lines=[frame], stderr="rg: unreadable directory", returncode=2)

    monkeypatch.setattr(fs_module.subprocess, "Popen", erroring_popen)
    (tmp_path / "a.txt").write_text("hello\nhello\n")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    with caplog.at_level(logging.WARNING, logger=fs_module.logger.name):
        result = be.grep("hello", path=str(tmp_path))

    assert any("ripgrep exited 2" in record.getMessage() for record in caplog.records)
    assert result.error is None
    assert result.matches is not None
    assert len(result.matches) == 2


def _rg_match_frame(path: str, line_number: int, text: str) -> str:
    """Build a ripgrep `--json` match frame line for the streaming fake."""
    return json.dumps({"type": "match", "data": {"path": {"text": path}, "lines": {"text": text}, "line_number": line_number}}) + "\n"


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_streaming_caps_total_and_terminates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The streaming ripgrep path stops at `max_count` total matches and kills the process early."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("hello\nhello\nhello\n")

    # More frames than the cap, spread across files, so the cap must apply
    # across files rather than per file.
    frames = [
        _rg_match_frame("a.txt", 1, "hello\n"),
        _rg_match_frame("a.txt", 2, "hello\n"),
        _rg_match_frame("b.txt", 1, "hello\n"),
        _rg_match_frame("b.txt", 2, "hello\n"),
    ]
    created: dict[str, _FakePopen] = {}

    def fake_popen(cmd: list[str], **_kwargs: object) -> _FakePopen:
        # `-m <cap + 1>` is passed to ripgrep as a secondary per-file guard; the
        # `+ 1` lets a single file emit one match past the cap so truncation is
        # detectable.
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "3"
        proc = _FakePopen(stdout_lines=frames)
        created["proc"] = proc
        return proc

    monkeypatch.setattr(fs_module.subprocess, "Popen", fake_popen)
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("hello", path=str(tmp_path), max_count=2)

    assert result.truncated is True
    assert result.matches is not None
    assert len(result.matches) == 2
    # The process was terminated once the cap was reached instead of draining.
    assert created["proc"].terminated is True


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_streaming_below_cap_not_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When fewer matches than `max_count` are emitted, the result completes untruncated."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    (tmp_path / "a.txt").write_text("hello\n")
    frames = [_rg_match_frame("a.txt", 1, "hello\n")]

    def fake_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        return _FakePopen(stdout_lines=frames)

    monkeypatch.setattr(fs_module.subprocess, "Popen", fake_popen)
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("hello", path=str(tmp_path), max_count=100)

    assert result.truncated is False
    assert result.matches is not None
    assert len(result.matches) == 1


@pytest.mark.usefixtures("_isolate_rg_cache")
def test_ripgrep_streaming_exact_cap_not_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exactly `max_count` matches with none dropped is reported complete."""
    monkeypatch.setattr(fs_module.shutil, "which", lambda _name: "/usr/bin/rg")
    (tmp_path / "a.txt").write_text("hello\nhello\n")
    # Exactly `max_count` frames and no more: ripgrep (`-m cap + 1`) would have
    # emitted a third if one existed, so the stream ending at the cap proves the
    # result is complete.
    frames = [_rg_match_frame("a.txt", 1, "hello\n"), _rg_match_frame("a.txt", 2, "hello\n")]

    def fake_popen(_cmd: list[str], **_kwargs: object) -> _FakePopen:
        return _FakePopen(stdout_lines=frames)

    monkeypatch.setattr(fs_module.subprocess, "Popen", fake_popen)
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("hello", path=str(tmp_path), max_count=2)

    assert result.truncated is False
    assert result.matches is not None
    assert len(result.matches) == 2


def test_python_fallback_caps_total_matches_across_files(tmp_path: Path) -> None:
    """The Python fallback caps total matches across files and flags truncation.

    ripgrep is absent in this environment, so `grep` uses `_python_search`.
    """
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("needle\nneedle\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("needle", path=str(tmp_path), max_count=3)

    assert result.truncated is True
    assert result.matches is not None
    assert len(result.matches) == 3


def test_python_fallback_no_cap_returns_all_matches(tmp_path: Path) -> None:
    """With `max_count=None` the Python fallback returns every match, untruncated."""
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("needle\nneedle\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("needle", path=str(tmp_path))

    assert result.truncated is False
    assert result.matches is not None
    assert len(result.matches) == 6


def test_python_fallback_below_cap_not_truncated(tmp_path: Path) -> None:
    """A cap larger than the number of matches leaves the result untruncated."""
    (tmp_path / "a.txt").write_text("needle\nneedle\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("needle", path=str(tmp_path), max_count=100)

    assert result.truncated is False
    assert result.matches is not None
    assert len(result.matches) == 2


def test_python_fallback_exact_cap_not_truncated(tmp_path: Path) -> None:
    """The Python fallback reports exactly `max_count` matches as complete."""
    (tmp_path / "a.txt").write_text("needle\nneedle\n")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    result = be.grep("needle", path=str(tmp_path), max_count=2)

    assert result.truncated is False
    assert result.matches is not None
    assert len(result.matches) == 2


def _install_flaky_rglob(monkeypatch: pytest.MonkeyPatch, exc: Exception, after_yields: int = 1) -> None:
    """Replace `Path.rglob` with a generator that yields N entries then raises."""
    real_rglob = Path.rglob

    def flaky_rglob(self: Path, pattern: str):
        for idx, entry in enumerate(sorted(real_rglob(self, pattern)), start=1):
            yield entry
            if idx >= after_yields:
                raise exc

    monkeypatch.setattr(Path, "rglob", flaky_rglob)


@pytest.mark.parametrize("virtual_mode", [False, True])
def test_grep_python_fallback_survives_mid_iteration_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, virtual_mode: bool) -> None:
    """Python grep fallback returns accumulated matches when `rglob` aborts mid-walk.

    `Path.rglob` can raise `FileNotFoundError` (or other `OSError` subclasses)
    when a directory entry is unlinked or renamed while the walk is in
    progress. The fallback must surface a partial result rather than letting
    the exception escape and fail the whole tool invocation.
    """
    root = tmp_path / "project"
    root.mkdir()
    (root / "first.txt").write_text("hello world\n")
    (root / "second.txt").write_text("hello world\n")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=virtual_mode)
    monkeypatch.setattr(be, "_ripgrep_search", lambda *_a, **_k: (None, False))
    _install_flaky_rglob(monkeypatch, FileNotFoundError("simulated mid-walk unlink"))

    grep_path = "/" if virtual_mode else str(root)
    result = be.grep("hello", path=grep_path)

    assert result.matches is not None
    assert result.error is not None
    assert "aborted" in result.error
    if virtual_mode:
        # The real `root_dir` must not leak into the agent-visible error.
        assert str(root) not in result.error
    else:
        assert str(root) in result.error

    matched_paths = {m["path"] for m in result.matches}
    if virtual_mode:
        assert "/first.txt" in matched_paths
        assert "/second.txt" not in matched_paths
    else:
        assert any(p.endswith("first.txt") for p in matched_paths)
        assert not any(p.endswith("second.txt") for p in matched_paths)


def test_grep_python_fallback_survives_runtime_error_mid_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`RuntimeError` from `rglob` (e.g. symlink-loop detection) is also recoverable."""
    root = tmp_path
    (root / "a.txt").write_text("hello\n")
    (root / "b.txt").write_text("hello\n")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)
    monkeypatch.setattr(be, "_ripgrep_search", lambda *_a, **_k: (None, False))
    _install_flaky_rglob(monkeypatch, RuntimeError("symlink loop"))

    result = be.grep("hello", path=str(root))

    assert result.error is not None
    assert "symlink loop" in result.error
    assert result.matches


def test_grep_virtual_mode_sanitizes_runtime_error_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Virtual grep fallback must not expose real paths from `RuntimeError` text."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("hello\n")
    (root / "b.txt").write_text("hello\n")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    monkeypatch.setattr(be, "_ripgrep_search", lambda *_a, **_k: (None, False))
    _install_flaky_rglob(monkeypatch, RuntimeError(f"symlink loop under {root}"))

    result = be.grep("hello", path="/")

    assert result.error is not None
    assert "aborted" in result.error
    assert "RuntimeError" in result.error
    assert str(root) not in result.error
    assert "symlink loop under" not in result.error
    assert result.matches


def test_grep_virtual_mode_sanitizes_oserror_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Virtual grep fallback strips the real path embedded in a mid-walk `OSError`.

    A realistic `rglob` failure (an entry unlinked mid-walk) raises an `OSError`
    whose `str()` embeds the absolute filename. The agent-visible error must
    carry only the path-free `strerror` reason, never the real path.
    """
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("hello\n")
    (root / "b.txt").write_text("hello\n")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    monkeypatch.setattr(be, "_ripgrep_search", lambda *_a, **_k: (None, False))
    gone = str(root / "gone.txt")
    _install_flaky_rglob(monkeypatch, FileNotFoundError(2, "No such file or directory", gone))

    result = be.grep("hello", path="/")

    assert result.error is not None
    assert "aborted" in result.error
    assert "No such file or directory" in result.error  # path-free reason survives
    assert str(root) not in result.error  # real root must not leak
    assert "gone.txt" not in result.error  # embedded filename must not leak
    assert result.matches


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


class TestDisplayPath:
    """Tests for FilesystemBackend._display_path."""

    def test_non_virtual_returns_real_path(self, tmp_path: Path) -> None:
        """Non-virtual mode returns the real path unchanged."""
        target = tmp_path / "src" / "file.py"
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        assert be._display_path(target) == str(target)

    def test_virtual_returns_virtual_path(self, tmp_path: Path) -> None:
        """Virtual mode converts to the virtual path."""
        (tmp_path / "src").mkdir()
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        assert be._display_path(tmp_path / "src" / "file.py") == "/src/file.py"

    def test_virtual_out_of_root_falls_back_to_basename(self, tmp_path: Path) -> None:
        """A path outside the root makes `_to_virtual_path` raise; only the basename is shown."""
        sub = tmp_path / "sub"
        sub.mkdir()
        be = FilesystemBackend(root_dir=str(sub), virtual_mode=True)
        result = be._display_path(tmp_path / "secret" / "leak.txt")
        assert result == "leak.txt"  # bare name only
        assert str(tmp_path) not in result  # parent chain / real root absent

    def test_virtual_root_path_falls_back_to_slash(self, tmp_path: Path) -> None:
        """A root path (empty `.name`) falls back to `/`, not an empty string."""
        sub = tmp_path / "sub"
        sub.mkdir()
        be = FilesystemBackend(root_dir=str(sub), virtual_mode=True)
        assert be._display_path(Path("/")) == "/"


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


class _FailingHandle:
    """File-handle stub that yields one line, then raises `UnicodeDecodeError`.

    Simulates a file that decodes cleanly at first and fails partway through —
    the case a real undecodable file cannot reproduce, since a real binary file
    fails on the first read and exercises the silent-skip path instead.
    """

    def __init__(self, first_line: str = "needle before failure\n") -> None:
        self._first_line = first_line
        self._emitted = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        if not self._emitted:
            self._emitted = True
            return self._first_line
        encoding = "utf-8"
        data = b"\xff"
        reason = "invalid start byte"
        raise UnicodeDecodeError(encoding, data, 0, 1, reason)


def _fake_open_for(target: Path, handle: object) -> object:
    """Build a `Path.open` replacement that returns `handle` for `target` only."""
    original_open = Path.open

    def fake_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == target:
            return handle
        return original_open(path, *args, **kwargs)

    return fake_open


class TestGrepPythonFallbackTimeout:
    """Tests for the wall-clock timeout on the Python grep fallback."""

    def test_python_search_times_out_with_zero_timeout(self, tmp_path: Path) -> None:
        """`_python_search` flags `truncated` (not an error) when the deadline is exceeded."""
        (tmp_path / "file.txt").write_text("hello")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        _results, truncated, partial_error = be._python_search("hello", tmp_path, None, timeout=0)
        assert truncated is True
        # A timeout is not a hard error; matches are valid but incomplete.
        assert partial_error is None

    def test_python_search_matches_literal_substrings(self, tmp_path: Path) -> None:
        """The Python fallback does literal substring matching (no regex)."""
        (tmp_path / "code.py").write_text("def __init__(self):\n    return [a-z]\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("[a-z]", tmp_path, None)
        assert partial_error is None
        all_lines = [text for items in results.values() for _, text in items]
        assert any("[a-z]" in line for line in all_lines)

    def test_python_search_streams_large_file_with_per_line_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The per-line deadline check interrupts a single large file mid-stream.

        `time.monotonic` is stubbed to advance 1s per call. Inside
        `_python_search` it is called once to set the deadline, once for the
        outer per-file check, then on each 2048-line boundary (here only the
        check at line 2048 fires before the function returns). With the deadline
        1.5s out, the outer check passes (so the file is opened) and the first
        in-file check at line 2048 trips. This proves the mid-file branch runs
        rather than the outer guard short-circuiting before any read — which is
        what a `timeout=0` test (see `test_python_search_times_out_with_zero_timeout`)
        cannot distinguish.
        """
        lines = [f"line {i}" for i in range(1, 2501)]
        lines[0] = "needle"  # line 1 — scanned before the in-file timeout
        lines[2499] = "needle"  # line 2500 — never reached
        (tmp_path / "big.txt").write_text("\n".join(lines) + "\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        clock = {"t": 1000.0}

        def fake_monotonic() -> float:
            t = clock["t"]
            clock["t"] += 1.0
            return t

        monkeypatch.setattr(fs_module.time, "monotonic", fake_monotonic)
        results, truncated, partial_error = be._python_search("needle", tmp_path, None, timeout=1.5)

        assert truncated is True
        assert partial_error is None
        collected = results.get("/big.txt", [])
        assert (1, "needle") in collected
        assert all(line_num != 2500 for line_num, _ in collected)

    def test_python_search_does_not_match_regex_metacharacters(self, tmp_path: Path) -> None:
        """The fallback is a literal substring search: regex metacharacters are inert.

        Both patterns would match via `re.search` but must not match literally,
        which is the load-bearing distinction now that the regex compile is gone.
        """
        (tmp_path / "code.py").write_text("axb\nab\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        no_dot, _td, err_dot = be._python_search("a.b", tmp_path, None)  # regex would match "axb"
        assert err_dot is None
        assert no_dot == {}

        no_star, _ts, err_star = be._python_search("a*b", tmp_path, None)  # regex would match "ab"
        assert err_star is None
        assert no_star == {}

    def test_python_search_strips_carriage_returns(self, tmp_path: Path) -> None:
        """CRLF files yield clean match text via universal-newline translation on read."""
        (tmp_path / "crlf.txt").write_bytes(b"hit me\r\nother\r\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("hit", tmp_path, None)
        assert partial_error is None
        matches = results.get("/crlf.txt", [])
        assert matches == [(1, "hit me")]

    def test_python_search_skips_non_utf8_file(self, tmp_path: Path) -> None:
        """A wholly-undecodable file is skipped silently while valid files still match.

        The decode fails on the first byte, so no lines are scanned and the file
        is not reported as a partial read — mirroring ripgrep's binary-file skip.
        """
        (tmp_path / "good.txt").write_text("needle here\n")
        (tmp_path / "bad.bin").write_bytes(b"\xff\xfe needle \x00\x80")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)
        assert partial_error is None
        assert results.get("/good.txt") == [(1, "needle here")]
        assert "/bad.bin" not in results

    def test_python_search_reports_file_error_after_partial_scan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A per-file read error after scanning starts is surfaced with partial matches."""
        bad = tmp_path / "bad.txt"
        bad.write_text("")
        monkeypatch.setattr(Path, "open", _fake_open_for(bad, _FailingHandle()))
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)

        assert results == {"/bad.txt": [(1, "needle before failure")]}
        assert partial_error is not None
        assert "One or more files could not be fully searched" in partial_error
        assert "- /bad.txt: UnicodeDecodeError: invalid start byte" in partial_error

    def test_python_search_surfaces_open_failure_without_scanned_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An open failure is surfaced even when no lines were scanned.

        Unlike an undecodable binary (skipped silently), a file the caller asked
        to search but that could not be opened is reported via `partial_error`.
        """
        target = tmp_path / "locked.txt"
        target.write_text("needle\n")
        original_open = Path.open

        def fake_open(path: Path, *args: object, **kwargs: object) -> object:
            if path == target:
                msg = "nope"
                raise PermissionError(msg)
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fake_open)
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)

        assert results == {}
        assert partial_error is not None
        assert "- /locked.txt: PermissionError" in partial_error

    def test_python_search_returns_matches_when_another_file_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A clean file's matches are returned while a sibling's read error is surfaced."""
        good = tmp_path / "good.txt"
        good.write_text("needle good\n")
        bad = tmp_path / "bad.txt"
        bad.write_text("")
        monkeypatch.setattr(Path, "open", _fake_open_for(bad, _FailingHandle()))
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)

        assert results.get("/good.txt") == [(1, "needle good")]
        assert partial_error is not None
        assert "/bad.txt" in partial_error
        assert "/good.txt" not in partial_error

    def test_python_search_glob_filters_files(self, tmp_path: Path) -> None:
        """The compiled glob matcher restricts the fallback to matching filenames."""
        (tmp_path / "match.py").write_text("needle in py\n")
        (tmp_path / "skip.txt").write_text("needle in txt\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, "*.py")
        assert partial_error is None
        assert results.get("/match.py") == [(1, "needle in py")]
        assert "/skip.txt" not in results

    def test_python_search_glob_matches_directory_components(self, tmp_path: Path) -> None:
        """Directory-component globs filter nested files (GLOBSTAR/BRACE flags)."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("needle doc\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "b.md").write_text("needle src\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, "docs/*.md")
        assert partial_error is None
        assert results.get("/docs/a.md") == [(1, "needle doc")]
        assert "/src/b.md" not in results

    def test_python_search_skips_file_over_size_limit(self, tmp_path: Path) -> None:
        """Files exceeding `max_file_size_bytes` are skipped without a partial error."""
        (tmp_path / "big.txt").write_text("needle\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        be.max_file_size_bytes = 3  # smaller than the file; forces the size-skip branch
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)
        assert partial_error is None
        assert results == {}

    def test_python_search_non_virtual_mode_keys_on_absolute_path(self, tmp_path: Path) -> None:
        """In non-virtual mode the result key is the absolute filesystem path."""
        target = tmp_path / "f.txt"
        target.write_text("needle\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)
        assert partial_error is None
        assert results == {str(target): [(1, "needle")]}

    def test_python_search_accumulates_multiple_matches_per_file(self, tmp_path: Path) -> None:
        """All matching lines are collected in order with 1-based line numbers."""
        (tmp_path / "f.txt").write_text("needle\nno\nneedle\nno\nneedle\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("needle", tmp_path, None)
        assert partial_error is None
        assert results == {"/f.txt": [(1, "needle"), (3, "needle"), (5, "needle")]}

    def test_python_search_handles_cr_only_line_endings(self, tmp_path: Path) -> None:
        """Classic-Mac CR line endings are normalized by universal-newline translation."""
        (tmp_path / "cr.txt").write_bytes(b"hit me\rother\r")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        results, _truncated, partial_error = be._python_search("hit", tmp_path, None)
        assert partial_error is None
        assert results.get("/cr.txt") == [(1, "hit me")]

    def test_grep_fallback_treats_pattern_literally(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`grep` passes the raw pattern to the fallback (no `re.escape`), matching literally.

        Stubbing `_ripgrep_search` to `None` forces the Python fallback regardless
        of whether ripgrep is installed, locking the call-site contract.
        """
        (tmp_path / "f.txt").write_text("a.b\naxb\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        monkeypatch.setattr(be, "_ripgrep_search", lambda *_args, **_kwargs: (None, False))
        matches = be.grep("a.b", path="/").matches
        assert matches is not None
        texts = [m["text"] for m in matches]
        assert "a.b" in texts
        assert "axb" not in texts

    def test_grep_flags_timeout_as_truncated_with_partial_results(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`grep` flags a fallback timeout as `truncated` while still returning matches found so far."""
        (tmp_path / "file.txt").write_text("hello")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        monkeypatch.setattr(FilesystemBackend, "_ripgrep_search", lambda *_a, **_kw: (None, False))
        monkeypatch.setattr(
            be,
            "_python_search",
            lambda *_a, **_kw: ({"/file.txt": [(1, "hello")]}, True, None),
        )
        result = be.grep("hello", path="/")
        assert result.truncated is True
        assert result.error is None
        # Partial matches collected before the timeout are preserved.
        assert result.matches
        assert result.matches[0]["path"] == "/file.txt"


class TestGrepPythonFallbackIncludeGlob:
    """The Python grep fallback shares ripgrep-like include-glob semantics.

    Stubbing `_ripgrep_search` to `None` forces the Python fallback regardless
    of whether ripgrep is installed, so these lock the shared contract:

    - A slashless pattern (`*.py`) matches the basename at any depth.
    - A path-containing pattern (`src/**/*.py`) matches relative to the root.
    """

    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FilesystemBackend:
        (tmp_path / "src" / "app").mkdir(parents=True)
        (tmp_path / "src" / "app" / "main.py").write_text("import os\n")
        (tmp_path / "top.py").write_text("import sys\n")
        (tmp_path / "README.md").write_text("import note\n")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        monkeypatch.setattr(be, "_ripgrep_search", lambda *_a, **_k: (None, False))
        return be

    def _paths(self, be: FilesystemBackend, glob: str, path: str = "/") -> list[str]:
        result = be.grep("import", path=path, glob=glob)
        assert result.matches is not None
        return sorted(m["path"] for m in result.matches)

    def test_directory_glob_matches_nested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        be = self._setup(tmp_path, monkeypatch)
        assert self._paths(be, "src/**/*.py") == ["/src/app/main.py"]

    def test_recursive_glob_matches_all_python(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        be = self._setup(tmp_path, monkeypatch)
        assert self._paths(be, "**/*.py") == ["/src/app/main.py", "/top.py"]

    def test_slashless_glob_matches_at_any_depth(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        be = self._setup(tmp_path, monkeypatch)
        assert self._paths(be, "*.py") == ["/src/app/main.py", "/top.py"]

    def test_negative_glob(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        be = self._setup(tmp_path, monkeypatch)
        assert self._paths(be, "*.md") == ["/README.md"]

    def test_glob_relative_to_search_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        be = self._setup(tmp_path, monkeypatch)
        assert self._paths(be, "app/*.py", path="/src") == ["/src/app/main.py"]


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


def test_ls_nonexistent_path_sets_error(tmp_path: Path) -> None:
    """Ls on a missing path must surface the failure on .error, not return []."""
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    result = be.ls("/missing/")

    assert result.entries is None
    assert result.error == "Path '/missing/': path_not_found"


def test_ls_file_path_sets_not_a_directory_error(tmp_path: Path) -> None:
    """Ls on a file path must surface not_a_directory on .error."""
    write_file(tmp_path / "file.txt", "content")
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    result = be.ls("/file.txt")

    assert result.entries is None
    assert result.error == "Path '/file.txt': not_a_directory"


def test_ls_empty_directory_returns_empty_entries(tmp_path: Path) -> None:
    """Ls on an empty directory returns success with an empty entries list."""
    (tmp_path / "empty").mkdir()
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    result = be.ls("/empty/")

    assert result.error is None
    assert result.entries == []


def test_ls_symlink_loop_path_returns_structured_error(tmp_path: Path) -> None:
    """A resolver failure in `ls` should not escape the backend boundary."""
    make_symlink_loop(tmp_path / "loop")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    result = be.ls("loop")

    assert result.entries is None
    assert result.error is not None
    assert "Cannot list 'loop'" in result.error


def test_ls_virtual_mode_reports_child_symlink_loop(tmp_path: Path) -> None:
    """Virtual listings should report cyclic children without raising."""
    make_symlink_loop(tmp_path / "loop")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    result = be.ls("/")

    assert result.error is not None
    # Per-child failures are prefixed so callers can tell them apart from a
    # top-level listing failure.
    assert "child error:" in result.error
    assert "loop" in result.error
    assert result.entries == []


def test_file_operations_return_errors_for_symlink_loop_paths(tmp_path: Path) -> None:
    """Resolver failures should become operation errors for model-facing APIs.

    Asserts on the resolver-failure phrase so a regression that drops the
    `_resolve_path` guard (and falls back to "File not found") would fail.
    """
    make_symlink_loop(tmp_path / "loop")

    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    read_error = be.read("loop").error
    assert read_error is not None
    assert "Error reading file 'loop'" in read_error

    write_error = be.write("loop", "content").error
    assert write_error is not None
    assert "Error writing file 'loop'" in write_error

    edit_error = be.edit("loop", "old", "new").error
    assert edit_error is not None
    assert "Error editing file 'loop'" in edit_error

    grep_error = be.grep("needle", path="loop").error
    assert grep_error is not None
    assert "Error searching path 'loop'" in grep_error

    glob_error = be.glob("*", path="loop").error
    assert glob_error is not None
    assert "Error globbing path 'loop'" in glob_error

    assert be.upload_files([("loop", b"content")])[0].error == "invalid_path"
    assert be.download_files(["loop"])[0].error == "invalid_path"


class TestVirtualModeDefaultDeprecation:
    """`virtual_mode=None` (omitted) emits a deprecation; explicit values do not."""

    def test_omitted_virtual_mode_warns(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            be = FilesystemBackend(root_dir=str(tmp_path))

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert deprecations[0].category is LangChainDeprecationWarning
        assert "virtual_mode" in str(deprecations[0].message)
        # Default falls back to `False` for backwards compatibility.
        assert be.virtual_mode is False

    def test_explicit_virtual_mode_does_not_warn(self, tmp_path: Path) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
            FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning) and "virtual_mode" in str(w.message)]
        assert deprecations == []


class TestReadTrailingNewlineRoundtrip:
    """`FilesystemBackend.read` must round-trip the file's trailing-newline state.

    That state feeds `perform_string_replacement`'s EOF-mismatch detection.
    Dropping it here re-introduces the silent-failure loop from #2856.
    """

    def test_preserves_trailing_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "with_newline.txt"
        target.write_text("foo\nbar\n")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        result = be.read(str(target))

        assert isinstance(result, ReadResult)
        assert result.file_data is not None
        assert result.file_data["content"] == "foo\nbar\n"

    def test_preserves_no_trailing_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "no_newline.txt"
        target.write_text("foo\nbar")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        result = be.read(str(target))

        assert isinstance(result, ReadResult)
        assert result.file_data is not None
        assert result.file_data["content"] == "foo\nbar"

    def test_read_then_edit_eof_mismatch_surfaces_hint(self, tmp_path: Path) -> None:
        """End-to-end: read+edit on an unterminated file emits the EOF hint.

        Pins the #2856 fix at the boundary that matters — the model-facing
        flow — not just the inner predicate.
        """
        target = tmp_path / "memory.md"
        target.write_text("# Agent Role:\nyou are an assistant")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        result = be.edit(
            str(target),
            "# Agent Role:\nyou are an assistant\n",
            "# Agent Role:\nyou are an assistant\nYou can do anything\n",
        )

        assert result.error is not None
        assert "old_string ends with a newline" in result.error
        assert target.read_text() == "# Agent Role:\nyou are an assistant"


class TestFilesystemDelete:
    """Tests for FilesystemBackend.delete."""

    def test_delete_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        write_file(f, "hello")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        result = be.delete("/a.txt")
        assert isinstance(result, DeleteResult)
        assert result.error is None
        assert result.path == "/a.txt"
        assert not f.exists()

    def test_delete_missing_file_returns_error(self, tmp_path: Path) -> None:
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/missing.txt")
        assert result.path is None
        assert result.error is not None
        assert "not found" in result.error

    def test_delete_empty_directory(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/sub")
        assert result.error is None
        assert result.path == "/sub"
        assert not (tmp_path / "sub").exists()

    def test_delete_directory_recursively(self, tmp_path: Path) -> None:
        # A directory is removed along with all of its nested contents.
        sub = tmp_path / "sub"
        (sub / "deep").mkdir(parents=True)
        write_file(sub / "b.txt", "b")
        write_file(sub / "deep" / "d.txt", "d")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        result = be.delete("/sub")
        assert result.error is None
        assert result.path == "/sub"
        assert not sub.exists()

    def test_delete_directory_leaves_siblings(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        write_file(tmp_path / "sub" / "b.txt", "b")
        write_file(tmp_path / "keep.txt", "keep")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        assert be.delete("/sub").error is None
        assert not (tmp_path / "sub").exists()
        assert (tmp_path / "keep.txt").exists()

    def test_delete_symlink_to_dir_does_not_follow(self, tmp_path: Path) -> None:
        # Deleting a symlink that points at a directory removes only the link;
        target = tmp_path / "target"
        target.mkdir()
        write_file(target / "keep.txt", "keep")
        link = tmp_path / "link"
        link.symlink_to(target, target_is_directory=True)
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

        result = be.delete(str(link))
        assert result.error is None
        assert not link.exists()
        assert target.exists()
        assert (target / "keep.txt").exists()

    def test_delete_only_removes_target(self, tmp_path: Path) -> None:
        write_file(tmp_path / "keep.txt", "keep")
        write_file(tmp_path / "drop.txt", "drop")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        assert be.delete("/drop.txt").error is None
        assert not (tmp_path / "drop.txt").exists()
        assert (tmp_path / "keep.txt").exists()

    async def test_adelete_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        write_file(f, "hello")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        result = await be.adelete("/a.txt")
        assert result.error is None
        assert result.path == "/a.txt"
        assert not f.exists()

    def test_delete_resolve_failure_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A failure resolving the path surfaces as a deletion error, not a raise.
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        def _boom(_path: str) -> Path:
            raise OSError

        monkeypatch.setattr(be, "_resolve_path", _boom)
        result = be.delete("/a.txt")
        assert result.path is None
        assert result.error is not None
        assert "Error deleting" in result.error

    def test_delete_unlink_failure_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # An OSError from unlink (e.g. permission denied) is reported, file kept.
        f = tmp_path / "a.txt"
        write_file(f, "hello")
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

        def _boom(_self: Path, *_args: object, **_kwargs: object) -> None:
            raise OSError

        monkeypatch.setattr(Path, "unlink", _boom)
        result = be.delete("/a.txt")
        assert result.path is None
        assert result.error is not None
        assert "Error deleting" in result.error
        assert f.exists()
