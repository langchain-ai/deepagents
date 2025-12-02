"""Async tests for FilesystemBackend."""

from pathlib import Path

import pytest

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult, WriteResult


def write_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


async def test_filesystem_backend_async_normal_mode(tmp_path: Path):
    """Test async operations in normal (non-virtual) mode."""
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.py"
    write_file(f1, "hello fs")
    write_file(f2, "print('x')\nhello")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    # als_info absolute path - should only list files in root, not subdirectories
    infos = await be.als_info(str(root))
    paths = {i["path"] for i in infos}
    assert str(f1) in paths  # File in root should be listed
    assert str(f2) not in paths  # File in subdirectory should NOT be listed
    assert (str(root) + "/dir/") in paths  # Directory should be listed

    # aread, aedit, awrite
    txt = await be.aread(str(f1))
    assert "hello fs" in txt
    msg = await be.aedit(str(f1), "fs", "filesystem", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1
    msg2 = await be.awrite(str(root / "new.txt"), "new content")
    assert isinstance(msg2, WriteResult) and msg2.error is None and msg2.path.endswith("new.txt")

    # agrep_raw
    matches = await be.agrep_raw("hello", path=str(root))
    assert isinstance(matches, list) and any(m["path"].endswith("a.txt") for m in matches)

    # aglob_info
    g = await be.aglob_info("*.py", path=str(root))
    assert any(i["path"] == str(f2) for i in g)


async def test_filesystem_backend_async_virtual_mode(tmp_path: Path):
    """Test async operations in virtual mode."""
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.md"
    write_file(f1, "hello virtual")
    write_file(f2, "content")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # als_info from virtual root - should only list files in root, not subdirectories
    infos = await be.als_info("/")
    paths = {i["path"] for i in infos}
    assert "/a.txt" in paths  # File in root should be listed
    assert "/dir/b.md" not in paths  # File in subdirectory should NOT be listed
    assert "/dir/" in paths  # Directory should be listed

    # aread and aedit via virtual path
    txt = await be.aread("/a.txt")
    assert "hello virtual" in txt
    msg = await be.aedit("/a.txt", "virtual", "virt", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1

    # awrite new file via virtual path
    msg2 = await be.awrite("/new.txt", "x")
    assert isinstance(msg2, WriteResult) and msg2.error is None
    assert (root / "new.txt").exists()

    # agrep_raw limited to path
    matches = await be.agrep_raw("virt", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/a.txt" for m in matches)

    # aglob_info
    g = await be.aglob_info("**/*.md", path="/")
    assert any(i["path"] == "/dir/b.md" for i in g)

    # invalid regex returns error string
    err = await be.agrep_raw("[", path="/")
    assert isinstance(err, str)

    # path traversal blocked
    with pytest.raises(ValueError):
        await be.aread("/../a.txt")


async def test_filesystem_backend_als_nested_directories(tmp_path: Path):
    """Test async ls with nested directories."""
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

    root_listing = await be.als_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths

    src_listing = await be.als_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = await be.als_info("/src/utils/")
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = await be.als_info("/nonexistent/")
    assert empty_listing == []


async def test_filesystem_backend_als_normal_mode_nested(tmp_path: Path):
    """Test async ls_info with nested directories in normal (non-virtual) mode."""
    root = tmp_path

    files = {
        root / "file1.txt": "content1",
        root / "subdir" / "file2.txt": "content2",
        root / "subdir" / "nested" / "file3.txt": "content3",
    }

    for path, content in files.items():
        write_file(path, content)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=False)

    root_listing = await be.als_info(str(root))
    root_paths = [fi["path"] for fi in root_listing]

    assert str(root / "file1.txt") in root_paths
    assert str(root / "subdir") + "/" in root_paths
    assert str(root / "subdir" / "file2.txt") not in root_paths

    subdir_listing = await be.als_info(str(root / "subdir"))
    subdir_paths = [fi["path"] for fi in subdir_listing]
    assert str(root / "subdir" / "file2.txt") in subdir_paths
    assert str(root / "subdir" / "nested") + "/" in subdir_paths
    assert str(root / "subdir" / "nested" / "file3.txt") not in subdir_paths


async def test_filesystem_backend_als_trailing_slash(tmp_path: Path):
    """Test async ls_info edge cases with trailing slashes."""
    root = tmp_path

    files = {
        root / "file.txt": "content",
        root / "dir" / "nested.txt": "nested",
    }

    for path, content in files.items():
        write_file(path, content)

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    listing_with_slash = await be.als_info("/")
    assert len(listing_with_slash) > 0

    listing = await be.als_info("/")
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    listing1 = await be.als_info("/dir/")
    listing2 = await be.als_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]

    empty = await be.als_info("/nonexistent/")
    assert empty == []


async def test_filesystem_backend_intercept_large_tool_result_async(tmp_path: Path):
    """Test that FilesystemBackend properly handles large tool result interception in async context."""
    from langchain.tools import ToolRuntime
    from langchain_core.messages import ToolMessage

    from deepagents.middleware.filesystem import FilesystemMiddleware

    root = tmp_path
    rt = ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id="test_fs",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )

    middleware = FilesystemMiddleware(backend=lambda r: FilesystemBackend(root_dir=str(root), virtual_mode=True), tool_token_limit_before_evict=1000)

    large_content = "f" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_fs_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_fs_123" in result.content
    saved_file = root / "large_tool_results" / "test_fs_123"
    assert saved_file.exists()
    assert saved_file.read_text() == large_content


async def test_filesystem_aupload_single_file(tmp_path: Path):
    """Test async uploading a single binary file."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    test_path = "/test_upload.bin"
    test_content = b"Hello, Binary World!"

    responses = await be.aupload_files([(test_path, test_content)])

    assert len(responses) == 1
    assert responses[0].path == test_path
    assert responses[0].error is None

    # Verify file exists and content matches
    uploaded_file = root / "test_upload.bin"
    assert uploaded_file.exists()
    assert uploaded_file.read_bytes() == test_content


async def test_filesystem_aupload_multiple_files(tmp_path: Path):
    """Test async uploading multiple files in one call."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    files = [
        ("/file1.bin", b"Content 1"),
        ("/file2.bin", b"Content 2"),
        ("/subdir/file3.bin", b"Content 3"),
    ]

    responses = await be.aupload_files(files)

    assert len(responses) == 3
    for i, (path, content) in enumerate(files):
        assert responses[i].path == path
        assert responses[i].error is None

    # Verify all files created
    assert (root / "file1.bin").read_bytes() == b"Content 1"
    assert (root / "file2.bin").read_bytes() == b"Content 2"
    assert (root / "subdir" / "file3.bin").read_bytes() == b"Content 3"


async def test_filesystem_adownload_single_file(tmp_path: Path):
    """Test async downloading a single file."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create a file manually
    test_file = root / "test_download.bin"
    test_content = b"Download me!"
    test_file.write_bytes(test_content)

    responses = await be.adownload_files(["/test_download.bin"])

    assert len(responses) == 1
    assert responses[0].path == "/test_download.bin"
    assert responses[0].content == test_content
    assert responses[0].error is None


async def test_filesystem_adownload_multiple_files(tmp_path: Path):
    """Test async downloading multiple files in one call."""
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
    responses = await be.adownload_files(paths)

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


async def test_filesystem_aupload_download_roundtrip(tmp_path: Path):
    """Test async upload followed by download for data integrity."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test with binary content including special bytes
    test_path = "/roundtrip.bin"
    test_content = bytes(range(256))  # All possible byte values

    # Upload
    upload_responses = await be.aupload_files([(test_path, test_content)])
    assert upload_responses[0].error is None

    # Download
    download_responses = await be.adownload_files([test_path])
    assert download_responses[0].error is None
    assert download_responses[0].content == test_content


async def test_filesystem_adownload_errors(tmp_path: Path):
    """Test async download error handling."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test file_not_found
    responses = await be.adownload_files(["/nonexistent.txt"])
    assert len(responses) == 1
    assert responses[0].path == "/nonexistent.txt"
    assert responses[0].content is None
    assert responses[0].error == "file_not_found"

    # Test is_directory
    (root / "testdir").mkdir()
    responses = await be.adownload_files(["/testdir"])
    assert responses[0].error == "is_directory"
    assert responses[0].content is None

    # Test invalid_path (path traversal)
    responses = await be.adownload_files(["/../etc/passwd"])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"
    assert responses[0].content is None


async def test_filesystem_aupload_errors(tmp_path: Path):
    """Test async upload error handling."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Test invalid_path (path traversal)
    responses = await be.aupload_files([("/../bad/path.txt", b"content")])
    assert len(responses) == 1
    assert responses[0].error == "invalid_path"


async def test_filesystem_partial_success_aupload(tmp_path: Path):
    """Test partial success in async batch upload."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    files = [
        ("/valid1.txt", b"Valid content 1"),
        ("/../invalid.txt", b"Invalid path"),  # Path traversal
        ("/valid2.txt", b"Valid content 2"),
    ]

    responses = await be.aupload_files(files)

    assert len(responses) == 3
    # First file should succeed
    assert responses[0].error is None
    assert (root / "valid1.txt").exists()

    # Second file should fail
    assert responses[1].error == "invalid_path"

    # Third file should still succeed (partial success)
    assert responses[2].error is None
    assert (root / "valid2.txt").exists()


async def test_filesystem_partial_success_adownload(tmp_path: Path):
    """Test partial success in async batch download."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create one valid file
    valid_file = root / "exists.txt"
    valid_content = b"I exist!"
    valid_file.write_bytes(valid_content)

    paths = ["/exists.txt", "/doesnotexist.txt", "/../invalid"]
    responses = await be.adownload_files(paths)

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


async def test_filesystem_aedit_replace_all(tmp_path: Path):
    """Test async edit with replace_all option."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create file with multiple occurrences
    test_file = root / "test.txt"
    test_file.write_text("foo bar foo baz")

    # Edit with replace_all=False when string appears multiple times should error
    res1 = await be.aedit("/test.txt", "foo", "qux", replace_all=False)
    assert res1.error is not None
    assert "appears 2 times" in res1.error

    # Edit with replace_all=True - should replace all occurrences
    res2 = await be.aedit("/test.txt", "foo", "qux", replace_all=True)
    assert res2.error is None
    assert res2.occurrences == 2
    content = await be.aread("/test.txt")
    assert "qux bar qux baz" in content

    # Now test replace_all=False with unique string (should succeed)
    res3 = await be.aedit("/test.txt", "bar", "xyz", replace_all=False)
    assert res3.error is None
    assert res3.occurrences == 1
    content2 = await be.aread("/test.txt")
    assert "qux xyz qux baz" in content2


async def test_filesystem_aread_with_offset_and_limit(tmp_path: Path):
    """Test async read with offset and limit."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create file with multiple lines
    test_file = root / "multi.txt"
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    test_file.write_text(lines)

    # Read with offset and limit
    content = await be.aread("/multi.txt", offset=2, limit=3)
    assert "Line 3" in content
    assert "Line 4" in content
    assert "Line 5" in content
    assert "Line 1" not in content
    assert "Line 6" not in content


async def test_filesystem_agrep_with_glob(tmp_path: Path):
    """Test async grep with glob filter."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create multiple files
    (root / "test.py").write_text("import os")
    (root / "test.txt").write_text("import nothing")
    (root / "main.py").write_text("import sys")

    # agrep_raw with glob filter
    matches = await be.agrep_raw("import", path="/", glob="*.py")
    assert isinstance(matches, list)
    py_files = [m["path"] for m in matches]
    assert any("test.py" in p for p in py_files)
    assert any("main.py" in p for p in py_files)
    assert not any("test.txt" in p for p in py_files)


async def test_filesystem_aglob_recursive(tmp_path: Path):
    """Test async glob with recursive patterns."""
    root = tmp_path
    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # Create files in nested directories
    files = {
        root / "src" / "main.py": "code",
        root / "src" / "utils" / "helper.py": "utils",
        root / "tests" / "test_main.py": "tests",
        root / "readme.txt": "docs",
    }

    for path, content in files.items():
        write_file(path, content)

    # Recursive glob for all .py files
    infos = await be.aglob_info("**/*.py", path="/")
    py_files = [i["path"] for i in infos]
    assert any("main.py" in p for p in py_files)
    assert any("helper.py" in p for p in py_files)
    assert any("test_main.py" in p for p in py_files)
    assert not any("readme.txt" in p for p in py_files)
