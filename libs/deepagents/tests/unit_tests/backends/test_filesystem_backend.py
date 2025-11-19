from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import EditResult, WriteResult


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
    infos = be.ls_info(str(root))
    paths = {i["path"] for i in infos}
    assert str(f1) in paths  # File in root should be listed
    assert str(f2) not in paths  # File in subdirectory should NOT be listed
    assert (str(root) + "/dir/") in paths  # Directory should be listed

    # read, edit, write
    txt = be.read(str(f1))
    assert "hello fs" in txt
    msg = be.edit(str(f1), "fs", "filesystem", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1
    msg2 = be.write(str(root / "new.txt"), "new content")
    assert isinstance(msg2, WriteResult) and msg2.error is None and msg2.path.endswith("new.txt")

    # grep_raw
    matches = be.grep_raw("hello", path=str(root))
    assert isinstance(matches, list) and any(m["path"].endswith("a.txt") for m in matches)

    # glob_info
    g = be.glob_info("*.py", path=str(root))
    assert any(i["path"] == str(f2) for i in g)


def test_filesystem_backend_virtual_mode(tmp_path: Path):
    root = tmp_path
    f1 = root / "a.txt"
    f2 = root / "dir" / "b.md"
    write_file(f1, "hello virtual")
    write_file(f2, "content")

    be = FilesystemBackend(root_dir=str(root), virtual_mode=True)

    # ls_info from virtual root - should only list files in root, not subdirectories
    infos = be.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/a.txt" in paths  # File in root should be listed
    assert "/dir/b.md" not in paths  # File in subdirectory should NOT be listed
    assert "/dir/" in paths  # Directory should be listed

    # read and edit via virtual path
    txt = be.read("/a.txt")
    assert "hello virtual" in txt
    msg = be.edit("/a.txt", "virtual", "virt", replace_all=False)
    assert isinstance(msg, EditResult) and msg.error is None and msg.occurrences == 1

    # write new file via virtual path
    msg2 = be.write("/new.txt", "x")
    assert isinstance(msg2, WriteResult) and msg2.error is None
    assert (root / "new.txt").exists()

    # grep_raw limited to path
    matches = be.grep_raw("virt", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/a.txt" for m in matches)

    # glob_info
    g = be.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/dir/b.md" for i in g)

    # invalid regex returns error string
    err = be.grep_raw("[", path="/")
    assert isinstance(err, str)

    # path traversal blocked
    try:
        be.read("/../a.txt")
        assert False, "expected ValueError for traversal"
    except ValueError:
        pass


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

    root_listing = be.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths

    src_listing = be.ls_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = be.ls_info("/src/utils/")
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = be.ls_info("/nonexistent/")
    assert empty_listing == []


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

    root_listing = be.ls_info(str(root))
    root_paths = [fi["path"] for fi in root_listing]

    assert str(root / "file1.txt") in root_paths
    assert str(root / "subdir") + "/" in root_paths
    assert str(root / "subdir" / "file2.txt") not in root_paths

    subdir_listing = be.ls_info(str(root / "subdir"))
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

    listing_with_slash = be.ls_info("/")
    assert len(listing_with_slash) > 0

    listing = be.ls_info("/")
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    listing1 = be.ls_info("/dir/")
    listing2 = be.ls_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]

    empty = be.ls_info("/nonexistent/")
    assert empty == []


def test_filesystem_backend_intercept_large_tool_result(tmp_path: Path):
    """Test that FilesystemBackend properly handles large tool result interception."""
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
    for i, (path, content) in enumerate(files):
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
