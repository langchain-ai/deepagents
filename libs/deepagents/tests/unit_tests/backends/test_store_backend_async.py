"""Async tests for StoreBackend."""

from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.store import StoreBackend


def make_runtime():
    return ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


async def test_store_backend_async_crud_and_search():
    """Test async CRUD and search operations."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # awrite new file
    msg = await be.awrite("/docs/readme.md", "hello store")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.path == "/docs/readme.md"

    # aread
    txt = await be.aread("/docs/readme.md")
    assert "hello store" in txt

    # aedit
    msg2 = await be.aedit("/docs/readme.md", "hello", "hi", replace_all=False)
    assert isinstance(msg2, EditResult) and msg2.error is None and msg2.occurrences == 1

    # als_info (path prefix filter)
    infos = await be.als_info("/docs/")
    assert any(i["path"] == "/docs/readme.md" for i in infos)

    # agrep_raw
    matches = await be.agrep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/docs/readme.md" for m in matches)

    # aglob_info
    g = await be.aglob_info("*.md", path="/")
    assert len(g) == 0

    g2 = await be.aglob_info("**/*.md", path="/")
    assert any(i["path"] == "/docs/readme.md" for i in g2)


async def test_store_backend_als_nested_directories():
    """Test async ls with nested directories."""
    rt = make_runtime()
    be = StoreBackend(rt)

    files = {
        "/src/main.py": "main code",
        "/src/utils/helper.py": "helper code",
        "/src/utils/common.py": "common code",
        "/docs/readme.md": "readme",
        "/docs/api/reference.md": "api reference",
        "/config.json": "config",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    root_listing = await be.als_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths
    assert "/docs/readme.md" not in root_paths
    assert "/docs/api/reference.md" not in root_paths

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


async def test_store_backend_als_trailing_slash():
    """Test async ls with trailing slash behavior."""
    rt = make_runtime()
    be = StoreBackend(rt)

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    listing_from_root = await be.als_info("/")
    assert len(listing_from_root) > 0

    listing1 = await be.als_info("/dir/")
    listing2 = await be.als_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


async def test_store_backend_async_errors():
    """Test async error handling."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # aedit missing file
    err = await be.aedit("/missing.txt", "a", "b")
    assert isinstance(err, EditResult) and err.error and "not found" in err.error

    # aread missing file
    content = await be.aread("/nonexistent.txt")
    assert "Error" in content or "not found" in content.lower()


async def test_store_backend_aedit_replace_all():
    """Test async edit with replace_all option."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # Write file with multiple occurrences
    res = await be.awrite("/test.txt", "foo bar foo baz")
    assert res.error is None

    # Edit with replace_all=False when string appears multiple times should error
    res2 = await be.aedit("/test.txt", "foo", "qux", replace_all=False)
    assert res2.error is not None
    assert "appears 2 times" in res2.error

    # Edit with replace_all=True - should replace all occurrences
    res3 = await be.aedit("/test.txt", "foo", "qux", replace_all=True)
    assert res3.error is None
    assert res3.occurrences == 2

    content = await be.aread("/test.txt")
    assert "qux bar qux baz" in content

    # Now test replace_all=False with unique string (should succeed)
    res4 = await be.aedit("/test.txt", "bar", "xyz", replace_all=False)
    assert res4.error is None
    assert res4.occurrences == 1

    content2 = await be.aread("/test.txt")
    assert "qux xyz qux baz" in content2


async def test_store_backend_aread_with_offset_and_limit():
    """Test async read with offset and limit."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # Write file with multiple lines
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    res = await be.awrite("/multi.txt", lines)
    assert res.error is None

    # Read with offset
    content_offset = await be.aread("/multi.txt", offset=2, limit=3)
    assert "Line 3" in content_offset
    assert "Line 4" in content_offset
    assert "Line 5" in content_offset
    assert "Line 1" not in content_offset
    assert "Line 6" not in content_offset


async def test_store_backend_agrep_with_glob():
    """Test async grep with glob filter."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # Write multiple files
    files = {
        "/test.py": "import os",
        "/test.txt": "import nothing",
        "/main.py": "import sys",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    # agrep_raw with glob filter for .py files only
    matches = await be.agrep_raw("import", path="/", glob="*.py")
    assert isinstance(matches, list)
    py_matches = [m["path"] for m in matches if m["path"].endswith(".py")]
    assert len(py_matches) >= 2  # Should match test.py and main.py


async def test_store_backend_aglob_patterns():
    """Test async glob with various patterns."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # Write files in nested directories
    files = {
        "/src/main.py": "code",
        "/src/utils/helper.py": "utils",
        "/tests/test_main.py": "tests",
        "/readme.md": "docs",
        "/docs/api.md": "api docs",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    # Recursive glob for all .py files
    infos = await be.aglob_info("**/*.py", path="/")
    py_files = [i["path"] for i in infos]
    assert "/src/main.py" in py_files
    assert "/src/utils/helper.py" in py_files
    assert "/tests/test_main.py" in py_files

    # Glob for markdown files
    md_infos = await be.aglob_info("**/*.md", path="/")
    md_files = [i["path"] for i in md_infos]
    assert "/readme.md" in md_files
    assert "/docs/api.md" in md_files


async def test_store_backend_aupload_adownload():
    """Test async upload and download operations."""
    rt = make_runtime()
    be = StoreBackend(rt)

    # Upload files
    files_to_upload = [
        ("/file1.bin", b"Binary content 1"),
        ("/file2.bin", b"Binary content 2"),
    ]

    upload_responses = await be.aupload_files(files_to_upload)
    assert len(upload_responses) == 2
    assert all(r.error is None for r in upload_responses)

    # Download files
    download_responses = await be.adownload_files(["/file1.bin", "/file2.bin"])
    assert len(download_responses) == 2
    # Note: StoreBackend stores as text, so binary content gets decoded
    assert download_responses[0].error is None
    assert download_responses[1].error is None


async def test_store_backend_agrep_invalid_regex():
    """Test async grep with invalid regex pattern."""
    rt = make_runtime()
    be = StoreBackend(rt)

    res = await be.awrite("/test.txt", "some content")
    assert res.error is None

    # Invalid regex should return error string
    result = await be.agrep_raw("[invalid", path="/")
    assert isinstance(result, str)
    assert "Invalid regex" in result or "error" in result.lower()


async def test_store_backend_intercept_large_tool_result_async():
    """Test that StoreBackend properly handles large tool result interception in async context."""
    from langchain_core.messages import ToolMessage

    from deepagents.middleware.filesystem import FilesystemMiddleware

    rt = make_runtime()
    middleware = FilesystemMiddleware(backend=lambda r: StoreBackend(r), tool_token_limit_before_evict=1000)

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = rt.store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    assert stored_content.value["content"] == [large_content]
