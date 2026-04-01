"""Async tests for StoreBackend."""

from functools import partial

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends.protocol import EditResult, ReadResult, WriteResult
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware


async def test_store_backend_async_crud_and_search():
    """Test async CRUD and search operations."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # awrite new file
    msg = await be.awrite("/docs/readme.md", "hello store")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.path == "/docs/readme.md"

    # aread
    read_result = await be.aread("/docs/readme.md")
    assert isinstance(read_result, ReadResult) and read_result.file_data is not None
    assert "hello store" in read_result.file_data["content"]

    # aedit
    msg2 = await be.aedit("/docs/readme.md", "hello", "hi", replace_all=False)
    assert isinstance(msg2, EditResult) and msg2.error is None and msg2.occurrences == 1

    # als_info (path prefix filter)
    infos = (await be.als("/docs/")).entries
    assert infos is not None
    assert any(i["path"] == "/docs/readme.md" for i in infos)

    # agrep
    matches = (await be.agrep("hi", path="/")).matches
    assert matches is not None and any(m["path"] == "/docs/readme.md" for m in matches)

    # aglob
    g = (await be.aglob("*.md", path="/")).matches
    assert len(g) == 0

    g2 = (await be.aglob("**/*.md", path="/")).matches
    assert any(i["path"] == "/docs/readme.md" for i in g2)


async def test_store_backend_als_nested_directories():
    """Test async ls with nested directories."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

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

    root_listing = (await be.als("/")).entries
    assert root_listing is not None
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths
    assert "/docs/readme.md" not in root_paths
    assert "/docs/api/reference.md" not in root_paths

    src_listing = (await be.als("/src/")).entries
    assert src_listing is not None
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    utils_listing = (await be.als("/src/utils/")).entries
    assert utils_listing is not None
    utils_paths = [fi["path"] for fi in utils_listing]
    assert "/src/utils/helper.py" in utils_paths
    assert "/src/utils/common.py" in utils_paths
    assert len(utils_paths) == 2

    empty_listing = await be.als("/nonexistent/")
    assert empty_listing.entries == []


async def test_store_backend_als_trailing_slash():
    """Test async ls with trailing slash behavior."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    listing_from_root = (await be.als("/")).entries
    assert listing_from_root is not None
    assert len(listing_from_root) > 0

    listing1 = (await be.als("/dir/")).entries
    listing2 = (await be.als("/dir")).entries
    assert listing1 is not None
    assert listing2 is not None
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


async def test_store_backend_async_errors():
    """Test async error handling."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # aedit missing file
    err = await be.aedit("/missing.txt", "a", "b")
    assert isinstance(err, EditResult) and err.error and "not found" in err.error

    # aread missing file
    read_result = await be.aread("/nonexistent.txt")
    assert isinstance(read_result, ReadResult) and read_result.error is not None


async def test_store_backend_aedit_replace_all():
    """Test async edit with replace_all option."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

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

    read_result = await be.aread("/test.txt")
    assert read_result.file_data is not None
    assert "qux bar qux baz" in read_result.file_data["content"]

    # Now test replace_all=False with unique string (should succeed)
    res4 = await be.aedit("/test.txt", "bar", "xyz", replace_all=False)
    assert res4.error is None
    assert res4.occurrences == 1

    read_result2 = await be.aread("/test.txt")
    assert read_result2.file_data is not None
    assert "qux xyz qux baz" in read_result2.file_data["content"]


async def test_store_backend_aread_with_offset_and_limit():
    """Test async read with offset and limit."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # Write file with multiple lines
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    res = await be.awrite("/multi.txt", lines)
    assert res.error is None

    # Read with offset
    read_result = await be.aread("/multi.txt", offset=2, limit=3)
    assert read_result.file_data is not None
    content_offset = read_result.file_data["content"]
    assert "Line 3" in content_offset
    assert "Line 4" in content_offset
    assert "Line 5" in content_offset
    assert "Line 1" not in content_offset
    assert "Line 6" not in content_offset


async def test_store_backend_agrep_with_glob():
    """Test async grep with glob filter."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    # Write multiple files
    files = {
        "/test.py": "import os",
        "/test.txt": "import nothing",
        "/main.py": "import sys",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None

    # agrep with glob filter for .py files only
    matches = (await be.agrep("import", path="/", glob="*.py")).matches
    assert matches is not None
    py_matches = [m["path"] for m in matches if m["path"].endswith(".py")]
    assert len(py_matches) >= 2  # Should match test.py and main.py


async def test_store_backend_aglob_patterns():
    """Test async glob with various patterns."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

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
    infos = (await be.aglob("**/*.py", path="/")).matches
    py_files = [i["path"] for i in infos]
    assert "/src/main.py" in py_files
    assert "/src/utils/helper.py" in py_files
    assert "/tests/test_main.py" in py_files

    # Glob for markdown files
    md_infos = (await be.aglob("**/*.md", path="/")).matches
    md_files = [i["path"] for i in md_infos]
    assert "/readme.md" in md_files
    assert "/docs/api.md" in md_files


async def test_store_backend_aupload_adownload():
    """Test async upload and download operations."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

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
    """Test async grep with special characters (literal search, not regex)."""
    mem_store = InMemoryStore()
    be = StoreBackend(store=mem_store, namespace=lambda _rt: ("filesystem",))

    res = await be.awrite("/test.txt", "some content")
    assert res.error is None

    # Special characters are treated literally, not regex
    result = await be.agrep("[invalid", path="/")
    assert result.matches is not None  # Returns empty list, not error


@pytest.mark.parametrize("file_format", ["v1", "v2"])
async def test_store_backend_intercept_large_tool_result_async(file_format):
    """Test that StoreBackend properly handles large tool result interception in async context."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=partial(StoreBackend, store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format),
        tool_token_limit_before_evict=1000,
    )

    large_content = "y" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_456")
    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=mem_store,
        stream_writer=lambda _: None,
        config={},
    )
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_456" in result.content

    stored_content = mem_store.get(("filesystem",), "/large_tool_results/test_456")
    assert stored_content is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_content.value["content"] == expected


@pytest.mark.parametrize("file_format", ["v1", "v2"])
async def test_store_backend_aintercept_large_tool_result_async(file_format):
    """Test async intercept path uses async store methods (fixes InvalidStateError with BatchedStore)."""
    mem_store = InMemoryStore()
    middleware = FilesystemMiddleware(
        backend=partial(StoreBackend, store=mem_store, namespace=lambda _rt: ("filesystem",), file_format=file_format),
        tool_token_limit_before_evict=1000,
    )

    large_content = "z" * 5000
    artifact_payload = {"kind": "structured", "value": {"key": "v"}}
    tool_message = ToolMessage(
        content=large_content,
        tool_call_id="test_async_789",
        name="example_tool",
        id="tool_msg_async_1",
        artifact=artifact_payload,
        status="error",
        additional_kwargs={"trace": "abc"},
        response_metadata={"provider": "mock"},
    )

    rt = ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=mem_store,
        stream_writer=lambda _: None,
        config={},
    )

    # Use the async intercept path (what awrap_tool_call uses)
    result = await middleware._aintercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_async_789" in result.content
    assert result.name == "example_tool"
    assert result.id == "tool_msg_async_1"
    assert result.artifact == artifact_payload
    assert result.status == "error"
    assert result.additional_kwargs == {"trace": "abc"}
    assert result.response_metadata == {"provider": "mock"}

    # Verify content was stored via async path
    stored_content = await mem_store.aget(("filesystem",), "/large_tool_results/test_async_789")
    assert stored_content is not None
    expected = [large_content] if file_format == "v1" else large_content
    assert stored_content.value["content"] == expected
