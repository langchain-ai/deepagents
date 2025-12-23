"""Async tests for StateBackend."""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from deepagents.backends.protocol import EditResult, WriteResult
from deepagents.backends.state import StateBackend


def make_runtime(files=None):
    return ToolRuntime(
        state={
            "messages": [],
            "files": files or {},
        },
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


async def test_awrite_aread_aedit_als_agrep_aglob_state_backend():
    """Test async write, read, edit, ls, grep, and glob operations."""
    rt = make_runtime()
    be = StateBackend(rt)

    # awrite
    res = await be.awrite("/notes.txt", "hello world")
    assert isinstance(res, WriteResult)
    assert res.error is None and res.files_update is not None
    # apply state update
    rt.state["files"].update(res.files_update)

    # aread
    content = await be.aread("/notes.txt")
    assert "hello world" in content

    # aedit unique occurrence
    res2 = await be.aedit("/notes.txt", "hello", "hi", replace_all=False)
    assert isinstance(res2, EditResult)
    assert res2.error is None and res2.files_update is not None
    rt.state["files"].update(res2.files_update)

    content2 = await be.aread("/notes.txt")
    assert "hi world" in content2

    # als_info should include the file
    listing = await be.als_info("/")
    assert any(fi["path"] == "/notes.txt" for fi in listing)

    # agrep_raw
    matches = await be.agrep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/notes.txt" for m in matches)

    # invalid regex yields string error
    err = await be.agrep_raw("[", path="/")
    assert isinstance(err, str)

    # aglob_info
    infos = await be.aglob_info("*.txt", path="/")
    assert any(i["path"] == "/notes.txt" for i in infos)


async def test_state_backend_async_errors():
    """Test async error handling for StateBackend."""
    rt = make_runtime()
    be = StateBackend(rt)

    # aedit missing file
    err = await be.aedit("/missing.txt", "a", "b")
    assert isinstance(err, EditResult) and err.error and "not found" in err.error

    # awrite duplicate
    res = await be.awrite("/dup.txt", "x")
    assert isinstance(res, WriteResult) and res.files_update is not None
    rt.state["files"].update(res.files_update)
    dup_err = await be.awrite("/dup.txt", "y")
    assert isinstance(dup_err, WriteResult) and dup_err.error and "already exists" in dup_err.error


async def test_state_backend_als_nested_directories():
    """Test async ls with nested directories."""
    rt = make_runtime()
    be = StateBackend(rt)

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
        rt.state["files"].update(res.files_update)

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


async def test_state_backend_als_trailing_slash():
    """Test async ls with trailing slash behavior."""
    rt = make_runtime()
    be = StateBackend(rt)

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    listing_with_slash = await be.als_info("/")
    assert len(listing_with_slash) == 2
    assert "/file.txt" in [fi["path"] for fi in listing_with_slash]
    assert "/dir/" in [fi["path"] for fi in listing_with_slash]

    listing_from_dir = await be.als_info("/dir/")
    assert len(listing_from_dir) == 1
    assert listing_from_dir[0]["path"] == "/dir/nested.txt"


async def test_state_backend_aedit_replace_all():
    """Test async edit with replace_all option."""
    rt = make_runtime()
    be = StateBackend(rt)

    # Write file with multiple occurrences
    res = await be.awrite("/test.txt", "hello world hello universe")
    assert res.error is None
    rt.state["files"].update(res.files_update)

    # Edit with replace_all=False when string appears multiple times should error
    res2 = await be.aedit("/test.txt", "hello", "hi", replace_all=False)
    assert res2.error is not None
    assert "appears 2 times" in res2.error

    # Edit with replace_all=True - should replace all occurrences
    res3 = await be.aedit("/test.txt", "hello", "hi", replace_all=True)
    assert res3.error is None
    assert res3.occurrences == 2
    rt.state["files"].update(res3.files_update)

    content = await be.aread("/test.txt")
    assert "hi world hi universe" in content

    # Now test replace_all=False with unique string (should succeed)
    res4 = await be.aedit("/test.txt", "world", "galaxy", replace_all=False)
    assert res4.error is None
    assert res4.occurrences == 1
    rt.state["files"].update(res4.files_update)

    content2 = await be.aread("/test.txt")
    assert "hi galaxy hi universe" in content2


async def test_state_backend_aread_with_offset_and_limit():
    """Test async read with offset and limit parameters."""
    rt = make_runtime()
    be = StateBackend(rt)

    # Write file with multiple lines
    lines = "\n".join([f"Line {i}" for i in range(1, 11)])
    res = await be.awrite("/multi.txt", lines)
    assert res.error is None
    rt.state["files"].update(res.files_update)

    # Read with offset
    content_offset = await be.aread("/multi.txt", offset=2, limit=3)
    assert "Line 3" in content_offset
    assert "Line 4" in content_offset
    assert "Line 5" in content_offset
    assert "Line 1" not in content_offset
    assert "Line 6" not in content_offset


async def test_state_backend_agrep_with_pattern_and_glob():
    """Test async grep with pattern and glob filter."""
    rt = make_runtime()
    be = StateBackend(rt)

    # Write multiple files
    files = {
        "/test.py": "import os",
        "/test.txt": "import nothing",
        "/main.py": "import sys",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    # agrep_raw with glob filter for .py files only
    matches = await be.agrep_raw("import", path="/", glob="*.py")
    assert isinstance(matches, list)
    assert any(m["path"] == "/test.py" for m in matches)
    assert any(m["path"] == "/main.py" for m in matches)
    # test.txt should not be in matches even though it contains "import"
    assert not any(m["path"] == "/test.txt" for m in matches)


async def test_state_backend_aglob_recursive():
    """Test async glob with recursive patterns."""
    rt = make_runtime()
    be = StateBackend(rt)

    # Write files in nested directories
    files = {
        "/src/main.py": "code",
        "/src/utils/helper.py": "utils",
        "/tests/test_main.py": "tests",
        "/readme.txt": "docs",
    }

    for path, content in files.items():
        res = await be.awrite(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    # Recursive glob for all .py files
    infos = await be.aglob_info("**/*.py", path="/")
    py_files = [i["path"] for i in infos]
    assert "/src/main.py" in py_files
    assert "/src/utils/helper.py" in py_files
    assert "/tests/test_main.py" in py_files
    assert "/readme.txt" not in py_files


async def test_state_backend_intercept_large_tool_result_async():
    """Test that StateBackend properly handles large tool result interception in async context."""
    from langgraph.types import Command

    from deepagents.middleware.filesystem import FilesystemMiddleware

    rt = make_runtime()
    middleware = FilesystemMiddleware(backend=lambda r: StateBackend(r), tool_token_limit_before_evict=1000)

    large_content = "x" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, Command)
    assert "/large_tool_results/test_123" in result.update["files"]
    assert result.update["files"]["/large_tool_results/test_123"]["content"] == [large_content]
    assert "Tool result too large" in result.update["messages"][0].content
