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


def test_write_read_edit_ls_grep_glob_state_backend():
    rt = make_runtime()
    be = StateBackend(rt)

    # write
    res = be.write("/notes.txt", "hello world")
    assert isinstance(res, WriteResult)
    assert res.error is None and res.files_update is not None
    # apply state update
    rt.state["files"].update(res.files_update)

    # read
    content = be.read("/notes.txt")
    assert "hello world" in content

    # edit unique occurrence
    res2 = be.edit("/notes.txt", "hello", "hi", replace_all=False)
    assert isinstance(res2, EditResult)
    assert res2.error is None and res2.files_update is not None
    rt.state["files"].update(res2.files_update)

    content2 = be.read("/notes.txt")
    assert "hi world" in content2

    # ls_info should include the file
    listing = be.ls_info("/")
    assert any(fi["path"] == "/notes.txt" for fi in listing)

    # grep_raw
    matches = be.grep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/notes.txt" for m in matches)

    # invalid regex yields string error
    err = be.grep_raw("[", path="/")
    assert isinstance(err, str)

    # glob_info
    infos = be.glob_info("*.txt", path="/")
    assert any(i["path"] == "/notes.txt" for i in infos)


def test_state_backend_errors():
    rt = make_runtime()
    be = StateBackend(rt)

    # edit missing file
    err = be.edit("/missing.txt", "a", "b")
    assert isinstance(err, EditResult) and err.error and "not found" in err.error

    # write duplicate
    res = be.write("/dup.txt", "x")
    assert isinstance(res, WriteResult) and res.files_update is not None
    rt.state["files"].update(res.files_update)
    dup_err = be.write("/dup.txt", "y")
    assert isinstance(dup_err, WriteResult) and dup_err.error and "already exists" in dup_err.error


def test_state_backend_ls_nested_directories():
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
        res = be.write(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

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


def test_state_backend_ls_trailing_slash():
    rt = make_runtime()
    be = StateBackend(rt)

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None
        rt.state["files"].update(res.files_update)

    listing_with_slash = be.ls_info("/")
    assert len(listing_with_slash) == 2
    assert "/file.txt" in [fi["path"] for fi in listing_with_slash]
    assert "/dir/" in [fi["path"] for fi in listing_with_slash]

    listing_from_dir = be.ls_info("/dir/")
    assert len(listing_from_dir) == 1
    assert listing_from_dir[0]["path"] == "/dir/nested.txt"


def test_state_backend_intercept_large_tool_result():
    """Test that StateBackend properly handles large tool result interception."""
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
