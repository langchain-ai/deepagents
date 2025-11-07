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


def test_store_backend_crud_and_search():
    rt = make_runtime()
    be = StoreBackend(rt)

    # write new file
    msg = be.write("/docs/readme.md", "hello store")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.path == "/docs/readme.md"

    # read
    txt = be.read("/docs/readme.md")
    assert "hello store" in txt

    # edit
    msg2 = be.edit("/docs/readme.md", "hello", "hi", replace_all=False)
    assert isinstance(msg2, EditResult) and msg2.error is None and msg2.occurrences == 1

    # ls_info (path prefix filter)
    infos = be.ls_info("/docs/")
    assert any(i["path"] == "/docs/readme.md" for i in infos)

    # grep_raw
    matches = be.grep_raw("hi", path="/")
    assert isinstance(matches, list) and any(m["path"] == "/docs/readme.md" for m in matches)

    # glob_info
    g = be.glob_info("*.md", path="/")
    assert len(g) == 0

    g2 = be.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/docs/readme.md" for i in g2)


def test_store_backend_ls_nested_directories():
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
        res = be.write(path, content)
        assert res.error is None

    root_listing = be.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/config.json" in root_paths
    assert "/src/" in root_paths
    assert "/docs/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/src/utils/helper.py" not in root_paths
    assert "/docs/readme.md" not in root_paths
    assert "/docs/api/reference.md" not in root_paths

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


def test_store_backend_ls_trailing_slash():
    rt = make_runtime()
    be = StoreBackend(rt)

    files = {
        "/file.txt": "content",
        "/dir/nested.txt": "nested",
    }

    for path, content in files.items():
        res = be.write(path, content)
        assert res.error is None

    listing_from_root = be.ls_info("/")
    assert len(listing_from_root) > 0

    listing1 = be.ls_info("/dir/")
    listing2 = be.ls_info("/dir")
    assert len(listing1) == len(listing2)
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


def test_store_backend_intercept_large_tool_result():
    """Test that StoreBackend properly handles large tool result interception."""
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
