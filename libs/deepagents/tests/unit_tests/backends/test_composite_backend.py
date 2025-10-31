from pathlib import Path

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import WriteResult
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore


def make_runtime(tid: str = "tc"):
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tid,
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def build_composite_state_backend(runtime: ToolRuntime, *, routes):
    built_routes = {}
    for prefix, backend_or_factory in routes.items():
        if callable(backend_or_factory):
            built_routes[prefix] = backend_or_factory(runtime)
        else:
            built_routes[prefix] = backend_or_factory
    default_state = StateBackend(runtime)
    return CompositeBackend(default=default_state, routes=built_routes)


def test_composite_state_backend_routes_and_search(tmp_path: Path):
    rt = make_runtime("t3")
    # route /memories/ to store
    be = build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})

    # write to default (state)
    res = be.write("/file.txt", "alpha")
    assert isinstance(res, WriteResult) and res.files_update is not None

    # write to routed (store)
    msg = be.write("/memories/readme.md", "beta")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.files_update is None

    # ls_info at root returns both
    infos = be.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/file.txt" in paths and "/memories/" in paths

    # grep across both
    matches = be.grep_raw("alpha", path="/")
    assert any(m["path"] == "/file.txt" for m in matches)
    matches2 = be.grep_raw("beta", path="/")
    assert any(m["path"] == "/memories/readme.md" for m in matches2)

    # glob across both
    g = be.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/memories/readme.md" for i in g)


def test_composite_backend_filesystem_plus_store(tmp_path: Path):
    # default filesystem, route to store under /memories/
    root = tmp_path
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    rt = make_runtime("t4")
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # put files in both
    r1 = comp.write("/hello.txt", "hello")
    assert isinstance(r1, WriteResult) and r1.error is None and r1.files_update is None
    r2 = comp.write("/memories/notes.md", "note")
    assert isinstance(r2, WriteResult) and r2.error is None and r2.files_update is None

    # ls_info path routing
    infos_root = comp.ls_info("/")
    assert any(i["path"] == "/hello.txt" for i in infos_root)
    infos_mem = comp.ls_info("/memories/")
    assert any(i["path"] == "/memories/notes.md" for i in infos_mem)

    # grep_raw merges
    gm = comp.grep_raw("hello", path="/")
    assert any(m["path"] == "/hello.txt" for m in gm)
    gm2 = comp.grep_raw("note", path="/")
    assert any(m["path"] == "/memories/notes.md" for m in gm2)

    # glob_info
    gl = comp.glob_info("*.md", path="/")
    assert any(i["path"] == "/memories/notes.md" for i in gl)


def test_composite_backend_store_to_store():
    """Test composite with default store and routed store (two different stores)."""
    rt = make_runtime("t5")

    # Create two separate store backends (simulating different namespaces/stores)
    default_store = StoreBackend(rt)
    memories_store = StoreBackend(rt)

    comp = CompositeBackend(default=default_store, routes={"/memories/": memories_store})

    # Write to default store
    res1 = comp.write("/notes.txt", "default store content")
    assert isinstance(res1, WriteResult) and res1.error is None and res1.path == "/notes.txt"

    # Write to routed store
    res2 = comp.write("/memories/important.txt", "routed store content")
    assert isinstance(res2, WriteResult) and res2.error is None and res2.path == "/important.txt"

    # Read from both
    content1 = comp.read("/notes.txt")
    assert "default store content" in content1

    content2 = comp.read("/memories/important.txt")
    assert "routed store content" in content2

    # ls_info at root should show both
    infos = comp.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/notes.txt" in paths
    assert "/memories/" in paths

    # grep across both stores
    matches = comp.grep_raw("default", path="/")
    assert any(m["path"] == "/notes.txt" for m in matches)

    matches2 = comp.grep_raw("routed", path="/")
    assert any(m["path"] == "/memories/important.txt" for m in matches2)


def test_composite_backend_multiple_routes():
    """Test composite with state default and multiple store routes."""
    rt = make_runtime("t6")

    # State backend as default, multiple stores for different routes
    comp = build_composite_state_backend(
        rt,
        routes={
            "/memories/": (lambda r: StoreBackend(r)),
            "/archive/": (lambda r: StoreBackend(r)),
            "/cache/": (lambda r: StoreBackend(r)),
        },
    )

    # Write to state (default)
    res_state = comp.write("/temp.txt", "ephemeral data")
    assert res_state.files_update is not None  # State backend returns files_update
    assert res_state.path == "/temp.txt"

    # Write to /memories/ route
    res_mem = comp.write("/memories/important.md", "long-term memory")
    assert res_mem.files_update is None  # Store backend doesn't return files_update
    assert res_mem.path == "/important.md"

    # Write to /archive/ route
    res_arch = comp.write("/archive/old.log", "archived log")
    assert res_arch.files_update is None
    assert res_arch.path == "/old.log"

    # Write to /cache/ route
    res_cache = comp.write("/cache/session.json", "cached session")
    assert res_cache.files_update is None
    assert res_cache.path == "/session.json"

    # ls_info at root should aggregate all
    infos = comp.ls_info("/")
    paths = {i["path"] for i in infos}
    assert "/temp.txt" in paths
    assert "/memories/" in paths
    assert "/archive/" in paths
    assert "/cache/" in paths

    # ls_info at specific route
    mem_infos = comp.ls_info("/memories/")
    mem_paths = {i["path"] for i in mem_infos}
    assert "/memories/important.md" in mem_paths
    assert "/temp.txt" not in mem_paths
    assert "/archive/old.log" not in mem_paths

    # grep across all backends
    all_matches = comp.grep_raw(".", path="/")  # Match any character
    paths_with_content = {m["path"] for m in all_matches}
    assert "/temp.txt" in paths_with_content
    assert "/memories/important.md" in paths_with_content
    assert "/archive/old.log" in paths_with_content
    assert "/cache/session.json" in paths_with_content

    # glob across all backends
    glob_results = comp.glob_info("**/*.md", path="/")
    assert any(i["path"] == "/memories/important.md" for i in glob_results)

    # Edit in routed backend
    edit_res = comp.edit("/memories/important.md", "long-term", "persistent", replace_all=False)
    assert edit_res.error is None
    assert edit_res.occurrences == 1

    updated_content = comp.read("/memories/important.md")
    assert "persistent memory" in updated_content


def test_composite_backend_ls_nested_directories(tmp_path: Path):
    rt = make_runtime("t7")
    root = tmp_path

    files = {
        root / "local.txt": "local file",
        root / "src" / "main.py": "code",
        root / "src" / "utils" / "helper.py": "utils",
    }

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    comp.write("/memories/note1.txt", "note 1")
    comp.write("/memories/deep/note2.txt", "note 2")
    comp.write("/memories/deep/nested/note3.txt", "note 3")

    root_listing = comp.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/local.txt" in root_paths
    assert "/src/" in root_paths
    assert "/memories/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/memories/note1.txt" not in root_paths

    src_listing = comp.ls_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    mem_listing = comp.ls_info("/memories/")
    mem_paths = [fi["path"] for fi in mem_listing]
    assert "/memories/note1.txt" in mem_paths
    assert "/memories/deep/" in mem_paths
    assert "/memories/deep/note2.txt" not in mem_paths

    deep_listing = comp.ls_info("/memories/deep/")
    deep_paths = [fi["path"] for fi in deep_listing]
    assert "/memories/deep/note2.txt" in deep_paths
    assert "/memories/deep/nested/" in deep_paths
    assert "/memories/deep/nested/note3.txt" not in deep_paths


def test_composite_backend_ls_multiple_routes_nested():
    rt = make_runtime("t8")
    comp = build_composite_state_backend(
        rt,
        routes={
            "/memories/": (lambda r: StoreBackend(r)),
            "/archive/": (lambda r: StoreBackend(r)),
        },
    )

    state_files = {
        "/temp.txt": "temp",
        "/work/file1.txt": "work file 1",
        "/work/projects/proj1.txt": "project 1",
    }

    for path, content in state_files.items():
        res = comp.write(path, content)
        if res.files_update:
            rt.state["files"].update(res.files_update)

    memory_files = {
        "/memories/important.txt": "important",
        "/memories/diary/entry1.txt": "diary entry",
    }

    for path, content in memory_files.items():
        comp.write(path, content)

    archive_files = {
        "/archive/old.txt": "old",
        "/archive/2023/log.txt": "2023 log",
    }

    for path, content in archive_files.items():
        comp.write(path, content)

    root_listing = comp.ls_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/temp.txt" in root_paths
    assert "/work/" in root_paths
    assert "/memories/" in root_paths
    assert "/archive/" in root_paths
    assert "/work/file1.txt" not in root_paths
    assert "/memories/important.txt" not in root_paths

    work_listing = comp.ls_info("/work/")
    work_paths = [fi["path"] for fi in work_listing]
    assert "/work/file1.txt" in work_paths
    assert "/work/projects/" in work_paths
    assert "/work/projects/proj1.txt" not in work_paths

    mem_listing = comp.ls_info("/memories/")
    mem_paths = [fi["path"] for fi in mem_listing]
    assert "/memories/important.txt" in mem_paths
    assert "/memories/diary/" in mem_paths
    assert "/memories/diary/entry1.txt" not in mem_paths

    arch_listing = comp.ls_info("/archive/")
    arch_paths = [fi["path"] for fi in arch_listing]
    assert "/archive/old.txt" in arch_paths
    assert "/archive/2023/" in arch_paths
    assert "/archive/2023/log.txt" not in arch_paths


def test_composite_backend_ls_trailing_slash(tmp_path: Path):
    rt = make_runtime("t9")
    root = tmp_path

    (root / "file.txt").write_text("content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/store/": store})

    comp.write("/store/item.txt", "store content")

    listing = comp.ls_info("/")
    paths = [fi["path"] for fi in listing]
    assert paths == sorted(paths)

    empty_listing = comp.ls_info("/store/nonexistent/")
    assert empty_listing == []

    empty_listing2 = comp.ls_info("/nonexistent/")
    assert empty_listing2 == []

    listing1 = comp.ls_info("/store/")
    listing2 = comp.ls_info("/store")
    assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


def test_composite_backend_intercept_large_tool_result():
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    rt = make_runtime("t10")

    middleware = FilesystemMiddleware(
        backend=lambda r: build_composite_state_backend(r, routes={"/memories/": (lambda x: StoreBackend(x))}), tool_token_limit_before_evict=1000
    )
    large_content = "z" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_789")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, Command)
    assert "/large_tool_results/test_789" in result.update["files"]
    assert result.update["files"]["/large_tool_results/test_789"]["content"] == [large_content]
    assert "Tool result too large" in result.update["messages"][0].content


def test_composite_backend_intercept_large_tool_result_routed_to_store():
    """Test that large tool results can be routed to a specific backend like StoreBackend."""
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain_core.messages import ToolMessage

    rt = make_runtime("t11")

    middleware = FilesystemMiddleware(
        backend=lambda r: build_composite_state_backend(r, routes={"/large_tool_results/": (lambda x: StoreBackend(x))}),
        tool_token_limit_before_evict=1000,
    )

    large_content = "w" * 5000
    tool_message = ToolMessage(content=large_content, tool_call_id="test_routed_123")
    result = middleware._intercept_large_tool_result(tool_message, rt)

    assert isinstance(result, ToolMessage)
    assert "Tool result too large" in result.content
    assert "/large_tool_results/test_routed_123" in result.content

    stored_item = rt.store.get(("filesystem",), "/test_routed_123")
    assert stored_item is not None
    assert stored_item.value["content"] == [large_content]
