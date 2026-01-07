from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend


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


def test_composite_backend_grep_path_isolation():
    """Test that grep with path=/tools doesn't return results from /memories."""
    rt = make_runtime("t7")

    # Use StateBackend as default, StoreBackend for /memories/
    state = StateBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=state, routes={"/memories/": store})

    # Write to state backend (default) in /tools directory
    comp.write("/tools/hammer.txt", "tool for nailing")
    comp.write("/tools/saw.txt", "tool for cutting")

    # Write to memories route with content that would match our grep
    comp.write("/memories/workshop.txt", "tool shed location")
    comp.write("/memories/notes.txt", "remember to buy tools")

    # Grep for "tool" in /tools directory - should NOT return /memories results
    matches = comp.grep_raw("tool", path="/tools")
    match_paths = [m["path"] for m in matches] if isinstance(matches, list) else []

    # Should find results in /tools
    assert any("/tools/hammer.txt" in p for p in match_paths)
    assert any("/tools/saw.txt" in p for p in match_paths)

    # Should NOT find results in /memories (this is the bug)
    assert not any("/memories/" in p for p in match_paths), f"grep path=/tools should not return /memories results, but got: {match_paths}"


def test_composite_backend_ls_nested_directories(tmp_path: Path):
    rt = make_runtime("t8")
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
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from deepagents.middleware.filesystem import FilesystemMiddleware

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
    from langchain_core.messages import ToolMessage

    from deepagents.middleware.filesystem import FilesystemMiddleware

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


# Mock sandbox backend for testing execute functionality
class MockSandboxBackend(SandboxBackendProtocol, StateBackend):
    """Mock sandbox backend that implements SandboxBackendProtocol."""

    def execute(self, command: str, *, timeout: int = 30 * 60) -> ExecuteResponse:
        """Mock execute that returns the command as output."""
        return ExecuteResponse(
            output=f"Executed: {command}",
            exit_code=0,
            truncated=False,
        )

    @property
    def id(self) -> str:
        return "mock_sandbox_backend"


def test_composite_backend_execute_with_sandbox_default():
    """Test that CompositeBackend.execute() delegates to sandbox default backend."""
    rt = make_runtime("t_exec1")
    sandbox = MockSandboxBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=sandbox, routes={"/memories/": store})

    # Execute should work since default backend supports it
    result = comp.execute("ls -la")
    assert isinstance(result, ExecuteResponse)
    assert result.output == "Executed: ls -la"
    assert result.exit_code == 0
    assert result.truncated is False


def test_composite_backend_execute_without_sandbox_default():
    """Test that CompositeBackend.execute() fails when default doesn't support execution."""
    rt = make_runtime("t_exec2")
    state_backend = StateBackend(rt)  # StateBackend doesn't implement SandboxBackendProtocol
    store = StoreBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/memories/": store})

    # Execute should raise NotImplementedError since default backend doesn't support it
    with pytest.raises(NotImplementedError, match="doesn't support command execution"):
        comp.execute("ls -la")


def test_composite_backend_supports_execution_check():
    """Test the isinstance check works correctly for CompositeBackend."""
    rt = make_runtime("t_exec3")

    # CompositeBackend with sandbox default should pass isinstance check
    sandbox = MockSandboxBackend(rt)
    comp_with_sandbox = CompositeBackend(default=sandbox, routes={})
    # Note: CompositeBackend itself has execute() method, so isinstance will pass
    # but the actual support depends on the default backend
    assert hasattr(comp_with_sandbox, "execute")

    # CompositeBackend with non-sandbox default should still have execute() method
    # but will raise NotImplementedError when called
    state = StateBackend(rt)
    comp_without_sandbox = CompositeBackend(default=state, routes={})
    assert hasattr(comp_without_sandbox, "execute")


def test_composite_backend_execute_with_routed_backends():
    """Test that execution doesn't interfere with file routing."""
    rt = make_runtime("t_exec4")
    sandbox = MockSandboxBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=sandbox, routes={"/memories/": store})

    # Write files to both backends
    comp.write("/local.txt", "local content")
    comp.write("/memories/persistent.txt", "persistent content")

    # Execute should still work
    result = comp.execute("echo test")
    assert result.output == "Executed: echo test"

    # File operations should still work
    assert "local content" in comp.read("/local.txt")
    assert "persistent content" in comp.read("/memories/persistent.txt")


def test_composite_upload_routing(tmp_path: Path):
    """Test upload_files routing to different backends."""
    rt = make_runtime("t_upload1")
    root = tmp_path

    # Create composite with filesystem default and store route
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Upload files to default path (filesystem)
    default_files = [
        ("/file1.bin", b"Default content 1"),
        ("/file2.bin", b"Default content 2"),
    ]
    responses = comp.upload_files(default_files)
    assert len(responses) == 2
    assert all(r.error is None for r in responses)
    assert (root / "file1.bin").exists()
    assert (root / "file2.bin").read_bytes() == b"Default content 2"

    # Upload files to routed path (store)
    routed_files = [
        ("/memories/note1.bin", b"Memory content 1"),
        ("/memories/note2.bin", b"Memory content 2"),
    ]
    responses = comp.upload_files(routed_files)
    assert len(responses) == 2
    assert all(r.error is None for r in responses)

    # Verify files are accessible in store
    content1 = comp.read("/memories/note1.bin")
    assert "Memory content 1" in content1


def test_composite_download_routing(tmp_path: Path):
    """Test download_files routing to different backends."""
    rt = make_runtime("t_download1")
    root = tmp_path

    # Create composite with filesystem default and store route
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Pre-populate filesystem backend
    (root / "local.bin").write_bytes(b"Local binary data")

    # Pre-populate store backend
    comp.write("/memories/stored.txt", "Stored text data")

    # Download from default path (filesystem)
    responses = comp.download_files(["/local.bin"])
    assert len(responses) == 1
    assert responses[0].path == "/local.bin"
    assert responses[0].content == b"Local binary data"
    assert responses[0].error is None

    # Download from routed path (store) - Note: store backend doesn't implement download yet
    # So this test focuses on routing logic
    paths_to_download = ["/local.bin"]
    responses = comp.download_files(paths_to_download)
    assert len(responses) == 1
    assert responses[0].path == "/local.bin"


def test_composite_upload_download_roundtrip(tmp_path: Path):
    """Test upload and download roundtrip through composite backend."""
    rt = make_runtime("t_roundtrip1")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Upload binary content
    test_content = bytes(range(128))  # Binary data
    upload_responses = comp.upload_files([("/test.bin", test_content)])
    assert upload_responses[0].error is None

    # Download it back
    download_responses = comp.download_files(["/test.bin"])
    assert download_responses[0].error is None
    assert download_responses[0].content == test_content


def test_composite_partial_success_upload(tmp_path: Path):
    """Test partial success in batch upload with mixed valid/invalid paths."""
    rt = make_runtime("t_partial_upload")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    files = [
        ("/valid1.bin", b"Valid 1"),
        ("/../invalid.bin", b"Invalid path"),  # Path traversal
        ("/valid2.bin", b"Valid 2"),
    ]

    responses = comp.upload_files(files)

    assert len(responses) == 3
    # First should succeed
    assert responses[0].error is None
    assert (root / "valid1.bin").exists()

    # Second should fail
    assert responses[1].error == "invalid_path"

    # Third should still succeed (partial success)
    assert responses[2].error is None
    assert (root / "valid2.bin").exists()


def test_composite_partial_success_download(tmp_path: Path):
    """Test partial success in batch download with mixed valid/invalid paths."""
    rt = make_runtime("t_partial_download")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Create one valid file
    (root / "exists.bin").write_bytes(b"I exist!")

    paths = ["/exists.bin", "/doesnotexist.bin", "/../invalid"]
    responses = comp.download_files(paths)

    assert len(responses) == 3

    # First should succeed
    assert responses[0].error is None
    assert responses[0].content == b"I exist!"

    # Second should fail with file_not_found
    assert responses[1].error == "file_not_found"
    assert responses[1].content is None

    # Third should fail with invalid_path
    assert responses[2].error == "invalid_path"
    assert responses[2].content is None


def test_composite_upload_download_multiple_routes(tmp_path: Path):
    """Test upload/download with multiple routed backends."""
    rt = make_runtime("t_multi_route")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store1 = StoreBackend(rt)
    store2 = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store1, "/archive/": store2})

    # Upload to different backends
    files = [
        ("/default.bin", b"Default backend"),
        ("/memories/mem.bin", b"Memory backend"),
        ("/archive/arch.bin", b"Archive backend"),
    ]

    responses = comp.upload_files(files)
    assert len(responses) == 3
    assert all(r.error is None for r in responses)

    # Verify routing worked (filesystem file should exist)
    assert (root / "default.bin").exists()
    assert (root / "default.bin").read_bytes() == b"Default backend"


def test_composite_download_preserves_original_paths(tmp_path: Path):
    """Test that download responses preserve original composite paths."""
    rt = make_runtime("t_path_preserve")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Create files
    (root / "subdir").mkdir()
    (root / "subdir" / "file.bin").write_bytes(b"Nested file")

    # Download with composite path
    responses = comp.download_files(["/subdir/file.bin"])

    # Response should have the original composite path, not stripped
    assert responses[0].path == "/subdir/file.bin"
    assert responses[0].content == b"Nested file"


def test_composite_grep_targeting_specific_route(tmp_path: Path) -> None:
    """Test grep with path targeting a specific routed backend."""
    rt = make_runtime("t_grep1")
    root = tmp_path

    # Setup filesystem backend with some files
    (root / "default.txt").write_text("default backend content")
    (root / "default2.txt").write_text("more default stuff")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Write to memories route
    comp.write("/memories/note1.txt", "memory content alpha")
    comp.write("/memories/note2.txt", "memory content beta")

    # Grep with path="/memories/" should only search memories backend
    matches = comp.grep_raw("memory", path="/memories/")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find matches in /memories/
    assert any("/memories/note1.txt" in p for p in match_paths)
    assert any("/memories/note2.txt" in p for p in match_paths)

    # Should NOT find matches in default backend
    assert not any("/default" in p for p in match_paths)


def test_composite_grep_with_glob_filter(tmp_path: Path) -> None:
    """Test grep with glob parameter to filter files."""
    rt = make_runtime("t_grep2")
    root = tmp_path

    # Create files with different extensions
    (root / "script.py").write_text("python code here")
    (root / "config.json").write_text("json config here")
    (root / "readme.md").write_text("markdown docs here")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Add some files to memories route
    comp.write("/memories/notes.py", "python notes here")
    comp.write("/memories/data.json", "json data here")

    # Grep with glob="*.py" should only search Python files
    matches = comp.grep_raw("here", path="/", glob="*.py")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find .py files
    assert any("/script.py" in p for p in match_paths)
    assert any("/memories/notes.py" in p for p in match_paths)

    # Should NOT find non-.py files
    assert not any(".json" in p for p in match_paths)
    assert not any(".md" in p for p in match_paths)


def test_composite_grep_with_glob_in_specific_route(tmp_path: Path) -> None:
    """Test grep with glob parameter targeting a specific route."""
    rt = make_runtime("t_grep3")
    root = tmp_path

    (root / "local.md").write_text("local markdown")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Add files to memories
    comp.write("/memories/important.md", "important notes")
    comp.write("/memories/data.txt", "text data")

    # Grep memories with glob="*.md"
    matches = comp.grep_raw("notes", path="/memories/", glob="*.md")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find .md file in memories
    assert any("/memories/important.md" in p for p in match_paths)

    # Should NOT find .txt files or default backend files
    assert not any("/memories/data.txt" in p for p in match_paths)
    assert not any("/local.md" in p for p in match_paths)


def test_composite_grep_with_path_none(tmp_path: Path) -> None:
    """Test grep with path=None behaves like path='/'."""
    rt = make_runtime("t_grep4")
    root = tmp_path

    (root / "file1.txt").write_text("searchable content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    comp.write("/memories/file2.txt", "searchable memory")

    # Grep with path=None
    matches_none = comp.grep_raw("searchable", path=None)
    assert isinstance(matches_none, list)

    # Grep with path="/"
    matches_root = comp.grep_raw("searchable", path="/")
    assert isinstance(matches_root, list)

    # Both should return same results
    paths_none = sorted([m["path"] for m in matches_none])
    paths_root = sorted([m["path"] for m in matches_root])

    assert paths_none == paths_root
    assert len(paths_none) == 2


def test_composite_grep_invalid_regex(tmp_path: Path) -> None:
    """Test grep with invalid regex pattern returns error string."""
    rt = make_runtime("t_grep5")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Invalid regex patterns
    result = comp.grep_raw("[invalid(", path="/")
    assert isinstance(result, str)
    assert "Invalid regex" in result or "error" in result.lower()


def test_composite_grep_nested_path_in_route(tmp_path: Path) -> None:
    """Test grep with nested path within a routed backend."""
    rt = make_runtime("t_grep6")
    root = tmp_path

    (root / "local.txt").write_text("local content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Create nested structure in memories
    comp.write("/memories/docs/readme.md", "documentation here")
    comp.write("/memories/docs/guide.md", "guide here")
    comp.write("/memories/notes.txt", "notes here")

    # Grep with nested path
    matches = comp.grep_raw("here", path="/memories/docs/")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find files in /memories/docs/
    assert any("/memories/docs/readme.md" in p for p in match_paths)
    assert any("/memories/docs/guide.md" in p for p in match_paths)

    # Should NOT find files outside /memories/docs/
    assert not any("/memories/notes.txt" in p for p in match_paths)
    assert not any("/local.txt" in p for p in match_paths)


def test_composite_grep_empty_results(tmp_path: Path) -> None:
    """Test grep that matches nothing returns empty list."""
    rt = make_runtime("t_grep7")
    root = tmp_path

    (root / "file.txt").write_text("some content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    comp.write("/memories/note.txt", "memory content")

    # Search for pattern that doesn't exist
    matches = comp.grep_raw("nonexistent_pattern_xyz", path="/")
    assert isinstance(matches, list)
    assert len(matches) == 0


def test_composite_grep_route_prefix_restoration(tmp_path: Path) -> None:
    """Test that grep correctly restores route prefixes in results."""
    rt = make_runtime("t_grep8")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Write files to memories
    comp.write("/memories/alpha.txt", "test content alpha")
    comp.write("/memories/beta.txt", "test content beta")

    # Grep in memories route
    matches = comp.grep_raw("test", path="/memories/")
    assert isinstance(matches, list)
    assert len(matches) > 0

    # All paths should start with /memories/
    for match in matches:
        assert match["path"].startswith("/memories/")
        assert not match["path"].startswith("/memories//")  # No double slashes

    # Grep across all backends (path="/")
    matches_all = comp.grep_raw("test", path="/")
    assert isinstance(matches_all, list)

    # Filter matches from memories
    memory_matches = [m for m in matches_all if "/memories/" in m["path"]]
    for match in memory_matches:
        assert match["path"].startswith("/memories/")


def test_composite_grep_multiple_matches_per_file(tmp_path: Path) -> None:
    """Test grep returns multiple matches from same file."""
    rt = make_runtime("t_grep9")
    root = tmp_path

    # File with multiple matching lines
    (root / "multi.txt").write_text("line1 pattern\nline2 pattern\nline3 other")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    matches = comp.grep_raw("pattern", path="/")
    assert isinstance(matches, list)

    # Should have 2 matches from the same file
    multi_matches = [m for m in matches if "multi.txt" in m["path"]]
    assert len(multi_matches) == 2

    # Verify line numbers are correct
    line_numbers = sorted([m["line"] for m in multi_matches])
    assert line_numbers == [1, 2]


@pytest.mark.xfail(
    reason="StoreBackend instances share the same underlying store when using the same runtime, "
    "causing files written to one route to appear in all routes that use the same backend instance. "
    "This violates the expected isolation between routes."
)
def test_composite_grep_multiple_routes_aggregation(tmp_path: Path) -> None:
    """Test grep aggregates results from multiple routed backends with expected isolation.

    This test represents the intuitive expected behavior: files written to /memories/
    should only appear in /memories/, and files written to /archive/ should only appear
    in /archive/.
    """
    rt = make_runtime("t_grep10")
    root = tmp_path

    (root / "default.txt").write_text("default findme")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store1 = StoreBackend(rt)
    store2 = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store1, "/archive/": store2})

    # Write to each route
    comp.write("/memories/mem.txt", "memory findme")
    comp.write("/archive/arch.txt", "archive findme")

    # Grep across all backends
    matches = comp.grep_raw("findme", path="/")
    assert isinstance(matches, list)
    match_paths = sorted([m["path"] for m in matches])

    # Expected: each file appears only in its own route
    expected_paths = sorted(
        [
            "/archive/arch.txt",
            "/default.txt",
            "/memories/mem.txt",
        ]
    )
    assert match_paths == expected_paths


def test_composite_grep_error_in_routed_backend() -> None:
    """Test grep error handling when routed backend returns error string."""
    rt = make_runtime("t_grep_err1")

    # Create a mock backend that returns error strings for grep
    class ErrorBackend(StoreBackend):
        def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Invalid regex pattern error"

    error_backend = ErrorBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

    # When searching a specific route that errors, return the error
    result = comp.grep_raw("test", path="/errors/")
    assert result == "Invalid regex pattern error"


def test_composite_grep_error_in_routed_backend_at_root() -> None:
    """Test grep error handling when routed backend errors during root search."""
    rt = make_runtime("t_grep_err2")

    # Create a mock backend that returns error strings for grep
    class ErrorBackend(StoreBackend):
        def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Backend error occurred"

    error_backend = ErrorBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

    # When searching from root and a routed backend errors, return the error
    result = comp.grep_raw("test", path="/")
    assert result == "Backend error occurred"


def test_composite_grep_error_in_default_backend_at_root() -> None:
    """Test grep error handling when default backend errors during root search."""
    rt = make_runtime("t_grep_err3")

    # Create a mock backend that returns error strings for grep
    class ErrorDefaultBackend(StateBackend):
        def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Default backend error"

    error_default = ErrorDefaultBackend(rt)
    store_backend = StoreBackend(rt)

    comp = CompositeBackend(default=error_default, routes={"/store/": store_backend})

    # When searching from root and default backend errors, return the error
    result = comp.grep_raw("test", path="/")
    assert result == "Default backend error"


def test_composite_grep_non_root_path_on_default_backend(tmp_path: Path) -> None:
    """Test grep with non-root path on default backend."""
    rt = make_runtime("t_grep_default")
    root = tmp_path

    # Create nested structure
    (root / "work").mkdir()
    (root / "work" / "project.txt").write_text("project content")
    (root / "other.txt").write_text("other content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Search in /work directory (doesn't match any route)
    matches = comp.grep_raw("content", path="/work")
    match_paths = [m["path"] for m in matches]

    # Should only find files in /work, not /other.txt
    assert match_paths == ["/work/project.txt"]


def test_composite_glob_info_targeting_specific_route() -> None:
    """Test glob_info when path matches a specific route."""
    rt = make_runtime("t_glob1")

    store = StoreBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/memories/": store})

    # Write files to memories
    comp.write("/memories/test.py", "python file")
    comp.write("/memories/data.json", "json file")
    comp.write("/memories/docs/readme.md", "markdown file")

    # Write to default backend
    state_backend.write("/local.py", "local python")

    # Glob in specific route with pattern - should only find .py files in memories
    results = comp.glob_info("**/*.py", path="/memories/")
    result_paths = [fi["path"] for fi in results]

    assert result_paths == ["/memories/test.py"]


def test_composite_glob_info_nested_path_in_route() -> None:
    """Test glob_info with nested path within route."""
    rt = make_runtime("t_glob2")

    store = StoreBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/archive/": store})

    # Write nested files
    comp.write("/archive/2024/jan.log", "january logs")
    comp.write("/archive/2024/feb.log", "february logs")
    comp.write("/archive/2023/dec.log", "december logs")
    comp.write("/archive/notes.txt", "general notes")

    # Glob in nested path within route - should only find .log files in /archive/2024/
    results = comp.glob_info("*.log", path="/archive/2024/")
    result_paths = sorted([fi["path"] for fi in results])

    assert result_paths == ["/archive/2024/feb.log", "/archive/2024/jan.log"]
