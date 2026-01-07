"""Async tests for CompositeBackend."""

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

    async def aexecute(self, command: str) -> ExecuteResponse:
        """Async mock execute that returns the command as output."""
        return ExecuteResponse(
            output=f"Async Executed: {command}",
            exit_code=0,
            truncated=False,
        )

    @property
    def id(self) -> str:
        return "mock_sandbox_backend"


async def test_composite_state_backend_routes_and_search_async(tmp_path: Path):
    """Test async operations with composite backend routing."""
    rt = make_runtime("t3")
    be = build_composite_state_backend(rt, routes={"/memories/": (lambda r: StoreBackend(r))})

    # write to default (state)
    res = await be.awrite("/file.txt", "alpha")
    assert isinstance(res, WriteResult) and res.files_update is not None

    # write to routed (store)
    msg = await be.awrite("/memories/readme.md", "beta")
    assert isinstance(msg, WriteResult) and msg.error is None and msg.files_update is None

    # als_info at root returns both
    infos = await be.als_info("/")
    paths = {i["path"] for i in infos}
    assert "/file.txt" in paths and "/memories/" in paths

    # agrep across both
    matches = await be.agrep_raw("alpha", path="/")
    assert any(m["path"] == "/file.txt" for m in matches)
    matches2 = await be.agrep_raw("beta", path="/")
    assert any(m["path"] == "/memories/readme.md" for m in matches2)

    # aglob across both
    g = await be.aglob_info("**/*.md", path="/")
    assert any(i["path"] == "/memories/readme.md" for i in g)


async def test_composite_backend_filesystem_plus_store_async(tmp_path: Path):
    """Test async operations with filesystem and store backends."""
    root = tmp_path
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    rt = make_runtime("t4")
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # put files in both
    r1 = await comp.awrite("/hello.txt", "hello")
    assert isinstance(r1, WriteResult) and r1.error is None and r1.files_update is None
    r2 = await comp.awrite("/memories/notes.md", "note")
    assert isinstance(r2, WriteResult) and r2.error is None and r2.files_update is None

    # als_info path routing
    infos_root = await comp.als_info("/")
    assert any(i["path"] == "/hello.txt" for i in infos_root)
    infos_mem = await comp.als_info("/memories/")
    assert any(i["path"] == "/memories/notes.md" for i in infos_mem)

    # agrep_raw merges
    gm = await comp.agrep_raw("hello", path="/")
    assert any(m["path"] == "/hello.txt" for m in gm)
    gm2 = await comp.agrep_raw("note", path="/")
    assert any(m["path"] == "/memories/notes.md" for m in gm2)

    # aglob_info
    gl = await comp.aglob_info("*.md", path="/")
    assert any(i["path"] == "/memories/notes.md" for i in gl)


async def test_composite_backend_store_to_store_async():
    """Test async operations with default store and routed store."""
    rt = make_runtime("t5")

    # Create two separate store backends
    default_store = StoreBackend(rt)
    memories_store = StoreBackend(rt)

    comp = CompositeBackend(default=default_store, routes={"/memories/": memories_store})

    # Write to default store
    res1 = await comp.awrite("/notes.txt", "default store content")
    assert isinstance(res1, WriteResult) and res1.error is None and res1.path == "/notes.txt"

    # Write to routed store
    res2 = await comp.awrite("/memories/important.txt", "routed store content")
    assert isinstance(res2, WriteResult) and res2.error is None and res2.path == "/important.txt"

    # Read from both
    content1 = await comp.aread("/notes.txt")
    assert "default store content" in content1

    content2 = await comp.aread("/memories/important.txt")
    assert "routed store content" in content2

    # als_info at root should show both
    infos = await comp.als_info("/")
    paths = {i["path"] for i in infos}
    assert "/notes.txt" in paths
    assert "/memories/" in paths

    # agrep across both stores
    matches = await comp.agrep_raw("default", path="/")
    assert any(m["path"] == "/notes.txt" for m in matches)

    matches2 = await comp.agrep_raw("routed", path="/")
    assert any(m["path"] == "/memories/important.txt" for m in matches2)


async def test_composite_backend_multiple_routes_async():
    """Test async operations with state default and multiple store routes."""
    rt = make_runtime("t6")

    comp = build_composite_state_backend(
        rt,
        routes={
            "/memories/": (lambda r: StoreBackend(r)),
            "/archive/": (lambda r: StoreBackend(r)),
            "/cache/": (lambda r: StoreBackend(r)),
        },
    )

    # Write to state (default)
    res_state = await comp.awrite("/temp.txt", "ephemeral data")
    assert res_state.files_update is not None
    assert res_state.path == "/temp.txt"

    # Write to /memories/ route
    res_mem = await comp.awrite("/memories/important.md", "long-term memory")
    assert res_mem.files_update is None
    assert res_mem.path == "/important.md"

    # Write to /archive/ route
    res_arch = await comp.awrite("/archive/old.log", "archived log")
    assert res_arch.files_update is None
    assert res_arch.path == "/old.log"

    # Write to /cache/ route
    res_cache = await comp.awrite("/cache/session.json", "cached session")
    assert res_cache.files_update is None
    assert res_cache.path == "/session.json"

    # als_info at root should aggregate all
    infos = await comp.als_info("/")
    paths = {i["path"] for i in infos}
    assert "/temp.txt" in paths
    assert "/memories/" in paths
    assert "/archive/" in paths
    assert "/cache/" in paths

    # als_info at specific route
    mem_infos = await comp.als_info("/memories/")
    mem_paths = {i["path"] for i in mem_infos}
    assert "/memories/important.md" in mem_paths
    assert "/temp.txt" not in mem_paths
    assert "/archive/old.log" not in mem_paths

    # agrep across all backends
    all_matches = await comp.agrep_raw(".", path="/")  # Match any character
    paths_with_content = {m["path"] for m in all_matches}
    assert "/temp.txt" in paths_with_content
    assert "/memories/important.md" in paths_with_content
    assert "/archive/old.log" in paths_with_content
    assert "/cache/session.json" in paths_with_content

    # aglob across all backends
    glob_results = await comp.aglob_info("**/*.md", path="/")
    assert any(i["path"] == "/memories/important.md" for i in glob_results)

    # Edit in routed backend
    edit_res = await comp.aedit("/memories/important.md", "long-term", "persistent", replace_all=False)
    assert edit_res.error is None
    assert edit_res.occurrences == 1

    updated_content = await comp.aread("/memories/important.md")
    assert "persistent memory" in updated_content


async def test_composite_backend_als_nested_directories_async(tmp_path: Path):
    """Test async ls operations with nested directories."""
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

    await comp.awrite("/memories/note1.txt", "note 1")
    await comp.awrite("/memories/deep/note2.txt", "note 2")
    await comp.awrite("/memories/deep/nested/note3.txt", "note 3")

    root_listing = await comp.als_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/local.txt" in root_paths
    assert "/src/" in root_paths
    assert "/memories/" in root_paths
    assert "/src/main.py" not in root_paths
    assert "/memories/note1.txt" not in root_paths

    src_listing = await comp.als_info("/src/")
    src_paths = [fi["path"] for fi in src_listing]
    assert "/src/main.py" in src_paths
    assert "/src/utils/" in src_paths
    assert "/src/utils/helper.py" not in src_paths

    mem_listing = await comp.als_info("/memories/")
    mem_paths = [fi["path"] for fi in mem_listing]
    assert "/memories/note1.txt" in mem_paths
    assert "/memories/deep/" in mem_paths
    assert "/memories/deep/note2.txt" not in mem_paths

    deep_listing = await comp.als_info("/memories/deep/")
    deep_paths = [fi["path"] for fi in deep_listing]
    assert "/memories/deep/note2.txt" in deep_paths
    assert "/memories/deep/nested/" in deep_paths
    assert "/memories/deep/nested/note3.txt" not in deep_paths


async def test_composite_backend_als_multiple_routes_nested_async():
    """Test async ls with multiple routes and nested directories."""
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
        res = await comp.awrite(path, content)
        if res.files_update:
            rt.state["files"].update(res.files_update)

    memory_files = {
        "/memories/important.txt": "important",
        "/memories/diary/entry1.txt": "diary entry",
    }

    for path, content in memory_files.items():
        await comp.awrite(path, content)

    archive_files = {
        "/archive/old.txt": "old",
        "/archive/2023/log.txt": "2023 log",
    }

    for path, content in archive_files.items():
        await comp.awrite(path, content)

    root_listing = await comp.als_info("/")
    root_paths = [fi["path"] for fi in root_listing]
    assert "/temp.txt" in root_paths
    assert "/work/" in root_paths
    assert "/memories/" in root_paths
    assert "/archive/" in root_paths
    assert "/work/file1.txt" not in root_paths
    assert "/memories/important.txt" not in root_paths

    work_listing = await comp.als_info("/work/")
    work_paths = [fi["path"] for fi in work_listing]
    assert "/work/file1.txt" in work_paths
    assert "/work/projects/" in work_paths
    assert "/work/projects/proj1.txt" not in work_paths

    mem_listing = await comp.als_info("/memories/")
    mem_paths = [fi["path"] for fi in mem_listing]
    assert "/memories/important.txt" in mem_paths
    assert "/memories/diary/" in mem_paths
    assert "/memories/diary/entry1.txt" not in mem_paths

    arch_listing = await comp.als_info("/archive/")
    arch_paths = [fi["path"] for fi in arch_listing]
    assert "/archive/old.txt" in arch_paths
    assert "/archive/2023/" in arch_paths
    assert "/archive/2023/log.txt" not in arch_paths


async def test_composite_backend_aexecute_with_sandbox_default_async():
    """Test async execute with sandbox default backend."""
    rt = make_runtime("t_exec1")
    sandbox = MockSandboxBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=sandbox, routes={"/memories/": store})

    # Execute should work since default backend supports it
    result = await comp.aexecute("ls -la")
    assert isinstance(result, ExecuteResponse)
    assert result.output == "Async Executed: ls -la"
    assert result.exit_code == 0
    assert result.truncated is False


async def test_composite_backend_aexecute_without_sandbox_default_async():
    """Test async execute fails when default doesn't support execution."""
    rt = make_runtime("t_exec2")
    state_backend = StateBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/memories/": store})

    # Execute should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="doesn't support command execution"):
        await comp.aexecute("ls -la")


async def test_composite_backend_aexecute_with_routed_backends_async():
    """Test async execution doesn't interfere with file routing."""
    rt = make_runtime("t_exec4")
    sandbox = MockSandboxBackend(rt)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=sandbox, routes={"/memories/": store})

    # Write files to both backends
    await comp.awrite("/local.txt", "local content")
    await comp.awrite("/memories/persistent.txt", "persistent content")

    # Execute should still work
    result = await comp.aexecute("echo test")
    assert result.output == "Async Executed: echo test"

    # File operations should still work
    assert "local content" in await comp.aread("/local.txt")
    assert "persistent content" in await comp.aread("/memories/persistent.txt")


async def test_composite_aupload_routing_async(tmp_path: Path):
    """Test async upload_files routing to different backends."""
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
    responses = await comp.aupload_files(default_files)
    assert len(responses) == 2
    assert all(r.error is None for r in responses)
    assert (root / "file1.bin").exists()
    assert (root / "file2.bin").read_bytes() == b"Default content 2"

    # Upload files to routed path (store)
    routed_files = [
        ("/memories/note1.bin", b"Memory content 1"),
        ("/memories/note2.bin", b"Memory content 2"),
    ]
    responses = await comp.aupload_files(routed_files)
    assert len(responses) == 2
    assert all(r.error is None for r in responses)

    # Verify files are accessible in store
    content1 = await comp.aread("/memories/note1.bin")
    assert "Memory content 1" in content1


async def test_composite_adownload_routing_async(tmp_path: Path):
    """Test async download_files routing to different backends."""
    rt = make_runtime("t_download1")
    root = tmp_path

    # Create composite with filesystem default and store route
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)
    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Pre-populate filesystem backend
    (root / "local.bin").write_bytes(b"Local binary data")

    # Pre-populate store backend
    await comp.awrite("/memories/stored.txt", "Stored text data")

    # Download from default path (filesystem)
    responses = await comp.adownload_files(["/local.bin"])
    assert len(responses) == 1
    assert responses[0].path == "/local.bin"
    assert responses[0].content == b"Local binary data"
    assert responses[0].error is None


async def test_composite_aupload_download_roundtrip_async(tmp_path: Path):
    """Test async upload and download roundtrip through composite backend."""
    rt = make_runtime("t_roundtrip1")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Upload binary content
    test_content = bytes(range(128))  # Binary data
    upload_responses = await comp.aupload_files([("/test.bin", test_content)])
    assert upload_responses[0].error is None

    # Download it back
    download_responses = await comp.adownload_files(["/test.bin"])
    assert download_responses[0].error is None
    assert download_responses[0].content == test_content


async def test_composite_partial_success_aupload_async(tmp_path: Path):
    """Test partial success in async batch upload with mixed valid/invalid paths."""
    rt = make_runtime("t_partial_upload")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    files = [
        ("/valid1.bin", b"Valid 1"),
        ("/../invalid.bin", b"Invalid path"),  # Path traversal
        ("/valid2.bin", b"Valid 2"),
    ]

    responses = await comp.aupload_files(files)

    assert len(responses) == 3
    # First should succeed
    assert responses[0].error is None
    assert (root / "valid1.bin").exists()

    # Second should fail
    assert responses[1].error == "invalid_path"

    # Third should still succeed (partial success)
    assert responses[2].error is None
    assert (root / "valid2.bin").exists()


async def test_composite_partial_success_adownload_async(tmp_path: Path):
    """Test partial success in async batch download with mixed valid/invalid paths."""
    rt = make_runtime("t_partial_download")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Create one valid file
    (root / "exists.bin").write_bytes(b"I exist!")

    paths = ["/exists.bin", "/doesnotexist.bin", "/../invalid"]
    responses = await comp.adownload_files(paths)

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


async def test_composite_aupload_download_multiple_routes_async(tmp_path: Path):
    """Test async upload/download with multiple routed backends."""
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

    responses = await comp.aupload_files(files)
    assert len(responses) == 3
    assert all(r.error is None for r in responses)

    # Verify routing worked (filesystem file should exist)
    assert (root / "default.bin").exists()
    assert (root / "default.bin").read_bytes() == b"Default backend"


async def test_composite_adownload_preserves_original_paths_async(tmp_path: Path):
    """Test async download responses preserve original composite paths."""
    rt = make_runtime("t_path_preserve")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Create files
    (root / "subdir").mkdir()
    (root / "subdir" / "file.bin").write_bytes(b"Nested file")

    # Download with composite path
    responses = await comp.adownload_files(["/subdir/file.bin"])

    # Response should have the original composite path, not stripped
    assert responses[0].path == "/subdir/file.bin"
    assert responses[0].content == b"Nested file"


async def test_composite_agrep_targeting_specific_route_async(tmp_path: Path) -> None:
    """Test async grep with path targeting a specific routed backend."""
    rt = make_runtime("t_agrep1")
    root = tmp_path

    # Setup filesystem backend with some files
    (root / "default.txt").write_text("default backend content")
    (root / "default2.txt").write_text("more default stuff")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Write to memories route
    await comp.awrite("/memories/note1.txt", "memory content alpha")
    await comp.awrite("/memories/note2.txt", "memory content beta")

    # Grep with path="/memories/" should only search memories backend
    matches = await comp.agrep_raw("memory", path="/memories/")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find matches in /memories/
    assert any("/memories/note1.txt" in p for p in match_paths)
    assert any("/memories/note2.txt" in p for p in match_paths)

    # Should NOT find matches in default backend
    assert not any("/default" in p for p in match_paths)


async def test_composite_agrep_with_glob_filter_async(tmp_path: Path) -> None:
    """Test async grep with glob parameter to filter files."""
    rt = make_runtime("t_agrep2")
    root = tmp_path

    # Create files with different extensions
    (root / "script.py").write_text("python code here")
    (root / "config.json").write_text("json config here")
    (root / "readme.md").write_text("markdown docs here")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Add some files to memories route
    await comp.awrite("/memories/notes.py", "python notes here")
    await comp.awrite("/memories/data.json", "json data here")

    # Grep with glob="*.py" should only search Python files
    matches = await comp.agrep_raw("here", path="/", glob="*.py")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find .py files
    assert any("/script.py" in p for p in match_paths)
    assert any("/memories/notes.py" in p for p in match_paths)

    # Should NOT find non-.py files
    assert not any(".json" in p for p in match_paths)
    assert not any(".md" in p for p in match_paths)


async def test_composite_agrep_with_glob_in_specific_route_async(tmp_path: Path) -> None:
    """Test async grep with glob parameter targeting a specific route."""
    rt = make_runtime("t_agrep3")
    root = tmp_path

    (root / "local.md").write_text("local markdown")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Add files to memories
    await comp.awrite("/memories/important.md", "important notes")
    await comp.awrite("/memories/data.txt", "text data")

    # Grep memories with glob="*.md"
    matches = await comp.agrep_raw("notes", path="/memories/", glob="*.md")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find .md file in memories
    assert any("/memories/important.md" in p for p in match_paths)

    # Should NOT find .txt files or default backend files
    assert not any("/memories/data.txt" in p for p in match_paths)
    assert not any("/local.md" in p for p in match_paths)


async def test_composite_agrep_with_path_none_async(tmp_path: Path) -> None:
    """Test async grep with path=None behaves like path='/'."""
    rt = make_runtime("t_agrep4")
    root = tmp_path

    (root / "file1.txt").write_text("searchable content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    await comp.awrite("/memories/file2.txt", "searchable memory")

    # Grep with path=None
    matches_none = await comp.agrep_raw("searchable", path=None)
    assert isinstance(matches_none, list)

    # Grep with path="/"
    matches_root = await comp.agrep_raw("searchable", path="/")
    assert isinstance(matches_root, list)

    # Both should return same results
    paths_none = sorted([m["path"] for m in matches_none])
    paths_root = sorted([m["path"] for m in matches_root])

    assert paths_none == paths_root
    assert len(paths_none) == 2


async def test_composite_agrep_invalid_regex_async(tmp_path: Path) -> None:
    """Test async grep with invalid regex pattern returns error string."""
    rt = make_runtime("t_agrep5")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Invalid regex patterns
    result = await comp.agrep_raw("[invalid(", path="/")
    assert isinstance(result, str)
    assert "Invalid regex" in result or "error" in result.lower()


async def test_composite_agrep_nested_path_in_route_async(tmp_path: Path) -> None:
    """Test async grep with nested path within a routed backend."""
    rt = make_runtime("t_agrep6")
    root = tmp_path

    (root / "local.txt").write_text("local content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Create nested structure in memories
    await comp.awrite("/memories/docs/readme.md", "documentation here")
    await comp.awrite("/memories/docs/guide.md", "guide here")
    await comp.awrite("/memories/notes.txt", "notes here")

    # Grep with nested path
    matches = await comp.agrep_raw("here", path="/memories/docs/")
    assert isinstance(matches, list)
    match_paths = [m["path"] for m in matches]

    # Should find files in /memories/docs/
    assert any("/memories/docs/readme.md" in p for p in match_paths)
    assert any("/memories/docs/guide.md" in p for p in match_paths)

    # Should NOT find files outside /memories/docs/
    assert not any("/memories/notes.txt" in p for p in match_paths)
    assert not any("/local.txt" in p for p in match_paths)


async def test_composite_agrep_empty_results_async(tmp_path: Path) -> None:
    """Test async grep that matches nothing returns empty list."""
    rt = make_runtime("t_agrep7")
    root = tmp_path

    (root / "file.txt").write_text("some content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    await comp.awrite("/memories/note.txt", "memory content")

    # Search for pattern that doesn't exist
    matches = await comp.agrep_raw("nonexistent_pattern_xyz", path="/")
    assert isinstance(matches, list)
    assert len(matches) == 0


async def test_composite_agrep_route_prefix_restoration_async(tmp_path: Path) -> None:
    """Test async grep correctly restores route prefixes in results."""
    rt = make_runtime("t_agrep8")
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Write files to memories
    await comp.awrite("/memories/alpha.txt", "test content alpha")
    await comp.awrite("/memories/beta.txt", "test content beta")

    # Grep in memories route
    matches = await comp.agrep_raw("test", path="/memories/")
    assert isinstance(matches, list)
    assert len(matches) > 0

    # All paths should start with /memories/
    for match in matches:
        assert match["path"].startswith("/memories/")
        assert not match["path"].startswith("/memories//")  # No double slashes

    # Grep across all backends (path="/")
    matches_all = await comp.agrep_raw("test", path="/")
    assert isinstance(matches_all, list)

    # Filter matches from memories
    memory_matches = [m for m in matches_all if "/memories/" in m["path"]]
    for match in memory_matches:
        assert match["path"].startswith("/memories/")


async def test_composite_agrep_multiple_matches_per_file_async(tmp_path: Path) -> None:
    """Test async grep returns multiple matches from same file."""
    rt = make_runtime("t_agrep9")
    root = tmp_path

    # File with multiple matching lines
    (root / "multi.txt").write_text("line1 pattern\nline2 pattern\nline3 other")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    matches = await comp.agrep_raw("pattern", path="/")
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
async def test_composite_agrep_multiple_routes_aggregation_async(tmp_path: Path) -> None:
    """Test async grep aggregates results from multiple routed backends with expected isolation.

    This test represents the intuitive expected behavior: files written to /memories/
    should only appear in /memories/, and files written to /archive/ should only appear
    in /archive/.
    """
    rt = make_runtime("t_agrep10")
    root = tmp_path

    (root / "default.txt").write_text("default findme")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store1 = StoreBackend(rt)
    store2 = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store1, "/archive/": store2})

    # Write to each route
    await comp.awrite("/memories/mem.txt", "memory findme")
    await comp.awrite("/archive/arch.txt", "archive findme")

    # Grep across all backends
    matches = await comp.agrep_raw("findme", path="/")
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


async def test_composite_agrep_error_in_routed_backend_async() -> None:
    """Test async grep error handling when routed backend returns error string."""
    rt = make_runtime("t_agrep_err1")

    # Create a mock backend that returns error strings for grep
    class ErrorBackend(StoreBackend):
        async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Invalid regex pattern error"

    error_backend = ErrorBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

    # When searching a specific route that errors, return the error
    result = await comp.agrep_raw("test", path="/errors/")
    assert result == "Invalid regex pattern error"


async def test_composite_agrep_error_in_routed_backend_at_root_async() -> None:
    """Test async grep error handling when routed backend errors during root search."""
    rt = make_runtime("t_agrep_err2")

    # Create a mock backend that returns error strings for grep
    class ErrorBackend(StoreBackend):
        async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Backend error occurred"

    error_backend = ErrorBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

    # When searching from root and a routed backend errors, return the error
    result = await comp.agrep_raw("test", path="/")
    assert result == "Backend error occurred"


async def test_composite_agrep_error_in_default_backend_at_root_async() -> None:
    """Test async grep error handling when default backend errors during root search."""
    rt = make_runtime("t_agrep_err3")

    # Create a mock backend that returns error strings for grep
    class ErrorDefaultBackend(StateBackend):
        async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
            return "Default backend error"

    error_default = ErrorDefaultBackend(rt)
    store_backend = StoreBackend(rt)

    comp = CompositeBackend(default=error_default, routes={"/store/": store_backend})

    # When searching from root and default backend errors, return the error
    result = await comp.agrep_raw("test", path="/")
    assert result == "Default backend error"


async def test_composite_agrep_non_root_path_on_default_backend_async(tmp_path: Path) -> None:
    """Test async grep with non-root path on default backend."""
    rt = make_runtime("t_agrep_default")
    root = tmp_path

    # Create nested structure
    (root / "work").mkdir()
    (root / "work" / "project.txt").write_text("project content")
    (root / "other.txt").write_text("other content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    store = StoreBackend(rt)

    comp = CompositeBackend(default=fs, routes={"/memories/": store})

    # Search in /work directory (doesn't match any route)
    matches = await comp.agrep_raw("content", path="/work")
    match_paths = [m["path"] for m in matches]

    # Should only find files in /work, not /other.txt
    assert match_paths == ["/work/project.txt"]


async def test_composite_aglob_info_targeting_specific_route_async() -> None:
    """Test async glob_info when path matches a specific route."""
    rt = make_runtime("t_aglob1")

    store = StoreBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/memories/": store})

    # Write files to memories
    await comp.awrite("/memories/test.py", "python file")
    await comp.awrite("/memories/data.json", "json file")
    await comp.awrite("/memories/docs/readme.md", "markdown file")

    # Write to default backend
    await state_backend.awrite("/local.py", "local python")

    # Glob in specific route with pattern - should only find .py files in memories
    results = await comp.aglob_info("**/*.py", path="/memories/")
    result_paths = [fi["path"] for fi in results]

    assert result_paths == ["/memories/test.py"]


async def test_composite_aglob_info_nested_path_in_route_async() -> None:
    """Test async glob_info with nested path within route."""
    rt = make_runtime("t_aglob2")

    store = StoreBackend(rt)
    state_backend = StateBackend(rt)

    comp = CompositeBackend(default=state_backend, routes={"/archive/": store})

    # Write nested files
    await comp.awrite("/archive/2024/jan.log", "january logs")
    await comp.awrite("/archive/2024/feb.log", "february logs")
    await comp.awrite("/archive/2023/dec.log", "december logs")
    await comp.awrite("/archive/notes.txt", "general notes")

    # Glob in nested path within route - should only find .log files in /archive/2024/
    results = await comp.aglob_info("*.log", path="/archive/2024/")
    result_paths = sorted([fi["path"] for fi in results])

    assert result_paths == ["/archive/2024/feb.log", "/archive/2024/jan.log"]
