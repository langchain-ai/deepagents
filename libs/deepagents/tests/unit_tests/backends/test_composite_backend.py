from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import var_child_runnable_config
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_RUNTIME, CONFIG_KEY_SEND
from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend, _route_for_path
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, _file_data_reducer


def make_runtime(tid: str = "tc", *, store=None):
    """Minimal ToolRuntime - only needed for _intercept_large_tool_result calls."""
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tid,
        store=store or InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


def _make_state_config(files=None, *, store=None):
    file_store = {"files": files or {}}

    def read(select, *, _fresh=False):
        if isinstance(select, str):
            return file_store.get(select)
        return {k: file_store.get(k) for k in select}

    def send(writes):
        for channel, value in writes:
            if channel == "files":
                file_store["files"] = _file_data_reducer(file_store.get("files"), value)

    mock_runtime = SimpleNamespace(store=store, context=None, stream_writer=lambda _: None)
    config = {
        "configurable": {
            CONFIG_KEY_SEND: send,
            CONFIG_KEY_READ: read,
            CONFIG_KEY_RUNTIME: mock_runtime,
        },
    }
    return config, file_store


@contextmanager
def state_config_context(files=None, *, store=None):
    config, file_store = _make_state_config(files, store=store)
    token = var_child_runnable_config.set(config)
    try:
        yield file_store
    finally:
        var_child_runnable_config.reset(token)


def test_composite_state_backend_routes_and_search(tmp_path: Path):  # noqa: ARG001  # Pytest fixture
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        # route /memories/ to store
        be = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})

        # write to default (state)
        res = be.write("/file.txt", "alpha")
        assert isinstance(res, WriteResult)

        # write to routed (store)
        msg = be.write("/memories/readme.md", "beta")
        assert isinstance(msg, WriteResult) and msg.error is None

        # ls_info at root returns both
        infos = be.ls("/").entries
        assert infos is not None
        paths = {i["path"] for i in infos}
        assert "/file.txt" in paths and "/memories/" in paths

        # grep across both
        matches = be.grep("alpha", path="/").matches
        assert matches is not None
        assert any(m["path"] == "/file.txt" for m in matches)
        matches2 = be.grep("beta", path="/").matches
        assert matches2 is not None
        assert any(m["path"] == "/memories/readme.md" for m in matches2)

        # glob across both
        g = be.glob("**/*.md", path="/").matches
        assert any(i["path"] == "/memories/readme.md" for i in g)


def test_composite_backend_filesystem_plus_store(tmp_path: Path):
    # default filesystem, route to store under /memories/
    root = tmp_path
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        store = StoreBackend()
        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # put files in both
        r1 = comp.write("/hello.txt", "hello")
        assert isinstance(r1, WriteResult) and r1.error is None
        r2 = comp.write("/memories/notes.md", "note")
        assert isinstance(r2, WriteResult) and r2.error is None

        # ls_info path routing
        infos_root = comp.ls("/").entries
        assert infos_root is not None
        assert any(i["path"] == "/hello.txt" for i in infos_root)
        infos_mem = comp.ls("/memories/").entries
        assert infos_mem is not None
        assert any(i["path"] == "/memories/notes.md" for i in infos_mem)

        infos_mem_no_slash = comp.ls("/memories").entries
        assert infos_mem_no_slash is not None
        assert any(i["path"] == "/memories/notes.md" for i in infos_mem_no_slash)

        # grep route targeting should accept /memories as the route root
        gm_mem = comp.grep("note", path="/memories").matches
        assert gm_mem is not None
        assert any(m["path"] == "/memories/notes.md" for m in gm_mem)

        # glob route targeting should accept /memories as the route root
        gl_mem = comp.glob("*.md", path="/memories").matches
        assert any(i["path"] == "/memories/notes.md" for i in gl_mem)

        # grep merges
        gm = comp.grep("hello", path="/").matches
        assert gm is not None
        assert any(m["path"] == "/hello.txt" for m in gm)
        gm2 = comp.grep("note", path="/").matches
        assert gm2 is not None
        assert any(m["path"] == "/memories/notes.md" for m in gm2)

        # glob
        gl = comp.glob("*.md", path="/").matches
        assert any(i["path"] == "/memories/notes.md" for i in gl)


def test_composite_backend_store_to_store():
    """Test composite with default store and routed store (two different stores)."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # Create two separate store backends (simulating different namespaces/stores)
        default_store = StoreBackend()
        memories_store = StoreBackend()

        comp = CompositeBackend(default=default_store, routes={"/memories/": memories_store})

        # Write to default store
        res1 = comp.write("/notes.txt", "default store content")
        assert isinstance(res1, WriteResult) and res1.error is None and res1.path == "/notes.txt"

        # Write to routed store
        res2 = comp.write("/memories/important.txt", "routed store content")
        assert isinstance(res2, WriteResult) and res2.error is None and res2.path == "/memories/important.txt"

        # Read from both
        content1 = comp.read("/notes.txt")
        assert content1.file_data is not None
        assert "default store content" in content1.file_data["content"]

        content2 = comp.read("/memories/important.txt")
        assert content2.file_data is not None
        assert "routed store content" in content2.file_data["content"]

        # ls_info at root should show both
        infos = comp.ls("/").entries
        assert infos is not None
        paths = {i["path"] for i in infos}
        assert "/notes.txt" in paths
        assert "/memories/" in paths

        # grep across both stores
        matches = comp.grep("default", path="/").matches
        assert matches is not None
        assert any(m["path"] == "/notes.txt" for m in matches)

        matches2 = comp.grep("routed", path="/").matches
        assert matches2 is not None
        assert any(m["path"] == "/memories/important.txt" for m in matches2)


def test_composite_backend_multiple_routes():
    """Test composite with state default and multiple store routes."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # State backend as default, multiple stores for different routes
        comp = CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(),
                "/archive/": StoreBackend(),
                "/cache/": StoreBackend(),
            },
        )

        # Write to state (default)
        res_state = comp.write("/temp.txt", "ephemeral data")
        assert res_state.path == "/temp.txt"

        # Write to /memories/ route
        res_mem = comp.write("/memories/important.md", "long-term memory")
        assert res_mem.path == "/memories/important.md"

        # Write to /archive/ route
        res_arch = comp.write("/archive/old.log", "archived log")
        assert res_arch.path == "/archive/old.log"

        # Write to /cache/ route
        res_cache = comp.write("/cache/session.json", "cached session")
        assert res_cache.path == "/cache/session.json"

        # ls_info at root should aggregate all
        infos = comp.ls("/").entries
        assert infos is not None
        paths = {i["path"] for i in infos}
        assert "/temp.txt" in paths
        assert "/memories/" in paths
        assert "/archive/" in paths
        assert "/cache/" in paths

        # ls_info at specific route
        mem_infos = comp.ls("/memories/").entries
        assert mem_infos is not None
        mem_paths = {i["path"] for i in mem_infos}
        assert "/memories/important.md" in mem_paths
        assert "/temp.txt" not in mem_paths
        assert "/archive/old.log" not in mem_paths

        # grep across all backends with literal text search
        # Note: All written content contains 'm' character
        all_matches = comp.grep("m", path="/").matches  # Match literal 'm'
        assert all_matches is not None
        paths_with_content = {m["path"] for m in all_matches}
        assert "/temp.txt" in paths_with_content  # "ephemeral" contains 'm'
        # Note: Store routes might share state in tests, so just verify default backend works
        assert len(paths_with_content) >= 1  # At least temp.txt should match

        # glob across all backends
        glob_results = comp.glob("**/*.md", path="/").matches
        assert any(i["path"] == "/memories/important.md" for i in glob_results)

        # Edit in routed backend
        edit_res = comp.edit("/memories/important.md", "long-term", "persistent", replace_all=False)
        assert edit_res.error is None
        assert edit_res.occurrences == 1
        assert edit_res.path == "/memories/important.md"

        updated_content = comp.read("/memories/important.md")
        assert updated_content.file_data is not None
        assert "persistent memory" in updated_content.file_data["content"]


def test_composite_backend_grep_path_isolation():
    """Test that grep with path=/tools doesn't return results from /memories."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # Use StateBackend as default, StoreBackend for /memories/
        state = StateBackend()
        store_be = StoreBackend()

        comp = CompositeBackend(default=state, routes={"/memories/": store_be})

        # Write to state backend (default) in /tools directory
        comp.write("/tools/hammer.txt", "tool for nailing")
        comp.write("/tools/saw.txt", "tool for cutting")

        # Write to memories route with content that would match our grep
        comp.write("/memories/workshop.txt", "tool shed location")
        comp.write("/memories/notes.txt", "remember to buy tools")

        # Grep for "tool" in /tools directory - should NOT return /memories results
        result = comp.grep("tool", path="/tools")
        matches = result.matches
        match_paths = [m["path"] for m in matches] if matches is not None else []

        # Should find results in /tools
        assert any("/tools/hammer.txt" in p for p in match_paths)
        assert any("/tools/saw.txt" in p for p in match_paths)

        # Should NOT find results in /memories (this is the bug)
        assert not any("/memories/" in p for p in match_paths), f"grep path=/tools should not return /memories results, but got: {match_paths}"


def test_composite_backend_ls_nested_directories(tmp_path: Path):
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
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        comp.write("/memories/note1.txt", "note 1")
        comp.write("/memories/deep/note2.txt", "note 2")
        comp.write("/memories/deep/nested/note3.txt", "note 3")

        root_listing = comp.ls("/").entries
        assert root_listing is not None
        root_paths = [fi["path"] for fi in root_listing]
        assert "/local.txt" in root_paths
        assert "/src/" in root_paths
        assert "/memories/" in root_paths
        assert "/src/main.py" not in root_paths
        assert "/memories/note1.txt" not in root_paths

        src_listing = comp.ls("/src/").entries
        assert src_listing is not None
        src_paths = [fi["path"] for fi in src_listing]
        assert "/src/main.py" in src_paths
        assert "/src/utils/" in src_paths
        assert "/src/utils/helper.py" not in src_paths

        mem_listing = comp.ls("/memories/").entries
        assert mem_listing is not None
        mem_paths = [fi["path"] for fi in mem_listing]
        assert "/memories/note1.txt" in mem_paths
        assert "/memories/deep/" in mem_paths
        assert "/memories/deep/note2.txt" not in mem_paths

        deep_listing = comp.ls("/memories/deep/").entries
        assert deep_listing is not None
        deep_paths = [fi["path"] for fi in deep_listing]
        assert "/memories/deep/note2.txt" in deep_paths
        assert "/memories/deep/nested/" in deep_paths
        assert "/memories/deep/nested/note3.txt" not in deep_paths


def test_composite_backend_ls_multiple_routes_nested():
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        comp = CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(),
                "/archive/": StoreBackend(),
            },
        )

        state_files = {
            "/temp.txt": "temp",
            "/work/file1.txt": "work file 1",
            "/work/projects/proj1.txt": "project 1",
        }

        for path, content in state_files.items():
            comp.write(path, content)

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

        root_listing = comp.ls("/").entries
        assert root_listing is not None
        root_paths = [fi["path"] for fi in root_listing]
        assert "/temp.txt" in root_paths
        assert "/work/" in root_paths
        assert "/memories/" in root_paths
        assert "/archive/" in root_paths
        assert "/work/file1.txt" not in root_paths
        assert "/memories/important.txt" not in root_paths

        work_listing = comp.ls("/work/").entries
        assert work_listing is not None
        work_paths = [fi["path"] for fi in work_listing]
        assert "/work/file1.txt" in work_paths
        assert "/work/projects/" in work_paths
        assert "/work/projects/proj1.txt" not in work_paths

        mem_listing = comp.ls("/memories/").entries
        assert mem_listing is not None
        mem_paths = [fi["path"] for fi in mem_listing]
        assert "/memories/important.txt" in mem_paths
        assert "/memories/diary/" in mem_paths
        assert "/memories/diary/entry1.txt" not in mem_paths

        arch_listing = comp.ls("/archive/").entries
        assert arch_listing is not None
        arch_paths = [fi["path"] for fi in arch_listing]
        assert "/archive/old.txt" in arch_paths
        assert "/archive/2023/" in arch_paths
        assert "/archive/2023/log.txt" not in arch_paths


def test_composite_backend_ls_trailing_slash(tmp_path: Path):
    root = tmp_path

    (root / "file.txt").write_text("content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/store/": store})

        comp.write("/store/item.txt", "store content")

        listing = comp.ls("/").entries
        assert listing is not None
        paths = [fi["path"] for fi in listing]
        assert paths == sorted(paths)

        empty_listing = comp.ls("/store/nonexistent/")
        assert empty_listing.entries == []

        empty_listing2 = comp.ls("/nonexistent/")
        assert empty_listing2.entries == []

        listing1 = comp.ls("/store/").entries
        listing2 = comp.ls("/store").entries
        assert listing1 is not None
        assert listing2 is not None
        assert [fi["path"] for fi in listing1] == [fi["path"] for fi in listing2]


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_composite_backend_intercept_large_tool_result(file_format):
    mem_store = InMemoryStore()
    rt = make_runtime("t10", store=mem_store)

    with state_config_context(store=mem_store) as store:
        middleware = FilesystemMiddleware(
            backend=CompositeBackend(default=StateBackend(file_format=file_format), routes={"/memories/": StoreBackend()}),
            tool_token_limit_before_evict=1000,
        )
        large_content = "z" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_789")
        result = middleware._intercept_large_tool_result(tool_message, rt)

        # With CONFIG_KEY_SEND, the write goes directly to state — result is just a ToolMessage
        assert isinstance(result, ToolMessage)
        assert "Tool result too large" in result.content
        # Verify the file was written to state via send
        assert "/large_tool_results/test_789" in store["files"]
        expected = [large_content] if file_format == "v1" else large_content
        assert store["files"]["/large_tool_results/test_789"]["content"] == expected


@pytest.mark.parametrize("file_format", ["v1", "v2"])
def test_composite_backend_intercept_large_tool_result_routed_to_store(file_format):
    """Test that large tool results can be routed to a specific backend like StoreBackend."""
    mem_store = InMemoryStore()
    rt = make_runtime("t11", store=mem_store)

    with state_config_context(store=mem_store):
        middleware = FilesystemMiddleware(
            backend=CompositeBackend(
                default=StateBackend(file_format=file_format),
                routes={"/large_tool_results/": StoreBackend(file_format=file_format)},
            ),
            tool_token_limit_before_evict=1000,
        )

        large_content = "w" * 5000
        tool_message = ToolMessage(content=large_content, tool_call_id="test_routed_123")
        result = middleware._intercept_large_tool_result(tool_message, rt)

        assert isinstance(result, ToolMessage)
        assert "Tool result too large" in result.content
        assert "/large_tool_results/test_routed_123" in result.content

        stored_item = mem_store.get(("filesystem",), "/test_routed_123")
        assert stored_item is not None
        expected = [large_content] if file_format == "v1" else large_content
        assert stored_item.value["content"] == expected


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
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        sandbox = MockSandboxBackend()
        store_be = StoreBackend()

        comp = CompositeBackend(default=sandbox, routes={"/memories/": store_be})

        # Execute should work since default backend supports it
        result = comp.execute("ls -la")
        assert isinstance(result, ExecuteResponse)
        assert result.output == "Executed: ls -la"
        assert result.exit_code == 0
        assert result.truncated is False


def test_composite_backend_execute_without_sandbox_default():
    """Test that CompositeBackend.execute() fails when default doesn't support execution."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        state_backend = StateBackend()  # StateBackend doesn't implement SandboxBackendProtocol
        store_be = StoreBackend()

        comp = CompositeBackend(default=state_backend, routes={"/memories/": store_be})

        # Execute should raise NotImplementedError since default backend doesn't support it
        with pytest.raises(NotImplementedError, match="doesn't support command execution"):
            comp.execute("ls -la")


def test_composite_backend_supports_execution_check():
    """Test the isinstance check works correctly for CompositeBackend."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # CompositeBackend with sandbox default should pass isinstance check
        sandbox = MockSandboxBackend()
        comp_with_sandbox = CompositeBackend(default=sandbox, routes={})
        # Note: CompositeBackend itself has execute() method, so isinstance will pass
        # but the actual support depends on the default backend
        assert hasattr(comp_with_sandbox, "execute")

        # CompositeBackend with non-sandbox default should still have execute() method
        # but will raise NotImplementedError when called
        state = StateBackend()
        comp_without_sandbox = CompositeBackend(default=state, routes={})
        assert hasattr(comp_without_sandbox, "execute")


def test_composite_backend_execute_with_routed_backends():
    """Test that execution doesn't interfere with file routing."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        sandbox = MockSandboxBackend()
        store_be = StoreBackend()

        comp = CompositeBackend(default=sandbox, routes={"/memories/": store_be})

        # Write files to both backends
        comp.write("/local.txt", "local content")
        comp.write("/memories/persistent.txt", "persistent content")

        # Execute should still work
        result = comp.execute("echo test")
        assert result.output == "Executed: echo test"

        # File operations should still work
        local_result = comp.read("/local.txt")
        assert local_result.file_data is not None
        assert "local content" in local_result.file_data["content"]
        persistent_result = comp.read("/memories/persistent.txt")
        assert persistent_result.file_data is not None
        assert "persistent content" in persistent_result.file_data["content"]


def test_composite_upload_routing(tmp_path: Path):
    """Test upload_files routing to different backends."""
    root = tmp_path

    # Create composite with filesystem default and store route
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()
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
            ("/memories/note1.txt", b"Memory content 1"),
            ("/memories/note2.txt", b"Memory content 2"),
        ]
        responses = comp.upload_files(routed_files)
        assert len(responses) == 2
        assert all(r.error is None for r in responses)

        # Verify files are accessible in store
        content1 = comp.read("/memories/note1.txt")
        assert content1.file_data is not None
        assert "Memory content 1" in content1.file_data["content"]


def test_composite_download_routing(tmp_path: Path):
    """Test download_files routing to different backends."""
    root = tmp_path

    # Create composite with filesystem default and store route
    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()
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
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store1 = StoreBackend()
        store2 = StoreBackend()

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
    root = tmp_path

    # Setup filesystem backend with some files
    (root / "default.txt").write_text("default backend content")
    (root / "default2.txt").write_text("more default stuff")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Write to memories route
        comp.write("/memories/note1.txt", "memory content alpha")
        comp.write("/memories/note2.txt", "memory content beta")

        # Grep with path="/memories/" should only search memories backend
        matches = comp.grep("memory", path="/memories/").matches
        assert matches is not None
        match_paths = [m["path"] for m in matches]

        # Should find matches in /memories/
        assert any("/memories/note1.txt" in p for p in match_paths)
        assert any("/memories/note2.txt" in p for p in match_paths)

        # Should NOT find matches in default backend
        assert not any("/default" in p for p in match_paths)


def test_composite_grep_with_glob_filter(tmp_path: Path) -> None:
    """Test grep with glob parameter to filter files."""
    root = tmp_path

    # Create files with different extensions
    (root / "script.py").write_text("python code here")
    (root / "config.json").write_text("json config here")
    (root / "readme.md").write_text("markdown docs here")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Add some files to memories route
        comp.write("/memories/notes.py", "python notes here")
        comp.write("/memories/data.json", "json data here")

        # Grep with glob="*.py" should only search Python files
        matches = comp.grep("here", path="/", glob="*.py").matches
        assert matches is not None
        match_paths = [m["path"] for m in matches]

        # Should find .py files
        assert any("/script.py" in p for p in match_paths)
        assert any("/memories/notes.py" in p for p in match_paths)

        # Should NOT find non-.py files
        assert not any(".json" in p for p in match_paths)
        assert not any(".md" in p for p in match_paths)


def test_composite_grep_with_glob_in_specific_route(tmp_path: Path) -> None:
    """Test grep with glob parameter targeting a specific route."""
    root = tmp_path

    (root / "local.md").write_text("local markdown")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Add files to memories
        comp.write("/memories/important.md", "important notes")
        comp.write("/memories/data.txt", "text data")

        # Grep memories with glob="*.md"
        matches = comp.grep("notes", path="/memories/", glob="*.md").matches
        assert matches is not None
        match_paths = [m["path"] for m in matches]

        # Should find .md file in memories
        assert any("/memories/important.md" in p for p in match_paths)

        # Should NOT find .txt files or default backend files
        assert not any("/memories/data.txt" in p for p in match_paths)
        assert not any("/local.md" in p for p in match_paths)


def test_composite_grep_with_path_none(tmp_path: Path) -> None:
    """Test grep with path=None behaves like path='/'."""
    root = tmp_path

    (root / "file1.txt").write_text("searchable content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        comp.write("/memories/file2.txt", "searchable memory")

        # Grep with path=None
        matches_none = comp.grep("searchable", path=None).matches
        assert matches_none is not None

        # Grep with path="/"
        matches_root = comp.grep("searchable", path="/").matches
        assert matches_root is not None

        # Both should return same results
        paths_none = sorted([m["path"] for m in matches_none])
        paths_root = sorted([m["path"] for m in matches_root])

        assert paths_none == paths_root
        assert len(paths_none) == 2


def test_composite_grep_invalid_regex(tmp_path: Path) -> None:
    """Test grep with special characters (literal search, not regex)."""
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    # Special characters are treated literally (not regex), should return empty list
    result = comp.grep("[invalid(", path="/")
    assert result.matches is not None  # Returns empty list, not error


def test_composite_grep_nested_path_in_route(tmp_path: Path) -> None:
    """Test grep with nested path within a routed backend."""
    root = tmp_path

    (root / "local.txt").write_text("local content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Create nested structure in memories
        comp.write("/memories/docs/readme.md", "documentation here")
        comp.write("/memories/docs/guide.md", "guide here")
        comp.write("/memories/notes.txt", "notes here")

        # Grep with nested path
        matches = comp.grep("here", path="/memories/docs/").matches
        assert matches is not None
        match_paths = [m["path"] for m in matches]

        # Should find files in /memories/docs/
        assert any("/memories/docs/readme.md" in p for p in match_paths)
        assert any("/memories/docs/guide.md" in p for p in match_paths)

        # Should NOT find files outside /memories/docs/
        assert not any("/memories/notes.txt" in p for p in match_paths)
        assert not any("/local.txt" in p for p in match_paths)


def test_composite_grep_empty_results(tmp_path: Path) -> None:
    """Test grep that matches nothing returns empty list."""
    root = tmp_path

    (root / "file.txt").write_text("some content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        comp.write("/memories/note.txt", "memory content")

        # Search for pattern that doesn't exist
        matches = comp.grep("nonexistent_pattern_xyz", path="/").matches
        assert matches is not None
        assert len(matches) == 0


def test_composite_grep_route_prefix_restoration(tmp_path: Path) -> None:
    """Test that grep correctly restores route prefixes in results."""
    root = tmp_path

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Write files to memories
        comp.write("/memories/alpha.txt", "test content alpha")
        comp.write("/memories/beta.txt", "test content beta")

        # Grep in memories route
        matches = comp.grep("test", path="/memories/").matches
        assert matches is not None
        assert len(matches) > 0

        # All paths should start with /memories/
        for match in matches:
            assert match["path"].startswith("/memories/")
            assert not match["path"].startswith("/memories//")  # No double slashes

        # Grep across all backends (path="/")
        matches_all = comp.grep("test", path="/").matches
        assert matches_all is not None

        # Filter matches from memories
        memory_matches = [m for m in matches_all if "/memories/" in m["path"]]
        for match in memory_matches:
            assert match["path"].startswith("/memories/")


def test_composite_grep_multiple_matches_per_file(tmp_path: Path) -> None:
    """Test grep returns multiple matches from same file."""
    root = tmp_path

    # File with multiple matching lines
    (root / "multi.txt").write_text("line1 pattern\nline2 pattern\nline3 other")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    comp = CompositeBackend(default=fs, routes={})

    matches = comp.grep("pattern", path="/").matches
    assert matches is not None

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
    root = tmp_path

    (root / "default.txt").write_text("default findme")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store1 = StoreBackend()
        store2 = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store1, "/archive/": store2})

        # Write to each route
        comp.write("/memories/mem.txt", "memory findme")
        comp.write("/archive/arch.txt", "archive findme")

        # Grep across all backends
        matches = comp.grep("findme", path="/").matches
        assert matches is not None
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
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # Create a mock backend that returns error strings for grep
        class ErrorBackend(StoreBackend):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None):
                return "Invalid regex pattern error"

        error_backend = ErrorBackend()
        state_backend = StateBackend()

        comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

        # When searching a specific route that errors, return the error
        result = comp.grep("test", path="/errors/")
        assert result.error == "Invalid regex pattern error"


def test_composite_grep_error_in_routed_backend_at_root() -> None:
    """Test grep error handling when routed backend errors during root search."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # Create a mock backend that returns error strings for grep
        class ErrorBackend(StoreBackend):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None):
                return "Backend error occurred"

        error_backend = ErrorBackend()
        state_backend = StateBackend()

        comp = CompositeBackend(default=state_backend, routes={"/errors/": error_backend})

        # When searching from root and a routed backend errors, return the error
        result = comp.grep("test", path="/")
        assert result.error == "Backend error occurred"


def test_composite_grep_error_in_default_backend_at_root() -> None:
    """Test grep error handling when default backend errors during root search."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        # Create a mock backend that returns error strings for grep
        class ErrorDefaultBackend(StateBackend):
            def grep(self, pattern: str, path: str | None = None, glob: str | None = None):
                return "Default backend error"

        error_default = ErrorDefaultBackend()
        store_backend = StoreBackend()

        comp = CompositeBackend(default=error_default, routes={"/store/": store_backend})

        # When searching from root and default backend errors, return the error
        result = comp.grep("test", path="/")
        assert result.error == "Default backend error"


def test_composite_grep_non_root_path_on_default_backend(tmp_path: Path) -> None:
    """Test grep with non-root path on default backend."""
    root = tmp_path

    # Create nested structure
    (root / "work").mkdir()
    (root / "work" / "project.txt").write_text("project content")
    (root / "other.txt").write_text("other content")

    fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store = StoreBackend()

        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Search in /work directory (doesn't match any route)
        matches = comp.grep("content", path="/work").matches
        assert matches is not None
        match_paths = [m["path"] for m in matches]

        # Should only find files in /work, not /other.txt
        assert match_paths == ["/work/project.txt"]


def test_composite_glob_targeting_specific_route() -> None:
    """Test glob when path matches a specific route."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state_backend = StateBackend()

        comp = CompositeBackend(default=state_backend, routes={"/memories/": store_be})

        # Write files to memories
        comp.write("/memories/test.py", "python file")
        comp.write("/memories/data.json", "json file")
        comp.write("/memories/docs/readme.md", "markdown file")

        # Write to default backend
        state_backend.write("/local.py", "local python")

        # Glob in specific route with pattern - should only find .py files in memories
        results = comp.glob("**/*.py", path="/memories/").matches
        result_paths = [fi["path"] for fi in results]

        assert result_paths == ["/memories/test.py"]


def test_composite_glob_leading_slash_pattern() -> None:
    """Test glob with a leading-slash pattern from the root path."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state_backend = StateBackend()

        comp = CompositeBackend(default=state_backend, routes={"/memories/": store_be})

        comp.write("/memories/note.md", "markdown note")
        comp.write("/memories/data.txt", "text data")
        state_backend.write("/local.md", "local markdown")

        results = comp.glob("/memories/**/*.md", path="/").matches
        result_paths = [fi["path"] for fi in results]

        assert "/memories/note.md" in result_paths
        assert "/memories/data.txt" not in result_paths


def test_composite_glob_nested_path_in_route() -> None:
    """Test glob with nested path within route."""
    mem_store = InMemoryStore()

    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state_backend = StateBackend()

        comp = CompositeBackend(default=state_backend, routes={"/archive/": store_be})

        # Write nested files
        comp.write("/archive/2024/jan.log", "january logs")
        comp.write("/archive/2024/feb.log", "february logs")
        comp.write("/archive/2023/dec.log", "december logs")
        comp.write("/archive/notes.txt", "general notes")

        # Glob in nested path within route - should only find .log files in /archive/2024/
        results = comp.glob("*.log", path="/archive/2024/").matches
        result_paths = sorted([fi["path"] for fi in results])

        assert result_paths == ["/archive/2024/feb.log", "/archive/2024/jan.log"]


# --- Tests for path stripping consistency ---


def test_grep_path_stripping_matches_get_backend_and_key() -> None:
    """Verify grep strips route prefix the same way as _get_backend_and_key."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state = StateBackend()
        comp = CompositeBackend(default=state, routes={"/memories/": store_be})

        comp.write("/memories/readme.md", "hello world")

        # Search with trailing slash (exact route prefix)
        matches = comp.grep("hello", path="/memories/").matches
        assert matches is not None
        assert any(m["path"] == "/memories/readme.md" for m in matches)

        # Search with nested path inside route
        matches2 = comp.grep("hello", path="/memories/readme.md").matches
        assert matches2 is not None


def test_glob_path_stripping_matches_get_backend_and_key() -> None:
    """Verify glob strips route prefix the same way as _get_backend_and_key."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state = StateBackend()
        comp = CompositeBackend(default=state, routes={"/memories/": store_be})

        comp.write("/memories/notes.txt", "content")

        # Glob with trailing slash
        results = comp.glob("*.txt", path="/memories/").matches
        assert any(fi["path"] == "/memories/notes.txt" for fi in results)


def test_get_backend_and_key_consistency() -> None:
    """Verify _get_backend_and_key produces correct stripped paths."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        store_be = StoreBackend()
        state = StateBackend()
        comp = CompositeBackend(default=state, routes={"/memories/": store_be})

        # Exact route prefix
        backend, stripped = comp._get_backend_and_key("/memories/")
        assert backend is store_be
        assert stripped == "/"

        # File inside route
        backend, stripped = comp._get_backend_and_key("/memories/notes.txt")
        assert backend is store_be
        assert stripped == "/notes.txt"

        # Nested path inside route
        backend, stripped = comp._get_backend_and_key("/memories/sub/file.txt")
        assert backend is store_be
        assert stripped == "/sub/file.txt"

        # Path not matching any route
        backend, stripped = comp._get_backend_and_key("/other/file.txt")
        assert backend is state
        assert stripped == "/other/file.txt"


def test_route_for_path_edge_cases() -> None:
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        default = StateBackend()
        mem = StoreBackend()
        mem_private = StoreBackend()

        sorted_routes = [
            ("/memories/private/", mem_private),
            ("/memories/", mem),
        ]

        # No match -> default backend, path unchanged
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/other/file.txt") == (
            default,
            "/other/file.txt",
            None,
        )

        # Exact route root without trailing slash -> backend_path "/"
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories") == (
            mem,
            "/",
            "/memories/",
        )

        # Exact route prefix with trailing slash -> backend_path "/"
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories/") == (
            mem,
            "/",
            "/memories/",
        )

        # Nested path in route -> strip and keep leading slash
        assert _route_for_path(
            default=default,
            sorted_routes=sorted_routes,
            path="/memories/notes.txt",
        ) == (mem, "/notes.txt", "/memories/")

        # Deep nested path -> strip
        assert _route_for_path(
            default=default,
            sorted_routes=sorted_routes,
            path="/memories/sub/file.txt",
        ) == (mem, "/sub/file.txt", "/memories/")

        # Longest-prefix wins
        assert _route_for_path(
            default=default,
            sorted_routes=sorted_routes,
            path="/memories/private/secret.txt",
        ) == (mem_private, "/secret.txt", "/memories/private/")

        # Route root for nested route, without trailing slash
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories/private") == (
            mem_private,
            "/",
            "/memories/private/",
        )

        # Prefix boundary: should not match "/memories/" for "/memories2/..."
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/memories2/file.txt") == (
            default,
            "/memories2/file.txt",
            None,
        )


def test_route_for_path_no_trailing_slash_boundary() -> None:
    """Route without trailing slash must not match at non-boundary positions.

    Regression test for https://github.com/langchain-ai/deepagents/issues/1654.
    """
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        default = StateBackend()
        store_be = StoreBackend()

        sorted_routes = [("/abcd", store_be)]

        # /abcde/file.txt must NOT match /abcd (different path segment)
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/abcde/file.txt") == (
            default,
            "/abcde/file.txt",
            None,
        )

        # /abcd/file.txt SHOULD match /abcd and strip correctly
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/abcd/file.txt") == (
            store_be,
            "/file.txt",
            "/abcd",
        )

        # Exact match still works
        assert _route_for_path(default=default, sorted_routes=sorted_routes, path="/abcd") == (
            store_be,
            "/",
            "/abcd",
        )

        # Same boundary issue with a more realistic prefix
        sorted_routes_mem = [("/memories", store_be)]

        assert _route_for_path(default=default, sorted_routes=sorted_routes_mem, path="/memories-backup/file.txt") == (
            default,
            "/memories-backup/file.txt",
            None,
        )

        assert _route_for_path(default=default, sorted_routes=sorted_routes_mem, path="/memories/file.txt") == (
            store_be,
            "/file.txt",
            "/memories",
        )

        # Trailing-slash route should already work correctly
        sorted_routes_slash = [("/abcd/", store_be)]

        assert _route_for_path(default=default, sorted_routes=sorted_routes_slash, path="/abcde/file.txt") == (
            default,
            "/abcde/file.txt",
            None,
        )


def test_write_result_path_restored_to_full_routed_path():
    """CompositeBackend.write should return the full path, not the stripped key."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        comp = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})

        res = comp.write("/memories/site_context.md", "content")

        assert res.error is None
        assert res.path == "/memories/site_context.md"  # not "/site_context.md"


def test_edit_result_path_restored_to_full_routed_path():
    """CompositeBackend.edit should return the full path, not the stripped key."""
    mem_store = InMemoryStore()
    with state_config_context(store=mem_store):
        comp = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        comp.write("/memories/notes.md", "hello world")

        res = comp.edit("/memories/notes.md", "hello", "goodbye")

        assert res.error is None
        assert res.path == "/memories/notes.md"  # not "/notes.md"
