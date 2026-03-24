"""Tests for the delete method across all backend implementations."""

import json
import tempfile
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import (
    DeleteResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend

# -- helpers ------------------------------------------------------------------


def write_file(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def make_state_runtime(files=None):
    return ToolRuntime(
        state={"messages": [], "files": files or {}},
        context=None,
        tool_call_id="t1",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def make_store_runtime():
    return ToolRuntime(
        state={"messages": []},
        context=None,
        tool_call_id="t2",
        store=InMemoryStore(),
        stream_writer=lambda _: None,
        config={},
    )


# -- FilesystemBackend tests --------------------------------------------------


class TestFilesystemBackendDelete:
    def test_delete_file(self, tmp_path: Path) -> None:
        """Test deleting a file succeeds."""
        f = tmp_path / "a.txt"
        write_file(f, "hello")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/a.txt")

        assert isinstance(result, DeleteResult)
        assert result.error is None
        assert result.path == "/a.txt"
        assert result.files_update is None
        assert not f.exists()

    def test_delete_file_normal_mode(self, tmp_path: Path) -> None:
        """Test deleting a file in non-virtual mode."""
        f = tmp_path / "b.txt"
        write_file(f, "content")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        result = be.delete(str(f))

        assert result.error is None
        assert result.path == str(f)
        assert not f.exists()

    def test_delete_nonexistent_file(self, tmp_path: Path) -> None:
        """Test deleting a nonexistent file returns error."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/nonexistent.txt")

        assert result.error is not None
        assert "not found" in result.error

    def test_delete_empty_directory(self, tmp_path: Path) -> None:
        """Test deleting an empty directory succeeds."""
        d = tmp_path / "emptydir"
        d.mkdir()

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/emptydir")

        assert result.error is None
        assert result.path == "/emptydir"
        assert not d.exists()

    def test_delete_nonempty_directory(self, tmp_path: Path) -> None:
        """Test deleting a non-empty directory returns error."""
        d = tmp_path / "fulldir"
        d.mkdir()
        (d / "child.txt").write_text("content")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/fulldir")

        assert result.error is not None
        assert d.exists()

    def test_delete_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Test that path traversal is blocked in virtual mode."""
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        with pytest.raises(ValueError, match="traversal"):
            be.delete("/../secret.txt")

    def test_delete_then_write(self, tmp_path: Path) -> None:
        """Test that a deleted file can be recreated."""
        f = tmp_path / "reuse.txt"
        write_file(f, "original")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = be.delete("/reuse.txt")
        assert result.error is None

        result2 = be.write("/reuse.txt", "new content")
        assert result2.error is None
        assert f.read_text() == "new content"


# -- StateBackend tests -------------------------------------------------------


class TestStateBackendDelete:
    def test_delete_file(self) -> None:
        """Test deleting a file from state."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        # Write a file first
        res = be.write("/notes.txt", "hello")
        assert res.error is None
        rt.state["files"].update(res.files_update)

        # Delete
        result = be.delete("/notes.txt")
        assert isinstance(result, DeleteResult)
        assert result.error is None
        assert result.path == "/notes.txt"
        assert result.files_update is not None
        assert result.files_update["/notes.txt"] is None

    def test_delete_nonexistent_file(self) -> None:
        """Test deleting a nonexistent file returns error."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        result = be.delete("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error

    def test_delete_then_read(self) -> None:
        """Test that a deleted file cannot be read."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        res = be.write("/data.txt", "content")
        rt.state["files"].update(res.files_update)

        del_res = be.delete("/data.txt")
        assert del_res.error is None
        # Apply the deletion to state
        for k, v in del_res.files_update.items():
            if v is None:
                rt.state["files"].pop(k, None)

        read_res = be.read("/data.txt")
        assert read_res.error is not None
        assert "not found" in read_res.error

    def test_delete_then_write(self) -> None:
        """Test that a deleted file can be recreated."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        res = be.write("/reuse.txt", "original")
        rt.state["files"].update(res.files_update)

        del_res = be.delete("/reuse.txt")
        for k, v in del_res.files_update.items():
            if v is None:
                rt.state["files"].pop(k, None)

        res2 = be.write("/reuse.txt", "new content")
        assert res2.error is None
        rt.state["files"].update(res2.files_update)

        read_res = be.read("/reuse.txt")
        assert read_res.file_data is not None
        assert "new content" in read_res.file_data["content"]

    def test_delete_not_in_ls(self) -> None:
        """Test that a deleted file no longer shows up in ls."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        res = be.write("/visible.txt", "content")
        rt.state["files"].update(res.files_update)

        del_res = be.delete("/visible.txt")
        for k, v in del_res.files_update.items():
            if v is None:
                rt.state["files"].pop(k, None)

        listing = be.ls("/").entries
        assert listing is not None
        assert not any(fi["path"] == "/visible.txt" for fi in listing)


# -- StoreBackend tests -------------------------------------------------------


class TestStoreBackendDelete:
    def test_delete_file(self) -> None:
        """Test deleting a file from the store."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/doc.md", "hello store")
        result = be.delete("/doc.md")

        assert isinstance(result, DeleteResult)
        assert result.error is None
        assert result.path == "/doc.md"
        assert result.files_update is None

    def test_delete_nonexistent_file(self) -> None:
        """Test deleting a nonexistent file returns error."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        result = be.delete("/missing.md")
        assert result.error is not None
        assert "not found" in result.error

    def test_delete_then_read(self) -> None:
        """Test that a deleted file cannot be read."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/data.txt", "content")
        be.delete("/data.txt")

        read_res = be.read("/data.txt")
        assert read_res.error is not None
        assert "not found" in read_res.error

    def test_delete_then_write(self) -> None:
        """Test that a deleted file can be recreated."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/reuse.txt", "original")
        be.delete("/reuse.txt")

        res = be.write("/reuse.txt", "new content")
        assert res.error is None

        read_res = be.read("/reuse.txt")
        assert read_res.file_data is not None
        assert "new content" in read_res.file_data["content"]

    def test_delete_not_in_ls(self) -> None:
        """Test that a deleted file no longer shows up in ls."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/visible.txt", "content")
        be.delete("/visible.txt")

        listing = be.ls("/").entries
        assert listing is not None
        assert not any(fi["path"] == "/visible.txt" for fi in listing)

    def test_delete_not_in_grep(self) -> None:
        """Test that a deleted file doesn't appear in grep results."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/searchable.txt", "findme")
        be.delete("/searchable.txt")

        matches = be.grep("findme", path="/").matches
        assert matches is not None
        assert len(matches) == 0


# -- BaseSandbox tests --------------------------------------------------------


class MockSandbox(BaseSandbox):
    """Minimal concrete implementation of BaseSandbox for testing."""

    def __init__(self) -> None:
        self.last_command: str | None = None
        self._next_output: str = "1"
        self._uploaded: list[tuple[str, bytes]] = []
        self._file_store: dict[str, bytes] = {}

    @property
    def id(self) -> str:
        return "mock-sandbox"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.last_command = command
        output = self._next_output
        self._next_output = "1"
        return ExecuteResponse(output=output, exit_code=0, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        self._uploaded.extend(files)
        for path, content in files:
            self._file_store[path] = content
        return [FileUploadResponse(path=f[0], error=None) for f in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for p in paths:
            if p in self._file_store:
                results.append(FileDownloadResponse(path=p, content=self._file_store[p], error=None))
            else:
                results.append(FileDownloadResponse(path=p, content=None, error="file_not_found"))
        return results


class TestSandboxBackendDelete:
    def test_delete_success(self) -> None:
        """Test delete returns success when server script succeeds."""
        sandbox = MockSandbox()
        sandbox._next_output = json.dumps({"ok": True})

        result = sandbox.delete("/test/file.txt")

        assert result.error is None
        assert result.path == "/test/file.txt"
        assert result.files_update is None

    def test_delete_file_not_found(self) -> None:
        """Test delete returns error when file doesn't exist."""
        sandbox = MockSandbox()
        sandbox._next_output = json.dumps({"error": "file_not_found"})

        result = sandbox.delete("/test/missing.txt")

        assert result.error is not None
        assert "not found" in result.error

    def test_delete_os_error(self) -> None:
        """Test delete returns error from OS failure."""
        sandbox = MockSandbox()
        sandbox._next_output = json.dumps({"error": "[Errno 13] Permission denied: '/test/protected.txt'"})

        result = sandbox.delete("/test/protected.txt")

        assert result.error is not None
        assert "Permission denied" in result.error

    def test_delete_malformed_output(self) -> None:
        """Test delete handles non-JSON output gracefully."""
        sandbox = MockSandbox()
        sandbox._next_output = "not json at all"

        result = sandbox.delete("/test/file.txt")

        assert result.error is not None
        assert "unexpected server response" in result.error

    def test_delete_non_dict_json_output(self) -> None:
        """Test delete handles non-dict JSON output."""
        sandbox = MockSandbox()
        sandbox._next_output = "[1, 2, 3]"

        result = sandbox.delete("/test/file.txt")

        assert result.error is not None
        assert "unexpected server response" in result.error

    def test_delete_uses_execute(self) -> None:
        """Test that delete delegates to execute()."""
        sandbox = MockSandbox()
        sandbox._next_output = json.dumps({"ok": True})

        sandbox.delete("/test/file.txt")

        assert sandbox.last_command is not None
        assert "python3 -c" in sandbox.last_command


# -- LocalShellBackend tests --------------------------------------------------


class TestLocalShellBackendDelete:
    def test_delete_file(self) -> None:
        """Test deleting a file via LocalShellBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "file.txt"
            f.write_text("content")

            backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)
            result = backend.delete("/file.txt")

            assert result.error is None
            assert result.path == "/file.txt"
            assert not f.exists()

    def test_delete_nonexistent(self) -> None:
        """Test deleting nonexistent file via LocalShellBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)
            result = backend.delete("/missing.txt")

            assert result.error is not None
            assert "not found" in result.error


# -- CompositeBackend tests ---------------------------------------------------


class TestCompositeBackendDelete:
    def test_delete_default_backend(self, tmp_path: Path) -> None:
        """Test delete routes to default backend."""
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="tc",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )
        state = StateBackend(rt)
        store = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))
        comp = CompositeBackend(default=state, routes={"/memories/": store})

        # Write to default
        res = comp.write("/file.txt", "alpha")
        assert res.files_update is not None
        rt.state["files"].update(res.files_update)

        # Delete from default
        del_res = comp.delete("/file.txt")
        assert del_res.error is None
        assert del_res.path == "/file.txt"

    def test_delete_routed_backend(self, tmp_path: Path) -> None:
        """Test delete routes to the correct routed backend."""
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="tc",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )
        state = StateBackend(rt)
        store = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))
        comp = CompositeBackend(default=state, routes={"/memories/": store})

        # Write to routed
        comp.write("/memories/note.md", "beta")

        # Delete from routed
        del_res = comp.delete("/memories/note.md")
        assert del_res.error is None
        assert del_res.path == "/memories/note.md"

        # Verify it's gone
        read_res = comp.read("/memories/note.md")
        assert read_res.error is not None

    def test_delete_filesystem_routed(self, tmp_path: Path) -> None:
        """Test delete with filesystem backend in composite."""
        root = tmp_path
        write_file(root / "hello.txt", "hello")

        fs = FilesystemBackend(root_dir=str(root), virtual_mode=True)
        rt = make_store_runtime()
        store = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))
        comp = CompositeBackend(default=fs, routes={"/memories/": store})

        # Write to store
        comp.write("/memories/note.md", "note")

        # Delete from filesystem (default)
        del_res = comp.delete("/hello.txt")
        assert del_res.error is None
        assert not (root / "hello.txt").exists()

        # Delete from store (routed)
        del_res2 = comp.delete("/memories/note.md")
        assert del_res2.error is None

    def test_delete_nonexistent_routes_correctly(self) -> None:
        """Test delete of nonexistent file returns error from correct backend."""
        rt = ToolRuntime(
            state={"messages": [], "files": {}},
            context=None,
            tool_call_id="tc",
            store=InMemoryStore(),
            stream_writer=lambda _: None,
            config={},
        )
        state = StateBackend(rt)
        store = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))
        comp = CompositeBackend(default=state, routes={"/memories/": store})

        # Delete nonexistent from default
        result = comp.delete("/ghost.txt")
        assert result.error is not None
        assert "not found" in result.error

        # Delete nonexistent from routed
        result2 = comp.delete("/memories/ghost.md")
        assert result2.error is not None
        assert "not found" in result2.error


# -- Async tests --------------------------------------------------------------


@pytest.mark.asyncio
class TestAsyncDelete:
    async def test_filesystem_adelete(self, tmp_path: Path) -> None:
        """Test async delete on filesystem backend."""
        f = tmp_path / "async.txt"
        write_file(f, "async content")

        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        result = await be.adelete("/async.txt")

        assert result.error is None
        assert result.path == "/async.txt"
        assert not f.exists()

    async def test_state_adelete(self) -> None:
        """Test async delete on state backend."""
        rt = make_state_runtime()
        be = StateBackend(rt)

        res = be.write("/async.txt", "content")
        rt.state["files"].update(res.files_update)

        result = await be.adelete("/async.txt")
        assert result.error is None
        assert result.files_update is not None
        assert result.files_update["/async.txt"] is None

    async def test_store_adelete(self) -> None:
        """Test async delete on store backend."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        be.write("/async.txt", "content")
        result = await be.adelete("/async.txt")

        assert result.error is None
        assert result.path == "/async.txt"

        # Verify it's gone
        read_res = await be.aread("/async.txt")
        assert read_res.error is not None

    async def test_store_adelete_nonexistent(self) -> None:
        """Test async delete of nonexistent file in store."""
        rt = make_store_runtime()
        be = StoreBackend(rt, namespace=lambda _ctx: ("filesystem",))

        result = await be.adelete("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error
