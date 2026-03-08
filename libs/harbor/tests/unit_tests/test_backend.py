"""Unit tests for HarborSandbox backend.

These tests verify all file operations (read, write, edit, ls, grep, glob,
download_files) and command execution without requiring a real Harbor
environment.  A lightweight ``MockEnvironment`` stubs the
``BaseEnvironment.exec`` method, capturing commands and returning
configurable results.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import pytest

from deepagents_harbor.backend import HarborSandbox

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _ExecResult:
    """Minimal stand-in for ``harbor.environments.base.ExecResult``."""

    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class MockEnvironment:
    """Fake ``BaseEnvironment`` that records commands and replays responses.

    Attributes:
        session_id: Value returned by ``HarborSandbox.id``.
        commands: Captured command strings in execution order.
        responses: FIFO queue of ``_ExecResult`` to return from ``exec``.
            If exhausted, a default success result is returned.
    """

    def __init__(
        self,
        *,
        session_id: str = "test-session",
        responses: list[_ExecResult] | None = None,
    ) -> None:
        self.session_id = session_id
        self.commands: list[str] = []
        self._responses: list[_ExecResult] = list(responses or [])
        self._call_idx = 0

    async def exec(
        self,
        command: str,
        timeout_sec: int | None = None,  # noqa: ARG002
    ) -> _ExecResult:
        self.commands.append(command)
        if self._call_idx < len(self._responses):
            result = self._responses[self._call_idx]
            self._call_idx += 1
            return result
        return _ExecResult()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> MockEnvironment:
    return MockEnvironment()


@pytest.fixture
def sandbox(env: MockEnvironment) -> HarborSandbox:
    return HarborSandbox(env)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# aexecute / id
# ---------------------------------------------------------------------------


class TestExecute:
    async def test_id_returns_session_id(self, sandbox: HarborSandbox) -> None:
        assert sandbox.id == "test-session"

    async def test_aexecute_captures_command(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="hello")]
        result = await sandbox.aexecute("echo hello")
        assert result.output == "hello"
        assert result.exit_code == 0
        assert env.commands == ["echo hello"]

    async def test_aexecute_filters_bash_noise(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [
            _ExecResult(
                stdout="bash: cannot set terminal process group (-1): Inappropriate ioctl for device\nok",
                stderr="bash: no job control in this shell",
            ),
        ]
        result = await sandbox.aexecute("true")
        assert "ok" in result.output
        # The filtered messages should appear in stderr portion
        assert (
            "Inappropriate ioctl" not in result.output.split("\n stderr: ")[0]
            if "\n stderr: " in result.output
            else True
        )

    async def test_aexecute_nonzero_exit(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", stderr="not found", return_code=1)]
        result = await sandbox.aexecute("false")
        assert result.exit_code == 1
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# aread
# ---------------------------------------------------------------------------


class TestRead:
    async def test_read_success(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="     1\thello\n     2\tworld")]
        content = await sandbox.aread("/test.txt")
        assert "hello" in content
        assert "world" in content

    async def test_read_file_not_found(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="Error: File not found", return_code=1)]
        result = await sandbox.aread("/missing.txt")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# awrite
# ---------------------------------------------------------------------------


class TestWrite:
    async def test_write_success(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=0)]
        result = await sandbox.awrite("/new.txt", "content")
        assert result.error is None
        assert result.path == "/new.txt"
        # Verify base64 encoding was used
        assert len(env.commands) == 1
        assert "base64 -d" in env.commands[0]

    async def test_write_existing_file_errors(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [
            _ExecResult(
                stdout="Error: File '/new.txt' already exists",
                return_code=1,
            ),
        ]
        result = await sandbox.awrite("/new.txt", "oops")
        assert result.error is not None
        assert "already exists" in result.error


# ---------------------------------------------------------------------------
# aedit
# ---------------------------------------------------------------------------


class TestEdit:
    async def test_edit_single_replacement(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="1", return_code=0)]
        result = await sandbox.aedit("/f.py", "old", "new")
        assert result.error is None
        assert result.occurrences == 1

    async def test_edit_file_not_found(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=3)]
        result = await sandbox.aedit("/missing.py", "old", "new")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_string_not_found(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=1)]
        result = await sandbox.aedit("/f.py", "absent", "new")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_multiple_matches_no_replace_all(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=2)]
        result = await sandbox.aedit("/f.py", "dup", "new")
        assert result.error is not None
        assert "multiple" in result.error.lower()


# ---------------------------------------------------------------------------
# als_info
# ---------------------------------------------------------------------------


class TestLs:
    async def test_ls_returns_files_and_dirs(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="file.txt|false\nsubdir|true")]
        infos = await sandbox.als_info("/workspace")
        assert len(infos) == 2
        paths = {i["path"] for i in infos}
        assert "file.txt" in paths
        assert "subdir" in paths
        dirs = [i for i in infos if i.get("is_dir")]
        assert len(dirs) == 1

    async def test_ls_empty_dir(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=0)]
        infos = await sandbox.als_info("/empty")
        assert infos == []

    async def test_ls_nonexistent_dir(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=1)]
        infos = await sandbox.als_info("/nope")
        assert infos == []


# ---------------------------------------------------------------------------
# agrep_raw
# ---------------------------------------------------------------------------


class TestGrep:
    async def test_grep_returns_matches(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [
            _ExecResult(stdout="/a.py:10:def foo():\n/b.py:5:# foo"),
        ]
        matches = await sandbox.agrep_raw("foo", path="/workspace")
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert matches[0]["path"] == "/a.py"
        assert matches[0]["line"] == 10

    async def test_grep_no_matches(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="")]
        matches = await sandbox.agrep_raw("nonexistent")
        assert matches == []

    async def test_grep_uses_fixed_string_flag(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        """Grep must use -F for literal search, not regex interpretation."""
        env._responses = [_ExecResult(stdout="")]
        await sandbox.agrep_raw("def __init__(self):", path="/src")
        assert len(env.commands) == 1
        cmd = env.commands[0]
        # -F can appear combined (e.g., -rHnF)
        assert "-rHnF" in cmd, f"Expected -rHnF in grep command, got: {cmd}"

    async def test_grep_with_glob_filter(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="/a.py:1:match")]
        matches = await sandbox.agrep_raw("match", path="/src", glob="*.py")
        assert len(env.commands) == 1
        assert "--include=" in env.commands[0]
        assert isinstance(matches, list)
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# aglob_info
# ---------------------------------------------------------------------------


class TestGlob:
    async def test_glob_returns_file_infos(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="main.py|false\nsrc|true")]
        infos = await sandbox.aglob_info("*", path="/workspace")
        assert len(infos) == 2

    async def test_glob_empty_result(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=0)]
        infos = await sandbox.aglob_info("*.rs", path="/workspace")
        assert infos == []


# ---------------------------------------------------------------------------
# adownload_files
# ---------------------------------------------------------------------------


class TestDownloadFiles:
    async def test_download_single_file(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        content = b"file content here"
        encoded = base64.b64encode(content).decode("ascii")
        env._responses = [_ExecResult(stdout=encoded, return_code=0)]
        responses = await sandbox.adownload_files(["/app/data.txt"])
        assert len(responses) == 1
        assert responses[0].path == "/app/data.txt"
        assert responses[0].error is None
        assert responses[0].content == content

    async def test_download_file_not_found(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=1)]
        responses = await sandbox.adownload_files(["/missing.txt"])
        assert responses[0].error == "file_not_found"
        assert responses[0].content is None

    async def test_download_directory_error(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        env._responses = [_ExecResult(stdout="", return_code=2)]
        responses = await sandbox.adownload_files(["/some_dir"])
        assert responses[0].error == "is_directory"

    async def test_download_multiple_files(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        c1 = base64.b64encode(b"aaa").decode("ascii")
        c2 = base64.b64encode(b"bbb").decode("ascii")
        env._responses = [
            _ExecResult(stdout=c1, return_code=0),
            _ExecResult(stdout=c2, return_code=0),
        ]
        responses = await sandbox.adownload_files(["/a.txt", "/b.txt"])
        assert len(responses) == 2
        assert responses[0].content == b"aaa"
        assert responses[1].content == b"bbb"

    async def test_download_binary_content(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        binary = bytes(range(256))
        encoded = base64.b64encode(binary).decode("ascii")
        env._responses = [_ExecResult(stdout=encoded, return_code=0)]
        responses = await sandbox.adownload_files(["/bin.dat"])
        assert responses[0].content == binary

    async def test_download_partial_failure(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        ok_content = base64.b64encode(b"ok").decode("ascii")
        env._responses = [
            _ExecResult(stdout=ok_content, return_code=0),
            _ExecResult(stdout="", return_code=1),
        ]
        responses = await sandbox.adownload_files(["/ok.txt", "/missing.txt"])
        assert responses[0].error is None
        assert responses[0].content == b"ok"
        assert responses[1].error == "file_not_found"

    async def test_download_strips_stderr_noise(
        self,
        env: MockEnvironment,
        sandbox: HarborSandbox,
    ) -> None:
        """Stderr appended by aexecute must be stripped before base64 decoding."""
        content = b"payload"
        encoded = base64.b64encode(content).decode("ascii")
        env._responses = [_ExecResult(stdout=encoded, stderr="some warning", return_code=0)]
        responses = await sandbox.adownload_files(["/f.txt"])
        assert responses[0].error is None
        assert responses[0].content == content


# ---------------------------------------------------------------------------
# Sync methods raise NotImplementedError
# ---------------------------------------------------------------------------


class TestSyncNotSupported:
    def test_execute_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.execute("echo hi")

    def test_read_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.read("/f.txt")

    def test_write_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.write("/f.txt", "c")

    def test_edit_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.edit("/f.txt", "a", "b")

    def test_ls_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.ls_info("/")

    def test_grep_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.grep_raw("pat")

    def test_glob_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.glob_info("*.py")

    def test_download_sync_raises(self, sandbox: HarborSandbox) -> None:
        with pytest.raises(NotImplementedError):
            sandbox.download_files(["/f.txt"])
