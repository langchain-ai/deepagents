"""Observable Deep Agents sandbox behavior with a fake Smol VM."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import smol

from langchain_smol import SmolSandbox


@pytest.fixture
def machine() -> Mock:
    """A running VM with a small directory and realistic file errors."""
    files: dict[str, bytes] = {}
    vm = Mock(id="mach-123")

    def execute(_argv: list[str], _opts: smol.ExecOptions) -> SimpleNamespace:
        return SimpleNamespace(
            stdout="done\n",
            stderr="warning\n",
            exit_code=0,
            stdout_truncated=False,
            stderr_truncated=True,
        )

    def read_file(path: str) -> bytes:
        if path not in files:
            msg = "not found"
            code = "NOT_FOUND"
            raise smol.SmolError(code, msg)
        return files[path]

    def write_file(path: str, content: bytes) -> None:
        files[path] = content

    vm.exec.side_effect = execute
    vm.read_file.side_effect = read_file
    vm.write_file.side_effect = write_file
    return vm


def test_command_preserves_exit_status_and_truncation(machine: Mock) -> None:
    """Results preserve both output streams and report a truncated stream."""
    sandbox = SmolSandbox(machine=machine)
    deadline = 10
    result = sandbox.execute("echo done", timeout=deadline)
    assert result.output == "done\n\n<stderr>warning\n</stderr>"
    assert result.exit_code == 0
    assert result.truncated is True
    argv, opts = machine.exec.call_args.args
    assert argv == ["sh", "-lc", "echo done"]
    assert opts.timeout == deadline
    assert sandbox.id == "mach-123"
    machine.delete.assert_not_called()


def test_binary_file_roundtrip_and_missing_file(machine: Mock) -> None:
    """File transfers preserve bytes and correctly classify absent paths."""
    sandbox = SmolSandbox(machine=machine)
    assert sandbox.upload_files([("/workspace/data", b"\x00\xff")])[0].error is None
    assert sandbox.download_files(["/workspace/data"])[0].content == b"\x00\xff"
    assert sandbox.download_files(["/workspace/missing"])[0].error == "file_not_found"
    assert sandbox.download_files(["relative"])[0].error == "invalid_path"
    assert sandbox.upload_files([("relative", b"bad")])[0].error == "invalid_path"


def test_partial_failure_reports_errors_without_losing_other_files(
    machine: Mock,
) -> None:
    """A lost transfer reports its cause while another transfer still succeeds."""
    machine.write_file.side_effect = [smol.SmolError("CONNECTION", "offline"), None]
    results = SmolSandbox(machine=machine).upload_files(
        [("/workspace/first", b"a"), ("/workspace/second", b"b")]
    )
    assert [response.error for response in results] == ["connection", None]
    machine.read_file.side_effect = [
        smol.SmolError("SMOLVM_ERROR", "No such file or directory (os error 2)"),
        b"present",
    ]
    reads = SmolSandbox(machine=machine).download_files(
        ["/workspace/missing", "/workspace/present"]
    )
    assert [response.error for response in reads] == ["file_not_found", None]
    assert reads[1].content == b"present"


def test_sdk_rejects_nonpositive_timeout_before_running(machine: Mock) -> None:
    """An invalid timeout must not start a command with an unbounded deadline."""
    with pytest.raises(ValueError, match="greater than zero"):
        SmolSandbox(machine=machine, timeout=0)
    with pytest.raises(ValueError, match="greater than zero"):
        SmolSandbox(machine=machine).execute("sleep 1", timeout=0)
    machine.exec.assert_not_called()
