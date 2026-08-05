"""Unit tests for CloudRunSandbox class."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_cloud_run.sandbox import CloudRunSandbox


def test_sandbox_initialization() -> None:
    """Test default and custom initialization parameters."""
    sandbox_default = CloudRunSandbox()
    assert sandbox_default.allow_egress is False
    assert sandbox_default.sandbox_bin == "/usr/local/gcp/bin/sandbox"
    assert sandbox_default.default_timeout == 1800
    assert sandbox_default.env == {}
    assert sandbox_default.workdir is None
    assert sandbox_default.extra_sandbox_args == []
    assert sandbox_default.id.startswith("cloud-run-sandbox-")

    sandbox_custom = CloudRunSandbox(
        allow_egress=True,
        sandbox_bin="/custom/bin/sandbox",
        default_timeout=60,
        env={"FOO": "BAR"},
        workdir="/workspace",
        extra_sandbox_args=["--write"],
    )
    assert sandbox_custom.allow_egress is True
    assert sandbox_custom.sandbox_bin == "/custom/bin/sandbox"
    assert sandbox_custom.default_timeout == 60
    assert sandbox_custom.env == {"FOO": "BAR"}
    assert sandbox_custom.workdir == "/workspace"
    assert sandbox_custom.extra_sandbox_args == ["--write"]


@patch("subprocess.run")
def test_execute_default_no_egress(mock_run: MagicMock) -> None:
    """Test execute command shape when allow_egress is False."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Hello World\n",
        stderr="",
    )

    sandbox = CloudRunSandbox(allow_egress=False)
    res = sandbox.execute("echo 'Hello World'")

    mock_run.assert_called_once_with(
        [
            "/usr/local/gcp/bin/sandbox",
            "do",
            "-e",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "sh",
            "-c",
            "echo 'Hello World'",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert res.exit_code == 0
    assert res.output == "Hello World\n"
    assert res.truncated is False


@patch("subprocess.run")
def test_execute_allow_egress(mock_run: MagicMock) -> None:
    """Test execute command shape when allow_egress is True."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Network Output",
        stderr="",
    )

    sandbox = CloudRunSandbox(allow_egress=True)
    res = sandbox.execute("curl https://example.com", timeout=30)

    mock_run.assert_called_once_with(
        [
            "/usr/local/gcp/bin/sandbox",
            "do",
            "--allow-egress",
            "-e",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "sh",
            "-c",
            "curl https://example.com",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert res.exit_code == 0
    assert res.output == "Network Output"


@patch("subprocess.run")
def test_execute_env_and_workdir(mock_run: MagicMock) -> None:
    """Test execute command shape with env vars and workdir."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Output",
        stderr="",
    )

    sandbox = CloudRunSandbox(
        env={"GLOBAL_KEY": "GLOBAL_VAL"},
        workdir="/workspace",
    )
    res = sandbox.execute("pwd", env={"EXEC_KEY": "EXEC_VAL"}, workdir="/var/workspace")

    mock_run.assert_called_once_with(
        [
            "/usr/local/gcp/bin/sandbox",
            "do",
            "--workdir",
            "/var/workspace",
            "-e",
            "GLOBAL_KEY=GLOBAL_VAL",
            "-e",
            "EXEC_KEY=EXEC_VAL",
            "-e",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "sh",
            "-c",
            "pwd",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert res.exit_code == 0
    assert res.output == "Output"


@patch("subprocess.run")
def test_execute_list_command_and_extra_args(mock_run: MagicMock) -> None:
    """Test execute with list command and extra_sandbox_args."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="List Output",
        stderr="",
    )

    sandbox = CloudRunSandbox(
        extra_sandbox_args=["--write"],
    )
    res = sandbox.execute(["python3", "script.py", "arg1"])

    mock_run.assert_called_once_with(
        [
            "/usr/local/gcp/bin/sandbox",
            "do",
            "--write",
            "-e",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "python3",
            "script.py",
            "arg1",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert res.exit_code == 0
    assert res.output == "List Output"


@patch("subprocess.run")
def test_execute_with_stderr(mock_run: MagicMock) -> None:
    """Test stdout and stderr formatting."""
    mock_run.return_value = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="some stdout\n",
        stderr="some warning\n",
    )

    sandbox = CloudRunSandbox()
    res = sandbox.execute("ls non_existent")

    assert res.exit_code == 1
    assert "some stdout\n" in res.output
    assert (
        "<stderr>\nsome warning\n\n</stderr>" in res.output
        or "<stderr>\nsome warning\n</stderr>" in res.output
    )


@patch("subprocess.run")
def test_execute_binary_not_found(mock_run: MagicMock) -> None:
    """Test handling when sandbox binary does not exist."""
    mock_run.side_effect = FileNotFoundError()

    sandbox = CloudRunSandbox()
    res = sandbox.execute("ls")

    assert res.exit_code == 127
    assert (
        "Cloud Run sandbox binary '/usr/local/gcp/bin/sandbox' not found" in res.output
    )


@patch("subprocess.run")
def test_execute_timeout_expired(mock_run: MagicMock) -> None:
    """Test handling of subprocess TimeoutExpired."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="sleep 10", timeout=5)

    sandbox = CloudRunSandbox(default_timeout=5)
    res = sandbox.execute("sleep 10")

    assert res.exit_code == 124
    assert "Command execution timed out after 5 seconds" in res.output


def test_upload_and_download_files(tmp_path: Path) -> None:
    """Test uploading and downloading files on local disk."""
    sandbox = CloudRunSandbox()

    file1 = tmp_path / "subdir" / "test.txt"
    file2 = tmp_path / "data.bin"

    upload_res = sandbox.upload_files(
        [
            (str(file1), b"hello world"),
            (str(file2), b"\x00\x01\x02\x03"),
        ]
    )

    assert len(upload_res) == 2
    assert upload_res[0].error is None
    assert upload_res[1].error is None
    assert file1.read_text() == "hello world"
    assert file2.read_bytes() == b"\x00\x01\x02\x03"

    download_res = sandbox.download_files(
        [str(file1), str(file2), str(tmp_path / "missing.txt")]
    )
    assert len(download_res) == 3
    assert download_res[0].content == b"hello world"
    assert download_res[0].error is None
    assert download_res[1].content == b"\x00\x01\x02\x03"
    assert download_res[1].error is None
    assert download_res[2].content is None
    assert download_res[2].error is not None
