from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import langchain_docker
from langchain_docker import DockerError, DockerSandbox
from langchain_docker.backend import CONTAINER_WORKDIR, DEFAULT_IMAGE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

TIMEOUT_EXIT_CODE = 124
CONTAINER_ID_LENGTH = 12


def _ok(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["docker"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _exec_side_effect(
    **exec_config: object,
) -> object:
    def _run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "exec":
            error = exec_config.get("error")
            if isinstance(error, Exception):
                raise error
            exec_result = exec_config.get("exec")
            if isinstance(exec_result, subprocess.CompletedProcess):
                return exec_result
            return _ok()
        return _ok()

    return _run


@pytest.fixture
def docker_cli() -> Iterator[MagicMock]:
    with (
        patch.object(DockerSandbox, "_docker_available", return_value=True),
        patch.object(DockerSandbox, "_run_docker") as mock,
    ):
        mock.return_value = _ok()
        yield mock


def test_import_docker() -> None:
    assert langchain_docker is not None
    assert DockerSandbox is not None
    assert issubclass(DockerError, RuntimeError)


def test_default_image_is_python_bookworm(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        run_args = docker_cli.call_args_list[0].args[0]
        assert DEFAULT_IMAGE in run_args
        image_index = run_args.index(DEFAULT_IMAGE)
        assert run_args[image_index + 1 : image_index + 3] == ["sleep", "infinity"]
    finally:
        sandbox.close()


def test_start_container_mounts_shared_dir(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, image="test-image:local")
    try:
        run_args = docker_cli.call_args_list[0].args[0]
        assert run_args[0] == "run"
        assert run_args[run_args.index("--network") + 1] == "bridge"
        assert f"{tmp_path.resolve()}:{CONTAINER_WORKDIR}:rw" in run_args
        assert run_args[run_args.index("-w") + 1] == CONTAINER_WORKDIR
        assert len(sandbox.id) == CONTAINER_ID_LENGTH
        assert sandbox.shared_dir == tmp_path.resolve()
    finally:
        sandbox.close()


def test_start_container_disables_outbound_traffic(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, allow_outbound_traffic=False)
    try:
        run_args = docker_cli.call_args_list[0].args[0]
        assert run_args[run_args.index("--network") + 1] == "none"
    finally:
        sandbox.close()


def test_start_container_applies_resource_limits_and_extra_args(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(
        shared_dir=tmp_path,
        memory="1g",
        cpus=2.5,
        pids_limit=256,
        extra_run_args=["--env", "FOO=bar"],
    )
    try:
        run_args = docker_cli.call_args_list[0].args[0]
        assert run_args[run_args.index("--memory") + 1] == "1g"
        assert run_args[run_args.index("--cpus") + 1] == "2.5"
        assert run_args[run_args.index("--pids-limit") + 1] == "256"
        assert run_args[run_args.index("--env") + 1] == "FOO=bar"
    finally:
        sandbox.close()


def test_raises_when_container_start_fails(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    docker_cli.return_value = _ok(returncode=1, stderr="image not found")
    with pytest.raises(
        DockerError, match="failed to start sandbox container: image not found"
    ):
        DockerSandbox(shared_dir=tmp_path)


def test_start_failure_extracts_json_message(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    docker_cli.return_value = _ok(
        returncode=1,
        stderr=('{"message": "Conflict. The container name is already in use."}'),
    )
    with pytest.raises(DockerError, match="already in use"):
        DockerSandbox(shared_dir=tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout": 0}, "timeout must be positive"),
        ({"cpus": 0}, "cpus must be positive"),
        ({"pids_limit": -1}, "pids_limit must be positive"),
    ],
)
def test_constructor_rejects_invalid_limits(
    kwargs: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DockerSandbox(**kwargs)


def test_raises_when_docker_unavailable(tmp_path: Path) -> None:
    with (
        patch.object(DockerSandbox, "_run_docker", return_value=_ok(returncode=1)),
        pytest.raises(DockerError, match="Docker is not available"),
    ):
        DockerSandbox(shared_dir=tmp_path, image="missing:latest")


def test_execute_runs_command_in_container(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, image="test-image:local")
    try:
        docker_cli.side_effect = _exec_side_effect(exec=_ok(stdout="hello\n"))
        result = sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.output
        exec_args = docker_cli.call_args.args[0]
        assert exec_args[:2] == ["exec", f"deepagents-docker-{sandbox.id}"]
        assert exec_args[-1] == "echo hello"
    finally:
        sandbox.close()


def test_execute_formats_stderr_and_nonzero_exit(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        docker_cli.side_effect = _exec_side_effect(
            exec=_ok(returncode=2, stderr="something broke\n"),
        )
        result = sandbox.execute("false")
        assert result.exit_code == 2
        assert "[stderr] something broke" in result.output
        assert "Exit code: 2" in result.output
    finally:
        sandbox.close()


def test_execute_reports_no_output(docker_cli: MagicMock, tmp_path: Path) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        docker_cli.side_effect = _exec_side_effect(exec=_ok())
        result = sandbox.execute("true")
        assert result.output == "<no output>"
    finally:
        sandbox.close()


def test_execute_truncates_large_output(docker_cli: MagicMock, tmp_path: Path) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, max_output_bytes=50)
    try:
        docker_cli.side_effect = _exec_side_effect(exec=_ok(stdout="x" * 200))
        result = sandbox.execute("printf x")
        assert result.truncated is True
        assert "Output truncated at 50 bytes" in result.output
    finally:
        sandbox.close()


def test_execute_rejects_empty_command(docker_cli: MagicMock, tmp_path: Path) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        result = sandbox.execute("")
        assert result.exit_code == 1
        assert "non-empty string" in result.output
    finally:
        sandbox.close()


def test_execute_after_close_returns_error(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    sandbox.close()
    result = sandbox.execute("echo hello")
    assert result.exit_code == 1
    assert "closed" in result.output.lower()


def test_execute_timeout_with_custom_message(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, timeout=1)
    try:
        docker_cli.side_effect = _exec_side_effect(
            error=DockerError("docker command timed out after 1 seconds"),
        )
        result = sandbox.execute("sleep 10", timeout=1)
        assert result.exit_code == TIMEOUT_EXIT_CODE
        assert "custom timeout" in result.output.lower()
    finally:
        sandbox.close()


def test_execute_timeout_with_default_message(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        docker_cli.side_effect = _exec_side_effect(
            error=DockerError("docker command timed out after 120 seconds"),
        )
        result = sandbox.execute("sleep 10")
        assert result.exit_code == TIMEOUT_EXIT_CODE
        assert "timeout parameter" in result.output.lower()
    finally:
        sandbox.close()


def test_execute_when_docker_binary_missing(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        docker_cli.side_effect = _exec_side_effect(
            error=DockerError("docker executable not found on PATH"),
        )
        result = sandbox.execute("echo hello")
        assert result.exit_code == 1
        assert "not found on PATH" in result.output
    finally:
        sandbox.close()


def test_execute_rejects_non_positive_timeout(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        with pytest.raises(ValueError, match="timeout must be positive"):
            sandbox.execute("echo hello", timeout=0)
    finally:
        sandbox.close()


def test_write_and_read_via_virtual_paths(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, image="test-image:local")
    try:
        write_result = sandbox.write("/notes.txt", "alpha\n")
        assert write_result.error is None
        assert (tmp_path / "notes.txt").read_text() == "alpha\n"

        read_result = sandbox.read("/notes.txt")
        assert read_result.error is None
        assert read_result.file_data is not None
        assert "alpha" in read_result.file_data["content"]
    finally:
        sandbox.close()


def test_close_stops_and_removes_container(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    container_name = f"deepagents-docker-{sandbox.id}"
    sandbox.close()

    stop_call = docker_cli.call_args_list[1]
    rm_call = docker_cli.call_args_list[2]
    assert stop_call.args[0] == ["stop", "-t", "2", container_name]
    assert rm_call.args[0] == ["rm", "-f", container_name]


def test_close_is_idempotent(docker_cli: MagicMock, tmp_path: Path) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    sandbox.close()
    sandbox.close()
    assert len(docker_cli.call_args_list) == 3


def test_close_skips_remove_when_auto_remove_disabled(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path, auto_remove=False)
    sandbox.close()
    assert len(docker_cli.call_args_list) == 2
    assert docker_cli.call_args_list[1].args[0][0] == "stop"


def test_close_removes_owned_shared_dir(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    shared = tmp_path / "owned-shared"
    shared.mkdir()
    with patch("langchain_docker.backend.tempfile.mkdtemp", return_value=str(shared)):
        sandbox = DockerSandbox()
        sandbox.close()
    assert not shared.exists()


def test_close_preserves_user_shared_dir(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    sandbox.close()
    assert tmp_path.exists()


def test_context_manager_closes_sandbox(
    docker_cli: MagicMock,
    tmp_path: Path,
) -> None:
    with DockerSandbox(shared_dir=tmp_path) as sandbox:
        assert len(sandbox.id) == CONTAINER_ID_LENGTH
    assert len(docker_cli.call_args_list) == 3


def test_format_docker_error_variants(docker_cli: MagicMock, tmp_path: Path) -> None:
    sandbox = DockerSandbox(shared_dir=tmp_path)
    try:
        assert (
            sandbox._format_docker_error(_ok(returncode=1, stderr="image not found"))
            == "image not found"
        )
        conflict = '{"message": "Conflict. The container name is already in use."}'
        assert sandbox._format_docker_error(_ok(returncode=1, stderr=conflict)) == (
            "Conflict. The container name is already in use."
        )
        payload = '{"errorDetail":{"code":404}}'
        assert (
            sandbox._format_docker_error(_ok(returncode=1, stderr=payload)) == payload
        )
        assert sandbox._format_docker_error(_ok(returncode=7)) == "exit code 7"
    finally:
        sandbox.close()


def test_run_docker_returns_captured_output(tmp_path: Path) -> None:
    completed = _ok(stdout="ok\n")
    with (
        patch.object(DockerSandbox, "_docker_available", return_value=True),
        patch(
            "langchain_docker.backend.subprocess.run",
            return_value=completed,
        ) as subprocess_run,
    ):
        sandbox = DockerSandbox(shared_dir=tmp_path)
        try:
            result = sandbox._run_docker(["info"])
            assert result.returncode == 0
            assert result.stdout == "ok\n"
            subprocess_run.assert_called()
        finally:
            sandbox.close()


def test_run_docker_raises_on_timeout(tmp_path: Path) -> None:
    with (
        patch.object(DockerSandbox, "_docker_available", return_value=True),
        patch(
            "langchain_docker.backend.subprocess.run",
            return_value=_ok(),
        ) as subprocess_run,
    ):
        sandbox = DockerSandbox(shared_dir=tmp_path)
        try:
            subprocess_run.side_effect = subprocess.TimeoutExpired(
                cmd="docker",
                timeout=5,
            )
            with pytest.raises(DockerError, match="timed out after 5 seconds"):
                sandbox._run_docker(["exec", "cid", "true"], timeout=5)
            subprocess_run.side_effect = None
            subprocess_run.return_value = _ok()
        finally:
            sandbox.close()


def test_run_docker_raises_when_docker_missing(tmp_path: Path) -> None:
    with (
        patch.object(DockerSandbox, "_docker_available", return_value=True),
        patch(
            "langchain_docker.backend.subprocess.run",
            return_value=_ok(),
        ) as subprocess_run,
    ):
        sandbox = DockerSandbox(shared_dir=tmp_path)
        try:
            subprocess_run.side_effect = FileNotFoundError
            with pytest.raises(DockerError, match="not found on PATH"):
                sandbox._run_docker(["info"])
            subprocess_run.side_effect = None
            subprocess_run.return_value = _ok()
        finally:
            sandbox.close()
