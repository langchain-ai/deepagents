from __future__ import annotations

from unittest.mock import MagicMock

from sprites.exceptions import (
    FileNotFoundError_,
    FilesystemError,
    IsADirectoryError_,
    PermissionError_,
)
from sprites.exceptions import (
    TimeoutError as SpritesTimeoutError,
)

import langchain_sprites
from langchain_sprites.sandbox import SpritesSandbox

COMMAND_TIMEOUT_EXIT_CODE = 124
EXPECTED_EXIT_CODE = 2
EXPECTED_TIMEOUT = 5
EXPECTED_WRITE_COUNT = 2


def _make_sandbox(**kwargs: object) -> tuple[SpritesSandbox, MagicMock]:
    mock_sprite = MagicMock()
    mock_sprite.name = "my-sprite"
    sb = SpritesSandbox(sprite=mock_sprite, **kwargs)  # ty: ignore[invalid-argument-type]
    return sb, mock_sprite


def _completed(
    stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0
) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


def test_import_sprites() -> None:
    assert langchain_sprites is not None


def test_id_is_sprite_name() -> None:
    sb, _ = _make_sandbox()
    assert sb.id == "my-sprite"


def test_sprite_property_exposes_sdk_object() -> None:
    sb, mock_sprite = _make_sandbox()
    assert sb.sprite is mock_sprite


def test_execute_runs_bash_lc_in_workdir() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_sprite.run.return_value = _completed(stdout=b"hello\n")

    result = sb.execute("echo hello")

    mock_sprite.run.assert_called_once_with(
        "bash",
        "-lc",
        "echo hello",
        capture_output=True,
        cwd="/home/sprite",
        timeout=300,
    )
    assert result.output == "hello\n"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_uses_custom_workdir_and_timeout() -> None:
    sb, mock_sprite = _make_sandbox(timeout=10, workdir="/app")
    mock_sprite.run.return_value = _completed()

    sb.execute("true")

    mock_sprite.run.assert_called_once_with(
        "bash",
        "-lc",
        "true",
        capture_output=True,
        cwd="/app",
        timeout=10,
    )


def test_execute_explicit_timeout_overrides_default() -> None:
    sb, mock_sprite = _make_sandbox(timeout=300)
    mock_sprite.run.return_value = _completed()

    sb.execute("true", timeout=5)

    assert mock_sprite.run.call_args.kwargs["timeout"] == EXPECTED_TIMEOUT


def test_execute_wraps_stderr() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_sprite.run.return_value = _completed(stdout=b"out", stderr=b"warning\n")

    result = sb.execute("some command")

    assert result.output == "out\n<stderr>warning</stderr>"


def test_execute_returns_nonzero_exit_code() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_sprite.run.return_value = _completed(stderr=b"boom\n", returncode=2)

    result = sb.execute("exit 2")

    assert result.exit_code == EXPECTED_EXIT_CODE
    assert "<stderr>boom</stderr>" in result.output


def test_execute_timeout() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_sprite.run.side_effect = SpritesTimeoutError("command timed out after 10s")

    result = sb.execute("sleep 999", timeout=10)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output


def test_upload_files_writes_via_filesystem() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_path = MagicMock()
    mock_sprite.filesystem.return_value.__truediv__.return_value = mock_path

    responses = sb.upload_files([("/data/a.txt", b"aaa"), ("/data/dir/b.txt", b"bbb")])

    mock_sprite.filesystem.assert_called_with("/home/sprite")
    assert mock_path.write_bytes.call_count == EXPECTED_WRITE_COUNT
    assert [(r.path, r.error) for r in responses] == [
        ("/data/a.txt", None),
        ("/data/dir/b.txt", None),
    ]


def test_upload_files_partial_failure() -> None:
    sb, mock_sprite = _make_sandbox()
    ok_path = MagicMock()
    denied_path = MagicMock()
    denied_path.write_bytes.side_effect = PermissionError_("write", "b.txt")
    mock_sprite.filesystem.return_value.__truediv__.side_effect = [
        ok_path,
        denied_path,
    ]

    responses = sb.upload_files([("/data/a.txt", b"aaa"), ("/data/b.txt", b"bbb")])

    assert responses[0].error is None
    assert responses[1].error == "permission_denied"


def test_download_files_reads_via_filesystem() -> None:
    sb, mock_sprite = _make_sandbox()
    mock_path = MagicMock()
    mock_path.read_bytes.return_value = b"hello"
    mock_sprite.filesystem.return_value.__truediv__.return_value = mock_path

    responses = sb.download_files(["/data/a.txt"])

    assert responses[0].content == b"hello"
    assert responses[0].error is None


def test_download_files_maps_missing_file() -> None:
    sb, mock_sprite = _make_sandbox()
    ok_path = MagicMock()
    ok_path.read_bytes.return_value = b"hello"
    missing_path = MagicMock()
    missing_path.read_bytes.side_effect = FileNotFoundError_("read", "missing.txt")
    mock_sprite.filesystem.return_value.__truediv__.side_effect = [
        ok_path,
        missing_path,
    ]

    responses = sb.download_files(["/data/a.txt", "/data/missing.txt"])

    assert responses[0].error is None
    assert responses[1].content is None
    assert responses[1].error == "file_not_found"


def test_download_files_maps_directory_error() -> None:
    sb, mock_sprite = _make_sandbox()
    dir_path = MagicMock()
    dir_path.read_bytes.side_effect = IsADirectoryError_("read", "dir")
    mock_sprite.filesystem.return_value.__truediv__.return_value = dir_path

    responses = sb.download_files(["/data/dir"])

    assert responses[0].error == "is_directory"


def test_download_files_maps_unstructured_error_by_message() -> None:
    # The server does not always return a structured error code; a missing
    # file can surface as a generic FilesystemError with the raw OS error.
    sb, mock_sprite = _make_sandbox()
    odd_path = MagicMock()
    odd_path.read_bytes.side_effect = FilesystemError(
        "open /home/sprite/missing.txt: no such file or directory",
        "read",
        "missing.txt",
    )
    mock_sprite.filesystem.return_value.__truediv__.return_value = odd_path

    responses = sb.download_files(["/data/missing.txt"])

    assert responses[0].error == "file_not_found"


def test_upload_relative_path_returns_invalid_path() -> None:
    sb, mock_sprite = _make_sandbox()

    responses = sb.upload_files([("relative.txt", b"aaa")])

    assert responses[0].error == "invalid_path"
    mock_sprite.filesystem.return_value.__truediv__.assert_not_called()


def test_download_relative_path_returns_invalid_path() -> None:
    sb, mock_sprite = _make_sandbox()

    responses = sb.download_files(["relative/path.txt"])

    assert responses[0].content is None
    assert responses[0].error == "invalid_path"
    mock_sprite.filesystem.return_value.__truediv__.assert_not_called()
