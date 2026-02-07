"""Unit tests for Harbor backend image validation helpers."""

import base64

import pytest

from deepagents_harbor.backend import HarborSandbox, _is_valid_image


def test_valid_png_bytes_are_accepted() -> None:
    """Accept a valid PNG payload."""
    # 1x1 PNG
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5W8xkAAAAASUVORK5CYII="
    )
    assert _is_valid_image(data)


def test_truncated_png_is_rejected() -> None:
    """Reject truncated PNG bytes even when magic bytes are present."""
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5W8xkAAAAASUVORK5CYII="
    )
    assert not _is_valid_image(data[:-8])


def test_random_bytes_are_rejected() -> None:
    """Reject non-image payloads."""
    assert not _is_valid_image(b"not-an-image")


def test_download_files_method_is_defined_on_class() -> None:
    """Ensure download_files is a class method, not accidentally nested."""
    assert "download_files" in HarborSandbox.__dict__


class _FakeExecResult:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self.stdout = stdout
        self.return_code = return_code


class _FakeEnvironment:
    def __init__(self, stdout: str, return_code: int = 0) -> None:
        self._result = _FakeExecResult(stdout=stdout, return_code=return_code)

    async def exec(self, command: str) -> _FakeExecResult:
        return self._result


@pytest.mark.asyncio
async def test_adownload_files_allows_non_image_bytes() -> None:
    """Non-image files (e.g. markdown offload files) should not be image-validated."""
    payload = b"## Summarized at 2026-02-06T00:00:00Z\n\nexample\n"
    env = _FakeEnvironment(stdout=base64.b64encode(payload).decode("ascii"))
    sandbox = HarborSandbox(environment=env)  # type: ignore[arg-type]

    responses = await sandbox.adownload_files(["/conversation_history/thread.md"])

    assert len(responses) == 1
    assert responses[0].error is None
    assert responses[0].content == payload


@pytest.mark.asyncio
async def test_adownload_files_rejects_invalid_image_bytes_for_image_path() -> None:
    """Image extensions should still enforce image integrity checks."""
    env = _FakeEnvironment(stdout=base64.b64encode(b"not-an-image").decode("ascii"))
    sandbox = HarborSandbox(environment=env)  # type: ignore[arg-type]

    responses = await sandbox.adownload_files(["/tmp/image.png"])

    assert len(responses) == 1
    assert responses[0].content is None
    assert responses[0].error == "invalid_image_format"
