from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import langchain_createos
from langchain_createos.sandbox import CreateOSSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepagents.backends.protocol import ExecuteResponse

_FAKE_REQUEST = httpx.Request("GET", "https://test.example.com")


def _make_sandbox() -> CreateOSSandbox:
    return CreateOSSandbox(
        sandbox_id="sb-123",
        api_key="test-key",
        base_url="https://test.example.com",
    )


@pytest.fixture
def sandbox() -> Iterator[CreateOSSandbox]:
    backend = _make_sandbox()
    try:
        yield backend
    finally:
        backend.close()


def test_import_createos() -> None:
    assert langchain_createos is not None


def test_execute_returns_stdout(sandbox: CreateOSSandbox) -> None:
    mock_response = httpx.Response(
        200,
        request=_FAKE_REQUEST,
        json={
            "status": "success",
            "data": {
                "result": {
                    "stdout": "hello world",
                    "stderr": "",
                    "exit_code": 0,
                },
                "exec_ms": 10.5,
            },
        },
    )
    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = sandbox.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_includes_stderr(sandbox: CreateOSSandbox) -> None:
    mock_response = httpx.Response(
        200,
        request=_FAKE_REQUEST,
        json={
            "status": "success",
            "data": {
                "result": {
                    "stdout": "out",
                    "stderr": "err msg",
                    "exit_code": 1,
                },
                "exec_ms": 5.0,
            },
        },
    )
    with patch.object(httpx.Client, "post", return_value=mock_response):
        result = sandbox.execute("failing-cmd")

    assert "out" in result.output
    assert "<stderr>err msg</stderr>" in result.output
    assert result.exit_code == 1


def test_sandbox_id_property(sandbox: CreateOSSandbox) -> None:
    assert sandbox.id == "sb-123"


def test_download_files_invalid_path(sandbox: CreateOSSandbox) -> None:
    results = sandbox.download_files(["relative/path"])
    assert len(results) == 1
    assert results[0].error == "invalid_path"
    assert results[0].content is None


def test_upload_files_invalid_path(sandbox: CreateOSSandbox) -> None:
    results = sandbox.upload_files([("relative/path", b"data")])
    assert len(results) == 1
    assert results[0].error == "invalid_path"


def test_download_files_success(sandbox: CreateOSSandbox) -> None:
    mock_response = httpx.Response(200, request=_FAKE_REQUEST, content=b"file content")
    with patch.object(httpx.Client, "get", return_value=mock_response):
        results = sandbox.download_files(["/app/test.txt"])

    assert len(results) == 1
    assert results[0].content == b"file content"
    assert results[0].error is None


def test_upload_files_success(sandbox: CreateOSSandbox) -> None:
    mock_response = httpx.Response(
        200,
        request=_FAKE_REQUEST,
        json={"status": "success"},
    )
    with patch.object(httpx.Client, "put", return_value=mock_response):
        results = sandbox.upload_files([("/app/test.txt", b"content")])

    assert len(results) == 1
    assert results[0].error is None


def test_close_closes_http_client() -> None:
    with patch.object(httpx.Client, "close") as close:
        sandbox = _make_sandbox()
        sandbox.close()

    close.assert_called_once_with()


async def test_awrite_existing_file_fails(sandbox: CreateOSSandbox) -> None:
    check = ExecuteResponse(output="", exit_code=0)
    with (
        patch.object(sandbox, "aexecute", AsyncMock(return_value=check)),
        patch.object(sandbox, "aupload_files", AsyncMock()) as upload,
    ):
        result = await sandbox.awrite("/app/existing.txt", "new content")

    assert result.error == "Error: file '/app/existing.txt' already exists"
    upload.assert_not_awaited()
