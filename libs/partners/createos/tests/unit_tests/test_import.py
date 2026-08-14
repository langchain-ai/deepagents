from __future__ import annotations

from unittest.mock import patch

import httpx
import langchain_createos
from langchain_createos.sandbox import CreateOSSandbox

COMMAND_TIMEOUT_EXIT_CODE = 124

_FAKE_REQUEST = httpx.Request("GET", "https://test.example.com")


def _make_sandbox() -> CreateOSSandbox:
    return CreateOSSandbox(
        sandbox_id="sb-123",
        api_key="test-key",
        base_url="https://test.example.com",
    )


def test_import_createos() -> None:
    assert langchain_createos is not None


def test_execute_returns_stdout() -> None:
    sb = _make_sandbox()
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
    with patch.object(sb._client, "post", return_value=mock_response):
        result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False


def test_execute_includes_stderr() -> None:
    sb = _make_sandbox()
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
    with patch.object(sb._client, "post", return_value=mock_response):
        result = sb.execute("failing-cmd")

    assert "out" in result.output
    assert "<stderr>err msg</stderr>" in result.output
    assert result.exit_code == 1


def test_sandbox_id_property() -> None:
    sb = _make_sandbox()
    assert sb.id == "sb-123"


def test_download_files_invalid_path() -> None:
    sb = _make_sandbox()
    results = sb.download_files(["relative/path"])
    assert len(results) == 1
    assert results[0].error == "invalid_path"
    assert results[0].content is None


def test_upload_files_invalid_path() -> None:
    sb = _make_sandbox()
    results = sb.upload_files([("relative/path", b"data")])
    assert len(results) == 1
    assert results[0].error == "invalid_path"


def test_download_files_success() -> None:
    sb = _make_sandbox()
    mock_response = httpx.Response(200, request=_FAKE_REQUEST, content=b"file content")
    with patch.object(sb._client, "get", return_value=mock_response):
        results = sb.download_files(["/app/test.txt"])

    assert len(results) == 1
    assert results[0].content == b"file content"
    assert results[0].error is None


def test_upload_files_success() -> None:
    sb = _make_sandbox()
    mock_response = httpx.Response(200, request=_FAKE_REQUEST, json={"status": "success"})
    with patch.object(sb._client, "put", return_value=mock_response):
        results = sb.upload_files([("/app/test.txt", b"content")])

    assert len(results) == 1
    assert results[0].error is None
