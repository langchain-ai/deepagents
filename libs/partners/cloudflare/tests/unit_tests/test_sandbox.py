from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import httpx

from langchain_cloudflare.sandbox import (
    COMMAND_TIMEOUT_EXIT_CODE,
    CloudflareSandbox,
    _decode_sse_data,
    _extract_preceding_event,
)

SANDBOX_ID = "test-sandbox-123"
BASE_URL = "https://sandbox-bridge.example.workers.dev"


def _make_sandbox(
    *,
    api_key: str | None = "test-key",
    timeout: int = 300,
) -> CloudflareSandbox:
    return CloudflareSandbox(
        base_url=BASE_URL,
        sandbox_id=SANDBOX_ID,
        api_key=api_key,
        timeout=timeout,
    )


def _build_sse_body(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> str:
    """Build a mock SSE response body matching the bridge format."""
    parts: list[str] = []
    if stdout:
        encoded = base64.b64encode(stdout.encode()).decode()
        parts.append(f"event: stdout\ndata: {encoded}\n")
    if stderr:
        encoded = base64.b64encode(stderr.encode()).decode()
        parts.append(f"event: stderr\ndata: {encoded}\n")
    parts.append(f'event: exit\ndata: {{"exit_code": {exit_code}}}\n')
    return "\n".join(parts)


def test_id_property() -> None:
    sb = _make_sandbox()
    assert sb.id == SANDBOX_ID


def test_init_strips_trailing_slash() -> None:
    sb = CloudflareSandbox(
        base_url="https://example.com/",
        sandbox_id="sb-1",
    )
    assert sb._base_url == "https://example.com"


def test_init_without_api_key() -> None:
    sb = CloudflareSandbox(
        base_url=BASE_URL,
        sandbox_id=SANDBOX_ID,
    )
    assert "Authorization" not in sb._client.headers


def test_init_with_api_key() -> None:
    sb = _make_sandbox(api_key="my-secret")
    assert sb._client.headers["Authorization"] == "Bearer my-secret"


def test_execute_returns_stdout() -> None:
    sb = _make_sandbox()
    sse_body = _build_sse_body(stdout="hello world", exit_code=0)
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response) as mock_post:
        result = sb.execute("echo hello world")

    assert result.output == "hello world"
    assert result.exit_code == 0
    assert result.truncated is False

    call_args = mock_post.call_args
    assert f"/v1/sandbox/{SANDBOX_ID}/exec" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["argv"] == ["sh", "-lc", "echo hello world"]


def test_execute_returns_stderr() -> None:
    sb = _make_sandbox()
    sse_body = _build_sse_body(stdout="out", stderr="err", exit_code=1)
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response):
        result = sb.execute("bad-command")

    assert "out" in result.output
    assert "err" in result.output
    assert result.exit_code == 1


def test_execute_timeout_returns_124() -> None:
    sb = _make_sandbox(timeout=10)

    with patch.object(
        sb._client,
        "post",
        side_effect=httpx.TimeoutException("timed out"),
    ):
        result = sb.execute("sleep 999", timeout=5)

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert "timed out" in result.output.lower()


def test_execute_http_error() -> None:
    sb = _make_sandbox()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch.object(
        sb._client,
        "post",
        side_effect=httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        ),
    ):
        result = sb.execute("echo test")

    assert result.exit_code == 1
    assert "500" in result.output


def test_execute_uses_default_timeout() -> None:
    sb = _make_sandbox(timeout=600)
    sse_body = _build_sse_body(stdout="ok", exit_code=0)
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response) as mock_post:
        sb.execute("echo ok")

    payload = mock_post.call_args.kwargs["json"]
    assert payload["timeout_ms"] == 600_000


def test_execute_uses_explicit_timeout() -> None:
    sb = _make_sandbox(timeout=600)
    sse_body = _build_sse_body(stdout="ok", exit_code=0)
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response) as mock_post:
        sb.execute("echo ok", timeout=30)

    payload = mock_post.call_args.kwargs["json"]
    assert payload["timeout_ms"] == 30_000


def test_download_files_success() -> None:
    sb = _make_sandbox()
    mock_response = MagicMock()
    mock_response.content = b"file content"
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "get", return_value=mock_response):
        results = sb.download_files(["/workspace/test.txt"])

    assert len(results) == 1
    assert results[0].path == "/workspace/test.txt"
    assert results[0].content == b"file content"
    assert results[0].error is None


def test_download_files_not_found() -> None:
    sb = _make_sandbox()
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch.object(
        sb._client,
        "get",
        side_effect=httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response,
        ),
    ):
        results = sb.download_files(["/workspace/missing.txt"])

    assert len(results) == 1
    assert results[0].error == "file_not_found"
    assert results[0].content is None


def test_download_files_invalid_path() -> None:
    sb = _make_sandbox()
    results = sb.download_files(["relative/path.txt"])

    assert len(results) == 1
    assert results[0].error == "invalid_path"


def test_upload_files_success() -> None:
    sb = _make_sandbox()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "put", return_value=mock_response) as mock_put:
        results = sb.upload_files([("/workspace/test.txt", b"content")])

    assert len(results) == 1
    assert results[0].path == "/workspace/test.txt"
    assert results[0].error is None

    call_args = mock_put.call_args
    assert f"/v1/sandbox/{SANDBOX_ID}/file/workspace/test.txt" in call_args.args[0]


def test_upload_files_invalid_path() -> None:
    sb = _make_sandbox()
    results = sb.upload_files([("relative/path.txt", b"content")])

    assert len(results) == 1
    assert results[0].error == "invalid_path"


def test_upload_files_http_error() -> None:
    sb = _make_sandbox()
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch.object(
        sb._client,
        "put",
        side_effect=httpx.HTTPStatusError(
            "Forbidden",
            request=MagicMock(),
            response=mock_response,
        ),
    ):
        results = sb.upload_files([("/workspace/test.txt", b"content")])

    assert len(results) == 1
    assert results[0].error == "permission_denied"


def test_decode_sse_data_base64() -> None:
    encoded = base64.b64encode(b"hello world").decode()
    assert _decode_sse_data(encoded) == "hello world"


def test_decode_sse_data_plain_text() -> None:
    assert _decode_sse_data("not-valid-base64!!!") == "not-valid-base64!!!"


def test_extract_preceding_event_found() -> None:
    body = "event: stdout\ndata: abc\n"
    assert _extract_preceding_event(body, "data: abc") == "stdout"


def test_extract_preceding_event_not_found() -> None:
    body = "data: abc\n"
    assert _extract_preceding_event(body, "data: abc") is None


def test_execute_no_exit_event_returns_timeout() -> None:
    sb = _make_sandbox(timeout=10)
    sse_body = "event: stdout\ndata: partial\n"
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response):
        result = sb.execute("hanging-command")

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE


def test_multiple_file_operations() -> None:
    sb = _make_sandbox()

    mock_get = MagicMock()
    mock_get.content = b"data"
    mock_get.raise_for_status = MagicMock()

    mock_put = MagicMock()
    mock_put.raise_for_status = MagicMock()

    with (
        patch.object(sb._client, "get", return_value=mock_get),
        patch.object(sb._client, "put", return_value=mock_put),
    ):
        download_results = sb.download_files(
            ["/workspace/a.txt", "bad-path", "/workspace/b.txt"]
        )
        upload_results = sb.upload_files([("/workspace/c.txt", b"c"), ("bad", b"d")])

    assert len(download_results) == 3
    assert download_results[0].error is None
    assert download_results[1].error == "invalid_path"
    assert download_results[2].error is None

    assert len(upload_results) == 2
    assert upload_results[0].error is None
    assert upload_results[1].error == "invalid_path"


def test_sse_with_json_error_event() -> None:
    sb = _make_sandbox()
    sse_body = 'event: error\ndata: {"error": "sandbox crashed", "code": "INTERNAL"}\n'
    mock_response = MagicMock()
    mock_response.text = sse_body
    mock_response.raise_for_status = MagicMock()

    with patch.object(sb._client, "post", return_value=mock_response):
        result = sb.execute("crash")

    assert "sandbox crashed" in result.output
    assert result.exit_code == 1
