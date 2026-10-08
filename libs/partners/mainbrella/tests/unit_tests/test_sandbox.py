# Copyright (c) 2026 Mainbrella
from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import parse_qs, urlsplit

import pytest
from mainbrella import Mainbrella, MainbrellaError

from langchain_mainbrella import MainbrellaSandbox

GENERATION = "2026-10-05T12:00:00.000Z"
JOB_ID = "12345678-1234-1234-1234-123456789abc"
Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


def backend(transport: Transport) -> MainbrellaSandbox:
    client = Mainbrella("mb_" + "a" * 64, transport=transport)
    return MainbrellaSandbox(sandbox=client.connect("small", GENERATION))


@pytest.mark.parametrize("exit_code", [0, 7])
def test_execute_preserves_status_and_streams(exit_code: int) -> None:
    def transport(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        assert method == "POST"
        assert headers["Authorization"].startswith("Bearer mb_")
        assert timeout > 0
        assert urlsplit(url).path == "/containers/exec"
        assert parse_qs(urlsplit(url).query) == {
            "id": ["small"],
            "createdAt": [GENERATION],
        }
        assert json.loads(body or b"") == {"command": "echo hello", "timeoutMs": 2000}
        return 200, json.dumps(
            {
                "stdout": "hello",
                "stderr": "warning",
                "exitCode": exit_code,
                "timedOut": False,
                "outputTruncated": False,
            }
        ).encode()

    sandbox = backend(transport)
    result = sandbox.execute("echo hello", timeout=2)
    assert sandbox.id == f"small@{GENERATION}"
    assert result.output == "hello\n<stderr>warning</stderr>"
    assert result.exit_code == exit_code
    assert result.truncated is False


@pytest.mark.parametrize(
    ("timed_out", "truncated", "expected_exit"),
    [(True, False, 124), (False, True, None)],
)
def test_execute_partial_output(
    *, timed_out: bool, truncated: bool, expected_exit: int | None
) -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        return (
            200,
            json.dumps(
                {
                    "stdout": "partial",
                    "stderr": "",
                    "exitCode": None,
                    "timedOut": timed_out,
                    "outputTruncated": truncated,
                }
            ).encode(),
        )

    result = backend(transport).execute("command")
    assert result.output.startswith("partial")
    assert result.exit_code == expected_exit
    assert result.truncated == truncated


@pytest.mark.parametrize("timeout", [-1, 0, 901])
def test_invalid_timeouts_do_not_submit(timeout: int) -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        pytest.fail("Invalid timeout must not submit a command")

    with pytest.raises(ValueError, match="between 1 and 900"):
        backend(transport).execute("touch /workspace/side-effect", timeout=timeout)
    with pytest.raises(ValueError, match="between 1 and 900"):
        MainbrellaSandbox(sandbox=backend(transport)._sandbox, timeout=timeout)


def test_long_commands_use_managed_execution() -> None:
    def transport(
        url: str,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        path = urlsplit(url).path
        if method == "POST":
            assert path == "/containers/executions"
            assert headers["Idempotency-Key"]
            assert json.loads(body or b"")["timeoutMs"] == 120 * 1000
            return 202, json.dumps({"id": JOB_ID}).encode()
        assert path == f"/containers/executions/{JOB_ID}"
        return 200, json.dumps(
            {
                "status": "succeeded",
                "stdout": "done",
                "stderr": "",
                "exitCode": 0,
                "timedOut": False,
                "outputTruncated": False,
            }
        ).encode()

    result = backend(transport).execute("long command", timeout=120)
    assert result.output == "done"
    assert result.exit_code == 0


def test_observation_failure_cancels_managed_job() -> None:
    requests: list[str] = []

    def transport(
        _url: str,
        method: str,
        _headers: dict[str, str],
        _body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        requests.append(method)
        if method == "POST":
            return 202, json.dumps({"id": JOB_ID}).encode()
        if method == "GET":
            return 503, b'{"error":"execution_unavailable"}'
        return 202, b"{}"

    with pytest.raises(MainbrellaError, match="execution_unavailable"):
        backend(transport).execute("long command", timeout=120)
    assert requests == ["POST", "GET", "DELETE"]


def test_binary_transfer_and_independent_errors() -> None:
    contents: dict[str, bytes] = {}

    def transport(
        url: str,
        method: str,
        _headers: dict[str, str],
        body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        query = parse_qs(urlsplit(url).query)
        assert query["createdAt"] == [GENERATION]
        path = query["path"][0]
        if path == "/denied":
            return 403, b'{"error":"file_access_denied"}'
        if method == "PUT":
            assert body is not None
            contents[path] = body
            return 200, b"{}"
        return (
            (200, contents[path])
            if path in contents
            else (404, b'{"error":"file_not_found"}')
        )

    sandbox = backend(transport)
    payload = bytes([0, 128, 255])
    results = sandbox.upload_files(
        [
            ("/workspace/a & b.bin", payload),
            ("relative", b""),
            ("/workspace/empty", b""),
            ("/denied", b"no"),
        ]
    )
    assert [r.error for r in results] == [
        None,
        "invalid_path",
        None,
        "permission_denied",
    ]
    downloads = sandbox.download_files(
        ["/workspace/a & b.bin", "/workspace/empty", "/missing", "/denied"]
    )
    assert [r.content for r in downloads] == [payload, b"", None, None]
    assert [r.error for r in downloads] == [
        None,
        None,
        "file_not_found",
        "permission_denied",
    ]


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "/a/../b",
        "/a/./b",
        "/a//b",
        "/a/",
        "/nul\0",
        "/" + "a" * 4096,
        "/\ud800",
    ],
)
def test_invalid_paths_do_not_reach_provider(path: str) -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        pytest.fail("Invalid path must not reach the provider")

    sandbox = backend(transport)
    assert sandbox.upload_files([(path, b"")])[0].error == "invalid_path"
    assert sandbox.download_files([path])[0].error == "invalid_path"


def test_oversized_upload_does_not_write() -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        pytest.fail("Oversized upload must not reach the provider")

    assert (
        backend(transport)
        .upload_files([("/workspace/large", b"x" * (1024 * 1024 + 1))])[0]
        .error
        == "file_too_large"
    )


@pytest.mark.parametrize(
    "code",
    [
        "container_not_running",
        "files_unavailable",
        "file_too_large",
        "not_authenticated",
    ],
)
def test_service_errors_are_not_missing_files(code: str) -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        return 409, json.dumps({"error": code}).encode()

    sandbox = backend(transport)
    assert sandbox.download_files(["/workspace/test"])[0].error == code
    assert sandbox.upload_files([("/workspace/test", b"test")])[0].error == code


def test_failed_commands_are_not_retried() -> None:
    requests: list[str] = []

    def transport(
        _url: str,
        method: str,
        _headers: dict[str, str],
        _body: bytes | None,
        _timeout: float,
    ) -> tuple[int, bytes]:
        requests.append(method)
        return 503, b'{"error":"execution_unavailable"}'

    with pytest.raises(MainbrellaError, match="execution_unavailable"):
        backend(transport).execute("side effect")
    assert requests == ["POST"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("invalid_file_path", "invalid_path"),
        ("not_regular_file", "invalid_path"),
    ],
)
def test_provider_path_errors_are_normalized(code: str, expected: str) -> None:
    def transport(*_args: object) -> tuple[int, bytes]:
        return 409, json.dumps({"error": code}).encode()

    sandbox = backend(transport)
    assert sandbox.download_files(["/workspace/test"])[0].error == expected
    assert sandbox.upload_files([("/workspace/test", b"test")])[0].error == expected
