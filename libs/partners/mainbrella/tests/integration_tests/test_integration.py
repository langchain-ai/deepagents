# Copyright (c) 2026 Mainbrella
"""Live tests for Mainbrella's bounded HTTP sandbox operations."""

from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from mainbrella import Mainbrella

from langchain_mainbrella import MainbrellaSandbox

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.skipif(
    not os.environ.get("MAINBRELLA_API_KEY"), reason="MAINBRELLA_API_KEY is required"
)

COMMAND_FAILURE_EXIT_CODE = 7
COMMAND_TIMEOUT_EXIT_CODE = 124


@pytest.fixture(scope="module")
def sandbox() -> Iterator[MainbrellaSandbox]:
    client = Mainbrella(
        os.environ["MAINBRELLA_API_KEY"],
        base_url=os.environ.get("MAINBRELLA_API_URL") or "https://api.mainbrella.com",
    )
    with client.create(catalog_id="python") as handle:
        yield MainbrellaSandbox(sandbox=handle)


@pytest.fixture
def directory(sandbox: MainbrellaSandbox) -> str:
    path = f"/workspace/deepagents-test-{uuid4().hex}"
    assert sandbox.execute(f"mkdir -p {shlex.quote(path)}").exit_code == 0
    return path


def test_execute(sandbox: MainbrellaSandbox) -> None:
    result = sandbox.execute(
        f"printf hello; printf warning >&2; exit {COMMAND_FAILURE_EXIT_CODE}"
    )
    assert result.output == "hello\n<stderr>warning</stderr>"
    assert result.exit_code == COMMAND_FAILURE_EXIT_CODE
    assert result.truncated is False


def test_timeout(sandbox: MainbrellaSandbox) -> None:
    result = sandbox.execute("printf partial; sleep 5", timeout=1)
    assert "partial" in result.output
    assert "timed out" in result.output
    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE


def test_managed_execution(sandbox: MainbrellaSandbox) -> None:
    result = sandbox.execute("printf managed", timeout=120)
    assert result.output == "managed"
    assert result.exit_code == 0


def test_binary_transfer_at_limit(sandbox: MainbrellaSandbox, directory: str) -> None:
    path = f"{directory}/binary & unicode-☃.bin"
    payload = bytes(range(256)) * 4096
    assert sandbox.upload_files([(path, payload)])[0].error is None
    result = sandbox.download_files([path])[0]
    assert result.error is None
    assert result.content == payload


def test_file_limit(sandbox: MainbrellaSandbox, directory: str) -> None:
    path = f"{directory}/too-large.bin"
    payload = b"x" * (1024 * 1024 + 1)
    assert sandbox.upload_files([(path, payload)])[0].error == "file_too_large"
    assert (
        sandbox.execute(f"head -c 1048577 /dev/zero > {shlex.quote(path)}").exit_code
        == 0
    )
    assert sandbox.download_files([path])[0].error == "file_too_large"


def test_file_errors(sandbox: MainbrellaSandbox, directory: str) -> None:
    results = sandbox.download_files(["relative", f"{directory}/missing", directory])
    assert [result.error for result in results] == [
        "invalid_path",
        "file_not_found",
        "invalid_path",
    ]
    assert (
        sandbox.upload_files([(f"{directory}/missing-parent/file", b"test")])[0].error
        == "file_not_found"
    )


def test_inherited_file_tools(sandbox: MainbrellaSandbox, directory: str) -> None:
    path = f"{directory}/example.txt"
    assert sandbox.write(path, "first line\nsecond line").error is None
    result = sandbox.read(path)
    assert result.error is None
    assert result.file_data is not None
    assert result.file_data["content"] == "first line\nsecond line"
    assert sandbox.edit(path, "second", "last").error is None
    assert sandbox.execute(f"cat {shlex.quote(path)}").output == "first line\nlast line"
    listing = sandbox.ls(directory)
    assert listing.error is None
    assert listing.entries is not None
    assert path in [entry["path"] for entry in listing.entries]
    matches = sandbox.glob("*.txt", directory)
    assert matches.error is None
    assert matches.matches is not None
    assert path in [entry["path"] for entry in matches.matches]
    assert sandbox.grep("last", directory).error is None


async def test_async_file_roundtrip(sandbox: MainbrellaSandbox, directory: str) -> None:
    path = f"{directory}/async.txt"
    assert (await sandbox.aupload_files([(path, b"async")]))[0].error is None
    assert (await sandbox.adownload_files([path]))[0].content == b"async"
    assert (await sandbox.aexecute("printf async")).output == "async"
