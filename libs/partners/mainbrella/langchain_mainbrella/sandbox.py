# Copyright (c) 2026 Mainbrella
"""Mainbrella sandbox backend using the official Python SDK."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict, cast

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from mainbrella import MainbrellaError

if TYPE_CHECKING:
    from mainbrella import Sandbox

logger = logging.getLogger(__name__)

_FOREGROUND_TIMEOUT = 60
_MAX_TIMEOUT = 900
_MAX_FILE_BYTES = 1024 * 1024
_MAX_PATH_BYTES = 4096


class _CommandResult(TypedDict):
    stdout: str
    stderr: str
    exitCode: int | None
    timedOut: bool
    outputTruncated: bool


def _validate_timeout(timeout: int) -> None:
    """Reject timeouts that the provider cannot enforce."""
    if not 1 <= timeout <= _MAX_TIMEOUT:
        msg = f"Mainbrella command timeout must be between 1 and {_MAX_TIMEOUT} seconds"
        raise ValueError(msg)


def _valid_path(path: str) -> bool:
    """Match Mainbrella's absolute guest path rules."""
    try:
        size = len(path.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    return (
        path.startswith("/")
        and "\0" not in path
        and size <= _MAX_PATH_BYTES
        and (
            path == "/"
            or all(part not in ("", ".", "..") for part in path[1:].split("/"))
        )
    )


def _file_error(error: MainbrellaError) -> str:
    """Keep service failures distinct from missing files."""
    return {
        "file_not_found": FILE_NOT_FOUND,
        "invalid_file_path": INVALID_PATH,
        "not_regular_file": INVALID_PATH,
        "file_access_denied": PERMISSION_DENIED,
    }.get(error.code, error.code)


class MainbrellaSandbox(BaseSandbox):
    """Wrap an existing Mainbrella container generation.

    Inherited file tools require `python3` in the guest. Use the `python`
    catalog image or a custom image that includes Python and GNU coreutils.
    """

    def __init__(self, *, sandbox: Sandbox, timeout: int = 60) -> None:
        """Initialize the backend without provisioning a container.

        Args:
            sandbox: Existing Mainbrella SDK sandbox, including its generation.
            timeout: Default command timeout in seconds, from 1 to 900.

        Raises:
            ValueError: If the timeout exceeds Mainbrella's supported range.
        """
        _validate_timeout(timeout)
        self._sandbox = sandbox
        self._default_timeout = timeout

    @property
    def id(self) -> str:
        """Return `slot@createdAt`, preserving the exact container generation."""
        return f"{self._sandbox.id}@{self._sandbox.created_at}"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run a shell command without automatically retrying side effects.

        Args:
            command: Command interpreted by `/bin/sh -lc` in the guest.
            timeout: Command timeout in seconds, from 1 to 900. If omitted,
                uses the backend default. Commands over 60 seconds use managed
                execution and occupy a retained job slot for one hour.

        Returns:
            Combined stdout and stderr with exit status and truncation flag.

        Raises:
            ValueError: If the timeout is outside the supported range.
            MainbrellaError: If the command cannot be submitted or observed.
        """
        effective_timeout = self._default_timeout if timeout is None else timeout
        _validate_timeout(effective_timeout)
        result = self._run(command, effective_timeout)
        output = result["stdout"]
        if result["stderr"]:
            output += f"\n<stderr>{result['stderr']}</stderr>"
        if result["timedOut"]:
            output += f"\nCommand timed out after {effective_timeout} seconds"
        return ExecuteResponse(
            output=output,
            exit_code=124 if result["timedOut"] else result["exitCode"],
            truncated=result["outputTruncated"],
        )

    def _run(self, command: str, timeout: int) -> _CommandResult:
        if timeout <= _FOREGROUND_TIMEOUT:
            return cast(
                "_CommandResult",
                self._sandbox.commands.run(command, timeout_ms=timeout * 1000),
            )
        job = self._sandbox.commands.start(command, timeout_ms=timeout * 1000)
        try:
            # Allow the provider to report termination after its execution deadline.
            return cast("_CommandResult", job.wait(timeout=timeout + 30))
        except MainbrellaError:
            try:
                job.cancel()
            except MainbrellaError:
                logger.warning(
                    "Could not cancel Mainbrella job %s in %s",
                    job.id,
                    self.id,
                    exc_info=True,
                )
            raise

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download regular files as bytes, up to 1 MiB per file.

        Args:
            paths: Absolute guest paths.

        Returns:
            One result per path, preserving order and individual errors.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            if not _valid_path(path):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error=INVALID_PATH)
                )
                continue
            try:
                content = self._sandbox.files.read(path)
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except MainbrellaError as error:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error=_file_error(error)
                    )
                )
        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload binary files to existing directories, up to 1 MiB each.

        Args:
            files: Absolute guest paths paired with file bytes.

        Returns:
            One result per file, preserving order and individual errors.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            if not _valid_path(path):
                responses.append(FileUploadResponse(path=path, error=INVALID_PATH))
                continue
            if len(content) > _MAX_FILE_BYTES:
                responses.append(FileUploadResponse(path=path, error="file_too_large"))
                continue
            try:
                self._sandbox.files.write(path, content)
                responses.append(FileUploadResponse(path=path, error=None))
            except MainbrellaError as error:
                responses.append(
                    FileUploadResponse(path=path, error=_file_error(error))
                )
        return responses
