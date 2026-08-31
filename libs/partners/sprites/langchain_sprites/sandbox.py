"""Sprites sandbox backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from sprites.exceptions import (
    FileNotFoundError_,
    FilesystemError,
    IsADirectoryError_,
    PermissionError_,
)
from sprites.exceptions import (
    TimeoutError as SpritesTimeoutError,
)

if TYPE_CHECKING:
    import sprites

COMMAND_TIMEOUT_EXIT_CODE = 124

_DEFAULT_WORKDIR = "/home/sprite"


_ERROR_TYPES: list[tuple[type[FilesystemError], str]] = [
    (FileNotFoundError_, "file_not_found"),
    (PermissionError_, "permission_denied"),
    (IsADirectoryError_, "is_directory"),
]

# The server does not always return a structured error; a missing file can
# surface as a generic FilesystemError with the raw OS error string. Order
# matters: "no such file or directory" must match before the directory rule.
_ERROR_MESSAGES: list[tuple[tuple[str, ...], str]] = [
    (("no such file", "not found"), "file_not_found"),
    (("permission",), "permission_denied"),
    (("directory",), "is_directory"),
]


def _map_filesystem_error(error: Exception) -> str:
    """Map a Sprites SDK filesystem error to a protocol error code."""
    for error_type, code in _ERROR_TYPES:
        if isinstance(error, error_type):
            return code
    message = str(error).lower()
    for needles, code in _ERROR_MESSAGES:
        if any(needle in message for needle in needles):
            return code
    return "invalid_path"


class SpritesSandbox(BaseSandbox):
    """Fly.io Sprites sandbox implementation conforming to SandboxBackendProtocol.

    This implementation inherits all file operation methods from BaseSandbox and
    implements execute(), upload_files(), and download_files() using the Sprites
    SDK.

    Sprites are persistent, named Linux VMs. They start in 1 to 2 seconds, stop
    automatically when idle (at no cost while stopped), and support fast
    checkpoint and restore of the full machine state. Because a Sprite is
    persistent, the same sandbox can be picked up again later by name with all
    files and installed packages intact.
    """

    def __init__(
        self,
        *,
        sprite: sprites.Sprite,
        timeout: int = 300,
        workdir: str = _DEFAULT_WORKDIR,
    ) -> None:
        """Create a backend wrapping an existing Sprite.

        Args:
            sprite: Existing `sprites.Sprite` instance to wrap (from
                `SpritesClient.create_sprite()` or `SpritesClient.sprite()`).
            timeout: Default command timeout in seconds used when `execute()` is
                called without an explicit `timeout`.
            workdir: Working directory for command execution and for resolving
                relative file paths.
        """
        self._sprite = sprite
        self._default_timeout = timeout
        self._workdir = workdir

    @property
    def id(self) -> str:
        """Return the Sprite name.

        The name is the Sprite's persistent identity: it is stable across idle
        suspends and can be used to reconnect later via
        `SpritesClient.sprite(name)`.
        """
        return self._sprite.name

    @property
    def sprite(self) -> sprites.Sprite:
        """Return the wrapped Sprite for direct SDK access.

        Use this for Sprites features beyond the sandbox protocol, such as
        `create_checkpoint()`, `restore_checkpoint()`, and services.
        """
        return self._sprite

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command inside the sandbox.

        The command runs with `bash -lc` in the sandbox working directory, so
        shell features (pipes, redirection, globbing) work as expected.

        Args:
            command: Shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to
                complete. If None, uses the backend's default timeout.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        try:
            result = self._sprite.run(
                "bash",
                "-lc",
                command,
                capture_output=True,
                cwd=self._workdir,
                timeout=effective_timeout,
            )
        except SpritesTimeoutError:
            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=COMMAND_TIMEOUT_EXIT_CODE,
                truncated=False,
            )

        output = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if stderr.strip():
            output += f"\n<stderr>{stderr.strip()}</stderr>"

        return ExecuteResponse(
            output=output,
            exit_code=result.returncode,
            truncated=False,
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.

        Each file is read individually, allowing partial success when some
        files exist and others don't. Paths must be absolute.
        """
        fs = self._sprite.filesystem(self._workdir)
        responses: list[FileDownloadResponse] = []

        for path in paths:
            if not path.startswith("/"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
                continue
            try:
                content = (fs / path).read_bytes()
            except FilesystemError as error:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error=_map_filesystem_error(error),
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files into the sandbox.

        Parent directories are created automatically. Paths must be absolute.
        """
        fs = self._sprite.filesystem(self._workdir)
        responses: list[FileUploadResponse] = []

        for path, content in files:
            if not path.startswith("/"):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
                continue
            try:
                (fs / path).write_bytes(content)
            except FilesystemError as error:
                responses.append(
                    FileUploadResponse(path=path, error=_map_filesystem_error(error))
                )
            else:
                responses.append(FileUploadResponse(path=path, error=None))

        return responses
