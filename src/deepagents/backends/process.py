"""Abstraction for modeling a process."""

import abc
from typing import NotRequired, TypedDict


class ProcessCapabilities(TypedDict):
    """Capabilities of the process backend."""

    supports_exec: bool


class ExecuteResponse(TypedDict):
    """Result of code execution."""

    result: str
    """The output of the executed code.

    This will usually be the standard output of the command executed.
    """
    exit_code: NotRequired[int]
    """The exit code of the executed code, if applicable."""


class Process(abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        command: str,
        cwd: str | None = None,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the process.

        Args:
            command: Command to execute as a string.
            cwd: Working directory to execute the command in.
            timeout: Maximum execution time in seconds (default: 30 minutes).
        """
        ...

    @abc.abstractmethod
    def get_capabilities(self) -> ProcessCapabilities: ...
