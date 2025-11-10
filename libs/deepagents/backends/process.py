"""Abstraction for modeling a process."""

import abc
from dataclasses import dataclass
from typing import TypedDict


class ProcessCapabilities(TypedDict):
    """Capabilities of the process backend."""

    supports_exec: bool


@dataclass
class ExecuteResponse:
    """Result of code execution.

    Simplified schema optimized for LLM consumption.
    """

    output: str
    """Combined stdout and stderr output of the executed command."""

    exit_code: int | None = None
    """The process exit code. 0 indicates success, non-zero indicates failure."""

    signal: str | None = None
    """The signal that terminated the process (e.g., 'SIGTERM', 'SIGKILL'), if applicable."""

    truncated: bool = False
    """Whether the output was truncated due to backend limitations."""


class Process(abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        *,
        timeout: int = 30 * 60,
    ) -> ExecuteResponse:
        """Execute a command in the process.

        Simplified interface optimized for LLM consumption.

        Args:
            command: Full shell command string to execute.
            cwd: Working directory to execute the command in (absolute path).
            env: Environment variables for the command (dict of name -> value).
            timeout: Maximum execution time in seconds (default: 30 minutes).

        Returns:
            ExecuteResponse with combined output, exit code, optional signal, and truncation flag.
        """
        ...

    @abc.abstractmethod
    def get_capabilities(self) -> ProcessCapabilities: ...
