"""Failure classification for eval trial results.

Categorizes failures as infrastructure (OOM, timeout, sandbox) vs. model
capability using exit codes and text pattern matching.
"""

from __future__ import annotations

import re
from enum import Enum


class FailureCategory(Enum):
    """Classification of trial failures.

    Distinguishes infrastructure failures from model capabilitiy failures.
    """

    CAPABILITY = "capability"
    """Model produced wrong answer, incomplete solution, or logic error."""

    INFRA_OOM = "infra_oom"
    """Out-of-memory kill (exit code 137 / signal 9)."""

    INFRA_TIMEOUT = "infra_timeout"
    """Command or task exceeded time limit (exit code 124)."""

    INFRA_SANDBOX = "infra_sandbox"
    """Sandbox crash, network failure, or other environment error."""

    UNKNOWN = "unknown"
    """Could not determine failure category."""

    @property
    def is_infrastructure(self) -> bool:
        """Whether this failure is caused by infrastructure rather than model capability."""
        return self in {
            FailureCategory.INFRA_OOM,
            FailureCategory.INFRA_TIMEOUT,
            FailureCategory.INFRA_SANDBOX,
        }


_OOM_EXIT_CODES = {137}
"""Exit codes indicating the process was killed due to out-of-memory.

137 = 128 + SIGKILL(9), typically sent by the Linux OOM killer.
"""

_TIMEOUT_EXIT_CODES = {124}
"""Exit codes indicating the process exceeded a time limit.

124 = GNU coreutils `timeout` convention.
"""

_OOM_PATTERNS = (
    "oomkilled",
    "out of memory",
    "cannot allocate memory",
    "memory allocation failed",
    "signal 9",
    "sigkill",
    "exit code 137",
)
"""Case-insensitive substrings in exception text that signal an OOM kill."""

_TIMEOUT_PATTERNS = (
    "timed out",
    "deadline exceeded",
    "exit code 124",
)
"""Case-insensitive substrings in exception text that signal a timeout."""

_SANDBOX_PATTERNS = (
    "sandbox",
    "connection refused",
    "connection reset",
    "broken pipe",
    "network unreachable",
    "no route to host",
    "exec failed",
)
"""Case-insensitive substrings in exception text that signal a sandbox or
network-isolation failure."""


def extract_exit_codes(trajectory_text: str) -> list[int]:
    """Extract non-zero exit codes from trajectory JSON text.

    Scans observation results in the trajectory for exit code patterns commonly
    emitted by sandbox execute commands.

    Args:
        trajectory_text: Raw JSON text of the trajectory.

    Returns:
        List of non-zero exit codes found.
    """
    codes: list[int] = []
    # Match exit_code/exit code/exit-code variants (dot is a wildcard)
    # e.g. 'exit_code": 137', 'exit code: 1', 'exit-code 124'
    for match in re.finditer(r'(?:exit.code["\s:]+)(\d+)', trajectory_text, re.IGNORECASE):
        code = int(match.group(1))
        if code != 0:
            codes.append(code)
    return codes


def classify_failure(
    *,
    exception_text: str | None = None,
    exit_codes: list[int] | None = None,
    trajectory_text: str | None = None,
) -> FailureCategory:
    """Classify a trial failure as infrastructure or capability.

    Uses exit codes, exception messages, and trajectory content to determine
    whether a failure was caused by infrastructure issues (OOM, timeout, sandbox
    crash) or by the model's capability.

    Args:
        exception_text: Content of `exception.txt` if present.
        exit_codes: List of non-zero exit codes observed during the trial.
        trajectory_text: Raw trajectory JSON text for pattern matching.

    Returns:
        The determined failure category.
    """
    all_text = ""
    if exception_text:
        all_text += exception_text.lower()
    if trajectory_text:
        all_text += trajectory_text.lower()

    # Check exit codes first (most reliable signal)
    if exit_codes:
        for code in exit_codes:
            if code in _OOM_EXIT_CODES:
                return FailureCategory.INFRA_OOM
            if code in _TIMEOUT_EXIT_CODES:
                return FailureCategory.INFRA_TIMEOUT

    # Check text patterns
    if any(p in all_text for p in _OOM_PATTERNS):
        return FailureCategory.INFRA_OOM

    if any(p in all_text for p in _TIMEOUT_PATTERNS):
        return FailureCategory.INFRA_TIMEOUT

    if any(p in all_text for p in _SANDBOX_PATTERNS):
        return FailureCategory.INFRA_SANDBOX

    # Exception text present but no infra signals — classify as unknown
    if exception_text:
        return FailureCategory.UNKNOWN

    # Default: capability failure (wrong answer, not infra)
    return FailureCategory.CAPABILITY
