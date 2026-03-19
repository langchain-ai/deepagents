"""Failure classification for eval trial results.

Categorizes failures as infrastructure (OOM, timeout, sandbox) vs. model
capability using exit codes and text pattern matching.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Classification of trial failures.

    Distinguishes infrastructure failures from model capability failures.
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
    "sandbox crashed",
    "sandbox exited unexpectedly",
    "sandbox error",
    "sandbox failure",
    "connection refused",
    "connection reset",
    "broken pipe",
    "network unreachable",
    "no route to host",
    "exec failed",
)
"""Case-insensitive substrings in exception text that signal a sandbox or
network-isolation failure."""


def _extract_observation_texts(trajectory_json: str) -> list[str] | None:
    """Extract observation result content from parsed ATIF trajectory JSON.

    Only returns text from observation results (tool outputs).

    Args:
        trajectory_json: Raw JSON text of the trajectory.

    Returns:
        List of observation content strings, or `None` if the JSON could not be
            parsed as a valid ATIF trajectory (triggers raw fallback).
    """
    try:
        data = json.loads(trajectory_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("Failed to parse trajectory JSON for observation extraction")
        return None

    if not isinstance(data, dict) or "steps" not in data:
        return None

    texts: list[str] = []
    for step in data.get("steps", []):
        obs: dict[str, Any] | None = step.get("observation")
        if not obs:
            continue
        for result in obs.get("results", []):
            content = result.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                # ContentPart list (ATIF v1.6+)
                texts.extend(
                    part["text"] for part in content if isinstance(part, dict) and part.get("text")
                )
    return texts


def extract_exit_codes(trajectory_json: str) -> list[int]:
    """Extract non-zero exit codes from ATIF trajectory observation results.

    Parses the trajectory JSON structurally and only searches observation
    content (tool output) for exit code patterns, avoiding false positives from
    model-generated text that discusses exit codes.

    Args:
        trajectory_json: Raw JSON text of the ATIF trajectory.

    Returns:
        List of non-zero exit codes found in observation results.
    """
    observation_texts = _extract_observation_texts(trajectory_json)
    if observation_texts is None:
        # Fall back to regex on raw text if parsing fails (e.g. non-ATIF input)
        return _extract_exit_codes_raw(trajectory_json)
    if not observation_texts:
        return []

    codes: list[int] = []
    for text in observation_texts:
        codes.extend(_extract_exit_codes_raw(text))
    return codes


def _extract_exit_codes_raw(text: str) -> list[int]:
    """Extract non-zero exit codes from a text string using regex.

    Args:
        text: Text to search for exit code patterns.

    Returns:
        List of non-zero exit codes found.
    """
    codes: list[int] = []
    # Match exit_code/exit code/exit-code variants (dot is a wildcard)
    # e.g. 'exit_code": 137', 'exit code: 1', 'exit-code 124'
    for match in re.finditer(r'(?:exit.code["\s:]+)(\d+)', text, re.IGNORECASE):
        code = int(match.group(1))
        if code != 0:
            codes.append(code)
    return codes


def classify_failure(
    *,
    exception_text: str | None = None,
    exit_codes: list[int] | None = None,
) -> FailureCategory:
    """Classify a trial failure as infrastructure or capability.

    Uses exit codes and exception text to determine whether a failure was caused
    by infrastructure issues (OOM, timeout, sandbox crash) or by the
    model's capability.

    Pattern matching is restricted to `exception_text` only (structured,
    controlled output) to avoid false positives from model-generated content
    in trajectories.

    Args:
        exception_text: Content of `exception.txt` if present.
        exit_codes: List of non-zero exit codes observed during the trial.

    Returns:
        The determined failure category.
    """
    # Check exit codes first (most reliable signal)
    if exit_codes:
        for code in exit_codes:
            if code in _OOM_EXIT_CODES:
                return FailureCategory.INFRA_OOM
            if code in _TIMEOUT_EXIT_CODES:
                return FailureCategory.INFRA_TIMEOUT

    # Pattern match only against exception text (not trajectory)
    if exception_text:
        lower = exception_text.lower()

        if any(p in lower for p in _OOM_PATTERNS):
            return FailureCategory.INFRA_OOM

        if any(p in lower for p in _TIMEOUT_PATTERNS):
            return FailureCategory.INFRA_TIMEOUT

        if any(p in lower for p in _SANDBOX_PATTERNS):
            return FailureCategory.INFRA_SANDBOX

        # Exception present but no infra signals — ambiguous
        return FailureCategory.UNKNOWN

    # No exception, no infra exit codes — capability failure
    return FailureCategory.CAPABILITY
