"""Infrastructure metadata collection and statistical utilities for eval noise analysis.

Inspired by Anthropic's research on infrastructure noise in agentic coding evals:
https://www.anthropic.com/engineering/infrastructure-noise

Key insight: infrastructure configuration (resource limits, hardware, time-of-day)
can swing benchmark scores by several percentage points — often exceeding leaderboard
gaps between top models.
"""

from __future__ import annotations

import logging
import math
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SandboxLike(Protocol):
    """Structural protocol for objects usable by `collect_sandbox_metadata`.

    Any object exposing an `environment` attribute and an async `aexecute`
    method satisfies this protocol — including `HarborSandbox` and test fakes.
    """

    environment: Any

    async def aexecute(  # noqa: D102
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any: ...  # noqa: ANN401


logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Classification of trial failures.

    Distinguishes infrastructure failures (not the model's fault) from
    capability failures (genuine model shortcomings). This prevents
    misattributing infrastructure noise as model weakness.
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


# Exit codes and signals that indicate infrastructure failures
_OOM_EXIT_CODES = {137}  # 128 + SIGKILL(9)
_TIMEOUT_EXIT_CODES = {124}  # GNU timeout convention

# Patterns in exception text that indicate infrastructure failures
_OOM_PATTERNS = (
    "oomkilled",
    "out of memory",
    "cannot allocate memory",
    "memory allocation failed",
    "signal 9",
    "sigkill",
    "exit code 137",
)

_TIMEOUT_PATTERNS = (
    "timed out",
    "deadline exceeded",
    "exit code 124",
)

_SANDBOX_PATTERNS = (
    "sandbox",
    "connection refused",
    "connection reset",
    "broken pipe",
    "network unreachable",
    "no route to host",
    "exec failed",
)


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
    whether a failure was caused by infrastructure issues (OOM, timeout,
    sandbox crash) or by the model's capability.

    Args:
        exception_text: Content of exception.txt if present.
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

    # Exception text present but no infra signals — ambiguous, classify as unknown
    if exception_text:
        return FailureCategory.UNKNOWN

    # Default: capability failure (wrong answer, not infra)
    return FailureCategory.CAPABILITY


@dataclass
class InfraMetadata:
    """Infrastructure metadata captured at trial execution time.

    Enables post-hoc analysis of infrastructure noise by recording the execution
    environment details alongside eval results.
    """

    # Host info (captured from orchestrator machine)
    host_platform: str = ""
    host_python_version: str = ""

    # Sandbox info (captured from inside the sandbox)
    sandbox_type: str = ""
    sandbox_cpu_count: int | None = None
    sandbox_memory_total_mb: int | None = None
    sandbox_memory_available_mb: int | None = None
    sandbox_os: str = ""

    # Execution context
    timestamp_utc: str = ""
    concurrency_env: str = ""

    # Resource configuration
    resource_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)


def collect_host_metadata() -> dict[str, str]:
    """Collect metadata from the orchestrator host (non-sandbox).

    Returns:
        Dictionary with host platform and Python version.
    """
    return {
        "host_platform": platform.platform(),
        "host_python_version": platform.python_version(),
    }


async def collect_sandbox_metadata(backend: SandboxLike) -> InfraMetadata:
    """Collect infrastructure metadata from inside the sandbox environment.

    Runs lightweight shell commands to capture CPU, memory, and OS info.
    Designed to be called once at the start of a trial run.

    Args:
        backend: Harbor sandbox backend to query.

    Returns:
        Populated infrastructure metadata.
    """
    meta = InfraMetadata(
        timestamp_utc=datetime.now(UTC).isoformat(),
        concurrency_env=os.environ.get("HARBOR_CONCURRENCY", ""),
        sandbox_type=type(backend.environment).__name__,
    )

    # Collect host info
    host = collect_host_metadata()
    meta.host_platform = host["host_platform"]
    meta.host_python_version = host["host_python_version"]

    # Collect sandbox info via shell commands (best-effort — must never abort a trial)
    try:
        cpu_result = await backend.aexecute(
            "nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 0", timeout=10
        )
        cpu_str = cpu_result.output.strip().split("\n")[0]
        if cpu_str.isdigit():
            meta.sandbox_cpu_count = int(cpu_str)
        else:
            logger.debug("Sandbox CPU count returned non-numeric: %r", cpu_str)
    except Exception:  # noqa: BLE001  # best-effort metadata collection
        logger.debug("Failed to collect sandbox CPU count", exc_info=True)

    try:
        # Linux: /proc/meminfo, fallback to macOS sysctl
        mem_cmd = (
            "grep MemTotal /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' "
            "|| sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1048576)}' "
            "|| echo 0"
        )
        mem_result = await backend.aexecute(mem_cmd, timeout=10)
        mem_str = mem_result.output.strip().split("\n")[0]
        if mem_str.isdigit():
            meta.sandbox_memory_total_mb = int(mem_str)
        else:
            logger.debug("Sandbox memory total returned non-numeric: %r", mem_str)
    except Exception:  # noqa: BLE001  # best-effort metadata collection
        logger.debug("Failed to collect sandbox memory total", exc_info=True)

    try:
        # Available memory (Linux only via /proc/meminfo)
        avail_cmd = (
            "grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print int($2/1024)}' || echo 0"
        )
        avail_result = await backend.aexecute(avail_cmd, timeout=10)
        avail_str = avail_result.output.strip().split("\n")[0]
        if avail_str.isdigit():
            avail = int(avail_str)
            meta.sandbox_memory_available_mb = avail if avail > 0 else None
        else:
            logger.debug("Sandbox memory available returned non-numeric: %r", avail_str)
    except Exception:  # noqa: BLE001  # best-effort metadata collection
        logger.debug("Failed to collect sandbox available memory", exc_info=True)

    try:
        os_result = await backend.aexecute("uname -s -r 2>/dev/null || echo unknown", timeout=10)
        meta.sandbox_os = os_result.output.strip().split("\n")[0]
    except Exception:  # noqa: BLE001  # best-effort metadata collection
        logger.debug("Failed to collect sandbox OS info", exc_info=True)

    return meta


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def wilson_ci(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small samples and
    proportions near 0 or 1. Recommended by Anthropic's infrastructure
    noise research for eval score reporting.

    Args:
        successes: Number of successes (e.g., passed tasks).
        total: Total number of trials.
        z: Z-score for desired confidence level (1.96 = 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) as proportions in [0, 1].
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))

    return (max(0.0, center - margin), min(1.0, center + margin))


def format_ci(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> str:
    """Format a success rate with Wilson confidence interval.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        z: Z-score for desired confidence level.

    Returns:
        Formatted string like "72.3% [68.1%, 76.2%] (95% CI, n=90)".
    """
    if total == 0:
        return "N/A (no trials)"

    rate = (successes / total) * 100
    lo, hi = wilson_ci(successes, total, z=z)
    confidence = math.erf(z / math.sqrt(2)) * 100
    return f"{rate:.1f}% [{lo * 100:.1f}%, {hi * 100:.1f}%] ({confidence:.0f}% CI, n={total})"


def min_detectable_effect(total: int, *, z: float = 1.96, p: float = 0.5) -> float:
    """Estimate minimum detectable effect size for a given sample count.

    Uses the normal approximation for difference in two proportions.
    Conservative estimate assumes p=0.5 (maximum variance).

    Args:
        total: Number of tasks per run.
        z: Z-score for desired confidence level (1.96 = 95% CI).
        p: Assumed base proportion.

    Returns:
        Minimum detectable difference as a proportion (e.g., 0.042 = 4.2pp).
    """
    if total == 0:
        return 1.0
    # Two-sample proportion test: MDE ≈ z * sqrt(2 * p * (1-p) / n)
    return z * math.sqrt(2 * p * (1 - p) / total)
