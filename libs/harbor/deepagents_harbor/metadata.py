"""Infrastructure metadata collection for eval trials.

Captures host and sandbox environment details (CPU, memory, OS) to enable
post-hoc analysis of infrastructure noise in eval results.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SandboxLike(Protocol):
    """Structural protocol for objects usable by `collect_sandbox_metadata`.

    Any object exposing an `environment` attribute and an async `aexecute`
    method satisfies this protocol — including `HarborSandbox` and test fakes.
    """

    environment: Any
    """Harbor environment instance used to resolve sandbox type metadata."""

    async def aexecute(  # noqa: D102
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any: ...  # noqa: ANN401


logger = logging.getLogger(__name__)


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
