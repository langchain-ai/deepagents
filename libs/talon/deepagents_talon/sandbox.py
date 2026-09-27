"""Opt-in remote sandbox backend for Talon's agent tools.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from deepagents.backends import CompositeBackend
    from deepagents.backends.protocol import BackendProtocol, SandboxBackendProtocol

    from deepagents_talon.config import TalonConfig

logger = logging.getLogger(__name__)

_HOST_PATH_ENV_KEYS = (
    "DEEPAGENTS_TALON_SKILLS_DIRS",
    "SKILLS_DIRS",
    "DEEPAGENTS_TALON_MEMORY_PATHS",
    "AGENT_MEMORY_PATHS",
)


class SandboxStartupError(RuntimeError):
    """Raised when a configured sandbox cannot be started."""


@dataclass(frozen=True, slots=True)
class SandboxSettings:
    """Sandbox provider selection read from `DEEPAGENTS_TALON_SANDBOX*`.

    Args:
        provider: Sandbox provider name resolved by the `deepagents-code` registry.
        sandbox_id: Existing sandbox to attach to. Talon never deletes it.
        snapshot: Snapshot or blueprint for providers that support one.
        setup_script: Host path to a script run once after the sandbox starts.
    """

    provider: str
    sandbox_id: str | None = None
    snapshot: str | None = None
    setup_script: str | None = None


@dataclass(frozen=True, slots=True)
class SandboxSession:
    """A started sandbox plus the backend Talon's agent should use."""

    backend: CompositeBackend
    working_dir: str


@asynccontextmanager
async def open_sandbox(config: TalonConfig) -> AsyncIterator[SandboxSession | None]:
    """Start the configured sandbox for the host lifetime.

    Yields `None` when `DEEPAGENTS_TALON_SANDBOX` is unset. An owned sandbox is
    deleted on exit; one attached with `DEEPAGENTS_TALON_SANDBOX_ID` is kept.

    Args:
        config: Talon runtime configuration.

    Yields:
        The sandbox session, or `None` when sandboxing is disabled.

    Raises:
        SandboxStartupError: If the sandbox is configured but cannot start.
            Talon never falls back to host execution in that case.
    """
    settings = config.sandbox
    if settings is None:
        yield None
        return
    _warn_host_paths(config.env)
    stack = ExitStack()
    sandbox, working_dir = await asyncio.to_thread(_enter_sandbox, stack, settings)
    try:
        backend = sandbox_backend(sandbox, config.manifest_dir)
        yield SandboxSession(backend, working_dir)
    finally:
        await asyncio.to_thread(stack.close)


def sandbox_backend(sandbox: SandboxBackendProtocol, assistant_dir: Path) -> CompositeBackend:
    """Route everything to the sandbox except host-side skills and memory.

    Only `skills/` and `memory/` are routed to the host, so the sandboxed
    agent cannot rewrite `tools.json` or other assistant state.

    Args:
        sandbox: Started sandbox backend; also receives every `execute` call.
        assistant_dir: Talon assistant directory holding skills and memory.

    Returns:
        Composite backend for `DeepAgentRuntime`.
    """
    from deepagents.backends import (  # noqa: PLC0415  # keep CLI startup light
        CompositeBackend,
        FilesystemBackend,
    )

    routes: dict[str, BackendProtocol] = {}
    for name in ("skills", "memory"):
        root = assistant_dir / name
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        routes[f"{root}/"] = FilesystemBackend(root_dir=root, virtual_mode=True)
    return CompositeBackend(default=sandbox, routes=routes)


def _enter_sandbox(
    stack: ExitStack, settings: SandboxSettings
) -> tuple[SandboxBackendProtocol, str]:
    from deepagents_code.integrations.sandbox_factory import (  # noqa: PLC0415  # optional provider SDKs load only when sandboxing is on
        create_sandbox,
        get_default_working_dir,
    )

    try:
        working_dir = get_default_working_dir(settings.provider)
        sandbox = stack.enter_context(
            create_sandbox(
                settings.provider,
                sandbox_id=settings.sandbox_id,
                snapshot_name=settings.snapshot,
                setup_script_path=settings.setup_script,
            )
        )
    except Exception as exc:
        stack.close()
        msg = f"Could not start {settings.provider!r} sandbox: {exc}"
        raise SandboxStartupError(msg) from exc
    return sandbox, working_dir


def _warn_host_paths(env: Mapping[str, str]) -> None:
    for key in _HOST_PATH_ENV_KEYS:
        if env.get(key):
            logger.warning(
                "%s is set but sandbox mode only routes the assistant's skills/ and "
                "memory/ directories to the host; other paths resolve inside the sandbox",
                key,
            )
