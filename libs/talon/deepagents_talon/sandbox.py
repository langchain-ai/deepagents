"""Opt-in remote sandbox backend for Talon's agent tools.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
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

_SKILLS_ENV_KEYS = ("DEEPAGENTS_TALON_SKILLS_DIRS", "SKILLS_DIRS")
_SNAPSHOT_HINT = (
    "Snapshot names are shared across a LangSmith workspace; set "
    "DEEPAGENTS_TALON_SANDBOX_SNAPSHOT to a name you own"
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
        default_snapshot: Snapshot to use when neither `snapshot` nor the
            provider's own snapshot-name environment variable is set.
    """

    provider: str
    sandbox_id: str | None = None
    snapshot: str | None = None
    setup_script: str | None = None
    default_snapshot: str | None = None


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
    handoff = _Handoff()
    try:
        sandbox, working_dir = await asyncio.to_thread(_enter_sandbox, handoff, settings)
    except asyncio.CancelledError:
        if handoff.abandon():
            await asyncio.to_thread(handoff.stack.close)
        raise
    try:
        backend = sandbox_backend(sandbox, config.manifest_dir)
        yield SandboxSession(backend, working_dir)
    finally:
        await asyncio.to_thread(handoff.stack.close)


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


class _Handoff:
    """Pass a started sandbox from its worker thread to the awaiting task.

    Cancelling `asyncio.to_thread` does not stop the thread, so a sandbox can
    finish starting after its caller is gone. Whichever side learns last that
    the other has let go closes the stack, so an owned sandbox is never leaked.
    """

    def __init__(self) -> None:
        self.stack = ExitStack()
        self._lock = threading.Lock()
        self._abandoned = False
        self._delivered = False

    def deliver(self) -> None:
        """Hand the sandbox over, or close it if the caller was cancelled."""
        with self._lock:
            self._delivered = not self._abandoned
        if not self._delivered:
            self.stack.close()

    def abandon(self) -> bool:
        """Mark the caller cancelled; return whether it must close the stack."""
        with self._lock:
            self._abandoned = True
            return self._delivered


def _enter_sandbox(
    handoff: _Handoff, settings: SandboxSettings
) -> tuple[SandboxBackendProtocol, str]:
    from deepagents_code.integrations.sandbox_factory import (  # noqa: PLC0415  # optional provider SDKs load only when sandboxing is on
        create_sandbox,
        get_default_working_dir,
    )

    try:
        working_dir = get_default_working_dir(settings.provider)
        sandbox = handoff.stack.enter_context(
            create_sandbox(
                settings.provider,
                sandbox_id=settings.sandbox_id,
                snapshot_name=_snapshot_name(settings),
                setup_script_path=settings.setup_script,
            )
        )
    except Exception as exc:
        handoff.stack.close()
        msg = f"Could not start {settings.provider!r} sandbox: {exc}"
        if "snapshot" in str(exc).lower():
            msg += f". {_SNAPSHOT_HINT}"
        raise SandboxStartupError(msg) from exc
    handoff.deliver()
    return sandbox, working_dir


def _snapshot_name(settings: SandboxSettings) -> str | None:
    if settings.snapshot is not None or settings.default_snapshot is None:
        return settings.snapshot
    from deepagents_code.model_config import (  # noqa: PLC0415  # same lookup the provider uses
        resolve_env_var,
    )

    # Defer to the provider's own override, including its `DEEPAGENTS_CODE_` prefix.
    if resolve_env_var("LANGSMITH_SANDBOX_SNAPSHOT_NAME"):
        return None
    return settings.default_snapshot


def _warn_host_paths(env: Mapping[str, str]) -> None:
    for key in _SKILLS_ENV_KEYS:
        if env.get(key):
            logger.warning(
                "%s is set but sandbox mode only reads host skills from the assistant's "
                "skills/ directory; these paths resolve inside the sandbox",
                key,
            )
