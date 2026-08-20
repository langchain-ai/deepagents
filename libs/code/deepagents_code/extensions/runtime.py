"""Discover, load, and tear down Python extensions."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy
from deepagents_code.extensions.discovery import (
    discover_extension_files,
    project_extensions_dir,
)
from deepagents_code.extensions.loader import load_extension
from deepagents_code.extensions.models import ExtensionError
from deepagents_code.extensions.registry import ExtensionRegistry
from deepagents_code.extensions.settings import TrustPolicy, load_extension_settings
from deepagents_code.extensions.trust import is_project_extensions_trusted

if TYPE_CHECKING:
    from deepagents_code.extensions.models import SourceInfo

logger = logging.getLogger(__name__)
INTERACTIVE_MODE = "interactive"
HEADLESS_MODE = "headless"


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Registrations and isolated errors from one load pass."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    errors: tuple[str, ...] = ()


_shutdown_registry: ExtensionRegistry | None = None


def _prepare(
    cwd: Path | None,
    project_root: Path | None,
    project_trust_granted: bool,
) -> tuple[list[SourceInfo], Path] | None:
    if not is_env_truthy(EXPERIMENTAL):
        return None
    settings = load_extension_settings()
    if not settings.enabled:
        return None
    project_dir = None
    if project_root is not None and settings.trust is not TrustPolicy.NEVER:
        trusted = (
            project_trust_granted
            or settings.trust is TrustPolicy.ALWAYS
            or is_project_extensions_trusted(project_root)
        )
        if trusted:
            project_dir = project_extensions_dir(project_root)
    from deepagents_code.plugins import discover_plugins

    plugin_result = discover_plugins()
    for warning in plugin_result.warnings:
        logger.warning("Plugin extension discovery: %s", warning)
    sources = discover_extension_files(
        plugins=plugin_result.plugins,
        project_dir=project_dir,
    )
    return (sources, Path.cwd() if cwd is None else cwd) if sources else None


async def load_extensions(
    *,
    cwd: Path | None = None,
    mode: str = INTERACTIVE_MODE,
    project_root: Path | None = None,
    project_trust_granted: bool = False,
) -> ExtensionLoadResult:
    """Load every authorized extension while isolating failures.

    Args:
        cwd: Working directory exposed to factories.
        mode: Runtime mode.
        project_root: Project whose local extensions may be considered.
        project_trust_granted: Explicit one-run project grant.

    Returns:
        Successful registrations and error messages.
    """
    prepared = await asyncio.to_thread(
        _prepare,
        cwd,
        project_root,
        project_trust_granted,
    )
    if prepared is None:
        return ExtensionLoadResult()
    sources, session_cwd = prepared
    registry = ExtensionRegistry()
    errors: list[str] = []
    for source in sources:
        try:
            await load_extension(source, registry, cwd=session_cwd, mode=mode)
        except ExtensionError as exc:
            logger.warning("Skipping extension %s", source.path, exc_info=True)
            errors.append(str(exc))
        except Exception as exc:
            logger.exception("Unexpected extension failure: %s", source.path)
            errors.append(f"{source.path}: {type(exc).__name__}: {exc}")
    return ExtensionLoadResult(registry, tuple(errors))


def bind_server_extensions(registry: ExtensionRegistry) -> None:
    """Retain server-owned shutdown callbacks for lifespan teardown."""
    global _shutdown_registry  # noqa: PLW0603
    _shutdown_registry = registry if registry.shutdown_hooks else None


async def shutdown_extensions(registry: ExtensionRegistry) -> None:
    """Run teardown callbacks while isolating individual failures.

    Args:
        registry: Registry whose session is ending.
    """
    for hook in registry.shutdown_hooks:
        try:
            result = hook.unit()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "Shutdown hook from %s failed", hook.source.label, exc_info=True
            )


async def shutdown_server_extensions() -> None:
    """Release server-owned extensions on the persistent event loop."""
    global _shutdown_registry
    registry, _shutdown_registry = _shutdown_registry, None
    if registry is not None:
        await shutdown_extensions(registry)
