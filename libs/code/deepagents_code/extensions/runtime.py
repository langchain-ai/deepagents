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
    discover_extensions,
    project_extensions_dir,
)
from deepagents_code.extensions.loader import load_extension
from deepagents_code.extensions.registry import ExtensionError, ExtensionRegistry
from deepagents_code.extensions.settings import TrustPolicy, load_extension_settings
from deepagents_code.extensions.trust import is_project_extensions_trusted

if TYPE_CHECKING:
    from deepagents_code.extensions.registry import SourceInfo

logger = logging.getLogger(__name__)
INTERACTIVE_MODE = "interactive"
HEADLESS_MODE = "headless"


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Registrations and isolated errors from one load pass."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    errors: tuple[str, ...] = ()
    active: bool = False
    """Whether at least one authorized extension source activated the runtime."""


_shutdown_registry: ExtensionRegistry | None = None
_server_errors: tuple[str, ...] = ()


def _prepare(
    cwd: Path | None,
    project_root: Path | None,
    project_trust_granted: bool,
    cli_paths: tuple[Path, ...],
) -> tuple[tuple[SourceInfo, ...], tuple[str, ...], Path] | None:
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
    discovery = discover_extensions(
        plugins=plugin_result.plugins,
        config_files=settings.extra_files,
        config_dirs=settings.extra_dirs,
        cli_paths=cli_paths,
        project_dir=project_dir,
    )
    if not discovery.sources and not discovery.errors:
        return None
    return discovery.sources, discovery.errors, Path.cwd() if cwd is None else cwd


async def load_extensions(
    *,
    cwd: Path | None = None,
    mode: str = INTERACTIVE_MODE,
    project_root: Path | None = None,
    project_trust_granted: bool = False,
    cli_paths: tuple[Path, ...] = (),
) -> ExtensionLoadResult:
    """Load every authorized extension while isolating failures.

    Args:
        cwd: Working directory exposed to factories.
        mode: Runtime mode.
        project_root: Project whose local extensions may be considered.
        project_trust_granted: Explicit one-run project grant.
        cli_paths: Explicit one-run extension files or directories.

    Returns:
        Successful registrations and error messages.
    """
    prepared = await asyncio.to_thread(
        _prepare,
        cwd,
        project_root,
        project_trust_granted,
        cli_paths,
    )
    if prepared is None:
        return ExtensionLoadResult()
    sources, discovery_errors, session_cwd = prepared
    registry = ExtensionRegistry()
    errors = list(discovery_errors)
    for source in sources:
        try:
            await load_extension(source, registry, cwd=session_cwd, mode=mode)
        except ExtensionError as exc:
            logger.warning("Skipping extension %s", source.path, exc_info=True)
            errors.append(str(exc))
        except Exception as exc:
            logger.exception("Unexpected extension failure: %s", source.path)
            errors.append(f"{source.path}: {type(exc).__name__}: {exc}")
    return ExtensionLoadResult(registry, tuple(errors), active=True)


def bind_server_extensions(
    registry: ExtensionRegistry, *, errors: tuple[str, ...] = ()
) -> None:
    """Retain server-owned shutdown callbacks for lifespan teardown."""
    global _server_errors, _shutdown_registry  # noqa: PLW0603
    _shutdown_registry = registry
    _server_errors = errors


def server_extension_report() -> dict[str, object]:
    """Return sanitized metadata for the local provenance endpoint."""
    registry = _shutdown_registry
    registrations = () if registry is None else registry.registrations()
    return {
        "registrations": [
            {"kind": kind, "name": item.name, "source": item.source.as_dict()}
            for kind, item in registrations
            if kind != "shutdown"
        ],
        "errors": list(_server_errors),
        "restart_required": registry.restart_required
        if registry is not None
        else False,
    }


async def shutdown_extensions(registry: ExtensionRegistry) -> None:
    """Run teardown callbacks while isolating individual failures.

    Args:
        registry: Registry whose session is ending.
    """
    try:
        for hook in reversed(registry.shutdown_hooks):
            try:
                result = hook.unit()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning(
                    "Shutdown hook from %s failed", hook.source.label, exc_info=True
                )
    finally:
        registry.deactivate_apis()


async def shutdown_server_extensions() -> None:
    """Release server-owned extensions on the persistent event loop."""
    global _server_errors, _shutdown_registry  # noqa: PLW0603
    registry, _shutdown_registry = _shutdown_registry, None
    _server_errors = ()
    if registry is not None:
        await shutdown_extensions(registry)
