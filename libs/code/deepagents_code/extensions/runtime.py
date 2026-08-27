"""Discover, load, and tear down Python extensions."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code._env_vars import EXPERIMENTAL, is_env_truthy
from deepagents_code.extensions.api import ExtensionAPI, ExtensionMode
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


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Registrations and isolated errors from one load pass."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    errors: tuple[str, ...] = ()
    active: bool = False
    """Whether at least one authorized extension source activated the runtime."""
    _apis: tuple[ExtensionAPI, ...] = field(default=(), repr=False)


_server_extensions: ExtensionLoadResult | None = None


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
        config_paths=settings.extra_paths,
        cli_paths=cli_paths,
        project_dir=project_dir,
    )
    if not discovery.sources and not discovery.errors:
        return None
    return discovery.sources, discovery.errors, Path.cwd() if cwd is None else cwd


async def load_extensions(
    *,
    cwd: Path | None = None,
    mode: ExtensionMode = ExtensionMode.INTERACTIVE,
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
    apis: list[ExtensionAPI] = []
    for source in sources:
        try:
            api = await load_extension(source, registry, cwd=session_cwd, mode=mode)
        except ExtensionError as exc:
            logger.warning("Skipping extension %s", source.path, exc_info=True)
            errors.append(str(exc))
        except Exception as exc:
            logger.exception("Unexpected extension failure: %s", source.path)
            errors.append(f"{source.path}: {type(exc).__name__}: {exc}")
        else:
            apis.append(api)
    return ExtensionLoadResult(
        registry,
        tuple(errors),
        active=True,
        _apis=tuple(apis),
    )


def bind_server_extensions(extensions: ExtensionLoadResult) -> None:
    """Retain the server-owned extension runtime for lifespan teardown."""
    global _server_extensions  # noqa: PLW0603
    _server_extensions = extensions


def server_extension_report() -> dict[str, object]:
    """Return sanitized metadata for the local provenance endpoint."""
    extensions = _server_extensions
    registry = None if extensions is None else extensions.registry
    registrations = () if registry is None else registry.registrations()
    return {
        "registrations": [
            {"kind": kind, "name": item.name, "source": item.source.as_dict()}
            for kind, item in registrations
            if kind != "shutdown"
        ],
        "errors": [] if extensions is None else list(extensions.errors),
        "restart_required": registry.restart_required
        if registry is not None
        else False,
    }


async def shutdown_extensions(extensions: ExtensionLoadResult) -> None:
    """Run teardown callbacks while isolating individual failures.

    Args:
        extensions: Extension runtime whose session is ending.
    """
    registry = extensions.registry
    try:
        for hook in reversed(registry.shutdown_hooks):
            try:
                result = hook.unit()
                if inspect.isawaitable(result):
                    await result
            except (Exception, SystemExit):
                logger.warning(
                    "Shutdown hook from %s failed", hook.source.label, exc_info=True
                )
    finally:
        for api in extensions._apis:
            api._deactivate()


async def shutdown_server_extensions() -> None:
    """Release server-owned extensions on the persistent event loop."""
    global _server_extensions
    extensions, _server_extensions = _server_extensions, None
    if extensions is not None:
        await shutdown_extensions(extensions)
