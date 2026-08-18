"""Orchestrate extension discovery, trust, loading, and teardown."""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_code.extensions.discovery import (
    discover_extension_files,
    project_extensions_dir,
)
from deepagents_code.extensions.loader import load_extension
from deepagents_code.extensions.models import ExtensionError
from deepagents_code.extensions.registry import ExtensionRegistry
from deepagents_code.extensions.settings import (
    ExtensionSettings,
    TrustPolicy,
    load_extension_settings,
)
from deepagents_code.extensions.trust import is_project_extensions_trusted

if TYPE_CHECKING:
    from deepagents_code.extensions.models import ExtensionFile, LoadedExtension

logger = logging.getLogger(__name__)

INTERACTIVE_MODE = "interactive"
HEADLESS_MODE = "headless"


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Outcome of one extension discovery and loading pass."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    loaded: tuple[LoadedExtension, ...] = ()
    errors: tuple[str, ...] = ()


_server_result: ContextVar[ExtensionLoadResult | None] = ContextVar(
    "server_extension_result", default=None
)
_server_shutdown_registry: ExtensionRegistry | None = None


def project_extensions_trusted(
    project_root: Path | str,
    *,
    policy: TrustPolicy,
    granted: bool = False,
    store_path: Path | None = None,
) -> bool:
    """Resolve whether a project's extension directory may be scanned.

    Args:
        project_root: Project root under consideration.
        policy: Default policy when no decision is persisted.
        granted: Explicit one-run trust decision.
        store_path: Alternate persisted trust store for tests.

    Returns:
        `True` only when project extension execution is authorized.
    """
    if policy is TrustPolicy.NEVER:
        return False
    if granted or policy is TrustPolicy.ALWAYS:
        return True
    return is_project_extensions_trusted(project_root, store_path=store_path)


def _prepare_extension_load(
    *,
    cwd: Path | None,
    project_root: Path | None,
    project_trust_granted: bool,
    settings: ExtensionSettings | None,
    trust_store_path: Path | None,
) -> tuple[list[ExtensionFile], Path] | None:
    """Resolve settings, trust, sources, and cwd without executing code.

    Returns:
        Files and session cwd, or `None` when loading is disabled or empty.
    """
    resolved = load_extension_settings() if settings is None else settings
    if not resolved.enabled:
        return None

    trusted_project_dir: Path | None = None
    if project_root is not None and project_extensions_trusted(
        project_root,
        policy=resolved.trust,
        granted=project_trust_granted,
        store_path=trust_store_path,
    ):
        trusted_project_dir = project_extensions_dir(project_root)

    files = discover_extension_files(
        extra_paths=resolved.paths,
        project_dir=trusted_project_dir,
    )
    if not files:
        return None
    return files, Path.cwd() if cwd is None else Path(cwd)


async def load_extensions(
    *,
    cwd: Path | None = None,
    mode: str = INTERACTIVE_MODE,
    project_root: Path | None = None,
    project_trust_granted: bool = False,
    settings: ExtensionSettings | None = None,
    trust_store_path: Path | None = None,
) -> ExtensionLoadResult:
    """Discover and load every authorized extension.

    Args:
        cwd: Working directory for extension context.
        mode: Runtime mode, either `interactive` or `headless`.
        project_root: Project whose extension directory may be considered.
        project_trust_granted: Explicit one-run project trust decision.
        settings: Pre-resolved settings; loaded automatically when omitted.
        trust_store_path: Alternate persisted trust store for tests.

    Returns:
        Registry plus successful extensions and isolated error messages.
    """
    prepared = await asyncio.to_thread(
        _prepare_extension_load,
        cwd=cwd,
        project_root=project_root,
        project_trust_granted=project_trust_granted,
        settings=settings,
        trust_store_path=trust_store_path,
    )
    if prepared is None:
        return ExtensionLoadResult()

    files, session_cwd = prepared
    registry = ExtensionRegistry()
    loaded: list[LoadedExtension] = []
    errors: list[str] = []
    for file in files:
        try:
            loaded.append(
                await load_extension(file, registry, cwd=session_cwd, mode=mode)
            )
        except ExtensionError as exc:
            logger.warning("Skipping extension %s", file.path, exc_info=True)
            errors.append(str(exc))
        except Exception as exc:
            logger.exception("Unexpected failure loading extension %s", file.path)
            errors.append(f"{file.path}: {type(exc).__name__}: {exc}")

    return ExtensionLoadResult(
        registry=registry,
        loaded=tuple(loaded),
        errors=tuple(errors),
    )


def bind_server_extensions(
    result: ExtensionLoadResult,
) -> Token[ExtensionLoadResult | None]:
    """Expose a server-loop registry to threaded graph construction.

    Args:
        result: Extensions initialized on the persistent server loop.

    Returns:
        Context token used to restore the caller after graph construction.
    """
    global _server_shutdown_registry  # noqa: PLW0603
    _server_shutdown_registry = (
        result.registry if result.registry.shutdown_hooks else None
    )
    return _server_result.set(result)


def get_server_extensions() -> ExtensionLoadResult:
    """Return the bound server extensions or an empty result."""
    return _server_result.get() or ExtensionLoadResult()


def reset_server_extensions(token: Token[ExtensionLoadResult | None]) -> None:
    """Restore server extension context after graph construction.

    Args:
        token: Token returned by `bind_server_extensions`.
    """
    _server_result.reset(token)


async def shutdown_extensions(registry: ExtensionRegistry) -> None:
    """Run every teardown hook while isolating individual failures.

    Args:
        registry: Registry whose session resources are ending.
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
    """Release server-owned extensions on their persistent event loop."""
    global _server_shutdown_registry  # noqa: PLW0603
    registry = _server_shutdown_registry
    _server_shutdown_registry = None
    if registry is not None:
        await shutdown_extensions(registry)
