"""Orchestration and lifecycle for the extension system.

`load_extensions` walks the pipeline once: resolve settings, resolve project
trust, resolve sources into files, then load each file. Failures are isolated per
extension — a malformed extension is reported and skipped so it can never take
down the agent loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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

_server_extension_result: ContextVar[ExtensionLoadResult | None]
_server_shutdown_registry: ExtensionRegistry | None = None


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Outcome of one pass over the discovery/loading pipeline."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    """Registry holding everything the loaded extensions registered."""

    loaded: tuple[LoadedExtension, ...] = ()
    """Extensions that initialized successfully, in load order."""

    errors: tuple[str, ...] = ()
    """One message per extension that failed to load."""


_server_extension_result = ContextVar("server_extension_result", default=None)


def bind_server_extensions(
    result: ExtensionLoadResult,
) -> Token[ExtensionLoadResult | None]:
    """Expose a server-loop result to synchronous graph construction.

    `asyncio.to_thread` copies the current context, so `create_cli_agent` can
    consume this exact registry without reloading extensions on a temporary
    event loop. Registries with teardown hooks are also retained for the HTTP
    server lifespan.

    Args:
        result: Extensions initialized on the server event loop.

    Returns:
        Context token used to restore the caller after graph construction.
    """
    global _server_shutdown_registry  # noqa: PLW0603
    if result.registry.shutdown_hooks:
        _server_shutdown_registry = result.registry
    return _server_extension_result.set(result)


def reset_server_extensions(token: Token[ExtensionLoadResult | None]) -> None:
    """Restore the extension context after graph construction.

    Args:
        token: Token returned by `bind_server_extensions`.
    """
    _server_extension_result.reset(token)


def project_extensions_trusted(
    project_root: Path | str,
    *,
    policy: TrustPolicy,
    granted: bool = False,
    store_path: Path | None = None,
) -> bool:
    """Resolve whether a project's extensions may load.

    Args:
        project_root: Project root under consideration.
        policy: Default policy applied when no decision is recorded.
        granted: Whether this run was granted trust explicitly (an interactive
            prompt answered at startup, or a CLI flag).
        store_path: Alternate trust store path for tests.

    Returns:
        `True` when the project source may be scanned. Under `ask` with no
            persisted decision and no explicit grant the answer is `False`, so
            non-interactive runs stay closed by default.
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
    """Resolve settings, trust, files, and cwd off the caller's event loop.

    Returns:
        Discovered files and session cwd, or `None` when extensions are disabled
        or no files are found.
    """
    resolved_settings = load_extension_settings() if settings is None else settings
    if not resolved_settings.enabled:
        return None

    project_dir: Path | None = None
    if project_root is not None and project_extensions_trusted(
        project_root,
        policy=resolved_settings.trust,
        granted=project_trust_granted,
        store_path=trust_store_path,
    ):
        project_dir = project_extensions_dir(project_root)

    files = discover_extension_files(
        extra_paths=resolved_settings.paths,
        project_dir=project_dir,
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
    """Discover, trust-gate, and load every configured extension.

    Args:
        cwd: Working directory of the session; defaults to the process cwd.
        mode: Runtime mode: `interactive` or `headless`.
        project_root: Project root whose `.deepagents/extensions/` may load once
            trust resolves. `None` skips the project source entirely.
        project_trust_granted: Whether project trust was granted for this run.
        settings: Pre-resolved settings; read from config when omitted.
        trust_store_path: Alternate trust store path for tests.

    Returns:
        The populated registry alongside per-extension successes and failures.
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
        except (
            Exception
        ) as exc:  # error isolation: one bad extension must not kill the agent
            logger.exception("Unexpected failure loading extension %s", file.path)
            errors.append(f"{file.path}: {type(exc).__name__}: {exc}")

    return ExtensionLoadResult(
        registry=registry,
        loaded=tuple(loaded),
        errors=tuple(errors),
    )


def load_extensions_blocking(
    *,
    cwd: Path | None = None,
    mode: str = INTERACTIVE_MODE,
    project_root: Path | None = None,
    project_trust_granted: bool = False,
    settings: ExtensionSettings | None = None,
) -> ExtensionLoadResult:
    """Run `load_extensions` from synchronous code.

    A result initialized by the server loop is reused through its copied thread
    context. Other synchronous callers run loading on a private temporary loop;
    when their thread already owns a loop, loading moves to a worker thread.

    Args:
        cwd: Working directory of the session.
        mode: Runtime mode: `interactive` or `headless`.
        project_root: Project root whose extensions may load once trust resolves.
        project_trust_granted: Whether project trust was granted for this run.
        settings: Pre-resolved settings; read from config when omitted.

    Returns:
        The same result `load_extensions` produces.
    """
    server_result = _server_extension_result.get()
    if server_result is not None:
        return server_result

    def run() -> ExtensionLoadResult:
        return asyncio.run(
            load_extensions(
                cwd=cwd,
                mode=mode,
                project_root=project_root,
                project_trust_granted=project_trust_granted,
                settings=settings,
            )
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(run).result()


async def shutdown_extensions(registry: ExtensionRegistry) -> None:
    """Run every registered shutdown hook, isolating failures.

    Args:
        registry: Registry whose extensions are being torn down.
    """
    import inspect  # deferred: only needed during teardown

    for hook in registry.shutdown_hooks:
        try:
            result = hook.unit()
            if inspect.isawaitable(result):
                await result
        except Exception:  # one failing teardown must not block the rest
            logger.warning(
                "Shutdown hook from %s failed", hook.source.label, exc_info=True
            )


async def shutdown_server_extensions() -> None:
    """Tear down the server-owned extension registry on its event loop."""
    global _server_shutdown_registry  # noqa: PLW0603
    registry = _server_shutdown_registry
    _server_shutdown_registry = None
    if registry is not None:
        await shutdown_extensions(registry)
