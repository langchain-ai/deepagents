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
    from deepagents_code.extensions.models import LoadedExtension

logger = logging.getLogger(__name__)

INTERACTIVE_MODE = "interactive"
HEADLESS_MODE = "headless"


@dataclass(frozen=True, slots=True)
class ExtensionLoadResult:
    """Outcome of one pass over the discovery/loading pipeline."""

    registry: ExtensionRegistry = field(default_factory=ExtensionRegistry)
    """Registry holding everything the loaded extensions registered."""

    loaded: tuple[LoadedExtension, ...] = ()
    """Extensions that initialized successfully, in load order."""

    errors: tuple[str, ...] = ()
    """One message per extension that failed to load."""


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
    resolved_settings = load_extension_settings() if settings is None else settings
    if not resolved_settings.enabled:
        return ExtensionLoadResult()

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
        return ExtensionLoadResult()

    registry = ExtensionRegistry()
    session_cwd = Path.cwd() if cwd is None else Path(cwd)
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

    Extension factories may be `async def`, so loading is inherently async while
    the agent construction path (`create_cli_agent`) is synchronous. When a loop
    is already running in this thread, the work is handed to a short-lived worker
    thread rather than touching the caller's loop.

    Args:
        cwd: Working directory of the session.
        mode: Runtime mode: `interactive` or `headless`.
        project_root: Project root whose extensions may load once trust resolves.
        project_trust_granted: Whether project trust was granted for this run.
        settings: Pre-resolved settings; read from config when omitted.

    Returns:
        The same result `load_extensions` produces.
    """

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
