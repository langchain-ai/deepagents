"""Import and initialize Python extension factories."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING

from deepagents_code.extensions.api import ExtensionAPI, ExtensionMode
from deepagents_code.extensions.registry import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from deepagents_code.extensions.registry import ExtensionRegistry, SourceInfo


def _extension_module_name(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"deepagents_code_extension_{digest}"


def _import_factory(
    source: SourceInfo,
) -> tuple[str, Callable[[ExtensionAPI], Awaitable[None]]]:
    name = _extension_module_name(source.path)
    spec = importlib.util.spec_from_file_location(
        name,
        source.path,
        submodule_search_locations=[str(source.path.parent)]
        if source.is_package
        else None,
    )
    if spec is None or spec.loader is None:
        msg = f"Could not import extension {source.path}"
        raise ExtensionError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (KeyboardInterrupt, SystemExit, Exception) as exc:
        sys.modules.pop(name, None)
        if isinstance(exc, KeyboardInterrupt):
            raise
        msg = (
            f"Extension import in {source.path} attempted to exit: {exc}"
            if isinstance(exc, SystemExit)
            else f"Failed to import {source.path}: {exc}"
        )
        raise ExtensionError(msg) from exc
    factory = getattr(module, "extension", None)
    if not callable(factory):
        sys.modules.pop(name, None)
        msg = f"{source.path} does not define a callable 'extension' factory"
        raise ExtensionError(msg)
    if not inspect.iscoroutinefunction(factory):
        sys.modules.pop(name, None)
        msg = f"Extension factory in {source.path} must be declared with 'async def'"
        raise ExtensionError(msg)
    return name, factory


async def load_extension(
    source: SourceInfo,
    registry: ExtensionRegistry,
    *,
    cwd: Path,
    mode: ExtensionMode,
) -> ExtensionAPI:
    """Load one extension transactionally.

    Args:
        source: Extension entry file and import shape.
        registry: Destination for registrations.
        cwd: Session working directory.
        mode: Runtime mode.

    Returns:
        The active registrar owned by the extension runtime.

    Raises:
        ExtensionError: If import or initialization fails.
        KeyboardInterrupt: If extension code interrupts the process.
        asyncio.CancelledError: If initialization is cancelled.
    """
    name, factory = await asyncio.to_thread(_import_factory, source)
    snapshot = registry._snapshot()
    api = ExtensionAPI(registry, source, cwd=cwd, mode=mode)
    try:
        await factory(api)
    except (KeyboardInterrupt, asyncio.CancelledError):
        registry._rollback(snapshot)
        api._deactivate()
        sys.modules.pop(name, None)
        raise
    except (SystemExit, Exception) as exc:
        registry._rollback(snapshot)
        api._deactivate()
        sys.modules.pop(name, None)
        msg = (
            f"Extension factory in {source.path} attempted to exit: {exc}"
            if isinstance(exc, SystemExit)
            else f"Extension factory in {source.path} failed: {exc}"
        )
        raise ExtensionError(msg) from exc
    return api
