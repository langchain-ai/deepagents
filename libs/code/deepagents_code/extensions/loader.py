"""Import and initialize one discovered Python extension file."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING

from deepagents_code.extensions.api import ExtensionAPI
from deepagents_code.extensions.models import (
    ExtensionError,
    LoadedExtension,
    UnitOrigin,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from deepagents_code.extensions.models import ExtensionFile
    from deepagents_code.extensions.registry import ExtensionRegistry

FACTORY_NAME = "extension"
_MODULE_PREFIX = "deepagents_code_extension_"


def _module_name(path: Path, origin: UnitOrigin) -> str:
    """Build a deterministic import name unique to an extension path.

    Args:
        path: Extension entry file.
        origin: Whether the file is a package entry point.

    Returns:
        A private module name that cannot collide with the app namespace.
    """
    stem = path.parent.name if origin is UnitOrigin.PACKAGE else path.stem
    sanitized = "".join(
        char if char.isascii() and char.isalnum() else "_" for char in stem
    )
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    return f"{_MODULE_PREFIX}{sanitized}_{digest}"


def _import_factory(
    file: ExtensionFile,
) -> tuple[str, Callable[[ExtensionAPI], Any]]:
    """Import an extension module and locate its factory.

    Args:
        file: Discovered extension file and provenance.

    Returns:
        Synthesized module name and callable `extension` factory.

    Raises:
        ExtensionError: If import fails or the module has no callable factory.
        KeyboardInterrupt: If module import interrupts the process.
        SystemExit: If module import exits the process.
    """
    is_package = file.source.origin is UnitOrigin.PACKAGE
    name = _module_name(file.path, file.source.origin)
    spec = importlib.util.spec_from_file_location(
        name,
        file.path,
        submodule_search_locations=[str(file.path.parent)] if is_package else None,
    )
    if spec is None or spec.loader is None:
        msg = f"Could not build an import spec for {file.path}"
        raise ExtensionError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except (KeyboardInterrupt, SystemExit):
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        sys.modules.pop(name, None)
        msg = f"Failed to import {file.path}: {exc}"
        raise ExtensionError(msg) from exc

    factory = getattr(module, FACTORY_NAME, None)
    if not callable(factory):
        sys.modules.pop(name, None)
        msg = f"{file.path} does not define a callable {FACTORY_NAME!r} factory"
        raise ExtensionError(msg)
    return name, factory


async def load_extension(
    file: ExtensionFile,
    registry: ExtensionRegistry,
    *,
    cwd: Path,
    mode: str,
) -> LoadedExtension:
    """Import and initialize one extension transactionally.

    Synchronous import and factory work runs in a worker thread. Async factory
    results are awaited on the caller's event loop so loop-bound resources can
    remain valid for the full server lifetime.

    Args:
        file: Discovered extension file and provenance.
        registry: Registry receiving factory registrations.
        cwd: Working directory for the session.
        mode: Runtime mode, either `interactive` or `headless`.

    Returns:
        Registrations produced by this extension.

    Raises:
        asyncio.CancelledError: If initialization is cancelled.
        ExtensionError: If import or factory initialization fails.
        KeyboardInterrupt: If extension code interrupts the process.
        SystemExit: If extension code exits the process.
    """
    name, factory = await asyncio.to_thread(_import_factory, file)
    before = registry._snapshot()
    api = ExtensionAPI(registry, file.source, cwd=cwd, mode=mode)
    try:
        result = await asyncio.to_thread(factory, api)
        if inspect.isawaitable(result):
            await result
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        registry._rollback(before)
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        registry._rollback(before)
        sys.modules.pop(name, None)
        msg = f"Extension factory in {file.path} failed: {exc}"
        raise ExtensionError(msg) from exc
    finally:
        api._close()

    middleware, tools, backend_routes, _shutdown_hooks = before
    return LoadedExtension(
        path=file.path,
        source=file.source,
        middleware=tuple(unit.unit for unit in registry.middleware[middleware:]),
        tools=tuple(unit.unit for unit in registry.tools[tools:]),
        backend_routes=tuple(
            unit.name for unit in registry.backend_routes[backend_routes:]
        ),
    )
