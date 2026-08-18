"""File-in, extension-out loading of a single extension.

`load_extension` is the only loading mechanism in the system. It imports one
Python file with stdlib `importlib`, requires an `extension` factory callable,
awaits it when it is async, and returns what the factory registered. It takes a
path, never a directory or a manifest, so every discovery recipe funnels through
the same code path.

Extensions execute local Python code. Ungated sources (the user's own
directories) are trusted by definition; project-scoped sources are resolved by
`trust` before their paths ever reach this module.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
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

logger = logging.getLogger(__name__)

FACTORY_NAME = "extension"
"""Name of the factory callable every extension module must expose."""

_MODULE_PREFIX = "deepagents_code_extension_"
"""Prefix for synthesized module names, keeping extensions out of the app namespace."""


def _module_name(path: Path, origin: UnitOrigin) -> str:
    """Build a unique import name for an extension file.

    A directory extension is imported under its directory name so its sibling
    modules resolve as a package; a single file gets its stem. Both are prefixed
    and disambiguated by path hash so two extensions with the same name in
    different sources cannot shadow each other.

    Args:
        path: Entry file being imported.
        origin: Whether the entry file belongs to a package directory.

    Returns:
        A module name unique to this path.
    """
    stem = path.parent.name if origin is UnitOrigin.PACKAGE else path.stem
    sanitized = "".join(char if char.isalnum() else "_" for char in stem)
    digest = abs(hash(str(path))) % 1_000_000
    return f"{_MODULE_PREFIX}{sanitized}_{digest}"


def _import_factory(
    file: ExtensionFile,
) -> tuple[str, Callable[[ExtensionAPI], Any]]:
    """Import an extension module and return its factory.

    Args:
        file: Discovered extension entry file.

    Returns:
        Synthesized module name and its callable factory.

    Raises:
        ExtensionError: If the module cannot be imported or has no factory.
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
    """Import one extension file and run its factory.

    Args:
        file: Discovered entry file and the provenance its units will carry.
        registry: Registry receiving the factory's registrations.
        cwd: Working directory of the session.
        mode: Runtime mode: `interactive` or `headless`.

    Returns:
        A record of what the extension registered.

    Raises:
        asyncio.CancelledError: If extension initialization is cancelled.
        ExtensionError: If the file cannot be imported, exposes no `extension`
            factory, or the factory raises.
        KeyboardInterrupt: If the extension factory interrupts the process.
        SystemExit: If the extension factory exits the process.
    """
    name, factory = await asyncio.to_thread(_import_factory, file)

    before = registry._snapshot()
    api = ExtensionAPI(registry, file.source, cwd=cwd, mode=mode)
    try:
        # Synchronous factories stay off the server event loop. An async factory
        # returns its coroutine here without running it; awaiting below binds
        # its resources to the caller's loop.
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

    return LoadedExtension(
        path=file.path,
        source=file.source,
        middleware=tuple(unit.unit for unit in registry.middleware[before[0] :]),
        tools=tuple(unit.unit for unit in registry.tools[before[1] :]),
        commands=tuple(unit.name for unit in registry.commands[before[2] :]),
        backend_routes=tuple(
            unit.name for unit in registry.backend_routes[before[3] :]
        ),
    )
