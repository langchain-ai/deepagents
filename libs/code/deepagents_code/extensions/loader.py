"""Import and initialize Python extension factories."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING

from deepagents_code.extensions.api import ExtensionAPI
from deepagents_code.extensions.models import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from deepagents_code.extensions.models import SourceInfo
    from deepagents_code.extensions.registry import ExtensionRegistry


def _module_name(path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return f"deepagents_code_extension_{digest}"


def _import_factory(source: SourceInfo) -> tuple[str, Callable[[ExtensionAPI], Any]]:
    name = _module_name(source.path)
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
    except (KeyboardInterrupt, SystemExit):
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        sys.modules.pop(name, None)
        msg = f"Failed to import {source.path}: {exc}"
        raise ExtensionError(msg) from exc
    factory = getattr(module, "extension", None)
    if not callable(factory):
        sys.modules.pop(name, None)
        msg = f"{source.path} does not define a callable 'extension' factory"
        raise ExtensionError(msg)
    return name, factory


async def load_extension(
    source: SourceInfo,
    registry: ExtensionRegistry,
    *,
    cwd: Path,
    mode: str,
) -> None:
    """Load one extension transactionally.

    Args:
        source: Extension entry file and import shape.
        registry: Destination for registrations.
        cwd: Session working directory.
        mode: Runtime mode.

    Raises:
        ExtensionError: If import or initialization fails.
        KeyboardInterrupt: If extension code interrupts the process.
        SystemExit: If extension code exits the process.
        asyncio.CancelledError: If initialization is cancelled.
    """
    name, factory = await asyncio.to_thread(_import_factory, source)
    snapshot = registry._snapshot()
    api = ExtensionAPI(registry, source, cwd=cwd, mode=mode)
    try:
        result = await asyncio.to_thread(factory, api)
        if inspect.isawaitable(result):
            await result
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        registry._rollback(snapshot)
        sys.modules.pop(name, None)
        raise
    except Exception as exc:
        registry._rollback(snapshot)
        sys.modules.pop(name, None)
        msg = f"Extension factory in {source.path} failed: {exc}"
        raise ExtensionError(msg) from exc
    finally:
        api._close()
