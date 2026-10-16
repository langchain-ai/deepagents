"""Load tools from explicitly configured, trusted host Python files."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import CodeType, ModuleType


class LocalToolError(ValueError):
    """Invalid local tool configuration or import."""


@contextmanager
def load_local_tools(directories: Sequence[Path]) -> Iterator[tuple[BaseTool, ...]]:
    """Import public tool instances from trusted directories.

    Args:
        directories: Operator-selected host directories; importing executes Python.

    Yields:
        Tools discovered in directory order and sorted filename order.

    Raises:
        LocalToolError: A directory, import, or tool name is invalid.
    """
    tools: dict[str, BaseTool] = {}
    sources: dict[str, Path] = {}
    modules: list[ModuleType] = []
    try:
        for directory in dict.fromkeys(path.expanduser().resolve() for path in directories):
            for path in _tool_files(directory):
                module = _import_module(path)
                modules.append(module)
                _collect_tools(module, path, tools, sources)
        yield tuple(tools.values())
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)


def _collect_tools(
    module: ModuleType, path: Path, tools: dict[str, BaseTool], sources: dict[str, Path]
) -> None:
    for name, value in vars(module).items():
        if name.startswith("_") or not isinstance(value, BaseTool):
            continue
        if tools.get(value.name) is value:
            continue
        if value.name in tools:
            msg = f"Duplicate local tool {value.name!r} in {sources[value.name]} and {path}"
            raise LocalToolError(msg)
        tools[value.name], sources[value.name] = value, path


def _tool_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        msg = f"Local tools directory does not exist or is not a directory: {directory}"
        raise LocalToolError(msg)
    paths = sorted(path for path in directory.glob("*.py") if not path.name.startswith("_"))
    for path in paths:
        if path.resolve().parent != directory or not path.is_file():
            msg = f"Local tool file must remain inside its configured directory: {path}"
            raise LocalToolError(msg)
    return paths


class _FreshSourceLoader(SourceFileLoader):
    def get_code(self, fullname: str) -> CodeType:
        path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(path), path)


def _import_module(path: Path) -> ModuleType:
    name = f"_talon_local_tools_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        name, path, loader=_FreshSourceLoader(name, str(path))
    )
    if spec is None or spec.loader is None:
        msg = f"Cannot import local tools from {path}"
        raise LocalToolError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001  # redact arbitrary plugin errors, fail startup
        sys.modules.pop(name, None)
        msg = f"Cannot import local tools from {path} ({type(exc).__name__})"
        raise LocalToolError(msg) from None
    return module
