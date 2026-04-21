"""Deep Agents package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents._version import __version__

if TYPE_CHECKING:
    from deepagents.graph import create_deep_agent
    from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.permissions import FilesystemPermission
    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

__all__ = [
    "AsyncSubAgent",
    "AsyncSubAgentMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "FilesystemPermission",
    "MemoryMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "__version__",
    "create_deep_agent",
]

_LAZY: dict[str, str] = {
    "create_deep_agent": "deepagents.graph",
    "AsyncSubAgent": "deepagents.middleware.async_subagents",
    "AsyncSubAgentMiddleware": "deepagents.middleware.async_subagents",
    "FilesystemMiddleware": "deepagents.middleware.filesystem",
    "MemoryMiddleware": "deepagents.middleware.memory",
    "FilesystemPermission": "deepagents.middleware.permissions",
    "CompiledSubAgent": "deepagents.middleware.subagents",
    "SubAgent": "deepagents.middleware.subagents",
    "SubAgentMiddleware": "deepagents.middleware.subagents",
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
