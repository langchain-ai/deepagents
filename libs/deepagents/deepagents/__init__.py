"""Deep Agents package."""

from deepagents._version import __version__
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "MemoryMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "__version__",
    "create_deep_agent",
]
