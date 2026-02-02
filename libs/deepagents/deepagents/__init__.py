"""DeepAgents package."""

from deepagents.graph import create_deep_agent
from deepagents.version import VERSION

__version__ = VERSION
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "MemoryMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "create_deep_agent",
]
