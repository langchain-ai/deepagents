"""Deep Agents package."""

from deepagents._version import __version__
from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.permissions import FilesystemPermission
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

__all__ = [
    "AsyncSubAgent",
    "AsyncSubAgentMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "FilesystemPermission",
    "MemoryMiddleware",
    "PatchToolCallsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "__version__",
    "create_deep_agent",
]
