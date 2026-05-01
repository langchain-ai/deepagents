"""Deep Agents package."""

from deepagents._version import __version__
from deepagents.graph import create_deep_agent
from deepagents.middleware.async_subagents import AsyncSubAgent, AsyncSubAgentMiddleware
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    HarnessProfileConfig,
    ProviderProfile,
    register_harness_profile,
    register_provider_profile,
)

__all__ = [
    "AsyncSubAgent",
    "AsyncSubAgentMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "FilesystemPermission",
    "GeneralPurposeSubagentProfile",
    "HarnessProfile",
    "HarnessProfileConfig",
    "MemoryMiddleware",
    "ProviderProfile",
    "SubAgent",
    "SubAgentMiddleware",
    "__version__",
    "create_deep_agent",
    "register_harness_profile",
    "register_provider_profile",
]
