"""Local runtime host for long-running Deep Agents."""

from deepagents_talon._version import __version__
from deepagents_talon.config import TalonConfig
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    AgentRuntime,
    ChannelAdapter,
    ChannelMedia,
    ChannelMessage,
    ChannelStatus,
    CronScheduler,
)
from deepagents_talon.runtime import EchoAgentRuntime

__all__ = [
    "AgentRequest",
    "AgentResult",
    "AgentRuntime",
    "ChannelAdapter",
    "ChannelMedia",
    "ChannelMessage",
    "ChannelStatus",
    "CronScheduler",
    "EchoAgentRuntime",
    "TalonConfig",
    "TalonHost",
    "__version__",
]
