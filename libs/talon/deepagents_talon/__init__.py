"""Local runtime host for long-running Deep Agents.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from typing import TYPE_CHECKING

from deepagents_talon._version import __version__
from deepagents_talon.config import TalonConfig
from deepagents_talon.cron import (
    CronJob,
    CronJobError,
    CronJobStore,
    CronOrigin,
    CronSchedule,
    CronTools,
    PersistentCronScheduler,
)
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    AgentRuntime,
    ChannelAdapter,
    ChannelMedia,
    ChannelMessage,
    ChannelReaction,
    ChannelStatus,
    CronScheduler,
    ReactionChannelAdapter,
    ReactionHandler,
    SendResult,
    ToolApprovalDecision,
    ToolApprovalHandler,
    ToolApprovalRequest,
)
from deepagents_talon.speech import (
    DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL,
    LocalParakeetVoiceTranscriber,
    OpenAIVoiceTranscriber,
    VoiceTranscriber,
)

if TYPE_CHECKING:
    from deepagents_talon.runtime import DeepAgentRuntime, EchoAgentRuntime


def __getattr__(name: str) -> object:
    """Load runtime classes only when accessed."""
    if name in {"DeepAgentRuntime", "EchoAgentRuntime"}:
        from deepagents_talon import runtime  # noqa: PLC0415

        return getattr(runtime, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DEFAULT_LOCAL_VOICE_TRANSCRIPTION_MODEL",
    "AgentRequest",
    "AgentResult",
    "AgentRuntime",
    "ChannelAdapter",
    "ChannelMedia",
    "ChannelMessage",
    "ChannelReaction",
    "ChannelStatus",
    "CronJob",
    "CronJobError",
    "CronJobStore",
    "CronOrigin",
    "CronSchedule",
    "CronScheduler",
    "CronTools",
    "DeepAgentRuntime",
    "EchoAgentRuntime",
    "LocalParakeetVoiceTranscriber",
    "OpenAIVoiceTranscriber",
    "PersistentCronScheduler",
    "ReactionChannelAdapter",
    "ReactionHandler",
    "SendResult",
    "TalonConfig",
    "TalonHost",
    "ToolApprovalDecision",
    "ToolApprovalHandler",
    "ToolApprovalRequest",
    "VoiceTranscriber",
    "__version__",
]
