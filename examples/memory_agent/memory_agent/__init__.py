"""Memory Agent Example.

Demonstrates a deep agent that improves over time through learned memory:
- Global system prompt learned across all users
- Per-user system prompt personalized to each user
- Sleep-time cron job for background memory consolidation
- Eval set measuring improvement over N interactions
"""

from memory_agent.prompts import (
    AGENT_INSTRUCTIONS,
    CRON_CONSOLIDATION_PROMPT,
    GLOBAL_MEMORY_PROMPT,
    USER_MEMORY_PROMPT,
)

__all__ = [
    "AGENT_INSTRUCTIONS",
    "CRON_CONSOLIDATION_PROMPT",
    "GLOBAL_MEMORY_PROMPT",
    "USER_MEMORY_PROMPT",
]
