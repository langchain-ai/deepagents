"""Memory Agent Example.

Demonstrates a deep agent that improves over time through learned memory:
- Global memory learned across all users (persistent via StoreBackend)
- Per-user memory personalized to each user (persistent via StoreBackend)
- Live memory editing — agent reads/writes /memories/ during conversations
- Sleep-time cron job for background memory consolidation
- Eval set measuring improvement over N interactions
"""

from memory_agent.prompts import (
    AGENT_INSTRUCTIONS,
    CRON_CONSOLIDATION_PROMPT,
)

__all__ = [
    "AGENT_INSTRUCTIONS",
    "CRON_CONSOLIDATION_PROMPT",
]
