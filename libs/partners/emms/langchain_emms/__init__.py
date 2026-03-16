"""LangChain Deep Agents partner package for EMMS cognitive memory.

Provides EMmsMemoryMiddleware — biological 6-tier memory for AI agents:
working → short-term → long-term → semantic → procedural → SRS.

Usage::

    from deepagents import create_deep_agent
    from langchain_emms import EMmsMemoryMiddleware

    agent = create_deep_agent(middleware=[EMmsMemoryMiddleware()])
"""

from langchain_emms.middleware import (
    EMMS_SYSTEM_PROMPT,
    EMmsMemoryMiddleware,
    EMmsMemoryState,
    EMmsMemoryStateUpdate,
)

__version__ = "0.1.0"

__all__ = [
    "EMMS_SYSTEM_PROMPT",
    "EMmsMemoryMiddleware",
    "EMmsMemoryState",
    "EMmsMemoryStateUpdate",
]
