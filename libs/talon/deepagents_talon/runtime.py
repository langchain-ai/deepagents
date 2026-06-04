"""Minimal agent runtime used to bootstrap the Talon host."""

from __future__ import annotations

from deepagents_talon.interfaces import AgentRequest, AgentResult


class EchoAgentRuntime:
    """Small placeholder runtime for host bootstrapping and tests."""

    async def start(self) -> None:
        """Initialize the placeholder runtime."""

    async def stop(self) -> None:
        """Release placeholder runtime resources."""

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Return the request text as a trivial agent response.

        Args:
            request: Agent request supplied by the Talon host.

        Returns:
            Echo response tagged as placeholder runtime output.
        """
        return AgentResult(text=request.text)
