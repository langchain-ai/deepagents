"""Integration test: SanitizerMiddleware with a mock provider in a real agent."""

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware

from deepagents.middleware.sanitizer import (
    SanitizeFinding,
    SanitizeResult,
    SanitizerMiddleware,
)


class SecretRedactor:
    @property
    def name(self) -> str:
        return "test"

    def sanitize(self, content: str) -> SanitizeResult:
        if "SUPER_SECRET_TOKEN" in content:
            return SanitizeResult(
                content=content.replace("SUPER_SECRET_TOKEN", "<REDACTED:test-token>"),
                findings=[SanitizeFinding(rule_id="test-token", redacted_as="<REDACTED:test-token>")],
            )
        return SanitizeResult(content=content, findings=[])

    async def asanitize(self, content: str) -> SanitizeResult:
        return self.sanitize(content)


def test_sanitizer_middleware_registers_with_agent():
    """Verify middleware can be added to an agent without errors."""
    middleware: list[AgentMiddleware] = [
        SanitizerMiddleware(providers=[SecretRedactor()]),
    ]
    agent = create_agent(
        model="claude-sonnet-4-20250514",
        middleware=middleware,
        tools=[],
    )
    assert agent is not None
