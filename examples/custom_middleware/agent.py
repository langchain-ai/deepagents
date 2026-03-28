"""Custom Middleware — Cookbook Example.

Demonstrates how to write custom middleware for Deep Agents that:
1. Injects dynamic system prompt context on every LLM call
2. Logs all tool calls and their results
3. Modifies agent state before the agent loop starts

Shows both `wrap_model_call` (intercepts LLM requests) and
`before_agent` (modifies state at startup) hooks.
"""

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ContextT, ModelRequest, ModelResponse, ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime


class ToolCallLoggerMiddleware(AgentMiddleware):
    """Middleware that logs all tool calls made by the agent.

    This middleware intercepts every LLM response and logs any tool calls
    it contains, along with timing information. Useful for debugging,
    auditing, and understanding agent behavior.

    It also injects a "session context" system prompt addition so the
    agent knows what tools have been called so far in the session.
    """

    def __init__(self) -> None:
        self._tool_log: list[dict[str, Any]] = []
        self._start_time: float = time.time()

    @property
    def tool_log(self) -> list[dict[str, Any]]:
        """Get the accumulated tool call log."""
        return list(self._tool_log)

    def _log_tool_calls(self, response: ModelResponse[ResponseT]) -> None:
        """Extract and log tool calls from an LLM response."""
        message = response.message
        if isinstance(message, AIMessage) and message.tool_calls:
            for tc in message.tool_calls:
                entry = {
                    "timestamp": time.time() - self._start_time,
                    "tool": tc["name"],
                    "args": tc["args"],
                }
                self._tool_log.append(entry)
                print(f"[ToolLogger] {entry['timestamp']:.1f}s — {tc['name']}({json.dumps(tc['args'], default=str)[:100]})")

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Intercept the LLM call to log tool calls from the response."""
        response = handler(request)
        self._log_tool_calls(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async version of wrap_model_call."""
        response = await handler(request)
        self._log_tool_calls(response)
        return response


class SessionContextMiddleware(AgentMiddleware):
    """Middleware that injects dynamic session context into the system prompt.

    On every LLM call, this middleware appends session metadata (like
    the current time and a custom session ID) to the system message.
    This is useful for giving the agent awareness of its operating context.
    """

    def __init__(self, session_id: str, metadata: dict[str, str] | None = None) -> None:
        self._session_id = session_id
        self._metadata = metadata or {}

    def _build_context(self) -> str:
        """Build the session context string."""
        lines = [
            "\n\n## Session Context",
            f"- Session ID: {self._session_id}",
            f"- Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        for key, value in self._metadata.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject session context into the system message before every LLM call."""
        context = self._build_context()
        new_system = (request.system_message or "") + context
        return handler(request.override(system_message=new_system))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async version."""
        context = self._build_context()
        new_system = (request.system_message or "") + context
        return await handler(request.override(system_message=new_system))


class GuardrailMiddleware(AgentMiddleware):
    """Middleware that modifies agent state before the agent loop starts.

    Demonstrates the `before_agent` hook: prepend a safety reminder
    to the conversation if no prior messages exist.
    """

    def __init__(self, reminder: str = "Remember: always verify information before presenting it as fact.") -> None:
        self._reminder = reminder

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Inject a reminder into messages at the start of the conversation."""
        messages = state.get("messages", [])

        # Only inject on the first turn (when there are few messages)
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if len(tool_messages) == 0:
            # Add reminder as context for the agent
            print(f"[Guardrail] Injecting reminder: {self._reminder[:60]}...")
        return None


# --- Agent setup ---


def create_custom_middleware_agent():
    """Create a deep agent with custom middleware demonstrating all hook types."""
    # Initialize middleware instances
    tool_logger = ToolCallLoggerMiddleware()
    session_context = SessionContextMiddleware(
        session_id="demo-001",
        metadata={
            "environment": "development",
            "user_role": "developer",
        },
    )
    guardrail = GuardrailMiddleware()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        system_prompt="You are a helpful coding assistant.",
        middleware=[tool_logger, session_context, guardrail],
    )

    return agent, tool_logger


def main():
    """Run the agent with custom middleware and display the tool log."""
    agent, tool_logger = create_custom_middleware_agent()

    print("=" * 60)
    print("Custom Middleware Demo")
    print("=" * 60)
    print()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Create a simple Python function that checks if a "
                        "string is a palindrome, and write it to a file "
                        "called palindrome.py"
                    ),
                }
            ]
        }
    )

    # Display final response
    final_message = result["messages"][-1]
    print("\n" + "=" * 60)
    print("Final Response:")
    print("=" * 60)
    print(final_message.content if hasattr(final_message, "content") else str(final_message))

    # Display tool call log
    print("\n" + "=" * 60)
    print(f"Tool Call Log ({len(tool_logger.tool_log)} calls):")
    print("=" * 60)
    for entry in tool_logger.tool_log:
        print(f"  [{entry['timestamp']:.1f}s] {entry['tool']}")


if __name__ == "__main__":
    main()
