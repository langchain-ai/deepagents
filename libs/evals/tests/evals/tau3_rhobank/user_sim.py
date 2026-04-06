"""LLM-powered user simulator for multi-turn Rho-Bank customer service evaluation.

Extends the tau2 user simulator with tool-calling support. The user sim
can call tools (e.g. `call_discoverable_user_tool`, `request_human_agent_transfer`)
when the task scenario requires user-initiated actions.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import StructuredTool

STOP_TOKENS = frozenset({"###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"})

SIMULATION_GUIDELINES = """\
You are playing the role of a customer contacting a customer service representative.
Your goal is to simulate realistic customer interactions while following specific scenario instructions.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. \
Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language \
to convey the same information.
- Disclose information progressively. Wait for the agent to ask for specific information \
before providing it.

## Tool Usage
- When the scenario instructs you to perform an action (like submitting a dispute, \
providing card digits, or requesting a transfer), use the appropriate tool.
- When the agent gives you a tool to use, call `call_discoverable_user_tool` with \
the tool name and arguments the agent provided.
- When you want to request being transferred to a human agent, use `request_human_agent_transfer`.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisfied, generate the '###STOP###' token to end the conversation.
- If you are transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer.
- If you find yourself in a situation in which the scenario does not provide enough information \
for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to \
the provided instructions and maintaining character consistency.\
"""


@dataclass
class UserResponse:
    """Structured response from the user simulator.

    Attributes:
        text: The text content of the response (empty if only tool calls).
        tool_calls: List of tool call dicts with `id`, `name`, `args`.
    """

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _build_system_prompt(scenario: dict[str, Any]) -> str:
    """Build the user simulator's system prompt from a tau3 task scenario.

    Args:
        scenario: The `user_scenario` dict from a tau3 task.

    Returns:
        The full system prompt string.
    """
    instructions = scenario.get("instructions", {})
    parts = [SIMULATION_GUIDELINES, "\n\n<scenario>"]

    if isinstance(instructions, dict):
        if task_inst := instructions.get("task_instructions"):
            parts.append(f"\n## Task Instructions\n{task_inst}")
        if reason := instructions.get("reason_for_call"):
            parts.append(f"\n## Reason for Call\n{reason}")
        if known := instructions.get("known_info"):
            parts.append(f"\n## Known Information\n{known}")
        if domain := instructions.get("domain"):
            parts.append(f"\n## Domain\n{domain}")
    else:
        parts.append(f"\n{json.dumps(instructions)}")

    parts.append("\n</scenario>")
    return "".join(parts)


class UserSimulator:
    """Simulated customer driven by an LLM, with optional tool calling.

    Args:
        model: The chat model to use for generation.
        scenario: The `user_scenario` dict from a tau3 task.
        user_tools: Optional list of LangChain tools to bind to the model.
    """

    def __init__(
        self,
        model: BaseChatModel,
        scenario: dict[str, Any],
        user_tools: list[StructuredTool] | None = None,
    ) -> None:
        if user_tools:
            self._model = model.bind_tools(user_tools)
        else:
            self._model = model
        self._messages: list[BaseMessage] = [
            SystemMessage(content=_build_system_prompt(scenario)),
        ]
        self._done = False

    @property
    def is_done(self) -> bool:
        """Whether the simulator has emitted a stop token."""
        return self._done

    def get_opening_message(self) -> str:
        """Generate the customer's first message in the conversation.

        Returns:
            The customer's opening message.
        """
        greeting = "Hello! Welcome to Rho-Bank customer service. How may I assist you today?"
        return self.respond(greeting).text

    def respond(self, agent_message: str) -> UserResponse:
        """Generate the customer's response to an agent message.

        May return tool calls in addition to (or instead of) text.

        Args:
            agent_message: The agent's latest text.

        Returns:
            A UserResponse with text and/or tool calls.
        """
        self._messages.append(HumanMessage(content=agent_message))
        response: AIMessage = self._model.invoke(self._messages)
        self._messages.append(response)
        return self._parse_response(response)

    def _parse_response(self, response: AIMessage) -> UserResponse:
        """Parse a model response into normalized text and tool calls."""
        text = response.content if isinstance(response.content, str) else str(response.content)
        tool_calls: list[dict[str, Any]] = []

        if response.tool_calls:
            tool_calls.extend(
                {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in response.tool_calls
            )

        for token in STOP_TOKENS:
            if token in text:
                self._done = True
                text = text.replace(token, "").strip()

        return UserResponse(text=text, tool_calls=tool_calls)

    def receive_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """Feed a tool execution result back to the user sim.

        Args:
            tool_call_id: The ID of the tool call.
            tool_name: The name of the tool that was called.
            result: The string result from the tool execution.
        """
        self._messages.append(
            ToolMessage(content=result, tool_call_id=tool_call_id, name=tool_name)
        )

    def get_response_after_tools(self) -> UserResponse:
        """Get the user's next response after tool results have been fed back.

        Returns:
            A UserResponse with text and/or additional tool calls.
        """
        response: AIMessage = self._model.invoke(self._messages)
        self._messages.append(response)
        return self._parse_response(response)
