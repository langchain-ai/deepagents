"""LLM-powered user simulator for multi-turn airline customer service evaluation.

Uses a cheap model to play the customer role based on a tau2 task scenario.
The simulator follows tau2's simulation guidelines: disclose information
progressively, stay in character, and emit stop tokens when the task is done.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau-bench
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

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

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisfied, generate the '###STOP###' token to end the conversation.
- If you are transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer.
- If you find yourself in a situation in which the scenario does not provide enough information \
for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.

Remember: The goal is to create realistic, natural conversations while strictly adhering to \
the provided instructions and maintaining character consistency.\
"""


def _build_system_prompt(scenario: dict[str, Any]) -> str:
    """Build the user simulator's system prompt from a tau2 task scenario.

    Args:
        scenario: The `user_scenario` dict from a tau2 task.

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
    """Simulated customer driven by a cheap LLM.

    Args:
        model: The chat model to use for generation.
        scenario: The `user_scenario` dict from a tau2 task.
    """

    def __init__(self, model: BaseChatModel, scenario: dict[str, Any]) -> None:
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

        Feeds a generic agent greeting so the user sim responds naturally
        with their reason for calling.

        Returns:
            The customer's opening message.
        """
        greeting = "Hello! Welcome to our airline customer service. How may I assist you today?"
        return self.respond(greeting)

    def respond(self, agent_message: str) -> str:
        """Generate the customer's response to an agent message.

        Args:
            agent_message: The agent's latest text.

        Returns:
            The customer's response text (stop tokens stripped).
        """
        self._messages.append(HumanMessage(content=agent_message))
        response = self._model.invoke(self._messages)
        text = response.content if isinstance(response.content, str) else str(response.content)
        self._messages.append(AIMessage(content=text))

        for token in STOP_TOKENS:
            if token in text:
                self._done = True
                text = text.replace(token, "").strip()

        return text
