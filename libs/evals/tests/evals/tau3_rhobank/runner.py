"""Multi-turn conversation orchestrator for tau3 Rho-Bank evaluations.

Extends the tau2 runner with user tool execution support. Drives a
back-and-forth conversation between a deepagents agent and an LLM-powered
user simulator, handling user tool calls in between turns.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.evals.utils import run_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import StructuredTool
    from langgraph.graph.state import CompiledStateGraph

    from tests.evals.tau3_rhobank.domain import ToolCallEntry
    from tests.evals.tau3_rhobank.user_sim import UserResponse, UserSimulator

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 60
MAX_USER_TOOL_ROUNDS_PER_TURN = 12


@dataclass
class Message:
    """A single message in the conversation transcript."""

    role: str
    content: str


@dataclass
class ConversationResult:
    """Output of a multi-turn conversation run.

    Attributes:
        messages: Full conversation transcript (user + assistant messages).
        tool_call_log: Every tool invocation recorded during the conversation.
        terminated_by: How the conversation ended.
        turn_count: Number of agent turns completed.
    """

    messages: list[Message] = field(default_factory=list)
    tool_call_log: list[ToolCallEntry] = field(default_factory=list)
    terminated_by: str = "max_turns"
    turn_count: int = 0


def run_multi_turn(
    agent: CompiledStateGraph[Any, Any],
    user_sim: UserSimulator,
    *,
    model: BaseChatModel,
    tool_call_log: list[ToolCallEntry],
    user_tools: list[StructuredTool],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ConversationResult:
    """Run a multi-turn conversation between the agent and user simulator.

    Supports user tool calls: after the user sim responds, if it made tool
    calls the runner executes them against the shared DB and feeds results
    back to the user sim before collecting the text message.

    Args:
        agent: The compiled deepagents graph.
        user_sim: The LLM-powered user simulator.
        model: The agent's chat model (for logging only).
        tool_call_log: Shared mutable log populated by the banking tools.
        user_tools: LangChain StructuredTools bound to the user sim model.
        max_turns: Maximum number of agent turns before stopping.

    Returns:
        The full conversation result with transcript and tool call log.
    """
    thread_id = str(uuid.uuid4())
    result = ConversationResult(tool_call_log=tool_call_log)
    user_tools_by_name = {t.name: t for t in user_tools}

    def _resolve_user_tool_calls(user_response: UserResponse, label: str) -> str:
        """Execute user-sim tool calls and return the final text message.

        When the user simulator responds with tool calls (and possibly empty
        text), this helper runs those tools against the shared DB, feeds the
        results back, and re-invokes the sim until it produces a text message.
        """
        tool_rounds = 0
        while user_response.tool_calls:
            tool_rounds += 1
            if tool_rounds > MAX_USER_TOOL_ROUNDS_PER_TURN:
                logger.warning(
                    "Exceeded max user tool rounds (%d) at %s", MAX_USER_TOOL_ROUNDS_PER_TURN, label
                )
                break
            for tc in user_response.tool_calls:
                tool = user_tools_by_name.get(tc["name"])
                if tool is not None:
                    try:
                        tool_result = tool.invoke(tc["args"])
                    except Exception as e:  # noqa: BLE001
                        tool_result = f"Error: {e}"
                    user_sim.receive_tool_result(tc["id"], tc["name"], tool_result)
                else:
                    user_sim.receive_tool_result(
                        tc["id"], tc["name"], f"Error: Unknown tool '{tc['name']}'"
                    )
            user_response = user_sim.get_response_after_tools()
        return user_response.text

    opening = user_sim.get_opening_message()
    user_msg = _resolve_user_tool_calls(opening, "opening")
    result.messages.append(Message(role="user", content=user_msg))

    for turn in range(max_turns):
        logger.info("Turn %d: agent processing", turn + 1)
        trajectory = run_agent(
            agent,
            query=user_msg,
            model=model,
            thread_id=thread_id,
        )
        agent_msg = trajectory.answer
        result.messages.append(Message(role="assistant", content=agent_msg))
        result.turn_count = turn + 1

        if user_sim.is_done:
            result.terminated_by = "user_stop"
            break

        logger.info("Turn %d: user responding", turn + 1)
        user_response = user_sim.respond(agent_msg)
        user_msg = _resolve_user_tool_calls(user_response, f"turn {turn + 1}")

        result.messages.append(Message(role="user", content=user_msg))

        if user_sim.is_done:
            result.terminated_by = "user_stop"
            break
    else:
        result.terminated_by = "max_turns"

    logger.info(
        "Conversation ended: %s after %d turns, %d tool calls",
        result.terminated_by,
        result.turn_count,
        len(result.tool_call_log),
    )
    return result
