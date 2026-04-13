"""Multi-turn conversation orchestrator for tau2 airline evaluations.

Drives a back-and-forth conversation between a deepagents agent and an
LLM-powered user simulator, collecting the full transcript and tool call
log for later evaluation.

Based on τ-bench / τ²-bench / τ³-bench by Sierra Research (MIT License).
See LICENSE in this directory.

Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.evals.utils import AgentTrajectory, run_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from tests.evals.tau2_airline.domain import ToolCallEntry
    from tests.evals.tau2_airline.user_sim import UserSimulator

logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 30


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
        total_agent_steps: Total AI-message steps across all turns.
        total_tool_calls: Total tool-call requests across all turns.
        total_input_tokens: Cumulative agent input tokens (from usage_metadata).
        total_output_tokens: Cumulative agent output tokens (from usage_metadata).
    """

    messages: list[Message] = field(default_factory=list)
    tool_call_log: list[ToolCallEntry] = field(default_factory=list)
    terminated_by: str = "max_turns"
    turn_count: int = 0
    total_agent_steps: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _accumulate_trajectory_metrics(
    result: ConversationResult,
    trajectory: AgentTrajectory,
) -> None:
    """Add a single turn's trajectory metrics to the running totals.

    Args:
        result: The conversation result to update in place.
        trajectory: The trajectory returned by ``run_agent`` for one turn.
    """
    result.total_agent_steps += len(trajectory.steps)
    result.total_tool_calls += sum(len(s.action.tool_calls) for s in trajectory.steps)
    for step in trajectory.steps:
        usage = getattr(step.action, "usage_metadata", None)
        if usage:
            result.total_input_tokens += usage.get("input_tokens", 0) or 0
            result.total_output_tokens += usage.get("output_tokens", 0) or 0


def run_multi_turn(
    agent: CompiledStateGraph[Any, Any],
    user_sim: UserSimulator,
    *,
    model: BaseChatModel,
    tool_call_log: list[ToolCallEntry],
    max_turns: int = DEFAULT_MAX_TURNS,
) -> ConversationResult:
    """Run a multi-turn conversation between the agent and user simulator.

    Args:
        agent: The compiled deepagents graph.
        user_sim: The LLM-powered user simulator.
        model: The agent's chat model (for logging only).
        tool_call_log: Shared mutable log populated by the airline tools.
        max_turns: Maximum number of agent turns before stopping.

    Returns:
        The full conversation result with transcript and tool call log.
    """
    thread_id = str(uuid.uuid4())
    result = ConversationResult(tool_call_log=tool_call_log)

    user_msg = user_sim.get_opening_message()
    result.messages.append(Message(role="user", content=user_msg))

    for turn in range(max_turns):
        logger.info("Turn %d: agent processing", turn + 1)
        trajectory = run_agent(
            agent,
            query=user_msg,
            model=model,
            thread_id=thread_id,
        )
        _accumulate_trajectory_metrics(result, trajectory)
        agent_msg = trajectory.answer
        result.messages.append(Message(role="assistant", content=agent_msg))
        result.turn_count = turn + 1

        if user_sim.is_done:
            result.terminated_by = "user_stop"
            break

        logger.info("Turn %d: user responding", turn + 1)
        user_msg = user_sim.respond(agent_msg)
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
