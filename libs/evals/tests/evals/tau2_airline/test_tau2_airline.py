"""Parametrized pytest tests for 15 tau2 airline tasks.

Each test creates a fresh airline environment, runs a multi-turn conversation
between a deepagents agent and an LLM user simulator, then evaluates the
result using tau2's DB state + communicate info scoring.

Based on τ-bench / τ²-bench / τ³-bench by Sierra Research (MIT License).
See LICENSE in this directory. Task data updated to τ³-bench v1.0.0 which
includes corrections to all 15 evaluated tasks.
Source: https://github.com/sierra-research/tau2-bench (dev/tau3 branch)

Usage:
    uv run --group test pytest tests/evals/tau2_airline/ -v --model claude-sonnet-4-20250514
    uv run --group test pytest tests/evals/tau2_airline/ -k "task_2" --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t

from tests.evals.tau2_airline.domain import (
    create_airline_tools,
    load_db,
    load_policy,
    load_task,
)
from tests.evals.tau2_airline.evaluation import evaluate_task, score_tau2_episode
from tests.evals.tau2_airline.runner import run_multi_turn
from tests.evals.tau2_airline.user_sim import UserSimulator

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("conversation")]

logger = logging.getLogger(__name__)

TASK_IDS = [
    "2",
    "5",
    "7",
    "9",
    "14",
    "23",
    "27",
    "29",
    "32",
    "33",
    "35",
    "37",
    "38",
    "39",
    "44",
]

AGENT_SYSTEM_PROMPT = """\
You are a customer service agent that helps the user according to the <policy> provided below.
Use the available tools to look up information, verify customer identity, and take actions.
Always follow the policy. Be helpful, concise, and accurate.

<policy>
{domain_policy}
</policy>\
"""

USER_SIM_MODEL = "gpt-4.1-mini"


def _task_id_label(task_id: str) -> str:
    """Generate a readable pytest ID."""
    return f"task_{task_id}"


@pytest.mark.langsmith
@pytest.mark.parametrize("task_id", TASK_IDS, ids=_task_id_label)
def test_tau2_airline(model: BaseChatModel, task_id: str) -> None:
    """Run a multi-turn tau2 airline task and evaluate the result.

    Args:
        model: The agent's chat model (from --model CLI option).
        task_id: The tau2 task ID to run.
    """
    # Immediately override @pytest.mark.langsmith auto-capture so the dataset
    # example records clean metadata even if run_multi_turn() raises.
    _clean_inputs = {
        "task_id": task_id,
        "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
    }
    t.log_inputs(_clean_inputs)

    task = load_task(task_id)
    policy = load_policy()

    db = load_db()
    initial_state = task.get("initial_state")
    if initial_state:
        for key, value in initial_state.items():
            parts = key.split(".")
            obj = db
            for part in parts[:-1]:
                obj = getattr(obj, part) if not isinstance(obj, dict) else obj[part]
            final_key = parts[-1]
            if isinstance(obj, dict):
                obj[final_key] = value
            else:
                setattr(obj, final_key, value)

    tools, tool_log = create_airline_tools(db)

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT.format(domain_policy=policy),
        checkpointer=MemorySaver(),
    )

    user_model = init_chat_model(USER_SIM_MODEL)
    user_sim = UserSimulator(model=user_model, scenario=task.get("user_scenario", {}))

    conversation = run_multi_turn(
        agent,
        user_sim,
        model=model,
        tool_call_log=tool_log,
        max_turns=30,
    )

    # Override per-turn t.log_inputs() calls from run_agent() inside
    # run_multi_turn() with clean test-level metadata.
    t.log_inputs(_clean_inputs)

    reward = evaluate_task(
        actual_db=db,
        tool_log=tool_log,
        messages=conversation.messages,
        task=task,
    )
    episode_score = score_tau2_episode(reward)

    t.log_feedback(key="db_score", value=reward.db_score)
    t.log_feedback(key="communicate_score", value=reward.communicate_score)
    t.log_feedback(key="turn_count", value=conversation.turn_count)
    for key, value in episode_score.expected_metrics.items():
        t.log_feedback(key=key, value=value)

    logger.info(
        "Task %s: success=%s reasons=%s (%s), %d turns, %d tool calls",
        task_id,
        episode_score.success,
        ",".join(episode_score.success_reasons) if episode_score.success_reasons else "none",
        reward.details,
        conversation.turn_count,
        len(tool_log),
    )

    assert episode_score.success, (
        f"Task {task_id} failed: reasons={episode_score.success_reasons} details={reward.details}\n"
        f"Tool calls: {[e.name for e in tool_log]}\n"
        f"Terminated by: {conversation.terminated_by}"
    )
