"""Parametrized pytest tests for 4 tau3 Rho-Bank banking tasks.

Each test creates a fresh banking environment, runs a multi-turn conversation
between a deepagents agent and an LLM user simulator, then evaluates the
result using tau3's DB state + communicate info scoring.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench

Usage:
    uv run --group test pytest tests/evals/tau3_rhobank/ -v --model claude-sonnet-4-20250514
    uv run --group test pytest tests/evals/tau3_rhobank/ -k "task_017" --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langsmith import testing as t
from langsmith.run_helpers import get_current_run_tree

from tests.evals.tau3_rhobank.domain import (
    create_rhobank_tools,
    get_documents_dir,
    load_db,
    load_task,
)
from tests.evals.tau3_rhobank.evaluation import evaluate_task, score_tau3_episode
from tests.evals.tau3_rhobank.runner import run_multi_turn
from tests.evals.tau3_rhobank.user_sim import UserSimulator

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [
    pytest.mark.eval_category("conversation"),
    pytest.mark.eval_category("tau3"),
]

logger = logging.getLogger(__name__)

TASK_IDS = ["017", "041", "080", "081"]

AGENT_SYSTEM_PROMPT = """\
You are a customer service agent for Rho-Bank.

Rho-Bank's knowledge base of policy and procedure documents is available as \
JSON files in the following directory:

  {documents_dir}

Each file is a JSON object with "id", "title", and "content" fields. Use your \
file-search and reading tools to locate the relevant policies before taking \
action. Do NOT guess policies -- always verify by reading the documents first.

Use the available tools to look up information, verify identity, and take actions.
When the knowledge base tells you to unlock a tool, use unlock_discoverable_agent_tool.
When the knowledge base tells you to give a tool to the user, use give_discoverable_user_tool.\
"""

USER_SIM_MODEL = "gpt-4.1-mini"


def _task_id_label(task_id: str) -> str:
    """Generate a readable pytest ID."""
    return f"task_{task_id}"


@pytest.mark.langsmith
@pytest.mark.parametrize("task_id", TASK_IDS, ids=_task_id_label)
def test_tau3_rhobank(model: BaseChatModel, task_id: str) -> None:
    """Run a multi-turn tau3 Rho-Bank task and evaluate the result.

    Args:
        model: The agent's chat model (from --model CLI option).
        task_id: The tau3 task ID to run.
    """
    _clean_inputs = {
        "task_id": task_id,
        "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
    }
    t.log_inputs(_clean_inputs)
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.inputs = _clean_inputs
    else:
        logger.warning(
            "get_current_run_tree() returned None in @pytest.mark.langsmith test; "
            "dataset example inputs will not be overridden"
        )

    task = load_task(task_id)

    db = load_db()
    initial_state = task.get("initial_state")
    if initial_state:
        init_data = initial_state.get("initialization_data", {})
        agent_data = init_data.get("agent_data")
        if agent_data:
            for table_name, table_update in agent_data.items():
                table = getattr(db, table_name, None)
                if table is not None and isinstance(table_update, dict) and "data" in table_update:
                    table.data.update(table_update["data"])

    agent_tools, agent_tool_log, user_tools, user_tool_log = create_rhobank_tools(db)

    agent = create_deep_agent(
        model=model,
        tools=agent_tools,
        system_prompt=AGENT_SYSTEM_PROMPT.format(documents_dir=get_documents_dir()),
        checkpointer=MemorySaver(),
    )

    user_model = init_chat_model(USER_SIM_MODEL)

    task_user_tools_names = task.get("user_tools", [])
    active_user_tools = [ut for ut in user_tools if ut.name in task_user_tools_names]

    user_sim = UserSimulator(
        model=user_model,
        scenario=task.get("user_scenario", {}),
        user_tools=active_user_tools if active_user_tools else None,
    )

    combined_log = agent_tool_log

    conversation = run_multi_turn(
        agent,
        user_sim,
        model=model,
        tool_call_log=combined_log,
        user_tools=active_user_tools,
        max_turns=60,
    )

    t.log_inputs(_clean_inputs)

    all_tool_calls = agent_tool_log + user_tool_log

    reward = evaluate_task(
        actual_db=db,
        tool_log=all_tool_calls,
        messages=conversation.messages,
        task=task,
    )
    episode_score = score_tau3_episode(reward)

    t.log_feedback(key="db_score", score=reward.db_score)
    t.log_feedback(key="communicate_score", score=reward.communicate_score)
    t.log_feedback(key="turn_count", score=conversation.turn_count)
    for metric_key, metric_val in episode_score.expected_metrics.items():
        t.log_feedback(key=metric_key, score=metric_val)

    logger.info(
        "Task %s: success=%s reasons=%s (%s), %d turns, %d tool calls",
        task_id,
        episode_score.success,
        ",".join(episode_score.success_reasons) if episode_score.success_reasons else "none",
        reward.details,
        conversation.turn_count,
        len(all_tool_calls),
    )

    assert episode_score.success, (
        f"Task {task_id} failed: reasons={episode_score.success_reasons} details={reward.details}\n"
        f"Tool calls: {[e.name for e in all_tool_calls]}\n"
        f"Terminated by: {conversation.terminated_by}"
    )
