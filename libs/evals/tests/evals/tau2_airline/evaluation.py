"""Evaluation logic for tau2 airline tasks.

Reimplements the core tau2 evaluation strategy:
- **DB check**: replay expected actions on a fresh database, compare final
  state against the actual database after the conversation.
- **Communicate check**: verify that expected information substrings appear
  in agent messages.
- **Action check**: verify that expected tool calls were made (informational).

The overall reward mirrors tau2: product of DB and COMMUNICATE scores.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau-bench
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.evals.tau2_airline.domain import (
    FlightDB,
    ToolCallEntry,
    create_airline_tools,
    load_db,
)

if TYPE_CHECKING:
    from tests.evals.tau2_airline.runner import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class ActionCheckResult:
    """Result of checking a single expected action."""

    name: str
    expected_args: dict[str, Any]
    matched: bool


@dataclass
class TaskReward:
    """Combined evaluation result for a tau2 task.

    Attributes:
        reward: Final reward (product of db_score and communicate_score).
        db_score: 1.0 if DB states match, 0.0 otherwise.
        communicate_score: Fraction of communicate_info items found.
        action_checks: Per-action match results (informational).
        details: Human-readable summary.
    """

    reward: float
    db_score: float
    communicate_score: float
    action_checks: list[ActionCheckResult] = field(default_factory=list)
    details: str = ""


@dataclass
class EpisodeScore:
    """Episode-level success + expectation-style diagnostics.

    Attributes:
        success: ``True`` when the episode satisfies hard correctness criteria.
        success_reasons: Machine-readable failure reasons when ``success=False``.
        expected_metrics: Non-blocking diagnostic metrics for observability.
    """

    success: bool
    success_reasons: list[str] = field(default_factory=list)
    expected_metrics: dict[str, float | int | str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DB state comparison
# ---------------------------------------------------------------------------


def _hash_db(db: FlightDB) -> str:
    """Compute a canonical hash of the database state."""
    data = db.model_dump()
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _replay_expected_actions(task: dict[str, Any]) -> FlightDB:
    """Create a fresh DB and replay the task's expected actions on it.

    Args:
        task: The raw task dict from tasks.json.

    Returns:
        The FlightDB after replaying all expected actions.
    """
    fresh_db = load_db()

    initial_state = task.get("initial_state")
    if initial_state:
        _apply_initial_state(fresh_db, initial_state)

    tools, _ = create_airline_tools(fresh_db)
    tools_by_name = {t.name: t for t in tools}

    expected_actions = task.get("evaluation_criteria", {}).get("actions", [])
    for action in expected_actions:
        name = action["name"]
        args = action.get("arguments", {})
        tool = tools_by_name.get(name)
        if tool is None:
            logger.warning("Expected action %r not found in tools", name)
            continue
        try:
            tool.invoke(args)
        except (ValueError, KeyError, TypeError):
            logger.warning("Failed to replay expected action %s(%s)", name, args, exc_info=True)

    return fresh_db


def _apply_initial_state(db: FlightDB, initial_state: dict[str, Any]) -> None:
    """Apply a task's initial_state mutations to the DB.

    The initial_state dict maps dotted paths to values, e.g.
    ``{"users.alice.membership": "gold"}``. Currently supports direct
    top-level collection updates.

    Args:
        db: The FlightDB to mutate.
        initial_state: The initial_state dict from the task.
    """
    for key, value in initial_state.items():
        parts = key.split(".")
        obj: Any = db
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        final_key = parts[-1]
        if isinstance(obj, dict):
            obj[final_key] = value
        else:
            setattr(obj, final_key, value)


def check_db_state(actual_db: FlightDB, task: dict[str, Any]) -> float:
    """Compare actual DB state against expected state after replaying actions.

    Args:
        actual_db: The DB after the real conversation.
        task: The raw task dict.

    Returns:
        1.0 if states match, 0.0 otherwise.
    """
    expected_db = _replay_expected_actions(task)
    actual_hash = _hash_db(actual_db)
    expected_hash = _hash_db(expected_db)
    match = actual_hash == expected_hash
    if not match:
        logger.info(
            "DB state mismatch: actual=%s expected=%s", actual_hash[:12], expected_hash[:12]
        )
    return 1.0 if match else 0.0


# ---------------------------------------------------------------------------
# Action checks (informational)
# ---------------------------------------------------------------------------


def check_actions(
    tool_log: list[ToolCallEntry],
    task: dict[str, Any],
) -> list[ActionCheckResult]:
    """Check whether each expected action was called.

    Uses greedy matching: each expected action is matched against the first
    unmatched log entry with the same name and compatible arguments.

    Args:
        tool_log: The recorded tool invocations from the conversation.
        task: The raw task dict.

    Returns:
        Per-action check results.
    """
    expected = task.get("evaluation_criteria", {}).get("actions", [])
    used: set[int] = set()
    results: list[ActionCheckResult] = []

    for action in expected:
        name = action["name"]
        exp_args = action.get("arguments", {})
        matched = False

        for i, entry in enumerate(tool_log):
            if i in used or entry.name != name:
                continue
            if _args_match(entry.args, exp_args):
                matched = True
                used.add(i)
                break

        results.append(ActionCheckResult(name=name, expected_args=exp_args, matched=matched))

    return results


def _args_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check if actual tool args contain all expected key-value pairs."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if actual[key] != value:
            return False
    return True


# ---------------------------------------------------------------------------
# Communicate checks
# ---------------------------------------------------------------------------


def check_communicate(
    messages: list[Message],
    task: dict[str, Any],
) -> float:
    """Check that expected information appears in agent messages.

    Args:
        messages: The conversation transcript.
        task: The raw task dict.

    Returns:
        Fraction of communicate_info items found (1.0 if none expected).
    """
    expected = task.get("evaluation_criteria", {}).get("communicate_info", [])
    if not expected:
        return 1.0

    agent_text = " ".join(m.content for m in messages if m.role == "assistant")
    found = sum(1 for info in expected if str(info) in agent_text)
    return found / len(expected)


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_task(
    actual_db: FlightDB,
    tool_log: list[ToolCallEntry],
    messages: list[Message],
    task: dict[str, Any],
) -> TaskReward:
    """Run all evaluators and compute the final reward.

    The reward is ``db_score * communicate_score``, matching tau2's
    ``reward_basis=['DB', 'COMMUNICATE']`` logic for airline tasks.

    Args:
        actual_db: The DB state after the conversation.
        tool_log: All tool invocations recorded during the conversation.
        messages: The full conversation transcript.
        task: The raw task dict from tasks.json.

    Returns:
        The combined task reward.
    """
    db_score = check_db_state(actual_db, task)
    comm_score = check_communicate(messages, task)
    action_results = check_actions(tool_log, task)
    reward = db_score * comm_score

    action_summary = (
        f"{sum(a.matched for a in action_results)}/{len(action_results)} actions matched"
        if action_results
        else "no expected actions"
    )
    details = f"DB={db_score:.0f}, COMM={comm_score:.2f}, actions={action_summary}"

    return TaskReward(
        reward=reward,
        db_score=db_score,
        communicate_score=comm_score,
        action_checks=action_results,
        details=details,
    )


def score_tau2_episode(reward: TaskReward) -> EpisodeScore:
    """Map a tau2 task reward into success + expectation metrics.

    Hard correctness follows the tau2 airline reward basis: the episode only
    succeeds when both DB and communicate checks are perfect.

    Args:
        reward: The combined reward object produced by ``evaluate_task``.

    Returns:
        Episode-level success status and expectation-style diagnostics.
    """
    success = reward.db_score == 1.0 and reward.communicate_score == 1.0
    success_reasons: list[str] = []
    if reward.db_score < 1.0:
        success_reasons.append("db_state_mismatch")
    if reward.communicate_score < 1.0:
        success_reasons.append("communicate_mismatch")

    actions_expected = len(reward.action_checks)
    actions_matched = sum(1 for action in reward.action_checks if action.matched)
    actions_match_rate = actions_matched / actions_expected if actions_expected else 1.0

    expected_metrics: dict[str, float | int | str] = {
        "actions_match_rate": actions_match_rate,
    }

    return EpisodeScore(
        success=success,
        success_reasons=success_reasons,
        expected_metrics=expected_metrics,
    )
