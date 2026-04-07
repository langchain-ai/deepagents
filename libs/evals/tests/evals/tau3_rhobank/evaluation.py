"""Evaluation logic for tau3 Rho-Bank banking tasks.

Mirrors the tau2 evaluation strategy:
- **DB check**: replay expected actions on a fresh database, compare final
  state against the actual database after the conversation.
- **Communicate check**: verify that expected information substrings appear
  in agent messages.
- **Action check**: verify that expected tool calls were made (informational).

The overall reward mirrors tau2: product of DB and COMMUNICATE scores.
All four tau3 tasks have `reward_basis: ["DB"]` (no communicate_info),
so communicate_score will always be 1.0 and db_score is the sole determinant.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tests.evals.tau3_rhobank.domain import (
    ToolCallEntry,
    TransactionalDB,
    create_rhobank_tools,
    load_db,
)

if TYPE_CHECKING:
    from tests.evals.tau3_rhobank.runner import Message

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
    """Combined evaluation result for a tau3 task.

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
        success: True when the episode satisfies hard correctness criteria.
        success_reasons: Machine-readable failure reasons when success=False.
        expected_metrics: Non-blocking diagnostic metrics for observability.
    """

    success: bool
    success_reasons: list[str] = field(default_factory=list)
    expected_metrics: dict[str, float | int | str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DB state comparison
# ---------------------------------------------------------------------------


def _hash_db(db: TransactionalDB) -> str:
    """Compute a canonical hash of the database state."""
    data = db.model_dump()
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _apply_initial_state(db: TransactionalDB, agent_data: dict[str, Any]) -> None:
    """Apply a task's initial_state agent_data to the DB.

    tau3 uses `initialization_data.agent_data` which is a dict of table
    names to `{"data": {...}}` patches that get merged into the DB tables.

    Args:
        db: The TransactionalDB to mutate.
        agent_data: The agent_data dict from the task's initial_state.
    """
    for table_name, table_update in agent_data.items():
        table = getattr(db, table_name, None)
        if table is not None and isinstance(table_update, dict) and "data" in table_update:
            table.data.update(table_update["data"])


def _replay_expected_actions(task: dict[str, Any]) -> TransactionalDB:
    """Create a fresh DB and replay the task's expected actions on it.

    Args:
        task: The raw task dict.

    Returns:
        The TransactionalDB after replaying all expected actions.
    """
    fresh_db = load_db()

    initial_state = task.get("initial_state")
    if initial_state:
        init_data = initial_state.get("initialization_data", {})
        agent_data = init_data.get("agent_data")
        if agent_data:
            _apply_initial_state(fresh_db, agent_data)

    agent_tools, _, user_tools, _ = create_rhobank_tools(fresh_db)
    agent_tools_by_name = {t.name: t for t in agent_tools}
    user_tools_by_name = {t.name: t for t in user_tools}

    expected_actions = task.get("evaluation_criteria", {}).get("actions", [])
    for action in expected_actions:
        name = action["name"]
        args = action.get("arguments", {})
        requestor = action.get("requestor", "assistant")

        if requestor == "user":
            tool = user_tools_by_name.get(name)
        else:
            tool = agent_tools_by_name.get(name)

        if tool is None:
            logger.warning("Expected action %r (requestor=%s) not found in tools", name, requestor)
            continue
        try:
            tool.invoke(args)
        except (ValueError, KeyError, TypeError):
            logger.warning("Failed to replay expected action %s(%s)", name, args, exc_info=True)

    return fresh_db


def _diff_db(actual_db: TransactionalDB, expected_db: TransactionalDB) -> list[str]:
    """Compute human-readable diffs between two DB states."""
    actual = actual_db.model_dump()
    expected = expected_db.model_dump()
    diffs: list[str] = []

    def _recurse(a: object, e: object, path: str) -> None:
        if isinstance(a, dict) and isinstance(e, dict):
            all_keys = set(a) | set(e)
            for k in sorted(all_keys):
                if k not in e:
                    diffs.append(f"  + {path}.{k} (extra in actual)")
                elif k not in a:
                    diffs.append(f"  - {path}.{k} (missing in actual)")
                else:
                    _recurse(a[k], e[k], f"{path}.{k}")
        elif isinstance(a, list) and isinstance(e, list):
            if len(a) != len(e):
                diffs.append(f"  {path}: len {len(a)} vs expected {len(e)}")
            for i in range(min(len(a), len(e))):
                _recurse(a[i], e[i], f"{path}[{i}]")
            diffs.extend(f"  + {path}[{i}] (extra in actual)" for i in range(len(e), len(a)))
        elif a != e:
            diffs.append(f"  {path}: {a!r} != expected {e!r}")

    _recurse(actual, expected, "db")
    return diffs


def check_db_state(actual_db: TransactionalDB, task: dict[str, Any]) -> float:
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
        diffs = _diff_db(actual_db, expected_db)
        logger.info(
            "DB state mismatch: actual=%s expected=%s", actual_hash[:12], expected_hash[:12]
        )
        for line in diffs[:30]:
            logger.info(line)
        if len(diffs) > 30:
            logger.info("  ... and %d more diffs", len(diffs) - 30)
    return 1.0 if match else 0.0


# ---------------------------------------------------------------------------
# Action checks (informational)
# ---------------------------------------------------------------------------


def check_actions(
    tool_log: list[ToolCallEntry],
    task: dict[str, Any],
) -> list[ActionCheckResult]:
    """Check whether each expected action was called.

    Uses greedy matching. Supports tau3's `compare_args` field: when
    present, only those specific argument keys are compared.

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
        compare_args = action.get("compare_args")
        matched = False

        for i, entry in enumerate(tool_log):
            if i in used or entry.name != name:
                continue
            if _args_match(entry.args, exp_args, compare_args):
                matched = True
                used.add(i)
                break

        results.append(ActionCheckResult(name=name, expected_args=exp_args, matched=matched))

    return results


def _args_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    compare_args: list[str] | None = None,
) -> bool:
    """Check if actual tool args match expected ones.

    When `compare_args` is provided, only those keys are compared.
    Otherwise all expected keys must match.

    Args:
        actual: The actual arguments from the tool call.
        expected: The expected arguments from the task.
        compare_args: Optional list of argument keys to compare.

    Returns:
        True if the arguments match.
    """
    keys_to_check = compare_args if compare_args is not None else list(expected.keys())
    for key in keys_to_check:
        if key not in expected:
            continue
        if key not in actual:
            return False
        if actual[key] != expected[key]:
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
    actual_db: TransactionalDB,
    tool_log: list[ToolCallEntry],
    messages: list[Message],
    task: dict[str, Any],
) -> TaskReward:
    """Run all evaluators and compute the final reward.

    Args:
        actual_db: The DB state after the conversation.
        tool_log: All tool invocations recorded during the conversation.
        messages: The full conversation transcript.
        task: The raw task dict.

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


def score_tau3_episode(reward: TaskReward) -> EpisodeScore:
    """Map a tau3 task reward into success + expectation metrics.

    Success requires a perfect DB state match (all expected actions executed
    with the correct arguments). The `actions_match_rate` is logged as the
    primary continuous score so experiments show a 0-1 gradient rather than
    binary pass/fail.

    Args:
        reward: The combined reward object produced by `evaluate_task`.

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
        "score": actions_match_rate,
    }

    return EpisodeScore(
        success=success,
        success_reasons=success_reasons,
        expected_metrics=expected_metrics,
    )
