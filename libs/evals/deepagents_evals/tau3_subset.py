"""Curated tau3-bench subset for probing deep-agent conversation behavior.

30 tasks stratified by difficulty for a behavior spread, not leaderboard parity:
6 easy (telecom), 15 medium + 9 hard (banking_knowledge). Difficulty is a no-run
structural proxy — a per-task complexity score (required actions + env/nl
assertions + communicated info + banking retrieval documents). Domain carries
easy→medium (telecom vs banking domain pass rates); within-banking complexity
carries medium→hard.

Living selection: swap a task by editing its entry (and justification) here when
observed pass rates disagree with the assigned tier. ``INCLUDE_TASKS`` is derived
from ``TASKS`` — CI reads it with::

    python -c "from deepagents_evals.tau3_subset import INCLUDE_TASKS; print(INCLUDE_TASKS)"
"""

from __future__ import annotations

from dataclasses import dataclass

DATASET = "sierra-research/tau3-bench"


@dataclass(frozen=True)
class SubsetTask:
    """One curated task: its local id, difficulty tier, and why it sits there."""

    task_id: str
    tier: str  # "easy" | "medium" | "hard"
    justification: str


TASKS: tuple[SubsetTask, ...] = (
    # --- EASY ---
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-contract-end-suspension-unseat-sim-card-persona-easy",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-lock-sim-card-pin-persona-none",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-none",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-easy",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-lock-sim-card-pin-unseat-sim-card-persona-hard",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-break-apn-settings-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-easy",
        tier="easy",
        justification="Telecom (domain pass-rate ~82%); minimal grading criteria (complexity 2: 1 action(s), 1 env-assertion(s)) — expected to pass most runs.",
    ),
    # --- MEDIUM ---
    SubsetTask(
        task_id="tau3-banking_knowledge-task-018",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 18: 8 action(s), 10 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-029",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 18: 8 action(s), 10 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-072",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 18: 9 action(s), 9 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-026",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 19: 11 action(s), 8 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-040",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 19: 15 action(s), 4 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-050",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 19: 13 action(s), 6 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-052",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 20: 13 action(s), 7 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-056",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 20: 8 action(s), 12 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-064",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 20: 4 action(s), 16 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-093",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 20: 9 action(s), 11 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-039",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 21: 16 action(s), 5 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-061",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 21: 9 action(s), 12 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-070",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 21: 5 action(s), 16 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-073",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 21: 11 action(s), 10 required doc(s)) — expected to pass intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-043",
        tier="medium",
        justification="Banking knowledge-retrieval (domain pass-rate ~18%); mid-band complexity (complexity 22: 15 action(s), 7 required doc(s)) — expected to pass intermittently.",
    ),
    # --- HARD ---
    SubsetTask(
        task_id="tau3-banking_knowledge-task-048",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 34: 24 action(s), 10 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-077",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 35: 23 action(s), 12 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-079",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 35: 24 action(s), 11 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-096",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 35: 12 action(s), 23 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-071",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 36: 6 action(s), 30 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-091",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 38: 25 action(s), 13 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-097",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 43: 18 action(s), 25 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-081",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 46: 33 action(s), 13 required doc(s)) — expected to rarely pass.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-080",
        tier="hard",
        justification="Banking knowledge-retrieval; high complexity (complexity 49: 30 action(s), 19 required doc(s)) — expected to rarely pass.",
    ),
)

INCLUDE_TASKS = " ".join(f"{DATASET}__{t.task_id}" for t in TASKS)
"""Space-separated Harbor ``include_tasks`` value for the workflow's dataset."""
