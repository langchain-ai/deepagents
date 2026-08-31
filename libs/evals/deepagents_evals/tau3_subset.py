"""Curated tau3-bench subset for probing deep-agent conversation behavior.

30 tasks (drawn from telecom + banking_knowledge) stratified by difficulty for a
behavior spread, not leaderboard parity. Tiers are the **measured** pass rate of
`anthropic:claude-opus-4-8` over 3 rollouts per task at full agent timeout
(langsmith sandbox, tau3-runtime user simulator on gpt-5.2):

- easy   = solved 3/3 rollouts (reliably passes)
- medium = solved 1-2/3 rollouts (passes intermittently)
- hard   = solved 0/3 rollouts (not solved)

Opus finds most of this set hard, which is expected/acceptable headroom for a
difficulty probe. Living selection: re-run and re-tier here (updating each
`justification` with the new pass rate) as the reference model or task set
changes. `INCLUDE_TASKS` is derived from `TASKS` — CI reads it with::

    python -c "from deepagents_evals.tau3_subset import INCLUDE_TASKS; print(INCLUDE_TASKS)"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DATASET = "sierra-research/tau3-bench"

Tier = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class SubsetTask:
    """One curated task: its local id, difficulty tier, and why it sits there."""

    task_id: str
    tier: Tier
    justification: str

    def __post_init__(self) -> None:
        """Reject malformed rows at construction time.

        The module is built entirely of module-level `SubsetTask(...)`
        literals, so import doubles as a self-test. `tier` is additionally
        constrained statically by the `Tier` literal.
        """
        if not self.task_id.startswith("tau3-"):
            msg = f"task_id must start with 'tau3-': {self.task_id!r}"
            raise ValueError(msg)
        if not self.justification.strip():
            msg = f"justification must be non-empty for {self.task_id!r}"
            raise ValueError(msg)


TASKS: tuple[SubsetTask, ...] = (
    # --- EASY ---
    SubsetTask(
        task_id="tau3-banking_knowledge-task-050",
        tier="easy",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 3/3 rollouts (full timeout) — reliably passes.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-093",
        tier="easy",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 3/3 rollouts (full timeout) — reliably passes.",
    ),
    # --- MEDIUM ---
    SubsetTask(
        task_id="tau3-telecom-service-issue-break-apn-settings-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-easy",
        tier="medium",
        justification="Telecom; Opus 4.8 solved 2/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-043",
        tier="medium",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-056",
        tier="medium",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-072",
        tier="medium",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-contract-end-suspension-unseat-sim-card-persona-easy",
        tier="medium",
        justification="Telecom; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-none",
        tier="medium",
        justification="Telecom; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-lock-sim-card-pin-unseat-sim-card-persona-hard",
        tier="medium",
        justification="Telecom; Opus 4.8 solved 1/3 rollouts — passes intermittently.",
    ),
    # --- HARD ---
    SubsetTask(
        task_id="tau3-banking_knowledge-task-018",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-026",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-029",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-039",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-040",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-048",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-052",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-061",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-064",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-070",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-071",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-073",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-077",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-079",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-080",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-081",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-091",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-096",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-banking_knowledge-task-097",
        tier="hard",
        justification="Banking knowledge-retrieval; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-lock-sim-card-pin-persona-none",
        tier="hard",
        justification="Telecom; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
    SubsetTask(
        task_id="tau3-telecom-service-issue-airplane-mode-on-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-easy",
        tier="hard",
        justification="Telecom; Opus 4.8 solved 0/3 rollouts — not solved.",
    ),
)

# A duplicate task_id would double-weight a task and skew the difficulty
# distribution while silently passing every len()==30 check; reject it at import
# (a copy-paste slip during re-tiering is the likely cause).
if len({t.task_id for t in TASKS}) != len(TASKS):
    _msg = "duplicate task_id in TASKS"
    raise ValueError(_msg)

INCLUDE_TASKS = " ".join(f"{DATASET}__{t.task_id}" for t in TASKS)
"""Space-separated Harbor `include_tasks` value for the workflow's dataset."""
