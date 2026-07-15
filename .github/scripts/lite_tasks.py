"""Frozen 'lite' task subsets per category for the unified evals `profile=lite`.

A high-signal, low-cost slice: fewer tasks, FULL rollouts. Tasks are biased to the
difficulty frontier (partial-pass / hard-but-solvable) measured on a weaker model
(gpt-5.6-luna); saturated (all-pass) and verifier-unstable tasks are excluded.

Names are the exact harbor `--include-task-name` filters per category:
  autonomous   -> registry ref `harbor-index/<task>`
  conversation -> `sierra-research/tau3-bench__<task_id>` (same form as tau3_subset)
  context      -> local task dir basename `cb-cloud-<n>`

`include_tasks(category)` returns the space-separated string the workflow passes to
`_harbor_run.yml`. Keep this list under review; re-calibrate as models/tasks change.
"""

from __future__ import annotations

LITE_TASKS: dict[str, list[str]] = {
    # Isolated rerun of the three autonomous Docker-sandbox failures.
    "autonomous": [
        "harbor-index/swebenchverified-fix-span-selector-axes-limits",
        "harbor-index/featurebench-add-feature-mlflow-bedrock-autolog",
        "harbor-index/swebenchverified-fix-django-mti-parent-link",
    ],
    # 11 — luna is weak on banking (rich frontier); telecom is saturated (1 kept).
    "conversation": [
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-043",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-056",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-093",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-018",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-029",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-040",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-048",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-061",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-072",
        "sierra-research/tau3-bench__tau3-banking_knowledge-task-073",
        "sierra-research/tau3-bench__tau3-telecom-service-issue-airplane-mode-on-break-apn-settings-lock-sim-card-pin-overdue-bill-suspension-unseat-sim-card-persona-none",
    ],
    # 8 — luna is strong on context, so the frontier is thin: its partials + fails.
    "context": [
        "cb-cloud-10",
        "cb-cloud-41",
        "cb-cloud-47",
        "cb-cloud-0",
        "cb-cloud-26",
        "cb-cloud-49",
        "cb-cloud-15",
        "cb-cloud-5",
    ],
}


def include_tasks(category: str) -> str:
    """Space-separated include-task filter string for a category, or '' if none."""
    return " ".join(LITE_TASKS.get(category, []))
