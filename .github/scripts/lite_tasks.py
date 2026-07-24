"""Frozen 'lite' task subsets per category for the unified evals `profile=lite`.

A high-signal, low-cost slice: fewer tasks, FULL rollouts. Autonomous and
conversation use their calibrated difficulty frontiers. Context is a neutral,
paired Terra/Luna representative sample of the frozen 30-task corpus so a lite
run does not amplify either model's measured Context-Bench advantage.

Names are the exact harbor `--include-task-name` filters per category:
  autonomous   -> registry ref `harbor-index/<task>`
  conversation -> `sierra-research/tau3-bench__<task_id>` (same form as tau3_subset)
  context      -> local task dir basename `cb-cloud-<n>`

`include_tasks(category)` returns the space-separated string the workflow passes to
`_harbor_run.yml`. Keep this list under review; re-calibrate as models/tasks change.
"""

from __future__ import annotations

LITE_TASKS: dict[str, list[str]] = {
    # 15 — luna is weak here, so a rich frontier: partials + hard-but-solvable.
    # Excludes the bix* bioinformatics tasks: their ~6 GB `chenzizhao/bixbench`
    # image exhausts the Docker-sandbox runner disk (and fails the LangSmith
    # builder). Re-add once lite runs on a sandbox that builds big images.
    # `gpqadiamond-cope-rearrangement-products` and `swesmith-fix-oauth1-header-params`
    # replaced `replicationbench-find-galactic-vz-peaks` and `usaco-assign-cows-to-barns`:
    # both originals could run ~30-65 min (replicationbench's naive big-data script
    # hits the 1h per-command timeout), defeating lite's low-cost goal. The swaps are
    # <5 min and still frontier (partial-pass), re-picked on gpt-5.6-terra timing+signal.
    "autonomous": [
        "harbor-index/gpqadiamond-cope-rearrangement-products",
        "harbor-index/swebenchverified-fix-span-selector-axes-limits",
        "harbor-index/omnimath-find-perfect-square-functions",
        "harbor-index/swesmith-fix-oauth1-header-params",
        "harbor-index/featurebench-add-feature-mlflow-bedrock-autolog",
        "harbor-index/build-word2vec-pipeline",
        "harbor-index/tb-dna-insert",
        "harbor-index/swebenchverified-fix-django-mti-parent-link",
        "harbor-index/arcagi2-grid-transform-8b7b",
        "harbor-index/labbench-habenula-fluorescence-change",
        "harbor-index/labbench-read-asap2f-step-response",
        "harbor-index/gso-speedup-pydantic-enum",
        "harbor-index/swebenchpro-fix-file-suffix-chooser",
        "harbor-index/spider2-dbt-airport-arrivals",
        "harbor-index/arcagi2-grid-transform-a32d",
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
    # 10 — every Context-Bench query type, with extra deep-comparison and
    # multi-hop tasks. This 1 easy / 3 medium / 6 hard source-tier slice is
    # selected from the completed six-model, three-rollout full-30 run
    # (29883830538). It is the closest all-model profile among stable,
    # source-balanced candidates that preserves the full run's strict observed
    # order; it does not target an external leaderboard order.
    "context": [
        "cb-cloud-48",  # aggregation (medium)
        "cb-cloud-1",  # comparison_tiebreak (easy)
        "cb-cloud-21",  # cross_file_counting (medium)
        "cb-cloud-49",  # multi_entity_comparison (hard)
        "cb-cloud-65",  # multi_entity_comparison (hard)
        "cb-cloud-69",  # multi_hop_chain (hard)
        "cb-cloud-57",  # multi_hop_chain (hard)
        "cb-cloud-9",  # negation (medium)
        "cb-cloud-7",  # set_intersection (hard)
        "cb-cloud-4",  # temporal_reasoning (hard)
    ],
}


def include_tasks(category: str) -> str:
    """Space-separated include-task filter string for a category, or '' if none."""
    return " ".join(LITE_TASKS.get(category, []))
