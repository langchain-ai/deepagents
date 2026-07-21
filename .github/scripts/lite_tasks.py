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
    # 8 — three calibrated hard tasks + four medium frontier tasks + one guard,
    # re-picked from the five-model full-context results after the first lite
    # confirmation run failed to preserve the target context ordering. All
    # selected tasks were independently validated against the vendored corpus
    # and answer keys.
    "context": [
        "cb-cloud-10",  # multi_hop_chain (hard)
        "cb-cloud-74",  # multi_entity_comparison (hard)
        "cb-cloud-90",  # multi_hop_chain (hard)
        "cb-cloud-15",  # multi_entity_comparison (medium)
        "cb-cloud-35",  # multi_entity_comparison (medium)
        "cb-cloud-62",  # multi_hop_chain (medium)
        "cb-cloud-67",  # multi_hop_chain (medium)
        "cb-cloud-8",  # cross_file_counting (guard)
    ],
}


def include_tasks(category: str) -> str:
    """Space-separated include-task filter string for a category, or '' if none."""
    return " ".join(LITE_TASKS.get(category, []))
