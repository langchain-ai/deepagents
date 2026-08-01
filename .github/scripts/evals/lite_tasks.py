"""Frozen 'lite' task subsets per category for the unified evals `profile=lite`.

A high-signal, low-cost slice: fewer tasks, FULL rollouts. Autonomous and
conversation use their calibrated difficulty frontiers. Context is a neutral,
paired Terra/Luna representative sample of the frozen 30-task corpus so a lite
run does not amplify either model's measured Context-Bench advantage.

Names are the exact harbor `--include-task-name` filters per category:
  autonomous   -> registry ref `harbor-index/<task>`
  conversation -> `sierra-research/tau3-bench__<task_id>` (same form as tau3_subset)
  context      -> local task dir basename `cb-cloud-<n>`
  research     -> local task dir basename `DR<nnnn>`

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
    # 10 — one task per research domain, the ten domains DRBench covers most heavily
    # (after collapsing upstream's synonym labels: `market_analysis`/`market analysis`,
    # `itsm`/`it service management`, `crm`/`customer relationship management`,
    # `quality_assurance`/`quality assurance`; tasks whose `domain` is really an
    # industry label are excluded from the per-domain pick).
    #
    # Unlike the other categories this is not calibrated against a measured run —
    # there isn't one yet. It is a coverage slice: 4 easy / 4 medium / 2 hard, 76 gold
    # insights (24 external) across 172 documents, and 8 of 10 tasks carry external
    # (open-web) insights so a lite run still exercises search rather than corpus
    # reading alone. Per-domain pick favours
    # tasks with external insights, then a moderate corpus (<= 25 documents), then an
    # insight count near the dataset mean of 6. Re-calibrate once a full run exists.
    "research": [
        "DR0006",  # compliance / healthcare (easy) — 7 insights, 3 external, 12 docs
        "DR0003",  # crm / retail (medium) — 10 insights, 3 external, 22 docs
        "DR0025",  # csm / automobiles (hard) — 5 insights, 0 external, 20 docs
        "DR0013",  # cybersecurity / automobiles (medium) — 9 insights, 3 external, 18 docs
        "DR0007",  # itsm / healthcare (easy) — 9 insights, 3 external, 18 docs
        "DR0029",  # knowledge management / retail (hard) — 4 insights, 0 external, 18 docs
        "DR0004",  # market analysis / retail (medium) — 9 insights, 3 external, 20 docs
        "DR0012",  # quality assurance / automobiles (easy) — 6 insights, 3 external, 9 docs
        "DR0014",  # research / automobiles (medium) — 10 insights, 3 external, 21 docs
        "DR0002",  # sales / retail (easy) — 7 insights, 3 external, 14 docs
    ],
}


def include_tasks(category: str) -> str:
    """Space-separated include-task filter string for a category, or '' if none."""
    return " ".join(LITE_TASKS.get(category, []))
