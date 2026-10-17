"""Frozen task subsets per category for the unified evals run profiles.

`LITE_TASKS` backs `profile=lite`: a high-signal, low-cost slice — fewer tasks, FULL
rollouts. Autonomous and conversation use their calibrated difficulty frontiers.
Context is a neutral, paired Terra/Luna representative sample of the frozen 30-task
corpus so a lite run does not amplify either model's measured Context-Bench advantage.
Research is pinned to upstream DRBench's own 15-task subset (see below).

`FULL_TASKS` caps `profile=full` for the few categories where running the entire
dataset costs more than a representative sample is worth. A category absent from it
keeps `full` meaning every task, which is the default for all but `research`.

Names are the exact harbor `--include-task-name` filters per category:
  autonomous   -> registry ref `harbor-index/<task>`
  conversation -> `sierra-research/tau3-bench__<task_id>` (same form as tau3_subset)
  context      -> local task dir basename `cb-cloud-<n>`
  research     -> local task dir basename `DR<nnnn>`

`include_tasks(category)` returns the space-separated string the workflow passes to
`_harbor_run.yml`. Keep these lists under review; re-calibrate as models/tasks change.
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
    # Not ours to choose: this is upstream DRBench's own subset, the one the paper calls
    # MinEval ("restricted to 15 tasks for efficient ablation studies"). The paper names
    # it but never lists the ids; upstream ships them in `drbench/data/subsets/minival.jsonl`,
    # vendored at `libs/evals/harbor_adapters/drbench/vendor/subsets/minival.jsonl` and
    # asserted equal to this list by `test_lite_tasks.py`.
    #
    # It is a designed stratified block rather than the first 15 by coincidence:
    # 3 industries x 5 tasks, difficulty easy/easy/medium/medium/hard per industry, so
    # 6 easy / 6 medium / 3 hard over 10 distinct domains. DR0016 onward does NOT continue
    # the pattern — the rest of the corpus is unstructured and hard-skewed — which is why
    # `FULL_TASKS` extends it by stratified selection rather than by id order.
    "research": [
        "DR0001",  # compliance / retail (easy)
        "DR0002",  # sales / retail (easy)
        "DR0003",  # crm / retail (medium)
        "DR0004",  # market analysis / retail (medium)
        "DR0005",  # csm / retail (hard)
        "DR0006",  # compliance / healthcare (easy)
        "DR0007",  # itsm / healthcare (easy)
        "DR0008",  # cybersecurity / healthcare (medium)
        "DR0009",  # crm / healthcare (medium)
        "DR0010",  # marketing / healthcare (hard)
        "DR0011",  # compliance / automotive (easy)
        "DR0012",  # quality assurance / automotive (easy)
        "DR0013",  # cybersecurity / automotive (medium)
        "DR0014",  # research / automotive (medium)
        "DR0015",  # csm / automotive (hard)
    ],
}

# Categories whose `profile=full` runs a representative subset instead of the whole
# dataset. Anything absent keeps `full` meaning every task.
FULL_TASKS: dict[str, list[str]] = {
    # 30 tasks. DRBench ships 100, and running all of them costs ~$150-200 per model per
    # run, so this is a proportional sample instead: the full corpus is 20 easy / 23 medium
    # / 57 hard, which scaled to 30 is exactly 6 / 7 / 17. MinEval (see LITE_TASKS above)
    # already supplies 6 easy / 6 medium / 3 hard, so the 15 additions below are precisely
    # 1 medium and 14 hard. That lands on the benchmark's own difficulty ratio while making
    # lite a strict subset of full, so the two profiles share 15 comparable data points.
    #
    # Industry comes out 10 retail / 10 healthcare / 10 automotive across 13 distinct
    # domains, at most 3 per domain. Additions were picked by holding the difficulty target
    # and then repeatedly taking the task whose (domain, industry) pair was least
    # represented so far, lowest id breaking ties — reproducible rather than taste.
    #
    # A number from this set is NOT comparable to the paper's FullBenchmark, which is over
    # all 100 tasks. Say so wherever it is published.
    #
    # Two upstream `info.json` defects shape the picks. Four tasks carry an industry name
    # in their `domain` field (DR0082 "retail", DR0083 "virtual/remote healthcare", DR0084
    # and DR0085 "automotive -- electric vehicles"), so they earn no domain-diversity credit
    # here; an earlier pass selected three of them precisely because those bogus values
    # looked like new coverage. And domain labels are double-counted by formatting
    # (crm/customer relationship management, itsm/it service management,
    # market analysis/market_analysis), which normalizes 20 raw values down to 13.
    "research": [
        *(f"DR{n:04d}" for n in range(1, 16)),  # MinEval: 6 easy / 6 medium / 3 hard
        "DR0025",  # csm / retail (hard)
        "DR0029",  # knowledge management / retail (hard)
        "DR0030",  # knowledge management / retail (hard)
        "DR0031",  # knowledge management / retail (hard)
        "DR0037",  # public relations / retail (medium)
        "DR0040",  # crm / healthcare (hard)
        "DR0043",  # market analysis / healthcare (hard)
        "DR0044",  # market analysis / healthcare (hard)
        "DR0046",  # itsm / healthcare (hard)
        "DR0047",  # itsm / healthcare (hard)
        "DR0064",  # sales / automotive (hard)
        "DR0065",  # sales / automotive (hard)
        "DR0076",  # asset management / automotive (hard)
        "DR0077",  # asset management / automotive (hard)
        "DR0089",  # quality assurance / automotive (hard)
    ],
}


def include_tasks(category: str) -> str:
    """Space-separated include-task filter string for a category, or '' if none."""
    return " ".join(LITE_TASKS.get(category, []))
