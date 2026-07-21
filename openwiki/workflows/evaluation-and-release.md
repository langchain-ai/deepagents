---
type: Engineering Workflow
title: Evaluation, Harbor scorecards, CI, and releases
description: How Deep Agents runs unit and live evaluations, aggregates unified Harbor scorecards, compares branches, and turns validated package releases into publications.
tags: [evaluations, harbor, ci, release, langsmith]
---
# Evaluation, Harbor scorecards, CI, and releases

`libs/evals` contains both normal test-adjacent evaluation tooling and paid/live model evaluation paths. It validates the core harness and coding-agent behavior described in [Runtime and package architecture](../architecture/overview.md) and [Deep Agents Code](deep-agents-code.md), while GitHub workflows turn package checks into release candidates and publications.

## Two evaluation paths

### Pytest/LangSmith behavioral evaluations

`libs/evals/tests/evals/` contains agent evaluations that use real models and collect trajectories, mutations, correctness, and efficiency. Collection requires a model and LangSmith tracing; this is intentionally separate from normal unit testing.

The pytest reporter produces structured JSON with pass/fail/skip counts, correctness/category scores, step/tool-call efficiency, solve rate, median duration, LangSmith experiment links, and bounded failure detail. Once tests ran, ordinary evaluation failures are rewritten to pytest exit code 0 so artifact aggregation still runs; callers must inspect `counts.failed` rather than treating the process return code as the quality result. `deepagents_evals/cli.py` and its trial aggregation do exactly that.

Trials run sequentially locally because sharing provider/LangSmith work in parallel in-process is unsafe. CI can distribute jobs and then aggregate. The `deepagents-evals` CLI supports `run`, `trials`, `aggregate`, `radar`, catalog/model discovery, JSON output, and dry-run modes.

### Harbor unified scorecard

`.github/workflows/unified_evals.yml` is a manually dispatched Harbor workflow. It accepts models, categories, harness configurations, profile/rollout/concurrency/sandbox controls, optional Harbor/judge configuration, and optional branch variants. It prepares full or frozen-lite task lists, builds a model × branch matrix, invokes the reusable `_harbor_run.yml`, downloads `harbor-*` artifacts, combines them with `.github/scripts/aggregate_unified.py`, uploads the combined result, and attempts radar-chart publication.

The aggregate is keyed by **model, branch, harness configuration, and category**. It computes macro/micro metrics and marks leaves missing or incomplete rather than silently ranking partial results. Current category mapping is autonomous → `harbor-index`, conversation → a Tau3 subset, and context → the local context-retrieval dataset.

## Branch comparisons: a neutral harness

HEAD introduced `feat(evals): compare branch variants with a neutral harness` (`ae96a9082`). `unified_prep.py` resolves selected branch refs to immutable remote SHAs, constructs model/branch job rows, caps total jobs at 400, and bounds provider/sandbox concurrency. `_harbor_run.yml` then overlays **only** `libs/deepagents`, `libs/code`, and `libs/partners/quickjs` from each selected branch after installing a workflow-ref-locked baseline.

That is intentionally a bounded comparison, not a full branch-environment recreation: evaluator/harness code, datasets, verifiers, and locked dependencies remain controlled by the workflow ref. The change also removed harness system-prompt injection in bare/Tau3 graphs so the harness does not hide prompt differences between branches. Artifact names, LangSmith experiments, and aggregate keys are branch- and harness-scoped to avoid collisions.

## Local commands and credentials

Run from `libs/evals` after installing groups with `uv`:

```bash
uv sync --all-groups
make test
make lint
make type

# Live work: requires an appropriate provider credential, LANGSMITH_API_KEY,
# and LANGSMITH_TRACING=true.
make evals MODEL=openai:gpt-5.5
make evals-trials MODEL=openai:gpt-5.5 TRIALS=5
uv run deepagents-evals run --model openai:gpt-5.5 --eval-category memory --report /tmp/evals.json
```

`make test` is socket-restricted unit testing. `make lint` checks formatting, linting, typing, and generated evaluation catalog freshness. Do not claim live-eval coverage based on a green unit test; live providers, LangSmith, and Harbor have distinct external dependencies and cost.

## CI and release chain

`ci.yml` runs on pull requests, merge groups, and pushes to main. It uses changed-package detection; eval source **or eval workflow** changes intentionally trigger eval package lint/tests. Reusable lint/test workflows use frozen UV environments and verify the worktree remains clean after checks. The final CI aggregation job reports success only from the needed job outcomes.

`release-please.yml` runs on main, guards package scope/empty commits, updates generated lockfiles, and identifies merged conventional release commits. It dispatches `release.yml` separately to preserve PyPI Trusted Publishing behavior. The release workflow maps package names to paths, validates an immutable release SHA and package version, then performs:

```text
build -> pre-release wheel checks -> TestPyPI -> PyPI -> GitHub tag/release
```

Release checks install/import built wheels and run package unit tests; integration tests are intentionally disabled there. A release-ready package can therefore pass the pipeline without a fresh live LangSmith/Harbor evaluation unless another policy/workflow runs one.

## Change checklist

- Updating an eval test, reporter, trial aggregation, or command-line result must preserve the distinction between execution errors and evaluation quality (`counts.failed`).
- Updating Harbor prep/matrix/aggregation must retain dimensional keys and explicit missing/incomplete reporting; otherwise branch/harness comparisons can become misleading.
- Updating agent prompts or core/coding-agent runtime may need both unit coverage and a targeted real-model or Harbor run; see the affected tests/catalog before choosing categories.
- Updating package version/release paths should start in the package’s `pyproject.toml`, `CHANGELOG.md`, and release workflows; use the normal validation commands in [Operations and testing](../engineering/operations-and-testing.md).
