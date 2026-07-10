#!/usr/bin/env python3
"""Aggregate Harbor shard results into dataset-level pass@k and avg@k metrics.

Reads every per-trial ``result.json`` under a directory tree (the merged output
of all shard artifacts), groups trials by task, and computes:

  - pass@K  the fraction of tasks that passed at least once within K rollouts
            (K = rollouts_per_task; only this single k is reported).
  - avg@K   the fraction of all rollouts that passed, i.e. passing trials over
            total trials (== passing rollouts / (tasks * rollouts)).

A trial is a pass only when its verifier reward is >= PASS_THRESHOLD. Errored,
timed-out, or missing-verifier trials count as failures.

Aggregation is single-model on purpose: a run is one model's evaluation of one
dataset (a "category" in the wider harness). If results from more than one model
are present the script exits with an error rather than silently mixing them;
per-model aggregation is deferred until multi-model runs are supported.

Outputs two plain files in the output directory:
  - summary.json     dataset-level rollup (no per-task detail)
  - per_task.jsonl   one JSON object per task, one per line

The summary carries ``dataset`` and ``model`` so a future cross-category step
can combine several per-category summaries into an overall score.

When run inside GitHub Actions, a short markdown table is also appended to the
file named by GITHUB_STEP_SUMMARY.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PASS_THRESHOLD = 1.0


def iter_trial_results(root: Path):
    """Yield parsed per-trial result dicts found anywhere under root.

    Per-trial results carry a "task_name"; the job-level result.json does not,
    so it is skipped. Unreadable or non-object JSON is skipped with a warning.
    """
    for path in sorted(root.rglob("result.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {path}: {exc}", file=sys.stderr)
            continue
        if isinstance(data, dict) and data.get("task_name"):
            yield data


def trial_reward(result: dict):
    """Return the numeric verifier reward for a trial, or None if absent."""
    rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
    reward = rewards.get("reward")
    return reward if isinstance(reward, (int, float)) else None


def trial_errored(result: dict) -> bool:
    """True if the trial recorded an exception or produced no verifier reward."""
    if result.get("exception_info"):
        return True
    return trial_reward(result) is None


def trial_model(result: dict):
    """Return the model id recorded for a trial, or None."""
    agent = (result.get("config") or {}).get("agent") or {}
    return agent.get("model_name")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate for a task with n trials and c passes.

    Probability that at least one of k trials drawn without replacement from the
    n available is a pass. Requires 1 <= k <= n. At k == n this reduces to the
    empirical "did the task pass at least once" (1.0 if c >= 1 else 0.0).
    """
    if not 1 <= k <= n:
        raise ValueError(f"pass_at_k requires 1 <= k <= n (got k={k}, n={n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate(root: Path):
    """Walk the tree once and tally trials per task.

    Returns (models, job_ids, by_task) where by_task maps task name to
    {"trials", "passed", "errored"} counts.
    """
    models: set[str] = set()
    job_ids: set[str] = set()
    by_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trials": 0, "passed": 0, "errored": 0}
    )

    for result in iter_trial_results(root):
        task = result["task_name"]
        model = trial_model(result)
        if model:
            models.add(model)
        job_id = (result.get("config") or {}).get("job_id")
        if job_id:
            job_ids.add(job_id)

        stats = by_task[task]
        stats["trials"] += 1
        if trial_errored(result):
            stats["errored"] += 1
        reward = trial_reward(result)
        if reward is not None and reward >= PASS_THRESHOLD:
            stats["passed"] += 1

    return models, job_ids, dict(by_task)


def build_summary(by_task: dict, rollouts: int):
    """Compute per-task rows plus dataset-level pass@K and avg@K, where K = rollouts.

    Only the single k = rollouts is reported (no k < K sweep):
      - pass@K: fraction of tasks that passed at least once within K rollouts.
        Per task this is pass_at_k at k=min(rollouts, trials); with the usual
        trials == rollouts it is 1.0 iff the task passed at least once.
      - avg@K: passing rollouts / total rollouts across the whole run.
    """
    per_task = []
    passk_sum = 0.0
    passk_count = 0
    total_trials = total_passed = total_errored = 0

    for task in sorted(by_task):
        n = by_task[task]["trials"]
        c = by_task[task]["passed"]
        errored = by_task[task]["errored"]
        total_trials += n
        total_passed += c
        total_errored += errored

        # Cap k to the trials available so a task with fewer than K trials still
        # scores; normally n == rollouts so this is just pass@K.
        task_passk = round(pass_at_k(n, c, min(rollouts, n)), 6) if n else None
        if task_passk is not None:
            passk_sum += task_passk
            passk_count += 1
        per_task.append(
            {
                "task": task,
                "trials": n,
                "passed": c,
                "errored": errored,
                "pass_at_k": task_passk,
            }
        )

    dataset_passk = round(passk_sum / passk_count, 6) if passk_count else None
    # avg@K: passing rollouts / total rollouts run (== passing / (tasks * rollouts)).
    avg_at_k = round(total_passed / total_trials, 6) if total_trials else 0.0
    totals = {
        "tasks": len(by_task),
        "trials": total_trials,
        "passed": total_passed,
        "errored": total_errored,
    }
    return dataset_passk, avg_at_k, totals, per_task


def render_step_summary(summary: dict) -> str:
    """Render a compact markdown table for the GitHub run summary page."""
    lines = [
        "## Harbor results",
        "",
        f"- Dataset: {summary.get('dataset') or 'n/a'}",
        f"- Model: {summary.get('model') or 'n/a'}",
        f"- Rollouts per task: {summary.get('rollouts_per_task')}",
        f"- Shards: {summary.get('shards_found')}",
    ]
    totals = summary["totals"]
    lines.append(
        f"- Tasks: {totals['tasks']} | Trials: {totals['trials']} | "
        f"Passed: {totals['passed']} | Errored: {totals['errored']}"
    )
    k = summary.get("rollouts_per_task")
    passk = summary["pass_at_k"]
    lines.extend(["", "| metric | value |", "|---|---|"])
    lines.append(f"| pass@{k} | {passk:.3f} |" if passk is not None else f"| pass@{k} | n/a |")
    lines.append(f"| avg@{k} | {summary['avg_at_k']:.3f} |")
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict, per_task: list, out_dir: Path) -> None:
    """Write summary.json and per_task.jsonl, and the GitHub step summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "per_task.jsonl").open("w") as handle:
        for row in per_task:
            handle.write(json.dumps(row) + "\n")

    markdown = render_step_summary(summary)
    print(markdown)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write(markdown)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory of merged shard results.")
    parser.add_argument(
        "--rollouts",
        type=int,
        required=True,
        help="Rollouts per task (K); reports pass@1 through pass@K.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write summary.json and per_task.jsonl (default: root).",
    )
    parser.add_argument("--dataset", default=None, help="Dataset ref, recorded in the summary.")
    args = parser.parse_args(argv)

    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")

    out_dir = args.out_dir or args.root
    models, job_ids, by_task = aggregate(args.root)

    if not by_task:
        summary = {
            "dataset": args.dataset,
            "model": None,
            "rollouts_per_task": args.rollouts,
            "shards_found": len(job_ids),
            "totals": {"tasks": 0, "trials": 0, "passed": 0, "errored": 0},
            "pass_at_k": None,
            "avg_at_k": 0.0,
        }
        write_outputs(summary, [], out_dir)
        print("No trial results found; wrote empty summary.", file=sys.stderr)
        return 0

    if len(models) > 1:
        sys.exit(
            "error: results contain multiple models "
            f"({sorted(models)}); per-model aggregation is not implemented. "
            "Aggregate one model at a time."
        )

    dataset_passk, avg_at_k, totals, per_task = build_summary(by_task, args.rollouts)
    summary = {
        "dataset": args.dataset,
        "model": next(iter(models)) if models else None,
        "rollouts_per_task": args.rollouts,
        "shards_found": len(job_ids),
        "totals": totals,
        "pass_at_k": dataset_passk,
        "avg_at_k": avg_at_k,
    }
    write_outputs(summary, per_task, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
