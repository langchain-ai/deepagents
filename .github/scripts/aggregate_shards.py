#!/usr/bin/env python3
"""Aggregate Harbor shard results into dataset-level pass@K and avg@K metrics.

Reads every per-trial ``result.json`` under a directory tree (the merged output
of all shard artifacts), groups trials by task, and computes:

  - pass@K  the fraction of tasks that passed at least once within K rollouts
            (K = rollouts_per_task; only this single k is reported).
  - avg@K   passing trials (capped at K per task, so duplicated rollouts cannot
            push a task above 1) over the EXPECTED trial count (tasks * rollouts),
            so missing rollouts of a present task count as failures.

The summary is flagged ``incomplete`` when a full run cannot be vouched for:
the matrix job did not fully succeed (``--harbor-result`` != "success"); a
present task ran a number of trials other than K (missing OR duplicated
rollouts); a ``result.json`` could not be read; a reward was present but
non-numeric; or (when ``--expected-shards`` is given) fewer shards completed than
expected. Legitimately-empty shards upload ``empty-shard-*`` markers and count as
completed rather than missing.

A trial is a pass when its verifier reward is >= PASS_THRESHOLD. ``errored`` is
an independent diagnostic tally: a Harbor result that records ``exception_info``
and a verifier-passing reward counts as both passed and errored. Missing or
non-numeric verifier rewards are not passes and are counted as errored.

Aggregation is single-model on purpose: a run is one model's evaluation of one
dataset (a "category" in the wider harness). If results from more than one model
are present the script exits with an error rather than silently mixing them;
per-model aggregation is deferred until multi-model runs are supported.

Completeness is measured relative to the tasks and empty-shard markers that
reported. A whole shard whose artifact never uploaded contributes neither, so
pass ``--expected-shards`` when the caller knows the authoritative shard count.
Successful empty shards remain distinguishable from missing artifacts because
the workflow uploads one ``empty-shard-*`` marker for each no-op slice.

Outputs two plain files in the output directory:
  - summary.json     dataset-level rollup (no per-task detail)
  - per_task.jsonl   one JSON object per task, one per line

The summary carries ``dataset`` and ``model`` so a future cross-category step
can combine several per-category summaries into an overall score.

When run inside GitHub Actions, a short markdown table is also appended to the
file named by GITHUB_STEP_SUMMARY. Workflow-command annotations
(``::warning::`` / ``::error::``) are printed to stdout, which is the only
stream GitHub reliably parses them from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

PASS_THRESHOLD = 1.0


class Aggregation(NamedTuple):
    """Result of one walk of the merged shard tree.

    ``skipped_files`` counts ``result.json`` files that could not be read/parsed
    or were not JSON objects; ``malformed_rewards`` counts trials whose reward
    was present but not numeric. Both are data-integrity signals that flag the
    run ``incomplete``.
    """

    models: set[str]
    job_ids: set[str]
    empty_shards: set[str]
    by_task: dict[str, dict[str, int]]
    skipped_files: int
    malformed_rewards: int


class SummaryParts(NamedTuple):
    """Computed metrics for a non-empty run (fields named to avoid transposition)."""

    pass_at_k: float | None
    avg_at_k: float | None
    totals: dict[str, int]
    per_task: list[dict]


def emit_annotation(msg: str) -> None:
    """Print a GitHub Actions workflow command to stdout.

    ``::warning::`` / ``::error::`` are only reliably parsed from stdout, so
    annotations must not be routed to stderr.
    """
    print(msg)


def load_result(path: Path) -> dict | None:
    """Parse one ``result.json``; return the dict, or None if unusable.

    Unreadable/undecodable files and valid JSON that is not an object are
    skipped with a ``::warning::`` so a dropped result leaves a visible trace
    (a lost trial silently deflates the scores). A dict lacking ``task_name``
    (the job-level summary) is a legitimate skip and is handled by the caller,
    not here.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        emit_annotation(f"::warning::could not read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        emit_annotation(f"::warning::ignoring non-object result.json at {path}")
        return None
    return data


def raw_reward(result: dict) -> object:
    """Return the reward value recorded for a trial verbatim (any type, or None)."""
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    return rewards.get("reward")


def trial_reward(result: dict) -> float | None:
    """Return the numeric verifier reward for a trial, or None if absent/non-numeric.

    Numeric strings (e.g. ``"1.0"``) are coerced, so a stringified reward is not
    silently treated as a failure. A reward that is present but not numeric
    (e.g. a dict, or an unparseable string) returns None; use
    ``reward_is_malformed`` to distinguish that from a genuinely absent reward.
    ``bool`` is tolerated (Python treats it as an ``int``), mapping True/False
    to 1.0/0.0.
    """
    reward = raw_reward(result)
    if isinstance(reward, bool):
        return float(reward)
    if isinstance(reward, (int, float)):
        return float(reward)
    if isinstance(reward, str):
        try:
            return float(reward)
        except ValueError:
            return None
    return None


def reward_is_malformed(result: dict) -> bool:
    """True if a reward value is present but could not be read as a number."""
    return raw_reward(result) is not None and trial_reward(result) is None


def trial_errored(result: dict) -> bool:
    """True if the trial recorded an exception or produced no numeric reward."""
    if result.get("exception_info"):
        return True
    return trial_reward(result) is None


def trial_model(result: dict) -> str | None:
    """Return the model id recorded for a trial, or None."""
    agent = (result.get("config") or {}).get("agent") or {}
    return agent.get("model_name")


def aggregate(root: Path) -> Aggregation:
    """Walk the tree once and tally trials per task.

    Per-trial results carry a ``task_name``; the job-level ``result.json`` does
    not, so it is skipped silently (it is expected, not a loss). Files that
    cannot be read/parsed are counted in ``skipped_files``. ``passed`` follows
    the verifier reward, while ``errored`` separately tracks exception or
    missing-reward diagnostics.
    """
    models: set[str] = set()
    job_ids: set[str] = set()
    empty_shards = {path.name for path in root.rglob("empty-shard-*") if path.is_file()}
    by_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trials": 0, "passed": 0, "errored": 0}
    )
    skipped_files = 0
    malformed_rewards = 0

    for path in sorted(root.rglob("result.json")):
        result = load_result(path)
        if result is None:
            skipped_files += 1
            continue
        task = result.get("task_name")
        if not task:
            continue  # job-level summary, not a trial

        model = trial_model(result)
        if model:
            models.add(model)
        job_id = (result.get("config") or {}).get("job_id")
        if job_id:
            job_ids.add(job_id)
        if reward_is_malformed(result):
            malformed_rewards += 1
            emit_annotation(
                f"::warning::non-numeric reward {raw_reward(result)!r} in {path}; "
                "counting the trial as errored"
            )

        stats = by_task[task]
        stats["trials"] += 1
        if trial_errored(result):
            stats["errored"] += 1
        reward = trial_reward(result)
        if reward is not None and reward >= PASS_THRESHOLD:
            stats["passed"] += 1

    return Aggregation(
        models,
        job_ids,
        empty_shards,
        dict(by_task),
        skipped_files,
        malformed_rewards,
    )


def build_summary(by_task: dict[str, dict[str, int]], rollouts: int) -> SummaryParts:
    """Compute per-task rows plus dataset-level pass@K and avg@K, where K = rollouts.

    Missing rollouts of a present task are treated as failures, so an incomplete
    shard cannot inflate the scores:
      - pass@K: fraction of present tasks with at least one observed passing trial
        (a task that ran fewer than K rollouts still passes iff it passed once).
      - avg@K: passing trials, capped at K per task, divided by
        (present tasks * rollouts). The denominator is the EXPECTED trial count,
        and duplicated rollouts cannot inflate the score above 1.
    """
    per_task: list[dict] = []
    passk_sum = 0.0
    total_trials = total_passed = total_errored = 0
    capped_passed = 0

    for task in sorted(by_task):
        n = by_task[task]["trials"]
        c = by_task[task]["passed"]
        errored = by_task[task]["errored"]
        total_trials += n
        total_passed += c
        capped_passed += min(c, rollouts)
        total_errored += errored

        # A task passes @K iff it has >=1 observed pass; missing rollouts can only
        # be failures, so no unbiased estimator is needed.
        task_passk = 1.0 if c >= 1 else 0.0
        passk_sum += task_passk
        per_task.append(
            {
                "task": task,
                "trials": n,
                "passed": c,
                "errored": errored,
                f"pass@{rollouts}": task_passk,
            }
        )

    n_tasks = len(by_task)
    dataset_passk = round(passk_sum / n_tasks, 6) if n_tasks else None
    # Missing rollouts count as failures through the expected denominator, while
    # per-task capping prevents duplicated rollouts from inflating the numerator.
    expected_trials = n_tasks * rollouts
    avg_at_k = round(capped_passed / expected_trials, 6) if expected_trials else None
    totals = {
        "tasks": n_tasks,
        "trials": total_trials,
        "expected_trials": expected_trials,
        "passed": total_passed,
        "errored": total_errored,
    }
    return SummaryParts(dataset_passk, avg_at_k, totals, per_task)


def make_summary(
    *,
    dataset: str | None,
    model: str | None,
    category: str | None,
    rollouts: int,
    shards_found: int,
    expected_shards: int | None,
    skipped_files: int,
    harbor_result: str | None,
    incomplete: bool,
    totals: dict[str, int],
    pass_at_k: float | None,
    avg_at_k: float | None,
) -> dict:
    """Assemble the summary dict in one place, so the empty and populated paths
    cannot drift in schema. The metric keys are dynamic (``pass@{K}`` /
    ``avg@{K}``); ``rollouts_per_task`` carries K so a reader can reconstruct them.
    """
    return {
        "dataset": dataset,
        "model": model,
        "category": category,
        "rollouts_per_task": rollouts,
        "shards_found": shards_found,
        "expected_shards": expected_shards,
        "skipped_files": skipped_files,
        "harbor_result": harbor_result,
        "incomplete": incomplete,
        "totals": totals,
        f"pass@{rollouts}": pass_at_k,
        f"avg@{rollouts}": avg_at_k,
    }


def render_step_summary(summary: dict) -> str:
    """Render a compact markdown table for the GitHub run summary page."""
    lines = [
        "## Harbor results",
        "",
        f"- Dataset: {summary.get('dataset') or 'n/a'}",
        f"- Model: {summary.get('model') or 'n/a'}",
        f"- Rollouts per task: {summary.get('rollouts_per_task')}",
        f"- Shards completed: {summary.get('shards_found')}"
        + (
            f" / {summary['expected_shards']} expected"
            if summary.get("expected_shards")
            else ""
        ),
    ]
    totals = summary["totals"]
    lines.append(
        f"- Tasks: {totals['tasks']} | Trials: {totals['trials']}"
        + (
            f" / {totals['expected_trials']} expected"
            if totals.get("expected_trials")
            else ""
        )
        + f" | Passed: {totals['passed']} | Errored: {totals['errored']}"
    )
    if summary.get("skipped_files"):
        lines.append(f"- ⚠️ Unreadable result files skipped: {summary['skipped_files']}")
    if summary.get("incomplete"):
        lines.append(
            "- ⚠️ **Incomplete run** — some shards/rollouts are missing or unreadable; "
            "missing rollouts are counted as failures."
        )
    k = summary.get("rollouts_per_task")
    passk = summary.get(f"pass@{k}")
    avgk = summary.get(f"avg@{k}")
    lines.extend(["", "| metric | value |", "|---|---|"])
    lines.append(
        f"| pass@{k} | {passk:.3f} |" if passk is not None else f"| pass@{k} | n/a |"
    )
    lines.append(
        f"| avg@{k} | {avgk:.3f} |" if avgk is not None else f"| avg@{k} | n/a |"
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict, per_task: list[dict], out_dir: Path) -> None:
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


def _incomplete_reason(
    *,
    shard_failure: bool,
    shard_shortfall: bool,
    count_mismatch: bool,
    skipped_files: int,
    malformed_rewards: int,
    totals: dict[str, int],
    shards_found: int,
    expected_shards: int | None,
) -> str:
    """Human-readable summary of why a run was flagged incomplete (for the annotation)."""
    reasons = []
    if shard_failure:
        reasons.append("a shard job did not succeed")
    if shard_shortfall:
        reasons.append(f"only {shards_found}/{expected_shards} shards reported")
    if count_mismatch:
        reasons.append(
            "per-task rollout counts differ from K "
            f"(trials {totals['trials']}/{totals['expected_trials']})"
        )
    if skipped_files:
        reasons.append(f"{skipped_files} unreadable result file(s)")
    if malformed_rewards:
        reasons.append(f"{malformed_rewards} non-numeric reward(s)")
    return "; ".join(reasons) or "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory of merged shard results.")
    parser.add_argument(
        "--rollouts",
        type=int,
        required=True,
        help="Rollouts per task (K); the reported metrics are pass@K and avg@K.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write summary.json and per_task.jsonl (default: root).",
    )
    parser.add_argument(
        "--dataset", default=None, help="Dataset ref, recorded in the summary."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model spec, recorded authoritatively in the summary. Overrides the "
            "value detected from results, which is null when every trial errored."
        ),
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Eval category (autonomous|conversation|context), recorded in the summary.",
    )
    parser.add_argument(
        "--harbor-result",
        default=None,
        help=(
            "The matrix job result (GitHub needs.harbor.result). Anything other "
            "than 'success' -- excluding an unset/empty value, which is treated as "
            "success -- means a shard failed, so the run is flagged incomplete. "
            "Legitimately-empty shards (from task filtering) still count as success."
        ),
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=None,
        help=(
            "Authoritative shard count. When set, the run is flagged incomplete "
            "if fewer shards reported results or successful empty-shard markers."
        ),
    )
    args = parser.parse_args(argv)

    if args.rollouts < 1:
        parser.error("--rollouts must be >= 1")
    if args.expected_shards is not None and args.expected_shards < 1:
        parser.error("--expected-shards must be >= 1")

    out_dir = args.out_dir or args.root
    agg = aggregate(args.root)
    shards_found = len(agg.job_ids) + len(agg.empty_shards)
    # A shard actually failed only if the matrix job did not fully succeed. Empty
    # shards (filtered-out task slices) no-op successfully, so they are NOT losses.
    shard_failure = args.harbor_result not in (None, "", "success")
    # Fewer shards than the caller declared (only checkable when it passes a count).
    shard_shortfall = (
        args.expected_shards is not None and shards_found < args.expected_shards
    )
    # Files we found but couldn't trust: a lost result deflates the scores.
    data_loss = agg.skipped_files > 0 or agg.malformed_rewards > 0

    if not agg.by_task:
        incomplete = shard_failure or shard_shortfall or data_loss
        summary = make_summary(
            dataset=args.dataset,
            model=args.model,
            category=args.category,
            rollouts=args.rollouts,
            shards_found=shards_found,
            expected_shards=args.expected_shards,
            skipped_files=agg.skipped_files,
            harbor_result=args.harbor_result,
            incomplete=incomplete,
            totals={
                "tasks": 0,
                "trials": 0,
                "expected_trials": 0,
                "passed": 0,
                "errored": 0,
            },
            pass_at_k=None,
            avg_at_k=None,
        )
        write_outputs(summary, [], out_dir)
        if incomplete:
            # Symmetric with the populated path: a no-data run that *should* have
            # produced data is surfaced, not silently green.
            emit_annotation(
                "::warning::No trial results found for a run expected to produce them ("
                + _incomplete_reason(
                    shard_failure=shard_failure,
                    shard_shortfall=shard_shortfall,
                    count_mismatch=False,
                    skipped_files=agg.skipped_files,
                    malformed_rewards=agg.malformed_rewards,
                    totals=summary["totals"],
                    shards_found=shards_found,
                    expected_shards=args.expected_shards,
                )
                + ")."
            )
        print("No trial results found; wrote empty summary.", file=sys.stderr)
        return 0

    if len(agg.models) > 1:
        sys.exit(
            "error: results contain multiple models "
            f"({sorted(agg.models)}); per-model aggregation is not implemented. "
            "Aggregate one model at a time."
        )

    parts = build_summary(agg.by_task, args.rollouts)
    # Incomplete if a shard job failed, a shard is missing, a present task ran a
    # number of trials other than K (missing OR duplicated rollouts), or a result
    # was unreadable / had a non-numeric reward.
    count_mismatch = any(
        stats["trials"] != args.rollouts for stats in agg.by_task.values()
    )
    incomplete = shard_failure or shard_shortfall or data_loss or count_mismatch
    summary = make_summary(
        dataset=args.dataset,
        model=args.model or (next(iter(agg.models)) if agg.models else None),
        category=args.category,
        rollouts=args.rollouts,
        shards_found=shards_found,
        expected_shards=args.expected_shards,
        skipped_files=agg.skipped_files,
        harbor_result=args.harbor_result,
        incomplete=incomplete,
        totals=parts.totals,
        pass_at_k=parts.pass_at_k,
        avg_at_k=parts.avg_at_k,
    )
    if incomplete:
        emit_annotation(
            "::warning::Aggregated an incomplete run ("
            + _incomplete_reason(
                shard_failure=shard_failure,
                shard_shortfall=shard_shortfall,
                count_mismatch=count_mismatch,
                skipped_files=agg.skipped_files,
                malformed_rewards=agg.malformed_rewards,
                totals=parts.totals,
                shards_found=shards_found,
                expected_shards=args.expected_shards,
            )
            + "); missing rollouts counted as failures."
        )
    write_outputs(summary, parts.per_task, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
