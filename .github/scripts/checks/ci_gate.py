"""Evaluate the aggregate `ci_success` job result from per-job results.

Why this exists:
    `talon` editable-installs `deepagents-code`
    (`libs/talon/pyproject.toml`, `[tool.uv.sources]`), so the `talon` path
    filter in `ci.yml` watches `libs/code/**` and every libs/code PR runs the
    `lint-talon`/`test-talon` jobs. A libs/code change can therefore break
    talon CI, and failing `ci_success` used to force the author to fix the
    talon regression inside their libs/code PR — which also made
    release-please fan out changelog entries across both packages.

    The maintainers accept merging PRs that do not touch `libs/talon` with
    talon red at HEAD; the breakage is fixed in a follow-up `fix(talon): ...`
    PR. This script encodes exactly that waiver so the `ci_success` shell step
    stays readable and the decision matrix is unit-testable.

What it does:
    - On `pull_request` runs where the `talon` filter output is not `'true'`,
      `failure` results from `lint-talon`/`test-talon` are waived (reported as
      notices) instead of failing the gate. Every other job result is strict.
    - `cancelled` results are never waived — a cancelled talon job still
      blocks, matching the pre-existing rule that cancellations are not
      allowed on PR/merge_group runs.
    - On genuine talon PRs (`talon == 'true'`) and on `push`/`merge_group`
      runs, talon failures block exactly as before. Push runs stay strict so
      a broken talon stays visible on main, where every job runs
      unconditionally.

Inputs are a JSON object mapping job name to its result (the `skipped`
entries filtered out by the workflow step), the `talon` path-filter output,
and the event name. Output is a single JSON line consumed by the workflow
step. The exit code is always 0; the pass/fail decision lives only in the
JSON payload so a caller bug surfaces as an explicit failure, not a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TypedDict

# Jobs whose `failure` results may be waived on non-talon PRs.
WAIVABLE_JOBS = ("lint-talon", "test-talon")


class GateResult(TypedDict):
    """Structured gate report consumed by the `ci_success` workflow step."""

    ok: bool
    waived: list[str]
    failed: list[str]
    cancelled: list[str]


def evaluate_gate(results: dict[str, str], *, event: str, talon: str) -> GateResult:
    """Decide whether the aggregate CI gate passes.

    Args:
        results: Map of job name to job result (`success`/`failure`/
            `cancelled`); `skipped` jobs are omitted by the caller.
        event: `github.event_name` for the run.
        talon: The `changes` job's `talon` path-filter output.

    Returns:
        A `GateResult`. `ok` is True only when no blocking results remain
        after applying the talon waiver. `waived` names talon jobs whose
        failures were excused; `failed`/`cancelled` name blocking jobs.
    """
    waiver_applies = event == "pull_request" and talon != "true"
    waived: list[str] = []
    failed: list[str] = []
    cancelled: list[str] = []
    for job, result in results.items():
        if result == "failure":
            if waiver_applies and job in WAIVABLE_JOBS:
                waived.append(job)
            else:
                failed.append(job)
        elif result == "cancelled":
            cancelled.append(job)
    return GateResult(
        ok=not failed and not cancelled,
        waived=waived,
        failed=failed,
        cancelled=cancelled,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, evaluate the gate, and print one JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event",
        required=True,
        help="github.event_name (push, pull_request, merge_group, ...)",
    )
    parser.add_argument(
        "--talon",
        required=True,
        help="needs.changes.outputs.talon path-filter output",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="JSON object mapping job name to job result",
    )
    args = parser.parse_args(argv)
    results = json.loads(args.results)
    gate = evaluate_gate(results, event=args.event, talon=args.talon)
    json.dump(gate, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
