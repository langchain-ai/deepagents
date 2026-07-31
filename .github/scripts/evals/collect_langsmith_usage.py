#!/usr/bin/env python3
"""Collect deterministic token and cost metrics for Unified Eval experiments.

Takes the ``{experiment_name: expected_trials}`` map that prep computed up front
(``--experiments-json``), queries LangSmith for each experiment's root Harbor
rollout traces, and aggregates per experiment: total input/output tokens and
total cost (USD). Two totals are reported per experiment -- one over every
rollout ("true spend") and one restricted to rollouts that reached a terminal
result (no traced ``error``) -- so a leaf that erred out is comparable to a
clean one. Token/cost data lives only in LangSmith, so a missing
``LANGSMITH_API_KEY`` yields a stable "unavailable" shape rather than an error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol, cast


class RunLike(Protocol):
    """LangSmith root-run fields used by the collector."""

    tags: list[str] | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    total_cost: Decimal | None
    error: str | None


class ClientLike(Protocol):
    """Narrow ``langsmith.Client`` interface used by the collector."""

    def list_runs(
        self,
        *,
        project_name: str,
        is_root: bool,
        select: list[str],
    ) -> Iterable[RunLike]:
        """List runs from one LangSmith project."""


SELECT_FIELDS = [
    "id",
    "tags",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "total_cost",
    "error",
]
RETRY_DELAYS = (5.0, 10.0, 20.0, 30.0)


def load_experiments(path: Path) -> dict[str, int | None]:
    """Read the ``{experiment_name: expected_trials}`` map produced by prep.

    prep computes the experiment names up front (via ``experiment_name.py``) so
    the collector never has to scan shard artifacts to learn them. ``expected``
    is the per-experiment trace count (tasks * rollouts) used to tell "fully
    ingested" from "still catching up"; ``null`` (or a non-int) means prep could
    not determine it, so coverage is best-effort. Type-guarded — a malformed
    entry is dropped rather than aborting the whole collection.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"experiments file must be a JSON object: {path}"
        raise ValueError(msg)
    experiments: dict[str, int | None] = {}
    for name, expected in cast(dict[str, object], raw).items():
        if not isinstance(name, str) or not name:
            continue
        experiments[name] = (
            expected
            if isinstance(expected, int) and not isinstance(expected, bool) and expected >= 0
            else None
        )
    return experiments


def _token(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _cost(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return cost if cost.is_finite() and cost >= 0 else None


def _number(value: Decimal) -> float:
    """Convert an exact internal cost to a finite JSON number."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("cost is outside the finite JSON number range")
    return number


@dataclass
class _Totals:
    """Running token/cost tally over one subset of rollouts."""

    token_rollouts: int = 0
    priced_rollouts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: Decimal = Decimal(0)

    def add_tokens(self, prompt: int, completion: int, total: int) -> None:
        """Fold one rollout's token counts into the tally."""
        self.token_rollouts += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def add_cost(self, cost: Decimal) -> None:
        """Fold one rollout's cost into the tally."""
        self.priced_rollouts += 1
        self.cost += cost


def _totals_block(tally: _Totals) -> dict[str, object]:
    """Build a token/cost totals block, nulling metrics with no coverage."""
    return {
        "prompt_tokens": tally.prompt_tokens if tally.token_rollouts else None,
        "completion_tokens": tally.completion_tokens if tally.token_rollouts else None,
        "total_tokens": tally.total_tokens if tally.token_rollouts else None,
        "cost_usd": _number(tally.cost) if tally.priced_rollouts else None,
    }


def summarize_runs(
    runs: Iterable[RunLike], *, expected_rollouts: int | None
) -> dict[str, object]:
    """Aggregate root Harbor rollout traces without inspecting child runs.

    Accumulates two parallel totals: one over all rollouts ("true spend") and
    one over rollouts that reached a terminal result (``error`` unset), so a
    leaf's cost can be compared without being skewed by its failure rate.
    """
    counts = {"observed": 0, "completed": 0, "errored": 0}
    overall = _Totals()
    succeeded = _Totals()
    for run in runs:
        if "harbor-trial" not in (run.tags or []):
            continue
        counts["observed"] += 1
        is_errored = bool(run.error)
        counts["errored" if is_errored else "completed"] += 1
        # An errored rollout counts toward true spend but not completed-only totals.
        targets = (overall,) if is_errored else (overall, succeeded)

        prompt = _token(run.prompt_tokens)
        completion = _token(run.completion_tokens)
        total = _token(run.total_tokens)
        if prompt is not None and completion is not None and total is not None:
            for tally in targets:
                tally.add_tokens(prompt, completion, total)

        cost = _cost(run.total_cost)
        if cost is not None:
            for tally in targets:
                tally.add_cost(cost)

    status = "complete"
    if expected_rollouts is not None and any(
        count < expected_rollouts
        for count in (counts["observed"], overall.token_rollouts, overall.priced_rollouts)
    ):
        status = "partial"
    return {
        "status": status,
        "coverage": {
            "expected_rollouts": expected_rollouts,
            "observed_rollouts": counts["observed"],
            "token_rollouts": overall.token_rollouts,
            "priced_rollouts": overall.priced_rollouts,
            "completed_rollouts": counts["completed"],
            "errored_rollouts": counts["errored"],
        },
        "totals": _totals_block(overall),
        "completed_totals": _totals_block(succeeded),
    }


def unavailable_usage(expected_rollouts: int | None) -> dict[str, object]:
    """Return the stable empty shape used when LangSmith cannot be queried."""
    empty_totals: dict[str, object] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cost_usd": None,
    }
    return {
        "status": "unavailable",
        "coverage": {
            "expected_rollouts": expected_rollouts,
            "observed_rollouts": 0,
            "token_rollouts": 0,
            "priced_rollouts": 0,
            "completed_rollouts": 0,
            "errored_rollouts": 0,
        },
        "totals": dict(empty_totals),
        "completed_totals": dict(empty_totals),
    }


def _coverage_rank(usage: dict[str, object]) -> tuple[int, int, int]:
    coverage = cast(dict[str, int | None], usage["coverage"])
    return (
        cast(int, coverage["observed_rollouts"]),
        cast(int, coverage["token_rollouts"]),
        cast(int, coverage["priced_rollouts"]),
    )


def _fully_covered(usage: dict[str, object]) -> bool:
    coverage = cast(dict[str, int | None], usage["coverage"])
    expected = coverage["expected_rollouts"]
    if expected is None:
        return usage["status"] == "complete"
    return all(
        cast(int, coverage[field]) >= expected
        for field in ("observed_rollouts", "token_rollouts", "priced_rollouts")
    )


def _query_once(
    client: ClientLike, experiment: str, expected_rollouts: int | None
) -> dict[str, object]:
    runs = client.list_runs(
        project_name=experiment,
        is_root=True,
        select=SELECT_FIELDS,
    )
    return summarize_runs(runs, expected_rollouts=expected_rollouts)


def collect_all(
    experiments: dict[str, int | None],
    client: ClientLike | None,
    *,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    delays: tuple[float, ...] = RETRY_DELAYS,
) -> dict[str, object]:
    """Collect every experiment in shared retry rounds to bound total delay."""
    output = {
        experiment: unavailable_usage(expected)
        for experiment, expected in sorted(experiments.items())
    }
    if client is not None:
        pending = set(experiments)
        for attempt in range(attempts):
            for experiment in sorted(pending):
                try:
                    current = _query_once(client, experiment, experiments[experiment])
                except Exception as exc:  # noqa: BLE001  # API clients expose several transport exceptions
                    print(
                        f"::warning::LangSmith usage query failed for {experiment!r} "
                        f"(attempt {attempt + 1}/{attempts}): {type(exc).__name__}: {exc}"
                    )
                    continue
                if output[experiment]["status"] == "unavailable" or _coverage_rank(
                    current
                ) > _coverage_rank(output[experiment]):
                    output[experiment] = current
                if _fully_covered(current):
                    pending.remove(experiment)
            if not pending:
                break
            if attempt + 1 < attempts:
                sleep(delays[min(attempt, len(delays) - 1)])

    for experiment, usage in output.items():
        if usage["status"] != "complete" or not _fully_covered(usage):
            coverage = cast(dict[str, int | None], usage["coverage"])
            print(
                f"::warning::Incomplete LangSmith usage for {experiment!r}: "
                f"observed={coverage['observed_rollouts']}, "
                f"tokens={coverage['token_rollouts']}, "
                f"priced={coverage['priced_rollouts']}, "
                f"expected={coverage['expected_rollouts']}"
            )
    return {"schema_version": 1, "experiments": output}


def main(argv: list[str] | None = None) -> int:
    """CLI for the Unified Eval usage job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-json",
        type=Path,
        required=True,
        help="JSON map {experiment_name: expected_trials|null} produced by prep.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")

    experiments = load_experiments(args.experiments_json)
    client: ClientLike | None = None
    if experiments and os.environ.get("LANGSMITH_API_KEY"):
        from langsmith import Client

        client = Client()
    elif experiments:
        print("::warning::LANGSMITH_API_KEY is unavailable; usage analysis skipped")
    result = collect_all(experiments, client, attempts=args.attempts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
