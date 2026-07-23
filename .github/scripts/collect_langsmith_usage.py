#!/usr/bin/env python3
"""Collect deterministic token and cost metrics for Unified Eval leaves.

Discovers the LangSmith experiment (project) name recorded in each leaf's
``summary.json`` (written by ``aggregate_shards.py``), queries LangSmith for the
root Harbor rollout traces, and aggregates per experiment: total input/output
tokens and total cost (USD). Two totals are reported per experiment -- one over
every rollout ("true spend") and one restricted to rollouts that reached a
terminal result (no traced ``error``) -- so a leaf that erred out is comparable
to a clean one. Token/cost data lives only in LangSmith, so a missing
``LANGSMITH_API_KEY`` yields a stable "unavailable" shape rather than an error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Iterable
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


def _load_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, object], value)


def expected_rollouts(summary: dict[str, object]) -> int | None:
    """Return the authoritative expected trace count for a leaf summary."""
    expected_shards = summary.get("expected_shards")
    rollouts = summary.get("rollouts_per_task")
    if (
        isinstance(expected_shards, int)
        and not isinstance(expected_shards, bool)
        and expected_shards >= 0
        and isinstance(rollouts, int)
        and not isinstance(rollouts, bool)
        and rollouts > 0
    ):
        return expected_shards * rollouts
    totals = summary.get("totals")
    if isinstance(totals, dict):
        expected = cast(dict[str, object], totals).get("expected_trials")
        if (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected >= 0
        ):
            return expected
    return None


def discover_experiments(root: Path) -> dict[str, int | None]:
    """Discover experiment names + expected rollout counts from leaf summaries.

    Walks every ``summary.json`` under ``root`` (the downloaded ``harbor-*``
    leaf artifacts), reads its ``langsmith_experiment`` name and expected
    rollout count, dedupes shared experiments, and errors on conflicting
    expected counts for the same experiment.
    """
    experiments: dict[str, int | None] = {}
    for summary_path in sorted(root.rglob("summary.json")):
        summary = _load_object(summary_path, "leaf summary")
        experiment = summary.get("langsmith_experiment")
        if not isinstance(experiment, str) or not experiment:
            continue
        expected = expected_rollouts(summary)
        previous = experiments.get(experiment)
        if experiment in experiments and previous != expected:
            raise ValueError(
                f"conflicting expected rollout counts for {experiment!r}: "
                f"{previous!r} and {expected!r}"
            )
        experiments[experiment] = expected
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


def _totals_block(
    *,
    token_covered: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    priced: int,
    total_cost: Decimal,
) -> dict[str, object]:
    """Build a token/cost totals block, nulling metrics with no coverage."""
    return {
        "prompt_tokens": prompt_tokens if token_covered else None,
        "completion_tokens": completion_tokens if token_covered else None,
        "total_tokens": total_tokens if token_covered else None,
        "cost_usd": _number(total_cost) if priced else None,
    }


def summarize_runs(
    runs: Iterable[RunLike], *, expected_rollouts: int | None
) -> dict[str, object]:
    """Aggregate root Harbor rollout traces without inspecting child runs.

    Accumulates two parallel totals: one over all rollouts ("true spend") and
    one over rollouts that reached a terminal result (``error`` unset), so a
    leaf's cost can be compared without being skewed by its failure rate.
    """
    observed = token_covered = priced = completed = errored = 0
    completed_token_covered = completed_priced = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    completed_prompt = completed_completion = completed_total = 0
    total_cost = Decimal(0)
    completed_cost = Decimal(0)
    for run in runs:
        if "harbor-trial" not in (run.tags or []):
            continue
        observed += 1
        is_errored = bool(run.error)
        if is_errored:
            errored += 1
        else:
            completed += 1
        prompt = _token(run.prompt_tokens)
        completion = _token(run.completion_tokens)
        total = _token(run.total_tokens)
        has_tokens = prompt is not None and completion is not None and total is not None
        if has_tokens:
            token_covered += 1
            prompt_tokens += prompt
            completion_tokens += completion
            total_tokens += total
            if not is_errored:
                completed_token_covered += 1
                completed_prompt += prompt
                completed_completion += completion
                completed_total += total
        cost = _cost(run.total_cost)
        if cost is not None:
            priced += 1
            total_cost += cost
            if not is_errored:
                completed_priced += 1
                completed_cost += cost

    status = "complete"
    if expected_rollouts is not None and any(
        count < expected_rollouts for count in (observed, token_covered, priced)
    ):
        status = "partial"
    return {
        "status": status,
        "coverage": {
            "expected_rollouts": expected_rollouts,
            "observed_rollouts": observed,
            "token_rollouts": token_covered,
            "priced_rollouts": priced,
            "completed_rollouts": completed,
            "errored_rollouts": errored,
        },
        "totals": _totals_block(
            token_covered=token_covered,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            priced=priced,
            total_cost=total_cost,
        ),
        "completed_totals": _totals_block(
            token_covered=completed_token_covered,
            prompt_tokens=completed_prompt,
            completion_tokens=completed_completion,
            total_tokens=completed_total,
            priced=completed_priced,
            total_cost=completed_cost,
        ),
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


def collect_experiment(
    client: ClientLike,
    experiment: str,
    expected_rollouts: int | None,
    *,
    attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    delays: tuple[float, ...] = RETRY_DELAYS,
) -> dict[str, object]:
    """Query one experiment with bounded retries for delayed trace ingestion."""
    best: dict[str, object] | None = None
    for attempt in range(attempts):
        try:
            runs = client.list_runs(
                project_name=experiment,
                is_root=True,
                select=SELECT_FIELDS,
            )
            current = summarize_runs(runs, expected_rollouts=expected_rollouts)
            if best is None or _coverage_rank(current) > _coverage_rank(best):
                best = current
            if _fully_covered(current):
                return current
        except Exception as exc:  # noqa: BLE001  # API clients expose several transport exceptions
            print(
                f"::warning::LangSmith usage query failed for {experiment!r} "
                f"(attempt {attempt + 1}/{attempts}): {type(exc).__name__}"
            )
        if attempt + 1 < attempts:
            sleep(delays[min(attempt, len(delays) - 1)])
    return best or unavailable_usage(expected_rollouts)


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
                        f"(attempt {attempt + 1}/{attempts}): {type(exc).__name__}"
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
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args(argv)
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")

    experiments = discover_experiments(args.root)
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
