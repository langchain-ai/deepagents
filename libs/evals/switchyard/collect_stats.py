#!/usr/bin/env python3
# ruff: noqa: ANN401, PLC0415, T201  # Standalone CLI over dynamic JSON payloads.
"""Snapshot Switchyard's `/v1/stats` around an eval arm and price the result.

Switchyard reports calls, tokens, cache splits, and latency per model, but no
cost — there is no cost/price field anywhere in `crates/switchyard-server`
(the escalation doc's claim that the snapshot reports cost is a doc bug). So
cost is computed here from the token breakdown, reusing the repo's existing
pricing layer (`deepagents_code.cost_tracking`, genai-prices plus the
maintainer-curated bundled overrides) rather than a hand-rolled rate table.

The cache read/write split is why this has to come from Switchyard rather than
a naive input-token count: cache reads price at roughly a tenth of base input
and cache writes at roughly 1.25x, so a model that caches well looks far
cheaper than its raw input total suggests. Pricing those buckets separately is
the difference between a defensible cost comparison and a misleading one.

Usage, once per arm:

    python collect_stats.py reset
    ... run the eval arm ...
    python collect_stats.py snapshot runs/glm.json
    python collect_stats.py report runs/glm.json

`report` accepts several snapshots to print them side by side:

    python collect_stats.py report runs/glm.json runs/opus.json runs/esc.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Literal

DEFAULT_URL = "http://localhost:4000"

# genai-prices infers the provider from most model ids, but a self-hosted or
# marketplace id (a Baseten deployment slug, say) carries no provider hint.
# Map substrings to a LangChain provider key for those. `report --provider`
# appends to this at runtime.
PROVIDER_HINTS: dict[str, str] = {
    "claude": "anthropic",
    "gemini": "google_genai",
    "glm": "baseten",
}


def _http(url: str, method: str = "GET") -> Any:
    request = urllib.request.Request(url, method=method)  # noqa: S310  # operator-supplied localhost URL
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.loads(response.read())


def _provider_for(model: str, hints: dict[str, str]) -> str:
    lowered = model.lower()
    for needle, provider in hints.items():
        if needle in lowered:
            return provider
    return ""


def _tokens(stats: dict[str, Any], key: str) -> int:
    """Read a token count as an int.

    `cost_tracking._token_count` returns 0 for any non-int, so a float count is
    silently dropped and the estimate comes back too low rather than erroring.
    Switchyard's own `/v1/stats` emits u64s, but an averaged snapshot (mean over
    n rollouts) carries halves — round here so averaging a run set stays
    priceable instead of quietly costing nothing.
    """
    value = stats.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, round(value))


def _usage_metadata(
    stats: dict[str, Any],
    *,
    pricing: Literal["cache-aware", "uncached"] = "cache-aware",
) -> dict[str, Any]:
    """Map one Switchyard model-stats entry onto LangChain usage metadata.

    Switchyard's `prompt_tokens` is the inclusive input total (cache reads plus
    cache writes plus uncached), which is exactly what `estimate_cost` expects:
    it subtracts each detail bucket from the total before applying rates, so the
    buckets are not double-counted.
    """
    cache_read = _tokens(stats, "cached_tokens") if pricing == "cache-aware" else 0
    cache_creation = _tokens(stats, "cache_creation_tokens") if pricing == "cache-aware" else 0
    return {
        "input_tokens": _tokens(stats, "prompt_tokens"),
        "output_tokens": _tokens(stats, "completion_tokens"),
        "total_tokens": _tokens(stats, "total_tokens"),
        "input_token_details": {
            "cache_read": cache_read,
            "cache_creation": cache_creation,
        },
        "output_token_details": {"reasoning": _tokens(stats, "reasoning_tokens")},
    }


def _custom_cost(
    stats: dict[str, Any],
    model: str,
    rates: dict[str, tuple[float, float]],
) -> float | None:
    lowered = model.lower()
    for needle, (input_rate, output_rate) in rates.items():
        if needle in lowered:
            return (
                _tokens(stats, "prompt_tokens") * input_rate
                + _tokens(stats, "completion_tokens") * output_rate
            ) / 1_000_000
    return None


def _rows(
    snapshot: dict[str, Any],
    hints: dict[str, str],
    *,
    pricing: Literal["cache-aware", "uncached"] = "cache-aware",
    rates: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Flatten a snapshot into one row per model, judge calls included.

    Judge traffic lands in the `classifier` bucket rather than `models`, so it
    is pulled out separately and labelled — its tokens are a real cost of the
    routed arm and omitting them would overstate the saving.
    """
    from deepagents_code.cost_tracking import estimate_cost

    rows: list[dict[str, Any]] = []
    buckets: list[tuple[str, dict[str, Any]]] = [
        ("model", snapshot.get("models") or {}),
        ("judge", (snapshot.get("classifier") or {}).get("models") or {}),
    ]
    for role, models in buckets:
        for model, stats in sorted(models.items()):
            usage = _usage_metadata(stats, pricing=pricing)
            cost = _custom_cost(stats, model, rates or {})
            if cost is None:
                cost = estimate_cost(usage, model, _provider_for(model, hints))
            rows.append(
                {
                    "role": role,
                    "model": model,
                    "calls": stats.get("calls", 0),
                    "in": stats.get("prompt_tokens", 0),
                    "cache_r": stats.get("cached_tokens", 0),
                    "cache_w": stats.get("cache_creation_tokens", 0),
                    "out": stats.get("completion_tokens", 0),
                    "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
                    "p50_ms": (stats.get("model_call_latency") or {}).get("p50_ms", 0.0),
                    "cost_usd": cost,
                }
            )
    return rows


def _print_report(
    path: Path,
    hints: dict[str, str],
    *,
    pricing: Literal["cache-aware", "uncached"],
    rates: dict[str, tuple[float, float]],
) -> float | None:
    snapshot = json.loads(path.read_text())
    rows = _rows(snapshot, hints, pricing=pricing, rates=rates)
    print(f"\n=== {path.name} ({pricing}) ===")
    print(
        f"{'role':<6} {'model':<38} {'calls':>7} {'in':>12} {'cache R':>12} "
        f"{'cache W':>10} {'out':>10} {'hit%':>6} {'p50ms':>8} {'cost $':>10}"
    )
    total: float | None = 0.0
    for row in rows:
        cost = row["cost_usd"]
        if cost is None:
            # An unpriceable model must not silently read as free.
            total = None
            shown = "unpriced"
        else:
            shown = f"{cost:.4f}"
            if total is not None:
                total += cost
        print(
            f"{row['role']:<6} {row['model']:<38} {row['calls']:>7} {row['in']:>12,} "
            f"{row['cache_r']:>12,} {row['cache_w']:>10,} {row['out']:>10,} "
            f"{row['cache_hit_rate'] * 100:>5.1f}% {row['p50_ms']:>8.0f} {shown:>10}"
        )
    if total is None:
        print("  TOTAL: unavailable — at least one model has no price; pass --provider")
    else:
        print(f"  TOTAL: ${total:.4f}   ({snapshot.get('total_requests', 0)} requests)")
    return total


_SUM_FIELDS = (
    "calls",
    "errors",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)


def _snapshot_paths(sources: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for source in sources:
        if source.is_dir():
            paths.extend(source.rglob("switchyard-stats.json"))
        elif source.is_file():
            paths.append(source)
    return sorted(set(paths))


def _merge_models(target: dict[str, Any], source: dict[str, Any]) -> None:
    for model, stats in source.items():
        merged = target.setdefault(model, {})
        for field in _SUM_FIELDS:
            merged[field] = _tokens(merged, field) + _tokens(stats, field)


def _finalize_models(models: dict[str, Any]) -> None:
    for stats in models.values():
        calls = _tokens(stats, "calls")
        prompt = _tokens(stats, "prompt_tokens")
        stats["avg_prompt_tokens"] = prompt / calls if calls else 0.0
        stats["avg_completion_tokens"] = (
            _tokens(stats, "completion_tokens") / calls if calls else 0.0
        )
        stats["cache_hit_rate"] = _tokens(stats, "cached_tokens") / prompt if prompt else 0.0


def _aggregate_snapshots(paths: list[Path]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "total_requests": 0,
        "total_errors": 0,
        "models": {},
        "classifier": {"total_requests": 0, "models": {}},
        "snapshot_count": len(paths),
    }
    configs: set[str] = set()
    for path in paths:
        snapshot = json.loads(path.read_text())
        aggregate["total_requests"] += _tokens(snapshot, "total_requests")
        aggregate["total_errors"] += _tokens(snapshot, "total_errors")
        _merge_models(aggregate["models"], snapshot.get("models") or {})
        classifier = snapshot.get("classifier") or {}
        aggregate["classifier"]["total_requests"] += _tokens(classifier, "total_requests")
        _merge_models(
            aggregate["classifier"]["models"],
            classifier.get("models") or {},
        )
        config = snapshot.get("switchyard_config")
        if isinstance(config, str) and config:
            configs.add(config)
    _finalize_models(aggregate["models"])
    _finalize_models(aggregate["classifier"]["models"])
    aggregate["switchyard_configs"] = sorted(configs)
    return aggregate


def _validate_job(job_dir: Path) -> tuple[int, list[str]]:
    """Validate every trial and its Switchyard snapshot in one Harbor job.

    Args:
        job_dir: Harbor job directory containing one subdirectory per trial.

    Returns:
        The trial count and a list of validation failures.
    """
    if not job_dir.is_dir():
        return 0, [f"Harbor job directory does not exist: {job_dir}"]

    trials = sorted(path for path in job_dir.iterdir() if path.is_dir())
    if not trials:
        return 0, [f"No Harbor trial results found under {job_dir}"]

    failures: list[str] = []
    for trial in trials:
        result_path = trial / "result.json"
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            failures.append(f"{trial.name}: result.json is unreadable or invalid")
            continue
        if not isinstance(result, dict):
            failures.append(f"{trial.name}: result.json is not an object")
            continue
        if "exception_info" not in result or result["exception_info"] is not None:
            failures.append(f"{trial.name}: trial recorded an exception")
        if "verifier_result" not in result or result["verifier_result"] is None:
            failures.append(f"{trial.name}: verifier result is missing")

        stats_path = trial / "artifacts" / "switchyard-stats.json"
        try:
            stats = json.loads(stats_path.read_text())
        except FileNotFoundError:
            failures.append(f"{trial.name}: Switchyard stats are missing")
            continue
        except (OSError, json.JSONDecodeError):
            failures.append(f"{trial.name}: Switchyard stats are unreadable or invalid")
            continue
        if not isinstance(stats, dict):
            failures.append(f"{trial.name}: Switchyard stats are not an object")
            continue

        requests = stats.get("total_requests")
        errors = stats.get("total_errors")
        if type(requests) is not int or requests <= 0:
            failures.append(f"{trial.name}: Switchyard recorded no routed requests")
        if type(errors) is not int or errors != 0:
            failures.append(f"{trial.name}: Switchyard recorded upstream errors")

    return len(trials), failures


def main() -> int:
    """Run the stats reset, snapshot, aggregation, or reporting command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"Switchyard base URL (default {DEFAULT_URL})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("reset", help="Zero the counters. Run immediately before an arm.")

    snap = sub.add_parser("snapshot", help="Write the current /v1/stats to a file.")
    snap.add_argument("out", type=Path)

    validate = sub.add_parser("validate", help="Fail unless every routed Harbor trial succeeded.")
    validate.add_argument("job_dir", type=Path)

    aggregate = sub.add_parser("aggregate", help="Merge per-trial switchyard-stats.json artifacts.")
    aggregate.add_argument("sources", type=Path, nargs="+")
    aggregate.add_argument("--out", type=Path, required=True)

    report = sub.add_parser("report", help="Price one or more snapshots.")
    report.add_argument("snapshots", type=Path, nargs="+")
    report.add_argument(
        "--provider",
        action="append",
        default=[],
        metavar="SUBSTRING=PROVIDER",
        help="Force a provider for model ids containing SUBSTRING, e.g. --provider zai=fireworks",
    )
    report.add_argument(
        "--pricing",
        choices=("cache-aware", "uncached"),
        default="cache-aware",
        help="Use provider cache rates or price every input token at the base rate.",
    )
    report.add_argument(
        "--rate",
        action="append",
        default=[],
        metavar="SUBSTRING=INPUT,OUTPUT",
        help="Custom USD-per-million input/output rates for an uncatalogued model.",
    )

    args = parser.parse_args()

    if args.command == "reset":
        _http(f"{args.url}/v1/stats/reset", method="POST")
        print(f"reset {args.url}/v1/stats")
        return 0

    if args.command == "snapshot":
        snapshot = _http(f"{args.url}/v1/stats")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snapshot, indent=2))
        total = snapshot.get("total_requests", 0)
        print(f"wrote {args.out} ({total} requests)")
        if not total:
            print("  warning: zero requests recorded — did traffic actually route here?")
        return 0

    if args.command == "validate":
        trial_count, failures = _validate_job(args.job_dir)
        if failures:
            for failure in failures:
                print(f"error: {failure}", file=sys.stderr)
            return 1
        print(f"validated {trial_count} routed Harbor trial(s)")
        return 0

    if args.command == "aggregate":
        paths = _snapshot_paths(args.sources)
        if not paths:
            parser.error("aggregate found no switchyard-stats.json snapshots")
        merged = _aggregate_snapshots(paths)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(merged, indent=2))
        print(f"wrote {args.out} ({len(paths)} snapshots)")
        return 0

    hints = dict(PROVIDER_HINTS)
    for pair in args.provider:
        needle, _, provider = pair.partition("=")
        if not provider:
            parser.error(f"--provider expects SUBSTRING=PROVIDER, got {pair!r}")
        hints[needle.lower()] = provider
    rates: dict[str, tuple[float, float]] = {}
    for pair in args.rate:
        needle, separator, raw_rates = pair.partition("=")
        values = raw_rates.split(",") if separator else []
        try:
            input_rate, output_rate = (float(value) for value in values)
        except (TypeError, ValueError):
            parser.error(f"--rate expects SUBSTRING=INPUT,OUTPUT, got {pair!r}")
        if not needle or input_rate < 0 or output_rate < 0:
            parser.error(f"--rate values must be non-negative, got {pair!r}")
        rates[needle.lower()] = (input_rate, output_rate)
    for path in args.snapshots:
        _print_report(path, hints, pricing=args.pricing, rates=rates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
