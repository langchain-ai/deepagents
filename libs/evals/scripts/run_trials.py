"""Run the eval suite N times for the same model/config and aggregate stats.

Each trial invokes `pytest tests/evals` with the same flags as `make evals`,
writes its own per-trial report to `--out-dir`, and creates its own LangSmith
experiment. After all trials complete, per-metric mean / median / stdev / min /
max are computed across trials and written to `<out-dir>/trials_summary.json`.

Usage:
    python scripts/run_trials.py --model openai:gpt-5.5 --trials 5
    python scripts/run_trials.py --model openai:gpt-5.5 --trials 3 \
        --eval-category memory --openai-reasoning-effort medium

Within a single CLI invocation, trials run sequentially in-process — LangSmith
experiment creation and provider rate-limits make in-process parallelism
unsafe. CI achieves parallelism by running each trial as a separate GHA job
and then calling this script with `--aggregate-only` to merge their reports.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EVALS_DIR = Path(__file__).resolve().parents[1]
"""Root of the evals package (libs/evals/)."""

_DEFAULT_OUT_DIR = _EVALS_DIR / "trial_runs"
"""Default location for per-trial reports and the aggregated summary."""

_MAX_TRIALS = 50
"""Hard cap on `--trials` to bound runtime / cost from a typo."""

_SCALAR_METRICS = (
    "correctness",
    "solve_rate",
    "step_ratio",
    "tool_call_ratio",
    "median_duration_s",
)
"""Top-level numeric metrics aggregated across trials."""

_COUNT_FIELDS = ("passed", "failed", "skipped", "total")
"""Per-run counts surfaced alongside the scalar metrics."""

_MIN_SAMPLES_FOR_STDEV = 2
"""`statistics.stdev` requires at least two samples; below that we report None."""

_MODEL_ENV_VAR = "DEEPAGENTS_EVALS_MODEL"
"""When set, used as the default value for `--model` if the flag is omitted."""


def _warn(message: str) -> None:
    """Emit a warning to stderr, prefixed with `::warning::` under GitHub Actions.

    The annotation prefix surfaces the message in the workflow run's annotations
    panel; locally it just prints as a plain line.
    """
    prefix = "::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "warning: "
    print(f"{prefix}{message}", file=sys.stderr)


@dataclass(frozen=True)
class _Stats:
    """Summary statistics for a metric across trials.

    `n` is the count of trials that reported a non-null value for the metric.
    """

    n: int
    mean: float | None
    median: float | None
    stdev: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "min": self.min,
            "max": self.max,
        }


def _summarize(values: list[float]) -> _Stats:
    """Compute mean / median / stdev / min / max for a list of metric values.

    `stdev` is `None` for fewer than two samples; sample stdev (Bessel-corrected,
    which is what `statistics.stdev` returns) requires n >= 2.

    Args:
        values: Per-trial metric values; callers should filter out `None`s
            before passing.

    Returns:
        An `_Stats` record. All numeric fields are `None` when `values` is
        empty.
    """
    if not values:
        return _Stats(n=0, mean=None, median=None, stdev=None, min=None, max=None)
    return _Stats(
        n=len(values),
        mean=round(statistics.mean(values), 6),
        median=round(statistics.median(values), 6),
        stdev=(
            round(statistics.stdev(values), 6) if len(values) >= _MIN_SAMPLES_FOR_STDEV else None
        ),
        min=round(min(values), 6),
        max=round(max(values), 6),
    )


def _collect_numeric_values(
    reports: list[dict[str, Any]],
    key: str,
    *,
    nested_under: str | None = None,
) -> list[float]:
    """Collect a metric's values across reports, warning on type regressions.

    `None` is silently treated as "this trial didn't report it" — that's a
    legitimate schema variant (e.g. `solve_rate` is `null` when no test passed).
    Anything else that isn't numeric is a sign the upstream reporter shape
    changed; warn loudly so it doesn't get silently dropped from the stats.

    Args:
        reports: Per-trial report dicts.
        key: Metric name to extract.
        nested_under: If set, look up `report[nested_under][key]` instead of
            `report[key]` (used for `category_scores`).

    Returns:
        A list of `float`-coerced values from reports that supplied a real
        number; `None` and missing entries are skipped silently, all other
        types are skipped with a warning.
    """
    values: list[float] = []
    for idx, r in enumerate(reports):
        if nested_under is not None:
            container = r.get(nested_under) or {}
            if key not in container:
                continue
            raw = container[key]
        else:
            if key not in r:
                continue
            raw = r[key]
        if raw is None:
            continue
        # Exact-type check: JSON only produces int / float / bool / str / list /
        # dict / None, and `bool` is a subclass of `int`, so an `isinstance`
        # check would silently coerce `True` / `False` to 1.0 / 0.0.
        if type(raw) not in (int, float):
            location = f"{nested_under}.{key}" if nested_under else key
            _warn(
                f"trial {idx + 1}: non-numeric value for {location!r}: {raw!r} "
                f"(type {type(raw).__name__}); excluded from stats"
            )
            continue
        values.append(float(raw))
    return values


def aggregate_trials(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-trial eval reports into a single summary dict.

    Args:
        reports: Per-trial report dicts as written by the eval pytest reporter
            (one dict per `evals_report_trial_*.json`).

    Returns:
        A dict containing `trials` (list of per-trial input reports trimmed to
        the fields that matter for cross-trial comparison), `metrics` (mean /
        median / stdev / min / max for each scalar metric), `counts` (same
        stats for pass/fail/skip/total), and `category_scores` (per-category
        stats across trials).

    Raises:
        ValueError: If `reports` is empty.
    """
    if not reports:
        msg = "aggregate_trials requires at least one report"
        raise ValueError(msg)

    # Reports are expected to share model + sdk_version. Mismatches usually
    # mean the user mixed runs from two campaigns; the summary picks the
    # first report's values regardless, so warn before that becomes silent.
    models: set[str] = {str(r["model"]) for r in reports if r.get("model") is not None}
    if len(models) > 1:
        _warn(
            f"reports disagree on `model` ({sorted(models)!r}); "
            f"summary will use {reports[0].get('model')!r}"
        )
    sdks: set[str] = {
        str(r["sdk_version"]) for r in reports if r.get("sdk_version") is not None
    }
    if len(sdks) > 1:
        _warn(
            f"reports disagree on `sdk_version` ({sorted(sdks)!r}); "
            f"summary will use {reports[0].get('sdk_version')!r}"
        )

    metrics: dict[str, dict[str, Any]] = {}
    for key in _SCALAR_METRICS:
        metrics[key] = _summarize(_collect_numeric_values(reports, key)).to_dict()

    counts: dict[str, dict[str, Any]] = {}
    for key in _COUNT_FIELDS:
        counts[key] = _summarize(_collect_numeric_values(reports, key)).to_dict()

    all_categories = sorted({cat for r in reports for cat in (r.get("category_scores") or {})})
    category_stats: dict[str, dict[str, Any]] = {}
    for cat in all_categories:
        category_stats[cat] = _summarize(
            _collect_numeric_values(reports, cat, nested_under="category_scores")
        ).to_dict()

    return {
        "n_trials": len(reports),
        "model": reports[0].get("model"),
        "sdk_version": reports[0].get("sdk_version"),
        "metrics": metrics,
        "counts": counts,
        "category_scores": category_stats,
        "trials": [
            {
                "trial_index": idx + 1,
                "created_at": r.get("created_at"),
                "passed": r.get("passed"),
                "failed": r.get("failed"),
                "skipped": r.get("skipped"),
                "total": r.get("total"),
                "correctness": r.get("correctness"),
                "solve_rate": r.get("solve_rate"),
                "step_ratio": r.get("step_ratio"),
                "tool_call_ratio": r.get("tool_call_ratio"),
                "median_duration_s": r.get("median_duration_s"),
                "category_scores": r.get("category_scores", {}),
                "experiment_urls": r.get("experiment_urls", []),
                "pytest_returncode": r.get("pytest_returncode"),
            }
            for idx, r in enumerate(reports)
        ],
    }


def _build_pytest_args(args: argparse.Namespace, report_path: Path) -> list[str]:
    """Build the `uv run ... pytest tests/evals` argv for one trial.

    Args:
        args: Parsed CLI arguments for the trial runner.
        report_path: Path the trial should write its JSON report to.

    Returns:
        The full argv list passed to `subprocess.run`.
    """
    cmd: list[str] = [
        "uv",
        "run",
        "--group",
        "test",
        "pytest",
        "tests/evals",
        "-v",
        "--tb=short",
        "--model",
        args.model,
        "--evals-report-file",
        str(report_path),
    ]
    for cat in args.eval_category or []:
        cmd.extend(["--eval-category", cat])
    for tier in args.eval_tier or []:
        cmd.extend(["--eval-tier", tier])
    if args.openai_reasoning_effort:
        cmd.extend(["--openai-reasoning-effort", args.openai_reasoning_effort])
    if args.openrouter_provider:
        cmd.extend(["--openrouter-provider", args.openrouter_provider])
    if args.openrouter_allow_fallbacks:
        cmd.append("--openrouter-allow-fallbacks")
    if args.repl:
        cmd.extend(["--repl", args.repl])
    cmd.extend(args.pytest_extra)
    return cmd


@dataclass(frozen=True)
class _TrialOutcome:
    """Result of running one trial.

    `report_path` is `None` when pytest exited without writing a report.
    `returncode` is the pytest process exit code; non-zero values can occur
    *with* a usable report (e.g. session-scoped teardown crashed after the
    reporter wrote) and should be surfaced rather than silently ignored.
    """

    report_path: Path | None
    returncode: int


def _run_trial(
    *,
    trial_index: int,
    n_trials: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> _TrialOutcome:
    """Execute a single trial.

    Args:
        trial_index: One-based index of this trial in the sweep.
        n_trials: Total number of trials being run; only used for logging.
        args: Parsed CLI arguments (forwarded to `_build_pytest_args`).
        out_dir: Directory the trial's report file is written under.

    Returns:
        A `_TrialOutcome` with the report path (`None` if not produced) and
        the pytest exit code.
    """
    report_path = out_dir / f"evals_report_trial_{trial_index:03d}.json"
    cmd = _build_pytest_args(args, report_path)

    env = os.environ.copy()
    env.setdefault("LANGSMITH_TEST_SUITE", "deepagents-evals")
    env["DEEPAGENTS_TRIAL_INDEX"] = str(trial_index)
    env["DEEPAGENTS_TRIAL_TOTAL"] = str(n_trials)

    print(
        f"\n=== trial {trial_index}/{n_trials} === model={args.model} report={report_path}",
        flush=True,
    )
    print(f"$ {' '.join(cmd)}", flush=True)

    # `check=False`: a failing trial may still write a partial report we want
    # to aggregate; the caller inspects the returncode to decide what to do.
    result = subprocess.run(
        cmd,
        cwd=_EVALS_DIR,
        env=env,
        check=False,
    )

    if not report_path.is_file():
        _warn(
            f"trial {trial_index} produced no report at {report_path} "
            f"(pytest exit code {result.returncode}); skipping in aggregation"
        )
        return _TrialOutcome(report_path=None, returncode=result.returncode)
    if result.returncode != 0:
        _warn(
            f"trial {trial_index} pytest exited non-zero ({result.returncode}); "
            f"report at {report_path} will be aggregated but flagged"
        )
    return _TrialOutcome(report_path=report_path, returncode=result.returncode)


def _discover_reports(root: Path) -> list[Path]:
    """Recursively find eval report JSON files under `root`.

    Matches both `evals_report_trial_*.json` (written by this script when
    running locally) and `evals_report.json` (written directly by the eval
    pytest reporter, which is the filename inside each `_eval.yml` artifact
    bundle). Sorted for deterministic ordering; the set-union dedupes the
    rare case of both patterns matching the same path.
    """
    if not root.is_dir():
        return []
    matches = {
        *root.rglob("evals_report.json"),
        *root.rglob("evals_report_trial_*.json"),
    }
    return sorted(matches)


def _load_report(path: Path) -> dict[str, Any] | None:
    """Read a per-trial report file.

    Returns `None` on filesystem errors (`OSError`), encoding errors
    (`UnicodeDecodeError`), JSON syntax errors, or when the top-level value is
    not a dict. Each failure mode emits a warning so a single bad file in an
    aggregate sweep doesn't get silently dropped.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _warn(f"could not read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        _warn(f"{path} is not a JSON object; skipping")
        return None
    return data


def _print_summary(summary: dict[str, Any]) -> None:
    """Pretty-print the aggregate summary to stdout."""
    print()
    header = f"=== {summary['n_trials']} trials · model={summary.get('model')} ==="
    nonzero = [
        t["trial_index"]
        for t in summary.get("trials", [])
        if isinstance(t.get("pytest_returncode"), int) and t["pytest_returncode"] != 0
    ]
    if nonzero:
        header += f" ({len(nonzero)} had non-zero pytest exit: {nonzero})"
    print(header)
    metrics = summary["metrics"]
    width = max(len(k) for k in (*_SCALAR_METRICS, *_COUNT_FIELDS))
    for key in _SCALAR_METRICS:
        s = metrics[key]
        if s["n"] == 0:
            print(f"  {key:<{width}}  (no data)")
            continue
        stdev = "n/a" if s["stdev"] is None else f"{s['stdev']:.4f}"
        print(
            f"  {key:<{width}}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
            f"stdev={stdev}  min={s['min']:.4f}  max={s['max']:.4f}  (n={s['n']})"
        )
    for key in _COUNT_FIELDS:
        s = summary["counts"][key]
        if s["n"] == 0:
            continue
        stdev = "n/a" if s["stdev"] is None else f"{s['stdev']:.2f}"
        print(
            f"  {key:<{width}}  mean={s['mean']:.2f}  median={s['median']:.2f}  "
            f"stdev={stdev}  min={s['min']:.0f}  max={s['max']:.0f}"
        )
    if summary.get("category_scores"):
        print("\n  per-category correctness (mean ± stdev across trials):")
        for cat, s in sorted(summary["category_scores"].items()):
            if s["n"] == 0:
                continue
            stdev = "n/a" if s["stdev"] is None else f"±{s['stdev']:.3f}"
            print(f"    {cat}: {s['mean']:.3f} {stdev}  (n={s['n']})")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv into a Namespace.

    Enforces `--model` and `--trials` (with bounds) unless `--aggregate-only`
    is set, in which case the script just merges existing reports.
    """
    parser = argparse.ArgumentParser(
        description="Run the eval suite N times and aggregate metrics."
    )
    parser.add_argument(
        "--aggregate-only",
        type=Path,
        default=None,
        help=(
            "Skip trial execution; aggregate existing reports under DIR (recursive "
            "search for evals_report*.json). Used by CI after artifacts are downloaded."
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Where to write trials_summary.json (default: <out-dir>/trials_summary.json).",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=os.environ.get(_MODEL_ENV_VAR),
        help=(
            "Model identifier, e.g. openai:gpt-5.5 (passed through to pytest). "
            f"Required unless --aggregate-only is set. Defaults to ${_MODEL_ENV_VAR} when set."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        required=False,
        help=f"Number of trials to run (1..{_MAX_TRIALS}). Required unless --aggregate-only is set.",
    )
    parser.add_argument(
        "--eval-category",
        action="append",
        default=[],
        help="Restrict to one eval category (repeatable).",
    )
    parser.add_argument(
        "--eval-tier",
        action="append",
        default=[],
        help="Restrict to one eval tier (repeatable: baseline | hillclimb).",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument(
        "--openrouter-provider",
        default=None,
        help=(
            "Pin OpenRouter to one or more providers (comma-separated allowlist), "
            "e.g. MiniMax or MiniMax,Fireworks."
        ),
    )
    parser.add_argument(
        "--openrouter-allow-fallbacks",
        action="store_true",
        default=False,
        help=(
            "Allow OpenRouter to fall back outside --openrouter-provider when the "
            "listed providers are unavailable. Default is strict (no fallbacks)."
        ),
    )
    parser.add_argument(
        "--repl",
        choices=("quickjs",),
        default=None,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=f"Where per-trial reports and the summary are written (default: {_DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help=(
            "Emit the aggregated summary as compact JSON on stdout (in addition "
            "to writing trials_summary.json). Useful for downstream agents."
        ),
    )
    parser.add_argument(
        "pytest_extra",
        nargs=argparse.REMAINDER,
        help="Extra args to forward to pytest (use `--` to separate).",
    )
    args = parser.parse_args(argv)

    if args.aggregate_only is None:
        if not args.model:
            parser.error(
                f"--model is required (unless --aggregate-only is set). "
                f"Set ${_MODEL_ENV_VAR} or pass --model. "
                f"Run `deepagents-evals list models` for known specs."
            )
        if args.trials is None:
            parser.error("--trials is required (unless --aggregate-only is set)")
        if not 1 <= args.trials <= _MAX_TRIALS:
            parser.error(f"--trials must be between 1 and {_MAX_TRIALS}")
    if args.pytest_extra and args.pytest_extra[0] == "--":
        args.pytest_extra = args.pytest_extra[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    """Run N eval trials and write `<out-dir>/trials_summary.json`.

    With `--aggregate-only DIR`, skip trial execution and aggregate report
    files already on disk (e.g. CI artifacts).

    Returns:
        Process exit code: `0` on success, `1` when no usable reports were
        found (either no trial produced one, or every produced file was
        unreadable).
    """
    args = _parse_args(argv)

    # Maps a report path to the pytest exit code that produced it; only
    # populated for the live-execution path so partial failures can be
    # surfaced in the summary header and per-trial JSON.
    returncodes: dict[Path, int] = {}

    if args.aggregate_only is not None:
        report_paths = _discover_reports(args.aggregate_only)
        if not report_paths:
            print(
                f"error: no eval report JSON files found under {args.aggregate_only}",
                file=sys.stderr,
            )
            return 1
        out_dir = args.aggregate_only
    else:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        report_paths = []
        for i in range(args.trials):
            outcome = _run_trial(
                trial_index=i + 1,
                n_trials=args.trials,
                args=args,
                out_dir=out_dir,
            )
            if outcome.report_path is not None:
                report_paths.append(outcome.report_path)
                returncodes[outcome.report_path] = outcome.returncode

        if not report_paths:
            print("error: no trial produced a report; nothing to aggregate", file=sys.stderr)
            return 1

    reports: list[dict[str, Any]] = []
    for p in report_paths:
        data = _load_report(p)
        if data is None:
            continue
        if p in returncodes:
            data.setdefault("pytest_returncode", returncodes[p])
        reports.append(data)

    if not reports:
        print("error: no readable trial reports found", file=sys.stderr)
        return 1

    summary = aggregate_trials(reports)
    summary_path = args.summary_out or (out_dir / "trials_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        # Compact single-line JSON keeps stdout machine-parseable; the human
        # summary still goes to the on-disk `trials_summary.json` and to
        # stderr below for terminal users.
        print(json.dumps(summary, sort_keys=True))
        print(f"wrote {summary_path}", file=sys.stderr)
    else:
        _print_summary(summary)
        print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
