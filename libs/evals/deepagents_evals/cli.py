"""Unified, agent-friendly CLI for the Deep Agents evaluation suite.

Wraps the scattered `Makefile` targets and `scripts/*.py` entry points behind
a single `deepagents-evals` console script with discoverable subcommands and
machine-readable output.

Subcommands:
    run            Run the eval suite once (single trial).
    trials         Run the eval suite N times and aggregate metrics.
    aggregate      Aggregate previously-written trial reports.
    radar          Generate a radar chart from results.
    catalog        Regenerate or check `EVAL_CATALOG.md`.
    model-groups   Regenerate or check `MODEL_GROUPS.md`.
    list           Discover categories / tiers / models / evals.

Exit codes:
    0   Success.
    1   Eval failures: `run` saw a non-zero pytest exit, or `trials` /
        `aggregate` produced a summary whose aggregated `counts.failed.mean`
        is greater than zero. `radar` non-zero exits also map here.
    2   Configuration error: missing `--model`, registry-load failure, or
        a `--check` drift detector found that a generated file is stale.
    3   No usable reports: `trials` / `aggregate` produced no summary,
        or `--retry-failed` could not parse any prior reports.

Most subcommands accept `--json` for structured stdout and `--dry-run` for
preview-only execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_OK = 0
"""Successful run."""

EXIT_EVAL_FAILURES = 1
"""At least one test failed in this run.

For `run`, set when `pytest` exited non-zero. For `trials` / `aggregate`,
set when the aggregated summary's `counts.failed.mean` is greater than
zero — `pytest_reporter` rewrites `session.exitstatus` to 0 even on test
failures, so the per-trial returncodes are not a reliable signal.
"""

EXIT_CONFIG = 2
"""Bad CLI arguments, missing env vars, or unknown discovery values."""

EXIT_NO_REPORTS = 3
"""No readable reports were produced; nothing to aggregate."""

_PACKAGE_DIR = Path(__file__).resolve().parent
"""`libs/evals/deepagents_evals/`."""

_EVALS_DIR = _PACKAGE_DIR.parent
"""`libs/evals/`."""

_REPO_ROOT = _EVALS_DIR.parent.parent
"""Monorepo root."""

_CATEGORIES_JSON = _PACKAGE_DIR / "categories.json"
"""Source of truth for eval categories."""

_KNOWN_TIERS = ("baseline", "hillclimb")
"""Eval tier marker values recognized by `--eval-tier` / `pytest.mark.eval_tier`."""

_MODEL_ENV_VAR = "DEEPAGENTS_EVALS_MODEL"
"""When set, used as the default value for `--model` on `run` and `trials`."""


def _load_categories() -> list[str]:
    """Return the canonical list of eval categories from `categories.json`."""
    data = json.loads(_CATEGORIES_JSON.read_text(encoding="utf-8"))
    return list(data.get("categories", []))


def _load_module_by_path(name: str, path: Path) -> Any:
    """Import a Python file by absolute path, caching it in `sys.modules`.

    Registering the module before `exec_module` runs is required because
    classes defined in the loaded file resolve `cls.__module__` via
    `sys.modules[name]` during construction. Without it, `@dataclass`
    decorators (used in `scripts/run_trials.py`) fail with `AttributeError`.

    Subsequent calls return the cached module so monkeypatched attributes
    survive across `_import_*` helpers (the helpers are called on every
    subcommand invocation; without caching, a fresh module would be loaded
    each time and tests' patches would silently miss).
    """
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        msg = f"could not import {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_models_module() -> Any:
    """Import the registry module from `.github/scripts/models.py` lazily.

    Kept lazy because the module isn't installable as a package and importing
    it requires a `spec_from_file_location` dance — we only pay the cost when
    `list models` is actually invoked.
    """
    return _load_module_by_path(
        "_deepagents_evals_models", _REPO_ROOT / ".github" / "scripts" / "models.py"
    )


def _list_known_models() -> list[dict[str, Any]]:
    """Return all eval-tagged models with their groups and provider labels."""
    mod = _import_models_module()
    out: list[dict[str, Any]] = []
    for m in mod.REGISTRY:
        eval_groups = sorted(g.split(":", 1)[1] for g in m.groups if g.startswith("eval:"))
        if not eval_groups:
            continue
        out.append(
            {
                "spec": m.spec,
                "display_name": m.display_name,
                "provider_label": m.provider_label,
                "groups": eval_groups,
            }
        )
    out.sort(key=lambda r: r["spec"])
    return out


def _list_known_evals() -> list[dict[str, Any]]:
    """Return all discovered eval functions with their categories.

    Reuses the AST walker from `scripts/generate_eval_catalog.py` rather than
    importing the test modules (which would require LangSmith env config and
    the full eval dependency graph).
    """
    mod = _load_module_by_path(
        "_deepagents_eval_catalog", _EVALS_DIR / "scripts" / "generate_eval_catalog.py"
    )
    catalog: dict[str, list[tuple[str, str, int]]] = mod._collect_evals()  # noqa: SLF001
    out: list[dict[str, Any]] = []
    for category, entries in catalog.items():
        for name, path, line in entries:
            out.append(
                {
                    "name": name,
                    "category": category,
                    "path": path,
                    "line": line,
                }
            )
    out.sort(key=lambda r: (r["category"], r["path"], r["line"]))
    return out


def _emit_json(payload: object) -> None:
    """Print `payload` as compact JSON on a single line."""
    print(json.dumps(payload, sort_keys=True))


def _emit_table(rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    """Render `rows` as a fixed-width table on stdout."""
    if not rows:
        return
    widths = {col: max(len(col), *(len(str(r.get(col, ""))) for r in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for r in rows:
        print("  ".join(str(r.get(col, "")).ljust(widths[col]) for col in columns))


def _cmd_list(args: argparse.Namespace) -> int:
    """Discover available categories / tiers / models / evals."""
    if args.target == "categories":
        cats = _load_categories()
        if args.json:
            _emit_json(cats)
        else:
            print("\n".join(cats))
        return EXIT_OK

    if args.target == "tiers":
        if args.json:
            _emit_json(list(_KNOWN_TIERS))
        else:
            print("\n".join(_KNOWN_TIERS))
        return EXIT_OK

    if args.target == "models":
        try:
            models = _list_known_models()
        except Exception as exc:  # noqa: BLE001
            print(
                f"error: failed to load model registry from .github/scripts/models.py: {exc}",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        if args.group:
            models = [m for m in models if args.group in m["groups"]]
        if args.provider:
            models = [m for m in models if m["spec"].split(":", 1)[0] == args.provider]
        if args.json:
            _emit_json(models)
        else:
            _emit_table(models, ("spec", "display_name", "provider_label", "groups"))
        return EXIT_OK

    if args.target == "evals":
        try:
            evals = _list_known_evals()
        except Exception as exc:  # noqa: BLE001
            print(
                f"error: failed to discover evals from tests/evals/: {exc}",
                file=sys.stderr,
            )
            return EXIT_CONFIG
        if args.category:
            evals = [e for e in evals if e["category"] == args.category]
        if args.json:
            _emit_json(evals)
        else:
            _emit_table(evals, ("category", "name", "path", "line"))
        return EXIT_OK

    msg = f"unknown list target: {args.target!r}"
    raise AssertionError(msg)


def _build_single_trial_argv(args: argparse.Namespace) -> list[str]:
    """Build the `pytest tests/evals` argv for a single-trial run.

    Mirrors `make evals` but adds optional report path and lets every flag from
    `tests/evals/conftest.py:pytest_addoption` flow through.
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
    ]
    if args.report:
        cmd.extend(["--evals-report-file", str(args.report)])
    for cat in args.eval_category or []:
        cmd.extend(["--eval-category", cat])
    for cat in args.eval_category_exclude or []:
        cmd.extend(["--eval-category-exclude", cat])
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


def _validate_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Resolve `--model` from args / env, exit-2 on failure with a helpful message."""
    if args.model:
        return
    env_model = os.environ.get(_MODEL_ENV_VAR)
    if env_model:
        args.model = env_model
        return
    sample_groups: list[str] = []
    try:
        models = _list_known_models()
        all_groups = sorted({g for m in models for g in m["groups"]})
        sample_groups = all_groups[:8]
    except Exception as exc:  # noqa: BLE001
        # Discovery is best-effort, but a failing registry probe is itself
        # worth surfacing — silent regressions in `.github/scripts/models.py`
        # would otherwise rot until someone runs `list models` directly.
        print(
            f"warning: could not enumerate model groups for help text: {exc}",
            file=sys.stderr,
        )
    extra = (
        f"\n  Known model groups (run `deepagents-evals list models --group <name>`): "
        f"{', '.join(sample_groups)}"
        if sample_groups
        else ""
    )
    parser.exit(
        EXIT_CONFIG,
        f"error: --model is required (or set {_MODEL_ENV_VAR}). "
        f"Example: --model claude-sonnet-4-6{extra}\n",
    )


def _cmd_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the eval suite once via `pytest tests/evals`."""
    _validate_model(args, parser)
    cmd = _build_single_trial_argv(args)
    if args.dry_run:
        if args.json:
            _emit_json({"dry_run": True, "argv": cmd, "cwd": str(_EVALS_DIR)})
        else:
            print("$ " + " ".join(cmd))
        return EXIT_OK
    env = os.environ.copy()
    env.setdefault("LANGSMITH_TEST_SUITE", "deepagents-evals")
    result = subprocess.run(cmd, cwd=_EVALS_DIR, env=env, check=False)
    if args.json:
        _emit_json(
            {
                "model": args.model,
                "returncode": result.returncode,
                "report": str(args.report) if args.report else None,
            }
        )
    if result.returncode == 0:
        return EXIT_OK
    return EXIT_EVAL_FAILURES


def _import_run_trials() -> Any:
    """Import `scripts/run_trials.py` lazily (the script lives outside the package)."""
    return _load_module_by_path("_deepagents_run_trials", _EVALS_DIR / "scripts" / "run_trials.py")


def _collect_failed_nodeids(summary_or_dir: Path) -> tuple[list[str], int, int]:
    """Read failure node IDs from per-trial reports.

    `summary_or_dir` may be a `trials_summary.json` file (its parent directory
    is searched) or a directory containing `evals_report_trial_*.json` files.
    Per-trial reports each carry a `failures` array of
    `{test_name, category, failure_message}` dicts; node IDs are deduped
    across trials so a flake that failed once is retried once.

    Returns:
        A tuple `(nodeids, n_total, n_loaded)` where `n_total` is the number
            of report files discovered and `n_loaded` is the subset that parsed
            successfully. Callers use the difference to distinguish "every trial
            passed" from "every report was unreadable".
    """
    rt = _import_run_trials()
    root = summary_or_dir.parent if summary_or_dir.is_file() else summary_or_dir
    nodeids: set[str] = set()
    paths = rt._discover_reports(root)  # noqa: SLF001
    n_loaded = 0
    for path in paths:
        data = rt._load_report(path)  # noqa: SLF001
        if not data:
            continue
        n_loaded += 1
        failures = data.get("failures")
        if failures is None:
            continue
        if not isinstance(failures, list):
            print(
                f"warning: ignoring non-list `failures` in {path}: {type(failures).__name__}",
                file=sys.stderr,
            )
            continue
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            nodeid = failure.get("test_name")
            if isinstance(nodeid, str) and nodeid:
                nodeids.add(nodeid)
    return sorted(nodeids), len(paths), n_loaded


def _cmd_trials(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run N trials and aggregate, delegating to `scripts/run_trials.py`."""
    _validate_model(args, parser)
    rt = _import_run_trials()

    rt_argv: list[str] = ["--model", args.model, "--trials", str(args.trials)]
    for cat in args.eval_category or []:
        rt_argv.extend(["--eval-category", cat])
    for tier in args.eval_tier or []:
        rt_argv.extend(["--eval-tier", tier])
    if args.openai_reasoning_effort:
        rt_argv.extend(["--openai-reasoning-effort", args.openai_reasoning_effort])
    if args.openrouter_provider:
        rt_argv.extend(["--openrouter-provider", args.openrouter_provider])
    if args.openrouter_allow_fallbacks:
        rt_argv.append("--openrouter-allow-fallbacks")
    if args.repl:
        rt_argv.extend(["--repl", args.repl])
    if args.out_dir:
        rt_argv.extend(["--out-dir", str(args.out_dir)])
    if args.summary_out:
        rt_argv.extend(["--summary-out", str(args.summary_out)])

    if args.retry_failed:
        nodeids, n_total, n_loaded = _collect_failed_nodeids(args.retry_failed)
        if not nodeids:
            if n_total > 0 and n_loaded == 0:
                print(
                    f"error: discovered {n_total} report(s) under {args.retry_failed} "
                    f"but none parsed; see warnings above",
                    file=sys.stderr,
                )
            else:
                print(
                    f"error: no failed test node IDs found under {args.retry_failed}",
                    file=sys.stderr,
                )
            return EXIT_NO_REPORTS
        if args.dry_run:
            payload = {"retry_failed": nodeids, "trials": args.trials, "model": args.model}
            if args.json:
                _emit_json(payload)
            else:
                print(f"would retry {len(nodeids)} failed test(s):\n  " + "\n  ".join(nodeids))
            return EXIT_OK
        rt_argv.extend(["--", *nodeids])
    else:
        rt_argv.extend(args.pytest_extra or [])

    if args.dry_run:
        # Show the argv that would be passed to run_trials.main().
        msg = {"argv": rt_argv, "script": str(_EVALS_DIR / "scripts" / "run_trials.py")}
        if args.json:
            _emit_json(msg)
        else:
            print("would run: scripts/run_trials.py " + " ".join(rt_argv))
        return EXIT_OK

    rc = rt.main(rt_argv)
    if rc != 0:
        # `run_trials.main` only returns 0 (aggregation done) or 1 (no reports).
        return EXIT_NO_REPORTS
    summary_path = _resolve_summary_path(args)
    return _exit_code_from_summary(summary_path)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Aggregate existing reports under a directory."""
    rt = _import_run_trials()
    rt_argv = ["--aggregate-only", str(args.directory)]
    if args.summary_out:
        rt_argv.extend(["--summary-out", str(args.summary_out)])
    if args.json:
        rt_argv.append("--json")
    rc = rt.main(rt_argv)
    if rc != 0:
        return EXIT_NO_REPORTS
    summary_path = args.summary_out or (args.directory / "trials_summary.json")
    return _exit_code_from_summary(summary_path)


def _resolve_summary_path(args: argparse.Namespace) -> Path:
    """Compute where `run_trials.main()` wrote its `trials_summary.json`.

    Mirrors `scripts/run_trials.py` defaults: `--summary-out` wins, otherwise
    `<--out-dir or trial_runs>/trials_summary.json` next to the per-trial
    reports.
    """
    if args.summary_out:
        return args.summary_out
    out_dir = args.out_dir or (_EVALS_DIR / "trial_runs")
    return out_dir / "trials_summary.json"


def _exit_code_from_summary(summary_path: Path) -> int:
    """Map an aggregated `trials_summary.json` to a CLI exit code.

    `pytest_reporter` rewrites the per-trial `session.exitstatus` to 0 when
    tests fail (so CI doesn't fail the workflow shell), which means the
    aggregated `pytest_returncode` is also 0 for failure-laden runs. The
    only reliable signal is the aggregated `counts.failed.mean` produced by
    `aggregate_trials`. A non-zero failed mean ⇒ at least one test failed
    in at least one trial.
    """
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {summary_path}: {exc}", file=sys.stderr)
        return EXIT_OK
    failed = data.get("counts", {}).get("failed", {}).get("mean")
    if isinstance(failed, (int, float)) and failed > 0:
        return EXIT_EVAL_FAILURES
    return EXIT_OK


def _shell_out(
    cmd: list[str],
    *,
    dry_run: bool,
    json_mode: bool,
    nonzero_exit: int = EXIT_EVAL_FAILURES,
) -> int:
    """Run an external command from `_EVALS_DIR`, honoring dry-run / JSON modes.

    `nonzero_exit` controls how a non-zero subprocess return is mapped to a
    CLI exit code; defaults to `EXIT_EVAL_FAILURES` for the eval-running
    paths and is overridden to `EXIT_CONFIG` for `--check` drift detectors,
    where a non-zero exit means "files are out of date", not "evals failed".
    """
    if dry_run:
        if json_mode:
            _emit_json({"dry_run": True, "argv": cmd, "cwd": str(_EVALS_DIR)})
        else:
            print("$ " + " ".join(cmd))
        return EXIT_OK
    result = subprocess.run(cmd, cwd=_EVALS_DIR, check=False)
    return EXIT_OK if result.returncode == 0 else nonzero_exit


def _cmd_radar(args: argparse.Namespace) -> int:
    """Generate a radar chart by delegating to `scripts/generate_radar.py`."""
    cmd = ["uv", "run", "--extra", "charts", "python", "scripts/generate_radar.py"]
    if args.toy:
        cmd.append("--toy")
    if args.summary:
        cmd.extend(["--summary", str(args.summary)])
    if args.results:
        cmd.extend(["--results", str(args.results)])
    if args.output:
        cmd.extend(["-o", str(args.output)])
    cmd.extend(args.extra or [])
    return _shell_out(cmd, dry_run=args.dry_run, json_mode=args.json)


def _cmd_catalog(args: argparse.Namespace) -> int:
    """Regenerate or check `EVAL_CATALOG.md`."""
    cmd = ["uv", "run", "python", "scripts/generate_eval_catalog.py"]
    if args.check:
        cmd.append("--check")
    # `--check` non-zero means drift, which is a configuration condition
    # (the file is stale), not an eval failure.
    nonzero = EXIT_CONFIG if args.check else EXIT_EVAL_FAILURES
    return _shell_out(cmd, dry_run=args.dry_run, json_mode=args.json, nonzero_exit=nonzero)


def _cmd_model_groups(args: argparse.Namespace) -> int:
    """Regenerate or check `MODEL_GROUPS.md`."""
    cmd = ["uv", "run", "python", "scripts/generate_model_groups.py"]
    if args.check:
        cmd.append("--check")
    nonzero = EXIT_CONFIG if args.check else EXIT_EVAL_FAILURES
    return _shell_out(cmd, dry_run=args.dry_run, json_mode=args.json, nonzero_exit=nonzero)


def _add_run_options(p: argparse.ArgumentParser) -> None:
    """Attach the shared `--model` / category / tier / provider flags."""
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"Model identifier. Defaults to ${_MODEL_ENV_VAR} when --model is omitted; "
            f"the explicit flag wins when both are set."
        ),
    )
    p.add_argument(
        "--eval-category",
        action="append",
        default=[],
        help="Restrict to one eval category (repeatable).",
    )
    p.add_argument(
        "--eval-category-exclude",
        action="append",
        default=[],
        help="Exclude one eval category (repeatable).",
    )
    p.add_argument(
        "--eval-tier",
        action="append",
        default=[],
        choices=_KNOWN_TIERS,
        help="Restrict to one eval tier (repeatable).",
    )
    p.add_argument(
        "--openai-reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default=None,
    )
    p.add_argument("--openrouter-provider", default=None)
    p.add_argument("--openrouter-allow-fallbacks", action="store_true", default=False)
    p.add_argument("--repl", choices=("quickjs",), default=None)
    p.add_argument("--dry-run", action="store_true", help="Print the command instead of running.")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON on stdout.")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser for `deepagents-evals`."""
    parser = argparse.ArgumentParser(
        prog="deepagents-evals",
        description=(
            "Unified, agent-friendly CLI for the Deep Agents evaluation suite. "
            "See `deepagents-evals <subcommand> --help` for details."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<subcommand>")

    # run
    p_run = sub.add_parser("run", help="Run the eval suite once.")
    _add_run_options(p_run)
    p_run.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the per-run JSON report (passes --evals-report-file to pytest).",
    )
    p_run.add_argument(
        "pytest_extra",
        nargs=argparse.REMAINDER,
        help="Extra pytest args (use `--` to separate).",
    )
    p_run.set_defaults(func=lambda a: _cmd_run(a, p_run))

    # trials
    p_trials = sub.add_parser("trials", help="Run the eval suite N times and aggregate.")
    _add_run_options(p_trials)
    p_trials.add_argument("--trials", type=int, default=None)
    p_trials.add_argument("--out-dir", type=Path, default=None)
    p_trials.add_argument("--summary-out", type=Path, default=None)
    p_trials.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        help=(
            "Re-run only the failing tests from a prior run. PATH may be a "
            "trials_summary.json file or a directory containing per-trial reports."
        ),
    )
    p_trials.add_argument("pytest_extra", nargs=argparse.REMAINDER)
    p_trials.set_defaults(func=lambda a: _cmd_trials(a, p_trials))

    # aggregate
    p_agg = sub.add_parser("aggregate", help="Aggregate existing trial reports.")
    p_agg.add_argument("directory", type=Path, help="Directory to recursively scan for reports.")
    p_agg.add_argument("--summary-out", type=Path, default=None)
    p_agg.add_argument("--json", action="store_true")
    p_agg.set_defaults(func=_cmd_aggregate)

    # radar
    p_radar = sub.add_parser("radar", help="Generate a radar chart.")
    p_radar.add_argument("--toy", action="store_true")
    p_radar.add_argument("--summary", type=Path, default=None)
    p_radar.add_argument("--results", type=Path, default=None)
    p_radar.add_argument("-o", "--output", type=Path, default=None)
    p_radar.add_argument("--dry-run", action="store_true")
    p_radar.add_argument("--json", action="store_true")
    p_radar.add_argument("extra", nargs=argparse.REMAINDER)
    p_radar.set_defaults(func=_cmd_radar)

    # catalog
    p_cat = sub.add_parser("catalog", help="Regenerate or check EVAL_CATALOG.md.")
    p_cat.add_argument("--check", action="store_true")
    p_cat.add_argument("--dry-run", action="store_true")
    p_cat.add_argument("--json", action="store_true")
    p_cat.set_defaults(func=_cmd_catalog)

    # model-groups
    p_mg = sub.add_parser("model-groups", help="Regenerate or check MODEL_GROUPS.md.")
    p_mg.add_argument("--check", action="store_true")
    p_mg.add_argument("--dry-run", action="store_true")
    p_mg.add_argument("--json", action="store_true")
    p_mg.set_defaults(func=_cmd_model_groups)

    # list
    p_list = sub.add_parser("list", help="Discover categories / tiers / models / evals.")
    list_sub = p_list.add_subparsers(dest="target", required=True, metavar="<target>")
    p_list_cat = list_sub.add_parser("categories", help="List eval categories.")
    p_list_cat.add_argument("--json", action="store_true")
    p_list_tiers = list_sub.add_parser("tiers", help="List eval tiers.")
    p_list_tiers.add_argument("--json", action="store_true")
    p_list_models = list_sub.add_parser("models", help="List eval-tagged models.")
    p_list_models.add_argument("--group", default=None, help="Filter by group (e.g. set0).")
    p_list_models.add_argument("--provider", default=None, help="Filter by provider prefix.")
    p_list_models.add_argument("--json", action="store_true")
    p_list_evals = list_sub.add_parser("evals", help="List discovered eval functions.")
    p_list_evals.add_argument("--category", default=None, help="Filter by category.")
    p_list_evals.add_argument("--json", action="store_true")
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `deepagents-evals` console script.

    Args:
        argv: Optional argv override (mostly for tests). When `None`,
            `argparse` reads from `sys.argv`.

    Returns:
        One of the `EXIT_*` constants.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "trials" and args.trials is None:
        parser.error("--trials is required")
    # `argparse.REMAINDER` keeps a leading `--`; downstream callers don't want it.
    if hasattr(args, "pytest_extra") and args.pytest_extra and args.pytest_extra[0] == "--":
        args.pytest_extra = args.pytest_extra[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
