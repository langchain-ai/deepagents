from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from deepagents._version import __version__
from deepagents.graph import get_default_model

import tests.evals.utils as _evals_utils

_RESULTS: dict[str, int] = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0,
}
"""Aggregate pass/fail/skip/total counters across the entire session."""

_DURATIONS_S: list[float] = []
"""Wall-clock duration (seconds) of each test's `call` phase."""

_EFFICIENCY_RESULTS: list[_evals_utils.EfficiencyResult] = []
"""Per-test efficiency data (steps, tool calls) collected via the utils callback."""

_NODEID_TO_CATEGORY: dict[str, str] = {}
"""Mapping of pytest node ID to its `eval_category` mark value, built during collection."""

_CATEGORY_RESULTS: dict[str, dict[str, int]] = {}
"""Per-category pass/fail/total counters, keyed by category name."""

_EXPERIMENT_URLS: list[str] = []
"""LangSmith experiment comparison URLs collected during session teardown."""


def _micro_step_ratio() -> float | None:
    """Compute sum(actual_steps) / sum(expected_steps).

    Returns `None` when no tests specified expected step counts.
    """
    total_expected = 0
    total_actual = 0
    for r in _EFFICIENCY_RESULTS:
        if r.expected_steps is not None:
            total_expected += r.expected_steps
            total_actual += r.actual_steps
    if total_expected == 0:
        return None
    return round(total_actual / total_expected, 2)


def _micro_tool_call_ratio() -> float | None:
    """Compute sum(actual_tool_calls) / sum(expected_tool_calls).

    Returns `None` when no tests specified expected tool call counts.
    """
    total_expected = 0
    total_actual = 0
    for r in _EFFICIENCY_RESULTS:
        if r.expected_tool_calls is not None:
            total_expected += r.expected_tool_calls
            total_actual += r.actual_tool_calls
    if total_expected == 0:
        return None
    return round(total_actual / total_expected, 2)


def _solve_rate() -> float | None:
    """Compute solve rate: mean of per-test `expected_steps / duration_s` for eligible tests.

    For each test that passed and has both `expected_steps` and `duration_s`,
    the per-test contribution is `expected_steps / duration_s`. Tests that
    did not pass contribute zero. The result is the mean across all eligible
    tests.

    Returns `None` when no tests have the required data.
    """
    values: list[float] = []
    for r in _EFFICIENCY_RESULTS:
        if r.expected_steps is None or r.duration_s is None:
            continue
        if r.passed:
            values.append(r.expected_steps / r.duration_s if r.duration_s > 0 else 0.0)
        else:
            values.append(0.0)
    if not values:
        return None
    return round(statistics.mean(values), 4)


def pytest_configure(config: pytest.Config) -> None:
    _ = config
    _evals_utils._on_efficiency_result = _EFFICIENCY_RESULTS.append


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    for item in items:
        marker = item.get_closest_marker("eval_category")
        if marker and marker.args:
            _NODEID_TO_CATEGORY[item.nodeid] = str(marker.args[0])


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--evals-report-file",
        action="store",
        default=os.environ.get("DEEPAGENTS_EVALS_REPORT_FILE"),
        help=(
            "Write a JSON eval report to this path. If omitted, no JSON report is written. Can also be set via DEEPAGENTS_EVALS_REPORT_FILE."
        ),
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return

    _RESULTS["total"] += 1

    duration = float(report.duration)
    _DURATIONS_S.append(duration)

    outcome = report.outcome
    if outcome in {"passed", "failed", "skipped"}:
        _RESULTS[outcome] += 1

    category = _NODEID_TO_CATEGORY.get(report.nodeid)
    if category and outcome in {"passed", "failed"}:
        bucket = _CATEGORY_RESULTS.setdefault(category, {"passed": 0, "failed": 0, "total": 0})
        bucket[outcome] += 1
        bucket["total"] += 1

    if _EFFICIENCY_RESULTS and _EFFICIENCY_RESULTS[-1].duration_s is None:
        _EFFICIENCY_RESULTS[-1].duration_s = duration
        _EFFICIENCY_RESULTS[-1].passed = outcome == "passed"


def _collect_experiment_urls() -> list[str]:
    """Extract LangSmith experiment comparison URLs from registered test suites."""
    try:
        from langsmith.testing._internal import _LangSmithTestSuite
    except ImportError:
        return []

    try:
        instances = _LangSmithTestSuite._instances
        if not instances:
            return []

        urls: list[str] = []
        for suite in instances.values():
            dataset = getattr(suite, "_dataset", None)
            if dataset is None:
                continue
            dataset_url = getattr(dataset, "url", None)
            experiment_id = getattr(suite, "experiment_id", None)
            if dataset_url and experiment_id:
                urls.append(f"{dataset_url}/compare?selectedSessions={experiment_id}")
    except Exception as exc:  # noqa: BLE001  # private API; best-effort
        msg = f"warning: failed to collect experiment URLs: {exc}"
        print(msg, file=sys.stderr)  # noqa: T201
        return []
    else:
        return urls


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = exitstatus
    if session.exitstatus == 1:
        session.exitstatus = 0

    _EXPERIMENT_URLS.extend(_collect_experiment_urls())

    correctness = round((_RESULTS["passed"] / _RESULTS["total"]) if _RESULTS["total"] else 0.0, 2)
    step_ratio = _micro_step_ratio()
    tool_call_ratio = _micro_tool_call_ratio()
    solve_rate = _solve_rate()
    median_duration_s = round(statistics.median(_DURATIONS_S), 4) if _DURATIONS_S else 0.0

    category_scores: dict[str, float] = {}
    for cat, counts in sorted(_CATEGORY_RESULTS.items()):
        if counts["total"] > 0:
            category_scores[cat] = round(counts["passed"] / counts["total"], 2)

    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sdk_version": __version__,
        "model": session.config.getoption("--model")
        or str(session.config._inicache.get("model", ""))
        or str(get_default_model().model),
        **_RESULTS,
        "correctness": correctness,
        "category_scores": category_scores,
        "step_ratio": step_ratio,
        "tool_call_ratio": tool_call_ratio,
        "solve_rate": solve_rate,
        "median_duration_s": median_duration_s,
        "experiment_urls": _EXPERIMENT_URLS,
    }

    terminal_reporter = session.config.pluginmanager.getplugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_sep("=", "deepagents evals summary")
        terminal_reporter.write_line(f"created_at: {payload['created_at']}")
        terminal_reporter.write_line(f"sdk_version: {payload['sdk_version']}")
        terminal_reporter.write_line(f"model: {payload['model']}")
        terminal_reporter.write_line(
            f"results: {payload['passed']} passed, {payload['failed']} failed, {payload['skipped']} skipped (total={payload['total']})"
        )
        terminal_reporter.write_line(f"correctness: {correctness:.2f}")
        if category_scores:
            terminal_reporter.write_sep("-", "per-category correctness")
            for cat, score in sorted(category_scores.items()):
                counts = _CATEGORY_RESULTS[cat]
                terminal_reporter.write_line(
                    f"  {cat}: {score:.2f} ({counts['passed']}/{counts['total']})"
                )
        if step_ratio is not None:
            terminal_reporter.write_line(f"step_ratio: {step_ratio:.2f}")
        if tool_call_ratio is not None:
            terminal_reporter.write_line(f"tool_call_ratio: {tool_call_ratio:.2f}")
        if solve_rate is not None:
            terminal_reporter.write_line(f"solve_rate: {solve_rate:.4f}")
        terminal_reporter.write_line(f"median_duration_s: {median_duration_s:.4f}")
        if _EXPERIMENT_URLS:
            terminal_reporter.write_sep("-", "langsmith experiments")
            for url in _EXPERIMENT_URLS:
                terminal_reporter.write_line(f"  {url}")

    report_path_opt = session.config.getoption("--evals-report-file")
    if not report_path_opt:
        return

    report_path = Path(str(report_path_opt))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
