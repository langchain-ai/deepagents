from __future__ import annotations

import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from deepagents._version import __version__
from deepagents.graph import get_default_model

_RESULTS: dict[str, int] = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "total": 0,
}

_DURATIONS_S: list[float] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--evals-report-file",
        action="store",
        default=os.environ.get("DEEPAGENTS_EVALS_REPORT_FILE"),
        help=("Write a JSON eval report to this path. If omitted, no JSON report is written. Can also be set via DEEPAGENTS_EVALS_REPORT_FILE."),
    )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return

    _RESULTS["total"] += 1

    _DURATIONS_S.append(float(report.duration))

    outcome = report.outcome
    if outcome in {"passed", "failed", "skipped"}:
        _RESULTS[outcome] += 1


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = exitstatus
    if session.exitstatus == 1:
        session.exitstatus = 0

    correctness_accuracy = round((_RESULTS["passed"] / _RESULTS["total"]) if _RESULTS["total"] else 0.0, 2)
    median_duration_s = round(statistics.median(_DURATIONS_S), 4) if _DURATIONS_S else 0.0

    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sdk_version": __version__,
        "model": session.config.getoption("--model") or str(session.config._inicache.get("model", "")) or str(get_default_model().model),
        **_RESULTS,
        "correctness_accuracy": correctness_accuracy,
        "median_duration_s": median_duration_s,
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
        terminal_reporter.write_line(f"correctness_accuracy: {payload['correctness_accuracy']:.2f}")
        terminal_reporter.write_line(f"median_duration_s: {payload['median_duration_s']:.4f}")

    report_path_opt = session.config.getoption("--evals-report-file")
    if not report_path_opt:
        return

    report_path = Path(str(report_path_opt))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
