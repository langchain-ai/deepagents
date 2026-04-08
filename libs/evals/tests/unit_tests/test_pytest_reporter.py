"""Tests for the eval pytest reporter plugin — specifically the _FAILURES capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tests.evals.pytest_reporter as reporter


@dataclass
class _FakeReport:
    """Minimal stand-in for `pytest.TestReport`."""

    nodeid: str
    when: str
    outcome: str
    duration: float
    longreprtext: str = ""


class TestFailuresCapture:
    """Verify that _FAILURES is populated on test failures."""

    def setup_method(self):
        reporter._FAILURES.clear()
        reporter._RESULTS.update(passed=0, failed=0, skipped=0, total=0)
        reporter._DURATIONS_S.clear()
        reporter._EFFICIENCY_RESULTS.clear()
        reporter._NODEID_TO_CATEGORY.clear()
        reporter._CATEGORY_RESULTS.clear()

    def test_failed_test_appends_to_failures(self):
        reporter._NODEID_TO_CATEGORY["tests/evals/test_memory.py::test_recall"] = "memory"
        report = _FakeReport(
            nodeid="tests/evals/test_memory.py::test_recall",
            when="call",
            outcome="failed",
            duration=1.5,
            longreprtext="Expected 'TurboWidget' in final text, got 'unknown'",
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        assert len(reporter._FAILURES) == 1
        failure = reporter._FAILURES[0]
        assert failure["test_name"] == "tests/evals/test_memory.py::test_recall"
        assert failure["category"] == "memory"
        assert "TurboWidget" in failure["failure_message"]

    def test_passed_test_does_not_append(self):
        report = _FakeReport(
            nodeid="tests/evals/test_memory.py::test_ok",
            when="call",
            outcome="passed",
            duration=0.5,
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]
        assert reporter._FAILURES == []

    def test_skipped_test_does_not_append(self):
        report = _FakeReport(
            nodeid="tests/evals/test_memory.py::test_skip",
            when="call",
            outcome="skipped",
            duration=0.0,
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]
        assert reporter._FAILURES == []

    def test_setup_phase_ignored(self):
        report = _FakeReport(
            nodeid="tests/evals/test_memory.py::test_err",
            when="setup",
            outcome="failed",
            duration=0.0,
            longreprtext="fixture error",
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]
        assert reporter._FAILURES == []

    def test_missing_category_defaults_to_empty(self):
        report = _FakeReport(
            nodeid="tests/evals/test_misc.py::test_no_cat",
            when="call",
            outcome="failed",
            duration=1.0,
            longreprtext="some failure",
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        assert len(reporter._FAILURES) == 1
        assert reporter._FAILURES[0]["category"] == ""

    def test_multiple_failures_accumulate(self):
        for i in range(3):
            report = _FakeReport(
                nodeid=f"tests/evals/test_multi.py::test_{i}",
                when="call",
                outcome="failed",
                duration=1.0,
                longreprtext=f"failure {i}",
            )
            reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        assert len(reporter._FAILURES) == 3
        assert [f["failure_message"] for f in reporter._FAILURES] == [
            "failure 0",
            "failure 1",
            "failure 2",
        ]

    def test_long_failure_message_truncated(self):
        long_msg = "x" * (reporter._MAX_FAILURE_MSG_LEN + 1000)
        report = _FakeReport(
            nodeid="tests/evals/test_big.py::test_huge",
            when="call",
            outcome="failed",
            duration=1.0,
            longreprtext=long_msg,
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        assert len(reporter._FAILURES) == 1
        msg = reporter._FAILURES[0]["failure_message"]
        assert msg.endswith("... [truncated]")
        assert len(msg) < len(long_msg)


class _FakeTerminalReporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_sep(self, sep: str, title: str) -> None:
        _ = sep
        self.lines.append(title)

    def write_line(self, line: str) -> None:
        self.lines.append(line)


class _FakePluginManager:
    def __init__(self, terminal_reporter: _FakeTerminalReporter) -> None:
        self.terminal_reporter = terminal_reporter

    def getplugin(self, name: str) -> _FakeTerminalReporter | None:
        if name == "terminalreporter":
            return self.terminal_reporter
        return None


class _FakeConfig:
    def __init__(self, report_path: Path | None, terminal_reporter: _FakeTerminalReporter) -> None:
        self._report_path = report_path
        self._inicache: dict[str, str] = {}
        self.pluginmanager = _FakePluginManager(terminal_reporter)

    def getoption(self, name: str):
        if name == "--model":
            return "test-model"
        if name == "--evals-report-file":
            return str(self._report_path) if self._report_path is not None else None
        raise AssertionError(f"Unexpected option: {name}")


class _FakeSession:
    def __init__(self, config: _FakeConfig) -> None:
        self.config = config
        self.exitstatus = 0


class TestSessionSummary:
    def setup_method(self) -> None:
        reporter._RESULTS.update(passed=0, failed=0, skipped=0, total=0)
        reporter._DURATIONS_S.clear()
        reporter._EFFICIENCY_RESULTS.clear()
        reporter._NODEID_TO_CATEGORY.clear()
        reporter._CATEGORY_RESULTS.clear()
        reporter._EXPERIMENT_LINKS.clear()
        reporter._FAILURES.clear()

    def test_sessionfinish_writes_total_duration(self, tmp_path: Path) -> None:
        reporter._RESULTS.update(passed=2, failed=1, skipped=0, total=3)
        reporter._DURATIONS_S.extend([1.25, 2.5, 3.75])
        reporter._CATEGORY_RESULTS.update(
            {
                "memory": {"passed": 1, "failed": 1, "total": 2},
                "tool_use": {"passed": 1, "failed": 0, "total": 1},
            }
        )

        terminal_reporter = _FakeTerminalReporter()
        report_path = tmp_path / "summary.json"
        session = _FakeSession(_FakeConfig(report_path, terminal_reporter))

        reporter.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]

        payload = report_path.read_text(encoding="utf-8")
        assert '"median_duration_s": 2.5' in payload
        assert '"total_duration_s": 7.5' in payload
        assert any(line == "total_duration_s: 7.5000" for line in terminal_reporter.lines)
