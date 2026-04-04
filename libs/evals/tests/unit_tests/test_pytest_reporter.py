"""Tests for the eval pytest reporter plugin — specifically the _FAILURES capture."""

from __future__ import annotations

from dataclasses import dataclass

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
