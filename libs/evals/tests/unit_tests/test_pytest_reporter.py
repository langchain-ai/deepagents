"""Tests for the eval pytest reporter plugin — specifically the _FAILURES capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from _pytest.outcomes import Exit

import tests.evals.pytest_reporter as reporter

if TYPE_CHECKING:
    from _pytest.mark.structures import Mark


@dataclass
class _FakeReport:
    """Minimal stand-in for `pytest.TestReport`."""

    nodeid: str
    when: str
    outcome: str
    duration: float
    longreprtext: str = ""


@dataclass
class _FakeSession:
    """Minimal stand-in for `pytest.Session` for `sessionfinish` tests."""

    exitstatus: int
    config: Any = field(
        default_factory=lambda: SimpleNamespace(
            getoption=lambda *_args, **_kw: None,
            _inicache={},
            pluginmanager=SimpleNamespace(getplugin=lambda _name: None),
        )
    )


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


class TestSessionExitStatus:
    """Verify that pytest_sessionfinish preserves exit 1 when no tests ran."""

    def setup_method(self):
        reporter._FAILURES.clear()
        reporter._RESULTS.update(passed=0, failed=0, skipped=0, total=0)
        reporter._DURATIONS_S.clear()
        reporter._EFFICIENCY_RESULTS.clear()
        reporter._NODEID_TO_CATEGORY.clear()
        reporter._CATEGORY_RESULTS.clear()
        reporter._EXPERIMENT_LINKS.clear()

    def test_exit_1_swallowed_when_tests_ran(self):
        reporter._RESULTS.update(passed=2, failed=1, total=3)
        session = _FakeSession(exitstatus=1)
        reporter.pytest_sessionfinish(session, 1)  # type: ignore[arg-type]
        assert session.exitstatus == 0

    def test_exit_1_preserved_when_no_tests_ran(self):
        session = _FakeSession(exitstatus=1)
        reporter.pytest_sessionfinish(session, 1)  # type: ignore[arg-type]
        assert session.exitstatus == 1

    def test_exit_0_unchanged(self):
        reporter._RESULTS.update(passed=3, total=3)
        session = _FakeSession(exitstatus=0)
        reporter.pytest_sessionfinish(session, 0)  # type: ignore[arg-type]
        assert session.exitstatus == 0

    def test_exit_1_preserved_when_only_marked_skips(self):
        """Marked-skip tests (`@pytest.mark.skip`) emit only a `setup` phase
        report — pytest_runtest_logreport returns early and total stays 0. If
        an external exit 1 were to fire in this state, the reporter must not
        mask it.
        """
        report = _FakeReport(
            nodeid="tests/evals/test_x.py::test_y",
            when="setup",
            outcome="skipped",
            duration=0.0,
        )
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]
        assert reporter._RESULTS["total"] == 0

        session = _FakeSession(exitstatus=1)
        reporter.pytest_sessionfinish(session, 1)  # type: ignore[arg-type]
        assert session.exitstatus == 1

    @pytest.mark.parametrize("exitstatus", [2, 3, 4, 5])
    def test_non_1_exitstatus_unchanged(self, exitstatus: int):
        """Pytest exit codes 2 (interrupt), 3 (internal), 4 (usage), 5 (no
        tests collected) must pass through regardless of whether tests ran.
        """
        reporter._RESULTS.update(passed=3, total=3)
        session = _FakeSession(exitstatus=exitstatus)
        reporter.pytest_sessionfinish(session, exitstatus)  # type: ignore[arg-type]
        assert session.exitstatus == exitstatus


class _FakeItem:
    """Minimal stand-in for `pytest.Item` carrying a single `eval_category` mark."""

    def __init__(self, category: str) -> None:
        self._mark = pytest.mark.eval_category(category).mark

    def get_closest_marker(self, name: str) -> Mark | None:
        return self._mark if self._mark.name == name else None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
class TestFilterByMarker:
    """Integration check: unknown `--eval-category` values call `pytest.exit`.

    Pairs with TestSessionExitStatus to cover the full failure path — the
    filter raises `Exit(returncode=1)`, and the reporter preserves it.
    The `eval_category` mark is registered in the eval conftest, not here.
    """

    @staticmethod
    def _make_config(values: list[str]) -> SimpleNamespace:
        deselected: list[object] = []
        return SimpleNamespace(
            getoption=lambda _option: values,
            hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)),
        )

    def test_unknown_value_exits_with_code_1(self):
        from tests.evals.conftest import _filter_by_marker

        items = [_FakeItem("valid_cat")]
        config = self._make_config(["unknown_cat"])

        with pytest.raises(Exit) as exc_info:
            _filter_by_marker(
                config,  # ty: ignore[invalid-argument-type]
                items,  # ty: ignore[invalid-argument-type]
                option="--eval-category",
                marker_name="eval_category",
            )

        assert exc_info.value.returncode == 1
        msg = str(exc_info.value)
        assert "unknown_cat" in msg
        assert "valid_cat" in msg

    def test_known_value_does_not_exit(self):
        from tests.evals.conftest import _filter_by_marker

        items = [_FakeItem("valid_cat"), _FakeItem("other_cat")]
        config = self._make_config(["valid_cat"])

        _filter_by_marker(
            config,  # ty: ignore[invalid-argument-type]
            items,  # ty: ignore[invalid-argument-type]
            option="--eval-category",
            marker_name="eval_category",
        )
        assert len(items) == 1
        mark = items[0].get_closest_marker("eval_category")
        assert mark is not None
        assert mark.args == ("valid_cat",)

    def test_empty_option_is_noop(self):
        from tests.evals.conftest import _filter_by_marker

        items = [_FakeItem("valid_cat")]
        config = self._make_config([])

        _filter_by_marker(
            config,  # ty: ignore[invalid-argument-type]
            items,  # ty: ignore[invalid-argument-type]
            option="--eval-category",
            marker_name="eval_category",
        )
        assert len(items) == 1
