"""Tests for the eval pytest reporter plugin."""

from __future__ import annotations

from dataclasses import dataclass

import tests.evals.pytest_reporter as reporter
import tests.evals.utils as evals_utils


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
        reporter._NODEID_TO_VARIANT.clear()
        reporter._CATEGORY_RESULTS.clear()
        reporter._VARIANT_RESULTS.clear()
        reporter._VARIANT_DURATIONS.clear()
        reporter._PER_TEST_VARIANT_RESULTS.clear()
        reporter._PER_TEST_VARIANT_METRICS.clear()

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


class TestVariantMetricsCapture:
    """Verify that per-test variant run metrics are collected by the reporter."""

    def setup_method(self):
        reporter._FAILURES.clear()
        reporter._RESULTS.update(passed=0, failed=0, skipped=0, total=0)
        reporter._DURATIONS_S.clear()
        reporter._EFFICIENCY_RESULTS.clear()
        reporter._NODEID_TO_CATEGORY.clear()
        reporter._NODEID_TO_VARIANT.clear()
        reporter._CATEGORY_RESULTS.clear()
        reporter._VARIANT_RESULTS.clear()
        reporter._VARIANT_DURATIONS.clear()
        reporter._PER_TEST_VARIANT_RESULTS.clear()
        reporter._PER_TEST_VARIANT_METRICS.clear()
        evals_utils._pending_test_run_metrics = None

    def test_metrics_captured_for_variant_test(self):
        nodeid = "tests/evals/tau2_airline/test_tau2_airline.py::test_tau2_airline[task_14-tool]"
        reporter._NODEID_TO_VARIANT[nodeid] = "tool"

        evals_utils.report_test_run_metrics(
            evals_utils.TestRunMetrics(
                turns=5,
                agent_steps=12,
                tool_calls=30,
                input_tokens=8000,
                output_tokens=2000,
            )
        )

        report = _FakeReport(nodeid=nodeid, when="call", outcome="passed", duration=42.0)
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        base = reporter._base_test_name(nodeid)
        assert base in reporter._PER_TEST_VARIANT_METRICS
        assert "tool" in reporter._PER_TEST_VARIANT_METRICS[base]
        metrics = reporter._PER_TEST_VARIANT_METRICS[base]["tool"][0]
        assert metrics.turns == 5
        assert metrics.agent_steps == 12
        assert metrics.tool_calls == 30
        assert metrics.input_tokens == 8000
        assert metrics.output_tokens == 2000
        assert evals_utils._pending_test_run_metrics is None

    def test_metrics_not_captured_without_variant(self):
        nodeid = "tests/evals/test_basic.py::test_simple"

        evals_utils.report_test_run_metrics(
            evals_utils.TestRunMetrics(turns=1, agent_steps=3, tool_calls=5)
        )

        report = _FakeReport(nodeid=nodeid, when="call", outcome="passed", duration=1.0)
        reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        assert reporter._PER_TEST_VARIANT_METRICS == {}
        assert evals_utils._pending_test_run_metrics is None

    def test_aggregate_metrics_averages_trials(self):
        metrics = [
            evals_utils.TestRunMetrics(
                turns=4, agent_steps=10, tool_calls=20, input_tokens=6000, output_tokens=1000
            ),
            evals_utils.TestRunMetrics(
                turns=6, agent_steps=14, tool_calls=30, input_tokens=10000, output_tokens=3000
            ),
        ]
        agg = reporter._aggregate_metrics(metrics)
        assert agg["avg_turns"] == 5.0
        assert agg["avg_agent_steps"] == 12.0
        assert agg["avg_tool_calls"] == 25.0
        assert agg["avg_input_tokens"] == 8000
        assert agg["avg_output_tokens"] == 2000

    def test_variant_summary_includes_metrics(self):
        base = "tests/evals/tau2_airline/test_tau2_airline.py::test_tau2_airline"
        for variant in ("tool", "prompt"):
            nodeid = f"{base}[task_14-{variant}]"
            reporter._NODEID_TO_VARIANT[nodeid] = variant

            evals_utils.report_test_run_metrics(
                evals_utils.TestRunMetrics(
                    turns=5,
                    agent_steps=10,
                    tool_calls=25,
                    input_tokens=7000,
                    output_tokens=1500,
                )
            )
            report = _FakeReport(nodeid=nodeid, when="call", outcome="passed", duration=30.0)
            reporter.pytest_runtest_logreport(report)  # type: ignore[arg-type]

        summary = reporter._build_variant_summary()
        assert "variant_scores" in summary
        for variant in ("tool", "prompt"):
            scores = summary["variant_scores"][variant]
            assert "avg_agent_steps" in scores
            assert scores["avg_agent_steps"] == 10.0
            assert scores["avg_tool_calls"] == 25.0

        per_test = summary["per_test_variants"]
        assert base in per_test
        for variant in ("tool", "prompt"):
            entry = per_test[base][variant]
            assert "metrics" in entry
            assert entry["metrics"]["avg_input_tokens"] == 7000
