from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

_NODEID_TO_VARIANT: dict[str, str] = {}
"""Mapping of pytest node ID to its ``todo_mode`` parametrize value, built during collection."""

_VARIANT_RESULTS: dict[str, dict[str, int]] = {}
"""Per-variant pass/fail/total counters, keyed by todo_mode value (tool/prompt/filesystem)."""

_VARIANT_DURATIONS: dict[str, list[float]] = {}
"""Per-variant wall-clock durations."""

_PER_TEST_VARIANT_RESULTS: dict[str, dict[str, list[str]]] = {}
"""Per-test, per-variant outcomes. {base_test_name: {variant: [outcome, ...]}}."""

_PER_TEST_VARIANT_METRICS: dict[str, dict[str, list[_evals_utils.TestRunMetrics]]] = {}
"""Per-test, per-variant run metrics. {base_test_name: {variant: [metrics_per_trial]}}."""

_CATEGORY_RESULTS: dict[str, dict[str, int]] = {}
"""Per-category pass/fail/total counters, keyed by category name."""

_EXPERIMENT_LINKS: list[dict[str, str]] = []
"""LangSmith experiment link dicts with "name", "url", and optional "public_url" keys, collected at session teardown."""

_FAILURES: list[dict[str, str]] = []
"""Per-test failure details (`test_name`, `category`, `failure_message`) for post-run analysis."""

_MAX_FAILURE_MSG_LEN = 30_000
"""Truncate failure messages beyond this length (~7500 tokens) to stay within LLM context limits."""


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


def _langsmith_version() -> str:
    """Return the installed langsmith version, or "unknown" on failure."""
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("langsmith")
    except Exception:  # noqa: BLE001
        return "unknown"


def pytest_sessionstart(session: pytest.Session) -> None:
    """Pre-create the LangSmith experiment so the comparison URL is known upfront.

    Hydrates `_LangSmithTestSuite._instances` before any test runs. When the
    langsmith `@test` decorator calls `from_test` during the first test, it
    finds the pre-created instance and reuses it instead of creating a new
    experiment.

    This is the same private API that `_collect_experiment_links` already
    depends on.
    """
    test_suite_name = os.environ.get("LANGSMITH_TEST_SUITE")
    if not test_suite_name:
        return

    try:
        from langsmith import client as ls_client  # noqa: I001
        from langsmith.testing._internal import (
            _LangSmithTestSuite,
            _get_test_suite,
            _start_experiment,
        )
    except ImportError:
        return

    # Phase 1: create the experiment on the LangSmith server (irreversible).
    try:
        client = ls_client.Client()
        dataset = _get_test_suite(client, test_suite_name)

        model_opt = session.config.getoption("--model", default=None)
        model_name = model_opt or str(get_default_model().model)
        experiment_metadata = {
            "model": model_name,
            "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            "deepagents_version": __version__,
        }

        experiment = _start_experiment(client, dataset, experiment_metadata)
    except Exception as exc:  # noqa: BLE001
        msg = f"warning: could not create LangSmith experiment (langsmith=={_langsmith_version()}): {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201
        return

    # Phase 2: register the experiment locally so the @test decorator reuses it.
    suite_instance = None
    try:
        with _LangSmithTestSuite._lock:
            if not _LangSmithTestSuite._instances:
                _LangSmithTestSuite._instances = {}
            suite_instance = _LangSmithTestSuite(client, experiment, dataset, experiment_metadata)
            _LangSmithTestSuite._instances[test_suite_name] = suite_instance
    except Exception as exc:  # noqa: BLE001
        msg = f"warning: experiment created but could not register in _LangSmithTestSuite (langsmith=={_langsmith_version()}): {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201

    # Phase 3: build URLs and surface them in terminal output.
    try:
        dataset_url = getattr(dataset, "url", None)
        experiment_id = experiment.id
        if dataset_url and experiment_id:
            url = f"{dataset_url}/compare?selectedSessions={experiment_id}"
            public_url = (
                _get_public_experiment_url(suite_instance, experiment_id)
                if suite_instance is not None
                else None
            )
            _EXPERIMENT_LINKS.append(
                {
                    "name": experiment.name,
                    "url": url,
                    **({"public_url": public_url} if public_url else {}),
                }
            )
            terminal = session.config.pluginmanager.getplugin("terminalreporter")
            if terminal is not None:
                terminal.write_line(f"LangSmith experiment: {experiment.name}")
                if public_url:
                    terminal.write_line(f"  Public:   {public_url}")
                    terminal.write_line(f"  Internal: {url}")
                else:
                    terminal.write_line(f"  View results at: {url}")
                terminal.write_line("")
    except Exception as exc:  # noqa: BLE001
        msg = f"warning: experiment created but could not build URL (langsmith=={_langsmith_version()}): {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201


def _base_test_name(nodeid: str) -> str:
    """Strip parametrize brackets to get a stable base test name for grouping."""
    idx = nodeid.find("[")
    return nodeid[:idx] if idx != -1 else nodeid


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    for item in items:
        marker = item.get_closest_marker("eval_category")
        if marker and marker.args:
            _NODEID_TO_CATEGORY[item.nodeid] = str(marker.args[0])
        callspec = getattr(item, "callspec", None)
        if callspec and "todo_mode" in callspec.params:
            _NODEID_TO_VARIANT[item.nodeid] = str(callspec.params["todo_mode"])


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

    if outcome == "failed":
        msg = report.longreprtext
        if len(msg) > _MAX_FAILURE_MSG_LEN:
            msg = msg[:_MAX_FAILURE_MSG_LEN] + "\n\n... [truncated]"
        _FAILURES.append(
            {
                "test_name": report.nodeid,
                "category": _NODEID_TO_CATEGORY.get(report.nodeid, ""),
                "failure_message": msg,
            }
        )

    category = _NODEID_TO_CATEGORY.get(report.nodeid)
    if category and outcome in {"passed", "failed"}:
        bucket = _CATEGORY_RESULTS.setdefault(category, {"passed": 0, "failed": 0, "total": 0})
        bucket[outcome] += 1
        bucket["total"] += 1

    variant = _NODEID_TO_VARIANT.get(report.nodeid)
    if variant and outcome in {"passed", "failed"}:
        vbucket = _VARIANT_RESULTS.setdefault(variant, {"passed": 0, "failed": 0, "total": 0})
        vbucket[outcome] += 1
        vbucket["total"] += 1
        _VARIANT_DURATIONS.setdefault(variant, []).append(duration)
        base = _base_test_name(report.nodeid)
        _PER_TEST_VARIANT_RESULTS.setdefault(base, {}).setdefault(variant, []).append(outcome)

    # Consume per-test run metrics reported by the test function.
    pending = _evals_utils._pending_test_run_metrics
    if pending is not None:
        _evals_utils._pending_test_run_metrics = None
        if variant:
            base = _base_test_name(report.nodeid)
            _PER_TEST_VARIANT_METRICS.setdefault(base, {}).setdefault(variant, []).append(pending)

    if _EFFICIENCY_RESULTS and _EFFICIENCY_RESULTS[-1].duration_s is None:
        _EFFICIENCY_RESULTS[-1].duration_s = duration
        _EFFICIENCY_RESULTS[-1].passed = outcome == "passed"


def _get_public_experiment_url(suite: object, experiment_id: object) -> str | None:
    """Build the public comparison URL for an experiment.

    Uses `client.read_dataset_shared_schema` to obtain the public share URL,
    then appends the experiment ID as a comparison parameter. Returns `None`
    when the dataset is not shared or on any error.
    """
    try:
        client = getattr(suite, "client", None)
        dataset = getattr(suite, "_dataset", None)
        if client is None or dataset is None:
            return None
        dataset_id = getattr(dataset, "id", None)
        if dataset_id is None:
            return None
        share_schema = client.read_dataset_shared_schema(dataset_id=dataset_id)
        share_url = (
            share_schema.get("url")
            if isinstance(share_schema, dict)
            else getattr(share_schema, "url", None)
        )
        if share_url:
            return f"{share_url}/compare?selectedSessions={experiment_id}"
    except Exception as exc:  # noqa: BLE001
        msg = f"warning: could not resolve public URL for experiment: {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201
    return None


def _collect_experiment_links() -> list[dict[str, str]]:
    """Best-effort extraction of experiment name/URL pairs from langsmith internals.

    Accesses the private `_LangSmithTestSuite` API; returns an empty list on
    any failure.
    """
    try:
        from langsmith.testing._internal import _LangSmithTestSuite
    except ImportError:
        return []

    try:
        instances = _LangSmithTestSuite._instances
        if not instances:
            return []

        links: list[dict[str, str]] = []
        skipped = 0
        for suite in instances.values():
            dataset = getattr(suite, "_dataset", None)
            if dataset is None:
                skipped += 1
                continue
            dataset_url = getattr(dataset, "url", None)
            experiment_id = getattr(suite, "experiment_id", None)
            experiment = getattr(suite, "_experiment", None)
            name = getattr(experiment, "name", None) if experiment else None
            if dataset_url and experiment_id:
                url = f"{dataset_url}/compare?selectedSessions={experiment_id}"
                link: dict[str, str] = {"name": name or url, "url": url}
                public_url = _get_public_experiment_url(suite, experiment_id)
                if public_url:
                    link["public_url"] = public_url
                links.append(link)
            else:
                skipped += 1
        if skipped and not links:
            msg = f"warning: found {len(instances)} LangSmith test suite(s) but could not extract any experiment URLs"
            print(msg, file=sys.stderr)  # noqa: T201
    except Exception as exc:  # noqa: BLE001  # private API; best-effort
        try:
            from importlib.metadata import version as pkg_version

            ls_ver = pkg_version("langsmith")
        except Exception:  # noqa: BLE001
            ls_ver = "unknown"
        msg = f"warning: failed to collect experiment links (langsmith=={ls_ver}): {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201
        return []
    else:
        return links


def _aggregate_metrics(
    metrics_list: list[_evals_utils.TestRunMetrics],
) -> dict[str, float]:
    """Compute mean values across a list of ``TestRunMetrics``.

    Args:
        metrics_list: One or more metrics objects (one per trial).

    Returns:
        Dict with averaged metric values, rounded for readability.
    """
    n = len(metrics_list)
    if n == 0:
        return {}
    return {
        "avg_turns": round(sum(m.turns for m in metrics_list) / n, 1),
        "avg_agent_steps": round(sum(m.agent_steps for m in metrics_list) / n, 1),
        "avg_tool_calls": round(sum(m.tool_calls for m in metrics_list) / n, 1),
        "avg_input_tokens": round(sum(m.input_tokens for m in metrics_list) / n),
        "avg_output_tokens": round(sum(m.output_tokens for m in metrics_list) / n),
    }


def _build_variant_summary() -> dict[str, Any]:
    """Build structured variant-comparison data for the JSON report.

    Returns an empty dict when no variant-level data was collected (single-mode
    runs or tests that don't use the ``todo_mode`` fixture).
    """
    if not _VARIANT_RESULTS:
        return {}
    variant_scores: dict[str, Any] = {}
    for variant, counts in sorted(_VARIANT_RESULTS.items()):
        durations = _VARIANT_DURATIONS.get(variant, [])
        score_entry: dict[str, Any] = {
            **counts,
            "correctness": round(counts["passed"] / counts["total"], 2) if counts["total"] else 0.0,
            "median_duration_s": round(statistics.median(durations), 4) if durations else 0.0,
        }
        # Aggregate run-level metrics across all tests for this variant.
        all_metrics = [
            m
            for per_variant in _PER_TEST_VARIANT_METRICS.values()
            for m in per_variant.get(variant, [])
        ]
        if all_metrics:
            score_entry.update(_aggregate_metrics(all_metrics))
        variant_scores[variant] = score_entry

    per_test: dict[str, dict[str, Any]] = {}
    for base, variants in sorted(_PER_TEST_VARIANT_RESULTS.items()):
        per_test[base] = {}
        for v, outcomes in sorted(variants.items()):
            entry: dict[str, Any] = {
                "passed": outcomes.count("passed"),
                "total": len(outcomes),
            }
            metrics_list = _PER_TEST_VARIANT_METRICS.get(base, {}).get(v, [])
            if metrics_list:
                entry["metrics"] = _aggregate_metrics(metrics_list)
            per_test[base][v] = entry
    return {
        "variant_scores": variant_scores,
        "per_test_variants": per_test,
    }


def _render_variant_markdown(model: str, variant_data: dict[str, Any]) -> str:
    """Render a markdown comparison table from variant summary data."""
    variant_scores: dict[str, Any] = variant_data.get("variant_scores", {})
    per_test: dict[str, Any] = variant_data.get("per_test_variants", {})
    if not variant_scores:
        return ""

    lines: list[str] = []
    sorted_variants = sorted(variant_scores.keys())
    has_metrics = any("avg_agent_steps" in s for s in variant_scores.values())

    # High-level comparison table
    lines.append(f"### Todo Experiment: `{model}`")
    lines.append("")
    if has_metrics:
        header = "| Variant | Pass Rate | Passed | Failed | Total | Avg Turns | Avg Steps | Avg Tools | Avg In Tok | Avg Out Tok | Median Duration |"
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| Variant | Pass Rate | Passed | Failed | Total | Median Duration |"
        sep = "|---|---:|---:|---:|---:|---:|"
    lines.extend([header, sep])
    for v in sorted_variants:
        scores: dict[str, Any] = variant_scores[v]
        correctness = scores.get("correctness", 0)
        passed = scores.get("passed", 0)
        failed = scores.get("failed", 0)
        total = scores.get("total", 0)
        median_dur = scores.get("median_duration_s", 0)
        if has_metrics:
            avg_turns = scores.get("avg_turns", 0)
            avg_steps = scores.get("avg_agent_steps", 0)
            avg_tools = scores.get("avg_tool_calls", 0)
            avg_in = scores.get("avg_input_tokens", 0)
            avg_out = scores.get("avg_output_tokens", 0)
            lines.append(
                f"| `{v}` | {correctness:.2f} | {passed} | {failed} | {total}"
                f" | {avg_turns} | {avg_steps} | {avg_tools}"
                f" | {avg_in:,} | {avg_out:,} | {median_dur:.1f}s |"
            )
        else:
            lines.append(
                f"| `{v}` | {correctness:.2f} | {passed} | {failed} | {total} | {median_dur:.1f}s |"
            )

    # Per-test breakdown (compact pass/fail matrix)
    if per_test:
        lines.append("")
        lines.append("### Per-test breakdown")
        lines.append("")
        v_headers = " | ".join(f"`{v}`" for v in sorted_variants)
        header = f"| Test | {v_headers} |"
        sep = "|---|" + "|".join("---:" for _ in sorted_variants) + "|"
        lines.extend([header, sep])
        for base, variants in sorted(per_test.items()):
            short_name = base.rsplit("::", 1)[-1] if "::" in base else base
            cells = []
            for v in sorted_variants:
                vdata: dict[str, Any] = variants.get(v, {})
                p = vdata.get("passed", 0)
                t = vdata.get("total", 0)
                if t == 0:
                    cells.append("-")
                elif t == 1:
                    cells.append("pass" if p == t else "FAIL")
                else:
                    cells.append(f"{p}/{t}")
            lines.append(f"| `{short_name}` | {' | '.join(cells)} |")

    # Per-test detailed metrics table (one row per test x variant)
    any_test_metrics = any(
        "metrics" in vdata for variants in per_test.values() for vdata in variants.values()
    )
    if per_test and any_test_metrics:
        lines.append("")
        lines.append("### Per-test detailed metrics")
        lines.append("")
        lines.append("| Test | Variant | Result | Turns | Steps | Tools | In Tokens | Out Tokens |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for base, variants in sorted(per_test.items()):
            short_name = base.rsplit("::", 1)[-1] if "::" in base else base
            for v in sorted_variants:
                vdata: dict[str, Any] = variants.get(v, {})
                p = vdata.get("passed", 0)
                t = vdata.get("total", 0)
                if t == 0:
                    result = "-"
                elif t == 1:
                    result = "pass" if p == t else "FAIL"
                else:
                    result = f"{p}/{t}"
                m: dict[str, Any] = vdata.get("metrics", {})
                turns = m.get("avg_turns", "-")
                steps = m.get("avg_agent_steps", "-")
                tools = m.get("avg_tool_calls", "-")
                in_tok = m.get("avg_input_tokens", "-")
                out_tok = m.get("avg_output_tokens", "-")
                in_tok_fmt = f"{in_tok:,}" if isinstance(in_tok, int | float) else in_tok
                out_tok_fmt = f"{out_tok:,}" if isinstance(out_tok, int | float) else out_tok
                lines.append(
                    f"| `{short_name}` | `{v}` | {result}"
                    f" | {turns} | {steps} | {tools} | {in_tok_fmt} | {out_tok_fmt} |"
                )

    return "\n".join(lines) + "\n"


def _fmt_tokens(value: float | None) -> str:
    """Format a token count with thousands separators, or ``-`` if absent."""
    if value is None:
        return "-"
    return f"{int(value):,}"


def _render_terminal_detail_table(
    terminal: object,
    sorted_variants: list[str],
) -> None:
    """Write a detailed per-test x variant metrics table to the terminal.

    Args:
        terminal: The pytest terminal reporter plugin.
        sorted_variants: Alphabetically sorted variant names.
    """
    terminal.write_sep("-", "per-test variant details")  # type: ignore[union-attr]
    terminal.write_line(  # type: ignore[union-attr]
        f"  {'test':<35} {'variant':<12} {'result':>6}"
        f"  {'turns':>6} {'steps':>6} {'tools':>6}"
        f"  {'in_tok':>10} {'out_tok':>10}"
    )
    terminal.write_line(  # type: ignore[union-attr]
        f"  {'-' * 35} {'-' * 12} {'-' * 6}  {'-' * 6} {'-' * 6} {'-' * 6}  {'-' * 10} {'-' * 10}"
    )
    for base in sorted(_PER_TEST_VARIANT_RESULTS):
        short = base.rsplit("::", 1)[-1] if "::" in base else base
        for v in sorted_variants:
            outcomes = _PER_TEST_VARIANT_RESULTS[base].get(v, [])
            if not outcomes:
                continue
            if len(outcomes) == 1:
                result = "PASS" if outcomes[0] == "passed" else "FAIL"
            else:
                p = outcomes.count("passed")
                result = f"{p}/{len(outcomes)}"
            metrics_list = _PER_TEST_VARIANT_METRICS.get(base, {}).get(v, [])
            if metrics_list:
                agg = _aggregate_metrics(metrics_list)
                terminal.write_line(  # type: ignore[union-attr]
                    f"  {short:<35} {v:<12} {result:>6}"
                    f"  {agg.get('avg_turns', '-'):>6}"
                    f" {agg.get('avg_agent_steps', '-'):>6}"
                    f" {agg.get('avg_tool_calls', '-'):>6}"
                    f"  {_fmt_tokens(agg.get('avg_input_tokens')):>10}"
                    f" {_fmt_tokens(agg.get('avg_output_tokens')):>10}"
                )
            else:
                terminal.write_line(  # type: ignore[union-attr]
                    f"  {short:<35} {v:<12} {result:>6}"
                    f"  {'-':>6} {'-':>6} {'-':>6}"
                    f"  {'-':>10} {'-':>10}"
                )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _ = exitstatus
    if session.exitstatus == 1:
        session.exitstatus = 0

    if not _EXPERIMENT_LINKS:
        _EXPERIMENT_LINKS.extend(_collect_experiment_links())

    correctness = round((_RESULTS["passed"] / _RESULTS["total"]) if _RESULTS["total"] else 0.0, 2)
    step_ratio = _micro_step_ratio()
    tool_call_ratio = _micro_tool_call_ratio()
    solve_rate = _solve_rate()
    median_duration_s = round(statistics.median(_DURATIONS_S), 4) if _DURATIONS_S else 0.0

    category_scores: dict[str, float] = {}
    for cat, counts in sorted(_CATEGORY_RESULTS.items()):
        if counts["total"] > 0:
            category_scores[cat] = round(counts["passed"] / counts["total"], 2)

    variant_data = _build_variant_summary()

    model = (
        session.config.getoption("--model")
        or str(session.config._inicache.get("model", ""))
        or str(get_default_model().model)
    )

    payload: dict[str, object] = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sdk_version": __version__,
        "model": model,
        **_RESULTS,
        "correctness": correctness,
        "category_scores": category_scores,
        "step_ratio": step_ratio,
        "tool_call_ratio": tool_call_ratio,
        "solve_rate": solve_rate,
        "median_duration_s": median_duration_s,
        "experiment_urls": [link["url"] for link in _EXPERIMENT_LINKS],
        "experiment_links": _EXPERIMENT_LINKS,
        "failures": _FAILURES,
    }
    if variant_data:
        payload["variant_results"] = variant_data

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
        if len(category_scores) > 1:
            terminal_reporter.write_sep("-", "per-category correctness")
            for cat, score in sorted(category_scores.items()):
                counts = _CATEGORY_RESULTS[cat]
                terminal_reporter.write_line(
                    f"  {cat}: {score:.2f} ({counts['passed']}/{counts['total']})"
                )
        if _VARIANT_RESULTS:
            sorted_variants = sorted(_VARIANT_RESULTS.keys())
            has_run_metrics = bool(_PER_TEST_VARIANT_METRICS)
            terminal_reporter.write_sep("-", "per-variant summary")
            if has_run_metrics:
                terminal_reporter.write_line(
                    f"  {'variant':<14} {'pass':>6} {'rate':>6}"
                    f"  {'turns':>6} {'steps':>6} {'tools':>6}"
                    f"  {'in_tok':>10} {'out_tok':>10}  {'dur(s)':>8}"
                )
                terminal_reporter.write_line(
                    f"  {'-' * 14} {'-' * 6} {'-' * 6}"
                    f"  {'-' * 6} {'-' * 6} {'-' * 6}"
                    f"  {'-' * 10} {'-' * 10}  {'-' * 8}"
                )
                for v in sorted_variants:
                    counts = _VARIANT_RESULTS[v]
                    c = round(counts["passed"] / counts["total"], 2) if counts["total"] else 0.0
                    dur = _VARIANT_DURATIONS.get(v, [])
                    med_dur = round(statistics.median(dur), 1) if dur else 0.0
                    all_m = [
                        m for per_v in _PER_TEST_VARIANT_METRICS.values() for m in per_v.get(v, [])
                    ]
                    agg = _aggregate_metrics(all_m) if all_m else {}
                    terminal_reporter.write_line(
                        f"  {v:<14} {counts['passed']:>3}/{counts['total']:<3}"
                        f" {c:>5.2f}"
                        f"  {agg.get('avg_turns', '-'):>6}"
                        f" {agg.get('avg_agent_steps', '-'):>6}"
                        f" {agg.get('avg_tool_calls', '-'):>6}"
                        f"  {_fmt_tokens(agg.get('avg_input_tokens')):>10}"
                        f" {_fmt_tokens(agg.get('avg_output_tokens')):>10}"
                        f"  {med_dur:>7.1f}s"
                    )
            else:
                for v in sorted_variants:
                    counts = _VARIANT_RESULTS[v]
                    c = round(counts["passed"] / counts["total"], 2) if counts["total"] else 0.0
                    terminal_reporter.write_line(
                        f"  {v}: {c:.2f} ({counts['passed']}/{counts['total']})"
                    )
            if _PER_TEST_VARIANT_RESULTS:
                terminal_reporter.write_sep("-", "per-test variant matrix")
                v_cols = "  ".join(f"{v:>12}" for v in sorted_variants)
                terminal_reporter.write_line(f"  {'test':<55} {v_cols}")
                terminal_reporter.write_line(
                    f"  {'-' * 55} {'  '.join('-' * 12 for _ in sorted_variants)}"
                )
                for base in sorted(_PER_TEST_VARIANT_RESULTS):
                    short = base.rsplit("::", 1)[-1] if "::" in base else base
                    cells = []
                    for v in sorted_variants:
                        outcomes = _PER_TEST_VARIANT_RESULTS[base].get(v, [])
                        if not outcomes:
                            cells.append("-")
                        elif len(outcomes) == 1:
                            cells.append("PASS" if outcomes[0] == "passed" else "FAIL")
                        else:
                            p = outcomes.count("passed")
                            cells.append(f"{p}/{len(outcomes)}")
                    row = "  ".join(f"{c:>12}" for c in cells)
                    terminal_reporter.write_line(f"  {short:<55} {row}")
            if _PER_TEST_VARIANT_METRICS:
                _render_terminal_detail_table(terminal_reporter, sorted_variants)
        if step_ratio is not None:
            terminal_reporter.write_line(f"step_ratio: {step_ratio:.2f}")
        if tool_call_ratio is not None:
            terminal_reporter.write_line(f"tool_call_ratio: {tool_call_ratio:.2f}")
        if solve_rate is not None:
            terminal_reporter.write_line(f"solve_rate: {solve_rate:.4f}")
        terminal_reporter.write_line(f"median_duration_s: {median_duration_s:.4f}")
        if _EXPERIMENT_LINKS:
            terminal_reporter.write_sep("-", "langsmith experiments")
            for link in _EXPERIMENT_LINKS:
                public_url = link.get("public_url")
                if public_url:
                    terminal_reporter.write_line(f"  {link['name']}:")
                    terminal_reporter.write_line(f"    Public:   {public_url}")
                    terminal_reporter.write_line(f"    Internal: {link['url']}")
                else:
                    terminal_reporter.write_line(f"  {link['name']}: {link['url']}")

    report_path_opt = session.config.getoption("--evals-report-file")
    if not report_path_opt:
        return

    report_path = Path(str(report_path_opt))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if variant_data:
        md = _render_variant_markdown(str(model), variant_data)
        summary_md_path = report_path.with_name("variant_summary.md")
        summary_md_path.write_text(md, encoding="utf-8")
