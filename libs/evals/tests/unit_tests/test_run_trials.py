"""Tests for the multi-trial eval runner aggregator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_trials.py"
_MODULE_NAME = "_run_trials_under_test"


def _load_run_trials() -> ModuleType:
    """Import scripts/run_trials.py as a module without polluting sys.path.

    Registers the module in `sys.modules` before `exec_module` so dataclass
    forward-reference resolution (which looks the module up by name) works on
    Python 3.14+.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT)
    if spec is None or spec.loader is None:
        msg = f"could not load spec for {_SCRIPT}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


run_trials = _load_run_trials()


def _report(
    *,
    correctness: float,
    solve_rate: float | None,
    step_ratio: float | None,
    tool_call_ratio: float | None,
    median_duration_s: float,
    passed: int,
    failed: int,
    total: int,
    category_scores: dict[str, float] | None = None,
    skipped: int = 0,
) -> dict[str, Any]:
    """Build a fake per-trial report matching the pytest reporter schema."""
    return {
        "model": "openai:gpt-5.5",
        "sdk_version": "0.5.6",
        "created_at": "2026-05-04T00:00:00+00:00",
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "correctness": correctness,
        "solve_rate": solve_rate,
        "step_ratio": step_ratio,
        "tool_call_ratio": tool_call_ratio,
        "median_duration_s": median_duration_s,
        "category_scores": category_scores or {},
        "experiment_urls": [],
    }


class TestAggregateTrials:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one report"):
            run_trials.aggregate_trials([])

    def test_single_trial_has_no_stdev(self) -> None:
        summary = run_trials.aggregate_trials(
            [
                _report(
                    correctness=0.5,
                    solve_rate=0.2,
                    step_ratio=0.8,
                    tool_call_ratio=0.6,
                    median_duration_s=10.0,
                    passed=80,
                    failed=80,
                    total=160,
                ),
            ]
        )
        assert summary["n_trials"] == 1
        assert summary["metrics"]["correctness"]["mean"] == pytest.approx(0.5)
        assert summary["metrics"]["correctness"]["stdev"] is None
        assert summary["metrics"]["correctness"]["min"] == pytest.approx(0.5)
        assert summary["metrics"]["correctness"]["max"] == pytest.approx(0.5)

    def test_multi_trial_stats_match_statistics_module(self) -> None:
        correctness_values = [0.47, 0.49, 0.51]
        reports = [
            _report(
                correctness=c,
                solve_rate=0.20 + i * 0.01,
                step_ratio=0.80 + i * 0.01,
                tool_call_ratio=0.50 + i * 0.05,
                median_duration_s=8.0 + i,
                passed=80 + i,
                failed=80 - i,
                total=160,
            )
            for i, c in enumerate(correctness_values)
        ]
        summary = run_trials.aggregate_trials(reports)

        c_stats = summary["metrics"]["correctness"]
        assert c_stats["n"] == 3
        assert c_stats["mean"] == pytest.approx(statistics.mean(correctness_values))
        assert c_stats["median"] == pytest.approx(statistics.median(correctness_values))
        assert c_stats["stdev"] == pytest.approx(statistics.stdev(correctness_values))
        assert c_stats["min"] == pytest.approx(min(correctness_values))
        assert c_stats["max"] == pytest.approx(max(correctness_values))

        passed_stats = summary["counts"]["passed"]
        assert passed_stats["mean"] == pytest.approx(81.0)
        assert passed_stats["min"] == 80
        assert passed_stats["max"] == 82

    def test_null_metric_values_are_skipped(self) -> None:
        reports = [
            _report(
                correctness=0.5,
                solve_rate=None,
                step_ratio=0.8,
                tool_call_ratio=None,
                median_duration_s=10.0,
                passed=80,
                failed=80,
                total=160,
            ),
            _report(
                correctness=0.6,
                solve_rate=0.25,
                step_ratio=None,
                tool_call_ratio=0.7,
                median_duration_s=12.0,
                passed=90,
                failed=70,
                total=160,
            ),
        ]
        summary = run_trials.aggregate_trials(reports)
        assert summary["metrics"]["solve_rate"]["n"] == 1
        assert summary["metrics"]["solve_rate"]["mean"] == pytest.approx(0.25)
        assert summary["metrics"]["solve_rate"]["stdev"] is None
        assert summary["metrics"]["step_ratio"]["n"] == 1
        assert summary["metrics"]["tool_call_ratio"]["n"] == 1
        assert summary["metrics"]["correctness"]["n"] == 2

    def test_category_scores_aggregated_across_trials(self) -> None:
        reports = [
            _report(
                correctness=0.5,
                solve_rate=0.2,
                step_ratio=0.8,
                tool_call_ratio=0.6,
                median_duration_s=10.0,
                passed=80,
                failed=80,
                total=160,
                category_scores={"memory": 0.6, "tool_use": 0.2},
            ),
            _report(
                correctness=0.55,
                solve_rate=0.21,
                step_ratio=0.82,
                tool_call_ratio=0.65,
                median_duration_s=11.0,
                passed=85,
                failed=75,
                total=160,
                category_scores={"memory": 0.7, "tool_use": 0.18, "retrieval": 1.0},
            ),
        ]
        summary = run_trials.aggregate_trials(reports)
        cats = summary["category_scores"]
        assert cats["memory"]["n"] == 2
        assert cats["memory"]["mean"] == pytest.approx(0.65)
        # `retrieval` only appears in one trial; n=1, stdev=None
        assert cats["retrieval"]["n"] == 1
        assert cats["retrieval"]["stdev"] is None

    def test_per_trial_records_preserved_in_order(self) -> None:
        reports = [
            _report(
                correctness=0.4 + i * 0.05,
                solve_rate=0.2,
                step_ratio=0.8,
                tool_call_ratio=0.5,
                median_duration_s=10.0,
                passed=10 * i,
                failed=0,
                total=10 * i,
            )
            for i in range(3)
        ]
        summary = run_trials.aggregate_trials(reports)
        assert [t["trial_index"] for t in summary["trials"]] == [0, 1, 2]
        assert [t["correctness"] for t in summary["trials"]] == pytest.approx([0.4, 0.45, 0.5])

    def test_non_numeric_metric_value_is_excluded_with_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reports = [
            _report(
                correctness=0.5,
                solve_rate=0.2,
                step_ratio=0.8,
                tool_call_ratio=0.6,
                median_duration_s=10.0,
                passed=80,
                failed=80,
                total=160,
            ),
        ]
        # Bad upstream schema: correctness arrived as a string.
        reports[0]["correctness"] = "0.5"
        summary = run_trials.aggregate_trials(reports)
        assert summary["metrics"]["correctness"]["n"] == 0
        captured = capsys.readouterr()
        assert "non-numeric value for 'correctness'" in captured.err

    def test_bool_values_are_not_aggregated_as_ints(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reports = [
            _report(
                correctness=0.5,
                solve_rate=0.2,
                step_ratio=0.8,
                tool_call_ratio=0.6,
                median_duration_s=10.0,
                passed=80,
                failed=80,
                total=160,
            ),
        ]
        # `True` is `isinstance(_, int)` but should not be aggregated as 1.
        reports[0]["passed"] = True
        summary = run_trials.aggregate_trials(reports)
        assert summary["counts"]["passed"]["n"] == 0
        assert "non-numeric value for 'passed'" in capsys.readouterr().err

    def test_divergent_model_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        a = _report(
            correctness=0.5,
            solve_rate=0.2,
            step_ratio=0.8,
            tool_call_ratio=0.6,
            median_duration_s=10.0,
            passed=80,
            failed=80,
            total=160,
        )
        b = _report(
            correctness=0.6,
            solve_rate=0.2,
            step_ratio=0.8,
            tool_call_ratio=0.6,
            median_duration_s=10.0,
            passed=80,
            failed=80,
            total=160,
        )
        b["model"] = "anthropic:claude-4.6"
        summary = run_trials.aggregate_trials([a, b])
        assert summary["model"] == "openai:gpt-5.5"
        assert "disagree on `model`" in capsys.readouterr().err

    def test_pytest_returncode_passes_through_to_per_trial(self) -> None:
        r = _report(
            correctness=0.5,
            solve_rate=0.2,
            step_ratio=0.8,
            tool_call_ratio=0.6,
            median_duration_s=10.0,
            passed=80,
            failed=80,
            total=160,
        )
        r["pytest_returncode"] = 2
        summary = run_trials.aggregate_trials([r])
        assert summary["trials"][0]["pytest_returncode"] == 2


class TestSummarize:
    def test_empty_input_returns_all_none(self) -> None:
        stats = run_trials._summarize([]).to_dict()
        assert stats == {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }

    def test_single_value_has_no_stdev(self) -> None:
        stats = run_trials._summarize([0.5]).to_dict()
        assert stats["n"] == 1
        assert stats["stdev"] is None
        assert stats["mean"] == pytest.approx(0.5)


def _make_args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "model": "openai:gpt-5.5",
        "trials": 1,
        "eval_category": [],
        "eval_tier": [],
        "openai_reasoning_effort": None,
        "openrouter_provider": None,
        "openrouter_allow_fallbacks": False,
        "repl": None,
        "out_dir": Path("/tmp/out"),
        "pytest_extra": [],
        "aggregate_only": None,
        "summary_out": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestBuildPytestArgs:
    def test_minimal_args(self) -> None:
        cmd = run_trials._build_pytest_args(_make_args(), Path("/tmp/r.json"))
        assert cmd[:6] == ["uv", "run", "--group", "test", "pytest", "tests/evals"]
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "openai:gpt-5.5"
        assert "--evals-report-file" in cmd
        assert cmd[cmd.index("--evals-report-file") + 1] == "/tmp/r.json"

    def test_repeated_categories_and_tiers(self) -> None:
        args = _make_args(eval_category=["memory", "tool_use"], eval_tier=["baseline"])
        cmd = run_trials._build_pytest_args(args, Path("/tmp/r.json"))
        # Each repeated value should show up as its own --eval-category flag.
        assert cmd.count("--eval-category") == 2
        assert "memory" in cmd
        assert "tool_use" in cmd
        assert cmd.count("--eval-tier") == 1
        assert "baseline" in cmd

    def test_optional_flags_pass_through(self) -> None:
        args = _make_args(
            openai_reasoning_effort="medium",
            openrouter_provider="MiniMax",
            repl="quickjs",
        )
        cmd = run_trials._build_pytest_args(args, Path("/tmp/r.json"))
        assert "--openai-reasoning-effort" in cmd
        assert cmd[cmd.index("--openai-reasoning-effort") + 1] == "medium"
        assert "--openrouter-provider" in cmd
        assert cmd[cmd.index("--openrouter-provider") + 1] == "MiniMax"
        assert "--openrouter-allow-fallbacks" not in cmd
        assert "--repl" in cmd
        assert cmd[cmd.index("--repl") + 1] == "quickjs"

    def test_openrouter_provider_accepts_comma_separated_allowlist(self) -> None:
        args = _make_args(openrouter_provider="MiniMax,Fireworks")
        cmd = run_trials._build_pytest_args(args, Path("/tmp/r.json"))
        # Pytest does the parsing; the script just forwards the string verbatim.
        assert cmd[cmd.index("--openrouter-provider") + 1] == "MiniMax,Fireworks"

    def test_openrouter_allow_fallbacks_passed_as_bare_flag(self) -> None:
        args = _make_args(
            openrouter_provider="MiniMax,Fireworks",
            openrouter_allow_fallbacks=True,
        )
        cmd = run_trials._build_pytest_args(args, Path("/tmp/r.json"))
        assert "--openrouter-allow-fallbacks" in cmd

    def test_pytest_extra_forwarded(self) -> None:
        args = _make_args(pytest_extra=["-k", "smoke"])
        cmd = run_trials._build_pytest_args(args, Path("/tmp/r.json"))
        assert cmd[-2:] == ["-k", "smoke"]


class TestParseArgs:
    def test_requires_model_when_not_aggregate_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            run_trials._parse_args(["--trials", "3"])
        assert "--model is required" in capsys.readouterr().err

    def test_requires_trials_when_not_aggregate_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit):
            run_trials._parse_args(["--model", "openai:gpt-5.5"])
        assert "--trials is required" in capsys.readouterr().err

    def test_rejects_trials_below_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            run_trials._parse_args(["--model", "openai:gpt-5.5", "--trials", "0"])
        assert "between 1 and" in capsys.readouterr().err

    def test_rejects_trials_above_max(self, capsys: pytest.CaptureFixture[str]) -> None:
        too_many = run_trials._MAX_TRIALS + 1
        with pytest.raises(SystemExit):
            run_trials._parse_args(["--model", "openai:gpt-5.5", "--trials", str(too_many)])
        assert "between 1 and" in capsys.readouterr().err

    def test_aggregate_only_skips_model_and_trials_validation(self, tmp_path: Path) -> None:
        args = run_trials._parse_args(["--aggregate-only", str(tmp_path)])
        assert args.aggregate_only == tmp_path
        assert args.model is None
        assert args.trials is None

    def test_strips_leading_double_dash_from_pytest_extra(self) -> None:
        args = run_trials._parse_args(
            ["--model", "openai:gpt-5.5", "--trials", "1", "--", "-k", "smoke"]
        )
        assert args.pytest_extra == ["-k", "smoke"]


class TestDiscoverReports:
    def test_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        assert run_trials._discover_reports(tmp_path / "does_not_exist") == []

    def test_finds_per_trial_files(self, tmp_path: Path) -> None:
        (tmp_path / "evals_report_trial_000.json").write_text("{}")
        (tmp_path / "evals_report_trial_001.json").write_text("{}")
        found = run_trials._discover_reports(tmp_path)
        assert [p.name for p in found] == [
            "evals_report_trial_000.json",
            "evals_report_trial_001.json",
        ]

    def test_finds_ci_artifact_layout(self, tmp_path: Path) -> None:
        # `_eval.yml` writes `evals_report.json` inside each artifact dir.
        for i in range(2):
            d = tmp_path / f"evals-report-trial-{i:03d}-slug"
            d.mkdir()
            (d / "evals_report.json").write_text("{}")
        found = run_trials._discover_reports(tmp_path)
        assert len(found) == 2
        assert all(p.name == "evals_report.json" for p in found)

    def test_dedupes_when_a_file_matches_both_patterns(self, tmp_path: Path) -> None:
        # Implausible but valid: a file matching both globs should appear once.
        (tmp_path / "evals_report.json").write_text("{}")
        (tmp_path / "evals_report_trial_000.json").write_text("{}")
        found = run_trials._discover_reports(tmp_path)
        assert len(found) == 2

    def test_returns_sorted(self, tmp_path: Path) -> None:
        for name in ("evals_report_trial_002.json", "evals_report_trial_000.json"):
            (tmp_path / name).write_text("{}")
        found = run_trials._discover_reports(tmp_path)
        assert [p.name for p in found] == sorted(p.name for p in found)


class TestLoadReport:
    def test_returns_none_for_missing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run_trials._load_report(tmp_path / "nope.json") is None
        assert "could not read" in capsys.readouterr().err

    def test_returns_none_for_invalid_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json{")
        assert run_trials._load_report(path) is None
        assert "could not read" in capsys.readouterr().err

    def test_returns_none_for_non_dict_top_level(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        assert run_trials._load_report(path) is None
        assert "not a JSON object" in capsys.readouterr().err

    def test_returns_dict_on_success(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.json"
        path.write_text('{"a": 1}')
        assert run_trials._load_report(path) == {"a": 1}


class TestMainAggregateOnly:
    def test_writes_summary_from_existing_reports(self, tmp_path: Path) -> None:
        report = _report(
            correctness=0.5,
            solve_rate=0.2,
            step_ratio=0.8,
            tool_call_ratio=0.6,
            median_duration_s=10.0,
            passed=80,
            failed=80,
            total=160,
        )
        (tmp_path / "evals_report_trial_000.json").write_text(json.dumps(report))
        summary_out = tmp_path / "trials_summary.json"

        rc = run_trials.main(["--aggregate-only", str(tmp_path), "--summary-out", str(summary_out)])
        assert rc == 0
        assert summary_out.is_file()
        summary = json.loads(summary_out.read_text())
        assert summary["n_trials"] == 1
        assert summary["metrics"]["correctness"]["mean"] == pytest.approx(0.5)

    def test_returns_1_when_dir_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_trials.main(["--aggregate-only", str(tmp_path)])
        assert rc == 1
        assert "no eval report JSON files found" in capsys.readouterr().err

    def test_returns_1_when_all_reports_unreadable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "evals_report_trial_000.json").write_text("not json")
        rc = run_trials.main(["--aggregate-only", str(tmp_path)])
        assert rc == 1
        captured = capsys.readouterr()
        assert "no readable trial reports found" in captured.err

    def test_default_summary_path_is_under_aggregate_dir(self, tmp_path: Path) -> None:
        report = _report(
            correctness=0.5,
            solve_rate=0.2,
            step_ratio=0.8,
            tool_call_ratio=0.6,
            median_duration_s=10.0,
            passed=80,
            failed=80,
            total=160,
        )
        (tmp_path / "evals_report_trial_000.json").write_text(json.dumps(report))
        rc = run_trials.main(["--aggregate-only", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "trials_summary.json").is_file()


class TestMain:
    def test_returns_1_when_no_trial_produces_a_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail(**_: object) -> object:
            return run_trials._TrialOutcome(report_path=None, returncode=1)

        monkeypatch.setattr(run_trials, "_run_trial", fail)
        rc = run_trials.main(
            [
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "2",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert rc == 1
        assert "no trial produced a report" in capsys.readouterr().err

    def test_main_aggregates_when_run_trial_succeeds(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_run_trial(
            *,
            trial_index: int,
            n_trials: int,  # noqa: ARG001
            args: argparse.Namespace,  # noqa: ARG001
            out_dir: Path,
        ) -> object:
            path = out_dir / f"evals_report_trial_{trial_index:03d}.json"
            path.write_text(
                json.dumps(
                    _report(
                        correctness=0.5 + 0.1 * trial_index,
                        solve_rate=0.2,
                        step_ratio=0.8,
                        tool_call_ratio=0.6,
                        median_duration_s=10.0,
                        passed=80,
                        failed=80,
                        total=160,
                    )
                )
            )
            return run_trials._TrialOutcome(report_path=path, returncode=0)

        monkeypatch.setattr(run_trials, "_run_trial", fake_run_trial)
        rc = run_trials.main(
            [
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "2",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        summary = json.loads((tmp_path / "trials_summary.json").read_text())
        assert summary["n_trials"] == 2
        # Returncode passthrough on the live-execution path.
        assert summary["trials"][0]["pytest_returncode"] == 0
