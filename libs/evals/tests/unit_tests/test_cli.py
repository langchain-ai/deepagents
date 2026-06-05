"""Tests for the unified `deepagents-evals` CLI."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from deepagents_evals import cli

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts without the default-model env var leaking from the host.

    `monkeypatch` handles teardown automatically; no `yield` needed.
    """
    monkeypatch.delenv(cli._MODEL_ENV_VAR, raising=False)


class TestListSubcommand:
    def test_categories_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "categories"])
        out = capsys.readouterr().out
        assert rc == cli.EXIT_OK
        assert "memory" in out
        assert "tool_use" in out

    def test_categories_json_is_valid(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "categories", "--json"])
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert "memory" in payload
        assert payload == sorted(payload) or "memory" in payload

    def test_tiers_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "tiers"])
        out = capsys.readouterr().out.splitlines()
        assert rc == cli.EXIT_OK
        assert "baseline" in out
        assert "hillclimb" in out

    def test_evals_filtered_by_category(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "evals", "--category", "memory", "--json"])
        assert rc == cli.EXIT_OK
        evals = json.loads(capsys.readouterr().out)
        assert isinstance(evals, list)
        assert evals, "expected at least one memory eval"
        assert all(e["category"] == "memory" for e in evals)

    def test_models_lists_eval_tagged(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "models", "--json"])
        assert rc == cli.EXIT_OK
        models = json.loads(capsys.readouterr().out)
        assert isinstance(models, list)
        assert any(m["spec"].startswith("anthropic:") for m in models)
        assert all("groups" in m for m in models)

    def test_models_filtered_by_provider(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(["list", "models", "--provider", "anthropic", "--json"])
        assert rc == cli.EXIT_OK
        models = json.loads(capsys.readouterr().out)
        assert models
        assert all(m["spec"].startswith("anthropic:") for m in models)


class TestRunSubcommand:
    def test_dry_run_prints_argv(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(
            [
                "run",
                "--model",
                "openai:gpt-5.5",
                "--eval-category",
                "memory",
                "--eval-tier",
                "baseline",
                "--report",
                "/tmp/x.json",
                "--dry-run",
                "--json",
            ]
        )
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        argv = payload["argv"]
        assert "pytest" in argv
        assert "tests/evals" in argv
        assert "--model" in argv
        assert "openai:gpt-5.5" in argv
        assert "--eval-category" in argv
        assert "memory" in argv
        assert "--eval-tier" in argv
        assert "baseline" in argv
        assert "--evals-report-file" in argv

    def test_missing_model_is_config_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        # `parser.exit` raises SystemExit with the configured code.
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["run", "--dry-run"])
        assert excinfo.value.code == cli.EXIT_CONFIG
        err = capsys.readouterr().err
        assert "--model is required" in err
        assert cli._MODEL_ENV_VAR in err

    def test_env_var_supplies_model(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(cli._MODEL_ENV_VAR, "anthropic:claude-sonnet-4-6")
        rc = cli.main(["run", "--dry-run", "--json"])
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert "anthropic:claude-sonnet-4-6" in payload["argv"]


class TestTrialsSubcommand:
    def test_dry_run_emits_argv(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "2",
                "--eval-category",
                "tool_use",
                "--dry-run",
                "--json",
            ]
        )
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        argv = payload["argv"]
        assert "--model" in argv
        assert "openai:gpt-5.5" in argv
        assert "--trials" in argv
        assert "2" in argv
        assert "--eval-category" in argv
        assert "tool_use" in argv

    def test_retry_failed_collects_nodeids(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Seed a trial-report file with two failures.
        report = {
            "model": "openai:gpt-5.5",
            "passed": 1,
            "failed": 2,
            "skipped": 0,
            "total": 3,
            "failures": [
                {
                    "test_name": "tests/evals/test_memory.py::test_a",
                    "category": "memory",
                    "failure_message": "boom",
                },
                {
                    "test_name": "tests/evals/test_tool.py::test_b",
                    "category": "tool_use",
                    "failure_message": "boom",
                },
            ],
        }
        (tmp_path / "evals_report_trial_001.json").write_text(json.dumps(report))

        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "1",
                "--retry-failed",
                str(tmp_path),
                "--dry-run",
                "--json",
            ]
        )
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["model"] == "openai:gpt-5.5"
        assert sorted(payload["retry_failed"]) == [
            "tests/evals/test_memory.py::test_a",
            "tests/evals/test_tool.py::test_b",
        ]

    def test_retry_failed_no_failures_returns_no_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "1",
                "--retry-failed",
                str(tmp_path),
                "--dry-run",
            ]
        )
        assert rc == cli.EXIT_NO_REPORTS
        assert "no failed test node IDs" in capsys.readouterr().err

    def test_retry_failed_unreadable_reports_distinct_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Seed a corrupted report so `_load_report` discards it.
        (tmp_path / "evals_report_trial_001.json").write_text("not valid json")
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "1",
                "--retry-failed",
                str(tmp_path),
                "--dry-run",
            ]
        )
        err = capsys.readouterr().err
        assert rc == cli.EXIT_NO_REPORTS
        assert "discovered" in err
        assert "but none parsed" in err

    def test_retry_failed_forwards_separator_to_run_trials(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Capture the argv passed to run_trials.main when the dry-run shortcut
        # is bypassed (no --dry-run flag).
        report = {
            "failures": [{"test_name": "tests/evals/test_x.py::test_a"}],
        }
        (tmp_path / "evals_report_trial_001.json").write_text(json.dumps(report))

        captured: dict[str, list[str]] = {}

        def fake_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            # Write a passing summary so post-hoc resolution returns EXIT_OK.
            summary_path = tmp_path / "trials_summary.json"
            summary_path.write_text(json.dumps({"counts": {"failed": {"mean": 0}}}))
            return 0

        rt = cli._import_run_trials()
        monkeypatch.setattr(rt, "main", fake_main)
        # Force the summary path the CLI computes to land in tmp_path.
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "1",
                "--retry-failed",
                str(tmp_path),
                "--summary-out",
                str(tmp_path / "trials_summary.json"),
            ]
        )
        assert rc == cli.EXIT_OK
        argv = captured["argv"]
        # The `--` must precede any node ID forwarded to pytest, otherwise
        # `argparse.REMAINDER` parses node IDs as run_trials flags.
        sep_idx = argv.index("--")
        assert argv[sep_idx + 1] == "tests/evals/test_x.py::test_a"


class TestExitCodeMapping:
    def _summary_with_failures(self, path: Path, *, failed_mean: float) -> None:
        path.write_text(json.dumps({"counts": {"failed": {"mean": failed_mean}}}))

    def test_run_subprocess_zero_returns_ok(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_run(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        rc = cli.main(["run", "--model", "openai:gpt-5.5", "--json"])
        assert rc == cli.EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["returncode"] == 0

    def test_run_subprocess_nonzero_maps_to_eval_failures(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess(args=cmd, returncode=1),
        )
        rc = cli.main(["run", "--model", "openai:gpt-5.5"])
        assert rc == cli.EXIT_EVAL_FAILURES

    def test_trials_failures_in_summary_returns_eval_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        summary = tmp_path / "trials_summary.json"
        self._summary_with_failures(summary, failed_mean=2.5)
        rt = cli._import_run_trials()
        monkeypatch.setattr(rt, "main", lambda _argv: 0)
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "2",
                "--summary-out",
                str(summary),
            ]
        )
        assert rc == cli.EXIT_EVAL_FAILURES

    def test_trials_no_failures_returns_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        summary = tmp_path / "trials_summary.json"
        self._summary_with_failures(summary, failed_mean=0)
        rt = cli._import_run_trials()
        monkeypatch.setattr(rt, "main", lambda _argv: 0)
        rc = cli.main(
            [
                "trials",
                "--model",
                "openai:gpt-5.5",
                "--trials",
                "2",
                "--summary-out",
                str(summary),
            ]
        )
        assert rc == cli.EXIT_OK

    def test_trials_no_reports_maps_to_exit_no_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rt = cli._import_run_trials()
        monkeypatch.setattr(rt, "main", lambda _argv: 1)
        rc = cli.main(["trials", "--model", "openai:gpt-5.5", "--trials", "2"])
        assert rc == cli.EXIT_NO_REPORTS

    def test_aggregate_forwards_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, list[str]] = {}

        def fake_main(argv: list[str]) -> int:
            captured["argv"] = list(argv)
            (tmp_path / "trials_summary.json").write_text(
                json.dumps({"counts": {"failed": {"mean": 0}}})
            )
            return 0

        rt = cli._import_run_trials()
        monkeypatch.setattr(rt, "main", fake_main)
        rc = cli.main(["aggregate", str(tmp_path), "--json"])
        assert rc == cli.EXIT_OK
        assert "--json" in captured["argv"]

    def test_catalog_check_drift_maps_to_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess(args=cmd, returncode=1),
        )
        rc = cli.main(["catalog", "--check"])
        assert rc == cli.EXIT_CONFIG

    def test_model_groups_check_drift_maps_to_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli.subprocess,
            "run",
            lambda cmd, **_kw: subprocess.CompletedProcess(args=cmd, returncode=1),
        )
        rc = cli.main(["model-groups", "--check"])
        assert rc == cli.EXIT_CONFIG


class TestModelPrecedence:
    def test_explicit_model_beats_env_var(
        self,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(cli._MODEL_ENV_VAR, "from-env")
        rc = cli.main(
            [
                "run",
                "--model",
                "from-flag",
                "--dry-run",
                "--json",
            ]
        )
        assert rc == cli.EXIT_OK
        argv = json.loads(capsys.readouterr().out)["argv"]
        assert "from-flag" in argv
        assert "from-env" not in argv
