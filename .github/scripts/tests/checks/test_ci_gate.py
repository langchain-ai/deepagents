"""Tests for the `ci_success` talon-failure waiver matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "checks" / "ci_gate.py"

sys.path.insert(0, str(SCRIPT.parent))

from ci_gate import evaluate_gate

ALL_GREEN = {"lint-code": "success", "test-code": "success"}


def test_pr_without_talon_changes_waives_test_talon_failure() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "failure"},
        event="pull_request",
        talon="false",
    )

    assert gate["ok"] is True
    assert gate["waived"] == ["test-talon"]
    assert gate["failed"] == []


def test_pr_without_talon_changes_waives_lint_talon_failure() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "lint-talon": "failure"},
        event="pull_request",
        talon="false",
    )

    assert gate["ok"] is True
    assert gate["waived"] == ["lint-talon"]


def test_pr_without_talon_changes_still_fails_on_other_jobs() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "failure", "test-code": "failure"},
        event="pull_request",
        talon="false",
    )

    assert gate["ok"] is False
    assert gate["waived"] == ["test-talon"]
    assert gate["failed"] == ["test-code"]


def test_genuine_talon_pr_still_fails_on_talon_failure() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "failure"},
        event="pull_request",
        talon="true",
    )

    assert gate["ok"] is False
    assert gate["waived"] == []
    assert gate["failed"] == ["test-talon"]


def test_push_still_fails_on_talon_failure() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "failure"},
        event="push",
        talon="false",
    )

    assert gate["ok"] is False
    assert gate["failed"] == ["test-talon"]


def test_merge_group_still_fails_on_talon_failure() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "failure"},
        event="merge_group",
        talon="false",
    )

    assert gate["ok"] is False
    assert gate["failed"] == ["test-talon"]


def test_cancelled_talon_job_is_never_waived() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "test-talon": "cancelled"},
        event="pull_request",
        talon="false",
    )

    assert gate["ok"] is False
    assert gate["waived"] == []
    assert gate["cancelled"] == ["test-talon"]


def test_successful_talon_jobs_have_nothing_to_waive() -> None:
    gate = evaluate_gate(
        {**ALL_GREEN, "lint-talon": "success", "test-talon": "success"},
        event="pull_request",
        talon="false",
    )

    assert gate["ok"] is True
    assert gate["waived"] == []
    assert gate["failed"] == []
    assert gate["cancelled"] == []


def test_cli_emits_single_json_line() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--event",
            "pull_request",
            "--talon",
            "false",
            "--results",
            json.dumps({"test-talon": "failure"}),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    gate = json.loads(result.stdout)
    assert gate["ok"] is True
    assert gate["waived"] == ["test-talon"]
