"""Tests for the Context-Bench Harbor task generator CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harbor_adapters.contextbench import adapter
from harbor_adapters.contextbench.main import main

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_FIXTURE_RECORDS = [
    {
        "input": "Who has the highest total bank balance?",
        "ground_truth": "Linda Robbins",
        "agent_args": {
            "extra": {
                "question_type": "multi_hop_chain",
                "difficulty": "hard",
                "required_files": ["bank_accounts.txt", "people.txt"],
            }
        },
    },
    {
        "input": "Which resident owns the most vehicles?",
        "ground_truth": "Tammy Roberts",
        "agent_args": {
            "extra": {
                "question_type": "comparison_tiebreak",
                "difficulty": "easy",
                "required_files": ["pets.txt", "addresses.txt", "vehicles.txt", "people.txt"],
            }
        },
    },
    {
        "input": "Who owns the vehicle with license plate '7D U3378'?",
        "ground_truth": "George Peterson",
        "agent_args": {
            "extra": {
                "question_type": "negation",
                "difficulty": "medium",
                "required_files": ["vehicles.txt", "people.txt"],
            }
        },
    },
]


def _write_vendor_fixture(vendor_dir: Path) -> None:
    vendor_dir.mkdir(parents=True)
    lines = [json.dumps(record) for record in _FIXTURE_RECORDS]
    (vendor_dir / "filesystem_cloud.jsonl").write_text("\n".join(lines) + "\n")
    # The verifier ships the upstream grading rubric into each task.
    (vendor_dir / "rubric.txt").write_text(
        "Question: {input}\nExpected: {ground_truth}\nSubmission: {submission}\n"
    )
    files_dir = vendor_dir / "files"
    files_dir.mkdir()
    for filename in (
        "addresses.txt",
        "bank_accounts.txt",
        "people.txt",
        "pets.txt",
        "vehicles.txt",
    ):
        (files_dir / filename).write_text(f"{filename} source data\n")


def test_main_generates_task_by_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vendor_dir = tmp_path / "vendor"
    _write_vendor_fixture(vendor_dir)
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)

    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])

    task_dir = output_dir / "cb-cloud-1"
    task_toml = (task_dir / "task.toml").read_text()
    assert 'network_mode = "allowlist"' in task_toml
    solve_sh = (task_dir / "solution" / "solve.sh").read_text()
    assert "Tammy Roberts" in solve_sh


def test_populate_restores_corpus_from_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor_dir = tmp_path / "vendor"
    _write_vendor_fixture(vendor_dir)
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)

    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])

    # Simulate the git-ignored corpus AND single-sourced verifier files being
    # absent (fresh checkout): only the committed tests/case.json survives.
    files_dir = output_dir / "cb-cloud-1" / "environment" / "files"
    for corpus_file in files_dir.iterdir():
        corpus_file.unlink()
    files_dir.rmdir()
    tests_dir = output_dir / "cb-cloud-1" / "tests"
    for invariant in ("test.sh", "judge.py", "rubric.txt"):
        (tests_dir / invariant).unlink()

    # A non-contextbench sibling dir must be left untouched.
    other = output_dir / "not-a-cb-task"
    other.mkdir()
    (other / "task.toml").write_text('source = "elsewhere"\n')

    main(["--populate", str(output_dir)])

    restored = sorted(p.name for p in files_dir.iterdir())
    assert restored == [
        "addresses.txt",
        "bank_accounts.txt",
        "people.txt",
        "pets.txt",
        "vehicles.txt",
    ]
    assert (files_dir / "people.txt").read_text() == "people.txt source data\n"
    # The invariant verifier files are restored; the committed case.json stays.
    assert (tests_dir / "test.sh").read_text().endswith("python3 /tests/judge.py\n")
    assert (tests_dir / "judge.py").is_file()
    assert "{submission}" in (tests_dir / "rubric.txt").read_text()
    assert json.loads((tests_dir / "case.json").read_text())["ground_truth"] == "Tammy Roberts"
    assert not (other / "environment").exists()


def test_generate_task_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vendor_dir = tmp_path / "vendor"
    _write_vendor_fixture(vendor_dir)
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    output_dir = tmp_path / "dataset"

    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])
    # A second run over the same output dir must overwrite cleanly, not raise.
    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])

    assert (output_dir / "cb-cloud-1" / "task.toml").is_file()


def test_task_toml_records_source_difficulty_and_provider_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor_dir = tmp_path / "vendor"
    _write_vendor_fixture(vendor_dir)
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    output_dir = tmp_path / "dataset"

    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])
    task_toml = (output_dir / "cb-cloud-1" / "task.toml").read_text()

    # cb-cloud-1 fixture record is `easy`; both fields start at the source label.
    assert 'difficulty = "easy"' in task_toml
    assert 'source_difficulty = "easy"' in task_toml
    # Allowlist must admit non-Anthropic providers so any selectable model runs.
    for host in ("api.anthropic.com", "api.openai.com", "api.x.ai", "openrouter.ai"):
        assert host in task_toml


def test_stamp_calibrated_tiers_overwrites_difficulty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vendor_dir = tmp_path / "vendor"
    _write_vendor_fixture(vendor_dir)
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--task-ids", "cb-cloud-1"])

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"tasks": {"cb-cloud-1": {"tier": "hard"}}}))
    main(["--stamp-tiers", str(output_dir), "--calibration", str(calibration)])

    task_toml = (output_dir / "cb-cloud-1" / "task.toml").read_text()
    assert 'difficulty = "hard"' in task_toml  # calibrated tier stamped
    assert 'source_difficulty = "easy"' in task_toml  # provenance preserved
