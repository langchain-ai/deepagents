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

    # Simulate the git-ignored corpus being absent (fresh checkout).
    files_dir = output_dir / "cb-cloud-1" / "environment" / "files"
    for corpus_file in files_dir.iterdir():
        corpus_file.unlink()
    files_dir.rmdir()

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
    assert not (other / "environment").exists()
