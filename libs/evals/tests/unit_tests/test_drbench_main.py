"""Tests for the DRBench Harbor task generator CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from harbor_adapters.drbench import adapter
from harbor_adapters.drbench.main import main

if TYPE_CHECKING:
    from pathlib import Path

# Every test in this module needs the fixture vendor directory, and none of them need
# its path.
pytestmark = pytest.mark.usefixtures("vendor")


@pytest.fixture
def vendor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the adapter at fixture pins and a fixture upstream checkout of three tasks.

    `ensure_upstream_checkout` is the only seam that would reach the network, so replacing
    it keeps these tests offline while every config-reading path stays under test.
    """
    vendor_dir = tmp_path / "vendor"
    (vendor_dir / "subsets").mkdir(parents=True)
    checkout = tmp_path / "upstream"
    upstream_tasks = checkout / "drbench" / "data" / "tasks"
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    monkeypatch.setattr(adapter, "ensure_upstream_checkout", lambda: checkout)
    for task_id in ("DR0001", "DR0002", "DR0003"):
        task_root = upstream_tasks / task_id
        (task_root / "config").mkdir(parents=True)
        (task_root / "config" / "task.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "dr_question": f"Question for {task_id}?",
                    "date": "2025-08-27",
                    "company_info": {"name": "Acme"},
                    "persona": {"name": "Dana Ray"},
                }
            )
        )
        (task_root / "config" / "env.json").write_text(
            json.dumps(
                {
                    "env_files": [
                        {
                            "source": f"drbench/data/tasks/{task_id}/files/QA001/report.pdf",
                            "destination": "shared/report.pdf",
                            "app": "nextcloud",
                            "qa_type": "insight",
                        }
                    ]
                }
            )
        )
        (task_root / "config" / "eval.json").write_text(
            json.dumps(
                {
                    "dr_report_evaluation_qa": [
                        {
                            "id": "IN1",
                            "qa_type": "insight",
                            "type": "enterprise_fact",
                            "answer": "kept",
                        }
                    ]
                }
            )
        )
        (task_root / "info.json").write_text(
            json.dumps({"industry": "retail", "domain": "compliance", "difficulty": "easy"})
        )
    # `available_task_ids` reads upstream's own subset list, which stays vendored so the
    # ids resolve offline and before any task directory exists.
    (vendor_dir / "subsets" / "val.jsonl").write_text(
        "".join(
            json.dumps({"task_id": task_id, "path": f"drbench/data/tasks/{task_id}/config"}) + "\n"
            for task_id in ("DR0001", "DR0002", "DR0003")
        )
    )
    # Task generation pins the image by digest, so the record must exist offline.
    (vendor_dir / "image_digests.json").write_text(
        json.dumps(
            {
                "registry": adapter.IMAGE_REGISTRY,
                "digests": {
                    task_id: f"sha256:{index:064x}"
                    for index, task_id in enumerate(("DR0001", "DR0002", "DR0003"), 1)
                },
            }
        )
    )
    return vendor_dir


def test_available_task_ids_is_sorted() -> None:
    assert adapter.available_task_ids() == ["DR0001", "DR0002", "DR0003"]


def test_available_task_ids_excludes_the_sanity_task() -> None:
    # SANITY0 is upstream's install smoke test, not a scored task; including it in
    # `--all` would skew the dataset average.
    # Upstream's `val.jsonl` lists only the 100 scored tasks, so SANITY0 can exist in the
    # checkout without ever entering `--all`.
    sanity = adapter.ensure_upstream_checkout() / "drbench" / "data" / "tasks" / "SANITY0"
    (sanity / "config").mkdir(parents=True)
    for name in ("task.json", "env.json", "eval.json"):
        (sanity / "config" / name).write_text("{}")
    (sanity / "info.json").write_text("{}")
    assert "SANITY0" not in adapter.available_task_ids()
    # ...but it stays reachable by name for debugging.
    assert adapter.parse_task_id("SANITY0") == "SANITY0"


def test_main_generates_named_task_ids(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--task-ids", "DR0002"])
    assert sorted(p.name for p in output_dir.iterdir()) == ["DR0002"]


def test_main_limit_takes_the_first_n(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--limit", "2"])
    assert sorted(p.name for p in output_dir.iterdir()) == ["DR0001", "DR0002"]


def test_main_all_generates_every_vendored_task(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    main(["--output-dir", str(output_dir), "--all"])
    assert sorted(p.name for p in output_dir.iterdir()) == ["DR0001", "DR0002", "DR0003"]


def test_main_requires_a_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        main(["--output-dir", str(tmp_path / "dataset")])


def test_main_requires_output_dir_without_populate() -> None:
    with pytest.raises(ValueError, match="`--output-dir` is required"):
        main(["--task-ids", "DR0001"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--populate", "d", "--task-ids", "DR0001"],
        ["--populate", "d", "--limit", "1"],
        ["--populate", "d", "--all"],
    ],
)
def test_main_populate_is_exclusive(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        main(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["--task-ids", "DR0001", "--all"],
        ["--task-ids", "DR0001", "--limit", "1"],
        ["--all", "--limit", "1"],
    ],
)
def test_main_selection_flags_are_exclusive(tmp_path: Path, argv: list[str]) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        main(["--output-dir", str(tmp_path / "dataset"), *argv])


def test_main_refresh_digests_is_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        main(["--refresh-digests", "--output-dir", str(tmp_path / "d"), "--all"])


def test_main_rejects_an_unknown_task_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a DRBench id"):
        main(["--output-dir", str(tmp_path / "dataset"), "--task-ids", "not-a-task"])
