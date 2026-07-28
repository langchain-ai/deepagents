"""Tests for the OOLONG-synth Harbor task adapter.

These never touch the network: ``load_oolong_examples`` (the HuggingFace fetch)
is monkeypatched to return synthetic examples, so ``generate_dataset`` /
``--populate`` are exercised purely against the local filesystem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from harbor_adapters.oolong import generate_oolong_tasks
from harbor_adapters.oolong.generate_oolong_tasks import generate_dataset
from harbor_adapters.oolong.loader import OolongExample
from harbor_adapters.oolong.main import main as populate_main

if TYPE_CHECKING:
    from pathlib import Path

_GOLD = "SECRET-GOLD-42"


def _example(task_id: int, task_group: str) -> OolongExample:
    return OolongExample(
        task_id=task_id,
        dataset="trec_coarse",
        task_group=task_group,
        task_type="classification",
        context_len=1024,
        context_window_id=task_id * 10,
        context_window_text=f"long document body for task {task_id}",
        context_window_text_with_labels="labelled body",
        num_labels=6,
        question="How many entries fall in each category?",
        gold_answers=(_GOLD,),
        gold_answer_raw=_GOLD,
        answer_type="dict",
        input_subset=False,
    )


@pytest.fixture
def fake_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the HF fetch with two synthetic examples (no network)."""
    examples = (_example(1, "counting"), _example(2, "timeline"))

    def _fake_load(**_kwargs: object) -> tuple[OolongExample, ...]:
        return examples

    monkeypatch.setattr(generate_oolong_tasks, "load_oolong_examples", _fake_load)


@pytest.mark.usefixtures("fake_bucket")
def test_generate_dataset_writes_self_contained_tasks(tmp_path: Path) -> None:
    count = generate_dataset(out_dir=tmp_path, dataset="trec_coarse", context_len=1024)

    assert count == 2
    task_dirs = sorted(p.name for p in tmp_path.glob("oolong-synth-*"))
    assert task_dirs == [
        "oolong-synth-trec_coarse-1024-1",
        "oolong-synth-trec_coarse-1024-2",
    ]

    task_dir = tmp_path / "oolong-synth-trec_coarse-1024-1"
    for rel in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/context.txt",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/score.py",
        "tests/datapoint.json",
        "tests/official_scorer.py",
    ):
        assert (task_dir / rel).is_file(), f"missing {rel}"

    assert (task_dir / "environment/context.txt").read_text() == "long document body for task 1"


@pytest.mark.usefixtures("fake_bucket")
def test_gold_answer_only_in_verifier_datapoint(tmp_path: Path) -> None:
    generate_dataset(out_dir=tmp_path, dataset="trec_coarse", context_len=1024)
    task_dir = tmp_path / "oolong-synth-trec_coarse-1024-1"

    # Gold must not leak into anything the agent can read.
    for rel in ("instruction.md", "environment/context.txt", "task.toml"):
        assert _GOLD not in (task_dir / rel).read_text(), f"gold leaked into {rel}"

    # Gold lives only in the verifier-side datapoint (and the oracle solution).
    assert _GOLD in (task_dir / "tests/datapoint.json").read_text()
    assert _GOLD in (task_dir / "solution/solve.sh").read_text()


@pytest.mark.usefixtures("fake_bucket")
def test_populate_reads_bucket_and_preserves_committed_scaffolding(tmp_path: Path) -> None:
    # The bucket is read from the committed bucket.toml, and pre-existing
    # committed scaffolding must not be clobbered by populate.
    (tmp_path / "bucket.toml").write_text(
        'subset = "trec_coarse"\ncontext_len = 1024\nn_examples = 0\n'
    )
    sentinel_toml = '[dataset]\nname = "committed/sentinel"\n'
    sentinel_metric = "# committed metric sentinel\n"
    (tmp_path / "dataset.toml").write_text(sentinel_toml)
    (tmp_path / "metric.py").write_text(sentinel_metric)

    populate_main(["--populate", str(tmp_path)])

    assert (tmp_path / "dataset.toml").read_text() == sentinel_toml
    assert (tmp_path / "metric.py").read_text() == sentinel_metric
    assert len(list(tmp_path.glob("oolong-synth-*"))) == 2


def test_populate_without_bucket_toml_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"bucket\.toml"):
        populate_main(["--populate", str(tmp_path)])


@pytest.mark.usefixtures("fake_bucket")
def test_repopulate_clears_stale_task_dirs(tmp_path: Path) -> None:
    stale = tmp_path / "oolong-synth-trec_coarse-9999-77"
    (stale / "tests").mkdir(parents=True)
    (stale / "task.toml").write_text('version = "1.3"\n')

    generate_dataset(out_dir=tmp_path, dataset="trec_coarse", context_len=1024)

    assert not stale.exists()
    assert len(list(tmp_path.glob("oolong-synth-*"))) == 2
