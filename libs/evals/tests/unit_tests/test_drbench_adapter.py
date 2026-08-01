"""Tests for the DRBench Harbor task adapter."""

from __future__ import annotations

import json
import tarfile
from typing import TYPE_CHECKING

import pytest

from harbor_adapters.drbench import adapter

if TYPE_CHECKING:
    from pathlib import Path

_TASK_ID = "DR0001"


def _write_vendor(vendor: Path, *, env_files: list[dict], insights: list[dict]) -> None:
    """Write a minimal vendored config bundle for `_TASK_ID`."""
    task_root = vendor / "tasks" / _TASK_ID
    task_root.mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "task_id": _TASK_ID,
                "dr_question": "How should Acme respond to the new rules?",
                "date": "2025-08-27",
                "company_info": {"name": "Acme", "industry": "Retail"},
                "persona": {"name": "Dana Ray", "role": "Compliance Lead"},
            }
        )
    )
    (task_root / "env.json").write_text(json.dumps({"env_files": env_files}))
    (task_root / "eval.json").write_text(json.dumps({"dr_report_evaluation_qa": insights}))
    (task_root / "info.json").write_text(
        json.dumps({"industry": "retail", "domain": "compliance", "difficulty": "easy"})
    )


def _env_file(name: str, *, app: str = "nextcloud", qa_type: str = "insight") -> dict:
    return {
        "source": f"drbench/data/tasks/{_TASK_ID}/files/QA001/{name}",
        "destination": f"shared/{name}",
        "app": app,
        "qa_type": qa_type,
    }


@pytest.fixture
def vendor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the adapter at a fixture vendor directory."""
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    return vendor_dir


@pytest.mark.parametrize("task_id", ["DR0001", "SANITY0"])
def test_parse_task_id_accepts_drbench_ids(task_id: str) -> None:
    assert adapter.parse_task_id(task_id) == task_id


@pytest.mark.parametrize(
    "task_id",
    ["", "dr0001", "DR1", "DR00001", "../DR0001", "DR0001/x", ".", "..", "DR0001 "],
)
def test_parse_task_id_rejects_anything_else(task_id: str) -> None:
    """A task id is joined onto an output dir, so traversal must be impossible."""
    with pytest.raises(ValueError, match="must be a DRBench id"):
        adapter.parse_task_id(task_id)


def test_corpus_layout_groups_by_app_and_keeps_declared_name() -> None:
    layout = adapter.corpus_layout(
        {
            "env_files": [
                _env_file("report.pdf", app="nextcloud"),
                _env_file("inbox.jsonl", app="email"),
            ]
        }
    )
    assert layout == [
        (f"drbench/data/tasks/{_TASK_ID}/files/QA001/report.pdf", "nextcloud/report.pdf"),
        (f"drbench/data/tasks/{_TASK_ID}/files/QA001/inbox.jsonl", "email/inbox.jsonl"),
    ]


def test_corpus_layout_suffixes_duplicate_destinations() -> None:
    """Upstream declares one destination twice for different documents; keep both."""
    layout = adapter.corpus_layout(
        {"env_files": [_env_file("report.pdf"), _env_file("report.pdf")]}
    )
    assert [relative for _, relative in layout] == [
        "nextcloud/report.pdf",
        "nextcloud/report-2.pdf",
    ]


def test_corpus_layout_deduplicates_case_insensitively() -> None:
    """Names differing only by case must not collapse on a case-insensitive host."""
    layout = adapter.corpus_layout(
        {"env_files": [_env_file("Report.pdf"), _env_file("report.pdf")]}
    )
    assert [relative for _, relative in layout] == [
        "nextcloud/Report.pdf",
        "nextcloud/report-2.pdf",
    ]


def test_corpus_layout_rejects_unknown_app() -> None:
    with pytest.raises(ValueError, match="unknown app"):
        adapter.corpus_layout({"env_files": [_env_file("report.pdf", app="dropbox")]})


def test_corpus_layout_rejects_source_outside_task_files() -> None:
    entry = _env_file("report.pdf")
    entry["source"] = "../../etc/passwd"
    with pytest.raises(ValueError, match="not under a task's `files/`"):
        adapter.corpus_layout({"env_files": [entry]})


def test_insight_ground_truth_excludes_distractors_and_blanks() -> None:
    insights = adapter.insight_ground_truth(
        {
            "dr_report_evaluation_qa": [
                {"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"},
                {"id": "EX1", "qa_type": "insight", "type": "external_fact", "answer": "also kept"},
                {"id": "DI1", "qa_type": "distractor", "type": "enterprise_fact", "answer": "no"},
                {"id": "IN2", "qa_type": "insight", "type": "enterprise_fact", "answer": "  "},
            ]
        }
    )
    assert [insight["id"] for insight in insights] == ["IN1", "EX1"]
    assert [insight["type"] for insight in insights] == ["enterprise_fact", "external_fact"]


def test_generate_task_creates_self_contained_harbor_task(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("inbox.jsonl", app="email")],
        insights=[
            {
                "id": "IN1",
                "qa_type": "insight",
                "type": "enterprise_fact",
                "answer": "Acme tracks 250 SKUs.",
            },
            {
                "id": "DI1",
                "qa_type": "distractor",
                "type": "enterprise_fact",
                "answer": "Unrelated.",
            },
        ],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    for relative in (
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/.dockerignore",
        "environment/extract_text.py",
        "solution/solve.sh",
        "tests/case.json",
        "tests/test.sh",
        "tests/judge.py",
    ):
        assert (task_dir / relative).is_file(), relative

    task_toml = (task_dir / "task.toml").read_text()
    assert 'source = "drbench"' in task_toml
    assert f'task_id = "{_TASK_ID}"' in task_toml
    # Open-web egress is required: external_fact ground truth is not in the corpus.
    assert 'network_mode = "public"' in task_toml
    assert "insight_count = 1" in task_toml
    assert "document_count = 2" in task_toml

    instruction = (task_dir / "instruction.md").read_text()
    assert "Dana Ray" in instruction
    assert "/app/report.md" in instruction
    # The prompt must name only the apps this task actually has documents in.
    assert "/app/files/nextcloud/" in instruction
    assert "/app/files/email/" in instruction
    assert "/app/files/mattermost/" not in instruction
    # Ground truth must never appear in anything the agent can read.
    assert "Acme tracks 250 SKUs" not in instruction


def test_generate_task_keeps_ground_truth_only_under_tests(vendor: Path, tmp_path: Path) -> None:
    secret = "Acme tracks 250 high-risk SKUs."
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": secret}],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    case = json.loads((task_dir / "tests" / "case.json").read_text())
    assert case["task_id"] == _TASK_ID
    assert [insight["answer"] for insight in case["insights"]] == [secret]

    # `tests/` goes to the verifier and `solution/` is uploaded only by Harbor's
    # OracleAgent, never on a real agent run. Everything else -- the prompt, task.toml,
    # and the image build context -- is reachable by the agent under test, so the answer
    # may not appear anywhere in it.
    agent_visible = [
        path
        for path in task_dir.rglob("*")
        if path.is_file() and not {"tests", "solution"} & set(path.relative_to(task_dir).parts)
    ]
    assert agent_visible, "expected agent-visible files to check"
    for path in agent_visible:
        assert secret not in path.read_text(errors="replace"), path


def test_generate_task_is_idempotent(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"}],
    )
    output_dir = tmp_path / "dataset"
    first = adapter.generate_task(output_dir=output_dir, task_id=_TASK_ID)
    (first / "stale.txt").write_text("should be removed")
    second = adapter.generate_task(output_dir=output_dir, task_id=_TASK_ID)
    assert not (second / "stale.txt").exists()


def _archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Build an upstream-shaped tarball (single `<repo>-<sha>/` root)."""
    root = f"drbench-{adapter.UPSTREAM_SHA}"
    staging = tmp_path / "staging"
    archive_path = tmp_path / "corpus.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for relative, payload in members.items():
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
            tar.add(staged, arcname=f"{root}/{relative}")
    return archive_path


def test_populate_corpus_lays_down_only_declared_files(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("inbox.jsonl", app="email")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"}],
    )
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)

    prefix = f"drbench/data/tasks/{_TASK_ID}/files/QA001"
    archive_path = _archive(
        tmp_path,
        {
            f"{prefix}/report.pdf": b"pdf bytes",
            f"{prefix}/inbox.jsonl": b'{"type":"email"}',
            # Sidecars that carry the answers, and plaintext twins of the binary
            # documents. Neither is declared in env.json, so neither may be laid down.
            f"{prefix}/qa_dict.json": b'{"answer":"leaked"}',
            f"{prefix}/report.md": b"plaintext twin",
        },
    )
    assert adapter.populate_corpus(dataset_dir, archive=archive_path) == 1

    files_dir = dataset_dir / _TASK_ID / "environment" / "files"
    assert sorted(
        p.relative_to(files_dir).as_posix() for p in files_dir.rglob("*") if p.is_file()
    ) == [
        "email/inbox.jsonl",
        "nextcloud/report.pdf",
    ]


def test_populate_corpus_resolves_case_only_mismatch(vendor: Path, tmp_path: Path) -> None:
    """Upstream's DR0038 manifest disagrees with the tree on filename case."""
    _write_vendor(
        vendor,
        env_files=[_env_file("pr-engagements-overview.docx")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"}],
    )
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)

    prefix = f"drbench/data/tasks/{_TASK_ID}/files/QA001"
    archive_path = _archive(tmp_path, {f"{prefix}/PR-Engagements-Overview.docx": b"docx"})
    assert adapter.populate_corpus(dataset_dir, archive=archive_path) == 1

    laid_down = dataset_dir / _TASK_ID / "environment" / "files" / "nextcloud"
    assert (laid_down / "pr-engagements-overview.docx").read_bytes() == b"docx"


def test_populate_corpus_fails_when_a_document_is_absent(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("missing.pdf")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"}],
    )
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)

    prefix = f"drbench/data/tasks/{_TASK_ID}/files/QA001"
    archive_path = _archive(tmp_path, {f"{prefix}/report.pdf": b"pdf"})
    with pytest.raises(FileNotFoundError, match="missing 1 file"):
        adapter.populate_corpus(dataset_dir, archive=archive_path)


def test_populate_corpus_ignores_foreign_task_dirs(vendor: Path, tmp_path: Path) -> None:
    """Only tasks this adapter generated may be populated by it."""
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf")],
        insights=[{"id": "IN1", "qa_type": "insight", "type": "enterprise_fact", "answer": "kept"}],
    )
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)
    foreign = dataset_dir / "cb-cloud-1"
    foreign.mkdir()
    (foreign / "task.toml").write_text('version = "1.3"\n\n[metadata]\nsource = "contextbench"\n')

    prefix = f"drbench/data/tasks/{_TASK_ID}/files/QA001"
    archive_path = _archive(tmp_path, {f"{prefix}/report.pdf": b"pdf"})
    assert adapter.populate_corpus(dataset_dir, archive=archive_path) == 1
    assert not (foreign / "environment").exists()
