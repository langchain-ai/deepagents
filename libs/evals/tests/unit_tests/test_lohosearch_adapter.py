"""Tests for the LoHoSearch Harbor task adapter.

No network: every test builds its own encrypted CSV with the same scheme
upstream uses, so decryption, identity, and task generation are exercised
end-to-end without touching HuggingFace.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from csv import writer as csv_writer
from typing import TYPE_CHECKING

import pytest

from harbor_adapters.lohosearch import adapter

if TYPE_CHECKING:
    from pathlib import Path

_CANARY = (
    "BENCHMARK DATA SHOULD NEVER APPEAR AS PLAIN TEXT ONLINE. "
    "canary GUID lohosearch:00000000-0000-0000-0000-000000000000"
)


def _encrypt(plaintext: str, password: str = _CANARY) -> str:
    """Apply the BrowseComp XOR scheme, i.e. the inverse of `adapter.decrypt`."""
    data = plaintext.encode()
    key = adapter.derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key, strict=True))).decode()


def _csv(records: list[tuple[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv_writer(buffer)
    writer.writerow(["question", "answer", "canary"])
    for question, answer in records:
        writer.writerow([_encrypt(question), _encrypt(answer), _CANARY])
    return buffer.getvalue()


def test_decrypt_round_trips_the_browsecomp_scheme() -> None:
    assert adapter.decrypt(_encrypt("Which horse?"), _CANARY) == "Which horse?"


def test_parse_rows_decrypts_and_ids_every_record() -> None:
    rows = adapter.parse_rows(_csv([("Q one", "A one"), ("Q two", "A two")]))

    assert [row.question for row in rows] == ["Q one", "Q two"]
    assert [row.answer for row in rows] == ["A one", "A two"]
    # The id is the hash of the ciphertext, not of the plaintext, so it can be
    # computed against upstream without decrypting.
    assert rows[0].question_sha256 == hashlib.sha256(_encrypt("Q one").encode()).hexdigest()
    assert rows[0].question_sha256 != rows[1].question_sha256


def test_parse_rows_rejects_a_missing_column() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        adapter.parse_rows("question,canary\nabc,def\n")


def test_resolve_tasks_matches_by_hash_not_position() -> None:
    """Upstream re-ordering must not repoint a task at a different question."""
    rows = adapter.parse_rows(_csv([("Q one", "A one"), ("Q two", "A two")]))
    manifest = {"tasks": {"loho-01": {"question_sha256": rows[1].question_sha256}}}

    reordered = list(reversed(rows))
    resolved = adapter.resolve_tasks(manifest, reordered)

    assert resolved["loho-01"].question == "Q two"


def test_resolve_tasks_fails_loudly_when_a_question_changed() -> None:
    rows = adapter.parse_rows(_csv([("Q one", "A one")]))
    manifest = {"tasks": {"loho-01": {"question_sha256": "0" * 64}}}

    with pytest.raises(ValueError, match="no longer present upstream"):
        adapter.resolve_tasks(manifest, rows)


@pytest.mark.parametrize(
    "task_name",
    ["../escape", "loho-01/nested", "LOHO-01", ".hidden", ""],
)
def test_resolve_tasks_rejects_unsafe_task_names(task_name: str) -> None:
    rows = adapter.parse_rows(_csv([("Q one", "A one")]))
    manifest = {"tasks": {task_name: {"question_sha256": rows[0].question_sha256}}}

    with pytest.raises(ValueError, match="single lowercase path component"):
        adapter.resolve_tasks(manifest, rows)


def test_resolve_tasks_requires_a_hash() -> None:
    rows = adapter.parse_rows(_csv([("Q one", "A one")]))

    with pytest.raises(ValueError, match="must declare a `question_sha256`"):
        adapter.resolve_tasks({"tasks": {"loho-01": {"row_index": 0}}}, rows)


def test_generate_task_creates_a_self_contained_harbor_task(tmp_path: Path) -> None:
    rows = adapter.parse_rows(_csv([("Which horse?", "Sky High")]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    task_dir = adapter.generate_task(rows[0], "loho-01", dataset_dir)

    assert task_dir == dataset_dir / "loho-01"
    instruction = (task_dir / "instruction.md").read_text()
    assert instruction.startswith("Which horse?")
    assert "Exact Answer:" in instruction
    assert "/app/answer.txt" in instruction

    case = json.loads((task_dir / "tests" / "case.json").read_text())
    assert case == {"question": "Which horse?", "ground_truth": "Sky High"}

    for name in ("test.sh", "judge.py", "browsecomp_grader.txt", "simpleqa_grader.txt"):
        assert (task_dir / "tests" / name).is_file()


def test_generated_dockerfile_copies_no_task_content(tmp_path: Path) -> None:
    """The answer key lives only in tests/, which is mounted at verify time."""
    rows = adapter.parse_rows(_csv([("Which horse?", "Sky High")]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    task_dir = adapter.generate_task(rows[0], "loho-01", dataset_dir)

    assert "COPY" not in (task_dir / "environment" / "Dockerfile").read_text()


def test_solution_never_interpolates_the_answer_into_the_shell(tmp_path: Path) -> None:
    """The answer comes from a third-party CSV, so it must not reach the shell."""
    hostile = "'; rm -rf / #"
    rows = adapter.parse_rows(_csv([("Q", hostile)]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    task_dir = adapter.generate_task(rows[0], "loho-01", dataset_dir)

    solve = (task_dir / "solution" / "solve.sh").read_text()
    assert "rm -rf" not in solve
    payload = re.search(r"printf '%s' '([A-Za-z0-9+/=]+)'", solve)
    assert payload is not None, solve
    assert base64.b64decode(payload.group(1)).decode() == f"{hostile}\n"


def test_task_toml_grants_the_agent_public_egress(tmp_path: Path) -> None:
    rows = adapter.parse_rows(_csv([("Q", "A")]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    task_dir = adapter.generate_task(rows[0], "loho-01", dataset_dir)
    task_toml = (task_dir / "task.toml").read_text()

    assert '[agent]\nnetwork_mode = "public"' in task_toml
    assert f'question_sha256 = "{rows[0].question_sha256}"' in task_toml


def test_generate_task_regenerates_cleanly(tmp_path: Path) -> None:
    rows = adapter.parse_rows(_csv([("Q", "A")]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    task_dir = adapter.generate_task(rows[0], "loho-01", dataset_dir)
    (task_dir / "stale.txt").write_text("left over from a previous run")
    adapter.generate_task(rows[0], "loho-01", dataset_dir)

    assert not (task_dir / "stale.txt").exists()


@pytest.mark.parametrize("task_name", ["../escape", "sub/dir", "Loho-01"])
def test_generate_task_rejects_unsafe_names(task_name: str, tmp_path: Path) -> None:
    rows = adapter.parse_rows(_csv([("Q", "A")]))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    with pytest.raises(ValueError, match="single lowercase path component"):
        adapter.generate_task(rows[0], task_name, dataset_dir)


def test_populate_tasks_generates_every_manifest_entry(tmp_path: Path, monkeypatch) -> None:
    csv_text = _csv([("Q one", "A one"), ("Q two", "A two")])
    rows = adapter.parse_rows(csv_text)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "loho-01": {"question_sha256": rows[0].question_sha256},
                    "loho-02": {"question_sha256": rows[1].question_sha256},
                }
            }
        )
    )
    monkeypatch.setattr(adapter, "fetch_rows", lambda *_args, **_kwargs: rows)

    count = adapter.populate_tasks(dataset_dir)

    assert count == 2
    assert (dataset_dir / "loho-01" / "instruction.md").read_text().startswith("Q one")
    assert (dataset_dir / "loho-02" / "instruction.md").read_text().startswith("Q two")


def test_populate_tasks_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No manifest"):
        adapter.populate_tasks(tmp_path)


def test_render_grader_substitutes_without_cascading() -> None:
    """A value containing a placeholder must not be re-expanded."""
    rendered = adapter.render_grader(
        "Q: {question}\nA: {correct_answer}",
        question="{correct_answer}",
        correct_answer="secret",
    )

    assert rendered == "Q: {correct_answer}\nA: secret"


def test_render_grader_leaves_unknown_braces_alone() -> None:
    assert adapter.render_grader("{not_a_field} {question}", question="q") == ("{not_a_field} q")
