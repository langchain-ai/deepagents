"""Tests for search-artifact redaction.

The regression these guard against is concrete: a filename-based filter kept
`<trial>/agent/result.json` -- the full agent transcript, which shares its
basename with the trial record -- and published decrypted benchmark questions to
a public repo's artifacts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "redact_search_artifacts", Path(__file__).with_name("redact_search_artifacts.py")
)
assert _SPEC and _SPEC.loader
rsa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rsa)

_SECRET = "Identify the horse that satisfies all of the following conditions"


def _job_tree(root: Path) -> None:
    """Build a Harbor job tree shaped like a real search run."""
    trial = root / "2026-01-01__00-00-00" / "loho-01__abc123"
    (trial / "agent").mkdir(parents=True)
    (trial / "logs" / "verifier").mkdir(parents=True)

    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "loho-01",
                "verifier_result": {"rewards": {"reward": 0.5}},
                "exception_info": None,
                "config": {
                    "job_id": "job-1",
                    "agent": {"model_name": "anthropic:claude-haiku-4-5"},
                },
            }
        )
    )
    # The transcript: same basename, different path, full plaintext.
    (trial / "agent" / "result.json").write_text(
        json.dumps({"messages": [{"type": "human", "content": _SECRET}]})
    )
    (trial / "agent" / "instruction.txt").write_text(_SECRET)
    (trial / "logs" / "verifier" / "judges.json").write_text(
        json.dumps({"browsecomp": {"rationale": f"the answer to '{_SECRET}' is X"}})
    )
    (root / "2026-01-01__00-00-00" / "result.json").write_text(json.dumps({"stats": {}}))
    (root / "empty-shard-search-0").write_text("")


def test_transcript_sharing_a_basename_is_removed(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    _job_tree(root)

    rsa.redact(root)

    survivors = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert survivors == [
        "2026-01-01__00-00-00/loho-01__abc123/result.json",
        "empty-shard-search-0",
    ]


def test_no_plaintext_survives_anywhere(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    _job_tree(root)

    rsa.redact(root)

    blob = "\n".join(p.read_text() for p in root.rglob("*") if p.is_file())
    assert _SECRET not in blob


def test_kept_record_has_exactly_the_fields_the_aggregator_reads(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    _job_tree(root)

    rsa.redact(root)

    record = json.loads(
        (root / "2026-01-01__00-00-00" / "loho-01__abc123" / "result.json").read_text()
    )
    assert record == {
        "task_name": "loho-01",
        "verifier_result": {"rewards": {"reward": 0.5}},
        "config": {"job_id": "job-1", "agent": {"model_name": "anthropic:claude-haiku-4-5"}},
    }


def test_unknown_fields_are_dropped_by_default(tmp_path: Path) -> None:
    """A field a future Harbor adds must not be published just because it is new."""
    record = rsa.safe_record(
        {
            "task_name": "loho-01",
            "some_future_field": _SECRET,
            "verifier_result": {"rewards": {"reward": 1.0}, "notes": _SECRET},
        }
    )
    assert "some_future_field" not in record
    assert record["verifier_result"] == {"rewards": {"reward": 1.0}}


def test_exception_is_reduced_to_its_type(tmp_path: Path) -> None:
    """Presence drives `trial_errored`; messages and tracebacks can quote content."""
    record = rsa.safe_record(
        {
            "task_name": "loho-01",
            "exception_info": {
                "exception_type": "ResourceNotFoundError",
                "exception_message": _SECRET,
                "exception_traceback": _SECRET,
            },
        }
    )
    assert record["exception_info"] == {"exception_type": "ResourceNotFoundError"}


def test_empty_shard_markers_survive(tmp_path: Path) -> None:
    """The aggregator counts them to tell an empty shard from a missing one."""
    root = tmp_path / "job"
    root.mkdir()
    (root / "empty-shard-search-3").write_text("")

    rsa.redact(root)

    assert (root / "empty-shard-search-3").is_file()


def test_missing_root_is_a_no_op(tmp_path: Path) -> None:
    assert rsa.redact(tmp_path / "nope") == (0, 0)
