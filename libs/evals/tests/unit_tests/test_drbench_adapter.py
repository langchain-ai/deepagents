"""Tests for the DRBench Harbor task adapter (app mode)."""

from __future__ import annotations

import json
import tomllib
from typing import TYPE_CHECKING

import pytest
import yaml
from harbor.models.task.config import NetworkMode, TaskConfig

from harbor_adapters.drbench import adapter

if TYPE_CHECKING:
    from pathlib import Path

_TASK_ID = "DR0001"
_DIGEST = "sha256:" + "a" * 64


def _write_vendor(
    vendor: Path,
    *,
    env_files: list[dict],
    qa: list[dict],
    persona: dict | None = None,
    task_id: str = _TASK_ID,
) -> None:
    """Write a minimal vendored config bundle plus a pinned image digest."""
    task_root = vendor / "tasks" / task_id
    task_root.mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "dr_question": "How should Acme respond to the new rules?",
                "date": "2025-08-27",
                "company_info": {"name": "Acme", "industry": "Retail"},
                "persona": persona
                if persona is not None
                else {
                    "name": "Dana Ray",
                    "role": "Compliance Lead",
                    "username": "dana.ray",
                    "password": "my_drbench_pwd",
                },
            }
        )
    )
    (task_root / "env.json").write_text(json.dumps({"env_files": env_files}))
    (task_root / "eval.json").write_text(json.dumps({"dr_report_evaluation_qa": qa}))
    (task_root / "info.json").write_text(
        json.dumps({"industry": "retail", "domain": "compliance", "difficulty": "easy"})
    )
    (vendor / "image_digests.json").write_text(
        json.dumps({"registry": adapter.IMAGE_REGISTRY, "digests": {task_id: _DIGEST}})
    )


def _env_file(name: str, *, app: str = "nextcloud", qa_type: str = "insight") -> dict:
    return {
        "source": f"drbench/data/tasks/{_TASK_ID}/files/QA001/{name}",
        "destination": f"shared/{name}",
        "app": app,
        "qa_type": qa_type,
    }


def _qa(
    qa_id: str, answer: str, *, qa_type: str = "insight", kind: str = "enterprise_fact"
) -> dict:
    return {"id": qa_id, "qa_type": qa_type, "type": kind, "answer": answer, "question": "?"}


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
    [
        "",
        "dr0001",
        "DR1",
        "DR00001",
        "../DR0001",
        "DR0001/x",
        ".",
        "..",
        "DR0001 ",
        "DR0001:latest",
    ],
)
def test_parse_task_id_rejects_anything_else(task_id: str) -> None:
    """The id is joined onto an output dir and interpolated into an image tag."""
    with pytest.raises(ValueError, match="must be a DRBench id"):
        adapter.parse_task_id(task_id)


def test_task_apps_reports_only_the_apps_used() -> None:
    apps = adapter.task_apps(
        {
            "env_files": [
                _env_file("a.pdf", app="nextcloud"),
                _env_file("b.jsonl", app="email"),
                _env_file("c.pdf", app="nextcloud"),
            ]
        }
    )
    assert apps == ["email", "nextcloud"]


def test_task_apps_rejects_an_unknown_app() -> None:
    with pytest.raises(ValueError, match="unknown app"):
        adapter.task_apps({"env_files": [_env_file("a.pdf", app="dropbox")]})


def test_document_count_counts_the_manifest() -> None:
    assert adapter.document_count({"env_files": [_env_file("a.pdf"), _env_file("b.pdf")]}) == 2


def test_qa_ground_truth_separates_insights_from_distractors() -> None:
    eval_config = {
        "dr_report_evaluation_qa": [
            _qa("IN1", "kept"),
            _qa("EX1", "also kept", kind="external_fact"),
            _qa("DI1", "planted", qa_type="distractor"),
            _qa("IN2", "   "),
        ]
    }
    insights = adapter.qa_ground_truth(eval_config, "insight")
    distractors = adapter.qa_ground_truth(eval_config, "distractor")

    # Blank answers are dropped; upstream order is preserved.
    assert [i["id"] for i in insights] == ["IN1", "EX1"]
    assert [d["id"] for d in distractors] == ["DI1"]
    assert adapter.insight_ground_truth(eval_config) == insights


def test_qa_ground_truth_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="must be `insight` or `distractor`"):
        adapter.qa_ground_truth({"dr_report_evaluation_qa": []}, "bogus")


def test_persona_regime_uses_the_persona_login_everywhere() -> None:
    task_config = {"persona": {"username": "dana.ray", "password": "my_drbench_pwd"}}
    assert adapter.credential_regime(task_config) == "persona"
    creds = adapter.app_credentials(task_config)
    assert {c["username"] for c in creds.values()} == {"dana.ray"}
    assert {c["password"] for c in creds.values()} == {"my_drbench_pwd"}


def test_persona_regime_derives_a_missing_username() -> None:
    creds = adapter.app_credentials(
        {"persona": {"first_name": "Dana", "last_name": "Ray", "password": "pw"}}
    )
    assert creds["nextcloud"]["username"] == "dana.ray"


@pytest.mark.parametrize("password", [None, "", 0])
def test_default_regime_falls_back_to_the_app_logins(password: object) -> None:
    """85 of 100 upstream tasks carry no persona password, so the apps keep their own.

    Verified against the shipped images: DR0016's documents live under Nextcloud's
    `admin` user and its mailbox is `current.user`, not the persona.
    """
    task_config = {"persona": {"username": "dana.ray", "password": password}}
    assert adapter.credential_regime(task_config) == "default"
    creds = adapter.app_credentials(task_config)
    assert creds["nextcloud"] == {"username": "admin", "password": "admin_pwd"}
    assert creds["email"] == {"username": "current.user", "password": "current_user_pwd"}
    assert creds["mattermost"]["username"] == "admin@drbench.com"
    # The persona name must not be presented as a login it cannot use.
    assert all(c["username"] != "dana.ray" for c in creds.values())


def test_credential_regime_requires_a_persona() -> None:
    with pytest.raises(ValueError, match="must hold a `persona` object"):
        adapter.credential_regime({})


def test_image_reference_is_digest_pinned(vendor: Path) -> None:
    _write_vendor(vendor, env_files=[_env_file("a.pdf")], qa=[_qa("IN1", "kept")])
    reference = adapter.image_reference(_TASK_ID)
    assert reference == f"{adapter.IMAGE_REGISTRY}@{_DIGEST}"


def test_image_reference_fails_loudly_without_a_digest(vendor: Path) -> None:
    _write_vendor(vendor, env_files=[_env_file("a.pdf")], qa=[_qa("IN1", "kept")])
    with pytest.raises(KeyError, match="No vendored image digest"):
        adapter.image_reference("DR0099")


def test_load_image_digests_rejects_a_malformed_digest(vendor: Path) -> None:
    _write_vendor(vendor, env_files=[_env_file("a.pdf")], qa=[_qa("IN1", "kept")])
    (vendor / "image_digests.json").write_text(json.dumps({"digests": {"DR0001": "latest"}}))
    with pytest.raises(ValueError, match="Malformed image digest"):
        adapter.load_image_digests()


def test_generate_task_creates_a_two_service_app_mode_task(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("inbox.jsonl", app="email")],
        qa=[_qa("IN1", "Acme tracks 250 SKUs."), _qa("DI1", "Unrelated.", qa_type="distractor")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    for relative in (
        "task.toml",
        "instruction.md",
        "environment/docker-compose.yaml",
        "environment/main.Dockerfile",
        "environment/extract_text.py",
        "environment/.dockerignore",
        "solution/solve.sh",
        "tests/case.json",
        "tests/test.sh",
        "tests/judge.py",
    ):
        assert (task_dir / relative).is_file(), relative
    # App mode serves documents from the image; nothing is laid down on disk.
    assert not (task_dir / "environment" / "files").exists()

    compose = yaml.safe_load((task_dir / "environment" / "docker-compose.yaml").read_text())
    assert sorted(compose["services"]) == ["drbench", "main"]
    assert compose["services"]["drbench"]["image"].endswith(_DIGEST)
    assert compose["services"]["drbench"]["platform"] == adapter.IMAGE_PLATFORM
    # Harbor only ever overrides `command`, and only for `main`. Overriding either key
    # on the sidecar would stop its entrypoint starting supervisord, and the app stack
    # would never come up.
    assert not {"entrypoint", "command"} & set(compose["services"]["drbench"])
    assert compose["services"]["main"]["build"]["dockerfile"] == "main.Dockerfile"

    task_toml = (task_dir / "task.toml").read_text()
    assert 'source = "drbench"' in task_toml
    assert 'mode = "app"' in task_toml
    # Open web is required: external_fact ground truth is not in the app stack. An
    # allowlist would also drag in Harbor's egress sidecar, which puts every service
    # into one network namespace.
    assert 'network_mode = "public"' in task_toml
    assert "[environment.healthcheck]" in task_toml
    assert adapter.HEALTH_URL in task_toml
    assert "insight_count = 1" in task_toml
    assert "distractor_count = 1" in task_toml
    assert "document_count = 2" in task_toml


def test_generate_task_prompt_names_the_apps_and_their_logins(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("inbox.jsonl", app="email")],
        qa=[_qa("IN1", "Acme tracks 250 SKUs.")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)
    instruction = (task_dir / "instruction.md").read_text()

    assert "Dana Ray" in instruction
    assert "/app/report.md" in instruction
    # Only the apps this task actually uses, each with its own login.
    assert "http://drbench:8081" in instruction
    assert "drbench:1143" in instruction
    assert "http://drbench:8082" not in instruction
    assert "dana.ray" in instruction
    # The agent cannot discover the extractor or the health endpoint on its own.
    assert "extract-text" in instruction
    assert adapter.HEALTH_URL in instruction
    # No corpus on disk, so the prompt must not point at one.
    assert "/app/files" not in instruction
    assert "Acme tracks 250 SKUs" not in instruction


def test_generate_task_keeps_ground_truth_out_of_the_agents_reach(
    vendor: Path, tmp_path: Path
) -> None:
    secret = "Acme tracks 250 high-risk SKUs."
    planted = "Acme repainted its head office."
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf")],
        qa=[_qa("IN1", secret), _qa("DI1", planted, qa_type="distractor")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    case = json.loads((task_dir / "tests" / "case.json").read_text())
    assert [i["answer"] for i in case["insights"]] == [secret]
    assert [d["answer"] for d in case["distractors"]] == [planted]
    # The verifier re-fetches cited documents from the app stack, so it needs the login.
    assert case["credentials"]["nextcloud"]["username"] == "dana.ray"
    assert case["endpoints"]["nextcloud"] == "http://drbench:8081"
    assert case["credential_regime"] == "persona"

    # `tests/` goes to the verifier and `solution/` is uploaded only by Harbor's
    # OracleAgent, never on a real agent run. Everything else is agent-visible.
    agent_visible = [
        path
        for path in task_dir.rglob("*")
        if path.is_file() and not {"tests", "solution"} & set(path.relative_to(task_dir).parts)
    ]
    assert agent_visible
    for path in agent_visible:
        text = path.read_text(errors="replace")
        assert secret not in text, path
        # Knowing which facts are planted distractors would let the agent skip research.
        assert planted not in text, path


def test_generate_task_is_idempotent(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    output_dir = tmp_path / "dataset"
    first = adapter.generate_task(output_dir=output_dir, task_id=_TASK_ID)
    (first / "stale.txt").write_text("should be removed")
    second = adapter.generate_task(output_dir=output_dir, task_id=_TASK_ID)
    assert not (second / "stale.txt").exists()


def test_populate_lays_down_the_invariant_files(vendor: Path, tmp_path: Path) -> None:
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)

    invariants = (
        "environment/main.Dockerfile",
        "environment/extract_text.py",
        "tests/test.sh",
        "tests/judge.py",
    )
    for relative in invariants:
        (dataset_dir / _TASK_ID / relative).unlink()

    assert adapter.populate_corpus(dataset_dir) == 1
    for relative in invariants:
        assert (dataset_dir / _TASK_ID / relative).is_file(), relative


def test_populate_ignores_foreign_task_dirs(vendor: Path, tmp_path: Path) -> None:
    """Only tasks this adapter generated may be populated by it."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"
    adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)
    foreign = dataset_dir / "cb-cloud-1"
    foreign.mkdir()
    (foreign / "task.toml").write_text('version = "1.3"\n\n[metadata]\nsource = "contextbench"\n')

    assert adapter.populate_corpus(dataset_dir) == 1
    assert not (foreign / "environment").exists()


def test_generated_task_toml_validates_against_harbors_own_schema(
    vendor: Path, tmp_path: Path
) -> None:
    """Parse with Harbor's real model, not string matching.

    Harbor's config models accept unknown keys, so a field written under the wrong table
    validates and is then silently ignored. That is how `artifacts` first shipped as a
    no-op under `[verifier]`, which has no such field -- the report was never collected
    and a zero score was indistinguishable from a broken environment.
    """
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf")],
        qa=[_qa("IN1", "kept"), _qa("DI1", "planted", qa_type="distractor")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)
    config = TaskConfig.model_validate(tomllib.load((task_dir / "task.toml").open("rb")))

    # Each of these is only meaningful if Harbor actually parsed it into the model.
    assert config.artifacts == ["/app/report.md"]
    assert config.environment.network_mode is NetworkMode.PUBLIC
    assert config.environment.healthcheck is not None
    assert adapter.HEALTH_URL in config.environment.healthcheck.command
    assert config.environment.build_timeout_sec > 600.0
    assert config.agent.timeout_sec == 3600.0
    assert config.verifier.timeout_sec == 2400.0
