"""Tests for the DRBench Harbor task adapter (app mode)."""

from __future__ import annotations

import json
import re
import tomllib
from typing import TYPE_CHECKING

import pytest
import yaml
from harbor.models.task.config import NetworkMode, TaskConfig, VerifierEnvironmentMode
from harbor.models.task.verifier_mode import (
    resolve_effective_verifier_env_config,
    resolve_task_verifier_mode,
)

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
    """Write a minimal upstream config bundle plus the vendored pins for one task.

    Configs go into the fixture checkout in upstream's own layout (the three scoring
    configs under `config/`, the label file at the task root); the subset list and image
    digest go into the fixture vendor directory, which is where they stay committed.
    """
    task_root = adapter.ensure_upstream_checkout() / "drbench" / "data" / "tasks" / task_id
    (task_root / "config").mkdir(parents=True)
    (task_root / "config" / "task.json").write_text(
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
    (task_root / "config" / "env.json").write_text(json.dumps({"env_files": env_files}))
    (task_root / "config" / "eval.json").write_text(json.dumps({"dr_report_evaluation_qa": qa}))
    (task_root / "info.json").write_text(
        json.dumps({"industry": "retail", "domain": "compliance", "difficulty": "easy"})
    )
    (vendor / "subsets" / "val.jsonl").write_text(
        json.dumps({"task_id": task_id, "path": f"drbench/data/tasks/{task_id}/config"}) + "\n"
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
    """Point the adapter at fixture pins and a fixture upstream checkout.

    `ensure_upstream_checkout` is the single seam that would otherwise reach the network;
    replacing it keeps these tests offline while leaving every path that reads a config
    under test.
    """
    vendor_dir = tmp_path / "vendor"
    (vendor_dir / "subsets").mkdir(parents=True)
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    monkeypatch.setattr(adapter, "vendor_dir", lambda: vendor_dir)
    monkeypatch.setattr(adapter, "ensure_upstream_checkout", lambda: checkout)
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
    # No `platform:`: upstream ships a single-entry arm64 OCI index, so letting Docker
    # match the host makes an amd64 runner fail at pull rather than silently emulate.
    assert "platform" not in compose["services"]["drbench"]
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

    # The verifier installs upstream `drbench`, which ships both the gold `eval.json` and
    # the document corpus as package data, so it needs nothing but the task id. Ground
    # truth is therefore absent from the task directory entirely -- stricter than relying
    # on Harbor to upload `tests/` only after the agent has finished.
    case = json.loads((task_dir / "tests" / "case.json").read_text())
    assert set(case) == {"task_id", "upstream_sha"}
    assert case["task_id"] == _TASK_ID
    assert case["upstream_sha"] == adapter.UPSTREAM_SHA

    # No generated file may carry a gold answer, with one deliberate exception: the oracle
    # solution, which exists to write them out and is uploaded only by Harbor's OracleAgent.
    oracle = task_dir / "solution" / "solve.sh"
    assert secret in oracle.read_text()
    leaked = [
        str(path.relative_to(task_dir))
        for path in sorted(task_dir.rglob("*"))
        if path.is_file()
        and path != oracle
        and any(answer in path.read_text(errors="replace") for answer in (secret, planted))
    ]
    assert leaked == []

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
    with (task_dir / "task.toml").open("rb") as f:
        config = TaskConfig.model_validate(tomllib.load(f))

    # Each of these is only meaningful if Harbor actually parsed it into the model.
    assert config.artifacts == ["/app/report.md"]
    assert config.environment.network_mode is NetworkMode.PUBLIC
    assert config.environment.healthcheck is not None
    assert adapter.HEALTH_URL in config.environment.healthcheck.command
    assert config.environment.build_timeout_sec > 600.0
    assert config.agent.timeout_sec == 3600.0
    assert config.verifier.timeout_sec == 2400.0


# --- separate verifier environment ----------------------------------------------------


def test_generated_task_declares_a_separate_verifier_environment(
    vendor: Path, tmp_path: Path
) -> None:
    """Validate with Harbor's own models, not string matching.

    `TaskConfig` ignores unknown keys, so a misplaced or misspelled table validates and is
    then silently dropped -- which is exactly how an earlier `artifacts` key ended up
    doing nothing. Resolving the mode through Harbor's resolver is the only check that
    proves the setting took effect.
    """
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "A fact.")])
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)
    config = TaskConfig.model_validate(tomllib.loads((task_dir / "task.toml").read_text()))

    assert resolve_task_verifier_mode(config) is VerifierEnvironmentMode.SEPARATE

    verifier_env = resolve_effective_verifier_env_config(config, None)
    assert verifier_env is not None
    # Must be declared rather than inherited: without an explicit table Harbor deep-copies
    # `[environment]`, which would boot the app stack again and then fail its healthcheck,
    # since the verifier compose file has no such service.
    assert verifier_env.healthcheck is None
    assert verifier_env.network_mode.value == "public"

    # The report has to survive into the verifier environment, which only happens if
    # `artifacts` is a top-level key that actually validates.
    assert config.artifacts


def test_generated_task_ships_the_verifier_image_pinned_to_the_vendored_commit(
    vendor: Path, tmp_path: Path
) -> None:
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "A fact.")])
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    dockerfile = (task_dir / "tests" / "Dockerfile").read_text()
    # One pin for the metrics, prompts, ground truth, and corpus, shared with the
    # vendored task configs so a score is always traceable to one upstream commit.
    assert adapter.UPSTREAM_SHA in dockerfile
    assert "{drbench_ref}" not in dockerfile
    # The build must fail on an install without package data, or factuality would score
    # every claim unsupported while the image looks healthy.
    assert "data" in dockerfile
    assert "assert" in dockerfile

    # In separate-verifier mode Harbor passes `skip_tests_upload=True` and then executes
    # `/tests/test.sh` directly, so the verifier files must be baked into the image --
    # nothing puts them there at run time. Without this the trial dies with "not found".
    assert "COPY . /tests" in dockerfile
    # And the copy must come after the install, or a per-task `case.json` would invalidate
    # the pip layer and rebuild pandas/faiss/pymupdf for every task.
    assert dockerfile.index("pip install") < dockerfile.index("COPY . /tests")

    compose = (task_dir / "tests" / "docker-compose.yaml").read_text()
    # Harbor runs test.sh in the service named `main`.
    assert "main:" in compose
    # No app stack here: upstream resolves cited sources from the installed corpus.
    assert "drbench:" not in compose


# --- prompt contracts ------------------------------------------------------------------


def test_instruction_names_the_citation_forms_the_verifier_can_resolve(
    vendor: Path, tmp_path: Path
) -> None:
    # Upstream resolves an email citation by exact sender address and subject, and a chat
    # citation by channel/team/user. A citation it cannot resolve scores as unsupported no
    # matter how accurate the claim, so the prompt has to specify the forms.
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("mail.jsonl", app="email")],
        qa=[_qa("IN1", "A fact.")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)
    instruction = (task_dir / "instruction.md").read_text()

    assert "RoundCube-" in instruction
    assert "MatterMost-" in instruction
    # A display name cannot resolve -- every upstream pattern requires an `@` -- and the
    # subject is matched character for character.
    assert "email address" in instruction
    assert "exactly" in instruction


def test_instruction_passes_app_passwords_through_the_environment(
    vendor: Path, tmp_path: Path
) -> None:
    """No literal `-u user:password` in a committed prompt.

    These logins are public benchmark fixtures, not secrets, but repeating the pair across
    100 committed prompts trips secret scanners on every push -- a false positive that
    trains people to ignore real alerts. The values live in one labelled table per task.
    """
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("mail.jsonl", app="email")],
        qa=[_qa("IN1", "A fact.")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)
    instruction = (task_dir / "instruction.md").read_text()

    assert re.search(r"-u\s+[A-Za-z0-9._@-]+:[A-Za-z0-9._@-]+", instruction) is None
    assert "$DRBENCH_NEXTCLOUD_PASS" in instruction
    # `_USER` too, not just `_PASS`: the login-name substitution once rewrote the token
    # inside this variable's own name, and `_PASS` has no `USER` substring to notice it.
    assert "$DRBENCH_NEXTCLOUD_USER" in instruction

    env = tomllib.loads((task_dir / "task.toml").read_text())["environment"]["env"]
    assert env["DRBENCH_NEXTCLOUD_USER"] == "dana.ray"
    assert env["DRBENCH_NEXTCLOUD_PASS"]
    assert env["DRBENCH_EMAIL_USER"] == "dana.ray"


def test_no_generated_file_carries_a_literal_curl_credential_pair(
    vendor: Path, tmp_path: Path
) -> None:
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "A fact.")])
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    offenders = [
        str(path.relative_to(task_dir))
        for path in sorted(task_dir.rglob("*"))
        if path.is_file()
        and re.search(r"-u\s+[A-Za-z0-9._@-]+:[A-Za-z0-9._@-]+", path.read_text(errors="replace"))
    ]
    assert offenders == []


def test_populate_builds_every_task_from_an_empty_dataset_directory(
    vendor: Path, tmp_path: Path
) -> None:
    """No task directory is committed, so populate has to create them, not just fill them."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"

    assert adapter.populate_corpus(dataset_dir) == 1
    task_dir = dataset_dir / _TASK_ID
    for relative in (
        "task.toml",
        "instruction.md",
        "solution/solve.sh",
        "environment/docker-compose.yaml",
        "environment/main.Dockerfile",
        "tests/case.json",
        "tests/judge.py",
        "tests/Dockerfile",
    ):
        assert (task_dir / relative).is_file(), relative


def test_record_reads_upstreams_split_config_layout(vendor: Path) -> None:
    """Upstream keeps the scoring configs under `config/` and the labels at the task root."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    record = adapter.record_for_task_id(_TASK_ID)

    assert sorted(record) == ["env", "eval", "info", "task"]
    assert record["task"]["task_id"] == _TASK_ID
    assert record["info"]["difficulty"] == "easy"


def test_generate_refuses_to_remove_a_task_dir_outside_the_dataset(
    vendor: Path, tmp_path: Path
) -> None:
    """A symlinked task directory must not redirect the pre-generation cleanup."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("must survive")
    (dataset_dir / _TASK_ID).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="not a direct child"):
        adapter.generate_task(output_dir=dataset_dir, task_id=_TASK_ID)
    assert (outside / "keep.txt").is_file()


def test_upstream_checkout_rejects_a_sha_that_is_not_a_full_commit_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated before any git command runs, so a typo cannot become a fetch argument."""
    monkeypatch.setenv("DRBENCH_UPSTREAM_DIR", str(tmp_path / "upstream"))
    monkeypatch.setattr(adapter, "UPSTREAM_SHA", "main")

    with pytest.raises(ValueError, match="full 40-character commit hash"):
        adapter.ensure_upstream_checkout()
    assert not (tmp_path / "upstream").exists()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('["DR0001"]', "must hold a JSON object"),
        ('{"path": "x"}', "no string `task_id`"),
        ('{"task_id": 1}', "no string `task_id`"),
        ('{"task_id": "../escape"}', "must be a DRBench id"),
    ],
)
def test_subset_reader_rejects_a_malformed_entry(tmp_path: Path, line: str, message: str) -> None:
    subset = tmp_path / "val.jsonl"
    subset.write_text(line + "\n")

    with pytest.raises(ValueError, match=message):
        adapter.read_subset_task_ids(subset)


def test_the_vendored_pins_cover_exactly_the_same_tasks() -> None:
    """The two committed pins must agree, or generation fails partway through a run.

    `val.jsonl` decides which tasks `--all` builds and `image_digests.json` supplies the
    image each one runs, so a task in the first without an entry in the second is a
    `KeyError` after the dataset is already half written.
    """
    task_ids = adapter.available_task_ids()

    assert len(task_ids) == 100
    assert sorted(adapter.load_image_digests()) == task_ids


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"labels": []}', "must hold a `labels` object"),
        ('{"labels": {"DR0001": "easy"}}', "must be an object"),
        ('{"labels": {"DR0001": {"difficulty": "easy"}}}', "missing string"),
        ('{"labels": {"../escape": {}}}', "must be a DRBench id"),
    ],
)
def test_label_reader_rejects_a_malformed_record(vendor: Path, payload: str, message: str) -> None:
    (vendor / "task_labels.json").write_text(payload)

    with pytest.raises(ValueError, match=message):
        adapter.load_task_labels()


def test_label_verification_reports_a_stale_record(vendor: Path) -> None:
    """A bumped `UPSTREAM_SHA` with unrefreshed labels is the one drift worth catching."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    assert adapter.refresh_task_labels() == 1
    assert adapter.verify_task_labels() == []

    record = json.loads((vendor / "task_labels.json").read_text())
    record["labels"][_TASK_ID]["difficulty"] = "hard"
    record["upstream_sha"] = "0" * 40
    (vendor / "task_labels.json").write_text(json.dumps(record))

    problems = adapter.verify_task_labels()
    assert any("upstream_sha" in problem for problem in problems)
    assert any(_TASK_ID in problem for problem in problems)


def test_the_committed_labels_cover_every_task() -> None:
    """The label record and the subset list must name the same tasks."""
    assert sorted(adapter.load_task_labels()) == adapter.available_task_ids()


def test_every_env_var_the_prompt_names_is_actually_declared(vendor: Path, tmp_path: Path) -> None:
    """The stronger invariant: no prompt may reference a variable the task does not set.

    Checked as a property rather than per variable, because the way this breaks is a
    placeholder substitution *inventing* a name -- `$DRBENCH_NEXTCLOUD_USER` became
    `$DRBENCH_NEXTCLOUD_admin` in 96 of 100 prompts, which expands to nothing and sends an
    empty login. Asserting each known name individually would not have caught it.
    """
    _write_vendor(
        vendor,
        env_files=[_env_file("report.pdf"), _env_file("thread.jsonl", app="email")],
        qa=[_qa("IN1", "A fact.")],
    )
    task_dir = adapter.generate_task(output_dir=tmp_path / "dataset", task_id=_TASK_ID)

    declared = set(tomllib.loads((task_dir / "task.toml").read_text())["environment"]["env"])
    referenced = set(
        re.findall(r"\$(DRBENCH_[A-Za-z0-9_]+)", (task_dir / "instruction.md").read_text())
    )
    assert referenced, "the prompt names no DRBench variable at all"
    assert referenced <= declared, f"undeclared: {sorted(referenced - declared)}"


def test_populate_prunes_a_task_that_left_the_authoritative_set(
    vendor: Path, tmp_path: Path
) -> None:
    """A reused checkout must not keep serving tasks the current pin dropped.

    Harbor enumerates a local dataset by listing directories, so a leftover directory is
    a task that runs and scores while being outside the set the numbers claim to cover.
    `make dataset-check` cannot catch this: it builds into fresh directories.
    """
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    # A task from an earlier pin: this adapter generated it, and it is no longer listed.
    stale = dataset_dir / "DR0099"
    stale.mkdir()
    (stale / "task.toml").write_text('version = "1.3"\n\n[metadata]\nsource = "drbench"\n')

    assert adapter.populate_corpus(dataset_dir) == 1

    assert not (dataset_dir / "DR0099").exists()
    assert (dataset_dir / _TASK_ID / "task.toml").is_file()


def test_pruning_leaves_directories_this_adapter_did_not_generate(
    vendor: Path, tmp_path: Path
) -> None:
    """Four conditions gate the removal; failing any one keeps the directory."""
    _write_vendor(vendor, env_files=[_env_file("report.pdf")], qa=[_qa("IN1", "kept")])
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    # Right name shape, but another adapter's task.
    foreign = dataset_dir / "DR0098"
    foreign.mkdir()
    (foreign / "task.toml").write_text('version = "1.3"\n\n[metadata]\nsource = "contextbench"\n')
    # Not a DRBench id at all.
    unrelated = dataset_dir / "cb-cloud-1"
    unrelated.mkdir()
    (unrelated / "task.toml").write_text('version = "1.3"\n\n[metadata]\nsource = "contextbench"\n')

    assert adapter.prune_stale_tasks(dataset_dir, keep={_TASK_ID}) == []
    assert (foreign / "task.toml").is_file()
    assert (unrelated / "task.toml").is_file()


def test_subset_verification_reports_a_vendored_copy_that_drifted(vendor: Path) -> None:
    """The vendored list is the denominator of every score, so it must match the pin."""
    upstream_subsets = adapter.ensure_upstream_checkout() / "drbench" / "data" / "subsets"
    upstream_subsets.mkdir(parents=True)
    payload = json.dumps({"task_id": _TASK_ID, "path": "x"}) + "\n"
    (upstream_subsets / "val.jsonl").write_text(payload)
    (vendor / "subsets" / "val.jsonl").write_text(payload)
    assert adapter.verify_subsets() == []

    (vendor / "subsets" / "val.jsonl").write_text(payload + payload)
    assert [p for p in adapter.verify_subsets() if "differs" in p]

    (vendor / "subsets" / "extra.jsonl").write_text(payload)
    assert [p for p in adapter.verify_subsets() if "absent from upstream" in p]
