"""Test release workflow steps that remain inline in release.yml.

The release-notes job's body-construction steps (resolve-refs, generate-git-log,
generate-release-body, finalize-release-body) have been extracted into the
shared script at .github/scripts/release/build_release_notes.py. Their behavior
is covered by test_build_release_notes.py. This file retains tests for workflow
steps that are still inline bash, such as the setup job's resolve-sha step.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
PACKAGE_PATH = Path("libs/example")
REPOSITORY = "langchain-ai/deepagents"


def _git(repo: Path, *args: str) -> str:
    """Run git in *repo*, raising with git's own diagnosis when it fails.

    `check=True` would raise a `CalledProcessError` whose message carries only
    argv and the exit status, which turns an infrastructure failure into an
    undebuggable report.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed (exit {result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.email", "release-test@example.com")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "commit.gpgSign", "false")
    # Background maintenance forked by `git commit` can hold the repository
    # locks while the next command runs, failing it with a lock error.
    _git(repo, "config", "gc.auto", "0")
    _git(repo, "config", "maintenance.auto", "false")


def _commit(repo: Path, path: Path, content: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    _git(repo, "add", str(path))
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _workflow_step(step_id: str, *, job: str = "release-notes") -> str:
    with WORKFLOW_PATH.open() as file:
        workflow = yaml.safe_load(file)
    steps = workflow["jobs"][job]["steps"]
    return next(step["run"] for step in steps if step.get("id") == step_id)


def test_release_notes_script_checkout_uses_non_cone_mode() -> None:
    with WORKFLOW_PATH.open() as file:
        workflow = yaml.safe_load(file)
    steps = workflow["jobs"]["release-notes"]["steps"]
    checkout = next(
        step for step in steps if step.get("name") == "Check out release-notes script"
    )

    assert checkout["with"]["sparse-checkout-cone-mode"] is False


def test_temporary_repo_disables_inherited_commit_signing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("[commit]\n\tgpgSign = true\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    repo = tmp_path / "repo"
    repo.mkdir()

    _init_repo(repo)

    assert _git(repo, "config", "--local", "--bool", "commit.gpgSign") == "false"


@pytest.mark.parametrize(
    ("input_sha", "source"),
    [
        ("", "workflow SHA fallback because dangerous-nonmain-release is enabled"),
        ("HEAD", "explicit release-sha input"),
    ],
)
def test_resolve_release_sha_outputs_canonical_target_summary(
    tmp_path: Path,
    input_sha: str,
    source: str,
) -> None:
    _init_repo(tmp_path)
    sha = _commit(
        tmp_path,
        PACKAGE_PATH / "module.py",
        "BASE = 1\n",
        "feat(example): base",
    )
    output = tmp_path / "setup-output.txt"
    summary = tmp_path / "step-summary.md"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA_FALLBACK": "HEAD",
        "GITHUB_STEP_SUMMARY": str(summary),
        "INPUT_SHA": input_sha,
        "INPUT_VERSION": "1.1.0a1",
        "IS_DANGEROUS": "true",
        "PACKAGE": "example",
        "WORKING_DIR": str(PACKAGE_PATH),
    }

    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _workflow_step("resolve-sha", job="setup")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    outputs = dict(
        line.split("=", maxsplit=1) for line in output.read_text().splitlines()
    )
    assert outputs["sha"] == sha
    assert (
        f"| Release SHA | [`{sha[:7]}`](https://github.com/{REPOSITORY}/commit/{sha}) |"
        in summary.read_text()
    )
    assert f"| Resolution | {source} |" in summary.read_text()


def test_resolve_release_sha_rejects_unresolvable_sha(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    _commit(tmp_path, PACKAGE_PATH / "module.py", "BASE = 1\n", "feat(example): base")
    output = tmp_path / "setup-output.txt"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_SHA_FALLBACK": "HEAD",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
        "INPUT_SHA": "does-not-exist",
        "INPUT_VERSION": "1.1.0a1",
        "IS_DANGEROUS": "true",
        "PACKAGE": "example",
        "WORKING_DIR": str(PACKAGE_PATH),
    }

    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _workflow_step("resolve-sha", job="setup")],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "does not resolve to a commit" in result.stdout
    # The failure aborts before the `sha=` output is written.
    assert not output.exists() or "sha=" not in output.read_text()
