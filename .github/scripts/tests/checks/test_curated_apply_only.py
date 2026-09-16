"""Tests for changelog-only curated release-note apply detection."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from curated_apply_only import is_curated_apply_only

BOT_LOGIN = "release-bot[bot]"
BOT_ID = "12345"
COMPONENT = "example"
BRANCH = f"release-please--branches--main--components--{COMPONENT}"


def _run(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _commit(repo: Path, message: str, *, bot: bool = True) -> str:
    if bot:
        _run(repo, "config", "user.name", BOT_LOGIN)
        _run(
            repo,
            "config",
            "user.email",
            f"{BOT_ID}+{BOT_LOGIN}@users.noreply.github.com",
        )
    else:
        _run(repo, "config", "user.name", "Maintainer")
        _run(repo, "config", "user.email", "maintainer@example.com")
    _run(repo, "add", ".")
    _run(repo, "commit", "-m", message)
    return _run(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    repository.mkdir()
    _run(repository, "init", "-b", "main")
    _run(repository, "config", "user.name", "Maintainer")
    _run(repository, "config", "user.email", "maintainer@example.com")
    (repository / "pkg").mkdir()
    (repository / "pkg" / "CHANGELOG.md").write_text("generated\n")
    (repository / "pkg" / "source.py").write_text("value = 1\n")
    (repository / "release-please-config.json").write_text(
        json.dumps({"packages": {"pkg": {"component": COMPONENT}}})
    )
    _commit(repository, "initial", bot=False)
    return repository


def _detect(repo: Path, head: str) -> bool:
    return is_curated_apply_only(
        repo=repo,
        config_path=repo / "release-please-config.json",
        head=head,
        branch=BRANCH,
        bot_login=BOT_LOGIN,
        bot_id=BOT_ID,
    )


def test_accepts_bot_apply_that_only_modifies_managed_changelog(repo: Path) -> None:
    (repo / "pkg" / "CHANGELOG.md").write_text("curated\n")
    head = _commit(repo, f"chore({COMPONENT}): apply curated release notes")

    assert _detect(repo, head) is True


@pytest.mark.parametrize(
    "failure", ["message", "author", "extra-file", "added-changelog"]
)
def test_rejects_untrusted_or_non_changelog_only_commits(
    repo: Path, failure: str
) -> None:
    if failure == "added-changelog":
        (repo / "pkg" / "CHANGELOG.md").unlink()
        _commit(repo, "remove changelog", bot=False)
    (repo / "pkg" / "CHANGELOG.md").write_text("curated\n")
    if failure == "extra-file":
        (repo / "pkg" / "source.py").write_text("value = 2\n")
    message = (
        "not an apply"
        if failure == "message"
        else f"chore({COMPONENT}): apply curated release notes"
    )
    head = _commit(repo, message, bot=failure != "author")

    assert _detect(repo, head) is False


def test_rejects_configured_path_traversal(repo: Path) -> None:
    (repo / "release-please-config.json").write_text(
        json.dumps(
            {
                "packages": {
                    "pkg": {"component": COMPONENT, "changelog-path": "../source.py"}
                }
            }
        )
    )
    _commit(repo, "malformed config", bot=False)
    (repo / "pkg" / "CHANGELOG.md").write_text("curated\n")
    head = _commit(repo, f"chore({COMPONENT}): apply curated release notes")

    assert _detect(repo, head) is False
