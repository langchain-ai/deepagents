"""Detect a trusted release-bot commit that only updates one managed changelog."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

BRANCH_PREFIX = "release-please--branches--main--components--"
COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _git(repo: Path, *args: str) -> str:
    """Run a read-only git command."""
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _changelog_path(config_path: Path, component: str) -> str | None:
    """Resolve one component's changelog from release-please configuration."""
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        return None
    packages = config.get("packages")
    if not isinstance(packages, dict):
        return None
    matches = []
    for package_path, metadata in packages.items():
        if not isinstance(package_path, str) or not isinstance(metadata, dict):
            return None
        if metadata.get("component") != component:
            continue
        changelog = metadata.get("changelog-path", "CHANGELOG.md")
        normalized_package = package_path.rstrip("/")
        if not isinstance(changelog, str) or not normalized_package:
            return None
        raw_parts = normalized_package.split("/") + changelog.split("/")
        if normalized_package.startswith("/") or changelog.startswith("/"):
            return None
        if any(part in {"", ".", ".."} for part in raw_parts):
            return None
        target = PurePosixPath(normalized_package, changelog)
        if target.is_absolute() or any(
            part in {"", ".", ".."} for part in target.parts
        ):
            return None
        matches.append(target.as_posix())
    return matches[0] if len(matches) == 1 else None


def is_curated_apply_only(
    *,
    repo: Path,
    config_path: Path,
    head: str,
    branch: str,
    bot_login: str,
    bot_id: str,
) -> bool:
    """Return whether HEAD is the bot's single-changelog apply commit."""
    if not branch.startswith(BRANCH_PREFIX) or not bot_id.isdigit():
        return False
    component = branch.removeprefix(BRANCH_PREFIX)
    if not COMPONENT_PATTERN.fullmatch(component):
        return False
    changelog = _changelog_path(config_path, component)
    if changelog is None:
        return False
    metadata = (
        _git(repo, "show", "-s", "--format=%an%x00%ae%x00%P%x00%s", head)
        .rstrip("\n")
        .split("\0")
    )
    if len(metadata) != 4:
        return False
    author, email, parents, subject = metadata
    if len(parents.split()) != 1:
        return False
    expected_email = f"{bot_id}+{bot_login}@users.noreply.github.com"
    expected_subject = f"chore({component}): apply curated release notes"
    if (author, email, subject) != (bot_login, expected_email, expected_subject):
        return False
    changed = _git(
        repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head
    ).splitlines()
    modified = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--diff-filter=M",
        "--name-only",
        "-r",
        head,
    ).splitlines()
    return changed == [changelog] and modified == [changelog]


def main() -> int:
    """Print a GitHub Actions boolean and fail closed on invalid state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--bot-login", required=True)
    parser.add_argument("--bot-id", required=True)
    args = parser.parse_args()
    try:
        result = is_curated_apply_only(
            repo=args.repo,
            config_path=args.config,
            head=args.head,
            branch=args.branch,
            bot_login=args.bot_login,
            bot_id=args.bot_id,
        )
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        result = False
    print(str(result).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
