"""Write a release's dependency range changes into its CHANGELOG.md section.

Run by `release-please.yml`'s `update-lockfiles` job on each release PR, after
release-please refreshes the branch. It diffs the package's `pyproject.toml`
between the previous release tag and the release branch head, then adds,
replaces, or removes the `### Dependencies` subsection of the version being
released. The GitHub release body is extracted from that section, so the
dependency list ships in both places from one source.

Floor raises usually land as hidden `chore` commits, which release-please
leaves out of the generated changelog; this is what makes them visible.

The script is stdlib-only, like `build_release_notes.py`, and fails open: a
problem computing the list warns and leaves CHANGELOG.md untouched rather than
blocking the lockfile update.

Usage:
    python .github/scripts/release/update_changelog_dependencies.py \
        --component deepagents-code --version 0.2.14
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from build_release_notes import (
    PACKAGE_MAP,
    _git,
    _is_prerelease,
    dependency_change_lines,
    resolve_previous_tag,
    upsert_dependency_section,
)


def update_changelog(repo: Path, component: str, version: str) -> list[str]:
    """Update *component*'s CHANGELOG.md in place and return any warnings."""
    warnings: list[str] = []
    working_dir = PACKAGE_MAP.get(component)
    if working_dir is None:
        return [f"Unknown component '{component}'; CHANGELOG.md left unchanged"]
    changelog_path = repo / working_dir / "CHANGELOG.md"
    if not changelog_path.is_file():
        return [f"{changelog_path} does not exist; nothing to update"]

    head = _git(repo, "rev-parse", "HEAD")
    prev_tag = resolve_previous_tag(
        repo,
        component,
        version,
        head,
        is_prerelease=_is_prerelease(version),
        warnings=warnings,
    )
    lines = dependency_change_lines(repo, working_dir, prev_tag, head, warnings)

    changelog = changelog_path.read_text(encoding="utf-8")
    updated = upsert_dependency_section(changelog, version, lines)
    if updated is None:
        warnings.append(f"No {version} section in {changelog_path}; left unchanged")
    elif updated != changelog:
        changelog_path.write_text(updated, encoding="utf-8")
        print(f"Updated Dependencies in {changelog_path}", file=sys.stderr)
    return warnings


def main() -> None:
    """Parse arguments, update the changelog, and surface warnings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--component", required=True, help="release-please component")
    parser.add_argument("--version", required=True, help="Version being released")
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="Repository root"
    )
    args = parser.parse_args()
    try:
        warnings = update_changelog(args.repo_root, args.component, args.version)
    except (OSError, UnicodeDecodeError, subprocess.CalledProcessError) as err:
        detail = getattr(err, "stderr", "") or ""
        warnings = [f"Dependency list not updated: {err} {detail}".strip()]
    for warning in warnings:
        print(f"::warning::{warning}")


if __name__ == "__main__":
    main()
