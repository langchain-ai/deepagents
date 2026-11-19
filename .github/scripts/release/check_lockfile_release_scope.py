"""Flag bump-worthy PRs that would release more than one managed package.

Why this exists:
    release-please scopes a commit to a package by the file *paths* it touches,
    not by the title scope. A bump-worthy commit (e.g. a `feat:` in `libs/code`)
    that also edits release-relevant files under another managed package path
    (a `pyproject.toml` bound, source) is attributed to that package too, and
    release-please opens a separate release PR for each.

What counts as release-relevant:
    A file under a package path that is not equal to, or under, one of that
    package's `exclude-paths` in `release-please-config.json`. Every package
    excludes its `tests/` directory and its `uv.lock`, so test-only and
    lockfile-only changes never release a package. This mirrors the patched
    release-please `CommitExclude` (see `.github/release-please/`): a commit is
    skipped for a package iff every file it touches under that package's path is
    excluded.

What it detects:
    **multi_component** — two or more managed packages have at least one
    release-relevant change in a bump-worthy PR.

How it stays faithful to release-please:
    - Package paths, components, and exclude paths are read straight from
        `release-please-config.json` (the same source release-please scopes
        against) — no second list to drift.
    - The set of "bump-worthy" types is derived from the non-hidden entries in that
        config's `changelog-sections`. This is a deliberately conservative *superset*
        of release-please's actual bump triggers (`feat`/`fix` + breaking): `perf` and
        `revert` are visible sections that may not always bump. We accept the rare
        false positive on a `perf`/`revert`-only multi-package PR to avoid missing a
        real fan-out; such a PR can be cleared with the `ci:allow-lockfile-release`
        label.

    This script only *reports* offenders (and always exits 0 on success). The
    blocking decision — failing the check — lives in the workflow that calls it
    (`release_please_scope_check.yml`), not here.

Limitations:
    - Breaking changes are detected via the `!` title shorthand only. A
        `BREAKING CHANGE:` footer with no `!` is not inspected (the workflow passes
        the title, not the body). The worst case is an unflagged fan-out a maintainer
        catches at release-please time, so acceptable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"

# Conventional-commit title: lowercase type, optional (scope), optional `!`.
_TITLE_RE = re.compile(r"^([a-z]+)(?:\([^)]*\))?(!)?:\s")


class FanoutResult(TypedDict):
    """Structured fan-out report consumed by the CI workflow."""

    multi_component: list[str]
    # Always empty: lockfiles are excluded and never release a package. The key
    # stays so the scope-check workflow on PR branches cut before lockfiles were
    # excluded (which runs this base-revision detector) still parses the output.
    lockfile_only: list[str]


def bump_worthy_types(config: dict) -> frozenset[str]:
    """Return the conventional-commit types that may trigger a release.

    Derived from the non-hidden entries in the config's `changelog-sections`
    so the check tracks the repo's own configuration instead of a hardcoded
    list. See module docstring for why this is a conservative superset.

    Args:
        config: Parsed `release-please-config.json`.

    Returns:
        Set of lowercase commit types considered bump-worthy.
    """
    sections = config.get("changelog-sections", [])
    return frozenset(
        s["type"] for s in sections if s.get("type") and not s.get("hidden", False)
    )


def parse_title(title: str) -> tuple[str | None, bool]:
    """Return `(type, is_breaking)` parsed from a conventional-commit title.

    Args:
        title: The PR title (e.g. `feat(sdk): add thing` or `fix!: oops`).

    Returns:
        Tuple of the lowercase type (or `None` if the title is not
        conventional-commit-shaped) and whether the `!` breaking marker is set.
    """
    match = _TITLE_RE.match(title)
    if not match:
        return None, False
    return match.group(1), bool(match.group(2))


def is_bump_worthy(title: str, config: dict) -> bool:
    """Return whether the PR title would cut a release under our configuration.

    Args:
        title: The PR title.
        config: Parsed `release-please-config.json`.

    Returns:
        `True` when the title is breaking or uses a non-hidden changelog type.
    """
    ttype, breaking = parse_title(title)
    return bool(breaking or (ttype is not None and ttype in bump_worthy_types(config)))


def _normalize(path: str) -> str:
    """Strip leading/trailing slashes, as release-please's `normalizePaths` does."""
    return path.strip("/")


def _is_under(file: str, path: str) -> bool:
    """Return whether `file` equals `path` or lives under directory `path`."""
    return file == path or file.startswith(f"{path}/")


def release_relevant_files(changed: list[str], path: str, meta: dict) -> list[str]:
    """Return the files in `changed` that would release the package at `path`.

    A file counts when it is under the package path and not equal to, or under,
    any of the package's `exclude-paths`.

    Args:
        changed: Changed file paths, repo-root-relative (forward slashes).
        path: Package path (a key of the config's `packages` map).
        meta: The package's config entry.

    Returns:
        The release-relevant subset of `changed`, in input order.
    """
    path = _normalize(path)
    excludes = [_normalize(e) for e in meta.get("exclude-paths", []) or []]
    return [
        f
        for f in changed
        if f.startswith(f"{path}/") and not any(_is_under(f, e) for e in excludes)
    ]


def touched_components(changed: list[str], config: dict) -> list[str]:
    """Return sorted components that `changed` would release.

    A component is included when at least one changed file is release-relevant
    for it (see `release_relevant_files`). Files under `exclude-paths`, such as
    `tests/` and `uv.lock`, do not count.

    Args:
        changed: Changed file paths, repo-root-relative (forward slashes).
        config: Parsed `release-please-config.json`.

    Returns:
        Sorted list of component names.
    """
    return sorted(
        meta.get("component", path)
        for path, meta in config.get("packages", {}).items()
        if release_relevant_files(changed, path, meta)
    )


def find_fanout(title: str, changed: list[str], config: dict) -> FanoutResult:
    """Return the components a bump-worthy PR would release, if more than one.

    Args:
        title: The PR title.
        changed: Changed file paths, repo-root-relative (forward slashes).
        config: Parsed `release-please-config.json`.

    Returns:
        `FanoutResult` whose `multi_component` lists every component with a
        release-relevant change when there are two or more and the title is
        bump-worthy, and is empty otherwise.
    """
    components = []
    if is_bump_worthy(title, config):
        components = touched_components(changed, config)
    return {
        "multi_component": components if len(components) >= 2 else [],
        "lockfile_only": [],
    }


def main(title: str, changed: list[str], config_path: Path = DEFAULT_CONFIG) -> int:
    """Print a fan-out report as JSON to stdout.

    Errors are surfaced loudly (not swallowed) and exit non-zero so a broken
    config or unreadable input fails the CI step visibly rather than silently
    reporting "no offenders."

    Returns:
        `0` always on successful analysis — offenders are reported on stdout and
            the blocking decision is made by the calling workflow,
            not this script.
        `2` on an internal error (missing/invalid/empty config).
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"::error::Could not read release-please config {config_path}: {e}",
            file=sys.stderr,
        )
        return 2

    # Fail closed on config drift rather than silently degrading to a no-op gate
    # that passes everything: no packages means nothing is ever scoped, and no
    # non-hidden changelog-sections means no title is ever bump-worthy.
    if not isinstance(config.get("packages"), dict) or not config["packages"]:
        print(
            f"::error::release-please config {config_path} has no 'packages' map",
            file=sys.stderr,
        )
        return 2
    if not bump_worthy_types(config):
        print(
            f"::error::release-please config {config_path} has no non-hidden "
            "changelog-sections; cannot determine bump-worthy types",
            file=sys.stderr,
        )
        return 2

    result = find_fanout(title, changed, config)
    if result["multi_component"]:
        print(
            "multi-component release fan-out for: "
            + ", ".join(result["multi_component"]),
            file=sys.stderr,
        )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: check_lockfile_release_scope.py <pr-title>  (changed files on stdin)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    pr_title = sys.argv[1]
    changed_files = [line.strip() for line in sys.stdin if line.strip()]
    raise SystemExit(main(pr_title, changed_files))
