"""Flag open release-please PRs whose unreleased package delta is lockfile-only.

Post-merge safety net for the pre-merge gates in
`release_please_scope_check.yml`. Even when a bump-worthy multi-package PR
slips through, this report surfaces open `release(<component>):` PRs whose
component path — relative to the last-released git SHA in
`.release-please-manifest.json` — only changed lockfiles on `main`.

What counts as lockfile-only unreleased delta:
    For each managed package path, take `git diff --name-only <baseline> HEAD`
    restricted to that path. If the component has an open release-please PR and
    every changed path under the package is a lockfile name (`uv.lock`), the
    component is reported.

    Files that only exist because the open release PR rewrote version metadata
    on its branch are *not* considered here: the diff is against `main` at the
    checkout HEAD (post push), not against the release branch tip.

The script only *reports* offenders as JSON on stdout. The calling workflow
posts sticky comments / fails the advisory job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"
DEFAULT_MANIFEST = REPO_ROOT / ".release-please-manifest.json"

LOCKFILE_NAMES = frozenset({"uv.lock"})


def _run_git(args: list[str], *, cwd: Path) -> str:
    """Run a git command and return stdout text, raising on failure."""
    completed = subprocess.run(  # fixed argv, no shell
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        msg = (
            f"git {' '.join(args)!r} failed (rc={completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
        raise RuntimeError(msg)
    return completed.stdout


def package_unreleased_files(
    path: str,
    baseline: str,
    *,
    repo_root: Path,
    head: str = "HEAD",
) -> list[str]:
    """Return repo-root-relative files under `path` changed since `baseline`.

    Args:
        path: Managed package directory from `release-please-config.json`.
        baseline: Last-released git SHA recorded in the manifest.
        repo_root: Repository root used as the git cwd.
        head: Tip ref to diff against (default `HEAD`).

    Returns:
        Sorted list of changed file paths under the package directory.

    Raises:
        RuntimeError: If the git invocation fails.
    """
    # `path` is a directory; trailing slash keeps the path filter tight.
    filter_path = path if path.endswith("/") else f"{path}/"
    out = _run_git(
        ["diff", "--name-only", f"{baseline}..{head}", "--", filter_path],
        cwd=repo_root,
    )
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def is_lockfile_only(files: list[str]) -> bool:
    """Return whether every path is a known lockfile name."""
    return bool(files) and all(Path(f).name in LOCKFILE_NAMES for f in files)


def find_lockfile_only_components(
    config: dict,
    manifest: dict[str, str],
    *,
    repo_root: Path,
    head: str = "HEAD",
) -> list[dict[str, object]]:
    """Return components whose unreleased package delta is lockfile-only.

    Args:
        config: Parsed `release-please-config.json`.
        manifest: Parsed `.release-please-manifest.json` (path -> baseline SHA).
        repo_root: Repository root for git diffs.
        head: Tip ref to diff against.

    Returns:
        Sorted list of dicts with `component`, `path`, `baseline`, and `files`.
    """
    packages = config.get("packages", {})
    offenders: list[dict[str, object]] = []
    for path, meta in packages.items():
        if not isinstance(path, str) or not isinstance(meta, dict):
            continue
        baseline = manifest.get(path)
        if not baseline:
            continue
        # Baseline SHA missing from shallow clones raises RuntimeError so CI
        # fails closed rather than silently skipping a package.
        files = package_unreleased_files(
            path, baseline, repo_root=repo_root, head=head
        )
        if is_lockfile_only(files):
            offenders.append(
                {
                    "component": meta.get("component", path),
                    "path": path,
                    "baseline": baseline,
                    "files": files,
                }
            )
    return sorted(offenders, key=lambda o: str(o["component"]))


def main(
    *,
    config_path: Path = DEFAULT_CONFIG,
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
    head: str = "HEAD",
) -> int:
    """Print lockfile-only unreleased components as a JSON array.

    Returns:
        `0` on successful analysis.
        `2` on missing/invalid config, manifesto or git failure.
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"::error::Could not read release-please config/manifest: {e}", file=sys.stderr)
        return 2

    if not isinstance(config.get("packages"), dict) or not config["packages"]:
        print(
            f"::error::{config_path} has no non-empty 'packages' map",
            file=sys.stderr,
        )
        return 2
    if not isinstance(manifest, dict) or not manifest:
        print(
            f"::error::{manifest_path} is missing or empty",
            file=sys.stderr,
        )
        return 2

    try:
        offenders = find_lockfile_only_components(
            config, manifest, repo_root=repo_root, head=head
        )
    except RuntimeError as e:
        print(f"::error::{e}", file=sys.stderr)
        return 2

    if offenders:
        names = ", ".join(str(o["component"]) for o in offenders)
        print(f"Lockfile-only open-release candidates: {names}", file=sys.stderr)
    print(json.dumps(offenders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
