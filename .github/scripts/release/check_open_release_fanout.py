"""Flag open release-please PRs whose unreleased package delta is lockfile-only.

Post-merge safety net for the pre-merge gates in
`release_please_scope_check.yml`. Even when a bump-worthy multi-package PR
slips through, this report surfaces open `release(<component>):` PRs whose
component path — relative to the last *released package version* in
`.release-please-manifest.json` — only changed lockfiles on `main`.

What counts as lockfile-only unreleased delta:
    For each managed package path, resolve the manifest version to the package
    release tag (e.g. `deepagents-cli==0.2.2`, matching `tag-separator` /
    `include-component-in-tag` in `release-please-config.json`), then take
    `git diff --name-only <tag> HEAD` restricted to that path. If the component
    has an open release-please PR and every changed path under the package is a
    lockfile name (`uv.lock`), the component is reported.

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

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"
DEFAULT_MANIFEST = REPO_ROOT / ".release-please-manifest.json"

LOCKFILE_NAMES = frozenset({"uv.lock"})
DEFAULT_TAG_SEPARATOR = "=="


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


def tag_separator(config: dict) -> str:
    """Return the release-please tag separator (default `==`)."""
    sep = config.get("tag-separator", DEFAULT_TAG_SEPARATOR)
    if not isinstance(sep, str) or not sep:
        return DEFAULT_TAG_SEPARATOR
    return sep


def release_tag(component: str, version: str, *, separator: str = DEFAULT_TAG_SEPARATOR) -> str:
    """Build the git tag name for a released component version.

    Matches release-please settings used in this repo:
    `include-component-in-tag: true`, `include-v-in-tag: false`,
    `tag-separator: "=="` → `{component}=={version}`.

    Args:
        component: release-please component name (e.g. `deepagents-cli`).
        version: Manifest version string (e.g. `0.2.2`).
        separator: Tag separator from config.

    Returns:
        Tag name such as `deepagents-cli==0.2.2`.
    """
    return f"{component}{separator}{version}"


def _ref_exists(ref: str, *, repo_root: Path) -> bool:
    """Return whether `ref` resolves to a git object.

    Uses `rev-parse --verify` so a missing release tag is distinguishable from
    a `git diff` that failed for some other reason (see the caller in
    `find_lockfile_only_components`).

    Args:
        ref: Git ref to resolve.
        repo_root: Repository root used as the git cwd.

    Returns:
        `True` when the ref resolves and `False` when it is absent.

    Raises:
        RuntimeError: If git fails for a reason other than an absent ref.
    """
    completed = subprocess.run(  # fixed argv, no shell
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    msg = (
        f"git 'rev-parse --verify --quiet {ref}' failed "
        f"(rc={completed.returncode}): "
        f"{completed.stderr.strip() or completed.stdout.strip()}"
    )
    raise RuntimeError(msg)


def package_unreleased_files(
    path: str,
    baseline_ref: str,
    *,
    repo_root: Path,
    head: str = "HEAD",
) -> list[str]:
    """Return repo-root-relative files under `path` changed since `baseline_ref`.

    Args:
        path: Managed package directory from `release-please-config.json`.
        baseline_ref: Git ref for the last released tip (usually a release tag).
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
        ["diff", "--name-only", f"{baseline_ref}..{head}", "--", filter_path],
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
        manifest: Parsed `.release-please-manifest.json` (path -> version).
        repo_root: Repository root for git diffs.
        head: Tip ref to diff against.

    Returns:
        Sorted list of dicts with `component`, `path`, `version`, `baseline`
        (resolved release tag), and `files`. Components whose manifest version
        has no published release tag yet are skipped (with a stderr warning)
        rather than treated as offenders.

    Raises:
        RuntimeError: If a git diff fails for an existing release tag.
    """
    packages = config.get("packages", {})
    sep = tag_separator(config)
    offenders: list[dict[str, object]] = []
    for path, meta in packages.items():
        if not isinstance(path, str) or not isinstance(meta, dict):
            continue
        version = manifest.get(path)
        if not version or not isinstance(version, str):
            continue
        component = meta.get("component") or meta.get("package-name") or path
        if not isinstance(component, str) or not component:
            continue
        baseline = release_tag(component, version, separator=sep)
        if not _ref_exists(baseline, repo_root=repo_root):
            # The manifest was bumped (release PR merged) but the tag is not
            # published yet — pre-release checks may still be running or have
            # failed. There is no released baseline to diff against, so skip
            # this component for this run instead of failing closed; the watch
            # re-checks on the next release-please run / hourly cron, by which
            # point the tag exists. A genuinely deleted tag still surfaces here
            # on every run until restored.
            print(
                f"::warning::Skipping {component}: release tag '{baseline}' "
                f"(manifest {version}) not found; release likely unpublished.",
                file=sys.stderr,
            )
            continue
        # A diff failure for an existing tag (shallow clone without history,
        # corrupt ref) still raises so CI fails closed.
        files = package_unreleased_files(
            path, baseline, repo_root=repo_root, head=head
        )
        if is_lockfile_only(files):
            offenders.append(
                {
                    "component": component,
                    "path": path,
                    "version": version,
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
        `2` on missing/invalid config, manifest, or git failure.
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
