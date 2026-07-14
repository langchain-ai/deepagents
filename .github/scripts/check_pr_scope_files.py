"""Flag PRs whose package title scopes do not cover touched package dirs.

The PR labeler config already defines both sides of this relationship:
`scopeToLabel` maps conventional-commit scopes to package labels, and
`fileRules` maps package directories to the same labels. This helper reads that
config directly so the CI gate cannot drift from the labeler.

On successful analysis the script reports offenders on stdout and exits 0; the
workflow that calls it decides whether to fail, bypass, or comment. If the
labeler config cannot be read or validated, the script exits 2 so CI fails
closed.
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "scripts" / "pr-labeler-config.json"
DEFAULT_RELEASE_CONFIG = REPO_ROOT / "release-please-config.json"

_TITLE_RE = re.compile(r"^[a-z]+(?:\(([^)]*)\))?!?:\s")
_RELEASE_TITLE_RE = re.compile(r"^release\([^)]*\):\s")


@lru_cache(maxsize=None)
def _release_files(config_path: Path = DEFAULT_RELEASE_CONFIG) -> set[str]:
    """Return release-please managed artifact paths.

    Cached per `config_path` so a single run does not re-read and re-parse the
    config once per changed file. The config is immutable within a process, and
    raised `ValueError`s are not cached, so a malformed config keeps failing
    closed on every call.

    Args:
        config_path: Path to `release-please-config.json`.

    Returns:
        Set of repo-root-relative files release-please manages directly.

    Raises:
        ValueError: If the release-please config is missing required structure.
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        msg = f"could not read release-please config {config_path}: {e}"
        raise ValueError(msg) from e

    packages = config.get("packages")
    if not isinstance(packages, dict) or not packages:
        msg = "release-please config has no non-empty 'packages' map"
        raise ValueError(msg)

    files = {".release-please-manifest.json"}
    for root, package in packages.items():
        if not isinstance(root, str) or not isinstance(package, dict):
            msg = f"release-please package entry is malformed: {root!r}: {package!r}"
            raise ValueError(msg)
        changelog = package.get("changelog-path")
        if isinstance(changelog, str):
            files.add(str(Path(root) / changelog))
        extra_files = package.get("extra-files", [])
        if not isinstance(extra_files, list):
            msg = f"release-please package {root!r} has malformed 'extra-files'"
            raise ValueError(msg)
        for extra in extra_files:
            if not isinstance(extra, str):
                msg = f"release-please package {root!r} has non-string extra file"
                raise ValueError(msg)
            files.add(str(Path(root) / extra))
    return files


def is_release_file(
    file: str, *, release_config_path: Path = DEFAULT_RELEASE_CONFIG
) -> bool:
    """Return whether `file` is a file release-please may update in a release PR.

    Covers the version-of-record manifest, configured per-package changelog and
    extra files, and `uv.lock` files anywhere the uv workspace resolves.

    Args:
        file: Changed file path, repo-root-relative.
        release_config_path: Path to `release-please-config.json`.

    Returns:
        `True` when the path is one release-please may update in a release PR.

    Raises:
        ValueError: If the release-please config cannot be read or validated.
            Not reachable for `uv.lock` inputs, which return before any read.
    """
    path = Path(file)
    # release-please regenerates lockfiles wherever the uv workspace resolves —
    # not just package dirs but also `examples/*/uv.lock` (observed in real
    # partner release PRs). `uv.lock` is generated-only, so accept it at any
    # path rather than enumerating workspace-member dirs that drift over time.
    if path.name == "uv.lock":
        return True
    return file in _release_files(release_config_path)


def is_release_title(title: str) -> bool:
    """Return whether `title` is a release PR title.

    Args:
        title: PR title, e.g. `release(deepagents-code): 1.2.0`.

    Returns:
        `True` when the title uses the release PR conventional-commit shape.
    """
    return bool(_RELEASE_TITLE_RE.match(title))


def is_release_pr_change(
    title: str,
    changed: list[str],
    *,
    release_config_path: Path = DEFAULT_RELEASE_CONFIG,
) -> bool:
    """Return whether the PR looks like a release-please artifact update.

    The title is author-controlled (PR event payload) and the release component
    is not validated against real package names, so a non-release PR can adopt a
    `release(...)` title. Safety therefore rests entirely on the file allowlist:
    the bypass only applies when *every* changed file matches `is_release_file`,
    so a single source file re-arms the scope gate.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.
        release_config_path: Path to `release-please-config.json`.

    Returns:
        `True` only when the title is release-shaped and every changed file is a
        release-please generated/version artifact.

    Raises:
        ValueError: If the release-please config cannot be read or validated.
    """
    if not (changed and is_release_title(title)):
        return False
    # Read (and thereby validate) the release config up front so a malformed
    # `release-please-config.json` fails closed even when every changed file is
    # a `uv.lock` — those short-circuit inside `is_release_file` before any read,
    # which would otherwise let the gate stand down without validating the config.
    # `_release_files` is cached, so the per-file checks below reuse this result.
    _release_files(release_config_path)
    return all(
        is_release_file(file, release_config_path=release_config_path)
        for file in changed
    )


def parse_title_scopes(title: str) -> tuple[str, ...]:
    """Return conventional-commit scopes parsed from `title`.

    Args:
        title: PR title, e.g. `fix(cli,code): repair command`.

    Returns:
        Tuple of scope strings. Empty when the title is not conventional-commit
            shaped or has no scope.
    """
    match = _TITLE_RE.match(title)
    if not match or not match.group(1):
        return ()
    return tuple(scope.strip() for scope in match.group(1).split(",") if scope.strip())


def _package_rules(config: dict[str, Any]) -> list[dict[str, str]]:
    """Return package directory rules from the PR labeler config.

    Args:
        config: Parsed `.github/scripts/pr-labeler-config.json`.

    Returns:
        List of rules with `label` and normalized `prefix` keys.

    Raises:
        ValueError: If required config sections are missing or malformed.
    """
    rules = config.get("fileRules")
    if not isinstance(rules, list) or not rules:
        msg = "pr-labeler config has no non-empty 'fileRules' list"
        raise ValueError(msg)

    package_rules: list[dict[str, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            msg = f"pr-labeler fileRules entry is not an object: {rule!r}"
            raise ValueError(msg)
        label = rule.get("label")
        prefix = rule.get("prefix")
        # Rules without a 'prefix' (suffix/exact/pattern rules) are not package
        # directory rules and are legitimately skipped.
        if prefix is None:
            continue
        # A rule that *has* a 'prefix' but with a malformed type is config
        # corruption, not a non-package rule. Fail closed rather than silently
        # dropping it: a single dropped package rule would let a real
        # scope/file mismatch pass unnoticed (the gate's worst outcome).
        if not isinstance(label, str) or not isinstance(prefix, str):
            msg = f"pr-labeler fileRules entry has non-string label/prefix: {rule!r}"
            raise ValueError(msg)
        # A non-`libs/` prefix (e.g. `.github/workflows/`) is a legitimate
        # non-package rule.
        if not prefix.startswith("libs/"):
            continue
        package_rules.append({"label": label, "prefix": prefix.rstrip("/") + "/"})

    if not package_rules:
        msg = "pr-labeler config has no package directory fileRules"
        raise ValueError(msg)
    return sorted(package_rules, key=lambda r: r["prefix"])


def _scope_packages(config: dict[str, Any], package_labels: set[str]) -> set[str]:
    """Return scope names whose label points at a package label.

    Args:
        config: Parsed `.github/scripts/pr-labeler-config.json`.
        package_labels: Labels used by package directory rules.

    Returns:
        Set of scope names that represent package scopes.

    Raises:
        ValueError: If `scopeToLabel` is missing or malformed.
    """
    scope_to_label = config.get("scopeToLabel")
    if not isinstance(scope_to_label, dict) or not scope_to_label:
        msg = "pr-labeler config has no non-empty 'scopeToLabel' map"
        raise ValueError(msg)
    return {
        scope
        for scope, label in scope_to_label.items()
        if isinstance(scope, str) and isinstance(label, str) and label in package_labels
    }


def declared_packages(title: str, config: dict[str, Any]) -> set[str]:
    """Return package labels declared by the PR title scopes.

    Args:
        title: PR title.
        config: Parsed `.github/scripts/pr-labeler-config.json`.

    Returns:
        Set of package labels. Non-package scopes are ignored.

    Raises:
        ValueError: If required config sections are missing or malformed.
    """
    rules = _package_rules(config)
    package_labels = {rule["label"] for rule in rules}
    package_scopes = _scope_packages(config, package_labels)
    scope_to_label = config["scopeToLabel"]
    return {
        scope_to_label[scope]
        for scope in parse_title_scopes(title)
        if scope in package_scopes
    }


def changed_packages(
    changed: list[str], config: dict[str, Any]
) -> dict[str, list[str]]:
    """Return package labels and dirs touched by changed files.

    Args:
        changed: Changed file paths, repo-root-relative.
        config: Parsed `.github/scripts/pr-labeler-config.json`.

    Returns:
        Map of package label to touched package directories.

    Raises:
        ValueError: If required config sections are missing or malformed.
    """
    packages: dict[str, set[str]] = {}
    for rule in _package_rules(config):
        prefix = rule["prefix"]
        package_dir = prefix.rstrip("/")
        for file in changed:
            if file == package_dir or file.startswith(prefix):
                packages.setdefault(rule["label"], set()).add(prefix)
                break
    return {label: sorted(dirs) for label, dirs in sorted(packages.items())}


def find_offenders(
    title: str,
    changed: list[str],
    config: dict[str, Any],
    *,
    release_config_path: Path = DEFAULT_RELEASE_CONFIG,
) -> list[dict[str, object]]:
    """Return touched package dirs not covered by package scopes in `title`.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.
        config: Parsed `.github/scripts/pr-labeler-config.json`.
        release_config_path: Path to `release-please-config.json`.

    Returns:
        Sorted list of offender objects with `package` and `dirs` keys. Empty
            when the title declares no package scopes, when no package dirs are
            touched (for release PRs, after release-managed artifacts such as
            `uv.lock` files are filtered out), or when every touched package is
            covered by a declared scope.

    Raises:
        ValueError: If required config sections are missing or malformed.
    """
    # Computed before the release filtering so a malformed labeler config still
    # fails closed (raises) on release PRs rather than passing unchecked.
    declared = declared_packages(title, config)
    if is_release_title(title):
        # Validate the release config even when no package offenders remain. Then
        # ignore release-managed artifacts for package-dir matching: release PRs
        # often regenerate dependent `uv.lock` files in packages unrelated to the
        # release component, but ordinary source edits must still be checked.
        _release_files(release_config_path)
        changed = [
            file
            for file in changed
            if not is_release_file(file, release_config_path=release_config_path)
        ]

    if not changed or not declared:
        return []

    touched = changed_packages(changed, config)
    return [
        {"package": package, "dirs": dirs}
        for package, dirs in touched.items()
        if package not in declared
    ]


def main(
    title: str,
    changed: list[str],
    config_path: Path = DEFAULT_CONFIG,
    *,
    release_config_path: Path = DEFAULT_RELEASE_CONFIG,
) -> int:
    """Print offending packages as a JSON array to stdout.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.
        config_path: Path to the PR labeler config.
        release_config_path: Path to `release-please-config.json`.

    Returns:
        `0` after successful analysis. Offenders are reported on stdout and the
            workflow makes the blocking decision.
        `2` when the config cannot be read or validated, so CI fails closed.
    """
    try:
        # Wrap the labeler-config read so its errors name the labeler file. The
        # outer handler also catches ValueErrors raised while reading the
        # release config — by find_offenders (via `_release_files`) and by the
        # is_release_pr_change call below — whose messages name
        # release-please-config.json themselves, so the generic prefix below
        # does not mis-attribute one config's failure to the other.
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            msg = f"could not read PR labeler config {config_path}: {e}"
            raise ValueError(msg) from e
        offenders = find_offenders(
            title, changed, config, release_config_path=release_config_path
        )
        bypassed = is_release_pr_change(
            title, changed, release_config_path=release_config_path
        )
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(
            f"::error::PR scope/file check failed to read or validate config: {e}",
            file=sys.stderr,
        )
        return 2

    # Surface the release bypass so "gate stood down" is distinguishable from
    # "genuinely clean" in the Checks UI, mirroring the workflow's
    # detector-absent ::warning::. To stderr so it never corrupts the JSON
    # offenders the workflow captures from stdout. ::notice:: (not ::warning::)
    # because this is a designed, expected bypass.
    if bypassed:
        print(
            "::notice::Release-shaped title; scope/file gate bypassed because "
            "every changed file matched the release-please artifact allowlist.",
            file=sys.stderr,
        )

    if offenders:
        summary = ", ".join(
            f"{offender['package']} ({', '.join(offender['dirs'])})"
            for offender in offenders
        )
        print(
            f"PR title scope does not cover touched package dirs: {summary}",
            file=sys.stderr,
        )
    print(json.dumps(offenders))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: check_pr_scope_files.py <pr-title>  (changed files on stdin)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    pr_title = sys.argv[1]
    changed_files = [line.strip() for line in sys.stdin if line.strip()]
    raise SystemExit(main(pr_title, changed_files))
