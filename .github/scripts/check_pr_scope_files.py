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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "scripts" / "pr-labeler-config.json"

_TITLE_RE = re.compile(r"^[a-z]+(?:\(([^)]*)\))?!?:\s")
_RELEASE_TITLE_RE = re.compile(r"^release\([^)]*\):\s")


def is_release_file(file: str) -> bool:
    """Return whether `file` is a file release-please may update in a release PR.

    Covers the version-of-record manifest, per-package `CHANGELOG.md` /
    `pyproject.toml` / `_version.py`, and `uv.lock` files anywhere the uv
    workspace resolves.

    Keep in sync with `extra-files`/`changelog-path` in
    `release-please-config.json`. Drift fails closed — an unrecognized artifact
    means the bypass does not fire and the normal scope gate runs — so this is a
    maintenance note, not a safety hole.

    Args:
        file: Changed file path, repo-root-relative.

    Returns:
        `True` when the path is one release-please may update in a release PR.
    """
    path = Path(file)
    # The version-of-record manifest release-please rewrites on every release.
    if file == ".release-please-manifest.json":
        return True
    # release-please regenerates lockfiles wherever the uv workspace resolves —
    # not just package dirs but also `examples/*/uv.lock` (observed in real
    # partner release PRs). `uv.lock` is generated-only, so accept it at any
    # path rather than enumerating workspace-member dirs that drift over time.
    if path.name == "uv.lock":
        return True
    parts = path.parts
    if len(parts) < 3 or parts[0] != "libs":
        return False
    # Package-root files: depth 3 for top-level packages (`libs/<pkg>/...`) and
    # depth 4 for partners, which nest one level deeper under `libs/partners/`.
    if path.name in {"CHANGELOG.md", "pyproject.toml"}:
        return len(parts) == 3 or (len(parts) == 4 and parts[1] == "partners")
    # `_version.py` sits inside the import package, one level below the
    # package-root files above: depth 4 top-level, depth 5 for partners.
    if path.name == "_version.py":
        return len(parts) == 4 or (len(parts) == 5 and parts[1] == "partners")
    return False


def is_release_title(title: str) -> bool:
    """Return whether `title` is a release PR title.

    Args:
        title: PR title, e.g. `release(deepagents-code): 1.2.0`.

    Returns:
        `True` when the title uses the release PR conventional-commit shape.
    """
    return bool(_RELEASE_TITLE_RE.match(title))


def is_release_pr_change(title: str, changed: list[str]) -> bool:
    """Return whether the PR looks like a release-please artifact update.

    The title is author-controlled (PR event payload) and the release component
    is not validated against real package names, so a non-release PR can adopt a
    `release(...)` title. Safety therefore rests entirely on the file allowlist:
    the bypass only applies when *every* changed file matches `is_release_file`,
    so a single source file re-arms the scope gate.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.

    Returns:
        `True` only when the title is release-shaped and every changed file is a
        release-please generated/version artifact.
    """
    return (
        bool(changed)
        and is_release_title(title)
        and all(is_release_file(file) for file in changed)
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
    title: str, changed: list[str], config: dict[str, Any]
) -> list[dict[str, object]]:
    """Return touched package dirs not covered by package scopes in `title`.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.
        config: Parsed `.github/scripts/pr-labeler-config.json`.

    Returns:
        Sorted list of offender objects with `package` and `dirs` keys. Empty
            when the title declares no package scopes, when no package dirs are
            touched, or when every touched package is covered by a declared scope.

    Raises:
        ValueError: If required config sections are missing or malformed.
    """
    if is_release_pr_change(title, changed):
        return []

    declared = declared_packages(title, config)
    if not declared:
        return []

    touched = changed_packages(changed, config)
    return [
        {"package": package, "dirs": dirs}
        for package, dirs in touched.items()
        if package not in declared
    ]


def main(title: str, changed: list[str], config_path: Path = DEFAULT_CONFIG) -> int:
    """Print offending packages as a JSON array to stdout.

    Args:
        title: PR title.
        changed: Changed file paths, repo-root-relative.
        config_path: Path to the PR labeler config.

    Returns:
        `0` after successful analysis. Offenders are reported on stdout and the
            workflow makes the blocking decision.
        `2` when the config cannot be read or validated, so CI fails closed.
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        offenders = find_offenders(title, changed, config)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(
            f"::error::Could not read PR labeler config {config_path}: {e}",
            file=sys.stderr,
        )
        return 2

    # Surface the release bypass so "gate stood down" is distinguishable from
    # "genuinely clean" in the Checks UI, mirroring the workflow's
    # detector-absent ::warning::. To stderr so it never corrupts the JSON
    # offenders the workflow captures from stdout. ::notice:: (not ::warning::)
    # because this is a designed, expected bypass.
    if is_release_pr_change(title, changed):
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
