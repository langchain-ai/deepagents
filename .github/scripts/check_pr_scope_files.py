"""Flag PRs whose package title scopes do not cover touched package dirs.

The PR labeler config already defines both sides of this relationship:
`scopeToLabel` maps conventional-commit scopes to package labels, and
`fileRules` maps package directories to the same labels. This helper reads that
config directly so the CI gate cannot drift from the labeler.

The script only reports offenders and exits 0 after successful analysis. The
workflow that calls it decides whether to fail, bypass, or comment.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / ".github" / "scripts" / "pr-labeler-config.json"

_TITLE_RE = re.compile(r"^[a-z]+(?:\(([^)]*)\))?!?:\s")


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
            continue
        label = rule.get("label")
        prefix = rule.get("prefix")
        if not isinstance(label, str) or not isinstance(prefix, str):
            continue
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
