"""Check that the branch-name rules stay in sync across their three copies.

The branch naming convention `<github-username>/<scope>/<short-description>` is
enforced in three places that cannot share code:

- `.githooks/pre-push` — the local pre-push hook (bash)
- `.github/workflows/branch_name_check.yml` — the advisory CI check (bash in YAML)
- `.github/workflows/pr_lint.yml` — the PR *title* scope list, which the branch
  scope list mirrors (plus `docs`, which AGENTS.md lists as a branch scope)

Drift has an asymmetric cost: a scope added to `pr_lint.yml` alone means the
local hook hard-blocks a branch CI considers valid, and reports the stale scope
set as authoritative. The CI check is advisory, so it produces no red signal.
This check makes that drift a commit-time failure instead.
"""

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_HOOK = Path(".githooks/pre-push")
_BRANCH_CHECK = Path(".github/workflows/branch_name_check.yml")
_PR_LINT = Path(".github/workflows/pr_lint.yml")

# Scopes valid in a branch name but not in a PR title: AGENTS.md lists `docs` as
# a branch scope, while `pr_lint.yml` carries it as a Conventional Commits
# *type*. Keep this the only permitted delta between the two lists.
_BRANCH_ONLY_SCOPES = frozenset({"docs"})

# The three shell variables the two branch-name checks must agree on verbatim.
_SHARED_VARS = ("ALLOWED_RE", "ALLOWED_PREFIX_RE", "SCOPES_RE")


def _read(path: Path) -> str:
    """Return the text of a repo-relative path, or exit with a clear error."""
    full = _REPO_ROOT / path
    try:
        return full.read_text(encoding="utf-8")
    except OSError as exc:
        sys.exit(f"error: could not read {path}: {exc}")


def _shell_assignment(text: str, path: Path, var: str) -> str:
    """Return the single-quoted value assigned to `var` in a shell snippet.

    Requires exactly one assignment so a second, shadowing copy is a failure
    rather than a silently ignored one.
    """
    matches = re.findall(rf"^\s*{var}='([^']*)'\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        sys.exit(
            f"error: expected exactly one single-quoted `{var}=...` assignment in "
            f"{path}, found {len(matches)}. If the assignment was reformatted, "
            f"update {Path(__file__).name} to match."
        )
    return matches[0]


def _scopes_from_re(pattern: str) -> list[str]:
    """Return the alternatives from a `(a|b|c)` scope group."""
    if not (pattern.startswith("(") and pattern.endswith(")")):
        sys.exit(f"error: SCOPES_RE is not a parenthesized group: {pattern!r}")
    return pattern[1:-1].split("|")


def _pr_lint_scopes() -> list[str]:
    """Return the `scopes:` block entries from `pr_lint.yml`.

    Parsed with a regex rather than a YAML loader to keep this check
    dependency-free (it runs under `language: system`).
    """
    text = _read(_PR_LINT)
    match = re.search(
        r"^(?P<indent>[ ]*)scopes: \|\n(?P<body>(?:.*\n)*?)(?=\1\S)",
        text,
        re.MULTILINE,
    )
    if match is None:
        sys.exit(
            f"error: could not find a `scopes: |` block in {_PR_LINT}. If the "
            f"block was restructured, update {Path(__file__).name} to match."
        )
    scopes = [line.strip() for line in match.group("body").splitlines() if line.strip()]
    if not scopes:
        sys.exit(f"error: the `scopes: |` block in {_PR_LINT} is empty.")
    return scopes


def main() -> int:
    """Compare the three rule sets and report every mismatch found."""
    hook_text = _read(_HOOK)
    ci_text = _read(_BRANCH_CHECK)
    errors: list[str] = []

    # 1. The two branch-name checks must share all three patterns byte for byte.
    for var in _SHARED_VARS:
        hook_value = _shell_assignment(hook_text, _HOOK, var)
        ci_value = _shell_assignment(ci_text, _BRANCH_CHECK, var)
        if hook_value != ci_value:
            errors.append(
                f"{var} differs between the local hook and the CI check:\n"
                f"  {_HOOK}: {hook_value}\n"
                f"  {_BRANCH_CHECK}: {ci_value}"
            )

    # 2. The branch scope list must be the PR-title scope list plus `docs`.
    branch_scopes = _scopes_from_re(_shell_assignment(hook_text, _HOOK, "SCOPES_RE"))
    expected = set(_pr_lint_scopes()) | _BRANCH_ONLY_SCOPES
    actual = set(branch_scopes)
    if missing := sorted(expected - actual):
        errors.append(
            f"scopes in {_PR_LINT} (plus {sorted(_BRANCH_ONLY_SCOPES)}) but "
            f"missing from SCOPES_RE: {', '.join(missing)}"
        )
    if extra := sorted(actual - expected):
        errors.append(
            f"scopes in SCOPES_RE but not in {_PR_LINT}: {', '.join(extra)}. Add "
            f"them to pr_lint.yml, or to _BRANCH_ONLY_SCOPES if they are "
            f"deliberately branch-only."
        )

    if errors:
        print("Branch-name rules are out of sync:\n", file=sys.stderr)
        for error in errors:
            print(f"  - {error}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
