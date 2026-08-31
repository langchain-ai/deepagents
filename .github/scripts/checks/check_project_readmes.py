"""Detect edits to project landing-page READMEs on non-`docs` pull requests.

Why this exists:
    The root README and each package's `readme = "README.md"` (declared in
    every `libs/*/pyproject.toml`) are landing pages — they are what ships to
    PyPI and what a first-time reader sees. Incidental edits inside a feature
    or fix PR change that public face without anyone reviewing it as a
    documentation change. `.github/workflows/project_readme_check.yml` blocks
    such a PR until the author retitles it `docs:` or a maintainer applies the
    `readme: acknowledged` label.

What it does NOT do:
    This reads the PR title's Conventional Commit *type*, not the PR's
    contents. "Not a `docs` PR" is a proxy for intent, not a content check: a
    PR titled `docs(code): tidy README` that also rewrites 500 lines of Python
    passes the gate. Judging what a `docs:` PR actually contains stays a human
    job.

Exit-code contract:
    `main` returns a process exit code: 0 after printing the result, 2 when
    stdin is not a JSON array of strings. Nothing is written to stdout on the
    error path, so the workflow's `set -euo pipefail` aborts the step and the
    gate fails closed. Note this is deliberately the opposite convention from
    the sibling `ci_gate.py`, which always exits 0 and encodes its verdict in
    the payload: that script's caller needs to distinguish "gate says fail"
    from "gate crashed", whereas here both outcomes must block.
"""

from __future__ import annotations

import json
import re
import sys

# Duplicated as a JS literal in `.github/scripts/checks/readme-gate.js`, which
# is where the label is actually read; exported here only so the test suite can
# pin the two spellings together. Nothing in this module consumes it.
ACKNOWLEDGMENT_LABEL = "readme: acknowledged"

# The root README plus the `readme = "README.md"` of every distributed
# package. Hardcoded rather than derived so the detector needs no repo
# checkout to run, but a test walks `libs/*/pyproject.toml` and fails when a
# new package's README is missing here — without it a newly added package
# would silently fall outside the gate.
#
# `libs/README.md` is deliberately absent: it is a directory index, not a
# distributed package's landing page, and no pyproject declares it.
PROJECT_READMES = frozenset(
    {
        "README.md",
        "libs/acp/README.md",
        "libs/code/README.md",
        "libs/deepagents/README.md",
        "libs/evals/README.md",
        "libs/partners/daytona/README.md",
        "libs/partners/modal/README.md",
        "libs/partners/quickjs/README.md",
        "libs/partners/runloop/README.md",
        "libs/partners/sprites/README.md",
        "libs/partners/vercel/README.md",
        "libs/talon/README.md",
    }
)

# Anchored, lowercase-only, and intolerant of leading whitespace on purpose: a
# title that does not conform yields None, which reads as "not a docs PR" and
# fails closed. Relaxing this to be friendlier (case-insensitive, say) would
# silently widen the exemption to `Docs:` and `DOCS:`. The optional `!` is the
# Conventional Commits breaking-change marker, which does not change the type.
_TITLE_RE = re.compile(r"^([a-z]+)(?:\([^)]*\))?!?:\s")


def parse_pr_type(title: str) -> str | None:
    """Return the Conventional Commit type from a PR title.

    Args:
        title: The raw PR title.

    Returns:
        The lowercase type (`docs`, `fix`, ...), or None when the title does
        not start with a well-formed Conventional Commit prefix. Callers must
        treat None as "unknown, therefore not exempt".
    """
    match = _TITLE_RE.match(title)
    return match.group(1) if match else None


def find_readme_edits(title: str, changed: list[str]) -> dict[str, object]:
    """Return the protected READMEs a non-`docs` PR touches.

    Args:
        title: The raw PR title, used only for its Conventional Commit type.
        changed: Every path the PR touches. The caller is expected to include
            pre-rename paths, since renaming a protected README away is itself
            an edit to it.

    Returns:
        `{"pr_type": <parsed type or None>, "readmes": <sorted protected paths>}`.
        `readmes` is empty when the type is `docs` (the PR declares itself a
        documentation change) or when no protected path was touched. `pr_type`
        is echoed purely so CI logs show why a verdict was reached; the
        workflow branches only on `readmes`. Do not drop it as dead output.
    """
    pr_type = parse_pr_type(title)
    readmes = sorted(PROJECT_READMES.intersection(changed))
    return {"pr_type": pr_type, "readmes": [] if pr_type == "docs" else readmes}


def main(title: str) -> int:
    """Read changed paths as JSON on stdin and print the result as JSON.

    Args:
        title: The raw PR title.

    Returns:
        The process exit code: 0 after printing the detector result to stdout,
        or 2 when stdin is not a JSON array of strings. On the error path
        nothing reaches stdout, so the caller cannot mistake malformed input
        for a clean PR. See the module docstring's exit-code contract.
    """
    try:
        changed = json.load(sys.stdin)
        if not isinstance(changed, list) or any(
            not isinstance(path, str) for path in changed
        ):
            msg = "changed files must be a JSON array of strings"
            raise ValueError(msg)
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            f"::error::Project README check received invalid input: {exc}",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(find_readme_edits(title, changed), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_project_readmes.py <pr-title>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
