"""Detect project README edits on pull requests that are not documentation-only."""

from __future__ import annotations

import json
import re
import sys

ACKNOWLEDGMENT_LABEL = "readme: acknowledged"
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
        "libs/partners/vercel/README.md",
        "libs/talon/README.md",
    }
)
_TITLE_RE = re.compile(r"^([a-z]+)(?:\([^)]*\))?!?:\s")


def parse_pr_type(title: str) -> str | None:
    """Return the Conventional Commit type from a PR title."""
    match = _TITLE_RE.match(title)
    return match.group(1) if match else None


def find_readme_edits(title: str, changed: list[str]) -> dict[str, object]:
    """Return protected README edits when the PR type is not `docs`."""
    pr_type = parse_pr_type(title)
    readmes = sorted(PROJECT_READMES.intersection(changed))
    return {"pr_type": pr_type, "readmes": [] if pr_type == "docs" else readmes}


def main(title: str) -> int:
    """Read changed paths as JSON and print the detector result as JSON."""
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
