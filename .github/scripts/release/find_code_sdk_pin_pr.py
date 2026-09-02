"""Find the open PR maintained by the Code SDK pin bump workflow."""

from __future__ import annotations

import json
import sys

BODY_MARKER = "<!-- bump-code-sdk-pin-pr -->"
BRANCH_PREFIX = "chore/bump-code-sdk-pin-"
LEGACY_BODY_MARKER = "Opened automatically by `bump_code_sdk_pin.yml`"


def _is_automated_bump(pull_request: dict[str, object]) -> bool:
    """Return whether a PR belongs to the SDK pin bump workflow."""
    body = pull_request.get("body")
    branch = pull_request.get("headRefName")
    if not isinstance(body, str) or not isinstance(branch, str):
        return False
    if pull_request.get("isCrossRepository") is not False:
        return False
    return BODY_MARKER in body or (
        branch.startswith(BRANCH_PREFIX) and LEGACY_BODY_MARKER in body
    )


def find_existing_pr(
    pull_requests: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return the sole open automated bump PR, if one exists.

    Raises:
        ValueError: If multiple open automated bump PRs make the target ambiguous.
    """
    matches = [
        pull_request
        for pull_request in pull_requests
        if _is_automated_bump(pull_request)
    ]
    if len(matches) > 1:
        numbers = ", ".join(str(match.get("number", "unknown")) for match in matches)
        msg = f"Multiple open Code SDK pin bump PRs found: {numbers}"
        raise ValueError(msg)
    return matches[0] if matches else None


def main() -> int:
    """Read a PR list from stdin and print the selected PR as JSON."""
    pull_requests = json.load(sys.stdin)
    if not isinstance(pull_requests, list) or not all(
        isinstance(pull_request, dict) for pull_request in pull_requests
    ):
        msg = "Expected a JSON list of pull request objects"
        raise TypeError(msg)
    selected = find_existing_pr(pull_requests)
    json.dump(selected or {}, sys.stdout, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
