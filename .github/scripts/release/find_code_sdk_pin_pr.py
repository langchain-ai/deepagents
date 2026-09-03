"""Find the open PR maintained by the Code SDK pin bump workflow."""

from __future__ import annotations

import json
import sys

BODY_MARKER = "<!-- bump-code-sdk-pin-pr -->"
BRANCH_PREFIX = "chore/bump-code-sdk-pin-"
# Bodies written before `BODY_MARKER` was introduced. The workflow no longer
# writes this text, so it identifies only a PR that an older run opened.
LEGACY_BODY_MARKER = "Opened automatically by `bump_code_sdk_pin.yml`"


def _is_automated_bump(pull_request: dict[str, object]) -> bool:
    """Return whether a PR belongs to the SDK pin bump workflow.

    A PR must satisfy all three conditions:

    - Its head branch starts with `BRANCH_PREFIX`. Only this workflow creates
      such branches.
    - Its body holds `BODY_MARKER`, or the pre-marker `LEGACY_BODY_MARKER`.
    - It comes from this repository. An absent `isCrossRepository` field also
      fails this condition, because the caller must confirm the origin.

    The branch condition is necessary because `BODY_MARKER` is an HTML comment.
    It is invisible in a rendered body, so it moves silently when a person
    copies the body into their own PR. The caller pushes commits to the branch
    of the PR that this function selects, and it replaces the title and the
    body of that PR. Body text alone must not give a PR that authority.
    """
    body = pull_request.get("body")
    branch = pull_request.get("headRefName")
    if not isinstance(body, str) or not isinstance(branch, str):
        return False
    if pull_request.get("isCrossRepository") is not False:
        return False
    if not branch.startswith(BRANCH_PREFIX):
        return False
    return BODY_MARKER in body or LEGACY_BODY_MARKER in body


def find_existing_pr(
    pull_requests: list[dict[str, object]],
) -> dict[str, object] | None:
    """Return the sole automated bump PR in `pull_requests`, if one exists.

    Args:
        pull_requests: Open pull requests of this repository. This function
            does not read `state`, so the caller must exclude closed PRs.

    Raises:
        ValueError: If multiple automated bump PRs make the target ambiguous.
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
    """Read a PR list from stdin and print the selected PR as compact JSON.

    Prints `{}` when no automated bump PR is open. The workflow reads that
    empty object as "create a new PR".

    Returns:
        The process exit code. Always 0, because every failure raises.

    Raises:
        json.JSONDecodeError: If stdin does not hold valid JSON.
        TypeError: If stdin does not hold a JSON list of objects.
        ValueError: If multiple automated bump PRs are open.
    """
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
