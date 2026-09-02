"""Tests for selecting the PR maintained by the Code SDK pin bump workflow."""

import pytest
from find_code_sdk_pin_pr import BODY_MARKER, find_existing_pr


def _pull_request(
    number: int,
    *,
    body: str,
    branch: str = "chore/bump-code-sdk-pin-0.7.13",
    is_cross_repository: bool = False,
) -> dict[str, object]:
    return {
        "body": body,
        "headRefName": branch,
        "isCrossRepository": is_cross_repository,
        "number": number,
    }


def test_finds_marker_based_bump_pr() -> None:
    bump = _pull_request(6036, body=f"{BODY_MARKER}\nBumps the SDK pin.")
    unrelated = _pull_request(6000, body="A normal dependency update.")

    assert find_existing_pr([unrelated, bump]) == bump


def test_adopts_legacy_versioned_bump_pr() -> None:
    legacy = _pull_request(
        6036,
        body="Opened automatically by `bump_code_sdk_pin.yml` after an SDK release.",
    )

    assert find_existing_pr([legacy]) == legacy


@pytest.mark.parametrize(
    "pull_request",
    [
        _pull_request(
            6036,
            body="Opened automatically by `bump_code_sdk_pin.yml`.",
            branch="maintainer/manual-pin-bump",
        ),
        _pull_request(6036, body=BODY_MARKER, is_cross_repository=True),
    ],
)
def test_ignores_prs_not_owned_by_the_workflow(
    pull_request: dict[str, object],
) -> None:
    assert find_existing_pr([pull_request]) is None


def test_rejects_multiple_automated_bump_prs() -> None:
    first = _pull_request(6022, body=BODY_MARKER)
    second = _pull_request(6036, body=BODY_MARKER)

    with pytest.raises(ValueError, match="6022, 6036"):
        find_existing_pr([first, second])
