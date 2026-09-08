"""Compare two versions with PEP 440 ordering."""

from __future__ import annotations

import sys

from packaging.version import Version

BEHIND = "behind"
CURRENT = "current"
AHEAD = "ahead"


def compare(left: str, right: str) -> str:
    """Return where `left` sits relative to `right`.

    Args:
        left: The version under test, such as the pin on a branch.
        right: The version to compare against, such as the target version.

    Returns:
        `behind` if `left` is older than `right`, `ahead` if it is newer, and
        `current` if the two are equal.

    Raises:
        InvalidVersion: If either argument is not a PEP 440 version.
    """
    parsed_left = Version(left)
    parsed_right = Version(right)
    if parsed_left < parsed_right:
        return BEHIND
    if parsed_left > parsed_right:
        return AHEAD
    return CURRENT


def main(argv: list[str]) -> int:
    """Print the relation between the two versions given on the command line.

    Args:
        argv: The arguments after the program name: exactly two versions.

    Returns:
        The process exit code. Always 0, because every failure raises.

    Raises:
        SystemExit: If the caller does not pass exactly two versions.
        InvalidVersion: If either argument is not a PEP 440 version.
    """
    if len(argv) != 2:
        msg = f"Usage: compare_versions.py <left> <right> (got {len(argv)} args)"
        raise SystemExit(msg)
    sys.stdout.write(compare(argv[0], argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
