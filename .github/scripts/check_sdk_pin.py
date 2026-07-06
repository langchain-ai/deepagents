"""Check the deepagents-code SDK pin against the workspace SDK version.

`libs/code/pyproject.toml` pins an exact `deepagents==X.Y.Z`. This compares
that pin to the version declared in `libs/deepagents/pyproject.toml` and
reports stale pins so they surface locally instead of only at release time.

Advisory by design: the hard gate is the release workflow's pin-verification
step (the "Verify package pins SDK at or ahead of workspace version" step in
`release.yml`, which fails the publish job when the pin is stale). The
`check_sdk_pin.yml` workflow is a complementary advisory check that only
comments on release PRs — it does not block merge. During normal development
the editable workspace source means you always run against the local SDK
regardless of the pin, so a stale pin mid-feature is expected until you bump
the pin.

Exit codes when run as a script: 0 = pin in sync or ahead, 1 = stale pin
(advisory — callers may treat as non-fatal), 2 = could not determine
(malformed or missing pin/version). Callers must not treat exit 2 as a pass.
"""

import re
import tomllib
from pathlib import Path

# Capture the full version token after `==`, stopping at any PEP 508 delimiter
# (quote, comma, whitespace, marker `;`, or a range operator). The workflows
# import this module for extraction and comparison so local checks, PR comments,
# and release gating stay in lock-step.
_VERSION_RE = re.compile(r"==\s*([^\",\s;<>=]+)")
_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _sdk_version(repo_root: Path) -> str:
    """Return the version declared in the deepagents SDK pyproject.toml."""
    path = repo_root / "libs" / "deepagents" / "pyproject.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return data["project"]["version"]
    except KeyError:
        msg = f"Could not find project.version in {path}"
        raise ValueError(msg) from None


def _code_pin(repo_root: Path) -> str:
    """Return the pinned SDK version from the deepagents-code dependencies."""
    path = repo_root / "libs" / "code" / "pyproject.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    for dep in data.get("project", {}).get("dependencies", []):
        name_match = _NAME_RE.match(dep)
        if name_match and name_match.group(0).lower() == "deepagents":
            version_match = _VERSION_RE.search(dep)
            if version_match:
                return version_match.group(1)
    msg = f"No `deepagents==<version>` pin found in {path}"
    raise ValueError(msg)


def _parse_version(version: str) -> tuple[tuple[int, ...], tuple[int, int]]:
    """Return a comparable `(release, qualifier)` key for a version string.

    Handles a release segment (`X`, `X.Y`, `X.Y.Z`, ...) plus at most one
    trailing qualifier: a prerelease (`aN`/`bN`/`rcN`), a post-release
    (`.postN`), or a dev release (`.devN`). Qualifiers sort dev < pre < final
    < post, matching PEP 440 precedence.

    Combined qualifiers (e.g. `1.0a1.dev1`), epochs (`1!1.0`), and local
    versions (`1.0+local`) are intentionally rejected with `ValueError`. The
    repo never pins them, and a single comparable key cannot order them
    faithfully, so rejecting fails closed (callers surface it as exit 2 —
    "could not determine" — rather than a silently wrong comparison).
    """
    match = re.fullmatch(
        r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+)|\.post(\d+)|\.dev(\d+))?",
        version,
    )
    if not match:
        msg = f"Unsupported version format: {version}"
        raise ValueError(msg)
    release = tuple(int(part) for part in match.group(1).split("."))
    prerelease, prenum = match.group(2), match.group(3)
    postrelease = match.group(4)
    devrelease = match.group(5)
    # The regex alternation guarantees at most one qualifier is present.
    if devrelease is not None:
        pre = (-1, int(devrelease))
    elif prerelease is not None:
        pre = ({"a": 0, "b": 1, "rc": 2}[prerelease], int(prenum))
    elif postrelease is not None:
        pre = (4, int(postrelease))
    else:
        pre = (3, 0)
    return release, pre


def compare_versions(left: str, right: str) -> int:
    """Compare two version strings by PEP 440 precedence.

    Returns -1 if `left` is older than `right`, 0 if they are equal, and 1 if
    `left` is newer. Call sites rely on `compare_versions(pin, sdk) < 0`
    meaning the pin is stale.
    """
    left_release, left_pre = _parse_version(left)
    right_release, right_pre = _parse_version(right)
    length = max(len(left_release), len(right_release))
    left_key = left_release + (0,) * (length - len(left_release)), left_pre
    right_key = right_release + (0,) * (length - len(right_release)), right_pre
    if left_key < right_key:
        return -1
    if left_key > right_key:
        return 1
    return 0


def is_prerelease(version: str) -> bool:
    """Return whether a version is a prerelease or dev release.

    Derives the answer from the qualifier key `_parse_version` already
    computes, so prerelease classification stays in lock-step with the
    parser's grammar rather than duplicating it. The qualifier's first
    element is negative for dev releases and 0/1/2 for `a`/`b`/`rc`
    prereleases (`< 3`); final (`3`) and post (`4`) releases are not
    prereleases.

    Args:
        version: A PEP 440 version string.

    Returns:
        `True` for prerelease (`aN`/`bN`/`rcN`) or dev (`.devN`) versions.

    Raises:
        ValueError: If `version` is not a supported format (validated via
            `_parse_version`, which fails closed on epochs, local versions,
            and combined qualifiers).
    """
    _, qualifier = _parse_version(version)
    return qualifier[0] < 3


def main(repo_root: Path | None = None) -> int:
    """Compare the pin to the SDK version; return 1 when the pin is stale."""
    root = repo_root or Path(__file__).resolve().parents[2]
    sdk = _sdk_version(root)
    pin = _code_pin(root)
    comparison = compare_versions(pin, sdk)
    if comparison == 0:
        print(f"SDK pin is in sync: deepagents=={pin}")
        return 0
    if comparison > 0:
        print(
            f"SDK pin is ahead: libs/code pins deepagents=={pin} while the "
            f"workspace SDK is {sdk}."
        )
        return 0
    print(
        f"SDK pin is stale: libs/code pins deepagents=={pin} but the workspace "
        f"SDK is {sdk}.\n"
        f"If your change depends on the current SDK, set the pin in "
        f"libs/code/pyproject.toml to `deepagents=={sdk}`."
    )
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as e:
        # Exit 2 (distinct from the stale-pin exit 1) so a "couldn't determine"
        # failure is not laundered into an in-sync pass by callers that treat a
        # stale pin as advisory. See the `check` target in libs/code/Makefile.
        print(f"Could not determine SDK pin status: {e}")
        raise SystemExit(2) from None
