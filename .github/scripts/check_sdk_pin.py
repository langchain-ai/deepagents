"""Check the deepagents-code SDK pin against the workspace SDK version.

`libs/code/pyproject.toml` pins an exact `deepagents==X.Y.Z`. This compares
that pin to the version declared in `libs/deepagents/pyproject.toml` and
reports drift so it surfaces locally instead of only at release time.

Advisory by design: the release workflow is the hard gate (see
`.github/workflows/check_sdk_pin.yml` and `release.yml`). During normal
development the editable workspace source means you always run against the
local SDK regardless of the pin, so a mismatch mid-feature is expected until
you bump the pin.
"""

import re
import tomllib
from pathlib import Path

_VERSION_RE = re.compile(r"==\s*([0-9]+\.[0-9]+\.[0-9]+)")
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
    msg = f"No `deepagents==X.Y.Z` pin found in {path}"
    raise ValueError(msg)


def main(repo_root: Path | None = None) -> int:
    """Compare the pin to the SDK version; return 0 on match, 1 on drift."""
    root = repo_root or Path(__file__).resolve().parents[2]
    sdk = _sdk_version(root)
    pin = _code_pin(root)
    if sdk == pin:
        print(f"SDK pin is in sync: deepagents=={pin}")
        return 0
    print(
        f"SDK pin drift: libs/code pins deepagents=={pin} but the workspace "
        f"SDK is {sdk}.\n"
        f"If your change depends on the current SDK, set the pin in "
        f"libs/code/pyproject.toml to `deepagents=={sdk}`."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
