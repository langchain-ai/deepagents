"""Check that pyproject.toml and _version.py versions stay in sync.

Prevents releases with mismatched version numbers across the SDK and CLI
packages. Used by the CI workflow in .github/workflows/check_versions.yml
and as a pre-commit hook.
"""

import re
import sys
import tomllib
from pathlib import Path

PACKAGES = [
    ("libs/deepagents/pyproject.toml", "libs/deepagents/deepagents/_version.py"),
    ("libs/cli/pyproject.toml", "libs/cli/deepagents_cli/_version.py"),
]

_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


def _get_pyproject_version(path: Path) -> str:
    """Extract version from pyproject.toml.

    Args:
        path: Path to pyproject.toml.

    Returns:
        Version string.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return data["project"]["version"]
    except KeyError:
        msg = f"Could not find project.version in {path}"
        raise ValueError(msg) from None


def _get_version_py(path: Path) -> str:
    """Extract __version__ from _version.py.

    Args:
        path: Path to _version.py.

    Returns:
        Version string.

    Raises:
        ValueError: If __version__ is not found.
    """
    text = path.read_text()
    match = _VERSION_RE.search(text)
    if not match:
        msg = f"Could not find __version__ in {path}"
        raise ValueError(msg)
    return match.group(1)


def main() -> int:
    """Check version equality across packages.

    Returns:
        0 if all versions match, 1 if there are mismatches.
    """
    root = Path(__file__).resolve().parents[2]
    errors: list[str] = []

    for pyproject_rel, version_py_rel in PACKAGES:
        pyproject_path = root / pyproject_rel
        version_py_path = root / version_py_rel

        missing = [p for p in (pyproject_path, version_py_path) if not p.exists()]
        if missing:
            errors.append(
                f"  {pyproject_rel.split('/')[1]}: file(s) not found: "
                + ", ".join(str(m) for m in missing)
            )
            continue

        pyproject_ver = _get_pyproject_version(pyproject_path)
        version_py_ver = _get_version_py(version_py_path)

        if pyproject_ver != version_py_ver:
            pkg = pyproject_path.parent.name
            errors.append(
                f"  {pkg}: pyproject.toml={pyproject_ver}, "
                f"_version.py={version_py_ver}"
            )
        else:
            print(f"{pyproject_path.parent.name} versions match: {pyproject_ver}")

    if errors:
        print("Version mismatch detected:")
        print("\n".join(errors))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
