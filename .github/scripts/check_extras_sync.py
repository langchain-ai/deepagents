"""Check that optional extras stay in sync with required dependencies (openai).

When a package appears in both [project.dependencies] and
[project.optional-dependencies], we ensure their version constraints match.
This prevents silent version drift (e.g. bumping a required dep but
forgetting the corresponding extra).
"""

import sys
import tomllib
from pathlib import Path
from re import compile as re_compile

# Matches the package name at the start of a PEP 508 dependency string.
# Handles both hyphenated and underscored names (PEP 503 normalizes these).
_NAME_RE = re_compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize(name: str) -> str:
    """PEP 503 normalize a package name for comparison.

    Returns:
        Lowercased, underscore-normalized package name.
    """
    return name.lower().replace("-", "_").replace(".", "_")


def _parse_dep(dep: str) -> tuple[str, str]:
    """Return (normalized_name, version_spec) from a PEP 508 string.

    Returns:
        Tuple of normalized package name and version specifier.

    Raises:
        ValueError: If the dependency string cannot be parsed.
    """
    match = _NAME_RE.match(dep)
    if not match:
        msg = f"Cannot parse dependency: {dep}"
        raise ValueError(msg)
    name = match.group(1)
    version_spec = dep[match.end() :].strip()
    return _normalize(name), version_spec


def main(pyproject_path: Path) -> int:
    """Check extras sync and return exit code (0 = pass, 1 = mismatch).

    Returns:
        0 if all extras match, 1 if there are mismatches.
    """
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    required: dict[str, str] = {}
    for dep in data.get("project", {}).get("dependencies", []):
        name, spec = _parse_dep(dep)
        required[name] = spec

    mismatches: list[str] = []
    optional = data.get("project", {}).get("optional-dependencies", {})
    for group, deps in optional.items():
        for dep in deps:
            name, spec = _parse_dep(dep)
            if name in required and spec != required[name]:
                mismatches.append(
                    f"  [{group}] {name}: extra has '{spec}' "
                    f"but required dep has '{required[name]}'"
                )

    if mismatches:
        print("Extra / required dependency version mismatch:")
        print("\n".join(mismatches))
        print(
            "\nUpdate the optional extras in [project.optional-dependencies] "
            "to match [project.dependencies]."
        )
        return 1

    print("All extras are in sync with required dependencies.")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    raise SystemExit(main(path))
