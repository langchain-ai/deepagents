"""Resolve the Python versions a package's release checks should run on."""

from __future__ import annotations

import json
import sys
import tomllib
from typing import Final


SUPPORTED_PYTHON_VERSIONS: Final[tuple[str, ...]] = (
    "3.11",
    "3.12",
    "3.13",
    "3.14",
)
_OPERATORS: Final[tuple[str, ...]] = (">=", "<=", "!=", "==", ">", "<")


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted release version into comparable integers."""
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as error:
        msg = f"Cannot parse Python version {version!r}: {error}"
        raise ValueError(msg) from None


def _pad(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pad two version tuples to a common length."""
    width = max(len(left), len(right))
    return (
        left + (0,) * (width - len(left)),
        right + (0,) * (width - len(right)),
    )


def _satisfies(candidate: tuple[int, ...], clause: str) -> bool:
    """Check one clause from a `requires-python` specifier."""
    for operator in _OPERATORS:
        if not clause.startswith(operator):
            continue
        bound_text = clause[len(operator) :].strip()
        if bound_text.endswith(".*"):
            msg = f"Wildcard specifiers are not supported: {clause!r}"
            raise ValueError(msg)
        left, right = _pad(candidate, _parse_version(bound_text))
        if operator == ">=":
            return left >= right
        if operator == "<=":
            return left <= right
        if operator == "!=":
            return left != right
        if operator == "==":
            return left == right
        if operator == ">":
            return left > right
        return left < right
    msg = f"Unsupported requires-python clause: {clause!r}"
    raise ValueError(msg)


def resolve_python_versions(requires_python: str) -> list[str]:
    """Return supported interpreters allowed by `requires-python`."""
    clauses = [
        clause.strip() for clause in requires_python.split(",") if clause.strip()
    ]
    if not clauses:
        msg = "requires-python is empty"
        raise ValueError(msg)
    versions = [
        version
        for version in SUPPORTED_PYTHON_VERSIONS
        if all(_satisfies(_parse_version(version), clause) for clause in clauses)
    ]
    if not versions:
        msg = (
            f"requires-python {requires_python!r} matches none of "
            f"{list(SUPPORTED_PYTHON_VERSIONS)}"
        )
        raise ValueError(msg)
    return versions


def resolve_from_pyproject(contents: str) -> list[str]:
    """Resolve supported interpreters from `pyproject.toml` text."""
    data = tomllib.loads(contents)
    requires_python = data.get("project", {}).get("requires-python")
    if not isinstance(requires_python, str):
        msg = "pyproject.toml does not declare [project].requires-python"
        raise ValueError(msg)
    return resolve_python_versions(requires_python)


def main() -> int:
    """Print the baseline and matrix as GitHub Actions outputs."""
    try:
        versions = resolve_from_pyproject(sys.stdin.read())
    except (ValueError, tomllib.TOMLDecodeError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"python-version={versions[0]}")
    print(f"python-versions={json.dumps(versions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
