"""Run lockfile checks only for packages touched by changed paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_ROOT = REPO_ROOT / "libs"
EXAMPLES_ROOT = REPO_ROOT / "examples"


def _package_dirs() -> list[Path]:
    libs = [path.parent for path in LIBS_ROOT.glob("*/Makefile")]
    partners = [path.parent for path in (LIBS_ROOT / "partners").glob("*/Makefile")]
    examples = [path.parent for path in EXAMPLES_ROOT.glob("*/pyproject.toml")]
    return sorted([*libs, *partners, *examples], key=_repo_path)


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _python_version(package: Path) -> str:
    if package == LIBS_ROOT / "acp":
        return "3.14"
    return "3.12"


def _label(package: Path) -> str:
    try:
        return package.relative_to(LIBS_ROOT).as_posix()
    except ValueError:
        return f"../{package.relative_to(REPO_ROOT).as_posix()}"


def _touches_package(path: str, package: Path) -> bool:
    package_path = _repo_path(package)
    return path == package_path or path.startswith(f"{package_path}/")


def _packages_for_paths(paths: list[str]) -> list[Path]:
    packages = _package_dirs()
    if not paths:
        return packages
    return [
        package
        for package in packages
        if any(_touches_package(path, package) for path in paths)
    ]


def main(paths: list[str]) -> int:
    """Check lockfiles for packages touched by `paths`, or every package if empty."""
    packages = _packages_for_paths(paths)
    if not packages:
        print("✅ No package lockfiles need checking.")
        return 0
    for package in packages:
        print(f"🔍 Checking {_label(package)}")
        result = subprocess.run(
            [
                "uv",
                "lock",
                "--check",
                "--directory",
                str(package),
                "--python",
                _python_version(package),
            ],
            check=False,
        )
        if result.returncode != 0:
            return result.returncode
    print("✅ All applicable lockfiles are up-to-date!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
