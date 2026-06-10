"""Run the lockfile check for pre-commit without unrelated Talon churn."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_ROOT = REPO_ROOT / "libs"
EXAMPLES_ROOT = REPO_ROOT / "examples"
TALON_DIR = LIBS_ROOT / "talon"


def _package_dirs(*, include_talon: bool) -> list[Path]:
    libs = [path.parent for path in LIBS_ROOT.glob("*/Makefile")]
    partners = [path.parent for path in (LIBS_ROOT / "partners").glob("*/Makefile")]
    examples = [path.parent for path in EXAMPLES_ROOT.glob("*/pyproject.toml")]
    packages = [*libs, *partners, *examples]
    if not include_talon:
        packages = [package for package in packages if package != TALON_DIR]
    return sorted(packages, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def _python_version(package: Path) -> str:
    if package == LIBS_ROOT / "acp":
        return "3.14"
    return "3.12"


def _label(package: Path) -> str:
    try:
        return package.relative_to(LIBS_ROOT).as_posix()
    except ValueError:
        return f"../{package.relative_to(REPO_ROOT).as_posix()}"


def _touches_talon(paths: list[str]) -> bool:
    return any(Path(path).parts[:2] == ("libs", "talon") for path in paths)


def _include_talon(paths: list[str]) -> bool:
    return not paths or _touches_talon(paths)


def main(paths: list[str]) -> int:
    include_talon = _include_talon(paths)
    for package in _package_dirs(include_talon=include_talon):
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
