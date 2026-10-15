"""Run lockfile checks only for packages touched by changed paths."""

from __future__ import annotations

import shlex
import subprocess
import sys
import tomllib
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBS_ROOT = REPO_ROOT / "libs"
EXAMPLES_ROOT = REPO_ROOT / "examples"


def package_dirs() -> list[Path]:
    """Return every package directory that owns a `uv.lock`, in repo-path order."""
    libs = [path.parent for path in LIBS_ROOT.glob("*/Makefile")]
    partners = [path.parent for path in (LIBS_ROOT / "partners").glob("*/Makefile")]
    examples = [path.parent for path in EXAMPLES_ROOT.glob("*/pyproject.toml")]
    return sorted([*libs, *partners, *examples], key=_repo_path)


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def python_version(package: Path) -> str:
    """Return the interpreter version `package`'s lockfile must be resolved with."""
    if package == LIBS_ROOT / "acp":
        return "3.14"
    return "3.12"


def _path_source_dirs(manifest_dir: Path, manifest: Mapping[str, object]) -> set[str]:
    """Return repo-relative dirs `manifest` consumes via `[tool.uv.sources]` paths."""
    tool = manifest.get("tool")
    uv = tool.get("uv") if isinstance(tool, Mapping) else None
    sources = uv.get("sources") if isinstance(uv, Mapping) else None
    if not isinstance(sources, Mapping):
        return set()

    dirs: set[str] = set()
    for spec in sources.values():
        if not isinstance(spec, Mapping):
            continue
        raw = spec.get("path")
        if not isinstance(raw, str):
            continue
        resolved = (manifest_dir / raw).resolve()
        try:
            dirs.add(resolved.relative_to(REPO_ROOT).as_posix())
        except ValueError:
            continue
    return dirs


def stale_lock_dirs(changed_manifests: Collection[str]) -> list[str]:
    """Return changed package directories and their transitive local consumers."""
    consumes: dict[str, set[str]] = {}
    for package in package_dirs():
        manifest_file = package / "pyproject.toml"
        if not manifest_file.is_file():
            continue
        repo_path = package.relative_to(REPO_ROOT).as_posix()
        manifest = tomllib.loads(manifest_file.read_text(encoding="utf-8"))
        consumes[repo_path] = _path_source_dirs(package, manifest)

    stale = {PurePosixPath(path).parent.as_posix() for path in changed_manifests}
    while True:
        dependents = {
            package
            for package, sources in consumes.items()
            if sources & stale and package not in stale
        }
        if not dependents:
            return sorted(stale)
        stale |= dependents


def _label(package: Path) -> str:
    try:
        return package.relative_to(LIBS_ROOT).as_posix()
    except ValueError:
        return f"../{package.relative_to(REPO_ROOT).as_posix()}"


def _touches_package(path: str, package: Path) -> bool:
    package_path = _repo_path(package)
    return path == package_path or path.startswith(f"{package_path}/")


def _packages_for_paths(paths: list[str]) -> list[Path]:
    packages = package_dirs()
    if not paths:
        return packages
    return [
        package
        for package in packages
        if any(_touches_package(path, package) for path in paths)
    ]


def _lock_command(package: Path, *, check: bool) -> list[str]:
    command = ["uv", "lock"]
    if check:
        command.append("--check")
    return [
        *command,
        "--directory",
        package.relative_to(REPO_ROOT).as_posix(),
        "--python",
        python_version(package),
    ]


def _lockfile_error(package: Path) -> str:
    package_path = package.relative_to(REPO_ROOT).as_posix()
    lockfile = f"{package_path}/uv.lock"
    command = shlex.join(_lock_command(package, check=False))
    return (
        f"::error file={lockfile},title=Out-of-date uv.lock::"
        f"{lockfile} is out of sync with {package_path}/pyproject.toml. "
        f"From the repository root, run `{command}` and commit the updated lockfile."
    )


def main(paths: list[str]) -> int:
    """Check lockfiles for packages touched by `paths`, or every package if empty."""
    packages = _packages_for_paths(paths)
    if not packages:
        print("✅ No package lockfiles need checking.")
        return 0
    for package in packages:
        print(f"🔍 Checking {_label(package)}")
        result = subprocess.run(
            _lock_command(package, check=True),
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(_lockfile_error(package), file=sys.stderr)
            return result.returncode
    print("✅ All applicable lockfiles are up-to-date!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
