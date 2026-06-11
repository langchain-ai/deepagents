"""Resolve release PR dependencies against PyPI without same-PR package bumps.

Release-please PRs can bump several packages in one coordinated change. The
newly bumped wheels do not exist on PyPI until merge, so dependency resolution
must ignore only the intra-PR pins that the PR itself satisfies while still
checking every other runtime dependency against the real index.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"
BYPASS_LABEL = "release-deps: acknowledged"
RESOLVER_UV_KEYS = (
    "prerelease",
    "constraint-dependencies",
    "override-dependencies",
)
TRANSIENT_PATTERNS = re.compile(
    r"(error sending request|failed to fetch|connection|timed out|temporarily unavailable|"
    r"http (?:429|5\d\d)|status code: (?:429|5\d\d))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PackageBump:
    """A package whose own version was bumped by this PR."""

    name: str
    canonical_name: str
    version: str
    path: str


@dataclass(frozen=True)
class FilteredManifest:
    """Filtered manifest content plus dependency pins removed from it."""

    content: str
    skipped: tuple[str, ...]


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _error(message: str) -> None:
    print(f"::error::{message}")


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def _git_show(ref: str, path: str) -> str | None:
    proc = _run_git(["show", f"{ref}:{path}"], check=False)
    if proc.returncode != 0:
        _warning(
            f"Could not read {path} at base SHA; assuming it is new or unavailable "
            "and skipping same-PR bump detection for that package."
        )
        return None
    return proc.stdout


def load_release_packages(config_path: Path = DEFAULT_CONFIG) -> dict[str, str]:
    """Return release-please package paths mapped to component/package labels."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    packages = config.get("packages")
    if not isinstance(packages, dict) or not packages:
        msg = f"release-please config {config_path} has no packages map"
        raise ValueError(msg)
    return {
        path: meta.get("package-name") or meta.get("component") or path
        for path, meta in packages.items()
        if isinstance(meta, dict)
    }


def changed_manifests(base_sha: str, head_sha: str, package_paths: list[str]) -> list[str]:
    """Return changed release-package pyproject paths between base and head."""
    manifest_paths = [f"{path}/pyproject.toml" for path in package_paths]
    proc = _run_git(["diff", "--name-only", base_sha, head_sha, "--", *manifest_paths])
    changed = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return [path for path in manifest_paths if path in changed]


def _parse_toml(content: str, path: str) -> dict[str, Any] | None:
    try:
        return tomllib.loads(content)
    except tomllib.TOMLDecodeError as err:
        _warning(f"Could not parse {path}: {err}; skipping same-PR bump detection.")
        return None


def _project_name_version(data: dict[str, Any], path: str) -> tuple[str, str] | None:
    project = data.get("project")
    if not isinstance(project, dict):
        _warning(f"{path} has no [project] table; skipping same-PR bump detection.")
        return None
    dynamic = project.get("dynamic", [])
    if isinstance(dynamic, list) and ({"name", "version"} & set(dynamic)):
        _warning(f"{path} has dynamic project name/version; skipping same-PR bump detection.")
        return None
    name = project.get("name")
    version = project.get("version")
    if not isinstance(name, str) or not name:
        _warning(f"{path} has no static [project].name; skipping same-PR bump detection.")
        return None
    if not isinstance(version, str) or not version:
        _warning(f"{path} has no static [project].version; skipping same-PR bump detection.")
        return None
    return name, version


def detect_package_bumps(paths: list[str], base_sha: str) -> dict[str, PackageBump]:
    """Return packages whose own project version changed in this PR."""
    bumped: dict[str, PackageBump] = {}
    for path in paths:
        current_path = REPO_ROOT / path
        current = _parse_toml(current_path.read_text(encoding="utf-8"), path)
        if current is None:
            continue
        current_identity = _project_name_version(current, path)
        if current_identity is None:
            continue

        base_content = _git_show(base_sha, path)
        if base_content is None:
            continue
        base = _parse_toml(base_content, f"{base_sha}:{path}")
        if base is None:
            continue
        base_identity = _project_name_version(base, f"{base_sha}:{path}")
        if base_identity is None:
            continue

        current_name, current_version = current_identity
        base_name, base_version = base_identity
        if canonicalize_name(current_name) != canonicalize_name(base_name):
            _warning(
                f"{path} renamed package {base_name!r} -> {current_name!r}; "
                "skipping same-PR bump detection for that manifest."
            )
            continue
        if current_version == base_version:
            continue

        canonical = canonicalize_name(current_name)
        bumped[canonical] = PackageBump(
            name=current_name,
            canonical_name=canonical,
            version=current_version,
            path=path,
        )
    return bumped


def _requirement_is_satisfied_by_bump(dep: str, bumped: dict[str, PackageBump]) -> bool:
    try:
        requirement = Requirement(dep)
    except InvalidRequirement as err:
        _warning(f"Could not parse dependency {dep!r}: {err}; leaving it in place.")
        return False
    bump = bumped.get(canonicalize_name(requirement.name))
    if bump is None:
        return False
    return requirement.specifier.contains(bump.version, prereleases=True)


def _filter_dep_list(deps: Any, bumped: dict[str, PackageBump]) -> tuple[Any, list[str]]:
    if not isinstance(deps, list):
        return deps, []
    kept: list[Any] = []
    skipped: list[str] = []
    for dep in deps:
        if isinstance(dep, str) and _requirement_is_satisfied_by_bump(dep, bumped):
            skipped.append(dep)
        else:
            kept.append(dep)
    return kept, skipped


def _quote(value: str) -> str:
    return json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            inner = ",\n  ".join(_quote(item) for item in value)
            return f"[\n  {inner},\n]"
        inner = ", ".join(_toml_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_toml_value(val)}" for key, val in value.items())
        return f"{{ {inner} }}"
    msg = f"Unsupported TOML value for resolver manifest: {value!r}"
    raise TypeError(msg)


def build_filtered_manifest(data: dict[str, Any], bumped: dict[str, PackageBump]) -> FilteredManifest:
    """Build a resolver-equivalent pyproject with same-PR dependency pins removed."""
    filtered = copy.deepcopy(data)
    project = filtered.get("project", {})
    if not isinstance(project, dict):
        msg = "manifest has no [project] table"
        raise ValueError(msg)

    skipped: list[str] = []
    dependencies, skipped_dependencies = _filter_dep_list(project.get("dependencies", []), bumped)
    project["dependencies"] = dependencies
    skipped.extend(skipped_dependencies)

    optional_dependencies = project.get("optional-dependencies", {})
    if isinstance(optional_dependencies, dict):
        for extra, deps in list(optional_dependencies.items()):
            filtered_deps, skipped_extra = _filter_dep_list(deps, bumped)
            optional_dependencies[extra] = filtered_deps
            skipped.extend(f"{extra}: {dep}" for dep in skipped_extra)

    lines: list[str] = ["[project]"]
    for key in ("name", "version", "requires-python"):
        value = project.get(key)
        if isinstance(value, str):
            lines.append(f"{key} = {_toml_value(value)}")
    lines.append(f"dependencies = {_toml_value(project.get('dependencies', []))}")

    if isinstance(optional_dependencies, dict) and optional_dependencies:
        lines.extend(["", "[project.optional-dependencies]"])
        for extra, deps in optional_dependencies.items():
            lines.append(f"{extra} = {_toml_value(deps)}")

    tool = filtered.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    preserved = {
        key: uv[key]
        for key in RESOLVER_UV_KEYS
        if isinstance(uv, dict) and key in uv
    }
    if preserved:
        lines.extend(["", "[tool.uv]"])
        for key, value in preserved.items():
            lines.append(f"{key} = {_toml_value(value)}")

    return FilteredManifest(content="\n".join(lines) + "\n", skipped=tuple(skipped))


def is_transient_resolver_error(log: str) -> bool:
    """Return whether resolver output looks like a transient network/index failure."""
    return bool(TRANSIENT_PATTERNS.search(log))


def run_resolver(manifest: Path, log: Path) -> bool:
    """Resolve a filtered manifest against real PyPI and write combined output to log."""
    proc = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "--no-sources",
            "--universal",
            "--prerelease",
            "allow",
            "--all-extras",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode == 0:
        return True

    print(proc.stdout)
    if is_transient_resolver_error(proc.stdout):
        _warning(
            "Dependency resolution failed with a likely transient network/index error. "
            "Re-run the job before treating this as an unsatisfiable release dependency."
        )
    else:
        _error(
            "Dependency resolution failed. If the failing dependency is another package "
            "bumped by this same release PR, this check removes only pins satisfied by "
            "that same-PR bump before resolving the rest. If the remaining pin is "
            f"intentional, apply the `{BYPASS_LABEL}` label."
        )
    return False


def check_release_dependencies(base_sha: str, head_sha: str) -> int:
    """Resolve changed release-package manifests and return a process exit code."""
    packages = load_release_packages()
    manifests = changed_manifests(base_sha, head_sha, list(packages))
    if not manifests:
        _notice("No release-package pyproject.toml files changed; nothing to check.")
        return 0

    _notice(f"Changed package manifests: {', '.join(manifests)}")
    bumped = detect_package_bumps(manifests, base_sha)
    if bumped:
        introduced = ", ".join(
            f"{bump.name}=={bump.version} ({bump.path})"
            for bump in sorted(bumped.values(), key=lambda item: item.name)
        )
        _notice(f"Packages introduced by this PR: {introduced}")
    else:
        _notice("No same-PR package version bumps detected.")

    ok = True
    with tempfile.TemporaryDirectory(prefix="release-deps-") as tmp:
        tmpdir = Path(tmp)
        for index, manifest_path in enumerate(manifests):
            data = tomllib.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
            filtered = build_filtered_manifest(data, bumped)
            if filtered.skipped:
                _notice(
                    f"{manifest_path}: skipped same-PR dependency pins: "
                    + "; ".join(filtered.skipped)
                )
            else:
                _notice(f"{manifest_path}: no same-PR dependency pins skipped.")

            manifest_label = manifest_path.removesuffix("/pyproject.toml").replace("/", "__")
            manifest_dir = tmpdir / f"{index}-{manifest_label}"
            manifest_dir.mkdir()
            temp_manifest = manifest_dir / "pyproject.toml"
            temp_manifest.write_text(filtered.content, encoding="utf-8")
            log = tmpdir / f"{manifest_dir.name}.log"
            _notice(
                f"Resolving {manifest_path} against PyPI with "
                "uv pip compile --no-sources --universal --prerelease allow --all-extras"
            )
            if not run_resolver(temp_manifest, log):
                ok = False

    return 0 if ok else 1


def main() -> int:
    """CLI entry point used by the GitHub Actions workflow."""
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")
    if not base_sha or not head_sha:
        _error("BASE_SHA and HEAD_SHA must be set")
        return 2
    try:
        return check_release_dependencies(base_sha, head_sha)
    except Exception as err:  # noqa: BLE001  # fail closed with a clear CI annotation
        _error(f"Release dependency check failed unexpectedly: {err}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
