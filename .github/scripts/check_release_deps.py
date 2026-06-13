"""Resolve release PR runtime dependencies against real PyPI.

A release PR bumps a single package's version. Local development installs the
sibling packages as editable path dependencies via `[tool.uv.sources]`, which
hides whether a package's *published* dependencies actually resolve. This check
strips those local sources and resolves each changed release manifest against
the real index (`uv pip compile --no-sources`), so an unsatisfiable or
not-yet-published runtime dependency fails before merge/publish instead of at
user install time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

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


def build_resolver_manifest(data: dict[str, Any]) -> str:
    """Build a resolver-equivalent pyproject that drops local path sources.

    The result is a minimal manifest holding only resolver-relevant fields:
    `name`, `version`, `requires-python`, dependencies, optional-dependencies,
    and the `RESOLVER_UV_KEYS` subset of `[tool.uv]`. It omits `[tool.uv.sources]`
    so resolution runs against real PyPI (paired with `--no-sources`).
    """
    project = data.get("project", {})
    if not isinstance(project, dict):
        msg = "manifest has no [project] table"
        raise ValueError(msg)

    lines: list[str] = ["[project]"]
    for key in ("name", "version", "requires-python"):
        value = project.get(key)
        if isinstance(value, str):
            lines.append(f"{key} = {_toml_value(value)}")
    lines.append(f"dependencies = {_toml_value(project.get('dependencies', []))}")

    optional_dependencies = project.get("optional-dependencies", {})
    if isinstance(optional_dependencies, dict) and optional_dependencies:
        lines.extend(["", "[project.optional-dependencies]"])
        for extra, deps in optional_dependencies.items():
            lines.append(f"{extra} = {_toml_value(deps)}")

    tool = data.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    preserved = {key: uv[key] for key in RESOLVER_UV_KEYS if isinstance(uv, dict) and key in uv}
    if preserved:
        lines.extend(["", "[tool.uv]"])
        for key, value in preserved.items():
            lines.append(f"{key} = {_toml_value(value)}")

    return "\n".join(lines) + "\n"


def is_transient_resolver_error(log: str) -> bool:
    """Return whether resolver output looks like a transient network/index failure."""
    return bool(TRANSIENT_PATTERNS.search(log))


def run_resolver(manifest: Path, log: Path) -> bool:
    """Resolve a manifest against real PyPI and write combined output to log.

    Resolution ignores local path sources (`--no-sources`), spans every extra
    (`--all-extras`), allows prereleases (`--prerelease allow`), and is universal
    across platforms/Python versions (`--universal`).
    """
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
            "Dependency resolution failed against PyPI. A declared runtime dependency "
            "could not be satisfied by published packages (e.g. a pin on a version that "
            f"is not on PyPI yet). If the pin is intentional, apply the `{BYPASS_LABEL}` label."
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

    ok = True
    with tempfile.TemporaryDirectory(prefix="release-deps-") as tmp:
        tmpdir = Path(tmp)
        for index, manifest_path in enumerate(manifests):
            data = tomllib.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
            content = build_resolver_manifest(data)

            manifest_label = manifest_path.removesuffix("/pyproject.toml").replace("/", "__")
            manifest_dir = tmpdir / f"{index}-{manifest_label}"
            manifest_dir.mkdir()
            temp_manifest = manifest_dir / "pyproject.toml"
            temp_manifest.write_text(content, encoding="utf-8")
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
