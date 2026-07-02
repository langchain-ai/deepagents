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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"
BYPASS_LABEL = "release-deps: acknowledged"
COMMENT_MARKER = "<!-- release-deps-check -->"
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
class ResolverFailure:
    manifest_path: str
    """Path to the release package manifest that failed to resolve."""

    package_name: str
    """Published package name from the failed manifest."""

    log: str
    """Combined resolver output for the failed manifest."""

    transient: bool
    """Whether the resolver output looks like a network or package-index failure."""

    affected_extras: tuple[str, ...]
    """Optional dependency extras inferred from the resolver conflict."""


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _error(message: str) -> None:
    print(f"::error::{message}")


def _run_git(
    args: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
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


def changed_manifests(
    base_sha: str, head_sha: str, package_paths: list[str]
) -> list[str]:
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
    preserved = {
        key: uv[key] for key in RESOLVER_UV_KEYS if isinstance(uv, dict) and key in uv
    }
    if preserved:
        lines.extend(["", "[tool.uv]"])
        for key, value in preserved.items():
            lines.append(f"{key} = {_toml_value(value)}")

    return "\n".join(lines) + "\n"


def is_transient_resolver_error(log: str) -> bool:
    """Return whether resolver output looks like a transient network/index failure."""
    return bool(TRANSIENT_PATTERNS.search(log))


def _name_in_log(name: str, lower_log: str) -> bool:
    """Return whether `name` appears in `lower_log` as a whole package token.

    A naive substring test over-reports — `click` would match inside
    `clickhouse-driver` and `requests` inside `requests-toolbelt` — so require
    boundaries that are not distribution-name characters (`[\\w.-]`). `name` is
    an already-canonicalized, lowercased distribution name.
    """
    pattern = rf"(?<![\w.-]){re.escape(name)}(?![\w.-])"
    return re.search(pattern, lower_log) is not None


def _affected_extras(data: dict[str, Any], log: str) -> tuple[str, ...]:
    """Infer which optional-dependency extras a resolver conflict touches.

    Only enriches the failure comment with likely-affected install targets; it
    never influences the pass/fail decision. Returns a sorted, de-duplicated
    tuple of extra names.
    """
    project = data.get("project", {})
    if not isinstance(project, dict):
        return ()

    package_name = project.get("name")
    if not isinstance(package_name, str):
        return ()

    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        return ()

    affected: set[str] = set()
    lower_log = log.lower()
    self_name = canonicalize_name(package_name)
    parsed: dict[str, list[Requirement]] = {}
    for extra, specs in optional_dependencies.items():
        if not isinstance(extra, str) or not isinstance(specs, list):
            continue
        requirements = []
        for spec in specs:
            if not isinstance(spec, str):
                continue
            try:
                requirement = Requirement(spec)
            except InvalidRequirement:
                continue
            requirements.append(requirement)
            requirement_name = canonicalize_name(requirement.name)
            if requirement_name != self_name and _name_in_log(
                requirement_name, lower_log
            ):
                affected.add(extra)
        parsed[extra] = requirements

    # An extra that pulls in the package's own already-affected extras (e.g.
    # `all = ["pkg[sandbox]"]`) is affected too. Iterate to a fixpoint so these
    # transitive self-references propagate regardless of declaration order.
    changed = True
    while changed:
        changed = False
        for extra, requirements in parsed.items():
            if extra in affected:
                continue
            if any(
                canonicalize_name(requirement.name) == self_name
                and any(nested in affected for nested in requirement.extras)
                for requirement in requirements
            ):
                affected.add(extra)
                changed = True

    return tuple(sorted(affected))


def _relevant_log(log: str, *, max_lines: int = 80) -> str:
    lines = [line.rstrip() for line in log.strip().splitlines()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(
        [f"... {len(lines) - max_lines} earlier lines omitted ...", *lines[-max_lines:]]
    )


def _write_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with Path(summary_path).open("a", encoding="utf-8") as summary:
        summary.write(markdown)
        summary.write("\n")


def _write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    delimiter = f"__{name.upper()}_EOF__"
    while delimiter in value:
        delimiter += "_"
    with Path(output_path).open("a", encoding="utf-8") as output:
        output.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def _failure_markdown(failures: list[ResolverFailure], *, include_marker: bool) -> str:
    lines: list[str] = []
    if include_marker:
        lines.append(COMMENT_MARKER)
    lines.extend(
        [
            "## Release dependency resolution failed",
            "",
            "This release PR changes package metadata that does not currently resolve against published PyPI packages.",
            "The check ignores local editable sources and runs `uv pip compile --no-sources --universal --prerelease allow --all-extras`, so it catches install failures users would see after release.",
            "",
        ]
    )

    for failure in failures:
        lines.extend(
            [
                f"### `{failure.manifest_path}`",
                "",
            ]
        )
        if failure.affected_extras:
            targets = ", ".join(
                f"`{failure.package_name}[{extra}]`"
                for extra in failure.affected_extras
            )
            lines.extend(
                [
                    f"Likely affected install targets: {targets}.",
                    "",
                ]
            )
        elif failure.transient:
            lines.extend(
                [
                    "The resolver output looks like a transient network or package-index failure. Re-run the job before changing package metadata.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "The failing install target could not be inferred from the resolver output; review the conflict below for the affected dependency path.",
                    "",
                ]
            )
        lines.extend(
            [
                "```text",
                _relevant_log(failure.log),
                "```",
                "",
            ]
        )

    if any(not failure.transient for failure in failures):
        lines.extend(
            [
                f"If this is an intentional cross-package release-order issue, add the `{BYPASS_LABEL}` label and publish the compatible dependency shortly after this release.",
                f"Do not use `{BYPASS_LABEL}` for accidental unsatisfiable dependency ranges; fix the dependency metadata instead.",
            ]
        )

    return "\n".join(lines).rstrip()


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
        _write_output("failed", "false")
        _write_output("comment_body", "")
        return 0

    _notice(f"Changed package manifests: {', '.join(manifests)}")

    failures: list[ResolverFailure] = []
    with tempfile.TemporaryDirectory(prefix="release-deps-") as tmp:
        tmpdir = Path(tmp)
        for index, manifest_path in enumerate(manifests):
            data = tomllib.loads(
                (REPO_ROOT / manifest_path).read_text(encoding="utf-8")
            )
            content = build_resolver_manifest(data)

            manifest_label = manifest_path.removesuffix("/pyproject.toml").replace(
                "/", "__"
            )
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
                output = log.read_text(encoding="utf-8")
                project = data.get("project", {})
                package_name = (
                    project.get("name") if isinstance(project, dict) else None
                )
                failures.append(
                    ResolverFailure(
                        manifest_path=manifest_path,
                        package_name=package_name
                        if isinstance(package_name, str)
                        else packages[manifest_path.removesuffix("/pyproject.toml")],
                        log=output,
                        transient=is_transient_resolver_error(output),
                        affected_extras=_affected_extras(data, output),
                    )
                )

    if not failures:
        _write_output("failed", "false")
        _write_output("comment_body", "")
        return 0

    summary = _failure_markdown(failures, include_marker=False)
    _write_step_summary(summary)
    _write_output("failed", "true")
    non_transient_failures = [failure for failure in failures if not failure.transient]
    _write_output(
        "comment_body",
        _failure_markdown(non_transient_failures, include_marker=True)
        if non_transient_failures
        else "",
    )
    return 1


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
