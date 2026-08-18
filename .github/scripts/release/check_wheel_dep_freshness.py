"""Validate a release wheel's dependency metadata against published PyPI files."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import TYPE_CHECKING

import tomllib
from check_dep_freshness import extract_minimum
from check_release_deps import (
    DEFAULT_CONFIG,
    FetchPyPI,
    PyPIRequestError,
    _write_output,
    _write_step_summary,
    fetch_pypi_json,
)
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from email.message import Message

MAX_FETCH_WORKERS = 8
RELEASE_TITLE = re.compile(r"^release\(([^)]+)\):\s+\S+")


@dataclass(frozen=True)
class WheelMetadata:
    """Dependency metadata read from one built wheel."""

    name: str
    version: Version
    requires_python: str
    requires_dist: tuple[Requirement, ...]


@dataclass(frozen=True)
class ReleaseTarget:
    """Release package selected from a release-please PR title."""

    component: str
    package_name: str
    package_path: str
    python_version: str


@dataclass(frozen=True)
class DependencyCheck:
    """Result of comparing one wheel dependency with PyPI metadata."""

    dependency_name: str
    constraint: str
    passed: bool
    message: str


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _error(path: Path, message: str) -> None:
    print(f"::error file={path}::{message}")


def parse_wheel_metadata(wheel_path: Path) -> WheelMetadata:
    """Read package and dependency metadata from a wheel.

    Args:
        wheel_path: Wheel archive to inspect.

    Returns:
        Parsed wheel metadata.

    Raises:
        ValueError: If the wheel has missing, ambiguous, or invalid metadata.
        OSError: If the wheel cannot be read.
    """
    with zipfile.ZipFile(wheel_path) as wheel:
        metadata_paths = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            msg = (
                f"Expected exactly one .dist-info/METADATA file in {wheel_path}, "
                f"found {len(metadata_paths)}"
            )
            raise ValueError(msg)
        message = BytesParser(policy=default).parsebytes(
            wheel.read(metadata_paths[0]),
        )
    return _parse_metadata_message(message, wheel_path)


def _parse_metadata_message(message: Message, wheel_path: Path) -> WheelMetadata:
    name = message.get("Name")
    raw_version = message.get("Version")
    requires_python = message.get("Requires-Python")
    if not name or not raw_version or not requires_python:
        msg = f"{wheel_path} METADATA must declare Name, Version, and Requires-Python"
        raise ValueError(msg)
    try:
        version = Version(raw_version)
        python_specifier = SpecifierSet(requires_python)
    except (InvalidVersion, InvalidSpecifier) as err:
        msg = f"Invalid wheel metadata in {wheel_path}: {err}"
        raise ValueError(msg) from err
    if extract_minimum(python_specifier) is None:
        msg = f"{wheel_path} Requires-Python has no concrete lower bound: {requires_python}"
        raise ValueError(msg)

    requirements: list[Requirement] = []
    for raw_requirement in message.get_all("Requires-Dist", []):
        try:
            requirements.append(Requirement(raw_requirement))
        except InvalidRequirement as err:
            msg = f"Invalid Requires-Dist in {wheel_path}: {raw_requirement!r} ({err})"
            raise ValueError(msg) from err
    return WheelMetadata(
        name=name,
        version=version,
        requires_python=requires_python,
        requires_dist=tuple(requirements),
    )


def _load_release_config(config_path: Path = DEFAULT_CONFIG) -> Mapping[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    packages = config.get("packages")
    if not isinstance(packages, Mapping):
        msg = f"{config_path} has no packages mapping"
        raise TypeError(msg)
    return packages


def release_python_version(package_name: str) -> str:
    """Return the interpreter used by `release.yml` for a package.

    Args:
        package_name: Published distribution name.

    Returns:
        Python minor version used for release artifact tests.
    """
    if package_name in {"deepagents-code", "deepagents-talon"}:
        return "3.12"
    return "3.11"


def resolve_release_target(
    title: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> ReleaseTarget:
    """Resolve a release-please title to its configured package.

    Args:
        title: Pull request title.
        config_path: Release-please configuration path.

    Returns:
        Selected release package and interpreter.

    Raises:
        ValueError: If the title is not a known release-please title.
    """
    match = RELEASE_TITLE.match(title)
    if match is None:
        msg = f"Not a release-please title: {title}"
        raise ValueError(msg)
    component = match.group(1)
    for package_path, raw_metadata in _load_release_config(config_path).items():
        if not isinstance(package_path, str) or not isinstance(raw_metadata, Mapping):
            continue
        package_name = raw_metadata.get("package-name")
        configured_component = raw_metadata.get("component")
        if component not in {package_name, configured_component}:
            continue
        if not isinstance(package_name, str):
            msg = f"Release package {package_path} has no package-name"
            raise TypeError(msg)
        return ReleaseTarget(
            component=component,
            package_name=package_name,
            package_path=package_path,
            python_version=release_python_version(package_name),
        )
    msg = f"Release component {component!r} is not configured in {config_path}"
    raise ValueError(msg)


def load_sibling_python_metadata(
    *,
    repo_root: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, str]:
    """Load current `Requires-Python` values for repo-managed distributions.

    Args:
        repo_root: Repository root containing release package manifests.
        config_path: Release-please configuration path.

    Returns:
        Canonical distribution names mapped to current Python constraints.
    """
    constraints: dict[str, str] = {}
    for package_path in _load_release_config(config_path):
        if not isinstance(package_path, str):
            continue
        manifest_path = repo_root / package_path / "pyproject.toml"
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        project = manifest.get("project")
        if not isinstance(project, Mapping):
            continue
        name = project.get("name")
        requires_python = project.get("requires-python")
        if isinstance(name, str) and isinstance(requires_python, str):
            constraints[canonicalize_name(name)] = requires_python
    return constraints


def _wheel_requirements(metadata: WheelMetadata) -> tuple[Requirement, ...]:
    self_name = canonicalize_name(metadata.name)
    requirements: list[Requirement] = []
    seen: set[tuple[str, str, str]] = set()
    for requirement in metadata.requires_dist:
        name = canonicalize_name(requirement.name)
        if name == self_name:
            continue
        key = (name, str(requirement.specifier), requirement.url or "")
        if key in seen:
            continue
        seen.add(key)
        requirements.append(requirement)
    return tuple(requirements)


def _release_versions(
    payload: Mapping[str, object],
) -> list[tuple[Version, list[object]]]:
    releases = payload.get("releases")
    if not isinstance(releases, Mapping):
        return []
    versions: list[tuple[Version, list[object]]] = []
    for raw_version, raw_files in releases.items():
        if (
            not isinstance(raw_version, str)
            or not isinstance(raw_files, list)
            or not raw_files
        ):
            continue
        try:
            version = Version(raw_version)
        except InvalidVersion:
            continue
        versions.append((version, raw_files))
    return sorted(versions, key=lambda item: item[0], reverse=True)


def _file_python_constraints(files: Sequence[object]) -> tuple[str, ...]:
    constraints: list[str] = []
    for raw_file in files:
        if not isinstance(raw_file, Mapping):
            continue
        raw_constraint = raw_file.get("requires_python")
        constraint = raw_constraint if isinstance(raw_constraint, str) else ""
        if constraint not in constraints:
            constraints.append(constraint)
    return tuple(constraints)


def _python_constraint_matches(
    constraint: str,
    *,
    minimum_python: Version,
    expected_constraint: str | None,
) -> bool:
    try:
        specifier = SpecifierSet(constraint)
        expected = (
            SpecifierSet(expected_constraint)
            if expected_constraint is not None
            else None
        )
    except InvalidSpecifier:
        return False
    if not specifier.contains(minimum_python, prereleases=True):
        return False
    return expected is None or specifier == expected


def _constraint_label(requirement: Requirement) -> str:
    return str(requirement.specifier) or "(any version)"


def _python_label(constraints: Sequence[str]) -> str:
    labels = [constraint or "(not declared)" for constraint in constraints]
    return ", ".join(labels) if labels else "(not declared)"


def _failure_message(
    metadata: WheelMetadata,
    requirement: Requirement,
    versions: Sequence[tuple[Version, list[object]]],
    *,
    expected_constraint: str | None,
) -> str:
    constraint = _constraint_label(requirement)
    matching = [
        (version, files)
        for version, files in versions
        if requirement.specifier.contains(version)
    ]
    if not matching:
        latest = f"; latest on PyPI is {versions[0][0]}" if versions else ""
        return (
            f"No published {requirement.name} release satisfies {constraint}{latest}. "
            f"Publish a compatible {requirement.name} release first, then re-run this check."
        )

    version, files = matching[0]
    latest_label = (
        "latest on PyPI"
        if versions and version == versions[0][0]
        else f"newest on PyPI satisfying {constraint}"
    )
    published_python = _python_label(_file_python_constraints(files))
    if expected_constraint is not None:
        return (
            f"{requirement.name} {version} ({latest_label}) declares "
            f"requires-python{published_python}, but the current {requirement.name} "
            f"metadata requires {expected_constraint}. Constraint: {constraint}. "
            f"Release {requirement.name} first, then re-run this check."
        )
    return (
        f"{requirement.name} {version} ({latest_label}) declares "
        f"requires-python{published_python}, but {metadata.name} {metadata.version} "
        f"requires Python {metadata.requires_python}. Constraint: {constraint}. "
        f"Publish a compatible {requirement.name} release first, then re-run this check."
    )


def compare_wheel_with_pypi(
    metadata: WheelMetadata,
    pypi_payloads: Mapping[str, Mapping[str, object]],
    *,
    sibling_requires_python: Mapping[str, str] | None = None,
) -> tuple[DependencyCheck, ...]:
    """Compare built-wheel requirements with immutable PyPI release metadata.

    Repo-managed dependencies receive an additional freshness assertion: their newest
    eligible PyPI files must carry the same `Requires-Python` constraint as the current
    sibling source. This distinguishes a legitimately broad third-party Python range
    from an unpublished sibling metadata change such as the coordinated Code/Talon
    Python-floor bump.

    Args:
        metadata: Parsed metadata from the package being released.
        pypi_payloads: Canonical dependency names mapped to PyPI project JSON.
        sibling_requires_python: Current Python constraints for repo-managed siblings.

    Returns:
        One deterministic pass/fail result per distinct dependency constraint.

    Raises:
        ValueError: If the release wheel has no concrete Python lower bound.
    """
    minimum_python = extract_minimum(SpecifierSet(metadata.requires_python))
    if minimum_python is None:
        msg = f"{metadata.name} has no concrete Requires-Python lower bound"
        raise ValueError(msg)
    sibling_constraints = sibling_requires_python or {}
    checks: list[DependencyCheck] = []
    for requirement in _wheel_requirements(metadata):
        constraint = _constraint_label(requirement)
        if requirement.url:
            checks.append(
                DependencyCheck(
                    dependency_name=requirement.name,
                    constraint=requirement.url,
                    passed=False,
                    message=(
                        f"{requirement.name} uses the direct URL {requirement.url}. "
                        "Release dependencies must resolve from PyPI; publish the "
                        "dependency there and replace the direct reference first."
                    ),
                )
            )
            continue

        canonical_name = canonicalize_name(requirement.name)
        payload = pypi_payloads.get(canonical_name)
        if payload is None:
            checks.append(
                DependencyCheck(
                    dependency_name=requirement.name,
                    constraint=constraint,
                    passed=False,
                    message=(
                        f"PyPI metadata for {requirement.name} was not provided. "
                        "Re-run this check before releasing."
                    ),
                )
            )
            continue
        versions = _release_versions(payload)
        matching_versions = [
            (version, files)
            for version, files in versions
            if requirement.specifier.contains(version)
        ]
        expected_constraint = sibling_constraints.get(canonical_name)
        candidates = (
            matching_versions[:1]
            if expected_constraint is not None
            else matching_versions
        )
        passed = any(
            any(
                _python_constraint_matches(
                    python_constraint,
                    minimum_python=minimum_python,
                    expected_constraint=expected_constraint,
                )
                for python_constraint in _file_python_constraints(files)
            )
            for _, files in candidates
        )
        checks.append(
            DependencyCheck(
                dependency_name=requirement.name,
                constraint=constraint,
                passed=passed,
                message=(
                    f"{requirement.name} has a published release satisfying {constraint} "
                    f"and Python {metadata.requires_python}."
                    if passed
                    else _failure_message(
                        metadata,
                        requirement,
                        versions,
                        expected_constraint=expected_constraint,
                    )
                ),
            )
        )
    return tuple(checks)


def _fetch_payloads(
    names: set[str],
    fetcher: FetchPyPI,
) -> tuple[dict[str, Mapping[str, object]], dict[str, PyPIRequestError]]:
    payloads: dict[str, Mapping[str, object]] = {}
    failures: dict[str, PyPIRequestError] = {}
    if not names:
        return payloads, failures
    with ThreadPoolExecutor(max_workers=min(MAX_FETCH_WORKERS, len(names))) as executor:
        futures = {executor.submit(fetcher, name): name for name in sorted(names)}
        for future in as_completed(futures):
            name = futures[future]
            try:
                payloads[name] = future.result()
            except PyPIRequestError as err:
                failures[name] = err
    return payloads, failures


def validate_wheel(
    wheel_path: Path,
    *,
    repo_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    fetcher: FetchPyPI = fetch_pypi_json,
) -> int:
    """Validate a built wheel against live PyPI metadata.

    Args:
        wheel_path: Built release wheel.
        repo_root: Repository root used to read sibling package metadata.
        config_path: Release-please configuration path.
        fetcher: Injectable PyPI JSON client.

    Returns:
        Zero when every dependency is published with compatible metadata, otherwise
        one for dependency failures or two for an indeterminate PyPI query.
    """
    metadata = parse_wheel_metadata(wheel_path)
    requirements = _wheel_requirements(metadata)
    names = {
        canonicalize_name(requirement.name)
        for requirement in requirements
        if not requirement.url
    }
    payloads, fetch_failures = _fetch_payloads(names, fetcher)
    if fetch_failures:
        for name, error in sorted(fetch_failures.items()):
            _error(wheel_path, f"Could not query PyPI for {name}: {error}")
        return 2

    sibling_constraints = load_sibling_python_metadata(
        repo_root=repo_root,
        config_path=config_path,
    )
    checks = compare_wheel_with_pypi(
        metadata,
        payloads,
        sibling_requires_python=sibling_constraints,
    )
    failures = [check for check in checks if not check.passed]
    if not failures:
        _notice(
            f"All {len(checks)} dependency constraints in {wheel_path.name} have "
            "compatible published metadata."
        )
        return 0

    for failure in failures:
        _error(wheel_path, failure.message)
    summary = [
        "## Published dependency metadata is not ready",
        "",
        f"`{metadata.name}=={metadata.version}` cannot be released yet:",
        "",
    ]
    summary.extend(f"- {failure.message}" for failure in failures)
    _write_step_summary("\n".join(summary))
    return 1


def _target_command(title: str, config_path: Path) -> int:
    target = resolve_release_target(title, config_path=config_path)
    _write_output("package_name", target.package_name)
    _write_output("package_path", target.package_path)
    _write_output("python_version", target.python_version)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser("target", help="resolve a release PR package")
    target.add_argument("--title", required=True)
    target.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    validate = subparsers.add_parser("validate", help="validate one built wheel")
    validate.add_argument("--wheel", type=Path, required=True)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run release-target detection or built-wheel metadata validation."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "target":
            return _target_command(args.title, args.config)
        return validate_wheel(
            args.wheel,
            repo_root=args.repo_root,
            config_path=args.config,
        )
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as err:
        print(f"::error::Dependency metadata validation failed: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
