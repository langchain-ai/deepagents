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
from itertools import product
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
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from email.message import Message

MAX_FETCH_WORKERS = 8
MAX_MARKER_VALUES_PER_VARIABLE = 16
PYTHON_MINOR_PROBE_LIMIT = 100
# Newest CPython minor with a stable release, used to tell a "not tested on
# the next minor yet" upper bound (warning) from a cap that excludes an
# interpreter users actually run (hard failure). Pinned to the interpreter
# `check_dep_freshness.yml` runs this script under; bump both together.
CURRENT_CPYTHON_MINOR = 14
RELEASE_TITLE = re.compile(r"^release\(([^)]+)\):\s+\S+")
MARKER_VARIABLES = (
    "implementation_name",
    "implementation_version",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_release",
    "platform_system",
    "platform_version",
    "python_full_version",
    "python_version",
    "sys_platform",
    "extra",
)
PLATFORM_ENVIRONMENTS: tuple[Mapping[str, str], ...] = (
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "6.0.0",
        "platform_system": "Linux",
        "platform_version": "6.0.0",
        "sys_platform": "linux",
    },
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "aarch64",
        "platform_python_implementation": "CPython",
        "platform_release": "6.0.0",
        "platform_system": "Linux",
        "platform_version": "6.0.0",
        "sys_platform": "linux",
    },
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "ppc64le",
        "platform_python_implementation": "CPython",
        "platform_release": "6.0.0",
        "platform_system": "Linux",
        "platform_version": "6.0.0",
        "sys_platform": "linux",
    },
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "23.0.0",
        "platform_system": "Darwin",
        "platform_version": "23.0.0",
        "sys_platform": "darwin",
    },
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "arm64",
        "platform_python_implementation": "CPython",
        "platform_release": "23.0.0",
        "platform_system": "Darwin",
        "platform_version": "23.0.0",
        "sys_platform": "darwin",
    },
    {
        "implementation_name": "cpython",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_python_implementation": "CPython",
        "platform_release": "10",
        "platform_system": "Windows",
        "platform_version": "10.0.0",
        "sys_platform": "win32",
    },
    {
        "implementation_name": "cpython",
        "os_name": "nt",
        "platform_machine": "ARM64",
        "platform_python_implementation": "CPython",
        "platform_release": "10",
        "platform_system": "Windows",
        "platform_version": "10.0.0",
        "sys_platform": "win32",
    },
    {
        "implementation_name": "cpython",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "14.0",
        "platform_system": "FreeBSD",
        "platform_version": "14.0",
        "sys_platform": "freebsd",
    },
)
MARKER_OPERATOR = r"(?:not\s+in|in|===|==|!=|~=|<=|>=|<|>)"


@dataclass(frozen=True)
class WheelMetadata:
    """Dependency metadata read from one built wheel."""

    name: str
    version: Version
    requires_python: str
    requires_dist: tuple[Requirement, ...]
    provides_extra: tuple[str, ...]


@dataclass(frozen=True)
class ApplicableRequirement:
    """One dependency constraint and the marker variants that activate it."""

    requirement: Requirement
    variants: tuple[Requirement, ...]


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
    warning: bool = False


def _intersection_omits_only_unreleased_minors(
    wheel_requires_python: str,
    published_constraints: Sequence[str],
    *,
    current_minor: int = CURRENT_CPYTHON_MINOR,
) -> bool:
    """Whether the dependency's Python coverage gap touches no existing minor.

    A minor line is installable when at least one of its probed versions
    satisfies both the wheel's `Requires-Python` and every published file
    constraint. The gap must then start at the first unreleased minor and
    extend contiguously to the wheel's own ceiling, i.e. a "not tested on the
    next minor yet" upper bound such as `<3.15`. A gap at the wheel's minimum
    (the dependency rejects the release's install floor), in the middle of the
    claimed range, or covering already-released minors still fails.

    Failing probes are kept at full-version granularity: a patch-level
    exclusion on a released line (for example the wheel claims `>=3.12` but
    the dependency requires `>=3.12.1`) must not hide behind a working later
    patch of the same minor, so any released minor with a failed probe
    disqualifies the warning.

    Args:
        wheel_requires_python: `Requires-Python` declared by the release wheel.
        published_constraints: `requires_python` values from the published files.
        current_minor: Minor version of the newest released CPython line (e.g.
            14 when 3.14 is current). Defaults to `CURRENT_CPYTHON_MINOR`;
            tests override it to stay independent of the runner interpreter.

    Returns:
        `True` when only the next unreleased CPython minor and later are
        excluded by the dependency's upper bound.
    """
    wheel_specifier = SpecifierSet(wheel_requires_python)
    dep_specifiers = tuple(
        SpecifierSet(constraint) for constraint in published_constraints if constraint
    )
    wheel_minors: set[tuple[int, ...]] = set()
    dep_minors: set[tuple[int, ...]] = set()
    failed_minors: set[tuple[int, ...]] = set()
    for version in _supported_python_versions(
        wheel_requires_python,
        additional_constraints=published_constraints,
    ):
        minor = version.release[:2]
        wheel_minors.add(minor)
        if all(
            specifier.contains(version, prereleases=True)
            for specifier in dep_specifiers
        ):
            dep_minors.add(minor)
        else:
            failed_minors.add(minor)
    first_unreleased = current_minor + 1
    if any(minor[1] <= current_minor for minor in failed_minors):
        return False
    missing = sorted(wheel_minors - dep_minors)
    if not missing:
        return False
    floor = min(
        minor[1]
        for minor in wheel_minors
        if wheel_specifier.contains(Version(".".join(map(str, minor))))
    )
    if missing[0][1] <= floor:
        return False
    return missing[0][1] == first_unreleased and [
        minor for minor in sorted(wheel_minors) if minor[1] >= first_unreleased
    ] == missing


def _escape_command(value: object) -> str:
    """Escape a value for one GitHub Actions `::command::` line.

    Runner workflow commands end at a newline, so a PR-controlled string such as
    a `pyproject.toml` `requires-python` value must be percent-escaped before it
    is printed, or it could inject additional annotations.
    """
    return str(value).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _notice(message: str) -> None:
    print(f"::notice::{_escape_command(message)}")


def _warning(path: Path, message: str) -> None:
    print(f"::warning file={_escape_command(path)}::{_escape_command(message)}")


def _error(path: Path, message: str) -> None:
    print(f"::error file={_escape_command(path)}::{_escape_command(message)}")


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
        provides_extra=tuple(message.get_all("Provides-Extra", [])),
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


def _version_neighbors(version: Version) -> tuple[Version, ...]:
    release = list(version.release)
    neighbors = {version}
    above = release.copy()
    above[-1] += 1
    neighbors.add(Version(".".join(str(part) for part in above)))
    if len(release) == 2:
        neighbors.add(Version(f"{release[0]}.{release[1]}.1"))
    if release[-1] > 0:
        below = release.copy()
        below[-1] -= 1
        neighbors.add(Version(".".join(str(part) for part in below)))
    elif len(release) > 1 and release[-2] > 0:
        below = release[:-1]
        below[-1] -= 1
        below.append(999)
        neighbors.add(Version(".".join(str(part) for part in below)))
    return tuple(neighbors)


def _constraint_probe_versions(constraint: str) -> set[Version]:
    probes: set[Version] = set()
    try:
        specifier = SpecifierSet(constraint)
    except InvalidSpecifier:
        return probes
    for item in specifier:
        raw_version = item.version.removesuffix(".*")
        try:
            version = Version(raw_version)
        except InvalidVersion:
            continue
        probes.update(_version_neighbors(version))
    return probes


def _supported_python_versions(
    requires_python: str,
    *,
    additional_constraints: Sequence[str] = (),
    additional_versions: Sequence[Version] = (),
) -> tuple[Version, ...]:
    """Probe the supported versions in the package's current Python major.

    Python major versions that do not exist yet are not actionable release targets.
    Within the current major, every minor plus all constraint and marker boundaries
    are included so upper bounds and patch-level exclusions produce a witness.
    """
    specifier = SpecifierSet(requires_python)
    minimum = extract_minimum(specifier)
    if minimum is None:
        msg = f"Requires-Python has no concrete lower bound: {requires_python}"
        raise ValueError(msg)
    major = minimum.release[0]
    probes = {Version(f"{major}.{minor}") for minor in range(PYTHON_MINOR_PROBE_LIMIT)}
    for constraint in (requires_python, *additional_constraints):
        probes.update(_constraint_probe_versions(constraint))
    probes.update(additional_versions)
    supported = tuple(
        version
        for version in sorted(probes)
        if version.major == major and specifier.contains(version, prereleases=True)
    )
    if not supported:
        msg = f"Requires-Python has no supported {major}.x versions: {requires_python}"
        raise ValueError(msg)
    return supported


def _marker_literals(marker: str, variable: str) -> tuple[str, ...]:
    escaped = re.escape(variable)
    patterns = (
        rf"\b{escaped}\b\s*{MARKER_OPERATOR}\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
        rf"(?P<quote>['\"])(?P<value>.*?)(?P=quote)\s*{MARKER_OPERATOR}\s*\b{escaped}\b",
    )
    values: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, marker):
            value = match.group("value")
            values.add(value)
            values.update(part for part in re.split(r"[\s,]+", value) if part)
    return tuple(sorted(values))


def _marker_python_versions(requirement: Requirement) -> tuple[Version, ...]:
    if requirement.marker is None:
        return ()
    marker = str(requirement.marker)
    versions: set[Version] = set()
    for variable in (
        "implementation_version",
        "python_full_version",
        "python_version",
    ):
        for literal in _marker_literals(marker, variable):
            try:
                versions.update(_version_neighbors(Version(literal)))
            except InvalidVersion:
                continue
    return tuple(sorted(versions))


def _marker_environment_overrides(
    marker: str,
    provides_extra: Sequence[str],
) -> tuple[dict[str, str], ...]:
    platform_variables = [
        variable
        for variable in MARKER_VARIABLES
        if variable
        not in {
            "extra",
            "python_version",
            "python_full_version",
            "implementation_version",
        }
        and re.search(rf"\b{re.escape(variable)}\b", marker)
    ]
    environments = [
        {
            variable: value
            for variable, value in environment.items()
            if variable in platform_variables
        }
        for environment in PLATFORM_ENVIRONMENTS
    ]
    if not platform_variables:
        environments = [{}]

    unique_platforms = {
        tuple(sorted(environment.items())): environment for environment in environments
    }
    environments = list(unique_platforms.values())
    mutable_variables = sorted(
        {"platform_release", "platform_version"}.intersection(platform_variables)
    )
    if mutable_variables:
        expanded: list[dict[str, str]] = []
        for environment in environments:
            choices: list[tuple[str, ...]] = []
            for variable in mutable_variables:
                values = {environment.get(variable, "")}
                for literal in _marker_literals(marker, variable):
                    values.add(literal)
                    try:
                        values.update(
                            str(item) for item in _version_neighbors(Version(literal))
                        )
                    except InvalidVersion:
                        continue
                if len(values) > MAX_MARKER_VALUES_PER_VARIABLE:
                    msg = f"Marker has too many {variable} boundary values: {marker}"
                    raise ValueError(msg)
                choices.append(tuple(sorted(values)))
            expanded.extend(
                {**environment, **dict(zip(mutable_variables, values, strict=True))}
                for values in product(*choices)
            )
        environments = expanded

    extras = ("", *provides_extra) if re.search(r"\bextra\b", marker) else ("",)
    unique: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    for environment in environments:
        for extra in extras:
            candidate = {**environment, "extra": extra}
            unique[tuple(sorted(candidate.items()))] = candidate
    return tuple(unique.values())


def _active_python_versions(
    metadata: WheelMetadata,
    requirement: Requirement,
    supported_python: Sequence[Version],
) -> tuple[Version, ...]:
    if requirement.marker is None:
        return tuple(supported_python)

    marker = str(requirement.marker)
    overrides = _marker_environment_overrides(marker, metadata.provides_extra)
    baseline = default_environment()
    active: list[Version] = []
    for version in supported_python:
        python_version = f"{version.major}.{version.minor}"
        python_full_version = ".".join(str(part) for part in version.release)
        if len(version.release) == 2:
            python_full_version += ".0"
        for override in overrides:
            environment = {
                **baseline,
                **override,
                "implementation_version": python_full_version,
                "python_full_version": python_full_version,
                "python_version": python_version,
            }
            if requirement.marker.evaluate(environment=environment):
                active.append(version)
                break
    return tuple(active)


def _applicable_python_versions(
    metadata: WheelMetadata,
    variants: Sequence[Requirement],
    *,
    additional_constraints: Sequence[str] = (),
) -> tuple[Version, ...]:
    marker_versions = {
        version
        for requirement in variants
        for version in _marker_python_versions(requirement)
    }
    supported_python = _supported_python_versions(
        metadata.requires_python,
        additional_constraints=additional_constraints,
        additional_versions=tuple(marker_versions),
    )
    active: set[Version] = set()
    for requirement in variants:
        active.update(_active_python_versions(metadata, requirement, supported_python))
    return tuple(sorted(active))


def _applicable_requirements(
    metadata: WheelMetadata,
) -> tuple[ApplicableRequirement, ...]:
    self_name = canonicalize_name(metadata.name)
    grouped: dict[tuple[str, str, str], list[Requirement]] = {}
    for requirement in metadata.requires_dist:
        name = canonicalize_name(requirement.name)
        if name == self_name:
            continue
        key = (name, str(requirement.specifier), requirement.url or "")
        grouped.setdefault(key, []).append(requirement)

    applicable: list[ApplicableRequirement] = []
    for variants in grouped.values():
        python_versions = _applicable_python_versions(metadata, variants)
        if not python_versions:
            continue
        applicable.append(
            ApplicableRequirement(
                requirement=variants[0],
                variants=tuple(variants),
            )
        )
    return tuple(applicable)


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


def _python_constraints_match(
    constraints: Sequence[str],
    *,
    required_python: Sequence[Version],
    expected_constraint: str | None,
) -> bool:
    try:
        specifiers = tuple(SpecifierSet(constraint) for constraint in constraints)
        expected = (
            SpecifierSet(expected_constraint)
            if expected_constraint is not None
            else None
        )
    except InvalidSpecifier:
        return False
    if not specifiers:
        return False
    if expected is not None and any(specifier != expected for specifier in specifiers):
        return False
    return all(
        any(specifier.contains(version, prereleases=True) for specifier in specifiers)
        for version in required_python
    )


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

    A third-party dependency whose newest eligible release covers every existing
    CPython minor the wheel claims, but stops below a future minor (for example
    `requires-python<3.15` against the wheel's `<4.0`), degrades to a warning:
    resolvers already refuse that extra on the unreleased interpreter, so the
    release is not blocked on another project widening its bound. Gaps at real
    installation targets (the wheel's minimum, any minor the dependency excludes
    in the middle of the range, or a patch-level exclusion on a released line
    such as `>=3.12.1` against the wheel's `>=3.12`) stay hard failures.

    Args:
        metadata: Parsed metadata from the package being released.
        pypi_payloads: Canonical dependency names mapped to PyPI project JSON.
        sibling_requires_python: Current Python constraints for repo-managed siblings.

    Returns:
        One deterministic result per distinct dependency constraint; warnings have
        `passed=True` and `warning=True`.

    Raises:
        ValueError: If the release wheel has no concrete Python lower bound.
    """
    sibling_constraints = sibling_requires_python or {}
    checks: list[DependencyCheck] = []
    for applicable in _applicable_requirements(metadata):
        requirement = applicable.requirement
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
        passed = False
        warning = False
        for _, files in candidates:
            python_constraints = _file_python_constraints(files)
            required_python = _applicable_python_versions(
                metadata,
                applicable.variants,
                additional_constraints=python_constraints,
            )
            if _python_constraints_match(
                python_constraints,
                required_python=required_python,
                expected_constraint=expected_constraint,
            ):
                passed = True
                break
        if not passed and expected_constraint is None and matching_versions:
            _, files = matching_versions[0]
            published_python = _file_python_constraints(files)
            if _intersection_omits_only_unreleased_minors(
                metadata.requires_python,
                published_python,
            ):
                passed = True
                warning = True
        checks.append(
            DependencyCheck(
                dependency_name=requirement.name,
                constraint=constraint,
                passed=passed,
                warning=warning,
                message=(
                    f"{requirement.name} has a published release satisfying {constraint} "
                    f"and Python {metadata.requires_python}."
                    if passed and not warning
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
    requirements = _applicable_requirements(metadata)
    names = {
        canonicalize_name(item.requirement.name)
        for item in requirements
        if not item.requirement.url
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
    warnings = [check for check in checks if check.warning]
    for check in warnings:
        _warning(wheel_path, check.message)
    if not failures:
        _notice(
            f"All {len(checks)} dependency constraints in {wheel_path.name} have "
            "compatible published metadata."
        )
        if warnings:
            summary = [
                "## Dependency metadata warnings",
                "",
                f"`{metadata.name}=={metadata.version}` can be released, but these "
                "third-party dependencies do not yet publish support for every "
                "Python minor the wheel claims:",
                "",
            ]
            summary.extend(f"- {check.message}" for check in warnings)
            _write_step_summary("\n".join(summary))
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
        print(f"::error::Dependency metadata validation failed: {_escape_command(err)}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
