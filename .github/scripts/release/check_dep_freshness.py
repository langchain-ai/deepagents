"""Report release-package dependency minimums that trail published PyPI versions."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Self

from check_release_deps import (
    FetchPyPI,
    PyPIRequestError,
    _write_output,
    _write_step_summary,
    changed_manifests,
    fetch_pypi_json,
    load_release_packages,
)
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from packaging.specifiers import SpecifierSet

REPO_ROOT = Path(__file__).resolve().parents[3]
BYPASS_LABEL = "release-deps: acknowledged"
COMMENT_MARKER = "<!-- dep-freshness-check -->"
PRERELEASE_POLICY_ENV = "DEP_FRESHNESS_PRERELEASE_POLICY"
MAX_FETCH_WORKERS = 8
LOWER_BOUND_OPERATORS = frozenset({">=", "~=", "=="})
UPPER_BOUND_OPERATORS = frozenset({"<", "<=", "~=", "=="})


class PrereleasePolicy(StrEnum):
    """Control when published pre-releases participate in comparisons."""

    BOUND = "bound"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True)
class DependencyDeclaration:
    """A bounded dependency declared by one release package.

    Construct via `from_requirement`, which guarantees `minimum` is the concrete
    lower bound derived from `requirement`.
    """

    manifest_path: str
    package_name: str
    requirement: Requirement
    minimum: Version

    @classmethod
    def from_requirement(
        cls, manifest_path: str, package_name: str, requirement: Requirement
    ) -> Self | None:
        """Build a declaration when the requirement has a concrete minimum.

        Args:
            manifest_path: Repository-relative release manifest path.
            package_name: Name of the package being released.
            requirement: Parsed dependency requirement.

        Returns:
            A declaration, or `None` when no concrete lower bound is declared.

        """
        minimum = extract_minimum(requirement.specifier)
        if minimum is None:
            return None
        return cls(
            manifest_path=manifest_path,
            package_name=package_name,
            requirement=requirement,
            minimum=minimum,
        )

    @property
    def canonical_name(self) -> str:
        """Return the canonicalized distribution name."""
        return canonicalize_name(self.requirement.name)


@dataclass(frozen=True)
class StaleDependency:
    """A dependency whose published version is newer than its declared minimum."""

    manifest_path: str
    package_name: str
    dependency_name: str
    minimum: Version
    latest: Version
    within_upper_bound: bool

    def __post_init__(self) -> None:
        """Reject findings that are not actually stale.

        Raises:
            ValueError: If `latest` does not exceed `minimum`.

        """
        if self.latest <= self.minimum:
            msg = (
                f"StaleDependency requires latest > minimum, "
                f"got {self.latest} <= {self.minimum}"
            )
            raise ValueError(msg)


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _file_warning(path: str, message: str) -> None:
    print(f"::warning file={path}::{message}")


def _error(message: str) -> None:
    print(f"::error::{message}")


def extract_minimum(specifiers: SpecifierSet) -> Version | None:
    """Return the strongest concrete `>=`, `~=`, or `==` lower bound.

    Wildcard equality constraints such as `==1.2.*` do not identify one concrete
    minimum and are ignored. If another supported concrete lower bound is present,
    it can still establish the minimum.

    Args:
        specifiers: Parsed requirement specifiers.

    Returns:
        The greatest concrete lower-bound version, or `None` when none is declared.

    """
    candidates: list[Version] = []
    for specifier in specifiers:
        if specifier.operator not in LOWER_BOUND_OPERATORS:
            continue
        if specifier.operator == "==" and "*" in specifier.version:
            continue
        try:
            candidates.append(Version(specifier.version))
        except InvalidVersion:
            continue
    return max(candidates, default=None)


def includes_prereleases(policy: PrereleasePolicy, minimum: Version) -> bool:
    """Return whether pre-releases should be considered for one dependency.

    Args:
        policy: Configured pre-release comparison policy.
        minimum: Declared concrete minimum version.

    Returns:
        Whether eligible published pre-releases should participate.

    """
    if policy is PrereleasePolicy.ALWAYS:
        return True
    if policy is PrereleasePolicy.NEVER:
        return False
    return minimum.is_prerelease


def within_upper_bound(requirement: Requirement, version: Version) -> bool:
    """Return whether a version satisfies every upper-bound-like specifier.

    Args:
        requirement: Parsed dependency requirement.
        version: Published version being compared.

    Returns:
        `True` when no upper bound exists or the version remains below it.

    """
    upper_bounds = [
        specifier
        for specifier in requirement.specifier
        if specifier.operator in UPPER_BOUND_OPERATORS
    ]
    return all(
        specifier.contains(version, prereleases=True) for specifier in upper_bounds
    )


def _release_is_available(files: object) -> bool:
    if not isinstance(files, list) or not files:
        return False
    valid_files = [item for item in files if isinstance(item, Mapping)]
    return bool(valid_files) and any(
        item.get("yanked") is not True for item in valid_files
    )


def latest_pypi_version(
    payload: Mapping[str, object], *, include_prereleases: bool
) -> Version | None:
    """Select the newest usable PyPI release from a project JSON response.

    Empty releases and releases whose every file is yanked are ignored.

    Args:
        payload: Decoded PyPI project JSON.
        include_prereleases: Whether pre/dev releases are eligible.

    Returns:
        The greatest eligible PEP 440 version, or `None` when none is available.

    Raises:
        TypeError: If the payload's `releases` value is missing or not a mapping.

    """
    releases = payload.get("releases")
    if not isinstance(releases, Mapping):
        msg = "PyPI response has no releases mapping"
        raise TypeError(msg)

    candidates: list[Version] = []
    for raw_version, files in releases.items():
        if not isinstance(raw_version, str) or not _release_is_available(files):
            continue
        try:
            version = Version(raw_version)
        except InvalidVersion:
            continue
        if version.is_prerelease and not include_prereleases:
            continue
        candidates.append(version)
    return max(candidates, default=None)


def _source_is_local(source: object) -> bool:
    if isinstance(source, Mapping):
        local = isinstance(source.get("path"), str) or source.get("workspace") is True
        # A marker-guarded local source only applies on some platforms; the
        # dependency can still resolve from PyPI elsewhere, so keep checking it.
        return local and "marker" not in source
    if isinstance(source, list) and source:
        return all(_source_is_local(item) for item in source)
    return False


def local_dependency_names(manifest: Mapping[str, object]) -> frozenset[str]:
    """Return dependency names backed only by local path/workspace uv sources.

    Marker-guarded sources are excluded: because the marker can be false on some
    platforms, the dependency may still resolve from PyPI and remains in scope.

    Args:
        manifest: Parsed `pyproject.toml` data.

    Returns:
        Canonicalized local distribution names.

    """
    tool = manifest.get("tool")
    if not isinstance(tool, Mapping):
        return frozenset()
    uv = tool.get("uv")
    if not isinstance(uv, Mapping):
        return frozenset()
    sources = uv.get("sources")
    if not isinstance(sources, Mapping):
        return frozenset()
    return frozenset(
        canonicalize_name(name)
        for name, source in sources.items()
        if isinstance(name, str) and _source_is_local(source)
    )


def _declared_requirement_strings(project: Mapping[str, object]) -> list[str]:
    requirements: list[str] = []
    dependencies = project.get("dependencies", [])
    if isinstance(dependencies, list):
        requirements.extend(item for item in dependencies if isinstance(item, str))

    optional = project.get("optional-dependencies", {})
    if isinstance(optional, Mapping):
        for values in optional.values():
            if isinstance(values, list):
                requirements.extend(item for item in values if isinstance(item, str))
    return requirements


def load_declarations(
    manifest_path: str, package_name: str
) -> list[DependencyDeclaration]:
    """Load bounded, non-local runtime dependency declarations from a manifest.

    Args:
        manifest_path: Repository-relative release manifest path.
        package_name: Name of the package being released.

    Returns:
        De-duplicated dependency declarations with concrete minimums.

    Raises:
        TypeError: If the manifest has no valid project table.

    """
    manifest = tomllib.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
    project = manifest.get("project")
    if not isinstance(project, Mapping):
        msg = f"{manifest_path} has no [project] table"
        raise TypeError(msg)

    local_names = local_dependency_names(manifest)
    project_name = project.get("name")
    self_name = (
        canonicalize_name(project_name) if isinstance(project_name, str) else None
    )
    declarations: list[DependencyDeclaration] = []
    seen: set[tuple[str, str]] = set()

    for raw_requirement in _declared_requirement_strings(project):
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement as err:
            _warning(f"Skipping invalid requirement in {manifest_path}: {err}")
            continue

        canonical_name = canonicalize_name(requirement.name)
        if (
            requirement.url
            or canonical_name in local_names
            or canonical_name == self_name
        ):
            continue
        declaration = DependencyDeclaration.from_requirement(
            manifest_path, package_name, requirement
        )
        if declaration is None:
            continue
        key = (canonical_name, str(requirement.specifier))
        if key in seen:
            continue
        seen.add(key)
        declarations.append(declaration)
    return declarations


def _fetch_payloads(
    names: set[str], fetcher: FetchPyPI
) -> tuple[dict[str, Mapping[str, object]], dict[str, PyPIRequestError]]:
    payloads: dict[str, Mapping[str, object]] = {}
    failures: dict[str, PyPIRequestError] = {}
    if not names:
        return payloads, failures

    workers = min(MAX_FETCH_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetcher, name): name for name in sorted(names)}
        for future in as_completed(futures):
            name = futures[future]
            try:
                payloads[name] = future.result()
            except PyPIRequestError as err:
                failures[name] = err
    return payloads, failures


def _policy_description(policy: PrereleasePolicy) -> str:
    if policy is PrereleasePolicy.ALWAYS:
        return "Pre-releases are included for every dependency."
    if policy is PrereleasePolicy.NEVER:
        return "Pre-releases are ignored for every dependency."
    return (
        "Pre-releases are included only when the declared minimum is itself a "
        "pre-release."
    )


def freshness_markdown(
    findings: list[StaleDependency],
    *,
    policy: PrereleasePolicy,
    unavailable: tuple[str, ...] = (),
    include_marker: bool,
) -> str:
    """Render dependency freshness findings as Markdown.

    Args:
        findings: Lagging dependency minimums.
        policy: Pre-release comparison policy used by the check.
        unavailable: Dependencies that could not be queried completely.
        include_marker: Whether to prefix the PR-comment marker.

    Returns:
        A Markdown report suitable for a step summary or PR comment.

    """
    lines: list[str] = []
    if include_marker:
        lines.append(COMMENT_MARKER)
    lines.extend(
        [
            "## Dependency minimums trail PyPI",
            "",
            (
                "The following release-package dependencies have a newer published "
                "version than their declared minimum:"
            ),
            "",
            (
                "| Package | Dependency | Current min | Latest on PyPI | "
                "Within current upper bound? |"
            ),
            "|---|---|---:|---:|:---:|",
        ]
    )
    for finding in sorted(
        findings,
        key=lambda item: (item.package_name, canonicalize_name(item.dependency_name)),
    ):
        within = "Yes" if finding.within_upper_bound else "**No**"
        lines.append(
            f"| `{finding.package_name}` | `{finding.dependency_name}` | "
            f"`{finding.minimum}` | `{finding.latest}` | {within} |"
        )

    lines.extend(
        [
            "",
            _policy_description(policy),
            "Local path/workspace dependencies from `[tool.uv.sources]` are excluded.",
            "",
            (
                "This check is advisory. Bump dependency bounds where appropriate, "
                f"or apply the `{BYPASS_LABEL}` label to record that the release "
                "dependencies were reviewed."
            ),
        ]
    )
    if unavailable:
        names = ", ".join(f"`{name}`" for name in unavailable)
        lines.extend(
            [
                "",
                (
                    "> [!WARNING]\n> PyPI could not be queried completely for: "
                    f"{names}. Re-run the check before relying on this report as "
                    "exhaustive."
                ),
            ]
        )
    return "\n".join(lines)


def check_dependency_freshness(
    base_sha: str,
    head_sha: str,
    *,
    policy: PrereleasePolicy = PrereleasePolicy.BOUND,
    fetcher: FetchPyPI | None = None,
) -> int:
    """Check changed release manifests and emit advisory GitHub Actions outputs.

    Args:
        base_sha: Pull request base commit.
        head_sha: Pull request head commit.
        policy: Pre-release comparison policy.
        fetcher: Injectable PyPI fetcher for tests.

    Returns:
        Zero on normal completion; lagging bounds and known PyPI failures are
        advisory. Unexpected errors (e.g. a malformed manifest or git failure)
        propagate; `main` fail-closes on those with exit code 2.

    """
    packages = load_release_packages()
    manifests = changed_manifests(base_sha, head_sha, list(packages))
    if not manifests:
        _notice("No release-package pyproject.toml files changed; nothing to check.")
        _write_output("stale", "false")
        _write_output("indeterminate", "false")
        _write_output("comment_body", "")
        return 0

    _notice(f"Changed package manifests: {', '.join(manifests)}")
    declarations = [
        declaration
        for manifest_path in manifests
        for declaration in load_declarations(
            manifest_path,
            packages[manifest_path.removesuffix("/pyproject.toml")],
        )
    ]
    canonical_names = {declaration.canonical_name for declaration in declarations}
    payloads, failures = _fetch_payloads(
        canonical_names,
        fetcher or fetch_pypi_json,
    )

    findings: list[StaleDependency] = []
    unavailable = set(failures)
    for name, failure in sorted(failures.items()):
        kind = "transient " if failure.transient else ""
        _warning(f"Skipping {name} after a {kind}PyPI query failure: {failure}")

    for declaration in declarations:
        canonical_name = declaration.canonical_name
        payload = payloads.get(canonical_name)
        if payload is None:
            continue
        try:
            latest = latest_pypi_version(
                payload,
                include_prereleases=includes_prereleases(policy, declaration.minimum),
            )
        except TypeError as err:
            unavailable.add(canonical_name)
            _warning(f"Skipping malformed PyPI data for {canonical_name}: {err}")
            continue
        if latest is None:
            _notice(
                f"No eligible published release found for {canonical_name}; skipping."
            )
            continue
        if latest <= declaration.minimum:
            continue
        finding = StaleDependency(
            manifest_path=declaration.manifest_path,
            package_name=declaration.package_name,
            dependency_name=declaration.requirement.name,
            minimum=declaration.minimum,
            latest=latest,
            within_upper_bound=within_upper_bound(declaration.requirement, latest),
        )
        findings.append(finding)
        _file_warning(
            finding.manifest_path,
            f"{finding.dependency_name} minimum {finding.minimum} trails PyPI "
            f"{finding.latest}",
        )

    unavailable_names = tuple(sorted(unavailable))
    _write_output("stale", "true" if findings else "false")
    _write_output("indeterminate", "true" if unavailable_names else "false")
    if findings:
        summary = freshness_markdown(
            findings,
            policy=policy,
            unavailable=unavailable_names,
            include_marker=False,
        )
        _write_step_summary(summary)
        _write_output(
            "comment_body",
            freshness_markdown(
                findings,
                policy=policy,
                unavailable=unavailable_names,
                include_marker=True,
            ),
        )
    else:
        _write_output("comment_body", "")
    return 0


def main() -> int:
    """Run the dependency freshness check for a pull request diff."""
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")
    if not base_sha or not head_sha:
        _error("BASE_SHA and HEAD_SHA must be set")
        return 2

    raw_policy = os.environ.get(PRERELEASE_POLICY_ENV, PrereleasePolicy.BOUND.value)
    try:
        policy = PrereleasePolicy(raw_policy)
    except ValueError:
        choices = ", ".join(option.value for option in PrereleasePolicy)
        _error(f"{PRERELEASE_POLICY_ENV} must be one of: {choices}")
        return 2

    try:
        return check_dependency_freshness(base_sha, head_sha, policy=policy)
    except Exception as err:  # noqa: BLE001  # fail closed on script defects
        _error(f"Dependency freshness check failed unexpectedly: {err}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
