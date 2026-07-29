"""Resolve release PR runtime dependencies against real PyPI.

A release PR bumps a single package's version. Local development installs the
sibling packages as editable path dependencies via `[tool.uv.sources]`, which
hides whether a package's *published* dependencies actually resolve. This check
strips those local sources and resolves each changed release manifest against
the real index (`uv pip compile --no-sources`), so an unsatisfiable or
not-yet-published runtime dependency fails before merge/publish instead of at
user install time.

This module also owns the shared PyPI JSON client (`fetch_pypi_json`) used by
the dependency-freshness check; the import direction is one-way
(`check_dep_freshness` imports from here) to avoid a cycle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "release-please-config.json"
BYPASS_LABEL = "release-deps: acknowledged"
COMMENT_MARKER = "<!-- release-deps-check -->"
ACKED_ENV = "RELEASE_DEPS_ACKED"
FOLLOWUP_LIMIT = 10
DEFAULT_REQUEST_TIMEOUT = 5.0
DEFAULT_REQUEST_ATTEMPTS = 3
MAX_RETRY_DELAY = 5.0
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 600
TRANSIENT_HTTP_STATUSES = frozenset({408, 425, 429})
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


@dataclass(frozen=True)
class FollowUpConflict:
    """A sibling release package whose published metadata conflicts with a head requirement."""

    manifest_path: str
    """Path to the release package manifest declaring the requirement."""

    package_name: str
    """Published package name of the manifest declaring the requirement."""

    dependency_name: str
    """Sibling release package that must publish a compatible release."""

    requirement: str
    """Requirement specifier declared on this branch."""

    published_constraint: str | None
    """Published `requires_dist` constraint on PyPI that cannot satisfy the requirement."""

    reason: str
    """Why the published package cannot satisfy the head requirement."""


@dataclass
class CheckResult:
    """Outcome of resolving changed release manifests against PyPI."""

    changed: bool
    """Whether any release-package manifest changed on the PR."""

    failures: list[ResolverFailure] = field(default_factory=list)
    """Manifests whose dependencies failed to resolve against PyPI."""

    followups: list[FollowUpConflict] = field(default_factory=list)
    """Sibling release packages needing a compatible follow-up release."""

    unavailable: tuple[str, ...] = ()
    """Dependencies whose PyPI metadata could not be fetched."""

    @property
    def non_transient_failures(self) -> list[ResolverFailure]:
        """Failures that are not transient network or package-index errors."""
        return [failure for failure in self.failures if not failure.transient]


class HTTPResponse(Protocol):
    """Subset of an urllib response used by the PyPI client."""

    def __enter__(self) -> Self:
        """Enter the response context."""

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool | None:
        """Exit the response context."""

    def read(self) -> bytes:
        """Read the response body."""


class OpenUrl(Protocol):
    """A urllib-compatible opener that accepts a per-request timeout."""

    def __call__(self, request: Request, *, timeout: float) -> HTTPResponse:
        """Open a request and return a context-managed response."""


class PyPIRequestError(RuntimeError):
    """A known failure while querying or decoding a PyPI response."""

    def __init__(self, message: str, *, transient: bool) -> None:
        """Initialize a PyPI request failure.

        Args:
            message: Human-readable failure detail.
            transient: Whether retrying later could reasonably succeed.

        """
        super().__init__(message)
        self.transient = transient


class _InvalidPyPIResponseError(ValueError):
    """Raised when PyPI returns a response that cannot be decoded safely."""


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _error(message: str) -> None:
    print(f"::error::{message}")


def _decode_pypi_response(raw: bytes) -> Mapping[str, object]:
    try:
        payload: object = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        msg = f"invalid JSON response: {err}"
        raise _InvalidPyPIResponseError(msg) from err
    if not isinstance(payload, dict) or not isinstance(payload.get("releases"), dict):
        msg = "JSON response does not contain a releases object"
        raise _InvalidPyPIResponseError(msg)
    return payload


def _retry_delay(error: HTTPError | None, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After") if error and error.headers else None
    if retry_after:
        try:
            return min(MAX_RETRY_DELAY, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(MAX_RETRY_DELAY, 0.25 * (2**attempt))


def _is_transient_status(status: int) -> bool:
    return (
        status in TRANSIENT_HTTP_STATUSES
        or SERVER_ERROR_MIN <= status < SERVER_ERROR_MAX
    )


def fetch_pypi_json(
    name: str,
    *,
    attempts: int = DEFAULT_REQUEST_ATTEMPTS,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    opener: OpenUrl | None = None,
    sleep: Any = time.sleep,
) -> Mapping[str, object]:
    """Fetch one project's PyPI JSON with bounded retries.

    Args:
        name: Distribution name, which is canonicalized before querying.
        attempts: Maximum number of HTTP attempts.
        timeout: Timeout in seconds for each attempt.
        opener: Injectable urllib-compatible opener for tests.
        sleep: Injectable delay function for tests.

    Returns:
        Decoded PyPI JSON response.

    Raises:
        ValueError: If `attempts` is less than one.
        PyPIRequestError: If PyPI cannot provide a usable response.

    """
    if attempts < 1:
        msg = "attempts must be at least 1"
        raise ValueError(msg)

    canonical_name = canonicalize_name(name)
    url = f"https://pypi.org/pypi/{quote(canonical_name, safe='')}/json"
    request = Request(  # URL is fixed to the HTTPS PyPI origin.
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "langchain-ai/deepagents-release-deps-check",
        },
    )
    open_url = opener or urlopen

    for attempt in range(attempts):
        try:
            with open_url(request, timeout=timeout) as response:
                return _decode_pypi_response(response.read())
        except HTTPError as err:
            transient = _is_transient_status(err.code)
            if not transient or attempt == attempts - 1:
                msg = f"PyPI returned HTTP {err.code} for {canonical_name}"
                raise PyPIRequestError(msg, transient=transient) from err
            sleep(_retry_delay(err, attempt))
        except (
            URLError,
            TimeoutError,
            OSError,
            HTTPException,
            _InvalidPyPIResponseError,
        ) as err:
            if attempt == attempts - 1:
                msg = f"PyPI request failed for {canonical_name}: {err}"
                raise PyPIRequestError(msg, transient=True) from err
            sleep(_retry_delay(None, attempt))

    msg = f"PyPI request exhausted retries for {canonical_name}"
    raise PyPIRequestError(msg, transient=True)


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


def _self_and_local_names(data: dict[str, Any]) -> set[str]:
    """Return canonicalized names of the package itself and its local path sources."""
    names: set[str] = set()
    project = data.get("project", {})
    if isinstance(project, dict):
        name = project.get("name")
        if isinstance(name, str):
            names.add(canonicalize_name(name))
    tool = data.get("tool", {})
    uv = tool.get("uv", {}) if isinstance(tool, dict) else {}
    sources = uv.get("sources", {}) if isinstance(uv, dict) else {}
    if isinstance(sources, dict):
        for name, source in sources.items():
            if not isinstance(name, str):
                continue
            if isinstance(source, dict) and (
                isinstance(source.get("path"), str) or source.get("workspace") is True
            ):
                names.add(canonicalize_name(name))
    return names


def _manifest_requirements(data: dict[str, Any]) -> list[Requirement]:
    """Parse all runtime and optional dependency requirements from a manifest."""
    project = data.get("project", {})
    if not isinstance(project, dict):
        return []
    specs: list[str] = []
    dependencies = project.get("dependencies", [])
    if isinstance(dependencies, list):
        specs.extend(spec for spec in dependencies if isinstance(spec, str))
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for values in optional.values():
            if isinstance(values, list):
                specs.extend(spec for spec in values if isinstance(spec, str))
    requirements = []
    for spec in specs:
        try:
            requirements.append(Requirement(spec))
        except InvalidRequirement:
            continue
    return requirements


def _sibling_requirements(
    manifests: list[str],
    sibling_names: set[str],
) -> dict[str, list[tuple[str, str, Requirement]]]:
    """Collect head requirements on sibling release packages, keyed by sibling name.

    Skips requirements on the declaring package itself and on local path sources:
    those never need a published sibling release.
    """
    collected: dict[str, list[tuple[str, str, Requirement]]] = {}
    for manifest_path in manifests:
        data = tomllib.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
        project = data.get("project", {})
        package_name = (
            project.get("name")
            if isinstance(project, dict) and isinstance(project.get("name"), str)
            else manifest_path.removesuffix("/pyproject.toml")
        )
        skip = _self_and_local_names(data)
        for requirement in _manifest_requirements(data):
            canonical_name = canonicalize_name(requirement.name)
            if canonical_name in skip or canonical_name not in sibling_names:
                continue
            collected.setdefault(canonical_name, []).append(
                (manifest_path, package_name, requirement)
            )
    return collected


def _parse_requires_dist(requires_dist: Any) -> list[Requirement]:
    """Parse a `requires_dist` list, skipping entries that are not valid requirements."""
    requirements = []
    for spec in requires_dist if isinstance(requires_dist, list) else []:
        if not isinstance(spec, str):
            continue
        try:
            requirements.append(Requirement(spec))
        except InvalidRequirement:
            continue
    return requirements


def _latest_published_metadata(
    payload: Any, package_name: str
) -> tuple[str, list[Requirement]] | None:
    """Return the newest published version and its parsed `requires_dist`.

    Prefers the `info` block of the PyPI JSON response; falls back to the newest
    release carrying its own metadata (older PyPI uploads sometimes lack it).
    """
    info = payload.get("info") if isinstance(payload, dict) else None
    if isinstance(info, dict):
        version = info.get("version")
        if isinstance(version, str):
            return version, _parse_requires_dist(info.get("requires_dist"))

    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, dict):
        return None

    candidates = []
    for raw_version, files in releases.items():
        if not isinstance(files, list) or not files:
            continue
        first = files[0]
        if not isinstance(first, dict) or not isinstance(first.get("requires_dist"), str):
            continue
        try:
            candidates.append((Version(raw_version), first["requires_dist"]))
        except InvalidVersion:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    version, requires_dist = candidates[-1]
    return str(version), _parse_requires_dist([requires_dist])


def _published_constraint(
    requires_dist: list[Requirement], canonical_name: str
) -> Requirement | None:
    """Find the published requirement on `canonical_name` in a `requires_dist` set."""
    for requirement in requires_dist:
        if canonicalize_name(requirement.name) == canonical_name:
            return requirement
    return None


def _probe_versions(specifier: SpecifierSet) -> list[Version]:
    """Build probe versions that sample the edges of a specifier range.

    The returned versions cover each bound itself plus one step above or below
    it, so a probe lands on both sides of every edge. Wildcard and `!=`
    specifiers are skipped — they cannot pin a concrete edge. Zero bounds
    (e.g. `<0.1` with nothing below) contribute only the bound itself.
    """
    probes: list[Version] = []
    for spec in specifier:
        if spec.operator == "!=" or "*" in spec.version:
            continue
        try:
            bound = Version(spec.version)
        except InvalidVersion:
            continue
        probes.append(bound)
        release = list(bound.release)
        if spec.operator in (">=", ">", "~=", "=="):
            above = release.copy()
            above[-1] += 1
            probes.append(Version(".".join(str(part) for part in above)))
        if spec.operator in ("<", "<=", "~=") and release[-1] > 0:
            below = release.copy()
            below[-1] -= 1
            probes.append(Version(".".join(str(part) for part in below)))
    return probes


def _ranges_may_overlap(head: SpecifierSet, published: SpecifierSet) -> bool:
    """Return whether two specifier ranges could share a version.

    Uses probe versions sampled from both range edges. Conservative: it may
    report overlap for adjacent open ranges that technically do not intersect,
    which only costs an extra follow-up row, never a missed conflict.
    """
    if not str(head) or not str(published):
        return True
    for probe in _probe_versions(head) + _probe_versions(published):
        if head.contains(probe, prereleases=True) and published.contains(
            probe, prereleases=True
        ):
            return True
    return False


def _compare_with_published(
    manifest_path: str,
    package_name: str,
    requirement: Requirement,
    canonical_name: str,
    payloads: dict[str, Any],
) -> FollowUpConflict | None:
    """Compare one head requirement against the sibling's published PyPI metadata."""
    payload = payloads.get(canonical_name)
    if payload is None:
        return None
    latest = _latest_published_metadata(payload, canonical_name)
    if latest is None:
        return None
    published_version, requires_dist = latest
    published = _published_constraint(requires_dist, canonical_name)
    if published is None:
        return FollowUpConflict(
            manifest_path=manifest_path,
            package_name=package_name,
            dependency_name=requirement.name,
            requirement=str(requirement.specifier) or "(any)",
            published_constraint=None,
            reason=f"published {published_version} does not declare this dependency",
        )
    if not published.specifier:
        return None
    if _ranges_may_overlap(requirement.specifier, published.specifier):
        return None
    return FollowUpConflict(
        manifest_path=manifest_path,
        package_name=package_name,
        dependency_name=requirement.name,
        requirement=str(requirement.specifier) or "(any)",
        published_constraint=str(published.specifier),
        reason=(
            f"published {published_version} constrains it to "
            f"`{published.specifier}`"
        ),
    )


def find_follow_up_conflicts(
    manifests: list[str],
    package_paths: list[str],
    *,
    fetcher: Any = None,
) -> tuple[list[FollowUpConflict], tuple[str, ...]]:
    """Identify sibling release packages that must publish a compatible release.

    Compares each head requirement that targets another release-please package
    against that package's latest published `requires_dist` on PyPI. A conflict
    means the published sibling cannot satisfy the head requirement, so the
    sibling needs its own release before the public install graph is healthy.

    Args:
        manifests: Changed release-package manifest paths on this PR.
        package_paths: All release-please package paths.
        fetcher: Injectable PyPI JSON fetcher for tests.

    Returns:
        A (conflicts, unavailable) pair: the conflicting sibling dependencies,
        and the dependency names whose PyPI metadata could not be fetched.
    """
    package_names = {
        canonicalize_name(name)
        for name in load_release_packages().values()
        if isinstance(name, str)
    }
    requirements = _sibling_requirements(manifests, package_names)
    if not requirements:
        return [], ()

    payloads: dict[str, Any] = {}
    unavailable: list[str] = []
    fetch = fetcher or fetch_pypi_json
    for canonical_name in sorted(requirements):
        try:
            payloads[canonical_name] = fetch(canonical_name)
        except PyPIRequestError as err:
            _warning(f"Skipping follow-up analysis for {canonical_name}: {err}")
            unavailable.append(canonical_name)

    conflicts: list[FollowUpConflict] = []
    for canonical_name in sorted(requirements):
        if canonical_name not in payloads:
            continue
        for manifest_path, package_name, requirement in requirements[canonical_name]:
            conflict = _compare_with_published(
                manifest_path,
                package_name,
                requirement,
                canonical_name,
                payloads,
            )
            if conflict is not None:
                conflicts.append(conflict)
    return conflicts, tuple(sorted(unavailable))


def _follow_up_markdown(
    followups: list[FollowUpConflict],
    *,
    unavailable: tuple[str, ...] = (),
    acked: bool,
) -> list[str]:
    """Render the "follow-up releases" section for the PR comment/step summary."""
    if not followups and not unavailable:
        return []

    lines: list[str] = []
    if followups:
        by_manifest: dict[str, list[FollowUpConflict]] = {}
        for conflict in followups:
            by_manifest.setdefault(conflict.manifest_path, []).append(conflict)

        if acked:
            lines.extend(
                [
                    "### Follow-up releases needed after merge",
                    "",
                    f"The `{BYPASS_LABEL}` label is set, so the resolver ran in report-only mode and the check passed. The public install graph still depends on these releases landing after this one:",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "### Follow-up releases needed",
                    "",
                    "These sibling packages still constrain the dependency on PyPI; each needs its own release before the public install graph is healthy:",
                    "",
                ]
            )

        if len(by_manifest) > 1:
            # The resolver stops at the first unsatisfiable manifest, so list the
            # full class of conflicts rather than only the one that failed first.
            shown = followups[:FOLLOWUP_LIMIT]
            lines.append(
                f"_Conflicts span {len(by_manifest)} manifests; the resolver reports only the first unsatisfiable one._"
            )
            lines.append("")
        else:
            shown = followups

        lines.extend(
            [
                "| Declaring package | Dependency | Head requires | Published on PyPI |",
                "|---|---|---|---|",
            ]
        )
        for conflict in shown:
            published = (
                f"`{conflict.published_constraint}`"
                if conflict.published_constraint is not None
                else "(not declared)"
            )
            lines.append(
                f"| `{conflict.package_name}` | `{conflict.dependency_name}` | "
                f"`{conflict.requirement}` | {published} — {conflict.reason} |"
            )
        if len(followups) > len(shown):
            lines.append(
                f"| … | … | … | {len(followups) - len(shown)} more conflicts |"
            )
        lines.append("")

    if unavailable:
        names = ", ".join(f"`{name}`" for name in unavailable)
        lines.extend(
            [
                "> [!WARNING]",
                f"> PyPI metadata could not be fetched for: {names}. Re-run the job before relying on the follow-up list as exhaustive.",
                "",
            ]
        )
    return lines


def _failure_sections(failures: list[ResolverFailure]) -> list[str]:
    """Render per-manifest failure sections without a title or marker."""
    lines: list[str] = []
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
    return lines


def _failure_markdown(
    failures: list[ResolverFailure],
    *,
    include_marker: bool,
    followups: list[FollowUpConflict] | None = None,
    unavailable: tuple[str, ...] = (),
) -> str:
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

    follow_up_lines = _follow_up_markdown(
        followups or [], unavailable=unavailable, acked=False
    )
    if follow_up_lines:
        lines.extend(follow_up_lines)

    lines.extend(_failure_sections(failures))

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


def check_release_dependencies(
    base_sha: str,
    head_sha: str,
    *,
    fetcher: Any = None,
) -> CheckResult:
    """Resolve changed release-package manifests against PyPI.

    Args:
        base_sha: Pull request base commit.
        head_sha: Pull request head commit.
        fetcher: Injectable PyPI JSON fetcher for tests.

    Returns:
        The resolution outcome: changed flag, resolver failures, sibling
        follow-up conflicts, and any dependencies whose PyPI metadata could
        not be fetched.
    """
    packages = load_release_packages()
    package_paths = list(packages)
    manifests = changed_manifests(base_sha, head_sha, package_paths)
    if not manifests:
        _notice("No release-package pyproject.toml files changed; nothing to check.")
        return CheckResult(changed=False)

    _notice(f"Changed package manifests: {', '.join(manifests)}")

    result = CheckResult(changed=True)
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
                result.failures.append(
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

    # Run follow-up analysis whenever manifests changed — including on clean
    # resolves under the bypass label, where a soft-run still needs the
    # follow-up list to stay authoritative.
    result.followups, result.unavailable = find_follow_up_conflicts(
        manifests, package_paths, fetcher=fetcher
    )
    return result


def _comment_body(result: CheckResult, *, acked: bool) -> str:
    """Build the PR comment body for a completed (possibly soft) check run."""
    non_transient = result.non_transient_failures

    if not acked:
        return _failure_markdown(
            non_transient,
            include_marker=True,
            followups=result.followups,
            unavailable=result.unavailable,
        )

    lines: list[str] = [
        COMMENT_MARKER,
        "## Release dependency check (acknowledged bypass)",
        "",
        f"The `{BYPASS_LABEL}` label is set: resolution ran in report-only mode, so the check passes regardless of the outcome below.",
        f"Use `{BYPASS_LABEL}` only for coordinated release order — publish the listed follow-up packages right after this release. Do not use it for accidental unsatisfiable dependency ranges; fix the dependency metadata instead.",
        "",
    ]

    follow_up_lines = _follow_up_markdown(
        result.followups, unavailable=result.unavailable, acked=True
    )
    lines.extend(follow_up_lines)

    if non_transient:
        lines.extend(
            [
                "### Resolution failures (reported, not blocking)",
                "",
                "These manifests still fail to resolve against published PyPI packages:",
                "",
            ]
        )
        lines.extend(_failure_sections(non_transient))
    elif result.failures:
        lines.extend(
            [
                "Resolution hit only transient network or package-index errors; no metadata problems were detected.",
                "",
            ]
        )
    elif not result.followups and not result.unavailable:
        lines.extend(
            [
                "Resolution succeeded against PyPI, and no sibling follow-up releases are required.",
                "",
            ]
        )

    return "\n".join(lines).rstrip()


def run_check(
    base_sha: str,
    head_sha: str,
    *,
    acked: bool,
    fetcher: Any = None,
) -> int:
    """Run the release dependency check and emit GitHub Actions outputs.

    Args:
        base_sha: Pull request base commit.
        head_sha: Pull request head commit.
        acked: Whether the `release-deps: acknowledged` bypass label is set.
        fetcher: Injectable PyPI JSON fetcher for tests.

    Returns:
        Process exit code: 0 on success or a soft (acknowledged) run, 1 on an
        unacknowledged resolution failure, 2 on an unexpected error.
    """
    result = check_release_dependencies(base_sha, head_sha, fetcher=fetcher)

    if not result.changed:
        _write_output("failed", "false")
        _write_output("comment_body", "")
        return 0

    non_transient = result.non_transient_failures

    if acked:
        _notice(
            f"`{BYPASS_LABEL}` label is set; running resolution in report-only mode."
        )
        body = _comment_body(result, acked=True)
        # The step summary duplicates the comment body but drops the HTML
        # marker so it reads as a standalone report.
        _write_step_summary(body.removeprefix(COMMENT_MARKER).lstrip())
        _write_output("failed", "false")
        _write_output("comment_body", body)
        return 0

    if not non_transient:
        if result.failures:
            # Purely transient failures fail closed with no comment, so an
            # existing warning comment stays until the job is re-run.
            _write_output("failed", "true")
            _write_output("comment_body", "")
            return 1
        _write_output("failed", "false")
        _write_output("comment_body", "")
        return 0

    body = _comment_body(result, acked=False)
    _write_step_summary(body.removeprefix(COMMENT_MARKER).lstrip())
    _write_output("failed", "true")
    _write_output("comment_body", body)
    return 1


def main() -> int:
    """CLI entry point used by the GitHub Actions workflow."""
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")
    if not base_sha or not head_sha:
        _error("BASE_SHA and HEAD_SHA must be set")
        return 2
    acked = os.environ.get(ACKED_ENV, "").strip().lower() in {"1", "true", "yes"}
    try:
        return run_check(base_sha, head_sha, acked=acked)
    except Exception as err:  # noqa: BLE001  # fail closed with a clear CI annotation
        _error(f"Release dependency check failed unexpectedly: {err}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
