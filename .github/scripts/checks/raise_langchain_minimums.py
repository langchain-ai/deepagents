"""Raise LangChain-ecosystem dependency lower bounds to the latest stable PyPI release.

For every in-scope requirement (see `IN_SCOPE_PREFIXES`) that declares a
concrete lower bound (`>=` or `~=`) in `[project.dependencies]`,
`[project.optional-dependencies]`, or `[dependency-groups]`, this rewrites that
lower bound in place to the newest stable PyPI release that still has at least
one non-yanked file *and* remains within the requirement's existing range.
Upper bounds, extras, and environment markers are preserved. Exact pins (`==`)
are left alone — raising a floor only applies to range floors. A bound already
ahead of the latest stable release (intentional prerelease coordination) is
respected and never lowered.

A dependency whose PyPI metadata could not be fetched, or a manifest that could
not be rewritten, is reported as a failure rather than silently omitted: an
unattended weekly cron must never render "PyPI was unreachable" as "everything
is already up to date".
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import Specifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[3]

# Sibling helper scripts: `check_dep_freshness` and `check_release_deps` live
# under `release/`; `check_lockfiles_pre_commit` owns the package list and the
# per-package interpreter version used to resolve lockfiles. Import them rather
# than duplicating either, so a lockfile regenerated here always matches what
# `check_lockfiles.yml` will verify on the opened PR.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
for _domain in (_SCRIPTS_DIR / "release", _SCRIPTS_DIR / "checks"):
    if str(_domain) not in sys.path:
        sys.path.insert(0, str(_domain))

from check_dep_freshness import (  # noqa: E402
    available_pypi_versions,
    extract_minimum,
    local_dependency_names,
)
from check_lockfiles_pre_commit import package_dirs, python_version  # noqa: E402
from check_release_deps import (  # noqa: E402
    PyPIRequestError,
    _write_output,
    _write_step_summary,
    fetch_pypi_json,
    load_release_packages,
)

MAX_FETCH_WORKERS = 8
# Only `>=` / `~=` floors are raiseable. `==` is an exact pin (the only in-scope
# one today, `deepagents==` in `libs/code`, is bumped by `bump_code_sdk_pin.yml`
# after a `deepagents` publish), and a bare upper bound has no floor to raise.
RAISEABLE_OPERATORS = frozenset({">=", "~="})
# Distribution-name prefixes whose floors are raised. Covers the LangChain
# integrations (langchain-*, the base langchain/langgraph/langsmith packages,
# and langgraph-* companion packages) plus the deepagents-* workspace packages
# that one package pulls from PyPI (e.g. deepagents-code -> deepagents-acp).
# Workspace-local sources and a package's own name are excluded per manifest.
# Matching is a bare name prefix with no hyphen boundary, so an unrelated
# third-party distribution such as `langchainhub` is also in scope.
IN_SCOPE_PREFIXES = ("langchain", "langgraph", "langsmith", "deepagents")


def _notice(message: str) -> None:
    print(f"::notice::{message}")


def _warning(message: str) -> None:
    print(f"::warning::{message}")


def _error(message: str) -> None:
    print(f"::error::{message}")


@dataclass(frozen=True)
class RequirementEdit:
    """One lower-bound rewrite applied to a manifest requirement string.

    Attributes:
        manifest_path: Repository-relative, POSIX-separated manifest that was
            edited.
        dependency_name: Canonicalized in-scope distribution name.
        old_requirement: Original requirement string as written in the manifest.
        new_requirement: Requirement string with the raised lower bound.

    """

    manifest_path: str
    dependency_name: str
    old_requirement: str
    new_requirement: str


@dataclass(frozen=True)
class ManifestScope:
    """One manifest's text and its in-scope requirements, parsed exactly once.

    Parsing a manifest twice (once to collect names to fetch, once to rewrite)
    risks the two passes disagreeing about what is in scope, which shows up only
    as a silently missing bump. This carries the single parse between phases.

    Attributes:
        manifest_path: Repository-relative, POSIX-separated manifest path.
        text: The manifest's verbatim source text.
        requirements: `(raw_string, parsed_requirement)` pairs for every
            in-scope dependency, in manifest declaration order.

    """

    manifest_path: str
    text: str
    requirements: tuple[tuple[str, Requirement], ...]


@dataclass(frozen=True)
class ManifestPlan:
    """A pending rewrite of one manifest, not yet written to disk.

    Rewrites are planned for every manifest before any is written so a failure
    partway through cannot leave a half-updated working tree.

    Attributes:
        manifest_path: Repository-relative, POSIX-separated manifest path.
        new_text: Full manifest text with every raised bound applied.
        edits: The rewrites `new_text` embodies.

    """

    manifest_path: str
    new_text: str
    edits: tuple[RequirementEdit, ...]


def _project_requirement_strings(project: Mapping[str, object]) -> list[str]:
    """Collect requirement strings from `[project]`'s dependency tables.

    Reads `dependencies` and `optional-dependencies`. `[dependency-groups]` is a
    top-level table, not a `[project]` one — see `_group_requirement_strings`.
    """
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


def _group_requirement_strings(manifest: Mapping[str, object]) -> list[str]:
    """Collect requirement strings from `[dependency-groups]`.

    PEP 735 include-tables (`{include-group = "..."}`) are not requirement
    strings and are skipped; the group they name is visited on its own.
    """
    groups = manifest.get("dependency-groups", {})
    if not isinstance(groups, Mapping):
        return []
    requirements: list[str] = []
    for values in groups.values():
        if isinstance(values, list):
            requirements.extend(item for item in values if isinstance(item, str))
    return requirements


def _raiseable_specifier(specifiers: SpecifierSet) -> Specifier | None:
    """Return the strongest `>=`/`~=` lower-bound clause, or `None` if there is none.

    Selected by version rather than iteration order. `SpecifierSet.__iter__`
    order is an implementation detail — older `packaging` releases stored the
    clauses in a `frozenset` — and `extract_minimum`, which gates the comparison
    at the call site, reports the *strongest* floor. Choosing by `max` keeps the
    clause that gets rewritten and the floor that was compared in agreement.
    """
    candidates = [
        specifier
        for specifier in specifiers
        if specifier.operator in RAISEABLE_OPERATORS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda specifier: Version(specifier.version))


def _compatible_release_version(old_version: str, new_minimum: Version) -> str:
    """Render `new_minimum` for a `~=` clause without moving its ceiling.

    `~=X1...Xn` means `>=X1...Xn, ==X1...X(n-1).*`, so the implied ceiling is
    fixed by the *number* of release components, not their values. Rendering the
    raised floor with the same component count therefore leaves the ceiling
    untouched: `~=1.2` (ceiling `==1.*`) raised towards 1.9.4 becomes `~=1.9`,
    not `~=1.9.4`, which would silently narrow the ceiling to `==1.9.*`.

    `new_minimum` always satisfies the original clause (callers clamp candidates
    to the existing range), so its leading components already match and only the
    last one moves.
    """
    components = len(Version(old_version).release)
    if len(new_minimum.release) < components:
        return str(new_minimum)
    return ".".join(str(part) for part in new_minimum.release[:components])


def _raise_lower_bound(
    requirement_string: str, specifier: Specifier, new_minimum: Version
) -> str:
    """Rewrite one requirement string's lower-bound clause to `new_minimum`.

    Only the matched clause is replaced, so upper bounds, extras, and markers
    survive verbatim.

    Args:
        requirement_string: Raw requirement string as written in the manifest.
        specifier: The clause to rewrite, from `_raiseable_specifier`.
        new_minimum: The version to raise the lower bound to.

    Returns:
        The rewritten requirement string.

    Raises:
        ValueError: If the clause could not be located in `requirement_string`.

    """
    new_version = (
        _compatible_release_version(specifier.version, new_minimum)
        if specifier.operator == "~="
        else str(new_minimum)
    )
    # `packaging` discards whitespace when parsing, so a clause spelled
    # `>= 1.0` in the manifest renders as `>=1.0` once parsed. Match the
    # operator and version with optional whitespace between them instead of
    # assuming the normalized text appears verbatim — `langchain >= 1.0` is
    # legal PEP 508 and must not blow up the run.
    pattern = re.escape(specifier.operator) + r"\s*" + re.escape(specifier.version)
    replaced, count = re.subn(
        pattern, f"{specifier.operator}{new_version}", requirement_string, count=1
    )
    if count != 1:
        msg = (
            f"Could not rewrite lower bound "
            f"'{specifier.operator}{specifier.version}' in '{requirement_string}'"
        )
        raise ValueError(msg)
    return replaced


def _preserves_upper_bounds(
    old: SpecifierSet,
    new: SpecifierSet,
    new_minimum: Version,
    versions: Iterable[Version],
) -> bool:
    """Whether `new` still admits everything `old` did at or above the new floor.

    Raising a floor must never tighten a ceiling. Rather than reasoning about
    each operator's implied upper bound, check the property directly against the
    versions that actually exist on PyPI — the only ones the change can affect.
    """
    return all(
        new.contains(version, prereleases=True)
        for version in versions
        if version >= new_minimum and old.contains(version, prereleases=True)
    )


def _self_name(project: Mapping[str, object]) -> str | None:
    """Return the canonicalized name of the package the manifest publishes."""
    name = project.get("name")
    return canonicalize_name(name) if isinstance(name, str) else None


def _in_scope(
    requirement: Requirement,
    canonical_name: str,
    local_names: frozenset[str],
    self_name: str | None,
) -> bool:
    """Return whether a dependency is an in-scope, externally-resolved dependency.

    Whether it declares a raiseable floor is a separate question, answered by
    `extract_minimum` and `_raiseable_specifier` at the call site.
    """
    if requirement.url or canonical_name in local_names or canonical_name == self_name:
        return False
    return canonical_name.startswith(IN_SCOPE_PREFIXES)


def _load_scope(manifest_path: str) -> ManifestScope:
    """Read and parse one manifest, returning its in-scope requirements.

    Raises:
        TypeError: If the manifest has no `[project]` table.

    """
    text = (REPO_ROOT / manifest_path).read_text(encoding="utf-8")
    manifest = tomllib.loads(text)
    project = manifest.get("project")
    if not isinstance(project, Mapping):
        msg = f"{manifest_path} has no [project] table"
        raise TypeError(msg)

    local_names = local_dependency_names(manifest)
    self_name = _self_name(project)
    requirements: list[tuple[str, Requirement]] = []
    seen: set[str] = set()
    for raw in _project_requirement_strings(project) + _group_requirement_strings(
        manifest
    ):
        if raw in seen:
            continue
        seen.add(raw)
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as err:
            _warning(
                f"Skipping unparseable requirement in {manifest_path}: {raw!r} ({err})"
            )
            continue
        canonical_name = canonicalize_name(requirement.name)
        if not _in_scope(requirement, canonical_name, local_names, self_name):
            continue
        requirements.append((raw, requirement))
    return ManifestScope(
        manifest_path=manifest_path, text=text, requirements=tuple(requirements)
    )


def _fetch_available_versions(
    names: Collection[str],
) -> tuple[dict[str, list[Version]], list[str]]:
    """Fetch every usable stable PyPI version for each distribution name.

    Returns:
        A `(available, failures)` pair. `failures` names the distributions whose
        metadata could not be retrieved or understood, so callers can report
        them instead of mistaking an absent entry for "already up to date".

    """
    available: dict[str, list[Version]] = {}
    failures: list[str] = []
    if not names:
        return available, failures
    workers = min(MAX_FETCH_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_pypi_json, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                # `available_pypi_versions` raises `TypeError` on a payload with
                # no `releases` mapping; keep it inside the guard so one
                # malformed response cannot abort every other lookup.
                versions = available_pypi_versions(
                    future.result(), include_prereleases=False
                )
            except (PyPIRequestError, TypeError) as err:
                _warning(f"Could not resolve {name}: PyPI query failed ({err})")
                failures.append(name)
                continue
            if not versions:
                _warning(f"Could not resolve {name}: no stable PyPI release found")
                failures.append(name)
                continue
            available[name] = versions
    return available, sorted(failures)


def _latest_compatible_version(
    requirement: Requirement, versions: Iterable[Version]
) -> Version | None:
    """Return the newest available version satisfying the requirement range."""
    return max(
        (
            version
            for version in versions
            if requirement.specifier.contains(version, prereleases=True)
        ),
        default=None,
    )


def _apply_replacements(
    text: str, replacements: Mapping[str, str], manifest_path: str
) -> str:
    """Rewrite requirement string literals in a manifest's TOML text.

    Each requirement is matched only where it fills an entire quoted TOML
    string. That anchoring is what makes the rewrite safe: a requirement that is
    a prefix of another (`langchain-core>=1.1` inside `langchain-core>=1.1.1`)
    cannot corrupt its neighbour, and a mention inside a `#` comment is left
    alone. Longest requirement first, as belt-and-braces against overlap.

    A requirement that also appears in a table this script does not parse
    (`[tool.uv] constraint-dependencies`, say) is rewritten there too, keeping
    the two spellings consistent.

    Raises:
        ValueError: If a planned replacement matched nothing, which would
            otherwise be reported as an applied edit that never happened.

    """
    for raw in sorted(replacements, key=len, reverse=True):
        new = replacements[raw]
        applied = 0
        for quote in ('"', "'"):
            literal = f"{quote}{raw}{quote}"
            occurrences = text.count(literal)
            if occurrences:
                text = text.replace(literal, f"{quote}{new}{quote}")
                applied += occurrences
        if not applied:
            msg = (
                f"{manifest_path}: requirement {raw!r} was parsed from this "
                "manifest but no matching quoted literal could be rewritten"
            )
            raise ValueError(msg)
    return text


def _plan_manifest(
    scope: ManifestScope, available: Mapping[str, list[Version]]
) -> ManifestPlan:
    """Plan the in-scope lower-bound rewrites for one manifest.

    Raises:
        ValueError: If a rewrite could not be expressed or applied.

    """
    edits: list[RequirementEdit] = []
    replacements: dict[str, str] = {}
    for raw, requirement in scope.requirements:
        canonical_name = canonicalize_name(requirement.name)
        minimum = extract_minimum(requirement.specifier)
        if minimum is None:
            continue
        specifier = _raiseable_specifier(requirement.specifier)
        if specifier is None:
            continue
        versions = available.get(canonical_name)
        if not versions:
            continue
        new_minimum = _latest_compatible_version(requirement, versions)
        if new_minimum is None or new_minimum <= minimum:
            continue

        raised = _raise_lower_bound(raw, specifier, new_minimum)
        if raised == raw:
            continue
        if not _preserves_upper_bounds(
            requirement.specifier,
            Requirement(raised).specifier,
            new_minimum,
            versions,
        ):
            _warning(
                f"{scope.manifest_path}: leaving {raw!r} alone — raising it to "
                f"{new_minimum} would exclude versions the current range allows."
            )
            continue

        replacements[raw] = raised
        edits.append(
            RequirementEdit(
                manifest_path=scope.manifest_path,
                dependency_name=canonical_name,
                old_requirement=raw,
                new_requirement=raised,
            )
        )

    new_text = (
        _apply_replacements(scope.text, replacements, scope.manifest_path)
        if replacements
        else scope.text
    )
    return ManifestPlan(
        manifest_path=scope.manifest_path, new_text=new_text, edits=tuple(edits)
    )


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
    """Return every package dir whose `uv.lock` the changed manifests invalidate.

    A lockfile embeds the requirement specifiers of each package it resolves
    from a local `[tool.uv.sources]` path, so raising a floor in
    `libs/deepagents` staleness-marks the lockfile of every package that path-
    depends on it — `libs/evals/uv.lock` and friends — not only its own. This
    closes over those path edges transitively. `check_lockfiles.yml` inspects
    only the packages a PR diff touches, so a lockfile omitted here would go
    stale on `main` and fail later for an unrelated contributor.
    """
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


def edits_markdown(edits: Sequence[RequirementEdit], *, heading: str) -> str:
    """Render applied edits as a Markdown table for the PR body/summary."""
    lines = [heading, "", "| Manifest | Dependency | Change |", "|---|---|---|"]
    lines.extend(
        f"| `{edit.manifest_path}` | `{edit.dependency_name}` | "
        f"`{edit.old_requirement}` → `{edit.new_requirement}` |"
        for edit in edits
    )
    return "\n".join(lines)


def failures_markdown(dependencies: Sequence[str], manifests: Sequence[str]) -> str:
    """Render unresolved dependencies and manifests so a partial run looks partial."""
    lines: list[str] = ["### Not raised", ""]
    if dependencies:
        lines.append(
            "PyPI metadata could not be resolved for these dependencies, so their "
            "floors were left untouched: "
            + ", ".join(f"`{name}`" for name in dependencies)
            + "."
        )
        lines.append("")
    if manifests:
        lines.append(
            "These manifests could not be rewritten: "
            + ", ".join(f"`{path}`" for path in manifests)
            + "."
        )
        lines.append("")
    lines.append("Re-run the workflow once the cause is resolved.")
    return "\n".join(lines)


def _select_manifests(package: str, packages: Mapping[str, str]) -> list[str] | None:
    """Resolve a `--package` value to the manifests it selects, or `None` if unknown."""
    if package == "all":
        selected = sorted(packages)
    else:
        matched = [path for path, label in packages.items() if label == package]
        if not matched and package in packages:
            matched = [package]
        if not matched:
            labels = ", ".join(sorted(set(packages.values()))) or "none"
            _error(
                f"Unknown package '{package}'. Expected a release label "
                f"({labels}), a release path, or 'all'."
            )
            return None
        selected = sorted(matched)
    return [f"{path}/pyproject.toml" for path in selected]


def _run(package: str) -> int:
    """Raise in-scope lower bounds for `package`, writing manifests and outputs."""
    manifests = _select_manifests(package, load_release_packages())
    if manifests is None:
        return 1
    _notice(f"Raising dependency minimums for {package}: " + ", ".join(manifests))

    scopes = [_load_scope(manifest_path) for manifest_path in manifests]
    in_scope_names = {
        canonicalize_name(requirement.name)
        for scope in scopes
        for _, requirement in scope.requirements
    }
    if not in_scope_names:
        # Every release package declares in-scope dependencies, so an empty set
        # means the manifests are shaped differently than this script expects —
        # a defect to surface, not a quiet no-op.
        _error(
            f"No in-scope requirements found for {package}. Every release "
            "package declares some, so this most likely means the manifest "
            "layout changed and this script needs updating."
        )
        return 1

    available, fetch_failures = _fetch_available_versions(in_scope_names)

    plans: list[ManifestPlan] = []
    plan_failures: list[str] = []
    for scope in scopes:
        try:
            plan = _plan_manifest(scope, available)
        except (ValueError, InvalidRequirement) as err:
            # Isolate per manifest: a defect in one package's requirements must
            # not discard the valid bumps found for the other nine.
            _error(f"Could not rewrite {scope.manifest_path}: {err}")
            plan_failures.append(scope.manifest_path)
            continue
        if plan.edits:
            plans.append(plan)

    all_edits = [edit for plan in plans for edit in plan.edits]
    if not all_edits:
        if fetch_failures or plan_failures:
            _error(
                f"No minimums could be raised for {package} and some inputs "
                "failed to resolve; see the warnings above. Not reporting this "
                "as up to date."
            )
            return 1
        _notice(
            f"All in-scope minimums for {package} are already at or above the "
            "latest compatible stable PyPI releases; nothing to do."
        )
        return 0

    for plan in plans:
        (REPO_ROOT / plan.manifest_path).write_text(plan.new_text, encoding="utf-8")

    summary = edits_markdown(all_edits, heading=f"Raised {len(all_edits)} minimum(s):")
    if fetch_failures or plan_failures:
        summary += "\n\n" + failures_markdown(fetch_failures, plan_failures)
    print(summary)
    _write_step_summary(summary)

    changed_files = sorted(plan.manifest_path for plan in plans)
    lock_dirs = stale_lock_dirs(changed_files)
    _write_output("changed", "true")
    _write_output("changed_files", ",".join(changed_files))
    _write_output("lock_dirs", ",".join(lock_dirs))
    # `dir=python` pairs so the workflow locks with the same interpreter
    # `check_lockfiles.yml` will verify against.
    _write_output(
        "lock_specs",
        ",".join(f"{path}={python_version(REPO_ROOT / path)}" for path in lock_dirs),
    )
    _write_output("summary", summary)
    return 0


def main() -> int:
    """Entry point: raise in-scope lower bounds for the selected package(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        required=True,
        help=(
            "Release component label (or its release path), or 'all' for every "
            "release package."
        ),
    )
    args = parser.parse_args()
    try:
        return _run(args.package)
    except Exception as err:  # noqa: BLE001  # fail closed on script defects
        _error(f"Raising dependency minimums failed unexpectedly: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
