"""Raise LangChain-ecosystem dependency lower bounds to the latest stable PyPI release.

For every in-scope requirement (see `IN_SCOPE_PREFIXES`) that declares a
concrete lower bound (`>=` or `~=`) in `[project.dependencies]`,
`[project.optional-dependencies]`, or `[dependency-groups]`, this rewrites that
lower bound in place to the newest non-yanked stable PyPI version. Upper
bounds, extras, and environment markers are preserved. Exact pins (`==`) are
left alone — raising a floor only applies to range floors. A bound already
ahead of the latest stable release (intentional prerelease coordination) is
respected and never lowered.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tomllib
from check_dep_freshness import (
    extract_minimum,
    latest_pypi_version,
    local_dependency_names,
)
from check_release_deps import fetch_pypi_json, load_release_packages
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_FETCH_WORKERS = 8
# Only `>=` / `~=` floors are raiseable. `==` is an exact pin (handled by the
# dedicated pin-bump workflow), and a bare upper bound has no floor to raise.
RAISEABLE_OPERATORS = frozenset({">=", "~="})
# Distribution-name prefixes whose floors are raised. Covers the LangChain
# integrations (langchain-*, the base langchain/langgraph/langsmith packages,
# and langgraph-* companion packages) plus the deepagents-* workspace packages
# that one package pulls from PyPI (e.g. deepagents-code -> deepagents-acp).
# Workspace-local sources and a package's own name are excluded per manifest.
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
        manifest_path: Repository-relative manifest that was edited.
        dependency_name: Canonicalized `langchain*` distribution name.
        old_requirement: Original requirement string as written in the manifest.
        new_requirement: Requirement string with the raised lower bound.
        old_minimum: Previous concrete lower bound.
        new_minimum: Raised lower bound (the latest stable PyPI release).

    """

    manifest_path: str
    dependency_name: str
    old_requirement: str
    new_requirement: str
    old_minimum: Version
    new_minimum: Version


def _project_requirement_strings(project: Mapping[str, object]) -> list[str]:
    """Collect requirement strings from `[project]` and `[dependency-groups]`."""
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
    """Collect requirement strings from `[dependency-groups]`."""
    groups = manifest.get("dependency-groups", {})
    if not isinstance(groups, Mapping):
        return []
    requirements: list[str] = []
    for values in groups.values():
        if isinstance(values, list):
            requirements.extend(item for item in values if isinstance(item, str))
    return requirements


def _raise_lower_bound(requirement_string: str, new_minimum: Version) -> str | None:
    """Rewrite one requirement string's concrete lower bound to `new_minimum`.

    Only the version token of the first `>=`/`~=` lower-bound specifier is
    replaced, so upper bounds, extras, and markers are preserved verbatim.

    Args:
        requirement_string: Raw requirement string as written in the manifest.
        new_minimum: The version to set as the new lower bound.

    Returns:
        The rewritten requirement string, or `None` when the requirement has
        no raiseable lower bound (e.g. an exact `==` pin or only an upper
        bound) and should be left unchanged.

    """
    for specifier in Requirement(requirement_string).specifier:
        if specifier.operator in RAISEABLE_OPERATORS:
            break
    else:
        return None
    # Replace the version half of the matched lower-bound clause. The clause
    # itself was parsed from this exact string, so its operator+version text
    # is present verbatim and uniquely identifies the span to swap.
    old_clause = f"{specifier.operator}{specifier.version}"
    new_clause = f"{specifier.operator}{new_minimum}"
    replaced, count = re.subn(
        re.escape(old_clause), new_clause, requirement_string, count=1
    )
    if count != 1:
        msg = f"Could not rewrite lower bound '{old_clause}' in '{requirement_string}'"
        raise ValueError(msg)
    return replaced


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
    """Return whether a dependency is an in-scope, externally-resolved floor."""
    if requirement.url or canonical_name in local_names or canonical_name == self_name:
        return False
    return canonical_name.startswith(IN_SCOPE_PREFIXES)


def _collect_in_scope_names(manifest_path: str) -> set[str]:
    """Return the canonicalized in-scope dependency names declared in a manifest."""
    manifest = tomllib.loads((REPO_ROOT / manifest_path).read_text(encoding="utf-8"))
    project = manifest.get("project")
    if not isinstance(project, Mapping):
        msg = f"{manifest_path} has no [project] table"
        raise TypeError(msg)

    local_names = local_dependency_names(manifest)
    self_name = _self_name(project)
    names: set[str] = set()
    for raw in _project_requirement_strings(project) + _group_requirement_strings(
        manifest
    ):
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        canonical_name = canonicalize_name(requirement.name)
        if not _in_scope(requirement, canonical_name, local_names, self_name):
            continue
        names.add(canonical_name)
    return names


def _fetch_latest_versions(names: set[str]) -> dict[str, Version]:
    """Fetch the latest stable PyPI version for each distribution name."""
    latest: dict[str, Version] = {}
    if not names:
        return latest
    workers = min(MAX_FETCH_WORKERS, len(names))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_pypi_json, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                payload = future.result()
            except Exception as err:  # noqa: BLE001  # surface as warning, keep going
                _warning(f"Skipping {name}: PyPI query failed ({err})")
                continue
            version = latest_pypi_version(payload, include_prereleases=False)
            if version is None:
                _warning(f"Skipping {name}: no stable PyPI release found")
                continue
            latest[name] = version
    return latest


def _raise_manifest(
    manifest_path: str, latest: Mapping[str, Version]
) -> list[RequirementEdit]:
    """Rewrite in-scope lower bounds in one manifest, returning the edits."""
    path = REPO_ROOT / manifest_path
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    manifest = tomllib.loads(original)
    project = manifest.get("project")
    if not isinstance(project, Mapping):
        msg = f"{manifest_path} has no [project] table"
        raise TypeError(msg)

    local_names = local_dependency_names(manifest)
    self_name = _self_name(project)
    requirement_strings = set(
        _project_requirement_strings(project) + _group_requirement_strings(manifest)
    )

    edits: list[RequirementEdit] = []
    replacements: dict[str, str] = {}
    for raw in requirement_strings:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        canonical_name = canonicalize_name(requirement.name)
        if not _in_scope(requirement, canonical_name, local_names, self_name):
            continue
        minimum = extract_minimum(requirement.specifier)
        if minimum is None:
            continue
        new_minimum = latest.get(canonical_name)
        if new_minimum is None or new_minimum <= minimum:
            continue
        raised = _raise_lower_bound(raw, new_minimum)
        if raised is None or raised == raw:
            continue
        replacements[raw] = raised
        edits.append(
            RequirementEdit(
                manifest_path=manifest_path,
                dependency_name=canonical_name,
                old_requirement=raw,
                new_requirement=raised,
                old_minimum=minimum,
                new_minimum=new_minimum,
            )
        )

    if not replacements:
        return []

    rewritten_lines: list[str] = []
    for line in lines:
        rewritten = line
        for old, new in replacements.items():
            if old in rewritten:
                rewritten = rewritten.replace(old, new)
        rewritten_lines.append(rewritten)
    path.write_text("".join(rewritten_lines), encoding="utf-8")
    return edits


def edits_markdown(edits: list[RequirementEdit], *, heading: str) -> str:
    """Render applied edits as a Markdown bullet list for the PR body/summary."""
    lines = [heading, ""]
    for edit in edits:
        lines.append(
            f"- `{edit.manifest_path}`: `{edit.dependency_name}` "
            f"`{edit.old_requirement}` → `{edit.new_requirement}`"
        )
    return "\n".join(lines)


def main() -> int:
    """Entry point: raise `langchain*` lower bounds for the selected package(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        required=True,
        help="Release component/package label, or 'all' for every release package.",
    )
    args = parser.parse_args()

    packages = load_release_packages()
    label_to_paths: dict[str, list[str]] = {}
    for path, label in packages.items():
        label_to_paths.setdefault(label, []).append(path)

    if args.package == "all":
        selected = sorted(packages)
    else:
        matched = [p for p, label in packages.items() if label == args.package]
        if not matched and args.package in packages:
            matched = [args.package]
        if not matched:
            known = ", ".join(sorted(label_to_paths)) or "none"
            _error(
                f"Unknown package '{args.package}'. Expected a release label "
                f"({known}) or 'all'."
            )
            return 1
        selected = sorted(matched)

    manifests = [f"{path}/pyproject.toml" for path in selected]
    _notice(f"Raising dependency minimums for {args.package}: " + ", ".join(manifests))

    in_scope_names: set[str] = set()
    for manifest_path in manifests:
        in_scope_names |= _collect_in_scope_names(manifest_path)
    if not in_scope_names:
        _notice(f"No in-scope requirements found for {args.package}; nothing to do.")
        return 0

    latest = _fetch_latest_versions(in_scope_names)

    all_edits: list[RequirementEdit] = []
    for manifest_path in manifests:
        all_edits.extend(_raise_manifest(manifest_path, latest))

    if not all_edits:
        _notice(
            f"All in-scope minimums for {args.package} are already at or above "
            "the latest stable PyPI releases; nothing to do."
        )
        return 0

    summary = edits_markdown(all_edits, heading=f"Raised {len(all_edits)} minimum(s):")
    print(summary)

    changed_files = sorted({edit.manifest_path for edit in all_edits})
    # Package dirs (the parent of each edited pyproject.toml) whose adjacent
    # uv.lock must be regenerated so the lockfile check stays green.
    lock_dirs = sorted({path.rsplit("/", 1)[0] for path in changed_files})
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(summary + "\n", encoding="utf-8")
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write("changed=true\n")
            handle.write(f"changed_files={','.join(changed_files)}\n")
            handle.write(f"lock_dirs={','.join(lock_dirs)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
