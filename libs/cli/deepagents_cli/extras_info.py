"""Inspect optional-dependency install status for the running distribution.

Reads `Requires-Dist` metadata to report which packages declared under
`[project.optional-dependencies]` are installed, and renders that status
in either plain text (for stdout) or markdown (for rich UI contexts).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as pkg_version,
)

from packaging.requirements import InvalidRequirement, Requirement

logger = logging.getLogger(__name__)

_EXTRA_MARKER_RE = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")

_COMPOSITE_EXTRAS: frozenset[str] = frozenset({"all-providers", "all-sandboxes"})
"""Extras whose package set is already covered by other, more specific extras.

Build backends flatten these meta-extras into their component packages
rather than preserving the `deepagents-cli[a,b,...]` self-reference, so
name-based filtering is the only reliable way to drop them.
"""

MODEL_PROVIDER_EXTRAS: frozenset[str] = frozenset(
    {
        "anthropic",
        "baseten",
        "bedrock",
        "cohere",
        "deepseek",
        "fireworks",
        "google-genai",
        "groq",
        "huggingface",
        "ibm",
        "litellm",
        "mistralai",
        "nvidia",
        "ollama",
        "openai",
        "openrouter",
        "perplexity",
        "vertexai",
        "xai",
    }
)
"""Optional extras that add model-provider integrations.

Keep in sync with `[project.optional-dependencies]` in `pyproject.toml`.
"""

SANDBOX_EXTRAS: frozenset[str] = frozenset({"agentcore", "daytona", "modal", "runloop"})
"""Optional extras that add sandbox integrations."""

ExtrasStatus = dict[str, list[tuple[str, str]]]
"""Mapping from extra name to `(package, installed_version)` tuples.

Only packages that are actually installed are included. Extras whose
declared packages are all missing are omitted entirely.
"""


@dataclass(frozen=True)
class ExtraDependencyStatus:
    """Install status for one optional dependency extra."""

    name: str
    """Extra name, such as `anthropic` or `daytona`."""

    installed: tuple[tuple[str, str], ...]
    """Installed `(package, version)` pairs declared by this extra."""

    missing: tuple[str, ...]
    """Declared package names for this extra that are not installed."""

    @property
    def ready(self) -> bool:
        """Return whether all declared packages for this extra are installed."""
        return bool(self.installed) and not self.missing


def _extract_extra_name(marker_str: str) -> str | None:
    """Pull the extra name out of a marker like `extra == "anthropic"`.

    Args:
        marker_str: String form of a `packaging.markers.Marker`.

    Returns:
        The quoted extra name, or `None` when the marker does not carry an
            `extra == "..."` clause.
    """
    match = _EXTRA_MARKER_RE.search(marker_str)
    return match.group(1) if match else None


def get_extras_status(
    distribution_name: str = "deepagents-cli",
) -> ExtrasStatus:
    """Return installed optional dependencies grouped by extra.

    Reads `Requires-Dist` metadata from the named distribution, groups the
    entries gated by `extra == "..."` markers under their extra name, and
    resolves each package's installed version via `importlib.metadata`.
    Packages that are not installed are omitted; extras whose entire
    package list is absent are dropped.

    Composite meta-extras that only bundle other extras (see
    `_COMPOSITE_EXTRAS`) and self-references to the distribution itself
    are skipped — their components already appear under their own extras.

    Args:
        distribution_name: Name of the installed distribution to inspect.

    Returns:
        Mapping from extra name to a sorted list of `(package, version)`
            tuples for packages that are currently installed. An empty
            mapping is returned when the distribution itself is not found.
    """
    result: ExtrasStatus = {}
    for extra in get_optional_dependency_status(distribution_name):
        if extra.installed:
            result[extra.name] = list(extra.installed)
    return result


def get_optional_dependency_status(
    distribution_name: str = "deepagents-cli",
) -> tuple[ExtraDependencyStatus, ...]:
    """Return installed and missing optional dependencies grouped by extra.

    Args:
        distribution_name: Name of the installed distribution to inspect.

    Returns:
        Sorted tuple of optional extra statuses. An empty tuple is returned
            when the distribution itself is not found.
    """
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError:
        # Editable installs renamed by the user, dev checkouts without metadata,
        # or vendored copies all hit this path. The dependency screen otherwise
        # silently renders "none detected" twice; warn so the cause is visible.
        logger.warning(
            "Distribution %s not found; optional-dependency status will be empty",
            distribution_name,
        )
        return ()

    own_name = distribution_name.lower()
    installed: dict[str, list[tuple[str, str]]] = {}
    missing: dict[str, list[str]] = {}
    for raw in dist.requires or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            logger.warning("Could not parse Requires-Dist entry: %s", raw)
            continue
        if not req.marker:
            continue
        extra = _extract_extra_name(str(req.marker))
        if not extra:
            continue
        if extra in _COMPOSITE_EXTRAS:
            continue
        if req.name.lower() == own_name:
            continue
        try:
            version = pkg_version(req.name)
        except PackageNotFoundError:
            missing.setdefault(extra, []).append(req.name)
        else:
            installed.setdefault(extra, []).append((req.name, version))

    names = sorted(set(installed) | set(missing))
    return tuple(
        ExtraDependencyStatus(
            name=name,
            installed=tuple(sorted(installed.get(name, []))),
            missing=tuple(sorted(missing.get(name, []))),
        )
        for name in names
    )


def format_extras_status_plain(status: ExtrasStatus) -> str:
    """Render an `ExtrasStatus` mapping as column-aligned plain text.

    Suitable for stdout in non-interactive contexts (e.g. the `--version`
    CLI flag) where a markdown renderer is unavailable.

    Args:
        status: Mapping returned by `get_extras_status`.

    Returns:
        Multi-line string with a heading and one `extra  package  version`
            row per installed package.

            Returns an empty string when `status` is empty.
    """
    if not status:
        return ""
    rows: list[tuple[str, str, str]] = [
        (extra_name, pkg_name, version)
        for extra_name, pkgs in status.items()
        for pkg_name, version in pkgs
    ]
    extra_width = max(len(row[0]) for row in rows)
    package_width = max(len(row[1]) for row in rows)
    lines = ["Installed optional dependencies:"]
    lines.extend(
        f"  {extra.ljust(extra_width)}  {pkg.ljust(package_width)}  {version}"
        for extra, pkg, version in rows
    )
    return "\n".join(lines)


def format_extras_status(status: ExtrasStatus) -> str:
    """Render an `ExtrasStatus` mapping as a markdown fragment.

    Args:
        status: Mapping returned by `get_extras_status`.

    Returns:
        Multi-line markdown string containing a heading and a pipe table
            with `Extra`, `Package`, and `Version` columns, suitable for
            rendering via a markdown widget.

            Returns an empty string when `status` is empty.
    """
    if not status:
        return ""
    rows: list[tuple[str, str, str]] = [
        (extra_name, pkg_name, version)
        for extra_name, pkgs in status.items()
        for pkg_name, version in pkgs
    ]
    headers = ("Extra", "Package", "Version")

    def _row(cells: tuple[str, str, str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        "### Installed optional dependencies",
        "",
        _row(headers),
        "| " + " | ".join("---" for _ in headers) + " |",
        *(_row(row) for row in rows),
    ]
    return "\n".join(lines)
