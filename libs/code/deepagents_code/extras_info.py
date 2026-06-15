"""Inspect optional-dependency install status for the running distribution.

Reads `Requires-Dist` metadata to report which packages declared under
`[project.optional-dependencies]` are installed, and renders that status
in either plain text (for stdout) or markdown (for rich UI contexts).
"""

from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as pkg_version,
)

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

logger = logging.getLogger(__name__)

_EXTRA_MARKER_RE = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


class ExtrasIntrospectionError(RuntimeError):
    """Raised when installed extras cannot be determined safely."""


_COMPOSITE_EXTRAS: frozenset[str] = frozenset({"all-providers", "all-sandboxes"})
"""Extras whose package set is already covered by other, more specific extras.

Build backends flatten these meta-extras into their component packages
rather than preserving the `deepagents-code[a,b,...]` self-reference, so
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
        "together",
        "vertex",
        "xai",
    }
)
"""Optional extras that add model-provider integrations.

Keep in sync with `[project.optional-dependencies]` in `pyproject.toml`.
"""

SANDBOX_EXTRAS: frozenset[str] = frozenset(
    {"agentcore", "daytona", "modal", "runloop", "vercel"}
)
"""Optional extras that add sandbox integrations."""

STANDALONE_EXTRAS: frozenset[str] = frozenset({"quickjs"})
"""Optional extras that don't fit the provider/sandbox taxonomy.

These integrations layer onto the main agent (e.g. a JS REPL via
`langchain-quickjs`) and aren't grouped under `all-providers` or
`all-sandboxes`.
"""

KNOWN_EXTRAS: frozenset[str] = (
    MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS | STANDALONE_EXTRAS
)
"""Union of all individually-installable extras.

Excludes the composite meta-extras (`all-providers`, `all-sandboxes`) since
those expand to other extras and don't add anything on their own.
Drift-protected by `test_model_config.TestProviderApiKeyEnv` and the
model-provider-drift checks; new extras must be added to the corresponding
category frozenset above.
"""


def format_known_extras() -> str:
    """Render the installable extras grouped by category as plain text.

    Drives the no-argument `/install` slash-command help so users can
    discover valid extras without consulting `pyproject.toml`. Sourced from
    the category frozensets above, so it stays in sync with `KNOWN_EXTRAS`
    automatically.

    Returns:
        Multi-line string with one labeled line per category, each listing
            its extras alphabetically.
    """
    groups: tuple[tuple[str, frozenset[str]], ...] = (
        ("Model providers", MODEL_PROVIDER_EXTRAS),
        ("Sandboxes", SANDBOX_EXTRAS),
        ("Other", STANDALONE_EXTRAS),
    )
    lines = ["Available extras:"]
    lines.extend(
        f"  {label}: {', '.join(sorted(extras))}" for label, extras in groups if extras
    )
    return "\n".join(lines)


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
    distribution_name: str = "deepagents-code",
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


def installed_extra_names(
    distribution_name: str = "deepagents-code",
    *,
    strict: bool = False,
) -> set[str]:
    """Return extras with at least one installed dependency.

    Args:
        distribution_name: Name of the installed distribution to inspect.
        strict: Raise when the distribution metadata cannot be read or parsed
            reliably.

    Returns:
        Set of extra names whose optional dependency metadata has at least one
            installed package. Composite extras are excluded.
    """
    statuses = get_optional_dependency_status(distribution_name, strict=strict)
    return {extra.name for extra in statuses if extra.installed}


def get_optional_dependency_status(
    distribution_name: str = "deepagents-code",
    *,
    strict: bool = False,
) -> tuple[ExtraDependencyStatus, ...]:
    """Return installed and missing optional dependencies grouped by extra.

    Args:
        distribution_name: Name of the installed distribution to inspect.
        strict: Raise when the distribution metadata cannot be read or parsed
            reliably.

    Returns:
        Sorted tuple of optional extra statuses. An empty tuple is returned
            when the distribution itself is not found.

    Raises:
        ExtrasIntrospectionError: If `strict` is `True` and metadata
            introspection fails.
    """
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError:
        if strict:
            msg = (
                f"Distribution {distribution_name!r} not found; cannot preserve "
                "already-installed extras safely"
            )
            raise ExtrasIntrospectionError(msg) from None
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
            if strict:
                msg = (
                    "Could not parse optional-dependency metadata; cannot "
                    f"preserve already-installed extras safely: {raw}"
                )
                raise ExtrasIntrospectionError(msg) from None
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


def extra_for_package(
    package: str,
    distribution_name: str = "deepagents-code",
) -> str | None:
    """Return the installable extra that declares a package.

    Resolves recovery hints from the package that is actually missing
    instead of guessing from a provider identifier. For example,
    `langchain-google-vertexai` maps to the `vertex` extra even though the
    provider id is `google_vertexai`.

    Args:
        package: Distribution package name to find in optional dependencies.
        distribution_name: Name of the installed distribution to inspect.

    Returns:
        The known extra name that declares `package`, or `None` when the
            package is not declared by an individually-installable extra,
            or when the distribution's metadata could not be read (logged
            at `warning` level — callers should treat both cases the same
            since the right fallback in either is `install_package_command`).
    """
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError:
        logger.warning(
            "Distribution %s not found; cannot resolve extra for package %s",
            distribution_name,
            package,
        )
        return None

    own_name = canonicalize_name(distribution_name)
    target = canonicalize_name(package)
    for raw in dist.requires or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            logger.warning("Could not parse Requires-Dist entry: %s", raw)
            continue
        if canonicalize_name(req.name) != target:
            continue
        if canonicalize_name(req.name) == own_name:
            continue
        if not req.marker:
            continue
        extra = _extract_extra_name(str(req.marker))
        if extra in KNOWN_EXTRAS:
            return extra
    return None


def verify_interpreter_deps() -> None:
    """Check that `langchain-quickjs` is installed for the `--interpreter` flag.

    Uses `importlib.util.find_spec` for a lightweight check with no actual
    imports. Call this in the app process *before* spawning the server
    subprocess so users get a clear, actionable error instead of an opaque
    server crash when the optional `quickjs` extra is not installed.

    Returns silently when the package is importable.

    Raises:
        ImportError: If `langchain_quickjs` is not importable.
    """
    try:
        found = importlib.util.find_spec("langchain_quickjs") is not None
    except (ImportError, ValueError):
        # A broken-but-installed `langchain_quickjs` (e.g., parent package
        # raises during import) would otherwise masquerade as "not installed";
        # capture the underlying cause for debug logs.
        logger.debug("find_spec failed for langchain_quickjs", exc_info=True)
        found = False

    if not found:
        from deepagents_code.config import _is_editable_install

        if _is_editable_install():
            from deepagents_code.update_check import editable_extra_hint

            msg = (
                "Missing dependencies for --interpreter. Editable install "
                f"detected — {editable_extra_hint('quickjs')}"
            )
        else:
            msg = (
                "Missing dependencies for --interpreter. "
                "Install with: dcode --install quickjs"
            )
        raise ImportError(msg)


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


CORE_DEPENDENCIES: tuple[str, ...] = (
    "langchain",
    "langchain-core",
    "langgraph",
    "langgraph-checkpoint",
    "langgraph-prebuilt",
    "langgraph-sdk",
    "langsmith",
)
"""Core LangChain-ecosystem packages surfaced for editable installs.

The deepagents SDK is reported separately by `/version`, so it is omitted
here. These are the packages a local checkout is most likely to pin or
override, so their resolved versions help diagnose editable environments.
"""


def get_core_dependency_versions() -> list[tuple[str, str | None]]:
    """Return `(package, version)` pairs for the core ecosystem dependencies.

    Returns:
        One entry per package in `CORE_DEPENDENCIES`, in declaration order.
            The version is `None` when the package is not installed.
    """
    versions: list[tuple[str, str | None]] = []
    for name in CORE_DEPENDENCIES:
        try:
            versions.append((name, pkg_version(name)))
        except PackageNotFoundError:
            versions.append((name, None))
    return versions


def format_core_dependencies_plain() -> str:
    """Render core ecosystem dependency versions as column-aligned plain text.

    Suitable for stdout in non-interactive contexts (e.g. the `--version`
    CLI flag) where a markdown renderer is unavailable.

    Returns:
        Multi-line string with a heading and one `package  version` row per
            core dependency. Missing packages are reported as `not installed`.
    """
    rows = [
        (name, version or "not installed")
        for name, version in get_core_dependency_versions()
    ]
    package_width = max(len(name) for name, _ in rows)
    lines = ["Core dependencies:"]
    lines.extend(f"  {name.ljust(package_width)}  {version}" for name, version in rows)
    return "\n".join(lines)


def format_core_dependencies() -> str:
    """Render core ecosystem dependency versions as a markdown fragment.

    Returns:
        Multi-line markdown string with a heading and a pipe table listing
            each core package and its resolved version (or `not installed`).
    """
    rows = [
        (name, version or "not installed")
        for name, version in get_core_dependency_versions()
    ]
    headers = ("Package", "Version")

    def _row(cells: tuple[str, str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [
        "### Core dependencies",
        "",
        _row(headers),
        "| " + " | ".join("---" for _ in headers) + " |",
        *(_row(row) for row in rows),
    ]
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
