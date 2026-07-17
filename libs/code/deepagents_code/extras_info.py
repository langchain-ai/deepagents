"""Inspect optional-dependency install status for the running distribution.

Reads `Requires-Dist` metadata to report which packages declared under
`[project.optional-dependencies]` are installed, and renders that status
in either plain text (for stdout) or markdown (for rich UI contexts).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    version as pkg_version,
)
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import url2pathname

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

DistributionMetadataStatus = Literal["resolved", "not_installed", "error"]
"""Outcome of a distribution version lookup.

Used for any distribution (the `deepagents` SDK and the `deepagents-code`
CLI alike). `"not_installed"` means the package metadata is genuinely absent;
`"error"` means an unexpected failure occurred while reading it. Callers
that don't care which kind of failure happened can treat both the same.
"""


def _editable_sdk_source_root() -> Path | None:
    """Return the editable `deepagents` source root from package metadata."""
    try:
        raw = distribution("deepagents").read_text("direct_url.json")
        if not raw:
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.debug("Ignoring malformed deepagents direct_url.json metadata")
            return None
        dir_info = data.get("dir_info")
        if not isinstance(dir_info, dict):
            logger.debug("Ignoring malformed deepagents direct_url.json dir_info")
            return None
        if not dir_info.get("editable", False):
            return None
        url = data.get("url")
        if not isinstance(url, str):
            logger.debug("Ignoring editable deepagents metadata without a source URL")
            return None
        parsed = urlparse(url)
        if parsed.scheme != "file":
            logger.debug("Ignoring editable deepagents metadata with non-file URL")
            return None
        path = url2pathname(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            path = f"//{parsed.netloc}{path}"
        return Path(path)
    except (PackageNotFoundError, OSError, ValueError, TypeError):
        # `OSError` covers `FileNotFoundError`/`PermissionError`/etc. while
        # reading the metadata file; `ValueError` covers malformed JSON
        # (`json.JSONDecodeError`), bad encodings (`UnicodeDecodeError`), and an
        # invalid IPv6 host from `urlparse`; `TypeError` covers a non-text
        # `read_text` payload. `url2pathname` is intentionally lenient and adds
        # no new failure modes. This probe must never propagate, since callers
        # treat it as a best-effort refinement over the metadata version.
        return None


def _sdk_version_from_source(root: Path) -> str | None:
    """Read `deepagents.__version__` from a source tree rooted at `root`.

    Returns:
        The source SDK version, or `None` when it cannot be read.
    """
    version_file = root / "deepagents" / "_version.py"
    try:
        source = version_file.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(version_file))
    except (OSError, SyntaxError, ValueError):
        # Reached only for editable installs, where the package is known to be
        # present — so an unreadable or malformed version file is a broken local
        # checkout, not an absent dependency. Warn (not debug): the source
        # version is masked and the caller falls back to potentially stale
        # metadata.
        logger.warning("Failed to read deepagents SDK version file", exc_info=True)
        return None
    for node in module.body:
        # Match only a plain `__version__ = "..."` assignment. release-please
        # writes the SDK's `_version.py` that way, so annotated (`ast.AnnAssign`)
        # or tuple-target forms are intentionally ignored.
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            # A non-literal `__version__` RHS masks the source version just like
            # an unreadable file, so warn for parity with the read/parse failure
            # above rather than falling back to stale metadata silently.
            logger.warning(
                "Failed to evaluate deepagents SDK __version__ literal",
                exc_info=True,
            )
            return None
        return value if isinstance(value, str) and value else None
    return None


def _contract_home(path: Path) -> str:
    """Return `path` as text with a home-directory prefix contracted to `~`.

    Args:
        path: Filesystem path to render.

    Returns:
        The stringified path, with a leading home directory replaced by `~`
            when applicable. The raw path is returned when the home directory
            cannot be determined.
    """
    text = str(path)
    try:
        home = str(Path.home())
    except (OSError, RuntimeError):
        return text
    if text == home:
        return "~"
    prefix = home if home.endswith("/") else f"{home}/"
    if text.startswith(prefix):
        return f"~/{text[len(prefix) :]}"
    return text


@dataclass(frozen=True)
class DistributionVersion:
    """Structured version facts for a single installed distribution.

    Separates three facts that are easily conflated: the live version read from
    the running source, whether imported or parsed (`source_version`), the
    version recorded in the installed distribution metadata (`metadata_version`),
    and whether the install is editable (with its `source_path`). `status`
    records whether the metadata lookup succeeded so diagnostic callers can
    distinguish a genuinely absent package from an unexpected lookup failure.
    """

    name: str
    """Distribution name, such as `deepagents` or `deepagents-code`."""

    source_version: str | None
    """Version read from the imported/source `_version.py`, when available."""

    metadata_version: str | None
    """Version from `importlib.metadata`, when the distribution is installed."""

    editable: bool
    """Whether the distribution is installed in editable mode."""

    source_path: str | None
    """`~`-contracted editable source root, when known.

    Available for callers that want to display the editable root; the
    `--version`/`/version` surfaces render the CLI path from `config` instead,
    so this is not rendered on every surface today.
    """

    status: DistributionMetadataStatus
    """Outcome of the metadata lookup."""

    @property
    def primary_version(self) -> str | None:
        """Version to report as authoritative.

        Editable installs prefer the live source version because their metadata
        can lag the working tree after a local version change; every other
        install uses the metadata version, falling back to the source version
        when metadata is absent. This keeps the "a resolved lookup yields a
        non-`None` version" contract intact even when a source version is the
        only fact available.
        """
        if self.editable and self.source_version:
            return self.source_version
        return self.metadata_version or self.source_version

    @property
    def has_drift(self) -> bool:
        """Whether the source and metadata versions are both known and differ."""
        return (
            self.source_version is not None
            and self.metadata_version is not None
            and self.source_version != self.metadata_version
        )


@dataclass(frozen=True)
class VersionReport:
    """Network-free snapshot of the version facts diagnostics need.

    Bundles the running CLI and installed SDK version facts with the result of
    comparing the `deepagents` requirement that `deepagents-code` declares
    against the installed SDK metadata version.
    """

    cli: DistributionVersion
    """Version facts for the running `deepagents-code` distribution."""

    sdk: DistributionVersion
    """Version facts for the installed `deepagents` SDK distribution."""

    sdk_requirement: Requirement | None
    """The `deepagents` requirement declared by `deepagents-code`, if found."""

    sdk_requirement_satisfied: bool | None
    """Whether the installed SDK metadata satisfies `sdk_requirement`.

    `None` when the comparison cannot be made (no declared requirement, the SDK
    is not installed, or an unparseable version).
    """

    @property
    def sdk_requirement_mismatch(self) -> bool:
        """Whether the installed SDK metadata violates the declared requirement."""
        # `is False`, not `not`: an inconclusive comparison (`None`) is *not* a
        # mismatch. Collapsing this to `not self.sdk_requirement_satisfied` would
        # silently report every unknown as a mismatch and flag doctor unhealthy.
        return self.sdk_requirement_satisfied is False


def _read_cli_source_version() -> str | None:
    """Return the running `deepagents-code` source version, or `None`.

    Reads `deepagents_code._version.__version__`, which is always present for
    the running package but is imported defensively so best-effort diagnostics
    never crash on a broken checkout.
    """
    try:
        from deepagents_code._version import __version__
    except Exception:
        # `_version` is generated and always present for the running package, so
        # a failure here means a broken checkout that masks the source version
        # and forces a fall back to potentially stale metadata. Warn for parity
        # with `_sdk_version_from_source`, which handles the same failure shape.
        logger.warning("Could not read deepagents-code source version", exc_info=True)
        return None
    return __version__ if isinstance(__version__, str) and __version__ else None


def _cli_editable_info() -> tuple[bool, str | None]:
    """Return the editable status and source path for `deepagents-code`.

    Reuses the cached PEP 610 detection in `config` rather than reimplementing
    it, so both surfaces agree and stay patchable in tests.
    """
    try:
        from deepagents_code.config import (
            _get_editable_install_path,
            _is_editable_install,
        )

        return _is_editable_install(), _get_editable_install_path()
    except Exception:
        logger.debug(
            "Could not determine deepagents-code editable status", exc_info=True
        )
        return False, None


def collect_cli_version_info() -> DistributionVersion:
    """Collect version facts for the running `deepagents-code` distribution.

    Returns:
        The structured CLI version facts. The status is `"resolved"` whenever a
            source or metadata version is available, `"not_installed"` when the
            metadata is genuinely missing (`PackageNotFoundError`) and no source
            version is available, and `"error"` on an unexpected metadata failure
            with no source fallback.
    """
    source = _read_cli_source_version()
    try:
        metadata: str | None = pkg_version("deepagents-code")
        status: DistributionMetadataStatus = "resolved"
    except PackageNotFoundError:
        logger.debug("deepagents-code package metadata not found in environment")
        metadata = None
        status = "resolved" if source else "not_installed"
    except Exception:  # Best-effort lookup; never propagate to the caller
        logger.warning(
            "Unexpected error looking up deepagents-code metadata version",
            exc_info=True,
        )
        metadata = None
        status = "resolved" if source else "error"
    editable, path = _cli_editable_info()
    return DistributionVersion(
        name="deepagents-code",
        source_version=source,
        metadata_version=metadata,
        editable=editable,
        source_path=path,
        status=status,
    )


def collect_sdk_version_info() -> DistributionVersion:
    """Collect version facts for the installed `deepagents` SDK distribution.

    Editable SDK installs prefer the source tree's `_version.py`; everything
    else reports the installed metadata version. Distinguishes a genuinely
    missing package from an unexpected metadata error.

    Returns:
        The structured SDK version facts.
    """
    try:
        metadata = pkg_version("deepagents")
    except PackageNotFoundError:
        logger.debug("deepagents SDK package not found in environment")
        return DistributionVersion(
            name="deepagents",
            source_version=None,
            metadata_version=None,
            editable=False,
            source_path=None,
            status="not_installed",
        )
    except Exception:  # Best-effort lookup; never propagate to the caller
        logger.warning(
            "Unexpected error looking up deepagents SDK version", exc_info=True
        )
        return DistributionVersion(
            name="deepagents",
            source_version=None,
            metadata_version=None,
            editable=False,
            source_path=None,
            status="error",
        )

    source_root = _editable_sdk_source_root()
    editable = source_root is not None
    source_version = _sdk_version_from_source(source_root) if source_root else None
    source_path = _contract_home(source_root) if source_root else None
    return DistributionVersion(
        name="deepagents",
        source_version=source_version,
        metadata_version=metadata,
        editable=editable,
        source_path=source_path,
        status="resolved",
    )


def sdk_requirement_from_cli(
    distribution_name: str = "deepagents-code",
) -> Requirement | None:
    """Return the base `deepagents` requirement declared by `deepagents-code`.

    Reads the installed distribution's `Requires-Dist` metadata and parses each
    entry with `packaging.Requirement`, returning the applicable base
    `deepagents` dependency. Extras-gated or environment-conditional variants
    whose marker does not apply to the current environment are skipped.

    Args:
        distribution_name: Name of the installed distribution to inspect.

    Returns:
        The parsed `deepagents` requirement, or `None` when the distribution or
            requirement cannot be read.
    """
    try:
        dist = distribution(distribution_name)
    except PackageNotFoundError:
        logger.debug(
            "Distribution %s not found; cannot read deepagents requirement",
            distribution_name,
        )
        return None
    except Exception:  # Best-effort lookup; never propagate to the caller
        logger.warning(
            "Unexpected error reading %s requirements", distribution_name, exc_info=True
        )
        return None

    try:
        requirements = dist.requires or []
    except Exception:  # Best-effort lookup; malformed metadata must not be fatal
        logger.warning(
            "Unexpected error reading %s requirements", distribution_name, exc_info=True
        )
        return None

    target = canonicalize_name("deepagents")
    for raw in requirements:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            logger.warning("Could not parse Requires-Dist entry: %s", raw)
            continue
        if canonicalize_name(req.name) != target:
            continue
        if req.marker is not None:
            # Skip extras-gated (`extra == "..."`) or environment-conditional
            # variants that do not apply; only the base runtime dependency is
            # meaningful for a straight installed-version comparison.
            try:
                applicable = req.marker.evaluate()
            except Exception:
                # An unexpected marker-evaluation failure must not silently drop
                # the base requirement — that would hide the very SDK mismatch
                # this check exists to surface. Warn and treat it as applicable
                # so a genuine mismatch is still flagged rather than masked.
                logger.warning(
                    "Could not evaluate marker for Requires-Dist entry %s; "
                    "treating the requirement as applicable",
                    raw,
                    exc_info=True,
                )
            else:
                if not applicable:
                    continue
        return req
    return None


def _requirement_satisfied(
    requirement: Requirement | None, metadata_version: str | None
) -> bool | None:
    """Return whether `metadata_version` satisfies `requirement`.

    Args:
        requirement: The declared requirement, or `None`.
        metadata_version: The installed SDK metadata version, or `None`.

    Returns:
        `True`/`False` when the comparison can be made, or `None` when either
            input is absent or the version cannot be parsed. Prereleases are
            allowed so a prerelease pin (e.g. `==0.7.0a7`) is evaluated
            correctly.
    """
    if requirement is None or metadata_version is None:
        return None
    try:
        parsed = Version(metadata_version)
    except InvalidVersion:
        logger.debug(
            "Could not parse SDK metadata version %r for requirement comparison",
            metadata_version,
            exc_info=True,
        )
        return None
    return requirement.specifier.contains(parsed, prereleases=True)


def collect_version_report() -> VersionReport:
    """Collect the offline version report used by `--version`, `/version`, and doctor.

    Returns:
        A `VersionReport` bundling CLI and SDK version facts with the declared
            `deepagents` requirement and whether the installed SDK metadata
            satisfies it. Performs no network or subprocess calls.
    """
    cli = collect_cli_version_info()
    sdk = collect_sdk_version_info()
    requirement = sdk_requirement_from_cli()
    # Compare against the metadata version — what the environment actually
    # installed — rather than the (possibly ahead) editable source version, so
    # the mismatch reflects the real dependency resolution.
    satisfied = _requirement_satisfied(requirement, sdk.metadata_version)
    return VersionReport(
        cli=cli,
        sdk=sdk,
        sdk_requirement=requirement,
        sdk_requirement_satisfied=satisfied,
    )


def _format_requirement_display(requirement: Requirement) -> str:
    """Render a requirement's version constraint for display.

    A single exact pin (`==X`) is shown as the bare version for readability;
    any other constraint keeps its full specifier form.

    Returns:
        The display form of the requirement's version constraint.
    """
    specs = list(requirement.specifier)
    if len(specs) == 1 and specs[0].operator == "==":
        return specs[0].version
    return str(requirement.specifier) or "any"


def _join_annotation_parts(parts: list[str]) -> str:
    """Join annotation parts into a trailing ` (...)` suffix.

    Returns:
        The parts joined as ` (a; b)`, or an empty string when `parts` is empty.
    """
    return f" ({'; '.join(parts)})" if parts else ""


def format_cli_version_annotation(info: DistributionVersion) -> str:
    """Return the source/metadata drift annotation for the CLI version line.

    Editable status is surfaced separately by every caller (the dedicated
    `Editable install:` line on `--version`/`/version`, and the `Install method`
    item in `doctor`), so it is intentionally *not* repeated inline here — that
    kept the CLI reading "editable" twice on those surfaces.

    Args:
        info: The CLI version facts.

    Returns:
        A trailing ` (...)` suffix flagging any drift between the source and
            installed metadata versions, or an empty string when they agree (so
            normal installs stay unchanged).
    """
    parts: list[str] = []
    if info.has_drift and info.metadata_version is not None:
        parts.append(f"installed metadata: {info.metadata_version}")
    return _join_annotation_parts(parts)


def format_sdk_version_annotation(report: VersionReport) -> str:
    """Return the editable/drift/mismatch annotation for the SDK version line.

    Args:
        report: The collected version report.

    Returns:
        A trailing ` (...)` suffix flagging an editable SDK install, any drift
            between the source and installed metadata versions, and an
            actionable requirement mismatch when the installed SDK metadata does
            not satisfy the requirement declared by `deepagents-code`. Empty
            when none applies.

    The mismatch is judged against the installed SDK *metadata* version (what
    the environment actually resolved), not the source version shown as primary
    for editable installs. When the two differ, the accompanying
    `installed metadata:` note makes that basis explicit, so an editable source
    version that appears to match the requirement can still read as a mismatch.
    """
    info = report.sdk
    parts: list[str] = []
    if info.editable:
        parts.append("editable")
    if info.has_drift and info.metadata_version is not None:
        parts.append(f"installed metadata: {info.metadata_version}")
    if report.sdk_requirement_mismatch and report.sdk_requirement is not None:
        # Compared against `metadata_version` (see docstring), which is why the
        # drift note above is emitted alongside so the basis is visible.
        required = _format_requirement_display(report.sdk_requirement)
        parts.append(f"required by deepagents-code: {required} — mismatch")
    return _join_annotation_parts(parts)


def resolve_sdk_version() -> tuple[str | None, DistributionMetadataStatus]:
    """Resolve the installed `deepagents` SDK version.

    Compatibility wrapper over `collect_sdk_version_info` for callers that only
    need the primary version and lookup status. Editable installs can have stale
    package metadata after local version files change, so they prefer the source
    tree's `_version.py` and fall back to metadata when the source version is
    unavailable. Distinguishes a genuinely missing package from an unexpected
    metadata error so diagnostic callers can report the two differently, while
    collapse-friendly callers can ignore the split.

    Returns:
        `(version, status)`. `version` is the resolved version string when
            `status` is `"resolved"`, otherwise `None`.
    """
    info = collect_sdk_version_info()
    if info.status != "resolved":
        return None, info.status
    return info.primary_version, "resolved"


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
        "meta",
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

STANDALONE_EXTRAS: frozenset[str] = frozenset({"media", "quickjs"})
"""Optional extras that don't fit the provider/sandbox taxonomy.

`quickjs` is a core dependency as of 0.1.24, but the empty extra remains
installable so older `deepagents-code[quickjs]` and `/install quickjs` workflows
stay harmless.
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
        """Whether all declared packages for this extra are installed."""
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
    """Check that `langchain-quickjs` is installed for the interpreter.

    Uses `importlib.util.find_spec` for a lightweight check with no actual
    imports. Call this in the app process *before* spawning the server
    subprocess so users get a clear, actionable error instead of an opaque
    server crash when the core dependency is missing or broken.

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
            msg = (
                "Missing core dependency for the interpreter. Editable install "
                "detected — refresh the local environment with uv sync, or "
                "relaunch with --no-interpreter to skip it."
            )
        else:
            msg = (
                "Missing core dependency for the interpreter. "
                "Reinstall dcode to restore langchain-quickjs, or relaunch with "
                "--no-interpreter to skip it."
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
