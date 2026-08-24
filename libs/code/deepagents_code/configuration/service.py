"""Process-local snapshots for managed and user TOML configuration."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from deepagents_code.configuration.resolver import ConfigResolver, ResolvedValue

from deepagents_code.configuration.paths import (
    managed_config_path,
    resolve_managed_path,
)
from deepagents_code.configuration.providers import TomlFileProvider
from deepagents_code.configuration.resolver import merge_toml_tables
from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)

logger = logging.getLogger(__name__)

UNION_PATHS = frozenset(
    {
        ("mcp", "disabled_project_servers"),
        ("mcp", "disabled_servers"),
    }
)
"""Paths whose lists accumulate instead of being replaced.

Deny lists must union across layers: replacing a managed deny list with a user
one would be a fail-open.

Governs the two *table merges* — `merge_managed_over_user` and the resolver's
deep-merge strategy. It does not govern the two readers that union
name sets rather than TOML tables: `model_config.load_mcp_server_trust_lists`
and `mcp_disabled.get_disabled_servers` accumulate their own layers directly,
including the env tier this set knows nothing about. A third deny list therefore
needs an entry here *and* a decision in those two readers.
"""

MANAGED_TABLE_PATHS = frozenset(
    {
        ("themes",),
        ("ui", "terminal_themes"),
        ("models", "providers"),
        ("async_subagents",),
        ("sandboxes", "providers"),
        ("threads", "columns"),
        ("effort",),
        ("effort", "by_model"),
    }
)
"""Sections whose managed value must be a TOML table.

The manifest also uses `STRUCTURED` for list-valued settings such as MCP
allowlists, so this cannot be derived from `OptionKind` alone. Parent paths
for every manifest option are added dynamically by
`managed_section_shape_violations` below.

`("effort",)` has no manifest option at all — it is written and read by
`model_config.load_effort_for_model` — so nothing else here would cover it, and
a managed scalar would replace the user's whole `[effort]` table.
`("effort", "by_model")` needs its own entry for the same reason: a managed
scalar there passes the top-level shape check and no manifest option gives the
merge validator a type for that path, so it would replace the user's
`[effort.by_model]` table.
`test_every_managed_table_path_is_enforced` pins every entry, because a renamed
section would otherwise stop being guarded silently.
"""


def union_paths_under(prefix: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    """Return `UNION_PATHS` rebased onto a subtree rooted at `prefix`.

    `merge_toml_tables` matches paths relative to where it starts. A merge of
    one option's own subtree therefore never matches an absolute deny-list
    path. Passing `UNION_PATHS` to such a merge does nothing at all. The merge
    then replaces a nested deny list instead of unioning it, which is a
    fail-open.

    Returns:
        The union paths that fall under `prefix`, relative to it.
    """
    depth = len(prefix)
    return frozenset(path[depth:] for path in UNION_PATHS if path[:depth] == prefix)


MANAGED_SOURCE = "managed config"
"""Provenance label for a value managed policy decided."""

USER_SOURCE = "config.toml"
"""Provenance label for a value the user's own file decided."""


def managed_decided(source: str) -> bool:
    """Return whether managed policy decided a value with this source label.

    A structured option merges both layers and reports a combined label, so
    `source == MANAGED_SOURCE` is the wrong test and silently answers `False`
    for a leaf managed policy does control.

    Returns:
        Whether managed policy contributed the value.
    """
    return source == MANAGED_SOURCE or source.startswith(f"{MANAGED_SOURCE} + ")


def merge_managed_over_user(
    user: Mapping[str, Any],
    managed: Mapping[str, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge managed policy over user config with this project's precedence.

    The single statement of managed-over-user precedence. It was assembled by
    hand at three call sites, two carrying "must match" comments, and had
    already drifted once: the site that omitted the validator reported a user
    table as effective while its provenance credited managed policy.

    Args:
        user: Lower-precedence user table.
        managed: Higher-precedence managed table, whose leaves win.
        prefix: Absolute manifest path of the subtree being merged. The union
            set and the validator both match paths relative to where the merge
            starts, so a subtree merge needs them rebased: an absolute deny-list
            path never matches inside a subtree, and an unprefixed leaf path
            resolves to no manifest option, which makes the validator accept
            everything.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """

    def managed_leaf_is_valid(path: tuple[str, ...], value: object) -> bool:
        """Validate one leaf against its absolute manifest path.

        Returns:
            Whether managed policy may apply the value at this leaf.
        """
        return is_valid_managed_scalar((*prefix, *path), value)

    return merge_toml_tables(
        user,
        managed,
        lower_source=USER_SOURCE,
        higher_source=MANAGED_SOURCE,
        union_paths=union_paths_under(prefix),
        higher_leaf_is_valid=managed_leaf_is_valid,
    )


@dataclass(frozen=True, slots=True)
class ConfigSources:
    """Managed and user TOML snapshots from one resolution generation."""

    managed: TomlSnapshot
    user: TomlSnapshot

    def dropped_managed_detail(self) -> str | None:
        """Return why the managed layer is absent from `merged`, if it is.

        Every reader here gates on `user.status.usable` and then merges. None
        inspected the *managed* status, so an unreadable managed layer merged as
        `{}` and the user's own values applied unopposed — with a warning about
        the user's file and nothing about the policy that vanished. The startup
        gate normally prevents an unusable managed snapshot from being observed,
        so this covers the gate-exempt commands and any future entry point that
        forgets it.

        Returns:
            A short reason, or `None` when managed policy is participating.
        """
        if self.managed.status.usable:
            return None
        return self.managed.status.detail or self.managed.status.health.value

    def merged(self) -> tuple[dict[str, Any], dict[str, str]]:
        """Return a deep merge where managed leaves outrank user leaves.

        Lists at `UNION_PATHS` accumulate instead of being replaced. A managed
        scalar replaces a colliding user table whatever its depth, but only
        when the value matches the type the manifest declares; a wrong-typed
        managed value leaves the user value in place.

        Returns:
            Merged table and dotted leaf-to-source mapping.
        """
        return merge_managed_over_user(self.user.data, self.managed.data)


def is_valid_managed_scalar(path: tuple[str, ...], value: object) -> bool:
    """Return whether a managed scalar has the declared type for its path.

    Unknown and structured paths retain the deep-merger's existing behavior;
    their dedicated readers own validation. Manifest-backed scalar leaves are
    validated before they displace a lower-precedence value.
    """
    from deepagents_code.config_manifest import (
        OptionKind,
        option_accepts_toml,
        option_for_toml_path,
    )

    option = option_for_toml_path(path)
    if option is None or option.kind is OptionKind.STRUCTURED:
        return True
    return option_accepts_toml(option, value, source="managed config")


ENFORCED_MANAGED_KEYS = (
    "interpreter.enable_interpreter",
    "interpreter.ptc",
    "interpreter.ptc_acknowledge_unsafe",
    "models.allowed",
    "models.auto_classifier",
    "runtime.recursion_limit",
    "sandboxes.default",
    "shell.allow_list",
    "skills.extra_allowed_dirs",
    "startup.mode",
    "startup.yolo_switcher",
    "tracing.langsmith_redact",
)
"""Manifest keys whose managed value must never resolve in the user's favor.

Every key here either grants a privilege (approval mode, the YOLO entry in the
Shift+Tab cycle, shell auto-approval, the interpreter, its programmatic
tool-calling list, and the acknowledgement that exposes every tool to it) or
draws a containment boundary (the skill-content allowlist, the recursion limit,
the classifier that reviews gated actions, and trace redaction). Ignoring an
unusable value for one of these leaves the user's own flag or environment
variable in force, which is the escalation the policy meant to forbid, so the
launch stops instead.

`sandboxes.default` is enforced for its *value*, not as a containment boundary.
It names which backend a bare `--sandbox` selects; it does not decide whether to
sandbox, so a launch that asks for no sandbox still runs on the host. Enforcement
here rejects a value that cannot be applied, and `_apply_managed_sandbox` prints
a notice when a launch bypasses the named backend, so the gap is visible rather
than assumed away.

`startup.mode` and `startup.yolo_switcher` must both be listed. Pinning the
approval mode while leaving the YOLO entry unenforced keeps unrestricted mode one
keypress away in the same file.

Keys that cannot grant privilege keep the ordinary ignore-and-fall-through rule.
`test_every_enforced_managed_key_resolves_to_a_manifest_option` pins each entry
to a manifest option, because `managed_policy_violations` skips a key it cannot
resolve, so a rename would turn enforcement into a silent no-op.
"""


def managed_declaration(
    managed_data: dict[str, Any], toml_keys: tuple[str, ...]
) -> Literal["declared", "shadowed"] | None:
    """Classify what managed policy says at one manifest path.

    Returns:
        `"declared"` when a value is present, `"shadowed"` when an ancestor is
        not a table (so the key it should hold is unreachable), or `None` when
        the administrator wrote nothing at this path.
    """
    node: object = managed_data
    for part in toml_keys[:-1]:
        if not isinstance(node, dict):
            return "shadowed"
        if part not in node:
            return None
        node = node[part]
    if not isinstance(node, dict):
        # A scalar where the parent table belongs, e.g. `startup = "manual"`
        # instead of `[startup]` + `mode = "manual"`.
        return "shadowed"
    return "declared" if toml_keys[-1] in node else None


def managed_policy_violations(
    managed_data: dict[str, Any],
    *,
    status: ProviderStatus | None = None,
) -> tuple[str, ...]:
    """Return managed settings whose declaration cannot be safely applied.

    A key is a violation when an enforced managed policy declaration cannot be
    applied, when a known managed section has a non-table value, or when a
    managed `[models]` default, recent, or auto-classifier value contradicts a
    managed `models.allowed` list. The shape cases matter because "wrong shape"
    is not "absent": merging such a value can erase a user subtree before a
    reader falls back to a default.

    That last case is the only one that reports a key which is not itself in
    `ENFORCED_MANAGED_KEYS` (`models.default`, `models.recent`): the value is
    individually valid and merely inconsistent with the administrator's own
    ceiling, which would otherwise start a session whose pinned model the same
    policy forbids.

    Required rather than defaulted to the process snapshot: `managed_health`
    pairs this with the health of the *same* snapshot, and a default that
    silently read the cache is what let a refreshed status be reported next to
    stale violations.

    Args:
        managed_data: Managed table to inspect. Must come from a snapshot whose
            status is `usable`, since an unhealthy snapshot carries `{}` and
            would report no violations.
        status: Health and display metadata for the same managed snapshot.

    Returns:
        The violating keys, sorted, empty when policy is enforceable.
    """
    from deepagents_code.config_manifest import get_option, is_valid_recursion_limit
    from deepagents_code.configuration.resolver import MANAGED_RANK
    from deepagents_code.configuration.types import Found, Invalid

    if not managed_data:
        return ()

    violations = list(managed_section_shape_violations(managed_data))
    for key in ENFORCED_MANAGED_KEYS:
        resolved = resolve_managed_option(key, managed_data, status=status)
        if resolved is None:
            continue
        result = resolved.tier_health.get(MANAGED_RANK)
        if isinstance(result, Invalid):
            option = get_option(key)
            if (
                option is not None
                and option.toml_keys is not None
                and managed_declaration(managed_data, option.toml_keys) == "declared"
            ):
                # Preserve the existing health-check diagnostic in addition to
                # exposing the same reason structurally through `tier_health`.
                logger.warning("%s", result.reason)
            violations.append(key)
            continue
        if not isinstance(result, Found):
            continue
        if key == "runtime.recursion_limit" and not is_valid_recursion_limit(
            result.value
        ):
            # Provider coercion applies no range ceiling, and the flag outranks
            # the bounded resolver when the agent is built, so an out-of-range
            # managed value would otherwise be assigned verbatim.
            violations.append(key)

    allowed_option = get_option("models.allowed")
    if (
        allowed_option is not None
        and managed_declaration(managed_data, allowed_option.toml_keys or ())
        == "declared"
    ):
        allowed_resolved = resolve_managed_option(
            "models.allowed", managed_data, status=status
        )
        allowed_result = (
            allowed_resolved.tier_health.get(MANAGED_RANK)
            if allowed_resolved is not None
            else None
        )
        if isinstance(allowed_result, Found) and isinstance(
            allowed_result.value, tuple
        ):
            allowed_models = {
                entry for entry in allowed_result.value if isinstance(entry, str)
            }
            models = managed_data.get("models")
            if isinstance(models, dict):
                for field in ("default", "recent", "auto_classifier"):
                    candidate = models.get(field)
                    if not isinstance(candidate, str) or not candidate.strip():
                        continue
                    normalized = candidate.strip()
                    provider, separator, _model = normalized.partition(":")
                    if normalized not in allowed_models and (
                        not separator or f"{provider}:*" not in allowed_models
                    ):
                        violations.append(f"models.{field}")
    return tuple(sorted(set(violations)))


def resolve_managed_option(
    key: str,
    managed_data: dict[str, Any],
    *,
    status: ProviderStatus | None = None,
) -> ResolvedValue[object] | None:
    """Resolve one manifest option and retain its rank-keyed managed result.

    Returns:
        The ranked resolution, or `None` when `key` is not manifest-backed.
    """
    from deepagents_code.config_manifest import get_option
    from deepagents_code.configuration.resolver import resolver_from_snapshots

    option = get_option(key)
    if option is None:
        return None
    # The caller is inspecting a specific managed generation — often a
    # candidate being validated before it takes force — so resolution must not
    # read the process-wide snapshots behind the shared resolver.
    return resolver_from_snapshots(
        managed=TomlSnapshot(
            managed_data,
            status or ProviderStatus("managed config", None, ProviderHealth.OK),
        ),
        user=TomlSnapshot.declaring_nothing("config.toml"),
    ).get(option)


def managed_rejections(managed_data: dict[str, Any]) -> tuple[str, ...]:
    """Return manifest keys managed policy declares whose value was dropped.

    Not a launch failure: only `ENFORCED_MANAGED_KEYS` stops a launch, and every
    other rejected managed value deliberately falls through to the user tier.
    But the fall-through was announced only through `logger.warning`, which the
    package's in-memory log handler makes unreachable from stderr — the handler
    installed at import time means `logging.lastResort` never fires. An
    administrator who typed `max_tokens = "8000"` saw a clean green table and
    had no way to learn the value was ignored.

    Args:
        managed_data: Managed table to inspect. Must come from a snapshot whose
            status is `usable`.

    Returns:
        The rejected keys, sorted, empty when every declared value is readable.
    """
    from deepagents_code.config_manifest import OptionKind, get_config_options
    from deepagents_code.configuration.providers import ranked_toml_value
    from deepagents_code.configuration.resolver import MANAGED_RANK
    from deepagents_code.configuration.types import Invalid

    if not managed_data:
        return ()
    rejected: list[str] = []
    for option in get_config_options():
        if not option.toml_keys or option.kind is OptionKind.STRUCTURED:
            # Structured options own their own validation in typed readers, and
            # report through those readers' own diagnostics.
            continue
        if managed_declaration(managed_data, option.toml_keys) != "declared":
            continue
        provider = ranked_toml_value(
            option,
            managed_data,
            rank=MANAGED_RANK,
            durable=True,
            status=ProviderStatus(MANAGED_SOURCE, None, ProviderHealth.OK),
        )
        if isinstance(provider.result, Invalid):
            # Preserve the diagnostics contract: this inspection historically
            # announced the raw declaration that policy rejected.
            logger.warning("%s", provider.result.reason)
            rejected.append(option.key)
    return tuple(sorted(rejected))


def managed_section_shape_violations(
    managed_data: dict[str, Any],
) -> tuple[str, ...]:
    """Return known managed sections declared as non-table values.

    Unknown leaf keys remain forward compatible. Every parent path of a
    manifest-backed option is known to be a table, however, as are the
    structured options that specifically represent tables. Rejecting a scalar
    at one of those paths prevents it from replacing an entire user section.
    """
    violations: list[str] = []
    for path in _managed_table_paths():
        node: object = managed_data
        for part in path:
            if not isinstance(node, dict) or part not in node:
                break
            node = node[part]
        else:
            if not isinstance(node, dict):
                violations.append(".".join(path))
    return tuple(sorted(violations))


@lru_cache(maxsize=1)
def _managed_table_paths() -> frozenset[tuple[str, ...]]:
    """Return every managed path whose value must be a table.

    A pure function of `get_config_options`, which is itself cached, so the
    derived set is cached too: this runs at least twice on the startup path and
    six times per launch, and rebuilt a nested comprehension over 89+ options
    every time. Tests that monkeypatch the registry already call
    `get_config_options.cache_clear`, and must clear this alongside it.

    Returns:
        Declared table sections plus every manifest option's parent paths.
    """
    from deepagents_code.config_manifest import get_config_options

    table_paths: set[tuple[str, ...]] = set(MANAGED_TABLE_PATHS)
    for option in get_config_options():
        if option.toml_keys:
            table_paths.update(
                option.toml_keys[:depth] for depth in range(1, len(option.toml_keys))
            )
    return frozenset(table_paths)


class ManagedConfigError(RuntimeError):
    """Raised when an enforced managed source cannot be read safely."""

    def __init__(self, status: ProviderStatus, message: str | None = None) -> None:
        """Build a safe startup error from provider health metadata."""
        if message is None:
            path = status.path or managed_config_path()
            detail = f": {status.detail}" if status.detail else ""
            if status.health is ProviderHealth.INDETERMINATE:
                # The file was never the problem — the location is unknown, so
                # "repair or remove the file" would send the reader to a path
                # that may hold nothing on a correctly configured host.
                message = (
                    f"Managed config location could not be determined{detail}. "
                    "Ask your administrator to verify the managed-config path."
                )
            else:
                message = (
                    f"Managed config at {path} is {status.health.value}{detail}. "
                    "Ask your administrator to repair or remove the file."
                )
        super().__init__(message)
        self.status = status


class ManagedPolicyError(ManagedConfigError):
    """Raised when managed policy declares a value that cannot be enforced.

    The file parses, so provider health is `OK`; the policy it states is what
    cannot be applied. A subclass of `ManagedConfigError` so every caller that
    already fails closed on an unreadable file fails closed here too.
    """

    def __init__(self, status: ProviderStatus, keys: tuple[str, ...]) -> None:
        """Build a startup error naming the keys that stop the launch."""
        path = status.path or managed_config_path()
        super().__init__(
            status,
            f"Managed config at {path} rejects {', '.join(keys)}. "
            "Ask your administrator to correct the value.",
        )
        self.keys = keys


class _SnapshotState:
    """Mutable process snapshot guarded by `_snapshot_lock`."""

    __slots__ = ("managed",)

    def __init__(self) -> None:
        """Start with no cached snapshot."""
        self.managed: TomlSnapshot | None = None


_snapshot_lock = threading.RLock()
_snapshot_state = _SnapshotState()


def _load_managed(path: Path | None = None) -> TomlSnapshot:
    """Load the managed provider without applying startup policy.

    An explicit `path` is read as given: it is a test-and-tooling operation on
    one named file, so the OS path resolution — and any reason that path would
    be a guess — does not apply to it.

    Returns:
        Parsed managed snapshot and health.
    """
    if path is not None:
        return TomlFileProvider("managed config", path).load()
    resolved = resolve_managed_path()
    snapshot = TomlFileProvider("managed config", resolved.path).load()
    is_guess = resolved.fallback is not None
    if not is_guess or snapshot.status.health is not ProviderHealth.MISSING:
        return snapshot
    # "No file at a guessed path" is not "no policy deployed", so this is not a
    # clean `MISSING`. `INDETERMINATE` is not usable, which stops the launch
    # instead of letting every reader see an empty managed table and treat
    # unreadable policy as absent policy.
    return TomlSnapshot(
        snapshot.data,
        replace(
            snapshot.status,
            health=ProviderHealth.INDETERMINATE,
            detail=resolved.fallback,
        ),
    )


def get_managed_snapshot(
    *, refresh: bool = False, path: Path | None = None
) -> TomlSnapshot:
    """Return the process snapshot, or an isolated snapshot for an explicit path.

    A reload that cannot parse the file never evicts policy that parsed
    cleanly earlier. An unusable snapshot carries `data == {}`, which every
    reader would otherwise treat as "nothing is enforced", so caching it would
    turn one broken write by an administrator into a process-wide fail-open.
    The caller still receives the failed load, so health checks see the error.

    The same holds for a file that parses but cannot be enforced. Its health
    is `OK`, so a usability check alone would cache it. When the refresh
    raises `ManagedPolicyError` (as `require_healthy_managed_config` does),
    the rejected candidate must not stay in the cache: the reload keeps the
    previous settings, but a later non-refresh reader would otherwise observe
    the rejected snapshot and, for example, re-enable a managed MCP deny the
    edit removed. Validate enforceability before caching, so the cache holds
    only the last enforceable snapshot.

    Returns:
        The cached snapshot, or the freshly loaded one when refreshing.
    """
    if path is not None:
        return _load_managed(path)
    with _snapshot_lock:
        cached = _snapshot_state.managed
        if not refresh and cached is not None:
            return cached
        candidate = _load_managed()
        # Cache only a snapshot whose declared policy can actually be enforced.
        # `usable` admits a parseable-but-unenforceable file, so gate on the
        # policy check too, not just provider health.
        if candidate.status.usable and not managed_policy_violations(
            candidate.data, status=candidate.status
        ):
            _snapshot_state.managed = candidate
        return candidate


def get_config_sources(
    *,
    user_path: Path | None = None,
    managed_path: Path | None = None,
) -> ConfigSources:
    """Load one user snapshot and the current managed snapshot.

    Managed policy is included exactly when `user_path` is `None`, which is
    what every production caller passes. Reading an explicit path is a
    test-and-tooling operation on one file, and its result must not be mistaken
    for the effective configuration.

    Deliberately not a caller-supplied flag: an `include_managed=False` source
    is indistinguishable from a machine with no policy installed, because the
    fabricated status reports `MISSING` and the table is empty. Deriving it here
    keeps that state one keyword out of reach.

    Args:
        user_path: Read this file as the user layer instead of the default, and
            exclude managed policy. Intended for tests and for tooling that
            inspects one file.
        managed_path: Read managed policy from this file instead of the fixed
            OS path, bypassing the process snapshot. Intended for tests.

    Returns:
        Both snapshots from one resolution generation.
    """
    if user_path is not None:
        return ConfigSources(
            managed=TomlSnapshot(
                {},
                ProviderStatus(
                    "managed config",
                    managed_path,
                    ProviderHealth.MISSING,
                ),
            ),
            user=TomlFileProvider("config.toml", user_path).load(),
        )
    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    return ConfigSources(
        managed=get_managed_snapshot(path=managed_path),
        user=TomlFileProvider("config.toml", DEFAULT_CONFIG_PATH).load(),
    )


def invalidate_config_sources() -> None:
    """Drop the cached managed snapshot and the shared process resolver.

    Test-only. Production reloads pass `refresh=True` instead, which keeps the
    last snapshot that parsed cleanly if the new one fails; clearing the cache
    first would leave readers with an empty managed table on a failed reload.

    Both caches are cleared together because they are keyed differently:
    dropping only the managed snapshot leaves the resolver holding the previous
    generation, which is the half most tests actually read.
    """
    from deepagents_code.configuration.resolver import reset_config_resolver

    with _snapshot_lock:
        _snapshot_state.managed = None
    reset_config_resolver()


def get_healthy_managed_snapshot(*, refresh: bool = False) -> TomlSnapshot:
    """Return managed policy only when it can be enforced.

    A file that parses is not necessarily enforceable: a privilege-affecting
    key can carry a value the manifest rejects, or a known section can be a
    scalar instead of a table. Both can otherwise resolve in the user's favor
    or erase a user subtree, so they stop the launch here rather than at each
    consumer.

    Returns:
        The exact managed snapshot that passed validation.

    Raises:
        ManagedConfigError: If managed policy is present but unusable.
        ManagedPolicyError: If managed policy declares an unenforceable key or
            malformed known section.
    """
    snapshot = get_managed_snapshot(refresh=refresh)
    status = snapshot.status
    if not status.usable:
        raise ManagedConfigError(status)
    violations = managed_policy_violations(snapshot.data, status=status)
    if violations:
        raise ManagedPolicyError(status, violations)
    return snapshot


def require_healthy_managed_config(*, refresh: bool = False) -> None:
    """Fail startup when present managed policy cannot be parsed or enforced.

    Propagates from `get_healthy_managed_snapshot`: `ManagedConfigError` when
    a managed file is present but unreadable, and `ManagedPolicyError` when it
    parses but declares policy that cannot be enforced. Both fail startup at
    every call site.

    Args:
        refresh: Re-read the managed file before checking it.
    """
    get_healthy_managed_snapshot(refresh=refresh)


def _managed_resolver(snapshot: TomlSnapshot) -> ConfigResolver:
    """Build a resolver whose managed provider owns `snapshot`.

    Returns:
        Resolver bound to the supplied managed generation.
    """
    from deepagents_code.configuration.resolver import resolver_from_snapshots

    return resolver_from_snapshots(
        managed=snapshot,
        user=TomlSnapshot.absent("config.toml"),
    )


def managed_config_status(*, refresh: bool = False) -> ProviderStatus:
    """Return managed provider health for diagnostics and config inspection."""
    from deepagents_code.configuration.resolver import MANAGED_RANK

    snapshot = get_managed_snapshot(refresh=refresh)
    return _managed_resolver(snapshot).provider_statuses()[MANAGED_RANK]


@dataclass(frozen=True, slots=True)
class ManagedHealth:
    """Provider health and policy enforceability from one managed snapshot."""

    status: ProviderStatus
    violations: tuple[str, ...]
    rejections: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether managed policy is both readable and enforceable.

        Rejections are excluded on purpose: they are values policy declared and
        the runtime ignored, which is the documented behavior for a key that
        cannot grant a privilege. They still have to be *reported*, so they are
        carried here rather than folded into the verdict.
        """
        return self.status.usable and not self.violations


def managed_health(*, refresh: bool = False) -> ManagedHealth:
    """Return both halves of exit 78 for one managed snapshot.

    Reading health and violations as two calls is a live bug, not a style
    choice. `get_managed_snapshot` declines to cache a candidate it cannot
    enforce, so a refreshed status describes the file on disk while a second,
    non-refreshed violation read still sees the last enforceable snapshot and
    reports none. Every diagnostic surface then shows `ok` for the exact file
    that just refused to launch. One snapshot, both answers.

    Returns:
        Health, violations, and ignored rejections that cannot disagree.
    """
    from deepagents_code.configuration.resolver import MANAGED_RANK

    snapshot = get_managed_snapshot(refresh=refresh)
    status = _managed_resolver(snapshot).provider_statuses()[MANAGED_RANK]
    return managed_snapshot_health(replace(snapshot, status=status))


def managed_snapshot_health(snapshot: TomlSnapshot) -> ManagedHealth:
    """Evaluate provider health and policy diagnostics for one snapshot.

    Args:
        snapshot: Managed provider generation to inspect.

    Returns:
        Health, violations, and ignored rejections from exactly `snapshot`.
    """
    if not snapshot.status.usable:
        return ManagedHealth(snapshot.status, (), ())
    violations = managed_policy_violations(snapshot.data, status=snapshot.status)
    # A key that stops the launch is not also "ignored": reporting it in both
    # lists made `doctor` print "rejects startup.mode - ignores startup.mode".
    rejections = tuple(
        key for key in managed_rejections(snapshot.data) if key not in violations
    )
    return ManagedHealth(snapshot.status, violations, rejections)
