"""Pure ranked resolution and deep-merge logic for layered configuration.

The ranked engine is intentionally unaware of the manifest, UI, model, theme,
environment, or filesystem. Providers coerce their own domains before handing
`Found`, `Unset`, or `Invalid` results to this module. Human-readable source
labels likewise remain in `ProviderStatus`; provenance and health here use only
numeric ranks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

from deepagents_code.configuration.types import (
    Found,
    ProviderResult,
    ProviderStatus,
)

MANAGED_RANK = 200
"""Managed policy rank; lower numeric ranks have stronger precedence."""

CLI_RANK = 300
"""Reserved seam for a future CLI provider; no CLI provider ships today."""

ENVIRONMENT_RANK = 400
"""Process-environment rank."""

USER_RANK = 500
"""User `config.toml` rank."""

DEFAULT_RANK = 1000
"""Typed manifest-default rank."""


@dataclass(frozen=True, slots=True)
class RankedProviderValue[T]:
    """One provider's already-coerced result for an option."""

    rank: int
    durable: bool
    status: ProviderStatus
    result: ProviderResult[T]
    diagnostics: tuple[str, ...] = ()
    """Ordered warnings encountered while trying aliases inside this tier."""


@dataclass(frozen=True, slots=True)
class ResolvedValue[T]:
    """Resolved value with rank-keyed provenance and provider health."""

    value: T
    provenance: Mapping[int, frozenset[tuple[str, ...]]]
    tier_health: Mapping[int, ProviderResult[T]]
    provider_status: Mapping[int, ProviderStatus]
    masked_ranks: frozenset[int] = frozenset()
    selected_ranks: tuple[int, ...] = ()
    tier_diagnostics: Mapping[int, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def ranks(self) -> tuple[int, ...]:
        """Contributing ranks in precedence order."""
        return self.selected_ranks or tuple(sorted(self.provenance))


def resolve_ranked[T](
    providers: Sequence[RankedProviderValue[T]],
    *,
    strategy: str = "replace",
) -> ResolvedValue[T] | None:
    """Resolve provider results by numeric rank and per-option merge strategy.

    Lower ranks win. For replacement options, a `Found` from a durable tier
    masks lower-precedence non-durable tiers. The mask is intentionally
    directional: a persisted user value at rank 500 cannot retroactively hide
    a higher-precedence environment value at rank 400.

    Accumulating strategies combine tiers by definition, so they retain every
    valid contribution. This preserves the existing fail-closed deny-list
    unions and deep TOML composition; treating accumulation as replacement
    would silently discard restrictions or sibling table leaves.

    Args:
        providers: Already-coerced provider results. Ranks must be unique.
        strategy: `replace`, `union`, or `deep_merge`.

    Returns:
        A resolved value, or `None` when no provider returned `Found`.

    Raises:
        ValueError: If ranks repeat or `strategy` is unknown.
    """
    ordered = sorted(providers, key=lambda provider: provider.rank)
    ranks = [provider.rank for provider in ordered]
    if len(set(ranks)) != len(ranks):
        msg = "ranked config providers must have unique ranks"
        raise ValueError(msg)
    if strategy not in {"replace", "union", "deep_merge"}:
        msg = f"unknown config merge strategy: {strategy}"
        raise ValueError(msg)

    tier_health = MappingProxyType(
        {provider.rank: provider.result for provider in ordered}
    )
    provider_status = MappingProxyType(
        {provider.rank: provider.status for provider in ordered}
    )
    tier_diagnostics = MappingProxyType(
        {provider.rank: provider.diagnostics for provider in ordered}
    )
    found = [provider for provider in ordered if isinstance(provider.result, Found)]
    if not found:
        return None
    if strategy == "union":
        return _resolve_ranked_union(
            found,
            tier_health,
            provider_status,
            tier_diagnostics,
        )
    if strategy == "deep_merge":
        return _resolve_ranked_deep_merge(
            found,
            tier_health,
            provider_status,
            tier_diagnostics,
        )

    durable_ranks = tuple(provider.rank for provider in found if provider.durable)
    masked = frozenset(
        provider.rank
        for provider in found
        if not provider.durable
        and any(durable_rank < provider.rank for durable_rank in durable_ranks)
    )
    winner = next(provider for provider in found if provider.rank not in masked)
    return ResolvedValue(
        _provider_value(winner),
        MappingProxyType({winner.rank: frozenset({()})}),
        tier_health,
        provider_status,
        masked,
        (winner.rank,),
        tier_diagnostics,
    )


def _replace_with_strongest[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Resolve to the strongest-precedence provider when accumulation fails.

    The value is copied. Provider values alias the process-wide managed
    snapshot, so handing out a live reference would let a consumer mutate
    administrator policy for the rest of the session.

    Returns:
        The lowest-rank provider's value, deep-copied.
    """
    winner = found[0]
    return ResolvedValue(
        deepcopy(_provider_value(winner)),
        MappingProxyType({winner.rank: frozenset({()})}),
        tier_health,
        provider_status,
        selected_ranks=(winner.rank,),
        tier_diagnostics=tier_diagnostics,
    )


def _resolve_ranked_union[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Accumulate list-like providers from weakest to strongest rank.

    Returns:
        The union, or the strongest-precedence (lowest-rank) replacement when a
        value is not list-like.
    """
    entries = [union_entries(_provider_value(provider)) for provider in found]
    if any(value is None for value in entries):
        return _replace_with_strongest(
            found, tier_health, provider_status, tier_diagnostics
        )
    union: list[Any] = []
    for value in reversed(entries):
        union = union_lists(union, cast("list[Any]", value))
    provenance = MappingProxyType(
        {provider.rank: frozenset({()}) for provider in found}
    )
    return ResolvedValue(
        cast("T", union),
        provenance,
        tier_health,
        provider_status,
        selected_ranks=tuple(provider.rank for provider in found),
        tier_diagnostics=tier_diagnostics,
    )


def _resolve_ranked_deep_merge[T](
    found: Sequence[RankedProviderValue[T]],
    tier_health: Mapping[int, ProviderResult[T]],
    provider_status: Mapping[int, ProviderStatus],
    tier_diagnostics: Mapping[int, tuple[str, ...]],
) -> ResolvedValue[T]:
    """Deep-merge mapping providers from weakest to strongest rank.

    A tier that does not hold a mapping cannot be merged. Such a tier falls
    back to replacement by the strongest-precedence (lowest-rank) provider,
    matching `_resolve_ranked_union`. Returning the non-mapping tier itself
    would let a weaker tier displace managed policy.

    Returns:
        The merged mapping, or the strongest provider's value when any tier
        cannot be merged.
    """
    weakest = found[-1]
    value = _provider_value(weakest)
    if not isinstance(value, dict):
        return _replace_with_strongest(
            found, tier_health, provider_status, tier_diagnostics
        )
    merged = deepcopy(cast("dict[str, Any]", value))
    leaves = _ranked_leaf_provenance(merged, weakest.rank)
    for provider in reversed(found[:-1]):
        higher = _provider_value(provider)
        if not isinstance(higher, dict):
            return _replace_with_strongest(
                found, tier_health, provider_status, tier_diagnostics
            )
        merged, leaves = _merge_ranked_tables(
            merged,
            cast("dict[str, Any]", higher),
            leaves,
            provider.rank,
        )
    grouped: dict[int, set[tuple[str, ...]]] = {}
    for path, rank in leaves.items():
        grouped.setdefault(rank, set()).add(path)
    provenance = MappingProxyType(
        {rank: frozenset(paths) for rank, paths in grouped.items()}
    )
    return ResolvedValue(
        cast("T", merged),
        provenance,
        tier_health,
        provider_status,
        selected_ranks=tuple(provider.rank for provider in found),
        tier_diagnostics=tier_diagnostics,
    )


def _merge_ranked_tables(
    lower: dict[str, Any],
    higher: dict[str, Any],
    provenance: dict[tuple[str, ...], int],
    higher_rank: int,
    *,
    prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[tuple[str, ...], int]]:
    """Deep-merge two mappings while retaining tuple-path rank provenance.

    Returns:
        The merged table and tuple-path-to-rank provenance.
    """
    merged = deepcopy(lower)
    ranked = dict(provenance)
    for key, value in higher.items():
        path = (*prefix, key)
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key], ranked = _merge_ranked_tables(
                cast("dict[str, Any]", existing),
                cast("dict[str, Any]", value),
                ranked,
                higher_rank,
                prefix=path,
            )
            continue
        merged[key] = deepcopy(value)
        for leaf in tuple(ranked):
            if leaf[: len(path)] == path:
                ranked.pop(leaf)
        ranked.update(_ranked_leaf_provenance(value, higher_rank, path))
    return merged, ranked


def _ranked_leaf_provenance(
    value: object, rank: int, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], int]:
    """Attribute every leaf under `value` to a numeric provider rank.

    Returns:
        Tuple-path-to-rank provenance for every leaf.
    """
    if isinstance(value, dict):
        if not value:
            return {path: rank} if path else {}
        result: dict[tuple[str, ...], int] = {}
        for key, child in cast("dict[str, object]", value).items():
            result.update(_ranked_leaf_provenance(child, rank, (*path, key)))
        return result
    return {path: rank}


def _provider_value[T](provider: RankedProviderValue[T]) -> T:
    """Narrow a provider known by the resolver to hold `Found`.

    Returns:
        The provider's coerced value.

    Raises:
        RuntimeError: If an internal accumulating resolver receives a non-found tier.
    """
    result = provider.result
    if isinstance(result, Found):
        return cast("T", result.value)
    msg = f"rank {provider.rank} did not contain a found value"
    raise RuntimeError(msg)


def union_lists(lower: list[Any], higher: list[Any]) -> list[Any]:
    """Accumulate two deny-list layers, keeping order and dropping duplicates.

    Shared with the merger so a deny list cannot union in one reader and
    replace in another.

    Returns:
        The lower list followed by the higher entries it does not already hold.
    """
    union = deepcopy(lower)
    for item in higher:
        if item not in union:
            union.append(deepcopy(item))
    return union


def union_entries(value: object) -> list[Any] | None:
    """Normalize one deny-list layer to its entries.

    A deny list may be written as a TOML array or as a comma-separated string
    (`disabled_servers = "a, b"`), and the runtime readers treat the two as
    equivalent — `mcp_disabled._strict_entries` and `model_config._toml_str_list`
    both split on commas. The merge has to accept both spellings too. It did
    not, so a managed string layer was dropped in favor of the user's array and
    the provenance then credited the user's file for a leaf managed policy
    contributes to.

    Returns:
        The trimmed entries, or `None` when the value cannot hold entries.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return value
    return None


def merge_toml_tables(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]] = frozenset(),
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge TOML tables with higher-precedence leaf provenance.

    Args:
        lower: Lower-precedence table.
        higher: Higher-precedence table, whose leaves win.
        lower_source: Source label recorded for surviving `lower` leaves.
        higher_source: Source label recorded for surviving `higher` leaves.
        union_paths: Paths whose lists accumulate instead of being replaced.
            Deny lists must union, because replacing one would be a fail-open.
            Paths match relative to the tables passed here, so a merge of one
            subtree needs them rebased (see `service.union_paths_under`).
        higher_leaf_is_valid: Optional check applied to a `higher` value before
            it displaces a `lower` one. Return `False` to keep the lower value,
            which stops a wrong-typed higher value from discarding a valid
            lower subtree. Receives paths on the same relative basis as
            `union_paths`. Every managed merge passes one, by way of
            `service.merge_managed_over_user`; omitting it leaves the merger
            with no type information, so it displaces only a table that holds
            no nested table.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    merged, provenance = _merge(
        lower,
        higher,
        lower_source=lower_source,
        higher_source=higher_source,
        union_paths=union_paths,
        higher_leaf_is_valid=higher_leaf_is_valid,
    )
    return merged, _dotted(_drop_ancestor_entries(provenance))


def _dotted(provenance: dict[tuple[str, ...], str]) -> dict[str, str]:
    """Join tuple paths for display.

    Provenance is keyed by path tuple everywhere inside this module. TOML allows
    a quoted key that contains dots (`"a.b" = 1` parses to the single key
    `a.b`), so a dotted string is a lossy key: it made `_drop_ancestor_entries`
    delete a live sibling leaf named `a`, and credited the wrong tier for the
    flat key. Joining happens once, here, where the ambiguity is only cosmetic.

    Returns:
        Provenance keyed by dotted path.
    """
    return {".".join(path): source for path, source in provenance.items()}


def _drop_ancestor_entries(
    provenance: dict[tuple[str, ...], str],
) -> dict[tuple[str, ...], str]:
    """Remove entries that are a strict ancestor of another entry.

    A lower empty table that the higher table fills leaves an entry for the
    table itself: it enters the recursion through `lower_provenance`, which
    carries the parent's own path, and the level that fills it never removes it.
    The result claimed a table was a user-controlled leaf alongside the managed
    leaves inside it. A path cannot be both a leaf and a parent, so the ancestor
    is always the stale one.

    Returns:
        Provenance with only leaf entries.
    """
    keys = tuple(provenance)
    return {
        path: source
        for path, source in provenance.items()
        if not any(other[: len(path)] == path and other != path for other in keys)
    }


def _merge(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]],
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None,
    lower_provenance: dict[tuple[str, ...], str] | None = None,
    path_prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    """Recursive half of `merge_toml_tables`.

    Separate so the public signature carries no parameter a caller must not
    pass: `lower_provenance` has to arrive already scoped to `path_prefix`, and
    an unscoped mapping produces wrong provenance with no error.

    Returns:
        Merged table and path-keyed leaf-to-source mapping.
    """
    merged: dict[str, Any] = deepcopy(dict(lower))
    provenance = dict(
        lower_provenance or _leaf_provenance(lower, lower_source, path_prefix)
    )
    for key, value in higher.items():
        path = (*path_prefix, key)
        existing = merged.get(key)
        # A higher scalar must replace a lower table, whatever the table holds.
        # Keeping the table lets a shape collision defeat the higher value.
        # Typed readers then reject the table and use the built-in default.
        #   Example: a user `[threads.relative_time]` table against a managed
        #   `relative_time = false`.
        # Depth is not consulted, so deeper nesting cannot restore the bypass.
        # With `higher_leaf_is_valid`, the check below gates the replacement.
        # That keeps a wrong-typed higher scalar from discarding a valid lower
        # subtree. Without a validator there is no type information here, so
        # only a table that holds no nested table is displaced.
        if (
            isinstance(existing, dict)
            and not isinstance(value, dict)
            and higher_leaf_is_valid is None
            and not _overriding_table_is_scalar_only(existing)
        ):
            continue
        # Validate every managed value at a manifest-backed scalar path,
        # including TOML tables. A table cannot be passed to the validator as
        # a leaf through the recursive branch below, so validating only
        # non-dicts would let `[models.default]` replace a valid string with a
        # dictionary that later runtime readers cannot use.
        if higher_leaf_is_valid is not None and not higher_leaf_is_valid(path, value):
            continue
        if path in union_paths:
            lower_entries = union_entries(existing)
            higher_entries = union_entries(value)
            if lower_entries is not None and higher_entries is None:
                # A higher value that cannot hold names must never replace a
                # deny list: that would drop the lower layer's denials.
                continue
            if lower_entries is not None and higher_entries is not None:
                merged[key] = union_lists(lower_entries, higher_entries)
                provenance[path] = _combined_source(lower_source, higher_source)
                continue
        if isinstance(existing, dict) and isinstance(value, dict):
            nested, nested_provenance = _merge(
                existing,
                value,
                lower_source=lower_source,
                higher_source=higher_source,
                union_paths=union_paths,
                higher_leaf_is_valid=higher_leaf_is_valid,
                lower_provenance={
                    leaf: source
                    for leaf, source in provenance.items()
                    if leaf[: len(path)] == path
                },
                path_prefix=path,
            )
            merged[key] = nested
            # Drop this subtree's old leaves first. A nested merge can delete a
            # leaf (a higher scalar replacing a lower table), and keeping the
            # parent-scope entry would report a path that no longer exists as
            # user-controlled — in the output an administrator reads to audit
            # what policy enforces.
            for leaf in tuple(provenance):
                if leaf[: len(path)] == path:
                    provenance.pop(leaf)
            provenance.update(nested_provenance)
            continue
        merged[key] = deepcopy(value)
        for leaf in tuple(provenance):
            if leaf[: len(path)] == path:
                provenance.pop(leaf)
        provenance.update(_leaf_provenance(value, higher_source, path))
    return merged, provenance


def _overriding_table_is_scalar_only(table: dict[str, Any]) -> bool:
    """Return `True` when `table` holds no non-empty nested tables at any depth.

    Only direct children need checking: a nested table at any depth makes its
    own parent chain non-empty, so an empty direct child cannot hide one.
    Empty nested tables carry no lower values worth preserving, so they do not
    stop a higher-precedence scalar from replacing the table.
    """
    for child in cast("dict[str, object]", table).values():
        if isinstance(child, dict) and child:
            return False
    return True


def _leaf_provenance(
    value: object, source: str, path: tuple[str, ...]
) -> dict[tuple[str, ...], str]:
    """Return provenance entries for every leaf under `value`."""
    if isinstance(value, dict):
        if not value:
            # An empty table at the root is not a leaf: it would key the whole
            # mapping. Every merge on a machine with no user `config.toml`
            # produced that entry, in the output an administrator reads to audit
            # what policy enforces.
            if not path:
                return {}
            return {path: source}
        result: dict[tuple[str, ...], str] = {}
        for key, child in cast("dict[str, object]", value).items():
            result.update(_leaf_provenance(child, source, (*path, key)))
        return result
    return {path: source}


def _combined_source(lower: str, higher: str) -> str:
    """Combine distinct source labels in precedence order.

    Returns:
        One source or a higher-plus-lower label.
    """
    if lower == higher:
        return higher
    return f"{higher} + {lower}"
