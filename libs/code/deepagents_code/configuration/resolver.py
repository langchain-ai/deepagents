"""Pure deep-merge logic for layered TOML configuration.

Precedence itself lives in `service.ConfigSources.merged` and in
`config_manifest.resolve_scalar`; this module only composes two tables once a
caller has decided which one outranks the other.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def merge_toml_tables(
    lower: dict[str, Any],
    higher: dict[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]] = frozenset(),
    lower_provenance: dict[str, str] | None = None,
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None = None,
    _path: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge TOML tables with higher-precedence leaf provenance.

    Args:
        lower: Lower-precedence table.
        higher: Higher-precedence table, whose leaves win.
        lower_source: Source label recorded for surviving `lower` leaves.
        higher_source: Source label recorded for surviving `higher` leaves.
        union_paths: Paths whose lists accumulate instead of being replaced.
            Deny lists must union, because replacing one would be a fail-open.
        lower_provenance: Provenance for `lower`, already scoped to `_path`.
            Passing an unscoped mapping produces wrong provenance.
        higher_leaf_is_valid: Optional check applied to a `higher` scalar
            before it displaces a `lower` value. Return `False` to keep the
            lower value, which stops a wrong-typed higher scalar from
            discarding a valid lower subtree.
        _path: Internal key path used by the recursion.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    merged = deepcopy(lower)
    provenance = dict(lower_provenance or _leaf_provenance(lower, lower_source, _path))
    for key, value in higher.items():
        path = (*_path, key)
        dotted = ".".join(path)
        existing = merged.get(key)
        # A higher scalar must replace a lower table whatever the table holds:
        # keeping the table lets a shape-colliding lower entry (e.g. a user
        # `[threads.relative_time]` table against a managed
        # `relative_time = false`) defeat the higher value, after which typed
        # readers reject the table and fall back to the built-in default.
        # Nesting the table deeper must not restore that bypass, so depth is
        # not consulted. When the caller supplies `higher_leaf_is_valid`, the
        # replacement is gated on it below, which keeps a wrong-typed higher
        # scalar from discarding valid lower subtrees. Without a validator
        # there is no type information here, so only a table that holds no
        # nested table is displaced.
        if (
            isinstance(existing, dict)
            and not isinstance(value, dict)
            and higher_leaf_is_valid is None
            and not _overriding_table_is_scalar_only(existing)
        ):
            continue
        if (
            path in union_paths
            and isinstance(existing, list)
            and not isinstance(value, list)
        ):
            continue
        if isinstance(existing, dict) and isinstance(value, dict):
            nested, nested_provenance = merge_toml_tables(
                existing,
                value,
                lower_source=lower_source,
                higher_source=higher_source,
                union_paths=union_paths,
                lower_provenance={
                    leaf: source
                    for leaf, source in provenance.items()
                    if leaf == dotted or leaf.startswith(f"{dotted}.")
                },
                higher_leaf_is_valid=higher_leaf_is_valid,
                _path=path,
            )
            merged[key] = nested
            provenance.update(nested_provenance)
            continue
        if (
            path in union_paths
            and isinstance(existing, list)
            and isinstance(value, list)
        ):
            union = deepcopy(existing)
            for item in value:
                if item not in union:
                    union.append(deepcopy(item))
            merged[key] = union
            provenance[dotted] = _combined_source(lower_source, higher_source)
            continue
        if (
            key in merged
            and higher_leaf_is_valid is not None
            and not isinstance(value, dict)
            and not higher_leaf_is_valid(path, value)
        ):
            continue
        merged[key] = deepcopy(value)
        for leaf in tuple(provenance):
            if leaf == dotted or leaf.startswith(f"{dotted}."):
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
) -> dict[str, str]:
    """Return provenance entries for every leaf under `value`."""
    if isinstance(value, dict):
        if not value:
            return {".".join(path): source}
        result: dict[str, str] = {}
        for key, child in cast("dict[str, object]", value).items():
            result.update(_leaf_provenance(child, source, (*path, key)))
        return result
    return {".".join(path): source}


def _combined_source(lower: str, higher: str) -> str:
    """Combine distinct source labels in precedence order.

    Returns:
        One source or a higher-plus-lower label.
    """
    if lower == higher:
        return higher
    return f"{higher} + {lower}"
