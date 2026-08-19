"""Pure deep-merge logic for layered TOML configuration.

Precedence itself lives in `service.ConfigSources.merged` and in
`config_manifest.resolve_scalar`; this module only composes two tables once a
caller has decided which one outranks the other.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
            `union_paths`.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    return _merge(
        lower,
        higher,
        lower_source=lower_source,
        higher_source=higher_source,
        union_paths=union_paths,
        higher_leaf_is_valid=higher_leaf_is_valid,
    )


def _merge(
    lower: Mapping[str, Any],
    higher: Mapping[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]],
    higher_leaf_is_valid: Callable[[tuple[str, ...], object], bool] | None,
    lower_provenance: dict[str, str] | None = None,
    path_prefix: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Recursive half of `merge_toml_tables`.

    Separate so the public signature carries no parameter a caller must not
    pass: `lower_provenance` has to arrive already scoped to `path_prefix`, and
    an unscoped mapping produces wrong provenance with no error.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    merged: dict[str, Any] = deepcopy(dict(lower))
    provenance = dict(
        lower_provenance or _leaf_provenance(lower, lower_source, path_prefix)
    )
    for key, value in higher.items():
        path = (*path_prefix, key)
        dotted = ".".join(path)
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
        if (
            path in union_paths
            and isinstance(existing, list)
            and not isinstance(value, list)
        ):
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
                    if leaf == dotted or leaf.startswith(f"{dotted}.")
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
                if leaf == dotted or leaf.startswith(f"{dotted}."):
                    provenance.pop(leaf)
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
