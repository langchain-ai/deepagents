"""Pure precedence and merge logic for configuration providers."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from deepagents_code.configuration.providers import ConfigProvider
from deepagents_code.configuration.types import Found, Invalid, ResolvedValue, Unset


class MergeStrategy(StrEnum):
    """How values from multiple providers compose."""

    REPLACE = "replace"
    DEEP_MERGE = "deep_merge"
    UNION = "union"


class ConfigResolver:
    """Resolve ranked provider snapshots without provider-specific masking."""

    def __init__(self, providers: tuple[ConfigProvider[Any], ...]) -> None:
        """Create a resolver ordered by ascending rank."""
        self._providers = tuple(sorted(providers, key=lambda provider: provider.rank))

    def resolve[T](
        self,
        key: str,
        *,
        default: T,
        strategy: MergeStrategy = MergeStrategy.REPLACE,
    ) -> ResolvedValue[T]:
        """Resolve `key` using the requested merge strategy.

        Returns:
            Effective value, source, provenance, and invalid-layer metadata.

        Raises:
            TypeError: If a provider returns an unsupported result type.
        """
        found: list[Found[Any]] = []
        invalid: list[Invalid] = []
        for provider in self._providers:
            result = provider.get(key)
            if isinstance(result, Invalid):
                invalid.append(result)
            elif isinstance(result, Found):
                found.append(result)
            elif not isinstance(result, Unset):
                msg = f"Unsupported provider result: {type(result).__name__}"
                raise TypeError(msg)
        if not found:
            return ResolvedValue(default, "default", invalid=tuple(invalid))
        if strategy is MergeStrategy.REPLACE:
            selected = found[0]
            return ResolvedValue(
                cast("T", selected.value),
                selected.source,
                invalid=tuple(invalid),
            )
        if strategy is MergeStrategy.UNION:
            merged: list[Any] = []
            for item in reversed(found):
                values = item.value
                if not isinstance(values, (list, tuple, set, frozenset)):
                    invalid.append(
                        Invalid(item.source, "union values must be sequences")
                    )
                    continue
                for value in values:
                    if value not in merged:
                        merged.append(value)
            sources = " + ".join(item.source for item in found)
            return ResolvedValue(cast("T", merged), sources, invalid=tuple(invalid))
        merged_value: dict[str, Any] = {}
        provenance: dict[str, str] = {}
        for item in reversed(found):
            if not isinstance(item.value, dict):
                invalid.append(Invalid(item.source, "deep-merge values must be tables"))
                continue
            merged_value, provenance = merge_toml_tables(
                merged_value,
                item.value,
                lower_source="default",
                higher_source=item.source,
                lower_provenance=provenance,
            )
        sources = " + ".join(item.source for item in found)
        return ResolvedValue(
            cast("T", merged_value),
            sources,
            provenance,
            tuple(invalid),
        )


def merge_toml_tables(
    lower: dict[str, Any],
    higher: dict[str, Any],
    *,
    lower_source: str,
    higher_source: str,
    union_paths: frozenset[tuple[str, ...]] = frozenset(),
    lower_provenance: dict[str, str] | None = None,
    _path: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Deep-merge TOML tables with higher-precedence leaf provenance.

    Returns:
        Merged table and dotted leaf-to-source mapping.
    """
    merged = deepcopy(lower)
    provenance = dict(lower_provenance or _leaf_provenance(lower, lower_source, _path))
    for key, value in higher.items():
        path = (*_path, key)
        dotted = ".".join(path)
        existing = merged.get(key)
        if isinstance(existing, dict) and not isinstance(value, dict):
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
        merged[key] = deepcopy(value)
        for leaf in tuple(provenance):
            if leaf == dotted or leaf.startswith(f"{dotted}."):
                provenance.pop(leaf)
        provenance.update(_leaf_provenance(value, higher_source, path))
    return merged, provenance


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
