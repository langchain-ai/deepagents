"""Unit tests for ranked config precedence and durable masking."""

from typing import Any

import pytest

from deepagents_code.configuration.resolver import (
    CLI_RANK,
    ENVIRONMENT_RANK,
    MANAGED_RANK,
    USER_RANK,
    RankedProviderValue,
    resolve_ranked,
)
from deepagents_code.configuration.types import (
    Found,
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    Unset,
)


def _provider(
    rank: int,
    result: ProviderResult[Any],
    *,
    durable: bool,
) -> RankedProviderValue[Any]:
    """Build one synthetic ranked provider."""
    return RankedProviderValue(
        rank,
        durable,
        ProviderStatus(f"rank {rank}", None, ProviderHealth.OK),
        result,
    )


def test_durable_found_masks_only_lower_priority_ephemeral_tiers() -> None:
    """A durable policy boundary is directional and explicit in the result."""
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, Found("managed"), durable=True),
            _provider(ENVIRONMENT_RANK, Found("environment"), durable=False),
            _provider(USER_RANK, Found("user"), durable=True),
        )
    )

    assert resolved is not None
    assert resolved.value == "managed"
    assert resolved.ranks == (MANAGED_RANK,)
    assert resolved.masked_ranks == frozenset({ENVIRONMENT_RANK})


def test_lower_priority_durable_value_does_not_mask_environment() -> None:
    """Persistence cannot reverse numeric precedence after a tier has won."""
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, Unset(), durable=True),
            _provider(ENVIRONMENT_RANK, Found("environment"), durable=False),
            _provider(USER_RANK, Found("user"), durable=True),
        )
    )

    assert resolved is not None
    assert resolved.value == "environment"
    assert resolved.ranks == (ENVIRONMENT_RANK,)
    assert resolved.masked_ranks == frozenset()


def test_invalid_durable_tier_falls_through_and_retains_ranked_health() -> None:
    """Only `Found` masks; an invalid durable declaration stays inspectable."""
    invalid = Invalid("synthetic managed rejection")
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, invalid, durable=True),
            _provider(ENVIRONMENT_RANK, Found(7), durable=False),
        )
    )

    assert resolved is not None
    assert resolved.value == 7
    assert resolved.ranks == (ENVIRONMENT_RANK,)
    assert resolved.tier_health[MANAGED_RANK] == invalid


def test_union_keeps_all_restrictive_tiers_and_rank_provenance() -> None:
    """Accumulating deny lists preserve every tier despite replacement masks."""
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, Found(["managed", "shared"]), durable=True),
            _provider(ENVIRONMENT_RANK, Found(["environment"]), durable=False),
            _provider(USER_RANK, Found(["user", "shared"]), durable=True),
        ),
        strategy="union",
    )

    assert resolved is not None
    assert resolved.value == ["user", "shared", "environment", "managed"]
    assert resolved.ranks == (MANAGED_RANK, ENVIRONMENT_RANK, USER_RANK)
    assert resolved.masked_ranks == frozenset()


def test_deep_merge_provenance_uses_tuple_paths_and_numeric_ranks() -> None:
    """Quoted dotted leaves cannot collide with nested sibling provenance."""
    resolved = resolve_ranked(
        (
            _provider(
                MANAGED_RANK,
                Found({"a": {"managed": 2}, "a.b": 2}),
                durable=True,
            ),
            _provider(
                USER_RANK,
                Found({"a": {"user": 1}, "a.b": 1, "sibling": 1}),
                durable=True,
            ),
        ),
        strategy="deep_merge",
    )

    assert resolved is not None
    assert resolved.value == {
        "a": {"user": 1, "managed": 2},
        "a.b": 2,
        "sibling": 1,
    }
    assert resolved.provenance[MANAGED_RANK] == frozenset({("a", "managed"), ("a.b",)})
    assert resolved.provenance[USER_RANK] == frozenset({("a", "user"), ("sibling",)})


def test_rank_space_reserves_but_does_not_require_a_cli_provider() -> None:
    """The unwired CLI seam outranks environment and yields to managed policy."""
    assert MANAGED_RANK < CLI_RANK < ENVIRONMENT_RANK < USER_RANK

    over_environment = resolve_ranked(
        (
            _provider(CLI_RANK, Found("cli"), durable=False),
            _provider(ENVIRONMENT_RANK, Found("environment"), durable=False),
        )
    )
    under_managed = resolve_ranked(
        (
            _provider(MANAGED_RANK, Found("managed"), durable=True),
            _provider(CLI_RANK, Found("cli"), durable=False),
        )
    )

    assert over_environment is not None
    assert over_environment.value == "cli"
    assert under_managed is not None
    assert under_managed.value == "managed"


def test_deep_merge_scalar_tier_cannot_outrank_stronger_tables() -> None:
    """A mid-rank scalar falls back to the strongest tier, not to itself."""
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, Found({"a": 1}), durable=True),
            _provider(CLI_RANK, Found("scalar"), durable=False),
            _provider(USER_RANK, Found({"b": 2}), durable=True),
        ),
        strategy="deep_merge",
    )

    assert resolved is not None
    assert resolved.value == {"a": 1}
    assert resolved.selected_ranks == (MANAGED_RANK,)


def test_accumulating_fallbacks_agree_on_precedence() -> None:
    """`union` and `deep_merge` resolve an unmergeable tier the same way."""
    providers = (
        _provider(MANAGED_RANK, Found({"a": 1}), durable=True),
        _provider(ENVIRONMENT_RANK, Found("scalar"), durable=False),
        _provider(USER_RANK, Found({"b": 2}), durable=True),
    )

    merged = resolve_ranked(providers, strategy="deep_merge")
    united = resolve_ranked(providers, strategy="union")

    assert merged is not None
    assert united is not None
    assert merged.selected_ranks == united.selected_ranks == (MANAGED_RANK,)


def test_unmergeable_fallback_copies_the_winning_provider() -> None:
    """A fallback must not alias the process-wide managed snapshot."""
    managed_table = {"provider": {"params": {"api_key": "policy"}}}
    resolved = resolve_ranked(
        (
            _provider(MANAGED_RANK, Found(managed_table), durable=True),
            _provider(USER_RANK, Found(5), durable=True),
        ),
        strategy="deep_merge",
    )

    assert resolved is not None
    assert resolved.value == managed_table
    assert resolved.value is not managed_table

    resolved.value["provider"]["params"]["api_key"] = "mutated"
    assert managed_table["provider"]["params"]["api_key"] == "policy"


def test_duplicate_provider_ranks_are_rejected() -> None:
    """Rank-keyed health cannot silently overwrite a colliding provider."""
    providers = (
        _provider(USER_RANK, Found("first"), durable=True),
        _provider(USER_RANK, Found("second"), durable=True),
    )

    with pytest.raises(ValueError, match="unique ranks"):
        resolve_ranked(providers)
