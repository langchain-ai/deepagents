"""Unit tests for ranked config precedence and durable masking."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    MergeStrategy,
    OptionKind,
    get_config_options,
    get_option,
)
from deepagents_code.configuration.provider import CliProvider, ConfigProvider
from deepagents_code.configuration.providers import (
    DefaultProvider,
    EnvProvider,
    TomlFileProvider,
)
from deepagents_code.configuration.resolver import (
    CLI_RANK,
    DEFAULT_RANK,
    ENVIRONMENT_RANK,
    MANAGED_RANK,
    USER_RANK,
    ConfigResolver,
    RankedProviderValue,
    ResolvedValue,
    resolve_ranked,
    resolver_from_snapshots,
)
from deepagents_code.configuration.types import (
    Found,
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    TomlSnapshot,
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


class _TrackingProvider:
    """Synthetic protocol implementation with observable calls."""

    durable = True

    def __init__(
        self,
        rank: int,
        result: ProviderResult[Any],
        calls: list[int] | None = None,
    ) -> None:
        """Store a fixed result and optional shared call log."""
        self.name = f"rank {rank}"
        self.rank = rank
        self.result = result
        self.calls = calls if calls is not None else []
        self.reloads = 0

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Return the fixed result and record provider order."""
        del option
        self.calls.append(self.rank)
        return RankedProviderValue(
            self.rank,
            self.durable,
            self.status(),
            self.result,
        )

    def status(self) -> ProviderStatus:
        """Return synthetic healthy status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Record one propagated reload."""
        self.reloads += 1


def _bool_option(key: str, toml_key: str) -> ConfigOption:
    """Build a synthetic boolean manifest option."""
    return ConfigOption(
        key=key,
        group="Test",
        summary="test option",
        kind=OptionKind.BOOL,
        default=False,
        toml_keys=("test", toml_key),
    )


def test_concrete_providers_implement_protocol(tmp_path: Path) -> None:
    """Every built-in source satisfies the structural provider contract."""
    providers = (
        TomlFileProvider("config.toml", tmp_path / "config.toml"),
        EnvProvider(),
        DefaultProvider(),
    )

    assert all(isinstance(provider, ConfigProvider) for provider in providers)
    assert all(callable(provider.get) for provider in providers)
    assert all(callable(provider.status) for provider in providers)
    assert all(callable(provider.reload) for provider in providers)


def test_config_resolver_sorts_providers_by_rank() -> None:
    """Provider invocation and status mappings follow numeric precedence."""
    calls: list[int] = []
    user = _TrackingProvider(USER_RANK, Found("user"), calls)
    managed = _TrackingProvider(MANAGED_RANK, Found("managed"), calls)
    resolver = ConfigResolver((user, managed))

    resolved = resolver.get(_bool_option("test.enabled", "enabled"))

    assert resolved.value == "managed"
    assert calls == [MANAGED_RANK, USER_RANK]
    assert tuple(resolver.provider_statuses()) == (MANAGED_RANK, USER_RANK)


def test_config_resolver_rejects_duplicate_provider_ranks() -> None:
    """A colliding rank cannot overwrite provider health or provenance."""
    with pytest.raises(ValueError, match="unique ranks"):
        ConfigResolver(
            (
                _TrackingProvider(USER_RANK, Found("first")),
                _TrackingProvider(USER_RANK, Found("second")),
            )
        )


def test_config_resolver_reload_propagates_to_every_provider() -> None:
    """Reload reaches every provider in precedence order."""
    first = _TrackingProvider(MANAGED_RANK, Unset())
    second = _TrackingProvider(DEFAULT_RANK, Found(False))
    resolver = ConfigResolver((second, first))

    resolver.reload()

    assert first.reloads == 1
    assert second.reloads == 1


def test_resolve_all_uses_one_toml_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-manifest read cannot mix file generations."""
    from deepagents_code import config_manifest

    path = tmp_path / "config.toml"
    path.write_text("[test]\nfirst = true\nsecond = false\n", encoding="utf-8")
    user = TomlFileProvider("config.toml", path)
    assert user.status().health is ProviderHealth.OK
    path.write_text("[test]\nfirst = false\nsecond = true\n", encoding="utf-8")
    options = (
        _bool_option("test.first", "first"),
        _bool_option("test.second", "second"),
    )
    monkeypatch.setattr(config_manifest, "get_config_options", lambda: options)
    resolver = ConfigResolver((user, DefaultProvider()))

    before = resolver.resolve_all()
    resolver.reload()
    after = resolver.resolve_all()

    assert before["test.first"].value is True
    assert before["test.second"].value is False
    assert after["test.first"].value is False
    assert after["test.second"].value is True


@pytest.mark.parametrize(
    "health",
    [ProviderHealth.CORRUPT, ProviderHealth.UNREADABLE],
)
def test_initial_failed_toml_snapshot_falls_through_to_default(
    health: ProviderHealth,
    tmp_path: Path,
) -> None:
    """A failed first read must remain an empty resolvable generation."""
    path = tmp_path / "config.toml"
    snapshot = TomlSnapshot({}, ProviderStatus("config.toml", path, health))
    provider = TomlFileProvider(
        "config.toml",
        path,
        loader=lambda: snapshot,
    )
    option = _bool_option("test.enabled", "enabled")

    resolved = ConfigResolver((provider, DefaultProvider())).get(option)

    assert resolved.value is False
    assert resolved.ranks == (DEFAULT_RANK,)
    assert resolved.provider_status[USER_RANK].health is health


def test_failed_toml_reload_keeps_the_last_usable_snapshot(tmp_path: Path) -> None:
    """A corrupt re-read must not replace the values still being enforced.

    An unusable candidate carries an empty table, which resolution reads as
    "this source declares nothing"; installing it on reload would drop the
    file's values and let lower ranks win.
    """
    path = tmp_path / "managed_config.toml"
    path.write_text("[test]\nenabled = true\n", encoding="utf-8")
    provider = TomlFileProvider("managed config", path, MANAGED_RANK)
    option = _bool_option("test.enabled", "enabled")
    assert provider.get(option).result == Found(True)

    path.write_text("not toml [", encoding="utf-8")
    provider.reload()

    assert provider.get(option).result == Found(True)
    status = provider.status()
    assert status.health is ProviderHealth.CORRUPT

    path.write_text("[test]\nenabled = false\n", encoding="utf-8")
    provider.reload()

    assert provider.get(option).result == Found(False)
    assert provider.status().health is ProviderHealth.OK


def test_failed_reload_keeps_managed_policy_enforced(tmp_path: Path) -> None:
    """A failed managed reload through the resolver must not fail open.

    Regression: `get_managed_snapshot(refresh=True)` returns the failed
    candidate for diagnostics, and `reload` installed it, so the managed tier
    read as unset and the user tier won until the file was repaired.
    """
    managed_path = tmp_path / "managed_config.toml"
    user_path = tmp_path / "config.toml"
    managed_path.write_text("[test]\nenabled = false\n", encoding="utf-8")
    user_path.write_text("[test]\nenabled = true\n", encoding="utf-8")
    managed = TomlFileProvider(
        "managed config",
        managed_path,
        MANAGED_RANK,
        True,
        loader=lambda: TomlFileProvider("managed config", managed_path).load(),
    )
    user = TomlFileProvider(
        "config.toml",
        user_path,
        USER_RANK,
        True,
        loader=lambda: TomlFileProvider("config.toml", user_path).load(),
    )
    resolver = ConfigResolver((managed, user))
    option = _bool_option("test.enabled", "enabled")
    assert resolver.get(option).value is False

    managed_path.write_text("not toml [", encoding="utf-8")
    resolver.reload()

    resolved = resolver.get(option)
    assert resolved.value is False
    assert resolved.provider_status[MANAGED_RANK].health is ProviderHealth.OK
    assert resolver.provider_statuses()[MANAGED_RANK].health is ProviderHealth.CORRUPT

    managed_path.write_text("[test]\nenabled = false\n", encoding="utf-8")
    resolver.reload()

    assert resolver.provider_statuses()[MANAGED_RANK].health is ProviderHealth.OK


def test_rejected_managed_reload_keeps_last_enforceable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parseable policy violation must not replace managed resolution."""
    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, service
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed_config.toml"
    user_path = tmp_path / "config.toml"
    managed_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    user_path.write_text('[startup]\nmode = "yolo"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed_path)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_path)
    monkeypatch.setattr(
        resolver_module,
        "_resolver_cache",
        resolver_module._ResolverCache(),
    )
    service.invalidate_config_sources()
    try:
        resolver = resolver_module.get_config_resolver()
        option = get_option("startup.mode")
        assert option is not None
        assert resolver.get(option).value == "manual"

        managed_path.write_text("startup.mode = 5\n", encoding="utf-8")
        resolver.reload()

        assert resolver.get(option).value == "manual"
    finally:
        service.invalidate_config_sources()


def test_real_manifest_options_merge_across_tiers() -> None:
    """Managed policy must compose with the user tier, not replace it.

    `MergeStrategy` is declared per option in the manifest, and the other merge
    tests use synthetic providers -- so nothing else pins the wiring for a real
    option. A `UNION` deny-list flipped to `REPLACE` would silently drop every
    user entry the moment an administrator denies one server, and a
    `DEEP_MERGE` table flipped the same way would drop the user's sibling
    columns. Both read as "policy applied" rather than as data loss.
    """
    managed = TomlSnapshot(
        {
            "startup": {"mode": "manual"},
            "mcp": {"disabled_servers": ["alpha"]},
            "threads": {"columns": {"title": {"width": 10}}},
        },
        ProviderStatus("managed config", None, ProviderHealth.OK),
    )
    user = TomlSnapshot(
        {
            "startup": {"mode": "auto"},
            "mcp": {"disabled_servers": ["beta"]},
            "threads": {"columns": {"summary": {"width": 30}}},
        },
        ProviderStatus("config.toml", None, ProviderHealth.OK),
    )
    resolved = resolver_from_snapshots(managed=managed, user=user).resolve_all()

    # A REPLACE scalar is the control: the stronger rank wins outright.
    startup = resolved["startup.mode"]
    assert startup.value == "manual"
    assert startup.ranks == (MANAGED_RANK,)

    # A deny-list union keeps every tier's contribution rather than letting the
    # stronger rank replace the weaker one.
    disabled = resolved["mcp.disabled_servers"]
    assert set(disabled.ranks) == {MANAGED_RANK, USER_RANK}
    assert isinstance(disabled.value, list)
    assert set(disabled.value) == {"alpha", "beta"}

    # A deep merge composes sibling leaves from both tiers.
    columns = resolved["threads.columns"]
    assert set(columns.ranks) == {MANAGED_RANK, USER_RANK}
    assert isinstance(columns.value, dict)
    assert set(columns.value) == {"title", "summary"}


def test_toml_snapshot_returns_the_generation_in_force() -> None:
    """The accessor exists so a caller can share this resolver's generation.

    Previously reached only indirectly, through
    `_resolve_option_without_managed`.
    """
    managed = TomlSnapshot.from_table("managed config", {"startup": {"mode": "manual"}})
    user = TomlSnapshot.from_table("config.toml", {"startup": {"mode": "auto"}})
    resolver = resolver_from_snapshots(managed=managed, user=user)

    assert resolver.toml_snapshot(MANAGED_RANK) == managed
    assert resolver.toml_snapshot(USER_RANK) == user
    # Environment and default providers carry no file snapshot.
    assert resolver.toml_snapshot(ENVIRONMENT_RANK) is None
    assert resolver.toml_snapshot(DEFAULT_RANK) is None


def test_healthy_managed_snapshot_refuses_unenforceable_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Policy that parses but cannot be enforced must stop the launch.

    Covered only through `_reload_values` before this; the function is the
    startup gate, so its own contract deserves a direct test.
    """
    from deepagents_code.configuration import service
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed.toml"
    redirect_managed_config(monkeypatch, managed_path)

    managed_path.write_text("[shell]\nallow_list = []\n", encoding="utf-8")
    service.invalidate_config_sources()
    try:
        assert service.get_healthy_managed_snapshot().data == {
            "shell": {"allow_list": []}
        }

        # A known section as a scalar can erase a user subtree, so it is
        # rejected rather than resolved in the user's favor.
        managed_path.write_text("shell = 5\n", encoding="utf-8")
        service.invalidate_config_sources()
        with pytest.raises(service.ManagedPolicyError):
            service.get_healthy_managed_snapshot()

        # A file that does not parse at all is a different failure.
        managed_path.write_text("[shell\n", encoding="utf-8")
        service.invalidate_config_sources()
        with pytest.raises(service.ManagedConfigError):
            service.get_healthy_managed_snapshot()
    finally:
        service.invalidate_config_sources()


def test_reload_with_replacements_rejects_an_unknown_rank(tmp_path: Path) -> None:
    """A replacement for a rank this resolver does not have is a mistake."""
    managed_path = tmp_path / "managed.toml"
    managed_path.write_text('startup.mode = "manual"\n', encoding="utf-8")
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot(
            {"startup": {"mode": "manual"}},
            ProviderStatus("managed config", managed_path, ProviderHealth.OK),
        ),
        user=TomlSnapshot({}, ProviderStatus("config.toml", None, ProviderHealth.OK)),
    )

    with pytest.raises(ValueError, match="unknown provider ranks"):
        resolver.reload_with_replacements(
            {
                999: TomlFileProvider(
                    "managed config",
                    managed_path,
                    999,
                    True,
                )
            }
        )


def test_reload_with_replacements_rejects_an_unusable_source(tmp_path: Path) -> None:
    """An unusable replacement would silently drop the tier's restrictions.

    Replacements bypass `TomlFileProvider.reload`, and with it the guarantee
    that a snapshot the source cannot use never displaces the last usable one.
    An unusable snapshot carries an empty table, which resolves as "declares
    nothing" -- so installing one at `MANAGED_RANK` reads as "no policy".
    """
    managed_path = tmp_path / "managed.toml"
    managed_path.write_text("[startup\n", encoding="utf-8")
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot(
            {"startup": {"mode": "manual"}},
            ProviderStatus("managed config", managed_path, ProviderHealth.OK),
        ),
        user=TomlSnapshot({}, ProviderStatus("config.toml", None, ProviderHealth.OK)),
    )
    broken = TomlFileProvider("managed config", managed_path, MANAGED_RANK, True)
    broken.reload()

    with pytest.raises(ValueError, match="unusable"):
        resolver.reload_with_replacements({MANAGED_RANK: broken})


def test_an_ignored_managed_snapshot_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A snapshot that would be discarded must fail loudly instead.

    `managed_snapshot` is honored on a cache miss and installed on a refresh,
    but a cache hit without `refresh_managed` keeps the generation already in
    force. That is correct for the preview path, which passes the snapshot it
    is already enforcing -- and silently wrong for anyone who passes a newer
    one and expects it to take effect.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, service
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed.toml"
    managed_path.write_text('startup.mode = "manual"\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed_path)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(
        resolver_module, "_resolver_cache", resolver_module._ResolverCache()
    )
    service.invalidate_config_sources()
    try:
        # Populate the cache, then offer a different generation without asking
        # for it to be installed.
        resolver_module.get_config_resolver()
        newer = TomlSnapshot(
            {"startup": {"mode": "auto"}},
            ProviderStatus("managed config", managed_path, ProviderHealth.OK),
        )

        with pytest.raises(ValueError, match="different generation"):
            resolver_module.get_config_resolver(managed_snapshot=newer)

        # The same generation is what the preview path passes, and it is fine.
        installed = resolver_module.get_config_resolver().toml_snapshot(MANAGED_RANK)
        assert installed is not None
        assert (
            resolver_module.get_config_resolver(managed_snapshot=installed) is not None
        )
    finally:
        service.invalidate_config_sources()


def _nest(keys: tuple[str, ...], value: object) -> dict[str, Any]:
    """Wrap `value` in the nested tables named by a manifest option's keys."""
    nested: dict[str, Any] = {keys[-1]: value}
    for key in reversed(keys[:-1]):
        nested = {key: nested}
    return nested


# Every option that must compose its tiers rather than replace them.
#
# Frozen deliberately: deriving this from `option.merge_strategy` would be a
# tautology, because a downgraded option simply drops out of the list and
# stops being tested. The expected strategy has to be written down here for a
# downgrade to fail anything.
_COMPOSING_OPTIONS = {
    "display.themes": MergeStrategy.DEEP_MERGE,
    "display.terminal_themes": MergeStrategy.DEEP_MERGE,
    "models.providers": MergeStrategy.DEEP_MERGE,
    "agents.async_subagents": MergeStrategy.DEEP_MERGE,
    "sandboxes.providers": MergeStrategy.DEEP_MERGE,
    "threads.columns": MergeStrategy.DEEP_MERGE,
    "mcp.enabled_project_server_approvals": MergeStrategy.DEEP_MERGE,
    "mcp.disabled_project_servers": MergeStrategy.UNION,
    "mcp.disabled_servers": MergeStrategy.UNION,
}


def test_composing_options_are_the_expected_set() -> None:
    """A new or removed composing option must update the frozen table.

    Without this, `_COMPOSING_OPTIONS` silently stops covering the manifest:
    a new `DEEP_MERGE` table would ship with no merge coverage at all.
    """
    actual = {
        option.key: option.merge_strategy
        for option in get_config_options()
        if option.merge_strategy is not MergeStrategy.REPLACE
    }

    assert actual == _COMPOSING_OPTIONS


@pytest.mark.parametrize("key", sorted(_COMPOSING_OPTIONS))
def test_every_composing_option_composes(key: str) -> None:
    """Every non-`REPLACE` option must keep both tiers' contributions.

    The test above names three options, which is enough to pin those three and
    nothing else: flipping `display.terminal_themes`, `agents.async_subagents`,
    or `sandboxes.providers` to `REPLACE` passed the whole suite.

    Losing this is silent and looks like success: a `UNION` deny-list flipped
    to `REPLACE` drops every user entry the moment an administrator denies one
    server, and a `DEEP_MERGE` table drops the user's sibling leaves. Both
    read as "policy applied" rather than as data loss.
    """
    option = get_option(key)
    assert option is not None
    assert option.toml_keys, f"{key} must declare the table it composes"
    strategy = _COMPOSING_OPTIONS[key]
    assert option.merge_strategy is strategy, (
        f"{key} must stay {strategy}: a weaker strategy discards a tier"
    )

    if strategy is MergeStrategy.UNION:
        managed_value: object = ["from-managed"]
        user_value: object = ["from-user"]
    else:
        managed_value = {"from_managed": {"width": 10}}
        user_value = {"from_user": {"width": 20}}
    resolved = resolver_from_snapshots(
        managed=TomlSnapshot(
            _nest(option.toml_keys, managed_value),
            ProviderStatus("managed config", None, ProviderHealth.OK),
        ),
        user=TomlSnapshot(
            _nest(option.toml_keys, user_value),
            ProviderStatus("config.toml", None, ProviderHealth.OK),
        ),
    ).get(option)

    assert set(resolved.ranks) == {MANAGED_RANK, USER_RANK}, (
        f"{key} ({strategy}) dropped a tier"
    )
    if strategy is MergeStrategy.UNION:
        assert isinstance(resolved.value, list)
        assert set(resolved.value) == {"from-managed", "from-user"}, (
            f"{key} did not union both tiers"
        )
    else:
        assert isinstance(resolved.value, dict)
        assert set(resolved.value) == {"from_managed", "from_user"}, (
            f"{key} did not merge sibling leaves"
        )


def test_cli_rank_beats_environment_user_and_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    option = get_option("runtime.recursion_limit")
    assert option is not None
    monkeypatch.setenv("DEEPAGENTS_RECURSION_LIMIT", "50")
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot(
            {}, ProviderStatus("managed config", None, ProviderHealth.OK)
        ),
        user=TomlSnapshot(
            {"runtime": {"recursion_limit": 75}},
            ProviderStatus("config.toml", None, ProviderHealth.OK),
        ),
        cli_provider=CliProvider({"recursion_limit": 100}),
    )

    resolved = resolver.get(option)

    assert resolved.value == 100
    assert resolved.ranks == (CLI_RANK,)
    assert resolved.provider_status[CLI_RANK].name == "CLI argument"


def test_managed_rank_masks_cli_but_user_does_not() -> None:
    option = get_option("runtime.recursion_limit")
    assert option is not None
    resolver = resolver_from_snapshots(
        managed=TomlSnapshot(
            {"runtime": {"recursion_limit": 25}},
            ProviderStatus("managed config", None, ProviderHealth.OK),
        ),
        user=TomlSnapshot(
            {"runtime": {"recursion_limit": 75}},
            ProviderStatus("config.toml", None, ProviderHealth.OK),
        ),
        cli_provider=CliProvider({"recursion_limit": 100}),
    )

    resolved = resolver.get(option)

    assert resolved.value == 25
    assert resolved.ranks == (MANAGED_RANK,)
    assert resolved.masked_ranks == frozenset({CLI_RANK})


def test_settings_from_environment_does_not_import_textual(tmp_path: Path) -> None:
    """Building `Settings` must not drag the theme registry onto the hot path.

    `resolve_all()` would resolve `display.theme`, whose `THEME_DELEGATE`
    coercion reaches the theme registry and imports Textual (~470ms). The four
    CLI entry points in `skills/commands.py` build `Settings` without drawing
    a UI, so they must not pay for it.
    """
    home = tmp_path / "home"
    (home / ".deepagents").mkdir(parents=True)
    (home / ".deepagents" / "config.toml").write_text(
        '[ui]\ntheme = "monokai"\n',
        encoding="utf-8",
    )
    env: dict[str, str] = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env.pop("DEEPAGENTS_CODE_THEME", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "from deepagents_code.config import Settings\n"
                "Settings.from_environment()\n"
                "assert 'textual' not in sys.modules, 'Textual reached the "
                "startup path'\n"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_corrupt_user_toml_warns_instead_of_defaulting_silently(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file the reader rejected must not look like a file that says nothing.

    An unusable source coerces to `Unset` for every option, which resolution
    reads as "declares nothing". Without a diagnostic the user's typo silently
    replaces their `shell.allow_list` with the manifest default.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics

    config_path = tmp_path / "config.toml"
    config_path.write_text("[shell\nallow_list = []\n", encoding="utf-8")
    provider = TomlFileProvider("config.toml", config_path)
    resolver = ConfigResolver((provider, DefaultProvider()))
    option = get_option("shell.allow_list")
    assert option is not None

    resolved = resolver.get(option)
    assert resolved.ranks == (DEFAULT_RANK,)

    with caplog.at_level("WARNING"):
        _emit_ranked_diagnostics(option, resolved)

    assert any("config.toml" in record.message for record in caplog.records)
    assert any("CORRUPT" in record.message for record in caplog.records)


def test_unusable_source_warning_is_emitted_once_per_process(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The rejection belongs to the file, not to each of its hundred options."""
    from deepagents_code.config_manifest import _emit_ranked_diagnostics

    config_path = tmp_path / "config.toml"
    config_path.write_text("[shell\n", encoding="utf-8")
    resolver = ConfigResolver(
        (TomlFileProvider("config.toml", config_path), DefaultProvider())
    )

    with caplog.at_level("WARNING"):
        for option in get_config_options():
            _emit_ranked_diagnostics(option, resolver.get(option))

    rejections = [
        record for record in caplog.records if "using defaults" in record.message
    ]
    assert len(rejections) == 1


def test_healthy_source_emits_no_rejection_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file that parses must stay quiet."""
    from deepagents_code.config_manifest import _emit_ranked_diagnostics

    config_path = tmp_path / "config.toml"
    config_path.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    resolver = ConfigResolver(
        (TomlFileProvider("config.toml", config_path), DefaultProvider())
    )
    option = get_option("shell.allow_list")
    assert option is not None

    with caplog.at_level("WARNING"):
        resolved = resolver.get(option)
        _emit_ranked_diagnostics(option, resolved)

    assert resolved.value == ["ls"]
    assert not [
        record for record in caplog.records if "using defaults" in record.message
    ]


def test_default_path_write_is_visible_to_the_shared_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed settings write must not leave the process serving stale values."""
    from deepagents_code import model_config
    from deepagents_code.configuration import (
        resolver as resolver_module,
        service,
        writer,
    )

    config_path = tmp_path / "config.toml"
    config_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        service,
        "get_managed_snapshot",
        lambda refresh=False: TomlSnapshot(  # noqa: ARG005
            {},
            ProviderStatus("managed config", None, ProviderHealth.MISSING),
        ),
    )
    service.invalidate_config_sources()
    try:
        option = get_option("startup.mode")
        assert option is not None
        assert resolver_module.get_config_resolver().get(option).value == "manual"

        def set_auto(data: dict[str, Any]) -> bool:
            data.setdefault("startup", {})["mode"] = "auto"
            return True

        assert writer.update_user_config(set_auto, config_path=config_path).ok
        assert resolver_module.get_config_resolver().get(option).value == "auto"
    finally:
        service.invalidate_config_sources()


def test_write_to_an_override_path_leaves_the_default_resolver_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An override write must not re-read the real user and managed configs.

    `get_config_resolver` is keyed on `DEFAULT_CONFIG_PATH`, so reloading it
    after a write elsewhere touches files the caller never named - live reads
    from a test that deliberately passed a `tmp_path`.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, writer

    default_path = tmp_path / "default.toml"
    other_path = tmp_path / "other.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", default_path)
    calls: list[int] = []

    def record(**_: object) -> ConfigResolver:
        calls.append(1)
        msg = "the shared resolver must not be built for an override write"
        raise AssertionError(msg)

    monkeypatch.setattr(resolver_module, "get_config_resolver", record)

    def set_auto(data: dict[str, Any]) -> bool:
        data.setdefault("startup", {})["mode"] = "auto"
        return True

    assert writer.update_user_config(set_auto, config_path=other_path).ok
    assert calls == []


def test_a_failed_resolver_refresh_does_not_fail_a_landed_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The bytes are already on disk; reporting failure sends the user to retry."""
    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, writer

    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    def explode(**_: object) -> ConfigResolver:
        msg = "disk went away"
        raise OSError(msg)

    monkeypatch.setattr(resolver_module, "get_config_resolver", explode)

    def set_auto(data: dict[str, Any]) -> bool:
        data.setdefault("startup", {})["mode"] = "auto"
        return True

    with caplog.at_level("WARNING"):
        result = writer.update_user_config(set_auto, config_path=config_path)

    assert result.ok
    assert result.changed
    assert 'mode = "auto"' in config_path.read_text(encoding="utf-8")
    assert any("could not refresh" in record.message for record in caplog.records)


def test_a_pathless_provider_does_not_read_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A snapshot with no known origin must not be reloaded from a guessed path.

    `resolver_from_snapshots` used to substitute a bare relative filename, so
    reloading a diagnostic resolver would read `./managed_config.toml` from
    whatever directory the process was launched in and enforce it as policy.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "managed_config.toml").write_text(
        '[startup]\nmode = "auto"\n',
        encoding="utf-8",
    )
    managed = TomlSnapshot(
        {},
        ProviderStatus("managed config", None, ProviderHealth.MISSING),
    )
    user = TomlSnapshot({}, ProviderStatus("config.toml", None, ProviderHealth.MISSING))
    resolver = resolver_from_snapshots(managed=managed, user=user)
    option = get_option("startup.mode")
    assert option is not None

    resolver.reload()

    assert resolver.get(option).ranks == (DEFAULT_RANK,)
    assert (
        resolver.provider_statuses()[MANAGED_RANK].health
        is ProviderHealth.INDETERMINATE
    )


@pytest.mark.parametrize(
    ("provider", "expected"),
    [(EnvProvider(), False), (DefaultProvider(), True)],
)
def test_stateless_provider_durability_cannot_be_overridden(
    provider: ConfigProvider,
    *,
    expected: bool,
) -> None:
    """Durability decides masking, so the attribute must not be able to lie.

    Both providers delegate to helpers that stamp a hardcoded durability onto
    every result. While `durable` was a settable field, passing the opposite
    value type-checked and changed nothing.
    """
    option = get_option("startup.mode")
    assert option is not None

    assert provider.durable is expected
    assert provider.get(option).durable is expected
    # Built dynamically so the type checker does not reject the call before
    # the test can prove the constructor does.
    overridden: dict[str, Any] = {"durable": not expected}
    with pytest.raises((TypeError, AttributeError)):
        type(provider)(**overridden)


def test_a_failed_reload_warns_that_the_edit_did_not_take_effect(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retaining the last good generation must not be silent.

    Resolution keeps returning the previous values, so nothing looks wrong -
    while the edit the user just saved is not in effect and the file on disk no
    longer describes what the process enforces.
    """
    from deepagents_code.config_manifest import _emit_ranked_diagnostics

    config_path = tmp_path / "config.toml"
    config_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    provider = TomlFileProvider("config.toml", config_path)
    resolver = ConfigResolver((provider, DefaultProvider()))
    option = get_option("startup.mode")
    assert option is not None
    assert resolver.get(option).value == "manual"

    config_path.write_text("[startup\n", encoding="utf-8")
    resolver.reload()

    with caplog.at_level("WARNING"):
        resolved = resolver.get(option)
        _emit_ranked_diagnostics(option, resolved)

    assert resolved.value == "manual"
    assert any("still applying" in record.message for record in caplog.records)
    assert any("CORRUPT" in record.message for record in caplog.records)


def test_doctor_reports_a_corrupt_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command a user runs when settings do not apply must say the file broke."""
    from deepagents_code import doctor, model_config

    config_path = tmp_path / "config.toml"
    config_path.write_text("[shell\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    item = doctor._user_config_diagnostic()

    assert not item.ok
    assert "corrupt" in item.value.lower()


def test_doctor_is_green_for_a_config_that_parses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy or absent file must not be reported as a problem."""
    from deepagents_code import doctor, model_config

    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    assert doctor._user_config_diagnostic().ok

    config_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    assert doctor._user_config_diagnostic().ok


def test_invalidate_config_sources_also_drops_the_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing only the managed snapshot leaves the resolver serving stale values.

    The two caches are keyed differently. Tests escape pollution today only by
    incidentally monkeypatching `DEFAULT_CONFIG_PATH`; one that exercises the
    resolver at an unchanged path would inherit the previous test's generation.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, service

    config_path = tmp_path / "config.toml"
    config_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        service,
        "get_managed_snapshot",
        lambda refresh=False: TomlSnapshot(  # noqa: ARG005
            {},
            ProviderStatus("managed config", None, ProviderHealth.MISSING),
        ),
    )
    service.invalidate_config_sources()
    try:
        option = get_option("startup.mode")
        assert option is not None
        assert resolver_module.get_config_resolver().get(option).value == "manual"

        # Same path, so the cache key is unchanged: only an explicit reset can
        # make the edit visible.
        config_path.write_text('[startup]\nmode = "auto"\n', encoding="utf-8")
        assert resolver_module.get_config_resolver().get(option).value == "manual"

        service.invalidate_config_sources()
        assert resolver_module.get_config_resolver().get(option).value == "auto"
    finally:
        service.invalidate_config_sources()


def test_resolved_value_rejects_a_selected_rank_with_no_provider_status() -> None:
    """`_ranked_source` indexes `provider_status` by every selected rank.

    An instance whose halves disagree is a `KeyError` in the source column of
    user-facing `config` output, so it must not be constructible.
    """
    with pytest.raises(ValueError, match="no provider status"):
        ResolvedValue(
            "value",
            {USER_RANK: frozenset({()})},
            {},
            {},
            frozenset(),
            (USER_RANK,),
        )


def test_resolved_value_rejects_a_rank_that_is_both_selected_and_masked() -> None:
    """A tier cannot have won and been hidden by a stronger durable tier."""
    status = ProviderStatus("config.toml", None, ProviderHealth.OK)
    with pytest.raises(ValueError, match="both selected and masked"):
        ResolvedValue(
            "value",
            {USER_RANK: frozenset({()})},
            {USER_RANK: Found("value")},
            {USER_RANK: status},
            frozenset({USER_RANK}),
            (USER_RANK,),
        )


def test_resolved_value_does_not_alias_the_mappings_it_was_given() -> None:
    """`frozen=True` protects the bindings, not the contents."""
    status = ProviderStatus("config.toml", None, ProviderHealth.OK)
    provider_status = {USER_RANK: status}
    resolved = ResolvedValue(
        "value",
        {USER_RANK: frozenset({()})},
        {USER_RANK: Found("value")},
        provider_status,
        frozenset(),
        (USER_RANK,),
    )

    provider_status[MANAGED_RANK] = ProviderStatus(
        "managed config",
        None,
        ProviderHealth.OK,
    )

    assert set(resolved.provider_status) == {USER_RANK}


def test_provenance_rank_without_provider_status_is_rejected() -> None:
    """`ranks` falls back to `provenance`, so it needs the same guard.

    `_ranked_source` indexes `provider_status` by whatever `ranks` returns.
    Validating only `selected_ranks` left the documented `KeyError` reachable
    through the fallback branch, which is the one an external constructor takes
    when it leaves `selected_ranks` empty.
    """
    with pytest.raises(ValueError, match="contributing ranks"):
        ResolvedValue(
            value=None,
            provenance={USER_RANK: frozenset({()})},
            tier_health={},
            provider_status={},
        )


def test_selected_rank_without_provider_status_is_still_rejected() -> None:
    """The original guard is unchanged."""
    with pytest.raises(ValueError, match="contributing ranks"):
        ResolvedValue(
            value=None,
            provenance={},
            tier_health={},
            provider_status={},
            selected_ranks=(USER_RANK,),
        )
