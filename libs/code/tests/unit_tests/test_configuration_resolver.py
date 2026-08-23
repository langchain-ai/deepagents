"""Unit tests for ranked config precedence and durable masking."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    OptionKind,
    get_config_options,
    get_option,
    resolve_scalar,
)
from deepagents_code.configuration.provider import ConfigProvider
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


def test_resolver_get_matches_resolve_scalar_for_every_manifest_option() -> None:
    """The compatibility wrapper and provider resolver remain equivalent."""
    managed = TomlSnapshot(
        {},
        ProviderStatus("managed config", None, ProviderHealth.OK),
    )
    user = TomlSnapshot(
        {},
        ProviderStatus("config.toml", None, ProviderHealth.OK),
    )
    resolver = resolver_from_snapshots(managed, user)
    resolved = resolver.resolve_all()

    for option in get_config_options():
        value, source = resolve_scalar(
            option,
            toml_data=user.data,
            managed_toml_data=managed.data,
        )
        actual = resolved[option.key]
        actual_source = " + ".join(
            actual.provider_status[rank].name for rank in actual.ranks
        )
        assert (actual.value, actual_source) == (value, source), option.key


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
