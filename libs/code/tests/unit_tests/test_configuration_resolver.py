"""Unit tests for ranked config precedence and durable masking."""

from pathlib import Path
from typing import Any, Self, cast

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    MergeStrategy,
    OptionKind,
    get_option,
)
from deepagents_code.configuration.providers import DefaultProvider, TomlFileProvider
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
        result: ProviderResult[object],
        calls: list[int] | None = None,
    ) -> None:
        """Store a fixed result and optional shared call log."""
        self.name = f"rank {rank}"
        self.rank = rank
        self.result = result
        self.calls = calls if calls is not None else []
        self.reloads = 0

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Return the fixed result and record provider order."""
        del option
        self.calls.append(self.rank)
        # This test double intentionally injects arbitrary provider results so
        # resolver precedence can be exercised independently of coercion.
        result = cast("ProviderResult[T]", self.result)
        return RankedProviderValue(
            self.rank,
            self.durable,
            self.status(),
            result,
        )

    def status(self) -> ProviderStatus:
        """Return synthetic healthy status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Record one propagated reload."""
        self.reloads += 1


def _bool_option(key: str, toml_key: str) -> ConfigOption[bool]:
    """Build a synthetic boolean manifest option."""
    return ConfigOption(
        key=key,
        group="Test",
        summary="test option",
        kind=OptionKind.BOOL,
        default=False,
        toml_keys=("test", toml_key),
    )


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


def test_stale_managed_refresh_cannot_replace_a_newer_resolver_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A delayed resolver refresh must not restore superseded policy."""
    from threading import Event, Thread, current_thread

    from deepagents_code import model_config
    from deepagents_code.configuration import resolver as resolver_module, service
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed.toml"
    managed_path.write_text('[shell]\nallow_list = ["initial"]\n', encoding="utf-8")
    redirect_managed_config(monkeypatch, managed_path)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml")
    service.invalidate_config_sources()
    try:
        option = get_option("shell.allow_list")
        assert option is not None
        resolver = resolver_module.get_config_resolver()
        assert resolver.get(option).value == ["initial"]

        def snapshot(command: str) -> TomlSnapshot:
            return TomlSnapshot(
                {"shell": {"allow_list": [command]}},
                ProviderStatus("managed config", managed_path, ProviderHealth.OK, None),
            )

        stale_fetched = Event()
        release_stale = Event()
        original_get = service.get_managed_snapshot
        monkeypatch.setattr(
            service,
            "_load_managed",
            lambda _path=None: snapshot(
                "stale"
                if current_thread().name == "stale-resolver-refresh"
                else "current"
            ),
        )

        def delayed_get(
            *, refresh: bool = False, path: Path | None = None
        ) -> TomlSnapshot:
            loaded = original_get(refresh=refresh, path=path)
            if refresh and current_thread().name == "stale-resolver-refresh":
                stale_fetched.set()
                assert release_stale.wait(timeout=5)
            return loaded

        monkeypatch.setattr(service, "get_managed_snapshot", delayed_get)
        stale = Thread(
            target=lambda: resolver_module.get_config_resolver(refresh_managed=True),
            name="stale-resolver-refresh",
            daemon=True,
        )
        stale.start()
        assert stale_fetched.wait(timeout=5)

        resolver_module.get_config_resolver(refresh_managed=True)
        assert resolver.get(option).value == ["current"]

        release_stale.set()
        stale.join(timeout=5)
        assert not stale.is_alive()
        assert resolver.get(option).value == ["current"]
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


def test_default_path_write_retains_the_last_enforceable_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preference write must not install parseable but invalid policy."""
    from deepagents_code import model_config
    from deepagents_code.configuration import (
        resolver as resolver_module,
        service,
        writer,
    )
    from unit_tests.conftest import redirect_managed_config

    config_path = tmp_path / "config.toml"
    config_path.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    managed_path = tmp_path / "managed.toml"
    managed_path.write_text(
        '[shell]\nallow_list = ["safe-command"]\n', encoding="utf-8"
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    redirect_managed_config(monkeypatch, managed_path)
    service.invalidate_config_sources()
    try:
        option = get_option("shell.allow_list")
        assert option is not None
        resolver = resolver_module.get_config_resolver()
        assert resolver.get(option).value == ["safe-command"]

        managed_path.write_text("[shell]\nallow_list = 5\n", encoding="utf-8")

        def set_auto(data: dict[str, Any]) -> bool:
            data.setdefault("startup", {})["mode"] = "auto"
            return True

        assert writer.update_user_config(set_auto, config_path=config_path).ok
        assert resolver.get(option).value == ["safe-command"]
    finally:
        service.invalidate_config_sources()


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


class _StubRemoteResponse:
    """Minimal complete HTTPS response for remote managed-policy tests."""

    def __init__(self, payload: bytes) -> None:
        """Model a well-framed 200 carrying `payload`."""
        from email.message import Message
        from io import BytesIO
        from types import SimpleNamespace

        self._stream = BytesIO(payload)
        self.status = 200
        self.chunked = False
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self))
        self.headers = Message()
        self.headers["Content-Length"] = str(len(payload))

    def __enter__(self) -> Self:
        """Return self so production's `with` block works.

        Returns:
            This response.
        """
        return self

    def __exit__(self, *_args: object) -> None:
        """Release the response."""
        self.close()

    def close(self) -> None:
        """Model response cleanup."""

    def read1(self, size: int = -1) -> bytes:
        """Return the next chunk.

        Args:
            size: Most bytes to return.

        Returns:
            The next body chunk.
        """
        return self._stream.read(size)

    def settimeout(self, value: float | None) -> None:
        """Accept the deadline production applies to the socket."""

    def shutdown(self, how: int) -> None:
        """Accept the shutdown production issues on an aborted read."""


def test_failed_remote_refresh_keeps_policy_resolving_in_the_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dead policy host does not drop the tier that is already in force.

    Asserting the service cache is not enough: the resolver is what every
    option read goes through, and it takes its managed tier from a *replacement*
    provider on this path rather than from a `reload`. A replacement that
    installed the failed generation would resolve as "this source declares
    nothing" and let the user tier win.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import (
        providers as providers_module,
        resolver as resolver_module,
        service,
        writer,
    )
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed.toml"
    managed_path.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed_path)
    user_path = tmp_path / "config.toml"
    user_path.write_text('[shell]\nallow_list = ["user-only"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_path)
    healthy = True

    class Opener:
        def open(self, _request: object, *, timeout: float) -> object:
            assert timeout > 0
            if not healthy:
                msg = "policy host is down"
                raise OSError(msg)
            return _StubRemoteResponse(b'[shell]\nallow_list = ["managed-only"]\n')

    monkeypatch.setattr(providers_module, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()
    try:
        option = get_option("shell.allow_list")
        assert option is not None
        resolver = resolver_module.get_config_resolver()
        assert resolver.get(option).value == ["managed-only"]

        healthy = False
        writer.refresh_shared_resolver(user_path)

        resolved = resolver_module.get_config_resolver().get(option)
        # Still the managed value, still attributed to the managed tier.
        assert resolved.value == ["managed-only"]
        assert MANAGED_RANK in resolved.selected_ranks
        assert isinstance(resolved.tier_health[MANAGED_RANK], Found)
    finally:
        service.invalidate_config_sources()


def test_unenforceable_remote_policy_cannot_escalate_through_a_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A published policy that policy itself rejects must not unmanage a session.

    A document that parses is `usable`, so nothing about provider health stops
    it from being installed -- but an enforced key whose value cannot be applied
    resolves as `Invalid` at the managed rank and falls through to the user's
    own value. `startup.mode` is exactly the privilege that must never do that,
    so an in-app preference toggle cannot be a way to reach it.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import (
        providers as providers_module,
        resolver as resolver_module,
        service,
        writer,
    )
    from unit_tests.conftest import redirect_managed_config

    managed_path = tmp_path / "managed.toml"
    managed_path.write_text(
        '[managed_config]\nsource = "https://config.example.com/policy.toml"\n',
        encoding="utf-8",
    )
    redirect_managed_config(monkeypatch, managed_path)
    user_path = tmp_path / "config.toml"
    user_path.write_text('[startup]\nmode = "yolo"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user_path)
    body = b'[startup]\nmode = "manual"\n'

    class Opener:
        def open(self, _request: object, *, timeout: float) -> object:
            assert timeout > 0
            return _StubRemoteResponse(body)

    monkeypatch.setattr(providers_module, "_build_remote_opener", lambda: Opener())
    service.invalidate_config_sources()
    try:
        option = get_option("startup.mode")
        assert option is not None
        resolver = resolver_module.get_config_resolver()
        assert resolver.get(option).value == "manual"

        # The administrator publishes an edit policy cannot apply.
        body = b"[startup]\nmode = 42\n"
        writer.refresh_shared_resolver(user_path)

        resolved = resolver_module.get_config_resolver().get(option)
        assert resolved.value == "manual"
        assert MANAGED_RANK in resolved.selected_ranks
        # And specifically not the user's own escalation.
        assert resolved.value != "yolo"
    finally:
        service.invalidate_config_sources()
