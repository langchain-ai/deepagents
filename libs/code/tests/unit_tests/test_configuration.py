"""Managed configuration provider, precedence, merge, and write tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from deepagents_code.config_manifest import ConfigOption, OptionKind, resolve_scalar
from deepagents_code.configuration.paths import managed_config_path
from deepagents_code.configuration.providers import MappingProvider, TomlFileProvider
from deepagents_code.configuration.resolver import (
    ConfigResolver,
    MergeStrategy,
    merge_toml_tables,
)
from deepagents_code.configuration.service import (
    ManagedConfigError,
    invalidate_config_sources,
    require_healthy_managed_config,
)
from deepagents_code.configuration.types import ProviderHealth
from deepagents_code.configuration.writer import update_user_config

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.mark.parametrize(
    ("platform", "environ", "expected"),
    [
        (
            "darwin",
            {},
            Path("/Library/Application Support/dcode/managed_config.toml"),
        ),
        ("linux", {}, Path("/etc/dcode/managed_config.toml")),
        (
            "win32",
            {"ProgramData": "D:/SharedData"},
            Path("D:/SharedData/dcode/managed_config.toml"),
        ),
        (
            "win32",
            {},
            Path("C:/ProgramData/dcode/managed_config.toml"),
        ),
    ],
)
def test_managed_config_path_is_fixed_by_platform(
    platform: str,
    environ: dict[str, str],
    expected: Path,
) -> None:
    """Managed config uses an administrator-owned OS path."""
    assert managed_config_path(platform=platform, environ=environ) == expected


def test_managed_config_path_windows_ignores_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A redefined `%ProgramData%` must not redirect the managed-config lookup.

    An unprivileged user can set `ProgramData` in their own shell; if the app
    trusted it, the admin policy file could be replaced or silently dropped.
    Production (no injected environ) resolves via the registry or the
    hardcoded default — never the process environment.
    """
    from deepagents_code.configuration import paths

    monkeypatch.setattr(paths, "_program_data_from_registry", lambda: None)
    monkeypatch.setenv("ProgramData", "C:/attacker/fake")
    monkeypatch.setenv("PROGRAMDATA", "C:/attacker/fake")
    assert managed_config_path(platform="win32") == Path(
        "C:/ProgramData/dcode/managed_config.toml"
    )


def test_toml_provider_distinguishes_missing_corrupt_and_empty(
    tmp_path: Path,
) -> None:
    """TOML snapshots keep missing, invalid, and valid-empty states distinct."""
    path = tmp_path / "managed.toml"
    provider = TomlFileProvider("managed config", path, 100)
    assert provider.load().status.health is ProviderHealth.MISSING

    path.write_text("[broken", encoding="utf-8")
    assert provider.load().status.health is ProviderHealth.CORRUPT

    path.write_text("", encoding="utf-8")
    snapshot = provider.load()
    assert snapshot.status.health is ProviderHealth.OK
    assert snapshot.data == {}


def test_toml_provider_marks_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operating-system read failure is not mistaken for a missing file."""
    path = tmp_path / "managed.toml"
    provider = TomlFileProvider("managed config", path, 100)

    def denied(*_args: object, **_kwargs: object) -> Iterator[bytes]:
        raise PermissionError

    monkeypatch.setattr(Path, "open", denied)
    assert provider.load().status.health is ProviderHealth.UNREADABLE


def test_managed_provider_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present corrupt managed policy produces a real startup-gate error."""
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)
    finally:
        invalidate_config_sources()


def test_resolver_uses_ranked_replace_union_and_deep_merge() -> None:
    """Provider resolution preserves explicit empties and strategy semantics."""
    managed = MappingProvider(
        "managed config",
        100,
        {
            "scalar": False,
            "deny": ["managed"],
            "table": {"nested": {"managed": 2}},
        },
    )
    user = MappingProvider(
        "config.toml",
        500,
        {
            "scalar": True,
            "deny": ["user"],
            "table": {"nested": {"user": 1}},
        },
    )
    resolver = ConfigResolver((user, managed))

    assert resolver.resolve("scalar", default=None).value is False
    assert resolver.resolve("deny", default=[], strategy=MergeStrategy.UNION).value == [
        "user",
        "managed",
    ]
    assert resolver.resolve(
        "table", default={}, strategy=MergeStrategy.DEEP_MERGE
    ).value == {"nested": {"user": 1, "managed": 2}}


def test_deep_merge_tracks_managed_leaf_provenance() -> None:
    """Ordinary tables merge per leaf while managed values win conflicts."""
    merged, provenance = merge_toml_tables(
        {"providers": {"acme": {"api_url": "user", "model": "small"}}},
        {"providers": {"acme": {"api_url": "managed"}}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged == {"providers": {"acme": {"api_url": "managed", "model": "small"}}}
    assert provenance == {
        "providers.acme.api_url": "managed config",
        "providers.acme.model": "config.toml",
    }


def test_resolve_scalar_managed_beats_env_and_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid managed scalar outranks both environment and user TOML."""
    option = ConfigOption(
        key="feature.enabled",
        group="Test",
        summary="test",
        default=False,
        env_var="FEATURE_ENABLED",
        toml_keys=("feature", "enabled"),
        kind=OptionKind.BOOL,
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_FEATURE_ENABLED", "true")
    assert resolve_scalar(
        option,
        toml_data={"feature": {"enabled": True}},
        managed_toml_data={"feature": {"enabled": False}},
    ) == (False, "managed config")


def test_resolve_scalar_skips_one_wrong_typed_managed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed key falls through without discarding valid siblings."""
    option = ConfigOption(
        key="feature.enabled",
        group="Test",
        summary="test",
        default=False,
        env_var="FEATURE_ENABLED",
        toml_keys=("feature", "enabled"),
        kind=OptionKind.BOOL,
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_FEATURE_ENABLED", "true")
    assert resolve_scalar(
        option,
        toml_data={"feature": {"enabled": False}},
        managed_toml_data={"feature": {"enabled": "not-a-bool", "other": 1}},
    ) == (True, "env (DEEPAGENTS_CODE_FEATURE_ENABLED)")


def test_managed_allow_list_empty_replaces_user_grants() -> None:
    """An explicit managed empty grant list enforces lockdown."""
    option = ConfigOption(
        key="mcp.approved_servers",
        group="Test",
        summary="test",
        toml_keys=("mcp", "approved_servers"),
        kind=OptionKind.STRUCTURED,
    )
    assert resolve_scalar(
        option,
        toml_data={"mcp": {"approved_servers": ["user-grant"]}},
        managed_toml_data={"mcp": {"approved_servers": []}},
    ) == ([], "managed config")


def test_user_writer_never_modifies_managed_config(tmp_path: Path) -> None:
    """Central writes preserve sibling user tables and leave policy untouched."""
    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("[ui]\ntheme = 'user'\n", encoding="utf-8")
    managed.write_text("[ui]\ntheme = 'managed'\n", encoding="utf-8")

    def mutate(data: dict[str, object]) -> bool:
        data["runtime"] = {"recursion_limit": 42}
        return True

    result = update_user_config(mutate, config_path=user)
    assert result.ok
    assert result.changed
    assert 'theme = "user"' in user.read_text(encoding="utf-8")
    assert "recursion_limit = 42" in user.read_text(encoding="utf-8")
    assert managed.read_text(encoding="utf-8") == "[ui]\ntheme = 'managed'\n"


def test_user_table_cannot_shadow_managed_scalar() -> None:
    """A shape-colliding user table yields to a managed scalar and provenance.

    Regression: skipping the managed scalar let typed readers reject the
    surviving user table and fall back to the built-in default, so the managed
    value was never enforced.
    """
    merged, provenance = merge_toml_tables(
        {"threads": {"relative_time": {"user": "table"}}, "other": {"a": 1}},
        {"threads": {"relative_time": False}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged == {"threads": {"relative_time": False}, "other": {"a": 1}}
    assert provenance["threads.relative_time"] == "managed config"


def test_managed_wrong_typed_table_keeps_valid_user_siblings() -> None:
    """A malformed managed table does not erase a valid lower table."""
    merged, provenance = merge_toml_tables(
        {"sandboxes": {"default": "user", "providers": {"acme": {"token": "x"}}}},
        {"sandboxes": "not-a-table", "runtime": {"recursion_limit": 42}},
        lower_source="config.toml",
        higher_source="managed config",
    )
    assert merged["sandboxes"] == {
        "default": "user",
        "providers": {"acme": {"token": "x"}},
    }
    assert merged["runtime"] == {"recursion_limit": 42}
    assert provenance["sandboxes.default"] == "config.toml"
    assert provenance["runtime.recursion_limit"] == "managed config"


def test_managed_wrong_typed_deny_list_keeps_valid_user_denies() -> None:
    """An invalid managed deny value cannot discard valid lower denials."""
    sources_lower = {"mcp": {"disabled_servers": ["user-denied"]}}
    sources_higher = {"mcp": {"disabled_servers": "not-a-list"}}
    merged, _ = merge_toml_tables(
        sources_lower,
        sources_higher,
        lower_source="config.toml",
        higher_source="managed config",
        union_paths=frozenset({("mcp", "disabled_servers")}),
    )
    assert merged["mcp"]["disabled_servers"] == ["user-denied"]


def test_managed_mcp_lockdown_replaces_grants_and_unions_denies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed empty grants lock down approvals while every deny source accumulates."""
    from deepagents_code import _env_vars, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        '[mcp]\ndisabled_project_servers = ["user-denied"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[mcp]\nenabled_project_server_approvals = []\n"
        'disabled_project_servers = ["managed-denied"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    monkeypatch.setenv(
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "env-granted",
    )
    monkeypatch.setenv(_env_vars.DISABLED_PROJECT_MCP_SERVERS, "env-denied")
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.enabled == frozenset()
    assert trust.approvals == frozenset()
    assert trust.disabled == frozenset({"user-denied", "managed-denied", "env-denied"})


def test_wrong_typed_managed_mcp_allow_list_does_not_mask_env_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed grant key is skipped instead of becoming lockdown."""
    from deepagents_code import _env_vars, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\nenabled_project_server_approvals = "wrong"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    monkeypatch.setenv(
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        "env-granted",
    )
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
    finally:
        service.invalidate_config_sources()

    assert trust.enabled == frozenset({"env-granted"})


def test_custom_mcp_config_path_is_isolated_from_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit custom config paths retain their test and embedding isolation seam."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "custom.toml"
    user.write_text(
        '[mcp]\ndisabled_project_servers = ["custom-denied"]\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\ndisabled_project_servers = ["managed-denied"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists(user)
    finally:
        service.invalidate_config_sources()

    assert trust.disabled == frozenset({"custom-denied"})


def test_managed_structured_preferences_reach_runtime_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed thread and warning tables override default-path runtime reads."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        "[threads]\nrelative_time = true\n[warnings]\nsuppress = []\n",
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[threads]\nrelative_time = false\n[warnings]\nsuppress = ["ripgrep"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
        assert model_config.is_warning_suppressed("ripgrep") is True
        assert model_config.load_thread_relative_time(user) is True
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()


def test_managed_models_survive_an_unreadable_default_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed models remain available when the user config has no read bits."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[models]\ndefault = "user-model"\n', encoding="utf-8")
    user.chmod(0o000)
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models]\ndefault = "managed-model"\n'
        '[models.providers.acme]\nmodels = ["managed-model"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = model_config.ModelConfig.load()
        assert config.default_model == "managed-model"
        assert config.providers["acme"]["models"] == ["managed-model"]
    finally:
        user.chmod(0o644)
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_skill_dirs_outrank_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed skill containment roots cannot be replaced through the environment."""
    from deepagents_code import config, model_config
    from deepagents_code._env_vars import EXTRA_SKILLS_DIRS
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    managed_dir = tmp_path / "managed-skills"
    env_dir = tmp_path / "env-skills"
    managed.write_text(
        f'[skills]\nextra_allowed_dirs = ["{managed_dir}"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    monkeypatch.setenv(EXTRA_SKILLS_DIRS, str(env_dir))
    service.invalidate_config_sources()
    try:
        settings = config.Settings.from_environment(start_path=tmp_path)
        assert settings.extra_skills_dirs == [managed_dir]
    finally:
        service.invalidate_config_sources()


def test_managed_scalar_enforced_over_user_table_shape_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user `[threads.relative_time]` table cannot neutralize managed policy."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[threads.relative_time]\nnested = true\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[threads]\nrelative_time = false\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()
