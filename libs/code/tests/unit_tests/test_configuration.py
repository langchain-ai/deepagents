"""Managed configuration provider, precedence, merge, and write tests."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from deepagents_code.config_manifest import ConfigOption, OptionKind, resolve_scalar
from deepagents_code.configuration.paths import managed_config_path
from deepagents_code.configuration.providers import TomlFileProvider
from deepagents_code.configuration.resolver import merge_toml_tables
from deepagents_code.configuration.service import (
    ConfigSources,
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


def test_registry_program_data_outranks_a_poisoned_process_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful registry read wins over a redefined `%ProgramData%`.

    The sibling test proves the *fallback* ignores the environment. This is the
    real attack: a machine with a relocated ProgramData, where the user exports
    a path of their own and the registry holds the true one.
    """
    from deepagents_code.configuration import paths

    monkeypatch.setattr(paths, "_program_data_from_registry", lambda: "D:/RealShared")
    monkeypatch.setenv("ProgramData", "C:/attacker/fake")
    monkeypatch.setenv("PROGRAMDATA", "C:/attacker/fake")
    assert managed_config_path(platform="win32") == Path(
        "D:/RealShared/dcode/managed_config.toml"
    )


def test_user_config_writers_share_one_lock_object() -> None:
    """A second lock for the same file would not mutually exclude.

    The hazard is the whole-file replace, so a `[effort]` write in
    `model_config` and a `[ui]` write through the shared writer must contend on
    the same object. This is the invariant those docstrings rest on.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration.writer import USER_CONFIG_WRITE_LOCK

    assert model_config._config_write_lock is USER_CONFIG_WRITE_LOCK


def test_toml_provider_distinguishes_missing_corrupt_and_empty(
    tmp_path: Path,
) -> None:
    """TOML snapshots keep missing, invalid, and valid-empty states distinct."""
    path = tmp_path / "managed.toml"
    provider = TomlFileProvider("managed config", path)
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
    provider = TomlFileProvider("managed config", path)

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


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root reads a 0o000 file, so the unreadable case cannot be staged",
)
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


def test_invalid_managed_scalar_keeps_valid_user_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed scalar does not erase a valid user preference."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[threads]\nrelative_time = false\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[threads]\nrelative_time = "invalid"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.invalidate_thread_config_cache()


def test_managed_table_at_scalar_path_keeps_valid_user_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed managed table cannot replace a user model specification."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text('[models]\ndefault = "user:model"\n', encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[models.default]\ninvalid = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        assert model_config.ModelConfig.load().default_model == "user:model"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


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


@pytest.mark.parametrize(
    "managed_toml",
    [
        pytest.param('threads = "bad"\n', id="manifest-parent"),
        pytest.param('themes = "bad"\n', id="structured-table"),
        pytest.param('mcp = "bad"\n', id="security-adjacent-parent"),
    ],
)
def test_non_table_known_managed_section_stops_startup(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar cannot replace a known managed section and erase user settings."""
    from deepagents_code.configuration import service
    from deepagents_code.configuration.service import ManagedPolicyError

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedPolicyError):
            require_healthy_managed_config(refresh=True)
    finally:
        service.invalidate_config_sources()


def _sources(managed: dict[str, object], user: dict[str, object]) -> ConfigSources:
    """Build `ConfigSources` from two literal tables, both reported healthy."""
    from deepagents_code.configuration.types import (
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    def snapshot(name: str, data: dict[str, object]) -> TomlSnapshot:
        return TomlSnapshot(
            data,
            ProviderStatus(name, None, ProviderHealth.OK),
        )

    return ConfigSources(
        managed=snapshot("managed config", managed),
        user=snapshot("config.toml", user),
    )


@pytest.mark.parametrize(
    "colliding_user_value",
    [
        pytest.param({"nested": True}, id="scalar-only-table"),
        pytest.param({"nested": {}}, id="table-with-empty-table"),
        pytest.param({"nested": {"deep": True}}, id="table-with-nested-table"),
        pytest.param({"a": {"b": {"c": 1}}}, id="deeply-nested-table"),
    ],
)
def test_managed_scalar_beats_a_user_table_at_any_depth(
    colliding_user_value: dict[str, object],
) -> None:
    """A valid managed scalar wins however deeply the user table nests.

    Regression: the merge kept any user table that held a non-empty nested
    table, so adding one level of nesting to `[threads.relative_time]` let a
    user defeat a managed `relative_time = false`. Typed readers then rejected
    the surviving table and fell back to the built-in default, which silently
    voided administrator policy.

    Driven through `ConfigSources.merged` on purpose: calling the merger
    directly with a hand-passed validator would stay green if `merged` stopped
    passing one.
    """
    sources = _sources(
        {"threads": {"relative_time": False}},
        {"threads": {"relative_time": colliding_user_value}},
    )
    merged, provenance = sources.merged()
    assert merged == {"threads": {"relative_time": False}}
    assert provenance["threads.relative_time"] == "managed config"


def test_invalid_managed_scalar_keeps_a_nested_user_table() -> None:
    """A wrong-typed managed scalar must not discard a valid user subtree."""
    sources = _sources(
        {"threads": {"relative_time": "not-a-bool"}},
        {"threads": {"relative_time": {"a": {"b": 1}}}},
    )
    merged, _ = sources.merged()
    assert merged == {"threads": {"relative_time": {"a": {"b": 1}}}}


def test_managed_policy_survives_a_corrupt_default_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt user config drops only the user layer, never managed policy.

    Regression: the shared reader raised `OSError` on an unusable user file
    before consulting the managed layer, so every caller fell back to built-in
    defaults. The user owns that file, which made one invalid byte an
    unprivileged way to switch administrator policy off.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("this is not [valid toml\n", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[startup]\nmode = "auto"\n[threads]\nrelative_time = false\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    model_config.invalidate_thread_config_cache()
    try:
        assert model_config.load_startup_mode() == "auto"
        assert model_config.load_thread_relative_time() is False
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()
        model_config.invalidate_thread_config_cache()


def test_malformed_managed_deny_list_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong-typed managed deny list reports an error instead of allowing all.

    Regression: the managed branch discarded the malformed flag, so an
    administrator typo produced an empty deny set with no signal, while the
    same typo in the user file failed closed.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[mcp]\ndisabled_project_servers = 5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
        assert "disabled_project_servers" in trust.read_error
    finally:
        service.invalidate_config_sources()


def test_non_table_managed_mcp_section_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar managed `[mcp]` cannot silently void the deny list."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('mcp = "locked"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("all", ["__ALL__"], id="string-all"),
        pytest.param(["all"], ["__ALL__"], id="list-all"),
    ],
)
def test_shell_allow_list_sentinels_agree_across_spellings(
    raw: object,
    expected: list[str],
) -> None:
    """The TOML array honors the same sentinels as the comma-separated string.

    Regression: the list branch bypassed the parser, so a managed
    `allow_list = ["all"]` permitted one literal command named `all` instead
    of every command, silently inverting the administrator's intent.
    """
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, raw, source="managed config") == expected


def test_shell_allow_list_rejects_all_combined_with_commands() -> None:
    """`all` stays exclusive in the array form, matching the string form."""
    from deepagents_code.config_manifest import _INVALID, _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, ["all", "git"], source="managed config") is _INVALID


def test_shell_allow_list_array_expands_recommended() -> None:
    """A `recommended` element expands to the curated safe set."""
    from deepagents_code.config import RECOMMENDED_SAFE_SHELL_COMMANDS
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    resolved = _coerce_toml(option, ["recommended", "make"], source="managed config")
    assert isinstance(resolved, list)
    assert set(RECOMMENDED_SAFE_SHELL_COMMANDS) <= set(resolved)
    assert "make" in resolved


def test_shell_allow_list_array_preserves_comma_in_entry() -> None:
    """An element containing a comma stays a single command.

    Regression: the TOML array was joined with commas and reparsed as the
    string form, so `["my,tool"]` resolved to `["my", "tool"]` — auto-approving
    two executables the administrator never listed.
    """
    from deepagents_code.config_manifest import _coerce_toml, get_option

    option = get_option("shell.allow_list")
    assert option is not None
    assert _coerce_toml(option, ["my,tool", "git"], source="managed config") == [
        "my,tool",
        "git",
    ]


def test_corrupt_managed_config_does_not_empty_the_mcp_deny_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken managed file must not re-enable administrator-denied servers.

    An unusable snapshot carries an empty table, so returning it would read as
    "nothing is denied" and silently undo every managed deny.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            mcp_disabled.get_disabled_servers()
        # The user-facing predicate must fail closed rather than propagate.
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


@pytest.mark.parametrize(
    "value",
    ['"github"', "5", "true", "{ github = true }"],
    ids=["string", "int", "bool", "table"],
)
def test_unusable_managed_deny_list_fails_closed(
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed deny list that cannot hold names must deny, never allow.

    Regression: `_coerce_entries` reported every non-list as "key absent", so
    the lookup fell through to an empty set with no log at all and every
    administrator-denied server started.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(f"[mcp]\ndisabled_servers = {value}\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        # The file parses, so the startup gate cannot catch this.
        require_healthy_managed_config(refresh=True)
        if value == '"github"':
            # A bare string parses as the comma-separated form, matching
            # `[mcp].disabled_project_servers`, so it denies rather than fails.
            assert mcp_disabled.get_disabled_servers() == {"github"}
        else:
            with pytest.raises(ManagedConfigError):
                mcp_disabled.get_disabled_servers()
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_managed_deny_list_string_splits_on_commas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`disabled_servers = "a, b"` denies two servers, not one bogus name."""
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[mcp]\ndisabled_servers = "github, linear"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"github", "linear"}
    finally:
        service.invalidate_config_sources()


def test_non_table_managed_mcp_section_cannot_void_the_deny_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scalar managed `[mcp]` shadows the deny list, so it must fail closed."""
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('mcp = "github"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(ManagedConfigError):
            mcp_disabled.get_disabled_servers()
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_failed_reload_keeps_the_last_healthy_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed file that breaks mid-session must not empty the cached policy.

    Refreshing used to cache the failed load, so every later reader saw an
    empty managed table and treated enforced denies as absent.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[mcp]\ndisabled_servers = ["denied"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"denied"}

        managed.write_text("[broken", encoding="utf-8")
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)

        # The failed refresh reported the error but left policy in force.
        assert mcp_disabled.get_disabled_servers() == {"denied"}
    finally:
        service.invalidate_config_sources()


def test_rejected_reload_keeps_the_last_enforceable_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload rejected on policy grounds must not replace the cached policy.

    Regression: a parseable-but-unenforceable file has health `OK`, so
    `get_managed_snapshot(refresh=True)` cached it before
    `require_healthy_managed_config` rejected it. The reload kept the
    previous settings, but the process-wide cache already held the rejected
    file, so a later non-refresh reader observed it and re-enabled a managed
    MCP deny the edit had removed.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        'startup.mode = "manual"\n[mcp]\ndisabled_servers = ["denied"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        assert mcp_disabled.get_disabled_servers() == {"denied"}

        # Enforced-key violation plus removal of the managed deny, in one
        # parseable edit. The reload is blocked, but the deny must stay.
        managed.write_text("startup.mode = 5\n", encoding="utf-8")
        with pytest.raises(ManagedConfigError):
            require_healthy_managed_config(refresh=True)

        assert mcp_disabled.get_disabled_servers() == {"denied"}
    finally:
        service.invalidate_config_sources()


def test_union_paths_rebase_onto_an_option_subtree() -> None:
    """Deny-list paths must still match when a merge starts below the root."""
    from deepagents_code.configuration.service import UNION_PATHS, union_paths_under

    assert ("disabled_servers",) in union_paths_under(("mcp",))
    assert union_paths_under(()) == UNION_PATHS
    assert union_paths_under(("models",)) == frozenset()


def test_merged_unions_deny_lists_across_layers() -> None:
    """Both layers' deny entries survive, and provenance names both sources.

    Regression: nothing drove the merger's union branch, so a change to the
    `UNION_PATHS` match or the dedupe would have replaced a managed deny list
    with the user's — a fail-open — while every test stayed green.
    """
    sources = _sources(
        {"mcp": {"disabled_servers": ["managed-denied", "shared"]}},
        {"mcp": {"disabled_servers": ["user-denied", "shared"]}},
    )
    data, provenance = sources.merged()
    assert sorted(data["mcp"]["disabled_servers"]) == [
        "managed-denied",
        "shared",
        "user-denied",
    ]
    assert provenance["mcp.disabled_servers"] == "managed config + config.toml"


def test_merged_keeps_a_managed_deny_list_when_the_user_has_none() -> None:
    """A user layer with no deny list cannot dilute the managed one."""
    sources = _sources({"mcp": {"disabled_servers": ["denied"]}}, {"mcp": {}})
    data, provenance = sources.merged()
    assert data["mcp"]["disabled_servers"] == ["denied"]
    assert provenance["mcp.disabled_servers"] == "managed config"


def test_merged_purges_provenance_for_leaves_the_merge_removed() -> None:
    """A replaced user table leaves no provenance behind.

    Regression: the nested merge kept the parent-scope entry, so
    `dcode config --json --verbose` reported `threads.relative_time.x` as
    user-controlled after a managed scalar had removed that path.
    """
    sources = _sources(
        {"threads": {"relative_time": False}},
        {"threads": {"relative_time": {"x": 1}}},
    )
    data, provenance = sources.merged()
    assert data["threads"]["relative_time"] is False
    assert provenance == {"threads.relative_time": "managed config"}


def test_writer_reports_a_mis_encoded_config_instead_of_raising(
    tmp_path: Path,
) -> None:
    """A config that is not UTF-8 is reported, not raised.

    `tomllib` decodes the bytes itself, so the failure is a
    `UnicodeDecodeError`, which the read guard did not catch. It escaped past
    every caller's error handling and lost the real reason.
    """
    from deepagents_code.configuration.writer import update_user_config

    target = tmp_path / "config.toml"
    target.write_bytes('[ui]\ntheme = "dark"\n'.encode("utf-16"))
    result = update_user_config(
        lambda data: bool(data.setdefault("ui", {})), config_path=target
    )
    assert result.ok is False
    assert result.error is not None
    assert str(target) in result.error


def test_write_result_rejects_a_failure_with_no_detail() -> None:
    """Callers branch on `ok` alone, so a failure must carry something to act on."""
    from deepagents_code.configuration.writer import WriteResult

    with pytest.raises(ValueError, match="error detail"):
        WriteResult(False, False, None)


def test_write_result_rejects_a_change_on_a_failed_write() -> None:
    """A failed write cannot report that it changed the file."""
    from deepagents_code.configuration.writer import WriteResult

    with pytest.raises(ValueError, match="cannot have changed"):
        WriteResult(False, True, "boom")


def test_write_result_accepts_the_three_real_outcomes() -> None:
    """The guard must not reject an outcome the writer actually returns."""
    from deepagents_code.configuration.writer import WriteResult

    assert WriteResult(True, True).changed is True
    assert WriteResult(True, False).changed is False
    assert WriteResult(False, False, "boom").ok is False


def test_writer_reports_caller_bugs_separately_from_disk_errors(
    tmp_path: Path,
) -> None:
    """A bug in the caller's closure must not read as a filesystem failure."""
    config_path = tmp_path / "config.toml"

    def broken(_data: dict[str, object]) -> bool:
        msg = "bug in caller"
        raise TypeError(msg)

    with pytest.raises(TypeError, match="bug in caller"):
        update_user_config(broken, config_path=config_path)

    unchanged = update_user_config(lambda _data: False, config_path=config_path)
    assert unchanged.ok is True
    assert unchanged.changed is False
    assert not config_path.exists()


def test_writer_reports_an_unparseable_existing_config_as_an_error(
    tmp_path: Path,
) -> None:
    """A corrupt file is refused so sibling sections are not truncated."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[broken", encoding="utf-8")

    result = update_user_config(
        lambda data: data.setdefault("ui", {}).update(theme="x") or True,
        config_path=config_path,
    )
    assert result.ok is False
    assert result.error is not None
    assert "could not update" in result.error


def _managed_policy_args() -> argparse.Namespace:
    """Return a namespace shaped like the parsed agent-launch arguments.

    `sandbox` is `"none"` rather than `None` because that is what `parse_args`
    produces for an omitted `--sandbox` (the argument declares
    `default="none"`). Using `None` here made the "does not force a sandbox"
    regression tests pass against code that forced one.
    """
    return argparse.Namespace(
        model=None,
        auto_classifier_model=None,
        interpreter=None,
        recursion_limit=None,
        sandbox="none",
        interpreter_tools=None,
        shell_allow_list="all",
        auto_approve=False,
        yolo=True,
    )


@pytest.mark.parametrize(
    ("managed_toml", "expected_exit"),
    [
        ('[startup]\nmode = "YOLO"\n', True),
        ("[runtime]\nrecursion_limit = 3\n", True),
        ("[shell]\nallow_list = 5\n", True),
        ("[skills]\nextra_allowed_dirs = 5\n", True),
        ("[models]\nauto_classifier = 4\n", True),
        ("[sandboxes]\ndefault = 5\n", True),
        ("[interpreter]\nptc = 5\n", True),
        ("[interpreter]\nenable_interpreter = 5\n", True),
        # A scalar where the table belongs shadows the key it should hold.
        ('startup = "manual"\n', True),
        ('shell = "ls"\n', True),
        ("skills = 5\n", True),
        ('[startup]\nmode = "manual"\n', False),
        ("[runtime]\nrecursion_limit = 500\n", False),
    ],
    ids=[
        "bad-startup-mode",
        "out-of-range-limit",
        "bad-shell-allow-list",
        "bad-skills-dirs",
        "bad-auto-classifier",
        "bad-sandbox",
        "bad-ptc",
        "bad-interpreter-toggle",
        "shadowed-startup",
        "shadowed-shell",
        "shadowed-skills",
        "valid-startup-mode",
        "valid-limit",
    ],
)
def test_rejected_managed_privilege_value_stops_the_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    managed_toml: str,
    expected_exit: bool,
) -> None:
    """A privilege key the manifest rejects must not resolve in the user's favor.

    Skipping the value left `--yolo` and `--shell-allow-list all` in force, so
    an administrator typo granted exactly the escalation policy forbade.

    Regression: a *shadowed* path (`startup = "manual"` instead of `[startup]`
    plus `mode`) read as "the administrator wrote nothing", so the same typo
    one level up still granted the escalation, silently.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        if expected_exit:
            with pytest.raises(SystemExit) as excinfo:
                main._apply_managed_runtime_policy(args)
            assert excinfo.value.code == 78
        else:
            main._apply_managed_runtime_policy(args)
    finally:
        service.invalidate_config_sources()


def test_managed_auto_mode_does_not_set_the_headless_incompatible_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed policy revokes flags; it never sets `--auto-approve`.

    Regression: assigning the flag positively made every headless launch exit 2
    with "--auto-approve is only supported in interactive mode", naming a flag
    the user never passed. `_resolve_approval_mode` already ends at
    `coerce_approval_mode(load_startup_mode())`, which reads merged managed
    policy, so the positive value needs no flag.
    """
    from deepagents_code import main, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "auto"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_policy(args)
        assert args.auto_approve is False
        assert args.yolo is False
        # The mode still reaches the runtime through the merged config.
        assert model_config.load_startup_mode() == "auto"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_sandbox_default_does_not_force_a_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`sandboxes.default` names a backend; it does not turn sandboxing on.

    Assigning it unconditionally forced every launch into a remote sandbox,
    which is not what the key documents. Both spellings of "no sandbox" have to
    be left alone: an omitted `--sandbox` arrives as `"none"` from argparse, and
    an explicit `--sandbox none` arrives the same way.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[sandboxes]\ndefault = "modal"\n', encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        for unsandboxed in ("none", None):
            args = _managed_policy_args()
            args.sandbox = unsandboxed
            main._apply_managed_runtime_policy(args)
            assert args.sandbox == unsandboxed
    finally:
        service.invalidate_config_sources()


def test_unavailable_managed_sandbox_leaves_an_unsandboxed_launch_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch that asked for no sandbox must not die on a managed backend.

    Regression: the guard checked only `None`, so a bare `dcode` reached the
    availability check and exited 78 over a backend it was never going to use.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[sandboxes]\ndefault = "not-a-real-provider"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_policy(args)
        assert args.sandbox == "none"
    finally:
        service.invalidate_config_sources()


def test_unavailable_managed_sandbox_stops_a_sandboxed_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed backend no provider answers to must not reach the factory.

    `parse_args` validates `--sandbox`, but it runs before managed policy is
    applied, so the managed value skipped `is_available` entirely.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[sandboxes]\ndefault = "not-a-real-provider"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    args.sandbox = "not-a-real-provider"
    try:
        with pytest.raises(SystemExit) as excinfo:
            main._apply_managed_runtime_policy(args)
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


def _managed_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed_toml: str
) -> Path:
    """Point every layer at `tmp_path` with `managed_toml` as managed policy.

    Returns:
        The managed file path.
    """
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    return managed


@pytest.mark.parametrize(
    "argv",
    [
        ["dcode", "config", "path"],
        ["dcode", "doctor"],
        ["dcode", "--help"],
        ["dcode", "help"],
    ],
    ids=["config", "doctor", "help-flag", "help-command"],
)
def test_diagnostic_commands_run_with_unusable_managed_policy(
    argv: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The administrator must keep the tools that explain a broken policy file.

    If the startup gate moved above these early returns, the only commands that
    report the managed path and its parse health would be the ones the broken
    file blocks.
    """
    from deepagents_code import main
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    monkeypatch.setattr(sys, "argv", argv)
    # Some of these exit and some return; neither may be the policy gate.
    exit_code: object = None
    try:
        try:
            main.cli_main()
        except SystemExit as exc:
            exit_code = exc.code
    finally:
        service.invalidate_config_sources()
    assert exit_code != 78


def test_agent_launch_commands_stop_on_unusable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that runs the agent must not start without enforceable policy."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    monkeypatch.setattr(sys, "argv", ["dcode", "tools", "list"])
    try:
        with pytest.raises(SystemExit) as excinfo:
            main.cli_main()
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


@pytest.mark.self_managed_update_check
@pytest.mark.parametrize(
    "managed_toml",
    ["[broken", "[update]\ncheck = 5\n"],
    ids=["unparseable", "wrong-type"],
)
def test_update_checks_fail_closed_on_unusable_managed_policy(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feature that reaches the network turns off when policy cannot be read.

    A wrong type falls through to the lower layers instead, because the value
    itself is what could not be read — not the file.
    """
    from deepagents_code import update_check
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, managed_toml)
    try:
        if managed_toml == "[broken":
            assert update_check.is_update_check_enabled() is False
            assert update_check.is_auto_update_enabled() is False
        else:
            # The file is readable, so only this key is ignored.
            assert update_check.is_update_check_enabled() is True
    finally:
        service.invalidate_config_sources()


@pytest.mark.self_managed_update_check
def test_managed_update_policy_outranks_the_user_preference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[update].check = false` cannot be re-enabled locally."""
    from deepagents_code import update_check
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[update]\ncheck = false\n")
    try:
        assert update_check.is_update_check_enabled() is False
    finally:
        service.invalidate_config_sources()


def test_reenabling_a_managed_denied_server_reports_the_shadow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save succeeds, and the user is told policy keeps the server off."""
    from deepagents_code import mcp_disabled
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[mcp]\ndisabled_servers = ["github"]\n')
    try:
        ok, detail = mcp_disabled.set_server_disabled("github", False)
        assert ok is True
        assert detail is not None
        assert mcp_disabled.is_server_disabled("github") is True
    finally:
        service.invalidate_config_sources()


def test_reenabling_a_server_fails_closed_when_policy_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-enabling must not proceed while the deny list cannot be read."""
    from deepagents_code import mcp_disabled
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        ok, detail = mcp_disabled.set_server_disabled("github", False)
        assert ok is False
        assert detail is not None
    finally:
        service.invalidate_config_sources()


def test_managed_sandbox_settings_survive_an_unusable_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sandbox selection is a containment boundary, so policy outlives a typo."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service
    from deepagents_code.integrations import sandbox_config
    from deepagents_code.integrations.sandbox_config import SandboxConfig

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[sandboxes]\ndefault = "modal"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    # `sandbox_config` binds the default path at import, so patching
    # `model_config` alone leaves it reading the real user config.
    monkeypatch.setattr(sandbox_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = SandboxConfig.load()
        assert config.default == "modal"
        # The user layer failed, and that is reported without dropping policy.
        assert config.parse_error is not None
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_managed_async_subagents_survive_an_unusable_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[async_subagents]` table still defines its agents."""
    from deepagents_code import model_config
    from deepagents_code.agent import load_async_subagents
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'url = "https://example.langsmith.dev"\n'
        'graph_id = "agent"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        subagents = load_async_subagents()
        assert [entry["name"] for entry in subagents] == ["researcher"]
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_doctor_reports_managed_parse_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` must explain the file that just stopped the launch."""
    from deepagents_code.configuration import service
    from deepagents_code.doctor import _managed_config_diagnostic

    managed = _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        item = _managed_config_diagnostic()
        assert item.ok is False
        assert str(managed) in item.value
        assert "administrator" in item.value
    finally:
        service.invalidate_config_sources()


def test_managed_path_fallback_is_reported_as_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed registry read must not look like "no policy deployed".

    The guessed path holds no file, which is the same state as a machine with
    no managed policy at all, so the reason has to reach the status detail.
    """
    from deepagents_code.configuration import paths, service

    monkeypatch.setattr(paths._path_state, "registry_fallback", "registry unreadable")
    monkeypatch.setattr(
        service, "managed_config_path", lambda: Path("/nonexistent/managed.toml")
    )
    service.invalidate_config_sources()
    try:
        status = service.managed_config_status(refresh=True)
        assert status.health is ProviderHealth.MISSING
        assert status.detail == "registry unreadable"
    finally:
        service.invalidate_config_sources()


def test_default_read_includes_policy_and_an_explicit_path_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a default read carries policy, and that is not a caller's choice.

    An excluded managed layer reports `MISSING` with an empty table, which is
    indistinguishable from a machine with no policy installed. Deriving it from
    `user_path` keeps that state out of reach of a keyword argument.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\ndefault = "managed:model"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        default_read = service.get_config_sources()
        assert default_read.managed.data != {}
        assert default_read.merged()[0]["models"]["default"] == "managed:model"

        isolated = service.get_config_sources(user_path=user)
        assert isolated.managed.data == {}
        assert "models" not in isolated.merged()[0]
    finally:
        service.invalidate_config_sources()


async def test_unreadable_managed_policy_disables_every_mcp_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deny decision must reach "which servers start", not just a predicate.

    `is_server_disabled` failing closed is not enough: this is the call that
    turns policy into running processes.
    """
    import json
    from unittest.mock import AsyncMock, MagicMock

    from deepagents_code import mcp_tools
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    explicit = tmp_path / "mcp.json"
    explicit.write_text(
        json.dumps({"mcpServers": {"github": {"command": "npx", "args": []}}}),
        encoding="utf-8",
    )
    load = AsyncMock(return_value=([], None, []))
    monkeypatch.setattr(mcp_tools, "_load_tools_from_config", load)
    monkeypatch.setattr(mcp_tools, "discover_mcp_configs", MagicMock(return_value=[]))
    try:
        tools, manager, infos = await mcp_tools.resolve_and_load_mcp_tools(
            explicit_config_path=str(explicit),
            trust_project_mcp=True,
        )
        assert tools == []
        assert manager is None
        assert [info.status for info in infos] == ["disabled"]
        # No server may reach the loader at all.
        load.assert_not_called()
    finally:
        service.invalidate_config_sources()


async def test_server_mode_refuses_to_build_a_graph_without_enforceable_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The server gate is a second entry point with the same duty as the CLI."""
    from deepagents_code import server_graph
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    failures: list[BaseException] = []
    monkeypatch.setattr(server_graph, "emit_startup_failure", failures.append)
    try:
        factory = server_graph._build_graph_factory(builder=None)
        with pytest.raises(SystemExit) as excinfo:
            await factory()
        assert excinfo.value.code == 1
        assert isinstance(failures[0], ManagedConfigError)
    finally:
        service.invalidate_config_sources()


def test_corrupt_managed_policy_fails_the_mcp_trust_lists_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable managed file must set `read_error`, keyed on its health."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[broken")
    try:
        trust = model_config.load_mcp_server_trust_lists()
        assert trust.read_error is not None
    finally:
        service.invalidate_config_sources()


def test_managed_structured_table_displaces_a_valid_user_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented structured-table exception, pinned so it cannot drift.

    `is_valid_managed_scalar` accepts any value at a `STRUCTURED` path, so a
    wrong-typed managed table displaces the user's and the typed reader falls
    back to its default. `README.md` documents this; nothing tested it.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text(
        '[models.providers.custom]\napi_key_env = "CUSTOM_KEY"\n',
        encoding="utf-8",
    )
    managed = tmp_path / "managed.toml"
    managed.write_text('[models]\nproviders = "junk"\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        assert model_config.ModelConfig.load().providers == {}
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_loaded_config_cannot_mutate_the_shared_managed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer must not be able to rewrite policy for the rest of the session.

    The managed snapshot is cached process-wide, so handing out a live sub-dict
    would let one caller's edit outlive its own read.
    """
    from deepagents_code import model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("[broken", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text(
        '[models.providers.corp]\napi_key_env = "CORP_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        config = model_config.ModelConfig.load()
        assert "corp" in config.providers
        config.providers["corp"]["api_key_env"] = "ATTACKER_KEY"
        snapshot = service.get_managed_snapshot()
        assert snapshot.data["models"]["providers"]["corp"]["api_key_env"] == "CORP_KEY"
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def test_out_of_range_managed_recursion_limit_falls_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded resolver keeps the lower layer; the launch gate stops instead."""
    from deepagents_code.config_manifest import resolve_recursion_limit
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[runtime]\nrecursion_limit = 3\n")
    try:
        assert (
            resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 400}})
            == 400
        )
    finally:
        service.invalidate_config_sources()


def test_managed_theme_outranks_the_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed `[ui].theme` cannot be overridden by an exported variable."""
    from deepagents_code._env_vars import THEME
    from deepagents_code.config_manifest import get_option, resolve_scalar
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[ui]\ntheme = "textual-dark"\n')
    monkeypatch.setenv(THEME, "nord")
    option = get_option("display.theme")
    assert option is not None
    try:
        value, source = resolve_scalar(option, toml_data={})
        assert value == "textual-dark"
        assert source.startswith("managed config")
    finally:
        service.invalidate_config_sources()


def test_failed_write_leaves_no_temporary_file_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write that fails mid-flight must not litter the config directory."""
    import tomli_w

    from deepagents_code.configuration import writer

    target = tmp_path / "config.toml"
    target.write_text('[ui]\ntheme = "dark"\n', encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(tomli_w, "dump", explode)
    result = writer.update_user_config(
        lambda data: bool(data.__setitem__("ui", {"theme": "light"})) or True,
        config_path=target,
    )
    assert result.ok is False
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("env_value", "user_toml", "expected_source"),
    [
        (None, '[shell]\nallow_list = ["ls"]\n', "config.toml"),
        ("git", '[shell]\nallow_list = ["ls"]\n', "env"),
    ],
    ids=["user-toml-honored", "env-outranks-user-toml"],
)
def test_shell_allow_list_reads_the_user_toml_below_the_environment(
    env_value: str | None,
    user_toml: str,
    expected_source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key gained `toml_keys`, so the TOML tier needs its own coverage.

    It was env-only before this feature, and it grants shell auto-approval, so
    the new tier is a user-writable permission surface.
    """
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option, resolve_scalar

    user = tmp_path / "config.toml"
    user.write_text(user_toml, encoding="utf-8")
    if env_value is None:
        monkeypatch.delenv(SHELL_ALLOW_LIST, raising=False)
    else:
        monkeypatch.setenv(SHELL_ALLOW_LIST, env_value)
    option = get_option("shell.allow_list")
    assert option is not None
    import tomllib

    with user.open("rb") as handle:
        toml_data = tomllib.load(handle)
    _, source = resolve_scalar(option, toml_data=toml_data, managed_toml_data={})
    assert source.startswith(expected_source)


def test_managed_shell_allow_list_outranks_a_shell_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported allow list cannot defeat the managed one."""
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option, resolve_scalar
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, '[shell]\nallow_list = ["ls"]\n')
    monkeypatch.setenv(SHELL_ALLOW_LIST, "all")
    option = get_option("shell.allow_list")
    assert option is not None
    try:
        value, source = resolve_scalar(option, toml_data={})
        assert value == ["ls"]
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


def test_empty_managed_shell_allow_list_is_a_lockdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`allow_list = []` must remove every grant, not fall through to one."""
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config_manifest import get_option, resolve_scalar
    from deepagents_code.configuration import service

    _managed_only(tmp_path, monkeypatch, "[shell]\nallow_list = []\n")
    monkeypatch.setenv(SHELL_ALLOW_LIST, "all")
    option = get_option("shell.allow_list")
    assert option is not None
    try:
        value, source = resolve_scalar(option, toml_data={})
        assert value is None
        assert source == "managed config"
    finally:
        service.invalidate_config_sources()


def test_saving_a_shadowed_ui_preference_says_policy_still_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save succeeds, and the user learns why nothing changed on screen.

    `README.md` advertises this notice for the theme, terminal-mapping,
    UI-toggle, and MCP-server screens; nothing tested any of them.
    """
    from deepagents_code import app, model_config
    from deepagents_code.configuration import service

    user = tmp_path / "config.toml"
    user.write_text("", encoding="utf-8")
    managed = tmp_path / "managed.toml"
    managed.write_text("[ui]\nshow_scrollbar = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    model_config.clear_caches()
    try:
        result = app._save_show_scrollbar_result(visible=False)
        assert result.ok is True
        assert result.message == (
            "Preference saved, but managed config remains effective."
        )
        # The preference is still written, so removing policy reveals it.
        assert "show_scrollbar = false" in user.read_text(encoding="utf-8")
    finally:
        service.invalidate_config_sources()
        model_config.clear_caches()


def _reload_previous() -> dict[str, object]:
    """Return a `previous` mapping shaped like the reloadable settings."""
    from deepagents_code.config import _RELOADABLE_FIELDS

    previous: dict[str, object] = dict.fromkeys(_RELOADABLE_FIELDS)
    previous["shell_allow_list"] = ["ls"]
    previous["extra_skills_dirs"] = []
    return previous


@pytest.mark.parametrize(
    "managed_toml",
    ["[broken", "[shell]\nallow_list = 5\n"],
    ids=["unparseable", "unenforceable"],
)
def test_blocked_reload_keeps_policy_and_says_so(
    managed_toml: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload that cannot enforce policy keeps it and reports the block.

    Regression: an unenforceable managed value had no reload-time equivalent of
    the launch-time reject, so `/reload` silently downgraded the shell allow
    list to the user's env value. The empty change list also told the user
    nothing had happened.
    """
    from deepagents_code._env_vars import SHELL_ALLOW_LIST
    from deepagents_code.config import Settings
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text(managed_toml, encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    previous = _reload_previous()
    try:
        refreshed, blocked = Settings._reload_values(
            start_path=tmp_path,
            env={SHELL_ALLOW_LIST: "all"},
            previous=previous,
        )
        assert refreshed == previous
        assert blocked is not None
        assert str(managed) in blocked
    finally:
        service.invalidate_config_sources()


def test_managed_startup_mode_revokes_a_user_yolo_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Managed policy outranks the `--yolo` flag it was written to forbid."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[startup]\nmode = "manual"\n', encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_policy(args)
    finally:
        service.invalidate_config_sources()

    assert args.yolo is False
    assert args.auto_approve is False


def test_managed_shell_allow_list_clears_the_cli_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed allow list must displace `--shell-allow-list all`."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_policy(args)
    finally:
        service.invalidate_config_sources()

    assert args.shell_allow_list is None


def test_managed_recursion_limit_is_range_checked_before_it_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid managed limit is applied; `resolve_scalar` alone never bounds it."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[runtime]\nrecursion_limit = 500\n", encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    args = _managed_policy_args()
    try:
        main._apply_managed_runtime_policy(args)
    finally:
        service.invalidate_config_sources()

    assert args.recursion_limit == 500


@pytest.mark.parametrize(
    ("registry_value", "expected_root"),
    [
        ("D:/SharedData", "D:/SharedData"),
        ("", "C:/ProgramData"),
        (None, "C:/ProgramData"),
    ],
)
def test_windows_program_data_comes_from_the_registry(
    monkeypatch: pytest.MonkeyPatch,
    registry_value: str | None,
    expected_root: str,
) -> None:
    """The Windows root is read from HKLM, never from a user-settable env var.

    A relocated ProgramData is a real enterprise configuration, and reading
    `%ProgramData%` would let any unprivileged user redirect the lookup.
    """
    import sys as _sys
    import types

    from deepagents_code.configuration import paths

    class _FakeKey:
        def __enter__(self) -> _FakeKey:  # noqa: PYI034 - local test double
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def query(_key: object, _name: str) -> tuple[object, int]:
        if registry_value is None:
            raise OSError
        return registry_value, 1

    fake = types.SimpleNamespace(
        HKEY_LOCAL_MACHINE=object(),
        OpenKey=lambda *_a, **_k: _FakeKey(),
        QueryValueEx=query,
    )
    monkeypatch.setitem(_sys.modules, "winreg", fake)
    monkeypatch.setattr(paths.sys, "platform", "win32")

    # Target the helper directly: the autouse isolation fixture replaces
    # `managed_config_path` itself.
    assert paths._windows_program_data(None) == expected_root


def test_disabled_server_write_recomputes_inside_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale pre-lock snapshot must not drop a concurrently disabled server.

    The writer recomputes the deny set from the table it parses inside the
    lock, so a disable that landed after the caller's read still survives.
    """
    from deepagents_code import mcp_disabled

    config_path = tmp_path / "config.toml"
    config_path.write_text('[mcp]\ndisabled_servers = ["a"]\n', encoding="utf-8")

    # Stand in for a concurrent writer: the caller's snapshot predates the
    # disable of "a" that is already on disk.
    monkeypatch.setattr(mcp_disabled, "_load_config", lambda _path: {})

    ok, _detail = mcp_disabled.set_server_disabled("b", True, config_path=config_path)
    assert ok

    import tomllib

    with config_path.open("rb") as handle:
        written = tomllib.load(handle)
    assert written["mcp"]["disabled_servers"] == ["a", "b"]


def test_startup_gate_exits_78_on_unusable_managed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gated command must stop rather than run with policy unenforced."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    managed = tmp_path / "managed.toml"
    managed.write_text("[broken", encoding="utf-8")
    monkeypatch.setattr(service, "managed_config_path", lambda: managed)
    service.invalidate_config_sources()
    try:
        with pytest.raises(SystemExit) as excinfo:
            main._require_managed_config_or_exit()
        assert excinfo.value.code == 78
    finally:
        service.invalidate_config_sources()


def test_startup_gate_accepts_a_missing_managed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing file applies no policy and must not block any command."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    monkeypatch.setattr(
        service, "managed_config_path", lambda: tmp_path / "absent.toml"
    )
    service.invalidate_config_sources()
    try:
        main._require_managed_config_or_exit()
    finally:
        service.invalidate_config_sources()
