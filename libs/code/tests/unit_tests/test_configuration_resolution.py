"""Named integration tests for managed-config resolution semantics.

The pre-merge golden snapshot (2,398 recorded option/stack cases) proved the
ranked resolver reproduced the legacy engine and was retired once the
migration shipped. These named tests carry the deliberate behaviors forward:
enforced keys exit 78, unhealthy providers fail closed, deny-list spellings
cannot diverge between read paths, and reload/preview never swaps the process
snapshot. New behavior belongs in new named tests here, not snapshots.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

if TYPE_CHECKING:
    from collections.abc import Mapping

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    OptionKind,
    _emit_ranked_diagnostics,
    _ranked_source,
    get_option,
)
from deepagents_code.configuration.resolver import resolver_from_snapshots
from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)
from unit_tests.conftest import resolve_option_for_test


def _resolve(
    option: ConfigOption[object],
    *,
    toml_data: dict[str, Any],
    managed_toml_data: dict[str, Any] | None = None,
) -> tuple[Any, str]:
    """Resolve `option` through production code as `(value, source)`.

    Thin alias for the shared `resolve_option_for_test`; see it for why these
    resolutions must not be rebuilt in the test suite.
    """
    return resolve_option_for_test(
        option, toml_data=toml_data, managed_toml_data=managed_toml_data
    )


_MANAGED_PATH = Path("/managed/managed_config.toml")


def _at_path(keys: tuple[str, ...] | None, value: object) -> dict[str, Any]:
    """Nest `value` at one manifest path."""
    if not keys:
        return {}
    root: dict[str, Any] = {}
    node = root
    for key in keys[:-1]:
        child: dict[str, Any] = {}
        node[key] = child
        node = child
    node[keys[-1]] = value
    return root


def _invalid_toml_value(option: ConfigOption[object]) -> object:
    """Return a value the option's TOML coercer rejects."""
    kind = option.kind
    if kind in {
        OptionKind.BOOL,
        OptionKind.BOOL_MODE_DEFAULT,
        OptionKind.BOOL_PRESENCE,
        OptionKind.INT,
        OptionKind.NON_NEGATIVE_INT,
        OptionKind.FLOAT,
    }:
        return "invalid"
    if kind in {OptionKind.STR, OptionKind.LOG_LEVEL_DELEGATE}:
        return 7
    if kind is OptionKind.NON_EMPTY_STR:
        return "   "
    if kind is OptionKind.ISO_DATETIME:
        return "not-a-datetime"
    if kind is OptionKind.DURATION_SECONDS:
        return "not-a-duration"
    if kind is OptionKind.MODEL_LIST_DELEGATE:
        return "not-a-list-of-model-specs"
    if kind in {OptionKind.SHELL_LIST_DELEGATE, OptionKind.SKILLS_DIRS_DELEGATE}:
        return 7
    if kind is OptionKind.PTC_DELEGATE:
        return True
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        return "beam"
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        return "YOLO"
    if kind is OptionKind.THEME_DELEGATE:
        return "unknown-test-theme"
    if kind is OptionKind.STRUCTURED:
        # The scalar resolver intentionally passes structured values through;
        # typed consumers decide whether this is usable.
        return 7
    msg = f"unhandled option kind: {kind}"
    raise AssertionError(msg)


def _snapshot(data: dict[str, Any], status: ProviderStatus) -> TomlSnapshot:
    """Build a typed snapshot from a literal mapping."""
    return TomlSnapshot(data, status)


def _managed_args() -> argparse.Namespace:
    """Return user-controlled launch arguments that policy must constrain."""
    return argparse.Namespace(
        model="user:model",
        auto_classifier_model="user:model",
        interpreter=True,
        recursion_limit=999,
        sandbox="none",
        interpreter_tools="all",
        shell_allow_list="all",
        auto_approve=True,
        yolo=True,
    )


def test_every_enforced_key_exits_78_and_a_rejected_benign_key_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the enforced-key boundary and tolerant non-enforced behavior."""
    from deepagents_code import main
    from deepagents_code.configuration import service
    from deepagents_code.configuration.service import ENFORCED_MANAGED_KEYS

    status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.OK)
    for key in ENFORCED_MANAGED_KEYS:
        option = get_option(key)
        assert option is not None
        data = _at_path(option.toml_keys, _invalid_toml_value(option))
        monkeypatch.setattr(
            service,
            "get_managed_snapshot",
            lambda data=data, **_kwargs: _snapshot(data, status),
        )
        with pytest.raises(SystemExit) as excinfo:
            main._apply_managed_runtime_exceptions(_managed_args())
        assert excinfo.value.code == 78

    benign = get_option("display.cursor_style")
    assert benign is not None
    managed = _at_path(benign.toml_keys, "beam")
    user = _at_path(benign.toml_keys, "underline")
    assert _resolve(benign, toml_data=user, managed_toml_data=managed) == (
        "underline",
        "config.toml",
    )
    assert service.managed_rejections(managed) == ("display.cursor_style",)


def test_indeterminate_provider_path_exits_78(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guessed missing path is unhealthy, never equivalent to no policy."""
    from deepagents_code import main
    from deepagents_code.configuration import service

    status = ProviderStatus(
        "managed config",
        _MANAGED_PATH,
        ProviderHealth.INDETERMINATE,
        "synthetic registry fallback",
    )
    monkeypatch.setattr(
        service, "get_managed_snapshot", lambda **_kwargs: _snapshot({}, status)
    )
    with pytest.raises(SystemExit) as excinfo:
        main._require_managed_config_or_exit()
    assert excinfo.value.code == 78


def test_diagnostic_surfaces_pair_status_and_violations_from_one_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doctor and config must not pair a refreshed status with cached policy."""
    from deepagents_code.client.commands.config import (
        _MANAGED_PATH_LABEL,
        _config_path_status,
        _managed_health_warning,
    )
    from deepagents_code.configuration import service
    from deepagents_code.doctor import _managed_config_diagnostic

    status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.OK)
    cached = _snapshot({"startup": {"mode": "manual"}}, status)
    rejected = _snapshot({"startup": {"mode": "YOLO"}}, status)
    calls: list[bool] = []

    def get_snapshot(
        *, refresh: bool = False, path: Path | None = None
    ) -> TomlSnapshot:
        assert path is None
        calls.append(refresh)
        return rejected if refresh else cached

    monkeypatch.setattr(service, "get_managed_snapshot", get_snapshot)

    warning = _managed_health_warning()
    assert warning is not None
    assert "startup.mode" in warning
    assert calls == [True]
    calls.clear()

    assert _config_path_status(_MANAGED_PATH_LABEL, exists=True) == "rejected"
    assert calls == [True]
    calls.clear()

    item = _managed_config_diagnostic()
    assert item.ok is False
    assert "startup.mode" in item.value
    assert calls == [True]


def test_comma_string_denies_agree_across_scalar_merge_and_runtime_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented string spelling cannot diverge between read paths."""
    from deepagents_code import mcp_disabled, model_config
    from deepagents_code.configuration import service
    from unit_tests.conftest import redirect_managed_config

    cases = (
        (
            "mcp.disabled_servers",
            "disabled_servers",
            lambda: mcp_disabled.get_disabled_servers(),
        ),
        (
            "mcp.disabled_project_servers",
            "disabled_project_servers",
            lambda: model_config.load_mcp_server_trust_lists().disabled,
        ),
    )
    for option_key, toml_key, runtime in cases:
        user = tmp_path / f"{toml_key}-user.toml"
        managed = tmp_path / f"{toml_key}-managed.toml"
        user.write_text(f'[mcp]\n{toml_key} = ["user", "shared"]\n', encoding="utf-8")
        managed.write_text(f'[mcp]\n{toml_key} = "managed, shared"\n', encoding="utf-8")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
        monkeypatch.setattr(mcp_disabled, "_DEFAULT_CONFIG_PATH", user)
        redirect_managed_config(monkeypatch, managed)
        service.invalidate_config_sources()

        option = get_option(option_key)
        assert option is not None
        user_data = {"mcp": {toml_key: ["user", "shared"]}}
        managed_data = {"mcp": {toml_key: "managed, shared"}}
        scalar, _ = _resolve(
            option, toml_data=user_data, managed_toml_data=managed_data
        )
        merged, _ = service.ConfigSources(
            _snapshot(
                managed_data,
                ProviderStatus("managed config", managed, ProviderHealth.OK),
            ),
            _snapshot(
                user_data,
                ProviderStatus("config.toml", user, ProviderHealth.OK),
            ),
        ).merged()
        assert set(cast("list[str]", scalar)) == {"managed", "shared", "user"}
        assert set(merged["mcp"][toml_key]) == {"managed", "shared", "user"}
        assert set(runtime()) == {"managed", "shared", "user"}
        service.invalidate_config_sources()


def test_quoted_dotted_keys_keep_sibling_provenance() -> None:
    """Tuple paths prevent a quoted dotted key from aliasing a nested path."""
    from deepagents_code.configuration.service import merge_managed_over_user

    _, provenance = merge_managed_over_user(
        {"a": "user", "sibling": 1}, {"a.b": "managed"}
    )
    assert provenance == {
        "a": "config.toml",
        "sibling": "config.toml",
        "a.b": "managed config",
    }


def test_user_write_to_managed_key_succeeds_but_stays_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer stores shadowed preferences and the resolver reports policy."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service
    from deepagents_code.configuration.writer import update_user_config
    from unit_tests.conftest import redirect_managed_config

    user = tmp_path / "config.toml"
    managed_path = tmp_path / "managed.toml"
    managed_path.write_text("[ui]\nshow_scrollbar = true\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed_path)

    def mutate(data: dict[str, Any]) -> bool:
        data["ui"] = {"show_scrollbar": False}
        return True

    result = update_user_config(mutate, config_path=user)
    assert result.ok
    assert result.changed
    option = get_option("display.show_scrollbar")
    assert option is not None
    value, source = _resolve(
        option,
        toml_data={"ui": {"show_scrollbar": False}},
        managed_toml_data={"ui": {"show_scrollbar": True}},
    )
    assert (value, source) == (True, "managed config")

    # The same writer structurally refuses the managed path.
    refused = update_user_config(mutate, config_path=managed_path)
    assert refused.ok is False
    assert refused.error == "managed config is read-only"
    service.invalidate_config_sources()


def test_wrong_typed_update_policy_matches_an_unreadable_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both managed-policy failures force network/update behavior off."""
    from deepagents_code.configuration import service
    from deepagents_code.update_check import _managed_update_value

    ok_status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.OK)
    corrupt_status = ProviderStatus(
        "managed config", _MANAGED_PATH, ProviderHealth.CORRUPT, "broken"
    )
    snapshots = (
        _snapshot({"update": {"auto_update": "false"}}, ok_status),
        _snapshot({}, corrupt_status),
    )
    results: list[tuple[bool, bool]] = []
    for candidate in snapshots:
        monkeypatch.setattr(
            service,
            "get_managed_snapshot",
            lambda candidate=candidate, **_kwargs: candidate,
        )
        results.append(_managed_update_value("auto_update"))
    assert results == [(True, False), (True, False)]


def test_rejected_non_enforced_reload_purges_stale_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly rejected value cannot retain the prior managed attribution."""
    from deepagents_code import model_config
    from deepagents_code.configuration import service
    from unit_tests.conftest import redirect_managed_config

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("[threads]\nrelative_time = true\n", encoding="utf-8")
    managed.write_text("[threads]\nrelative_time = false\n", encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    service.require_healthy_managed_config(refresh=True)
    _, before = service.get_config_sources().merged()
    assert before["threads.relative_time"] == "managed config"

    managed.write_text('[threads]\nrelative_time = "invalid"\n', encoding="utf-8")
    service.require_healthy_managed_config(refresh=True)
    data, after = service.get_config_sources().merged()
    assert data["threads"]["relative_time"] is True
    assert after["threads.relative_time"] == "config.toml"
    service.invalidate_config_sources()


def test_preview_reload_does_not_swap_the_process_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run observes the cached policy without accepting a disk edit."""
    from deepagents_code import model_config
    from deepagents_code.config import Credentials
    from deepagents_code.configuration import service
    from unit_tests.conftest import redirect_managed_config

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("", encoding="utf-8")
    managed.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    runtime = Credentials.from_environment(start_path=tmp_path)
    assert service.get_managed_snapshot().data["shell"]["allow_list"] == ["ls"]

    managed.write_text('[shell]\nallow_list = ["git"]\n', encoding="utf-8")
    runtime.preview_reload_from_environment(start_path=tmp_path)
    assert service.get_managed_snapshot().data["shell"]["allow_list"] == ["ls"]
    service.invalidate_config_sources()


# One representative option per coercion kind whose resolution runs through
# the ranked scalar engine with all of managed, env, and user tiers. Kinds
# that intentionally read only a subset of tiers (log level is env-only,
# themes read the user file) are covered by their own targeted tests instead.
_MATRIX_ENV_VARS = {
    "display.show_scrollbar": "DEEPAGENTS_CODE_SHOW_SCROLLBAR",
    "runtime.recursion_limit": "DEEPAGENTS_CODE_RECURSION_LIMIT",
    "shell.allow_list": "DEEPAGENTS_CODE_SHELL_ALLOW_LIST",
}

_MATRIX_MANAGED_VALUES = {
    "display.show_scrollbar": True,
    "runtime.recursion_limit": 3000,
    "shell.allow_list": ["managed-command"],
}
_MATRIX_USER_VALUES = {
    "display.show_scrollbar": True,
    "runtime.recursion_limit": 2000,
    "shell.allow_list": ["user-command"],
}
_MATRIX_ENV_VALUES = {
    "display.show_scrollbar": "true",
    "runtime.recursion_limit": "4000",
    "shell.allow_list": "env-command",
}

# The env tier coerces to the option's typed domain, so the expected resolved
# value is not always the raw string above.
_MATRIX_EXPECTED_ENV = {
    "display.show_scrollbar": True,
    "runtime.recursion_limit": 4000,
    "shell.allow_list": ["env-command"],
}

# Env values that the option's coercer rejects. Every tier falls through, so
# the expectation is always the user value and the warning text is not
# asserted here; per-kind rejection messages have their own targeted tests.
_MATRIX_INVALID_ENV_VALUES = {
    "display.show_scrollbar": "maybe",
    "runtime.recursion_limit": "not-an-int",
    "shell.allow_list": "all, env-command",
}


def _resolve_in_stack(
    option_key: str,
    *,
    managed: str,
    env: str,
    user: str,
) -> tuple[Any, str]:
    """Resolve one option in a synthetic managed/env/user stack.

    `managed` is `present` or `absent`; `env` is `set`, `unset`, or `invalid`;
    `user` is `set` or `unset`. A set tier supplies the per-kind valid (or
    invalid) value from the matrices above.
    """
    option = get_option(option_key)
    assert option is not None
    managed_data = (
        _at_path(option.toml_keys, _MATRIX_MANAGED_VALUES[option_key])
        if managed == "present"
        else {}
    )
    user_data = (
        _at_path(option.toml_keys, _MATRIX_USER_VALUES[option_key])
        if user == "set"
        else {}
    )
    env_name = _MATRIX_ENV_VARS[option_key]
    environ: dict[str, str] = {}
    if env == "set":
        environ[env_name] = _MATRIX_ENV_VALUES[option_key]
    elif env == "invalid":
        environ[env_name] = _MATRIX_INVALID_ENV_VALUES[option_key]
    with patch.dict(os.environ, environ, clear=True):
        return _resolve(option, toml_data=user_data, managed_toml_data=managed_data)


@pytest.mark.parametrize("option_key", sorted(_MATRIX_ENV_VARS))
@pytest.mark.parametrize("user", ["set", "unset"])
@pytest.mark.parametrize("env", ["set", "unset", "invalid"])
@pytest.mark.parametrize("managed", ["present", "absent"])
def test_option_kind_matrix_resolves_ranked_precedence(
    option_key: str, managed: str, env: str, user: str
) -> None:
    """Every option kind honors managed > env > user in a full tier stack.

    This is the named replacement for the retired golden matrix: one
    representative option per coercion kind, asserting the correct precedence
    rather than recorded output, so a future option that mis-wires its
    coercion or rank fails here instead of passing silently.
    """
    value, source = _resolve_in_stack(option_key, managed=managed, env=env, user=user)

    expected_user = _MATRIX_USER_VALUES[option_key]
    if managed == "present":
        assert (value, source) == (_MATRIX_MANAGED_VALUES[option_key], "managed config")
    elif env == "set":
        assert (value, source) == (
            _MATRIX_EXPECTED_ENV[option_key],
            f"env ({_MATRIX_ENV_VARS[option_key]})",
        )
    elif user == "set":
        # An invalid env tier falls through without reversing precedence.
        assert (value, source) == (expected_user, "config.toml")
    else:
        assert source == "default"
