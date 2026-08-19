"""Golden behavior net for the managed-config resolver migration.

The ranked resolver is intentionally being introduced behind the existing
engine.  This module records the existing engine's complete observable result
for every manifest option across synthetic provider stacks.  The observable
result includes warning text and the managed-health diagnostic, not just the
resolved value and source, because coercion is moving across a module boundary
and silently changing a rejection message would make the migration look safer
than it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import patch

import pytest

from deepagents_code.config_manifest import (
    ConfigOption,
    OptionKind,
    get_config_options,
    get_option,
    resolve_scalar,
)
from deepagents_code.configuration.types import (
    ProviderHealth,
    ProviderStatus,
    TomlSnapshot,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_GOLDEN_PATH = Path(__file__).with_name("snapshots") / "configuration_resolver.json"
_MANAGED_PATH = Path("/managed/managed_config.toml")
_USER_PATH = Path("/user/config.toml")
_PREFIX = "DEEPAGENTS_CODE_"


@dataclass(frozen=True, slots=True)
class _Scenario:
    """One synthetic provider stack resolved by the legacy engine."""

    name: str
    managed: Literal["present", "absent", "corrupt", "indeterminate"]
    env: Literal["set", "unset", "invalid", "empty", "whitespace", "prefixed"]
    user: Literal["set", "unset", "invalid"]


_SCENARIOS = (
    *(
        _Scenario(
            f"managed_{managed}__env_{env}__user_{user}",
            managed,
            env,
            user,
        )
        for managed in ("present", "absent", "corrupt", "indeterminate")
        for env in ("set", "unset")
        for user in ("set", "unset")
    ),
    _Scenario("managed_invalid__env_set__user_set", "present", "set", "set"),
    _Scenario("managed_absent__env_invalid__user_set", "absent", "invalid", "set"),
    _Scenario("managed_absent__env_unset__user_invalid", "absent", "unset", "invalid"),
    _Scenario("managed_absent__env_empty__user_set", "absent", "empty", "set"),
    _Scenario(
        "managed_absent__env_whitespace__user_set",
        "absent",
        "whitespace",
        "set",
    ),
    _Scenario("managed_absent__env_prefixed__user_set", "absent", "prefixed", "set"),
)


_TABLE_STRUCTURED_KEYS = frozenset(
    {
        "display.themes",
        "display.terminal_themes",
        "models.providers",
        "agents.async_subagents",
        "sandboxes.providers",
        "threads.columns",
    }
)


def _valid_value(option: ConfigOption, tier: str) -> object:
    """Return a deterministic valid input in the provider's TOML domain."""
    kind = option.kind
    if kind in {
        OptionKind.BOOL,
        OptionKind.BOOL_MODE_DEFAULT,
        OptionKind.BOOL_PRESENCE,
    }:
        return tier != "user"
    if kind in {OptionKind.INT, OptionKind.NON_NEGATIVE_INT}:
        # The bounded recursion-limit wrapper accepts this value too.
        return {"managed": 3000, "user": 2000, "env": 4000}[tier]
    if kind is OptionKind.FLOAT:
        return {"managed": 11.5, "user": 22.5, "env": 33.5}[tier]
    if kind in {OptionKind.STR, OptionKind.NON_EMPTY_STR}:
        return f"{tier}-value"
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        return {"managed": "ERROR", "user": "INFO", "env": "WARNING"}[tier]
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        return [f"{tier}-command"]
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        return [f"/golden/{tier}-skills"]
    if kind is OptionKind.PTC_DELEGATE:
        return "safe" if tier == "managed" else [f"{tier}_tool"]
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        return "block" if tier == "managed" else "underline"
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        return "manual" if tier == "managed" else "auto"
    if kind is OptionKind.THEME_DELEGATE:
        return {
            "managed": "textual-dark",
            "user": "langchain-light",
            "env": "nord",
        }[tier]
    if kind is OptionKind.STRUCTURED:
        if option.key in _TABLE_STRUCTURED_KEYS:
            return {tier: {"enabled": True}}
        if option.key == "mcp.enabled_project_server_approvals":
            return [{"project": tier, "server": "golden", "fingerprint": tier}]
        return [f"{tier}-entry"]
    msg = f"unhandled option kind: {kind}"
    raise AssertionError(msg)


def _valid_env_value(option: ConfigOption) -> str:
    """Return a valid input in the environment provider's string domain."""
    kind = option.kind
    if kind in {OptionKind.BOOL, OptionKind.BOOL_MODE_DEFAULT}:
        return "true"
    if kind is OptionKind.BOOL_PRESENCE:
        return "present"
    if kind in {OptionKind.INT, OptionKind.NON_NEGATIVE_INT}:
        return "4000"
    if kind is OptionKind.FLOAT:
        return "33.5"
    if kind in {OptionKind.STR, OptionKind.NON_EMPTY_STR}:
        return "env-value"
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        return "WARNING"
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        return "env-command"
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        return "/golden/env-skills"
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        return "underline"
    if kind is OptionKind.THEME_DELEGATE:
        return "nord"
    # No shipping PTC/structured/startup-mode option declares an env source.
    return "env-value"


def _invalid_toml_value(option: ConfigOption) -> object:
    """Return a value the option's legacy TOML coercer rejects."""
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
    if kind in {OptionKind.SHELL_LIST_DELEGATE, OptionKind.SKILLS_DIRS_DELEGATE}:
        return 7
    if kind is OptionKind.PTC_DELEGATE:
        return True
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        return "beam"
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        return "YOLO"
    if kind is OptionKind.THEME_DELEGATE:
        return "unknown-golden-theme"
    if kind is OptionKind.STRUCTURED:
        # The old scalar resolver intentionally passes structured values through;
        # typed consumers decide whether this is usable.
        return 7
    msg = f"unhandled option kind: {kind}"
    raise AssertionError(msg)


def _invalid_env_value(option: ConfigOption) -> str:
    """Return a string the option's legacy environment coercer rejects."""
    kind = option.kind
    if kind in {OptionKind.BOOL, OptionKind.BOOL_MODE_DEFAULT}:
        return "maybe"
    if kind is OptionKind.NON_NEGATIVE_INT:
        return "-1"
    if kind is OptionKind.INT:
        return "not-an-int"
    if kind is OptionKind.FLOAT:
        return "not-a-number"
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        return "TRACE"
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        return "all, env-command"
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        return "~golden-user-that-does-not-exist/config"
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        return "beam"
    if kind is OptionKind.THEME_DELEGATE:
        return "unknown-golden-theme"
    # Raw strings are intrinsically valid for STR/BOOL_PRESENCE, and no
    # shipping PTC/structured/startup-mode option is env-backed. Whitespace is
    # still an invalid/unset provider input and exercises its diagnostic.
    return "   "


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


def _provider_snapshot(
    option: ConfigOption, scenario: _Scenario
) -> tuple[TomlSnapshot, Mapping[str, Any]]:
    """Build the managed provider snapshot and data passed to `resolve_scalar`."""
    if scenario.name.startswith("managed_invalid"):
        data = _at_path(option.toml_keys, _invalid_toml_value(option))
        status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.OK)
        return TomlSnapshot(data, status), data
    if scenario.managed == "present":
        data = _at_path(option.toml_keys, _valid_value(option, "managed"))
        status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.OK)
        return TomlSnapshot(data, status), data
    if scenario.managed == "absent":
        status = ProviderStatus("managed config", _MANAGED_PATH, ProviderHealth.MISSING)
    elif scenario.managed == "corrupt":
        status = ProviderStatus(
            "managed config",
            _MANAGED_PATH,
            ProviderHealth.CORRUPT,
            "synthetic parse error",
        )
    else:
        status = ProviderStatus(
            "managed config",
            _MANAGED_PATH,
            ProviderHealth.INDETERMINATE,
            "synthetic registry fallback",
        )
    return TomlSnapshot({}, status), {}


def _user_data(option: ConfigOption, scenario: _Scenario) -> dict[str, Any]:
    """Build the user-file input for one scenario."""
    if scenario.user == "unset":
        return {}
    value = (
        _valid_value(option, "user")
        if scenario.user == "set"
        else _invalid_toml_value(option)
    )
    return _at_path(option.toml_keys, value)


def _option_env_names(option: ConfigOption) -> tuple[str, ...]:
    """Return every environment spelling the option resolver may inspect."""
    if option.env_var is None:
        return option.fallback_env_vars
    canonical = option.env_var
    prefixed = canonical if canonical.startswith(_PREFIX) else f"{_PREFIX}{canonical}"
    return tuple(dict.fromkeys((prefixed, canonical, *option.fallback_env_vars)))


def _scenario_env(option: ConfigOption, scenario: _Scenario) -> dict[str, str]:
    """Build the environment input for one scenario."""
    if option.env_var is None:
        return {}
    canonical = option.env_var
    prefixed = canonical if canonical.startswith(_PREFIX) else f"{_PREFIX}{canonical}"
    if scenario.env == "unset":
        return {}
    if scenario.env == "set":
        return {canonical: _valid_env_value(option)}
    if scenario.env == "prefixed":
        return {prefixed: _valid_env_value(option)}
    if scenario.env == "invalid":
        return {canonical: _invalid_env_value(option)}
    if scenario.env == "empty":
        return {canonical: ""}
    return {canonical: " \t "}


class _Capture(logging.Handler):
    """Collect warning/error messages without changing package logging."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Record the level and rendered message."""
        self.messages.append(
            {"level": record.levelname, "message": record.getMessage()}
        )


@contextmanager
def _capture_diagnostics() -> Iterator[_Capture]:
    """Capture package warnings emitted during one resolution."""
    package_logger = logging.getLogger("deepagents_code")
    previous_level = package_logger.level
    capture = _Capture()
    package_logger.addHandler(capture)
    if previous_level > logging.WARNING:
        package_logger.setLevel(logging.WARNING)
    try:
        yield capture
    finally:
        package_logger.removeHandler(capture)
        package_logger.setLevel(previous_level)


def _normalize(value: object) -> object:
    """Convert resolver output into stable JSON-shaped data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_normalize(child) for child in value), key=repr)
    return value


def _resolve_record(option: ConfigOption, scenario: _Scenario) -> dict[str, object]:
    """Resolve one option/stack through the current engine."""
    from deepagents_code import config_manifest
    from deepagents_code.client.commands.config import _managed_health_warning
    from deepagents_code.configuration import service

    snapshot, managed_data = _provider_snapshot(option, scenario)
    user_data = _user_data(option, scenario)
    env = _scenario_env(option, scenario)
    theme_patch = nullcontext()
    if option.kind is OptionKind.THEME_DELEGATE:
        from deepagents_code import theme

        # `get_registry()` normally reads config on first use. Keep that
        # separate managed-aware callsite out of this one-option resolver
        # snapshot and use the same deterministic built-in registry in both a
        # standalone regeneration process and pytest's pre-warmed process.
        registry = MappingProxyType(theme._builtin_themes())
        theme_patch = patch.object(theme, "get_registry", return_value=registry)
    config_manifest._warned_non_table_paths.clear()
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(service, "get_managed_snapshot", return_value=snapshot),
        theme_patch,
        _capture_diagnostics() as captured,
    ):
        value, source = resolve_scalar(
            option,
            toml_data=user_data,
            managed_toml_data=managed_data,
        )
        diagnostic = _managed_health_warning()
    config_manifest._warned_non_table_paths.clear()
    return {
        "value": _normalize(value),
        "source": source,
        "warnings": captured.messages,
        "diagnostic": diagnostic,
    }


def build_legacy_golden() -> dict[str, dict[str, dict[str, object]]]:
    """Return every legacy-engine snapshot record.

    Kept public within the test module so the checked-in JSON can be regenerated
    deliberately while the old engine is still authoritative.
    """
    return {
        option.key: {
            scenario.name: _resolve_record(option, scenario) for scenario in _SCENARIOS
        }
        for option in get_config_options()
    }


def encode_legacy_golden(
    golden: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    """Deduplicate records while keeping every option/scenario mapping explicit."""
    records: dict[str, Mapping[str, object]] = {}
    options: dict[str, dict[str, str]] = {}
    for key, scenarios in golden.items():
        encoded: dict[str, str] = {}
        for scenario, record in scenarios.items():
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
            previous = records.setdefault(digest, record)
            assert previous == record, "golden-record hash collision"
            encoded[scenario] = digest
        options[key] = encoded
    return {"format": 1, "records": records, "options": options}


def test_every_manifest_option_matches_the_legacy_golden() -> None:
    """Pin value, source, warnings, and health diagnostics for every option."""
    snapshot = cast(
        "dict[str, object]",
        json.loads(_GOLDEN_PATH.read_text(encoding="utf-8")),
    )
    assert snapshot["format"] == 1
    records = cast("dict[str, dict[str, object]]", snapshot["records"])
    expected = cast("dict[str, dict[str, str]]", snapshot["options"])
    actual = build_legacy_golden()
    assert actual.keys() == expected.keys()
    for key, scenarios in actual.items():
        assert scenarios.keys() == expected[key].keys()
        for scenario, record in scenarios.items():
            assert record == records[expected[key][scenario]], f"{key}: {scenario}"


def _snapshot(data: Mapping[str, Any], status: ProviderStatus) -> TomlSnapshot:
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
            main._apply_managed_runtime_policy(_managed_args())
        assert excinfo.value.code == 78

    benign = get_option("display.cursor_style")
    assert benign is not None
    managed = _at_path(benign.toml_keys, "beam")
    user = _at_path(benign.toml_keys, "underline")
    assert resolve_scalar(benign, toml_data=user, managed_toml_data=managed) == (
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
        scalar, _ = resolve_scalar(
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
    value, source = resolve_scalar(
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
    from deepagents_code.config import Settings
    from deepagents_code.configuration import service
    from unit_tests.conftest import redirect_managed_config

    user = tmp_path / "config.toml"
    managed = tmp_path / "managed.toml"
    user.write_text("", encoding="utf-8")
    managed.write_text('[shell]\nallow_list = ["ls"]\n', encoding="utf-8")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", user)
    redirect_managed_config(monkeypatch, managed)
    service.invalidate_config_sources()
    runtime = Settings.from_environment(start_path=tmp_path)
    assert service.get_managed_snapshot().data["shell"]["allow_list"] == ["ls"]

    managed.write_text('[shell]\nallow_list = ["git"]\n', encoding="utf-8")
    runtime.preview_reload_from_environment(start_path=tmp_path)
    assert service.get_managed_snapshot().data["shell"]["allow_list"] == ["ls"]
    service.invalidate_config_sources()
