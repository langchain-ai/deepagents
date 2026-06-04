"""Drift, resolution, and behavior tests for the configuration manifest.

These guard the contract that the manifest is the single source of truth for
the scalar config surface, that its resolver matches what the runtime reads,
and that secret-flagged options are never rendered by value.
"""

from __future__ import annotations

import argparse

from deepagents_code import _env_vars
from deepagents_code.config_commands import (
    _display_value,
    _resolve,
    _run_get,
    run_config_command,
)
from deepagents_code.config_manifest import (
    NON_OPTION_ENV_VARS,
    ConfigOption,
    OptionKind,
    get_config_options,
    get_option,
    option_keys,
    resolve_interpreter_kwargs,
    resolve_scalar,
)
from deepagents_code.model_config import PROVIDER_API_KEY_ENV


def _declared_deepagents_env_vars() -> set[str]:
    """Every `DEEPAGENTS_CODE_*` constant declared in `_env_vars`."""
    return {
        value
        for name, value in vars(_env_vars).items()
        if not name.startswith("_")
        and isinstance(value, str)
        and value.startswith("DEEPAGENTS_CODE_")
    }


# --- Drift / coverage -------------------------------------------------------


def test_manifest_covers_every_deepagents_env_var() -> None:
    """Every `DEEPAGENTS_CODE_*` env var must have a manifest entry."""
    manifest_env_vars = {opt.env_var for opt in get_config_options() if opt.env_var}
    declared = _declared_deepagents_env_vars() - NON_OPTION_ENV_VARS
    missing = declared - manifest_env_vars
    assert not missing, (
        f"`DEEPAGENTS_CODE_*` env vars without a manifest entry: {sorted(missing)}. "
        "Add a ConfigOption in config_manifest.py or list it in NON_OPTION_ENV_VARS."
    )


def test_manifest_covers_every_provider_credential() -> None:
    """Every provider in `PROVIDER_API_KEY_ENV` must have a credential option."""
    manifest_env_vars = {opt.env_var for opt in get_config_options() if opt.env_var}
    missing = set(PROVIDER_API_KEY_ENV.values()) - manifest_env_vars
    assert not missing, (
        f"Provider credential env vars without a manifest entry: {sorted(missing)}."
    )


def test_option_keys_unique() -> None:
    """Manifest keys must be unique so `config get` lookups are unambiguous."""
    keys = option_keys()
    assert len(keys) == len(set(keys))


# --- Secrets ----------------------------------------------------------------


def test_api_key_credentials_are_secret() -> None:
    """Credential options backed by key/token env vars must be secret-flagged."""
    for opt in get_config_options():
        if opt.group != "Credentials" or not opt.env_var:
            continue
        looks_secret = any(
            marker in opt.env_var for marker in ("KEY", "TOKEN", "APIKEY")
        )
        assert opt.secret is looks_secret, (
            f"{opt.key} secret={opt.secret} but env_var {opt.env_var!r} "
            f"implies secret={looks_secret}"
        )


def test_google_cloud_project_is_not_secret() -> None:
    """The Vertex project identifier is not secret material and shows its value."""
    opt = get_option("credentials.google_vertexai")
    assert opt is not None
    assert opt.env_var == "GOOGLE_CLOUD_PROJECT"
    assert opt.secret is False


def test_display_value_redacts_secrets() -> None:
    """A secret option never renders its raw value, only set/not set."""
    secret = ConfigOption(
        key="x",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        secret=True,
    )
    assert _display_value(secret, is_set=True, value="sk-supersecret") == "set"
    assert _display_value(secret, is_set=False, value=None) == "not set"


def test_run_get_json_omits_secret_value(monkeypatch, capsys) -> None:
    """JSON output for a secret option reports presence but never the value."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["set"] is True
    assert payload["data"]["value"] is None


# --- Single-source defaults -------------------------------------------------


def test_interpreter_defaults_match_settings() -> None:
    """Manifest interpreter defaults are the same objects `Settings` uses.

    This is what makes the manifest the single source of truth: the dataclass
    default and the manifest default cannot diverge because they are one value.
    """
    from deepagents_code.config import Settings

    settings = Settings.from_environment()
    for opt in get_config_options():
        if opt.group != "Interpreter" or opt.settings_field is None:
            continue
        assert getattr(settings, opt.settings_field) == opt.default


def test_every_settings_field_names_a_real_settings_attribute() -> None:
    """Catch a typo'd `settings_field` on any option, not just interpreter ones.

    `settings_field` is a free-form string with no compile-time link to the
    `Settings` dataclass, so a misspelling would only surface at runtime
    `getattr`. This locks the mapping across the whole catalog.
    """
    from dataclasses import fields

    from deepagents_code.config import Settings

    valid = {f.name for f in fields(Settings)}
    bad = {
        opt.key: opt.settings_field
        for opt in get_config_options()
        if opt.settings_field is not None and opt.settings_field not in valid
    }
    assert not bad, f"options reference unknown Settings fields: {bad}"


# --- Resolution -------------------------------------------------------------


def test_resolve_prefers_prefixed_env(monkeypatch) -> None:
    """A `DEEPAGENTS_CODE_`-prefixed env var wins over the canonical name."""
    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
    value, source = resolve_scalar(opt, toml_data={})
    assert source == "env (DEEPAGENTS_CODE_OPENAI_API_KEY)"
    assert value == "prefixed"


def test_resolve_empty_env_is_unset_matching_resolve_env_var(monkeypatch) -> None:
    """An empty (prefixed) env var is unset for `config show`, as the app sees it.

    The runtime `resolve_env_var` returns `None` for an empty prefixed var (and
    a prefixed empty suppresses the canonical). `resolve_scalar` must agree, or
    `config show` would report a credential as "set" that the app treats as
    unset — the exact drift this feature exists to prevent.
    """
    from deepagents_code.model_config import resolve_env_var

    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "")

    value, source = resolve_scalar(opt, toml_data={})
    assert resolve_env_var("OPENAI_API_KEY") is None
    assert source == "default"
    assert value is None


def test_run_show_json_redacts_every_secret(monkeypatch, capsys) -> None:
    """The `config show` aggregate (separate path from `get`) never leaks a secret."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    args = argparse.Namespace(config_command="show", output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    assert any(r["key"] == "credentials.anthropic" and r["set"] for r in rows)
    assert all(r["value"] is None for r in rows if r["secret"])


def test_resolve_int_falls_back_to_toml_then_default() -> None:
    """config.toml is consulted when env is unset; default is the last resort."""
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"interpreter": {"memory_limit_mb": 128}}) == (
        128,
        "config.toml",
    )
    assert resolve_scalar(opt, toml_data={}) == (64, "default")


def test_resolve_malformed_toml_int_falls_back_with_warning(caplog) -> None:
    """A bad TOML scalar is logged and falls back to the default, never raising."""
    import logging

    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(
            opt, toml_data={"interpreter": {"memory_limit_mb": "oops"}}
        )
    assert (value, source) == (64, "default")
    assert any("memory_limit_mb" in r.getMessage() for r in caplog.records)


def test_resolve_bool_env_uses_truthy_semantics(monkeypatch) -> None:
    """BOOL options honor is_env_truthy semantics ('0' is falsy, not 'set')."""
    opt = get_option("display.hide_cwd")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "1")
    assert resolve_scalar(opt, toml_data={})[0] is True
    monkeypatch.setenv(opt.env_var, "0")
    assert resolve_scalar(opt, toml_data={})[0] is False


def test_resolve_ptc_delegates_to_parser() -> None:
    """The PTC kind routes through the dedicated allowlist parser."""
    opt = get_option("interpreter.ptc")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"interpreter": {"ptc": "safe"}}) == (
        "safe",
        "config.toml",
    )
    # Invalid PTC value is rejected by the parser and falls back to default.
    value, source = resolve_scalar(opt, toml_data={"interpreter": {"ptc": "bogus"}})
    assert (value, source) == (opt.default, "default")


def test_resolve_interpreter_kwargs_maps_settings_fields() -> None:
    """The interpreter resolver returns Settings-constructor kwargs."""
    kwargs = resolve_interpreter_kwargs(
        toml_data={"interpreter": {"memory_limit_mb": 256, "enable_interpreter": True}}
    )
    assert kwargs["interpreter_memory_limit_mb"] == 256
    assert kwargs["enable_interpreter"] is True
    # Unspecified fields resolve to their manifest defaults.
    assert kwargs["interpreter_timeout_seconds"] == 5.0


# --- Misc -------------------------------------------------------------------


def test_get_option_unknown_returns_none() -> None:
    assert get_option("does.not.exist") is None


def test_run_get_unknown_key_returns_error_code(capsys) -> None:
    args = argparse.Namespace(config_command="get", key="nope", output_format="text")
    assert run_config_command(args) == 1
    assert "Unknown config option" in capsys.readouterr().err


def test_config_registered_in_help_specs() -> None:
    """The `config` group must be wired for the startup fast-path help dispatch."""
    from deepagents_code import ui
    from deepagents_code.main import _HELP_SPECS

    assert _HELP_SPECS.get("config") == ("config_command", "show_config_help")
    assert callable(ui.show_config_help)
