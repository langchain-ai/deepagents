"""Drift, resolution, and behavior tests for the configuration manifest.

These guard the contract that the manifest is the single source of truth for
the scalar config surface, that its resolver matches what the runtime reads,
and that secret-flagged options are never rendered by value.
"""

from __future__ import annotations

import argparse

import pytest

from deepagents_code import _env_vars
from deepagents_code.config_commands import (
    _display_value,
    _missing_extra_hint,
    _resolve,
    _run_get,
    _source_label,
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
    """A secret option never renders its raw value, only configured state."""
    secret = ConfigOption(
        key="x",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        secret=True,
    )
    assert _display_value(secret, is_set=True, value="sk-supersecret") == "configured"
    assert _display_value(secret, is_set=False, value=None) == "not configured"


def test_display_value_uses_credential_language_for_non_secret_unset() -> None:
    """Non-secret credential identifiers still use configured-state language."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        secret=False,
    )
    assert _display_value(option, is_set=False, value=None) == "not configured"


def test_missing_extra_hint_checks_provider_dependency(monkeypatch) -> None:
    """Credential rows can show when their provider integration is unavailable."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        secret=True,
        dependency_module="langchain_missing_provider",
        install_extra="missing-provider",
    )
    monkeypatch.setattr(
        "deepagents_code.config_commands.importlib.util.find_spec",
        lambda name: None if name == "langchain_missing_provider" else object(),
    )
    assert _missing_extra_hint(option) is True
    assert (
        _display_value(option, is_set=True, value="sk-secret")
        == "configured, unavailable"
    )
    assert _source_label("default") == "default"


def test_run_get_json_omits_secret_value(monkeypatch, capsys) -> None:
    """JSON output for a secret option reports presence but never the value."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["set"] is True
    assert payload["data"]["value"] is None


def test_charset_auto_display_value_includes_effective_glyph_mode() -> None:
    """The charset auto value says which glyph mode is actually being used."""
    option = get_option("display.charset")
    assert option is not None
    value = _display_value(option, is_set=False, value="auto")
    assert value in {
        "auto (using Unicode glyphs)",
        "auto (using ASCII glyphs)",
    }
    assert _source_label("default") == "default"


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
    assert kwargs["interpreter_timeout_seconds"] == pytest.approx(5.0)


def test_resolve_theme_uses_terminal_mapping_before_saved_theme(monkeypatch) -> None:
    """Theme resolution mirrors startup: terminal mapping wins over `[ui].theme`."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    value, source = resolve_scalar(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "ansi-dark"},
            }
        },
    )

    assert value == "ansi-dark"
    assert source == "config.toml [ui.terminal_themes.vscode]"


def test_resolve_theme_uses_saved_theme_without_terminal_match(monkeypatch) -> None:
    """A saved `[ui].theme` is reported when no terminal mapping applies."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "unknown-terminal")

    value, source = resolve_scalar(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "ansi-dark"},
            }
        },
    )

    assert value == "atom-one-light"
    assert source == "config.toml [ui.theme]"


def test_resolve_theme_env_wins_over_config(monkeypatch) -> None:
    """The explicit theme env var wins over saved config, matching startup."""
    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_THEME", "ansi-dark")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    value, source = resolve_scalar(
        opt,
        toml_data={
            "ui": {
                "theme": "atom-one-light",
                "terminal_themes": {"vscode": "langchain"},
            }
        },
    )

    assert value == "ansi-dark"
    assert source == "env (DEEPAGENTS_CODE_THEME)"


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


# --- ConfigOption validation ------------------------------------------------


def test_config_option_rejects_type_mismatched_default() -> None:
    """A default whose type contradicts `kind` fails at construction."""
    import pytest

    with pytest.raises(TypeError, match="not valid for kind int"):
        ConfigOption(key="x", group="g", summary="s", kind=OptionKind.INT, default="5")


def test_config_option_rejects_bool_default_for_int() -> None:
    """`bool` is an `int` subclass but must not pass as an INT/FLOAT default."""
    import pytest

    with pytest.raises(TypeError, match="not valid for kind int"):
        ConfigOption(key="x", group="g", summary="s", kind=OptionKind.INT, default=True)


def test_config_option_rejects_mutable_default() -> None:
    """A mutable default would be shared by reference through the lru_cache."""
    import pytest

    with pytest.raises(TypeError, match="mutable default"):
        ConfigOption(
            key="x", group="g", summary="s", kind=OptionKind.STR, default=["a"]
        )


def test_config_option_rejects_default_on_structured() -> None:
    """STRUCTURED options are display-only pass-throughs and take no default."""
    import pytest

    with pytest.raises(TypeError, match="must not declare a default"):
        ConfigOption(
            key="x", group="g", summary="s", kind=OptionKind.STRUCTURED, default="x"
        )


# --- Coercion matrix --------------------------------------------------------


def test_resolve_bool_presence_enables_on_any_value(monkeypatch) -> None:
    """BOOL_PRESENCE treats any non-empty value as set, including '0'.

    This is the one branch whose semantics differ from BOOL, where '0' is
    falsy; here `bool(raw)` makes a literal '0' enable the flag.
    """
    opt = get_option("debug.notifications")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL_PRESENCE
    monkeypatch.setenv(opt.env_var, "0")
    assert resolve_scalar(opt, toml_data={})[0] is True
    monkeypatch.setenv(opt.env_var, "")
    # An empty value is unset (see resolve_scalar), so it falls back to default.
    assert resolve_scalar(opt, toml_data={}) == (False, "default")


def test_resolve_malformed_int_env_falls_back_with_warning(monkeypatch, caplog) -> None:
    """A non-numeric env value for an INT option logs and falls back.

    Interpreter options are TOML-only, so the int env-coercion branch is
    exercised through a synthetic option with an env var.
    """
    import logging

    int_opt = ConfigOption(
        key="t.int",
        group="g",
        summary="s",
        kind=OptionKind.INT,
        default=7,
        env_var="DEEPAGENTS_CODE_TEST_INT",
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_INT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(int_opt, toml_data={})
    assert (value, source) == (7, "default")
    assert any("TEST_INT" in r.getMessage() for r in caplog.records)


def test_resolve_toml_int_rejects_bool() -> None:
    """A TOML boolean must not coerce to an INT (bool is an int subclass)."""
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    assert resolve_scalar(
        opt, toml_data={"interpreter": {"memory_limit_mb": True}}
    ) == (64, "default")


def test_resolve_toml_float_rejects_bool() -> None:
    """A TOML boolean must not coerce to a FLOAT."""
    opt = get_option("interpreter.timeout_seconds")
    assert opt is not None
    assert resolve_scalar(
        opt, toml_data={"interpreter": {"timeout_seconds": True}}
    ) == (5.0, "default")


def test_resolve_structured_passes_value_through() -> None:
    """STRUCTURED options return the raw table verbatim for display."""
    opt = get_option("threads.columns")
    assert opt is not None
    assert opt.kind is OptionKind.STRUCTURED
    table = {"created": True, "updated": False}
    assert resolve_scalar(opt, toml_data={"threads": {"columns": table}}) == (
        table,
        "config.toml",
    )


def test_resolve_malformed_skills_dir_env_falls_back(monkeypatch, caplog) -> None:
    """An unresolvable skills-dir env path logs and falls back, never raising."""
    import logging

    opt = get_option("skills.extra_allowed_dirs")
    assert opt is not None
    # `~nobodyuser_xyz` cannot resolve to a home directory; `expanduser` raises
    # RuntimeError, which the resolver must catch.
    monkeypatch.setenv(opt.env_var, "~nobodyuser_xyz/skills")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == (None, "default")
    assert any("could not resolve" in r.getMessage() for r in caplog.records)


def test_resolve_malformed_skills_dir_toml_falls_back(caplog) -> None:
    """An unresolvable skills-dir in config.toml logs and falls back."""
    import logging

    opt = get_option("skills.extra_allowed_dirs")
    assert opt is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(
            opt,
            toml_data={"skills": {"extra_allowed_dirs": ["~nobodyuser_xyz/skills"]}},
        )
    assert (value, source) == (None, "default")
    assert any("could not resolve" in r.getMessage() for r in caplog.records)


# --- load_config_toml -------------------------------------------------------


def test_load_config_toml_absent_returns_empty(monkeypatch, tmp_path) -> None:
    """An absent config file is not an error: returns {} silently."""
    from deepagents_code import config_manifest, model_config

    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", tmp_path / "missing.toml")
    assert config_manifest.load_config_toml() == {}


def test_load_config_toml_corrupt_returns_empty_with_warning(
    monkeypatch, tmp_path, caplog
) -> None:
    """A corrupt config file logs a warning and falls back to {}."""
    import logging

    from deepagents_code import config_manifest, model_config

    bad = tmp_path / "config.toml"
    bad.write_text("this is = not valid = toml ][")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", bad)
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert config_manifest.load_config_toml() == {}
    assert any("Could not read config" in r.getMessage() for r in caplog.records)


def test_load_config_toml_valid_parses(monkeypatch, tmp_path) -> None:
    """A valid config file is parsed into a mapping."""
    from deepagents_code import config_manifest, model_config

    good = tmp_path / "config.toml"
    good.write_text("[interpreter]\nmemory_limit_mb = 128\n")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", good)
    assert config_manifest.load_config_toml() == {
        "interpreter": {"memory_limit_mb": 128}
    }


# --- Display rendering ------------------------------------------------------


def test_display_value_unset_renders_placeholder() -> None:
    """A non-secret option with no value renders the unset placeholder."""
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)
    assert _display_value(opt, is_set=False, value=None) == "(unset)"


def test_display_value_truncates_long_values() -> None:
    """A long value is truncated to 60 chars with a trailing ellipsis."""
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)
    rendered = _display_value(opt, is_set=True, value="a" * 100)
    assert len(rendered) == 60
    assert rendered.endswith("\N{HORIZONTAL ELLIPSIS}")


def test_config_show_text_survives_markup_in_value(monkeypatch) -> None:
    """A value containing Rich close-tag markup must not crash text rendering."""
    monkeypatch.setenv(
        _env_vars.EXTERNAL_EVENT_SOCKET_PATH,
        "/tmp/sock[/]oops",
    )
    args = argparse.Namespace(config_command="show", output_format="text")
    assert run_config_command(args) == 0


# --- Command smoke (text paths) ---------------------------------------------


def test_run_show_text_returns_zero() -> None:
    """The default (text) `config show` rendering path runs without error."""
    args = argparse.Namespace(config_command="show", output_format="text")
    assert run_config_command(args) == 0


def test_run_list_text_returns_zero() -> None:
    """The default (text) `config list` rendering path runs without error."""
    args = argparse.Namespace(config_command="list", output_format="text")
    assert run_config_command(args) == 0


def test_run_get_text_returns_zero() -> None:
    """The default (text) `config get` rendering path runs without error."""
    args = argparse.Namespace(
        config_command="get", key="interpreter.memory_limit_mb", output_format="text"
    )
    assert run_config_command(args) == 0


def test_run_path_text_returns_zero() -> None:
    """The `config path` rendering path runs without error."""
    args = argparse.Namespace(config_command="path", output_format="text")
    assert run_config_command(args) == 0
