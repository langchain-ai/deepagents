"""Drift, resolution, and behavior tests for the configuration manifest.

These guard the contract that the manifest is the single source of truth for
the scalar config surface, that its resolver matches what the runtime reads,
and that secret-flagged options are never rendered by value.
"""

from __future__ import annotations

import argparse
import os

import pytest

from deepagents_code import _env_vars
from deepagents_code.client.commands.config import (
    _display_value,
    _missing_extra_hint,
    _resolve,
    _run_get,
    _source_label,
    run_config_command,
)
from deepagents_code.config_manifest import (
    CURSOR_STYLE_DEFAULT,
    NON_OPTION_ENV_VARS,
    ConfigOption,
    OptionKind,
    get_config_options,
    get_option,
    is_provider_package_installed,
    option_keys,
    provider_install_extra,
    resolve_interpreter_kwargs,
    resolve_scalar,
)
from deepagents_code.model_config import DEFAULT_STARTUP_MODE, PROVIDER_API_KEY_ENV

# Most unit tests set `DEEPAGENTS_CODE_NO_UPDATE_CHECK=1` to avoid accidental
# PyPI/DNS work. This module checks whether update settings came from the env,
# config file, or built-in defaults; adding the env var here would hide the
# config/default cases these tests are trying to verify.
pytestmark = pytest.mark.self_managed_update_check


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


def test_memory_auto_save_defaults_enabled(monkeypatch) -> None:
    """`memory.auto_save` resolves to enabled when nothing overrides it."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    assert resolve_scalar(option, toml_data={}) == (True, "default")


def test_memory_auto_save_env_disables(monkeypatch) -> None:
    """A falsy `DEEPAGENTS_CODE_MEMORY_AUTO_SAVE` disables auto-save."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "0")
    value, _ = resolve_scalar(option, toml_data={})
    assert value is False


def test_memory_auto_save_empty_env_disables(monkeypatch) -> None:
    """An explicitly empty env override disables automatic memory saving."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "")
    assert resolve_scalar(option, toml_data={}) == (
        False,
        f"env ({_env_vars.MEMORY_AUTO_SAVE})",
    )


def test_memory_auto_save_toml_disables(monkeypatch) -> None:
    """`[memory].auto_save = false` in config.toml disables auto-save."""
    option = get_option("memory.auto_save")
    assert option is not None
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    value, _ = resolve_scalar(option, toml_data={"memory": {"auto_save": False}})
    assert value is False


def test_is_memory_auto_save_enabled_reads_env(monkeypatch) -> None:
    """The `config.is_memory_auto_save_enabled` helper honors the env override."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_memory_auto_save_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    assert is_memory_auto_save_enabled() is True

    monkeypatch.setenv(_env_vars.MEMORY_AUTO_SAVE, "false")
    assert is_memory_auto_save_enabled() is False


def test_is_memory_auto_save_enabled_reads_toml(monkeypatch) -> None:
    """The helper honors `[memory].auto_save` from `config.toml` when env is unset."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_memory_auto_save_enabled

    monkeypatch.delenv(_env_vars.MEMORY_AUTO_SAVE, raising=False)
    monkeypatch.setattr(
        config_manifest,
        "load_config_toml",
        lambda: {"memory": {"auto_save": False}},
    )
    assert is_memory_auto_save_enabled() is False


def test_goal_auto_accept_criteria_defaults_to_review(monkeypatch) -> None:
    """Auto mode reviews generated goal criteria when no preference is set."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.delenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raising=False)

    assert resolve_scalar(option, toml_data={}) == (False, "default")


def test_goal_auto_accept_criteria_reads_toml(monkeypatch) -> None:
    """The goals table can opt Auto into applying criteria automatically."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.delenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raising=False)

    assert resolve_scalar(
        option,
        toml_data={"goals": {"auto_accept_criteria": True}},
    ) == (True, "config.toml")


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("0", False)])
def test_goal_auto_accept_criteria_env_overrides_toml(
    monkeypatch,
    raw: str,
    expected: bool,
) -> None:
    """A recognized env value takes priority over config.toml."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.setenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, raw)

    assert resolve_scalar(
        option,
        toml_data={"goals": {"auto_accept_criteria": not expected}},
    ) == (expected, f"env ({_env_vars.GOAL_AUTO_ACCEPT_CRITERIA})")


@pytest.mark.parametrize(
    ("toml_data", "expected"),
    [
        ({"goals": {"auto_accept_criteria": True}}, (True, "config.toml")),
        ({}, (False, "default")),
    ],
)
def test_invalid_goal_auto_accept_env_falls_through(
    monkeypatch,
    toml_data: dict[str, object],
    expected: tuple[bool, str],
) -> None:
    """An invalid env value should not mask TOML or the safe default."""
    option = get_option("goals.auto_accept_criteria")
    assert option is not None
    monkeypatch.setenv(_env_vars.GOAL_AUTO_ACCEPT_CRITERIA, "maybe")

    assert resolve_scalar(option, toml_data=toml_data) == expected


def test_debug_log_level_resolves_dynamic_default(monkeypatch) -> None:
    """The effective log level follows debug mode when no level is explicit."""
    option = get_option("debug.log_level")
    assert option is not None
    monkeypatch.delenv(_env_vars.LOG_LEVEL, raising=False)

    monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    assert resolve_scalar(option, toml_data={}) == ("INFO", "default")

    monkeypatch.setenv(_env_vars.DEBUG, "true")
    assert resolve_scalar(option, toml_data={}) == ("DEBUG", "default")


def test_debug_log_level_validates_explicit_value(monkeypatch) -> None:
    """Explicit log levels are normalized and invalid values use the runtime default."""
    option = get_option("debug.log_level")
    assert option is not None
    monkeypatch.setenv(_env_vars.DEBUG, "true")

    monkeypatch.setenv(_env_vars.LOG_LEVEL, " warning ")
    assert resolve_scalar(option, toml_data={}) == (
        "WARNING",
        f"env ({_env_vars.LOG_LEVEL})",
    )

    monkeypatch.setenv(_env_vars.LOG_LEVEL, "TRACE")
    assert resolve_scalar(option, toml_data={}) == ("DEBUG", "default")


@pytest.mark.parametrize(("debug", "expected"), [(None, "INFO"), ("1", "DEBUG")])
def test_config_get_and_aggregate_report_dynamic_log_level(
    monkeypatch, capsys, debug: str | None, expected: str
) -> None:
    """Both config command paths report the runtime's effective default."""
    import json

    monkeypatch.delenv(_env_vars.LOG_LEVEL, raising=False)
    if debug is None:
        monkeypatch.delenv(_env_vars.DEBUG, raising=False)
    else:
        monkeypatch.setenv(_env_vars.DEBUG, debug)

    get_args = argparse.Namespace(
        config_command="get",
        key="debug.log_level",
        output_format="json",
    )
    assert run_config_command(get_args) == 0
    get_payload = json.loads(capsys.readouterr().out)
    assert get_payload["data"]["value"] == expected

    config_args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(config_args) == 0
    config_payload = json.loads(capsys.readouterr().out)
    row = next(
        item for item in config_payload["data"] if item["key"] == "debug.log_level"
    )
    assert row["value"] == expected


# --- Provider install helpers ----------------------------------------------


def test_provider_install_extra_known_provider() -> None:
    """Known providers resolve to their installing extra."""
    assert provider_install_extra("baseten") == "baseten"
    # Provider name uses underscores; the extra uses hyphens.
    assert provider_install_extra("google_genai") == "google-genai"


def test_provider_install_extra_extra_only_provider() -> None:
    """Providers without required API keys can still resolve to extras."""
    assert provider_install_extra("bedrock") == "bedrock"
    assert provider_install_extra("ollama") == "ollama"


def test_provider_install_extra_unknown_provider() -> None:
    """Providers without a curated extra resolve to `None`."""
    assert provider_install_extra("not-a-real-provider") is None


def test_is_provider_package_installed_unknown_provider() -> None:
    """Providers without a curated extra are reported as installed."""
    assert is_provider_package_installed("not-a-real-provider") is True


def test_is_provider_package_installed_core_provider() -> None:
    """Core providers ship as base dependencies and are always importable."""
    assert is_provider_package_installed("openai") is True


def test_is_provider_package_installed_missing_extra() -> None:
    """A known provider whose package is absent is reported as not installed."""
    import importlib.util

    if importlib.util.find_spec("langchain_baseten") is not None:
        pytest.skip("langchain_baseten is installed in this environment")
    assert is_provider_package_installed("baseten") is False


# --- Secrets ----------------------------------------------------------------


def test_api_key_credentials_are_secret() -> None:
    """Credential options backed by key/token env vars must be secret-flagged."""
    for opt in get_config_options():
        if opt.group != "Credentials" or not opt.env_var:
            continue
        looks_secret = any(
            marker in opt.env_var for marker in ("KEY", "TOKEN", "APIKEY")
        )
        assert opt.redacted is looks_secret, (
            f"{opt.key} redacted={opt.redacted} but env_var {opt.env_var!r} "
            f"implies redacted={looks_secret}"
        )


def test_google_cloud_project_is_not_secret() -> None:
    """The Vertex project identifier is not secret material and shows its value."""
    opt = get_option("credentials.google_vertexai")
    assert opt is not None
    assert opt.env_var == "GOOGLE_CLOUD_PROJECT"
    assert opt.redacted is False


def test_display_value_redacts_secrets() -> None:
    """A secret option never renders its raw value, only configured state."""
    option = ConfigOption(
        key="x",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=True,
    )
    assert _display_value(option, is_set=True, value="sk-supersecret") == "configured"
    assert _display_value(option, is_set=False, value=None) == "not configured"


def test_display_value_uses_credential_language_for_non_secret_unset() -> None:
    """Non-secret credential identifiers still use configured-state language."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=False,
    )
    assert _display_value(option, is_set=False, value=None) == "not configured"


def test_missing_extra_hint_checks_provider_dependency(monkeypatch) -> None:
    """Credential rows can show when their provider integration is unavailable."""
    option = ConfigOption(
        key="credentials.example",
        group="Credentials",
        summary="",
        kind=OptionKind.STR,
        redacted=True,
        dependency_module="langchain_missing_provider",
        install_extra="missing-provider",
    )
    monkeypatch.setattr(
        "deepagents_code.client.commands.config.importlib.util.find_spec",
        lambda name: None if name == "langchain_missing_provider" else object(),
    )
    assert _missing_extra_hint(option) is True
    assert (
        _display_value(option, is_set=True, value="sk-secret")
        == "configured, unavailable"
    )
    assert _source_label("default") == "default"


def test_source_label_marks_prefixed_credential_env_as_session_override() -> None:
    """Only prefixed credential env sources get the session-override note."""
    credential = get_option("credentials.anthropic")
    assert credential is not None
    display = get_option("display.theme")
    assert display is not None

    assert (
        _source_label(
            "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY)",
            option=credential,
        )
        == "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY); session override"
    )
    assert (
        _source_label("env (ANTHROPIC_API_KEY)", option=credential)
        == "env (ANTHROPIC_API_KEY)"
    )
    assert (
        _source_label("env (DEEPAGENTS_CODE_THEME)", option=display)
        == "env (DEEPAGENTS_CODE_THEME)"
    )


def test_run_get_json_omits_secret_value(monkeypatch, capsys) -> None:
    """JSON output for a secret option reports presence but never the value."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["set"] is True
    assert payload["data"]["value"] is None


@pytest.fixture
def stored_auth_dir(tmp_path, monkeypatch):
    """Redirect the credential store into a temp dir so `/auth` keys are isolated."""
    state_dir = tmp_path / ".deepagents" / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)
    return state_dir


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_prefers_stored_over_env(monkeypatch):
    """A `/auth`-stored key wins over an env var, matching runtime precedence."""
    from deepagents_code import auth_store

    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "stored"
    assert value == "from-store"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_prefers_prefixed_env_over_stored(monkeypatch):
    """A prefixed credential env var stays authoritative over a stored key."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "from-prefix")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY)"
    assert value == "from-prefix"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_empty_prefixed_credential_blocks_stored(monkeypatch):
    """An empty prefixed credential suppresses the stored key like the runtime."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "")
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is False
    assert source == "default"
    assert value is None


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_uses_stored_when_env_unset(monkeypatch):
    """A stored key is surfaced even with no env var set."""
    from deepagents_code import auth_store

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", raising=False)
    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, _ = _resolve(option, {})
    assert is_set is True
    assert source == "stored"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_falls_back_to_env_without_stored(monkeypatch):
    """With no stored key, resolution falls through to the env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"


def test_resolve_credential_corrupt_store_falls_back_to_env(
    stored_auth_dir, monkeypatch, caplog
):
    """A corrupt `auth.json` is logged and treated as absent, not a hard error."""
    import logging

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    with caplog.at_level(logging.WARNING):
        is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"
    assert "treating as absent" in caplog.text


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_text_reports_stored_source(capsys):
    """`config get` shows a stored credential as configured from the store."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "configured" in out
    assert "stored" in out
    assert "from-store" not in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_text_reports_prefixed_env_as_session_override(monkeypatch, capsys):
    """`config get` labels a visible prefixed credential env var as session-scoped."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "from-prefix")
    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    compact = " ".join(out.split())
    assert "configured" in out
    assert "env (DEEPAGENTS_CODE_ANTHROPIC_API_KEY); session override" in compact
    assert "from-prefix" not in out
    assert "from-store" not in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_json_redacts_stored_secret_value(capsys):
    """`config get --json` reports a stored credential as set but never its value."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    assert _run_get("credentials.anthropic", "json") == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)["data"]
    assert payload["set"] is True
    assert payload["source"] == "stored"
    assert payload["value"] is None
    assert "from-store" not in raw


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_json_redacts_stored_secret_value(capsys):
    """`config --json` redacts a stored secret on the aggregate path too."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    raw = capsys.readouterr().out
    rows = json.loads(raw)["data"]
    row = next(r for r in rows if r["key"] == "credentials.anthropic")
    assert row["set"] is True
    assert row["source"] == "stored"
    assert row["value"] is None
    assert "from-store" not in raw


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_text_reports_stored_source(capsys):
    """`config` (aggregate text path) shows a stored credential as configured."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0
    out = capsys.readouterr().out
    assert "configured" in out
    assert "stored" in out
    assert "from-store" not in out


def test_resolve_empty_stored_key_falls_back_to_env(stored_auth_dir, monkeypatch):
    """A stored entry with a blank key does not mask a working env var."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "credentials": {
                    "anthropic": {"type": "api_key", "key": "", "added_at": ""}
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "env (ANTHROPIC_API_KEY)"
    assert value == "from-env"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_non_credential_ignores_store():
    """Non-credential options never consult the `/auth` store."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("anthropic", "from-store")
    option = get_option("display.show_header")
    assert option is not None
    _, source, _ = _resolve(option, {})
    assert source != "stored"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_tavily_service_prefers_stored(monkeypatch):
    """A stored key for the tavily *service* resolves with a stored source.

    Guards that the credential branch keys on group membership (and the
    `provider` field), not on model-provider-registry membership.
    """
    from deepagents_code import auth_store

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_TAVILY_API_KEY", raising=False)
    auth_store.set_stored_key("tavily", "from-store")
    option = get_option("credentials.tavily")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "stored"
    assert value == "from-store"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_tavily_prefixed_env_overrides_stored(monkeypatch):
    """A prefixed tavily env var wins over the stored key, matching runtime."""
    from deepagents_code import auth_store

    monkeypatch.setenv("DEEPAGENTS_CODE_TAVILY_API_KEY", "from-prefix")
    auth_store.set_stored_key("tavily", "from-store")
    option = get_option("credentials.tavily")
    assert option is not None
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "env (DEEPAGENTS_CODE_TAVILY_API_KEY)"
    assert value == "from-prefix"


def test_run_get_json_flags_unreadable_store(stored_auth_dir, capsys):
    """`config get --json` for a credential surfaces a store-read failure in-band."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("credentials.anthropic", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert "store_error" in payload
    # Redaction still holds even when the store is unreadable.
    assert payload["value"] is None


def test_run_get_json_non_credential_omits_store_error(stored_auth_dir, capsys):
    """A non-credential `config get --json` never carries a `store_error` key."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("display.show_header", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert "store_error" not in payload


def test_run_config_json_flags_unreadable_store(stored_auth_dir, capsys):
    """`config --json` marks credential rows when the store is unreadable."""
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    cred_rows = [r for r in rows if r["group"] == "Credentials"]
    assert cred_rows
    assert all("store_error" in r for r in cred_rows)
    assert all("store_error" not in r for r in rows if r["group"] != "Credentials")


def test_run_config_text_warns_on_unreadable_store(stored_auth_dir, capsys):
    """`config` text output warns when the credential store is unreadable.

    Guards the `_print_store_warning` call in both text renderers: without it a
    corrupt store would look identical to an empty one in the interactive view —
    the silent failure this warning exists to prevent.
    """
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    for verbose in (False, True):
        args = argparse.Namespace(
            config_command=None, output_format="text", verbose=verbose
        )
        assert run_config_command(args) == 0
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "unreadable" in out


def test_run_get_text_warns_on_unreadable_store(stored_auth_dir, capsys):
    """`config get` text output shows a warning banner for an unreadable store."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "unreadable" in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_config_reads_store_once(monkeypatch):
    """`config` parses the credential store once, not once per option."""
    from deepagents_code import auth_store

    calls = 0
    real_load = auth_store.load_credentials

    def _counting_load() -> dict:
        nonlocal calls
        calls += 1
        return real_load()

    monkeypatch.setattr(auth_store, "load_credentials", _counting_load)
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    # One read for the whole command, regardless of how many credential options
    # exist — guards the single-snapshot design against a per-option regression.
    assert calls == 1


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_non_redacted_credential_shows_stored_value(monkeypatch):
    """A non-redacted stored credential (the Vertex project) shows its value."""
    from deepagents_code import auth_store

    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_GOOGLE_CLOUD_PROJECT", raising=False)
    auth_store.set_stored_key("google_vertexai", "my-project")
    option = get_option("credentials.google_vertexai")
    assert option is not None
    assert option.redacted is False
    is_set, source, value = _resolve(option, {})
    assert is_set is True
    assert source == "stored"
    assert value == "my-project"


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_json_shows_non_redacted_stored_value(capsys):
    """`config get --json` surfaces a non-redacted stored value (not `None`)."""
    import json

    from deepagents_code import auth_store

    auth_store.set_stored_key("google_vertexai", "my-project")
    assert _run_get("credentials.google_vertexai", "json") == 0
    payload = json.loads(capsys.readouterr().out)["data"]
    assert payload["source"] == "stored"
    assert payload["redacted"] is False
    assert payload["value"] == "my-project"


def test_run_get_non_utf8_store_does_not_crash(stored_auth_dir, capsys):
    """A non-UTF-8 `auth.json` degrades to a warning banner, not a traceback."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_bytes(b"\xff\xfe not utf-8")
    assert _run_get("credentials.anthropic", "text") == 0
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "unreadable" in out


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
    """An empty (prefixed) env var is unset for `config`, as the app sees it.

    The runtime `resolve_env_var` returns `None` for an empty prefixed var (and
    a prefixed empty suppresses the canonical). `resolve_scalar` must agree, or
    `config` would report a credential as "set" that the app treats as
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


def test_langsmith_project_prefers_prefixed_env(monkeypatch) -> None:
    """The prefixed project env var wins over a bare `LANGSMITH_PROJECT`."""
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "prefixed")
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == ("prefixed", "env (DEEPAGENTS_CODE_LANGSMITH_PROJECT)")


def test_langsmith_project_falls_back_to_bare_env(monkeypatch) -> None:
    """A bare `LANGSMITH_PROJECT` resolves when the prefixed var is unset.

    Mirrors `get_langsmith_project_name`, so `config`/`config get` report the
    project agent traces actually route to.
    """
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_default_when_unset(monkeypatch) -> None:
    """With no project env var set, the default project name is rendered."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert resolve_scalar(opt, toml_data={}) == (LANGSMITH_PROJECT_DEFAULT, "default")


def test_langsmith_project_empty_prefixed_falls_through_to_bare(monkeypatch) -> None:
    """An empty prefixed var is skipped, so a set bare `LANGSMITH_PROJECT` wins.

    This is the opposite of the single-name credential path
    (`test_resolve_empty_env_is_unset_matching_resolve_env_var`): with a
    fallback declared, an empty prefixed var does not suppress resolution — it
    falls through to the next name, matching `get_langsmith_project_name`.
    """
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "")
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_empty_bare_is_default(monkeypatch) -> None:
    """An empty bare `LANGSMITH_PROJECT` is unset, so the default applies."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    assert resolve_scalar(opt, toml_data={}) == (LANGSMITH_PROJECT_DEFAULT, "default")


def test_fallback_env_vars_yield_to_toml_when_env_unset(monkeypatch) -> None:
    """A synthetic option exercises the empty-fallback → `config.toml` path.

    No shipping option both declares `fallback_env_vars` and has `toml_keys`,
    so this guards the generic resolver: an empty fallback env var must fall
    through to `config.toml`, while a set one still wins over it.
    """
    opt = ConfigOption(
        key="synthetic.fallback_toml",
        group="Synthetic",
        summary="Synthetic option for fallback + TOML precedence.",
        kind=OptionKind.STR,
        env_var=_env_vars.LANGSMITH_PROJECT,
        fallback_env_vars=("SYNTHETIC_FALLBACK",),
        toml_keys=("synthetic", "value"),
    )
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    toml_data = {"synthetic": {"value": "from-toml"}}

    monkeypatch.setenv("SYNTHETIC_FALLBACK", "")
    assert resolve_scalar(opt, toml_data=toml_data) == ("from-toml", "config.toml")

    monkeypatch.setenv("SYNTHETIC_FALLBACK", "from-env")
    assert resolve_scalar(opt, toml_data=toml_data) == (
        "from-env",
        "env (SYNTHETIC_FALLBACK)",
    )


@pytest.mark.parametrize(
    "bad_fallback",
    [
        ["LANGSMITH_PROJECT"],  # mutable list reintroduces the lru_cache hazard
        ("",),  # empty name never matches any env var
        ("LANGSMITH_PROJECT", ""),  # one valid, one empty
    ],
)
def test_fallback_env_vars_rejects_invalid(bad_fallback) -> None:
    """`__post_init__` rejects non-tuple or empty/non-str `fallback_env_vars`."""
    with pytest.raises(TypeError, match="fallback_env_vars must be a tuple"):
        ConfigOption(
            key="synthetic.bad_fallback",
            group="Synthetic",
            summary="Synthetic option with an invalid fallback.",
            kind=OptionKind.STR,
            fallback_env_vars=bad_fallback,
        )


def test_run_config_json_redacts_every_secret(monkeypatch, capsys) -> None:
    """The `config` aggregate (separate path from `get`) never leaks a secret."""
    import json

    monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-secret")
    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    assert any(r["key"] == "credentials.anthropic" and r["set"] for r in rows)
    assert all(r["value"] is None for r in rows if r["redacted"])


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


def test_cursor_style_option_definition() -> None:
    """Cursor style supports env and config.toml and defaults to a block."""
    opt = get_option("display.cursor_style")
    assert opt is not None
    assert opt.kind is OptionKind.CURSOR_STYLE_DELEGATE
    assert opt.default == CURSOR_STYLE_DEFAULT
    assert opt.toml_keys == ("ui", "cursor_style")
    assert opt.env_var == _env_vars.CURSOR_STYLE


def test_resolve_cursor_style_from_env(monkeypatch, caplog) -> None:
    """A valid env value wins; an invalid value falls through to config.toml."""
    import logging

    opt = get_option("display.cursor_style")
    assert opt is not None
    toml_data = {"ui": {"cursor_style": "underline"}}

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "block")
    assert resolve_scalar(opt, toml_data=toml_data) == (
        "block",
        f"env ({_env_vars.CURSOR_STYLE})",
    )

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "bar")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert resolve_scalar(opt, toml_data=toml_data) == (
            "underline",
            "config.toml",
        )
    assert any(
        _env_vars.CURSOR_STYLE in record.getMessage() for record in caplog.records
    )


def test_resolve_cursor_style_from_toml(caplog) -> None:
    """Only supported cursor styles are accepted from config.toml."""
    import logging

    opt = get_option("display.cursor_style")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"ui": {"cursor_style": "underline"}}) == (
        "underline",
        "config.toml",
    )

    for raw in ("bar", 1):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            value, source = resolve_scalar(opt, toml_data={"ui": {"cursor_style": raw}})
        assert (value, source) == (CURSOR_STYLE_DEFAULT, "default")
        assert any(
            "[ui].cursor_style" in record.getMessage() for record in caplog.records
        )

    assert resolve_scalar(opt, toml_data={}) == (CURSOR_STYLE_DEFAULT, "default")


def test_collapse_pastes_default_enabled() -> None:
    """Paste collapsing is enabled by default when unset."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL
    assert resolve_scalar(opt, toml_data={}) == (True, "default")


def test_collapse_pastes_env_disables(monkeypatch) -> None:
    """A falsy env var disables paste collapsing."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "0")
    assert resolve_scalar(opt, toml_data={})[0] is False


def test_collapse_pastes_toml_disables() -> None:
    """A `[ui].collapse_pastes = false` entry disables paste collapsing."""
    opt = get_option("display.collapse_pastes")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"ui": {"collapse_pastes": False}}) == (
        False,
        "config.toml",
    )


def test_thread_relative_time_default_matches_runtime_loader() -> None:
    """Fresh thread config shows relative timestamps by default."""
    opt = get_option("threads.relative_time")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={}) == (True, "default")


def test_auto_update_resolves_persisted_config() -> None:
    """`set_auto_update()` writes the TOML path surfaced by the manifest."""
    opt = get_option("update.auto_update")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"update": {"auto_update": True}}) == (
        True,
        "config.toml",
    )


def test_no_update_check_env_uses_presence_semantics(monkeypatch) -> None:
    """Any non-empty no-update-check env var disables checks, including '0'."""
    opt = get_option("update.no_update_check")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL_PRESENCE
    monkeypatch.setenv(_env_vars.NO_UPDATE_CHECK, "0")
    assert resolve_scalar(opt, toml_data={}) == (
        True,
        f"env ({_env_vars.NO_UPDATE_CHECK})",
    )


def test_no_update_check_resolves_inverted_persisted_check() -> None:
    """`[update].check = false` means the effective no-check flag is enabled."""
    opt = get_option("update.no_update_check")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"update": {"check": False}}) == (
        True,
        "config.toml",
    )
    assert resolve_scalar(opt, toml_data={"update": {"check": True}}) == (
        False,
        "config.toml",
    )


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
    err = capsys.readouterr().err
    assert "Unknown config option" in err
    assert "config --verbose" in err


def test_config_registered_as_bare_action_group() -> None:
    """Bare `config` must run its action instead of startup-fast-path help."""
    from deepagents_code import ui
    from deepagents_code.main import _BARE_ACTION_GROUPS, _HELP_SPECS

    assert "config" in _BARE_ACTION_GROUPS
    assert "config" not in _HELP_SPECS
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


def test_config_option_rejects_inverted_non_bool_toml() -> None:
    """Only boolean TOML options can use inverted config-file semantics."""
    import pytest

    with pytest.raises(TypeError, match="requires a boolean option kind"):
        ConfigOption(
            key="x",
            group="g",
            summary="s",
            kind=OptionKind.STR,
            default="x",
            toml_keys=("section", "key"),
            invert_toml_bool=True,
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


def test_config_text_survives_markup_in_value(monkeypatch) -> None:
    """A value containing Rich close-tag markup must not crash text rendering."""
    monkeypatch.setenv(
        _env_vars.EXTERNAL_EVENT_SOCKET_PATH,
        "/tmp/sock[/]oops",
    )
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0


def test_config_verbose_text_survives_markup_in_value(monkeypatch) -> None:
    """The verbose text path escapes markup in values so rendering can't break.

    `_print_config_verbose` renders with markup enabled and relies on manual
    `escape()`; the compact table path uses `Text` cells, so it needs its own
    guard.
    """
    monkeypatch.setenv(
        _env_vars.EXTERNAL_EVENT_SOCKET_PATH,
        "/tmp/sock[/]oops",
    )
    args = argparse.Namespace(config_command=None, output_format="text", verbose=True)
    assert run_config_command(args) == 0


# --- Command smoke (text paths) ---------------------------------------------


def test_run_config_text_returns_zero() -> None:
    """The default (text) `config` rendering path runs without error."""
    args = argparse.Namespace(config_command=None, output_format="text")
    assert run_config_command(args) == 0


def test_run_config_verbose_text_shows_descriptions(capsys) -> None:
    """`config --verbose` folds in each option's description and how-to-set.

    A plain exit-code check would still pass if the verbose path silently
    regressed to the compact table, so assert the distinguishing content — an
    option's summary and its `set via` line — is actually rendered.
    """
    opt = get_option("interpreter.memory_limit_mb")
    assert opt is not None
    args = argparse.Namespace(config_command=None, output_format="text", verbose=True)
    assert run_config_command(args) == 0
    # Normalize whitespace so Rich soft-wrapping at the test console width can't
    # break the substring match.
    rendered = " ".join(capsys.readouterr().out.split())
    assert " ".join(opt.summary.split()) in rendered
    assert "set via" in rendered


def test_config_parser_wires_default_and_verbose_flag(monkeypatch) -> None:
    """Real argparse wiring parses bare `config --all --json` as verbose JSON."""
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", ["dcode", "config", "--all", "--json"])
    ns = parse_args()
    assert ns.config_command is None
    assert ns.verbose is True
    assert ns.output_format == "json"


@pytest.mark.parametrize("removed_subcommand", ["show", "list", "ls"])
def test_config_parser_rejects_removed_subcommands(
    monkeypatch, removed_subcommand
) -> None:
    """Removed display subcommands are no longer registered on the parser."""
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", ["dcode", "config", removed_subcommand])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


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


# --- BOOL env coercion ------------------------------------------------------


def test_resolve_bool_unrecognized_env_falls_back_with_warning(
    monkeypatch, caplog
) -> None:
    """An unrecognized boolean env token logs and falls through, not source=env.

    `is_env_truthy` would silently return the default for `maybe`, but the
    resolver must not then credit the env var with that value: doing so would
    make `config` report `source=env` for a variable the runtime ignored.
    """
    import logging

    opt = get_option("display.hide_cwd")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "maybe")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == (False, "default")
    assert any("expected bool" in r.getMessage() for r in caplog.records)


# --- FLOAT / shell-list env coercion ---------------------------------------


def test_resolve_float_env_coerces_and_falls_back(monkeypatch, caplog) -> None:
    """The FLOAT env branch coerces a number and logs+falls back on garbage.

    Interpreter floats are TOML-only, so — like the INT branch — a synthetic
    env-backed option exercises both arms of `_coerce_env`'s FLOAT path.
    """
    import logging

    float_opt = ConfigOption(
        key="t.float",
        group="g",
        summary="s",
        kind=OptionKind.FLOAT,
        default=1.5,
        env_var="DEEPAGENTS_CODE_TEST_FLOAT",
    )
    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_FLOAT", "2.5")
    value, source = resolve_scalar(float_opt, toml_data={})
    assert value == pytest.approx(2.5)
    assert source == "env (DEEPAGENTS_CODE_TEST_FLOAT)"

    monkeypatch.setenv("DEEPAGENTS_CODE_TEST_FLOAT", "not-a-number")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(float_opt, toml_data={})
    assert (value, source) == (1.5, "default")
    assert any("TEST_FLOAT" in r.getMessage() for r in caplog.records)


def test_resolve_shell_list_env_happy_and_invalid(monkeypatch, caplog) -> None:
    """The shell-list env delegate parses a valid list and rejects bad input."""
    import logging

    opt = get_option("shell.allow_list")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "git status,ls")
    value, source = resolve_scalar(opt, toml_data={})
    assert source == f"env ({opt.env_var})"
    assert isinstance(value, list)
    assert "ls" in value

    # `'all'` cannot be combined with other commands; the parser raises and the
    # resolver logs + falls back rather than crashing.
    monkeypatch.setenv(opt.env_var, "all,ls")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={})
    assert source == "default"
    assert any("Ignoring invalid" in r.getMessage() for r in caplog.records)


def test_coerce_env_delegate_returns_invalid_not_raw(caplog) -> None:
    """A delegate kind reaching `_coerce_env` returns `_INVALID`, never raw.

    PTC/STRUCTURED options declare no env var, so this branch is unreachable in
    the live manifest. The guard exists so that if one ever gains an env var,
    an uncoerced raw string cannot leak into a typed `Settings` field.
    """
    import logging

    from deepagents_code.config_manifest import _INVALID, _coerce_env

    opt = get_option("interpreter.ptc")
    assert opt is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        result = _coerce_env(opt, "safe", "DEEPAGENTS_CODE_FAKE")
    assert result is _INVALID
    assert any("not env-backed" in r.getMessage() for r in caplog.records)


# --- TOML coercion (success + mismatch) ------------------------------------


def test_resolve_toml_str_success_and_type_mismatch(caplog) -> None:
    """A STR option reads a string from TOML and rejects a wrong-typed value."""
    import logging

    opt = get_option("threads.sort_order")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"threads": {"sort_order": "created_at"}}) == (
        "created_at",
        "config.toml",
    )

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={"threads": {"sort_order": 123}})
    assert (value, source) == ("updated_at", "default")
    assert any("sort_order" in r.getMessage() for r in caplog.records)


def test_startup_mode_option_definition() -> None:
    """`startup.mode` is config.toml-only and uses runtime mode validation."""
    opt = get_option("startup.mode")
    assert opt is not None
    assert opt.group == "Startup"
    assert opt.kind is OptionKind.STARTUP_MODE_DELEGATE
    assert opt.default == DEFAULT_STARTUP_MODE
    assert opt.toml_keys == ("startup", "mode")
    assert opt.env_var is None


def test_resolve_startup_mode_from_toml(caplog) -> None:
    """`startup.mode` resolves only valid runtime modes from config.toml."""
    import logging

    opt = get_option("startup.mode")
    assert opt is not None
    for mode in ("auto", "yolo"):
        assert resolve_scalar(opt, toml_data={"startup": {"mode": mode}}) == (
            mode,
            "config.toml",
        )
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(
            opt, toml_data={"startup": {"mode": "dangerously-auto"}}
        )
    assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
    assert any(
        "[startup].mode='dangerously-auto'" in r.getMessage() for r in caplog.records
    )

    for raw in (["manual"], {"name": "manual"}):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            value, source = resolve_scalar(opt, toml_data={"startup": {"mode": raw}})
        assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
        assert any("[startup].mode" in r.getMessage() for r in caplog.records)

    assert resolve_scalar(opt, toml_data={}) == (DEFAULT_STARTUP_MODE, "default")


def test_resolve_toml_float_success_non_bool() -> None:
    """A FLOAT option reads a real number from TOML and coerces an int to float."""
    opt = get_option("interpreter.timeout_seconds")
    assert opt is not None
    assert resolve_scalar(opt, toml_data={"interpreter": {"timeout_seconds": 2.5}}) == (
        2.5,
        "config.toml",
    )
    # A bare TOML integer is accepted and coerced to float.
    assert resolve_scalar(opt, toml_data={"interpreter": {"timeout_seconds": 3}}) == (
        3.0,
        "config.toml",
    )


# --- Theme resolution warnings ----------------------------------------------


def test_resolve_theme_unknown_env_warns(monkeypatch, caplog) -> None:
    """An unknown theme in the env var warns and falls back to the default."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_THEME", "no-such-theme")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={})
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("Unknown theme" in r.getMessage() for r in caplog.records)


def test_resolve_theme_non_table_ui_warns(monkeypatch, caplog) -> None:
    """A non-table `[ui]` value warns and falls back to the default theme."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(opt, toml_data={"ui": "oops"})
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("should be a table" in r.getMessage() for r in caplog.records)


def test_resolve_theme_unknown_saved_warns(monkeypatch, caplog) -> None:
    """An unknown saved `[ui].theme` warns and falls back to the default."""
    import logging

    from deepagents_code import theme

    opt = get_option("display.theme")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_THEME", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "no-mapping-terminal")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = resolve_scalar(
            opt, toml_data={"ui": {"theme": "no-such-theme"}}
        )
    assert (value, source) == (theme.DEFAULT_THEME, "default")
    assert any("Unknown theme" in r.getMessage() for r in caplog.records)


# --- config path: existence + OSError ---------------------------------------


def test_config_paths_logs_and_reports_missing_on_oserror(monkeypatch, caplog) -> None:
    """An `OSError` from `path.stat()` is logged and reported as missing."""
    import logging
    from pathlib import Path

    from deepagents_code import model_config
    from deepagents_code.client.commands.config import _config_paths

    target = model_config.DEFAULT_CONFIG_PATH
    real_stat = Path.stat

    def fake_stat(self, *, follow_symlinks: bool = True) -> object:
        if self == target:
            msg = "boom"
            raise OSError(msg)
        return real_stat(self, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", fake_stat)
    # The OSError guard and its debug log now live in the shared `_paths`
    # classifier that `_config_paths` delegates to.
    with caplog.at_level(logging.DEBUG, logger="deepagents_code._paths"):
        rows = _config_paths()
    config_row = next(row for row in rows if row[0] == "config.toml")
    assert config_row[2] is False
    assert any("Could not stat" in r.getMessage() for r in caplog.records)


def test_run_path_json_reports_existence(monkeypatch, tmp_path, capsys) -> None:
    """`config path --json` reports each location's existence and path."""
    import json

    from deepagents_code import model_config

    cfg = tmp_path / "config.toml"
    cfg.write_text("")
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", cfg)
    args = argparse.Namespace(config_command="path", output_format="json")
    assert run_config_command(args) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    row = next(r for r in rows if r["label"] == "config.toml")
    assert row["exists"] is True
    assert row["path"] == str(cfg)


def test_run_config_json_reports_effective_values(capsys) -> None:
    """`config --json` reports effective values without catalog fields."""
    import json

    args = argparse.Namespace(config_command=None, output_format="json")
    assert run_config_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config"
    rows = payload["data"]
    assert all(
        {"key", "group", "source", "set", "redacted", "value"} <= set(r) for r in rows
    )
    assert all("type" not in r for r in rows)


def test_run_config_verbose_json_serializes_catalog(capsys) -> None:
    """`config --verbose --json` folds the catalog into each row."""
    import json

    args = argparse.Namespace(config_command=None, output_format="json", verbose=True)
    assert run_config_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config"
    rows = payload["data"]
    assert any(
        r["key"] == "interpreter.memory_limit_mb" and r["default"] == 64 for r in rows
    )
    assert all(
        {"key", "type", "default", "redacted", "env_var", "toml_path", "cli_flag"}
        <= set(r)
        for r in rows
    )


# --- Provider/credential drift ----------------------------------------------


def test_new_provider_surfaces_after_cache_clear(monkeypatch) -> None:
    """A provider added to the registry surfaces once the option cache is cleared.

    Exercises the `cache_clear` caveat documented on `get_config_options`: the
    credential surface is regenerated from `PROVIDER_API_KEY_ENV`, so a new
    provider must produce a `credentials.<name>` option after the cache resets.
    """
    from deepagents_code import config_manifest, model_config

    patched = {
        **model_config.PROVIDER_API_KEY_ENV,
        "synthetic_xyz": "SYNTHETIC_XYZ_API_KEY",
    }
    monkeypatch.setattr(model_config, "PROVIDER_API_KEY_ENV", patched)
    config_manifest.get_config_options.cache_clear()
    config_manifest._options_by_key.cache_clear()
    try:
        opt = config_manifest.get_option("credentials.synthetic_xyz")
        assert opt is not None
        assert opt.env_var == "SYNTHETIC_XYZ_API_KEY"
        # A *_API_KEY env var is treated as secret material.
        assert opt.redacted is True
    finally:
        # Restore the cache so later tests rebuild against the real registry.
        config_manifest.get_config_options.cache_clear()
        config_manifest._options_by_key.cache_clear()


def test_provider_dependency_metadata_is_exhaustive() -> None:
    """Provider dependency metadata must cover auth and install surfaces."""
    from deepagents_code.config_manifest import _PROVIDER_DEPENDENCIES
    from deepagents_code.extras_info import MODEL_PROVIDER_EXTRAS

    assert set(PROVIDER_API_KEY_ENV) <= set(_PROVIDER_DEPENDENCIES), (
        "_PROVIDER_DEPENDENCIES must include every provider credential so config "
        "availability hints stay complete"
    )
    assert {extra for _module, extra in _PROVIDER_DEPENDENCIES.values()} == set(
        MODEL_PROVIDER_EXTRAS
    ), (
        "_PROVIDER_DEPENDENCIES must include every model-provider extra so the "
        "model selector can surface install-required recommended models"
    )


# --- Recursion limit -------------------------------------------------------


def test_recursion_limit_option_metadata() -> None:
    """The recursion-limit option exposes the env/TOML/CLI override surface."""
    opt = get_option("runtime.recursion_limit")
    assert opt is not None
    assert opt.kind is OptionKind.INT
    assert opt.env_var == _env_vars.RECURSION_LIMIT
    assert opt.toml_keys == ("runtime", "recursion_limit")
    assert opt.cli_flag == "--recursion-limit"
    assert "runtime.recursion_limit" in option_keys()


def test_resolve_recursion_limit_default() -> None:
    """With no override, the resolver returns the manifest default."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_DEFAULT,
        resolve_recursion_limit,
    )

    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_DEFAULT


def test_resolve_recursion_limit_env_wins(monkeypatch) -> None:
    """A valid env var overrides both TOML and the default."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "3000")
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 3000
    )


def test_resolve_recursion_limit_toml_when_env_unset(monkeypatch) -> None:
    """`config.toml` is used when no env var is set."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 1500
    )


@pytest.mark.parametrize("raw", ["0", "-5", "10", "999999999", "notanint"])
def test_resolve_recursion_limit_out_of_range_falls_back(monkeypatch, raw) -> None:
    """Non-positive, sub-floor, above-ceiling, or malformed values fall back."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_DEFAULT,
        resolve_recursion_limit,
    )

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, raw)
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_DEFAULT


def test_resolve_recursion_limit_invalid_env_falls_through_to_toml(
    monkeypatch,
) -> None:
    """An out-of-range env value does not mask a valid TOML override."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "10")
    assert (
        resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
        == 1500
    )


def test_resolve_recursion_limit_invalid_env_leaves_env_intact(monkeypatch) -> None:
    """Fall-through resolution must restore the rejected env var afterward."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "10")
    resolve_recursion_limit(toml_data={"runtime": {"recursion_limit": 1500}})
    assert os.environ[_env_vars.RECURSION_LIMIT] == "10"


def test_resolve_recursion_limit_accepts_floor_and_ceiling(monkeypatch) -> None:
    """The inclusive floor and ceiling are accepted verbatim."""
    from deepagents_code.config_manifest import (
        RECURSION_LIMIT_CEILING,
        RECURSION_LIMIT_FLOOR,
        resolve_recursion_limit,
    )

    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, str(RECURSION_LIMIT_FLOOR))
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_FLOOR
    monkeypatch.setenv(_env_vars.RECURSION_LIMIT, str(RECURSION_LIMIT_CEILING))
    assert resolve_recursion_limit(toml_data={}) == RECURSION_LIMIT_CEILING


def test_delegate_static_defaults_are_parseable() -> None:
    """A delegate option's static default must satisfy its own parser.

    Delegate defaults bypass the resolver's coercion (they are returned verbatim
    on the default path), so `__post_init__` cannot type-check them. This guards
    the one class of typo it would otherwise miss (e.g. `ptc` default `'saef'`).
    """
    from deepagents_code.config import _parse_interpreter_ptc

    for opt in get_config_options():
        if opt.default is None:
            continue
        if opt.kind is OptionKind.PTC_DELEGATE:
            assert _parse_interpreter_ptc(opt.default) == opt.default
