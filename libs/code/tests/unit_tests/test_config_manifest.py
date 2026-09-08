"""Drift, resolution, and behavior tests for the configuration manifest.

These guard the contract that the manifest is the single source of truth for
the scalar config surface, that its resolver matches what the runtime reads,
and that secret-flagged options are never rendered by value.
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

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
    ConfigOption,
    OptionKind,
    get_config_options,
    get_option,
    options_with_key_prefix,
)
from deepagents_code.model_config import DEFAULT_STARTUP_MODE, PROVIDER_API_KEY_ENV
from unit_tests.conftest import resolve_option_for_test

if TYPE_CHECKING:
    from pathlib import Path

# Most unit tests set `DEEPAGENTS_CODE_NO_UPDATE_CHECK=1` to avoid accidental
# PyPI/DNS work. This module checks whether update settings came from the env,
# config file, or built-in defaults; adding the env var here would hide the
# config/default cases these tests are trying to verify.
pytestmark = pytest.mark.self_managed_update_check


def _resolve_manifest_option(
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


# --- Drift / coverage -------------------------------------------------------


def test_manifest_covers_every_provider_credential() -> None:
    """Every provider in `PROVIDER_API_KEY_ENV` must have a credential option."""
    manifest_env_vars = {opt.env_var for opt in get_config_options() if opt.env_var}
    missing = set(PROVIDER_API_KEY_ENV.values()) - manifest_env_vars
    assert not missing, (
        f"Provider credential env vars without a manifest entry: {sorted(missing)}."
    )


_UI_READER_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Terminal-default inspection and the theme provider retain bespoke
        # terminal-program mapping semantics.
        "app.py:_load_terminal_default",
        "providers.py:ranked_theme_toml_value",
        # Reads `[ui]` only to repair and rewrite it; not a value reader.
        "app.py:_replace_malformed_ui",
    }
)
"""Functions permitted to read the `[ui]` table without using the manifest."""


# Readers that resolve an option but deliberately emit no ranked diagnostics.
# Each reports what it rejected through its own channel: the shared primitive
# hands the `ResolvedValue` back for the caller to emit against,
# `dcode config` prints provenance per option, the managed validator inspects a
# candidate generation not yet in force, and the sandbox, theme, and update
# readers log their own rejections in terms of the setting they own.
#
# Adding to this set should be a deliberate act with a channel named here. It
# is not a place to park a reader that reports nothing: file health is not
# value health, and a rejected value in a file that parses is invisible to
# `dcode doctor`.
_SILENT_RESOLVER_READERS = frozenset(
    {
        # The shared resolution primitives, not readers: each returns the
        # `ResolvedValue` and every caller emits against it.
        "config_manifest.py:_resolve_option_without_managed",
        "client/commands/config.py:_option_provenance",
        "configuration/service.py:resolve_managed_option",
        "integrations/sandbox_config.py:load",
        "theme.py:_load_user_themes",
        "update_check.py:_resolve_update_setting",
    }
)


# Six `app.py` display toggles plus `display.show_usage_stats` in
# `_session_stats.py`. The `app.py` wrapper forwards variables, not literals,
# so it is deliberately not counted. Exact, not a floor — see the test.
_EXPECTED_LITERAL_CALL_SITES = 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7d", "7d"), ("2w", "2w"), (" 12H ", "12h")],
)
def test_max_resume_age_parses_duration(value, expected) -> None:
    """A rolling resume age is normalized during config resolution."""
    option = get_option("threads.max_resume_age")
    assert option is not None

    assert _resolve_manifest_option(
        option,
        toml_data={"threads": {"max_resume_age": value}},
    ) == (expected, "config.toml")


@pytest.mark.parametrize(
    "value", ["0d", "7", "1.5d", "-1d", "forever", f"{'9' * 5000}d"]
)
def test_max_resume_age_rejects_invalid_duration(value, caplog) -> None:
    """An invalid rolling resume age is rejected during config resolution."""
    option = get_option("threads.max_resume_age")
    assert option is not None

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option,
            toml_data={"threads": {"max_resume_age": value}},
        ) == (None, "default")
    assert "max_resume_age" in caplog.text


def test_resume_after_normalizes_managed_cutoff() -> None:
    """Managed resume policy wins and normalizes its cutoff to UTC."""
    option = get_option("threads.resume_after")
    assert option is not None

    assert _resolve_manifest_option(
        option,
        toml_data={"threads": {"resume_after": "2030-01-01"}},
        managed_toml_data={"threads": {"resume_after": "2026-06-01T08:00:00-04:00"}},
    ) == ("2026-06-01T12:00:00+00:00", "managed config")


def test_resume_after_rejects_naive_datetime(caplog) -> None:
    """A timezone-less cutoff cannot silently become managed policy."""
    option = get_option("threads.resume_after")
    assert option is not None

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option,
            toml_data={"threads": {"resume_after": "2026-06-01T08:00:00"}},
        ) == (None, "default")
    assert "resume_after" in caplog.text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2030, 1, 1), "2030-01-01T00:00:00+00:00"),
        (
            datetime(2026, 6, 1, 8, tzinfo=timezone(timedelta(hours=-4))),
            "2026-06-01T12:00:00+00:00",
        ),
    ],
)
def test_resume_after_accepts_toml_temporal_values(value, expected) -> None:
    """Native TOML dates and aware datetimes normalize consistently."""
    option = get_option("threads.resume_after")
    assert option is not None

    assert _resolve_manifest_option(
        option,
        toml_data={},
        managed_toml_data={"threads": {"resume_after": value}},
    ) == (expected, "managed config")


def test_negative_retry_count_is_not_reported_as_effective(caplog) -> None:
    """A retry value the runtime rejects must not appear active in `config`."""
    import logging

    option = get_option("retries.max_retries")
    assert option is not None
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option, toml_data={"retries": {"max_retries": -1}}
        ) == (
            None,
            "default",
        )
    assert any(
        "[retries].max_retries" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("key", ["agents.default", "agents.recent"])
def test_blank_saved_agent_is_not_reported_as_effective(key: str, caplog) -> None:
    """Blank agent preferences fall through just as the launch loader does."""
    import logging

    option = get_option(key)
    assert option is not None
    field = key.removeprefix("agents.")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option, toml_data={"agents": {field: "   "}}
        ) == (
            None,
            "default",
        )
    assert any(f"[agents].{field}" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("key", ["agents.default", "agents.recent"])
def test_saved_agent_is_trimmed_like_the_launch_loader(key: str) -> None:
    """The manifest reports the normalized saved-agent value used at launch."""
    option = get_option(key)
    assert option is not None
    field = key.removeprefix("agents.")
    assert _resolve_manifest_option(
        option, toml_data={"agents": {field: "  coder  "}}
    ) == (
        "coder",
        "config.toml",
    )


def test_show_message_timestamps_env_overrides_config(monkeypatch) -> None:
    """The env var must outrank a persisted `/timestamps` toggle."""
    option = get_option("display.show_message_timestamps")
    assert option is not None
    monkeypatch.setenv(_env_vars.SHOW_MESSAGE_TIMESTAMPS, "1")
    value, source = _resolve_manifest_option(
        option, toml_data={"ui": {"show_message_timestamps": False}}
    )
    assert value is True
    assert source == f"env ({_env_vars.SHOW_MESSAGE_TIMESTAMPS})"


def test_is_openai_prompt_cache_key_enabled_reads_env(monkeypatch) -> None:
    """`is_openai_prompt_cache_key_enabled` honors the env override."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.delenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, raising=False)
    assert is_openai_prompt_cache_key_enabled() is True

    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "false")
    assert is_openai_prompt_cache_key_enabled() is False


def test_is_openai_prompt_cache_key_enabled_empty_env_opts_out(monkeypatch) -> None:
    """An explicitly empty env value opts out (via `empty_env_is_false`)."""
    from deepagents_code import config_manifest
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setattr(config_manifest, "load_config_toml", dict)
    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "")
    assert is_openai_prompt_cache_key_enabled() is False


def test_is_openai_prompt_cache_key_enabled_unrecognized_env_falls_through(
    monkeypatch, tmp_path: Path
) -> None:
    """An unrecognized env token is ignored, so config.toml decides."""
    from deepagents_code.config import is_openai_prompt_cache_key_enabled

    monkeypatch.setenv(_env_vars.OPENAI_PROMPT_CACHE_KEY, "banana")
    (tmp_path / "config.toml").write_text(
        "[models]\nopenai_prompt_cache_key = false\n", encoding="utf-8"
    )
    assert is_openai_prompt_cache_key_enabled() is False


def test_resolve_auto_classifier_model_warns_on_blank_value(
    monkeypatch, caplog, tmp_path: Path
) -> None:
    """A blank configured classifier is reported, not dropped in silence.

    Reverting to the main agent model is a security control quietly changing
    behavior, so it gets the same audible treatment a malformed value gets.
    """
    from deepagents_code.config import resolve_auto_classifier_model

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "   "\n', encoding="utf-8"
    )

    with caplog.at_level("WARNING", logger="deepagents_code.config"):
        assert resolve_auto_classifier_model() is None

    assert any(
        "blank" in record.getMessage() and "auto_classifier" in record.getMessage()
        for record in caplog.records
    )


def test_resolve_auto_classifier_model_reports_malformed_problem(
    monkeypatch, tmp_path: Path
) -> None:
    """A wrong-typed value must not silently revert to self-review.

    TOML coercion drops a non-string to the option default, making a
    malformed entry indistinguishable from an absent one — so the agent resumed
    grading its own actions with nothing logged at all.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    monkeypatch.delenv(_env_vars.AUTO_CLASSIFIER_MODEL, raising=False)
    (tmp_path / "config.toml").write_text(
        "[models]\nauto_classifier = 3\n", encoding="utf-8"
    )

    spec, problem = resolve_auto_classifier_model_with_problem()

    assert spec is None
    assert problem is not None
    # The value, not just the fact of rejection: this string is shown to the
    # user, and "your setting is malformed" without saying which value was
    # rejected gives them nothing to act on.
    assert "auto_classifier=3" in problem
    assert "config.toml" in problem
    assert "provider:model" in problem
    assert "main agent model" in problem


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("0", False), ("", False)])
def test_resume_term_program_env_overrides_mode_default(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: bool,
) -> None:
    """An explicit feature env value wins over mode and TOML values."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.setenv(_env_vars.DEBUG, "1")
    monkeypatch.setenv(_env_vars.EXPERIMENTAL, "1")
    monkeypatch.setenv(_env_vars.RESUME_TERM_PROGRAM, raw)

    assert _resolve_manifest_option(
        option,
        toml_data={"features": {"resume_term_program": not expected}},
    ) == (expected, f"env ({_env_vars.RESUME_TERM_PROGRAM})")


def test_resume_term_program_unrecognized_env_falls_through_to_mode_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo'd feature flag must not silently defeat debug mode."""
    option = get_option("features.resume_term_program")
    assert option is not None
    monkeypatch.delenv(_env_vars.EXPERIMENTAL, raising=False)
    monkeypatch.setenv(_env_vars.DEBUG, "1")
    monkeypatch.setenv(_env_vars.RESUME_TERM_PROGRAM, "maybe")

    assert _resolve_manifest_option(option, toml_data={}) == (True, "default")


def test_debug_log_level_validates_explicit_value(monkeypatch) -> None:
    """Explicit log levels are normalized and invalid values use the runtime default."""
    option = get_option("debug.log_level")
    assert option is not None
    monkeypatch.setenv(_env_vars.DEBUG, "true")

    monkeypatch.setenv(_env_vars.LOG_LEVEL, " warning ")
    assert _resolve_manifest_option(option, toml_data={}) == (
        "WARNING",
        f"env ({_env_vars.LOG_LEVEL})",
    )

    monkeypatch.setenv(_env_vars.LOG_LEVEL, "TRACE")
    assert _resolve_manifest_option(option, toml_data={}) == ("DEBUG", "default")


# --- Provider install helpers ----------------------------------------------


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


@pytest.mark.parametrize(
    ("key", "env_var"),
    [
        ("credentials.google_cloud_location", "GOOGLE_CLOUD_LOCATION"),
        ("credentials.google_anthropic_vertex", "GOOGLE_CLOUD_PROJECT"),
    ],
)
def test_google_cloud_configuration_is_not_secret(key: str, env_var: str) -> None:
    """Google Cloud project and location identifiers are visible configuration."""
    opt = get_option(key)
    assert opt is not None
    assert opt.env_var == env_var
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


def test_display_value_redacts_structured_table() -> None:
    """A redacted table never renders its dict, only its presence."""
    option = ConfigOption(
        key="agents.async_subagents",
        group="Agents",
        summary="",
        kind=OptionKind.STRUCTURED,
        redacted=True,
    )
    secret = {"researcher": {"headers": {"Authorization": "Bearer sk-secret"}}}
    rendered = _display_value(option, is_set=True, value=secret)
    assert rendered == "configured"
    assert "sk-secret" not in rendered
    assert _display_value(option, is_set=False, value=None) == "(unset)"


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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    is_set, source, _ = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"


@pytest.mark.usefixtures("stored_auth_dir")
def test_resolve_credential_falls_back_to_env_without_stored(monkeypatch):
    """With no stored key, resolution falls through to the env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    option = get_option("credentials.anthropic")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
        is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    _, source, _ = _resolve(option, {}, managed_toml_data={})
    assert source != "stored"


@pytest.fixture
def clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every LangSmith key name so each test declares its own state.

    Not autouse: these are module-level functions, so it would leak into
    unrelated tests. Sharing it keeps the four names from drifting apart
    test-by-test.
    """
    for var in (
        "LANGSMITH_API_KEY",
        "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("stored_auth_dir", "clear_langsmith_env")
def test_resolve_langsmith_service_prefers_stored():
    """A stored LangSmith key resolves with a stored source."""
    from deepagents_code import auth_store

    auth_store.set_stored_key("langsmith", "from-store")
    option = get_option("credentials.langsmith")
    assert option is not None
    assert option.redacted is True
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "stored"
    assert value == "from-store"


@pytest.mark.usefixtures("stored_auth_dir", "clear_langsmith_env")
def test_resolve_langsmith_falls_back_to_prefixed_langchain_api_key(monkeypatch):
    """LangSmith credential display honors the prefixed runtime fallback."""
    monkeypatch.setenv("LANGCHAIN_API_KEY", "from-fallback")
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "from-prefix")
    option = get_option("credentials.langsmith")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (DEEPAGENTS_CODE_LANGCHAIN_API_KEY)"
    assert value == "from-prefix"


@pytest.mark.usefixtures("stored_auth_dir", "clear_langsmith_env")
def test_resolve_langsmith_empty_prefixed_fallback_shadows_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty fallback override keeps config display aligned with runtime."""
    from deepagents_code.model_config import (
        ProviderAuthState,
        get_service_auth_status,
    )

    monkeypatch.setenv("LANGCHAIN_API_KEY", "from-fallback")
    option = get_option("credentials.langsmith")
    assert option is not None

    monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
    assert get_service_auth_status("langsmith").state is ProviderAuthState.MISSING
    is_set, _, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is False
    assert value is None

    monkeypatch.delenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY")
    assert _resolve(option, {}, managed_toml_data={}) == (
        True,
        "env (LANGCHAIN_API_KEY)",
        "from-fallback",
    )


@pytest.mark.usefixtures("stored_auth_dir", "clear_langsmith_env")
def test_stored_key_with_empty_prefixed_primary_agrees_across_surfaces(monkeypatch):
    """`config` and `auth status` agree once a fallback can outrank the store."""
    from deepagents_code import auth_store
    from deepagents_code.model_config import (
        ProviderAuthSource,
        ProviderAuthState,
        get_service_auth_status,
    )

    auth_store.set_stored_key("langsmith", "from-store")
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "from-fallback")

    option = get_option("credentials.langsmith")
    assert option is not None
    assert _resolve(option, {}, managed_toml_data={}) == (
        True,
        "env (LANGCHAIN_API_KEY)",
        "from-fallback",
    )

    status = get_service_auth_status("langsmith")
    assert status.state is ProviderAuthState.CONFIGURED
    assert status.source is ProviderAuthSource.ENV
    assert status.env_var == "LANGCHAIN_API_KEY"


@pytest.mark.usefixtures("stored_auth_dir", "clear_langsmith_env")
def test_resolve_langsmith_primary_env_wins_over_fallback(monkeypatch):
    """The primary LangSmith env var retains precedence over its fallback."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "from-primary")
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "from-prefix")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "from-fallback")
    option = get_option("credentials.langsmith")
    assert option is not None
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
    assert is_set is True
    assert source == "env (LANGSMITH_API_KEY)"
    assert value == "from-primary"


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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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


def test_run_get_pairs_values_and_health_from_one_managed_snapshot(
    monkeypatch, capsys
) -> None:
    """A config invocation refreshes managed provider state exactly once."""
    from deepagents_code.configuration import service
    from deepagents_code.configuration.types import (
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    snapshot = TomlSnapshot(
        {},
        ProviderStatus("managed config", None, ProviderHealth.MISSING),
    )
    calls: list[bool] = []

    def get_snapshot(*, refresh: bool = False, path: object = None) -> TomlSnapshot:
        assert path is None
        calls.append(refresh)
        return snapshot

    monkeypatch.setattr(service, "get_managed_snapshot", get_snapshot)

    assert _run_get("display.show_header", "json") == 0
    assert calls == [True]
    capsys.readouterr()


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
    is_set, source, value = _resolve(option, {}, managed_toml_data={})
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


# --- Single-source defaults -------------------------------------------------


# --- Resolution -------------------------------------------------------------


def test_resolve_prefers_prefixed_env(monkeypatch) -> None:
    """A `DEEPAGENTS_CODE_`-prefixed env var wins over the canonical name."""
    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "prefixed")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == "env (DEEPAGENTS_CODE_OPENAI_API_KEY)"
    assert value == "prefixed"


def test_resolve_empty_env_is_unset_matching_resolve_env_var(monkeypatch) -> None:
    """An empty (prefixed) env var is unset for `config`, as the app sees it.

    The runtime `resolve_env_var` returns `None` for an empty prefixed var (and
    a prefixed empty suppresses the canonical). The resolver must agree, or
    `config` would report a credential as "set" that the app treats as
    unset — the exact drift this feature exists to prevent.
    """
    from deepagents_code.model_config import resolve_env_var

    opt = get_option("credentials.openai")
    assert opt is not None
    monkeypatch.setenv("OPENAI_API_KEY", "canonical")
    monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "")

    value, source = _resolve_manifest_option(opt, toml_data={})
    assert resolve_env_var("OPENAI_API_KEY") is None
    assert source == "default"
    assert value is None


def test_langsmith_project_prefers_prefixed_env(monkeypatch) -> None:
    """The prefixed project env var wins over a bare `LANGSMITH_PROJECT`."""
    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "prefixed")
    monkeypatch.setenv("LANGSMITH_PROJECT", "bare")
    value, source = _resolve_manifest_option(opt, toml_data={})
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
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_default_when_unset(monkeypatch) -> None:
    """With no project env var set, the default project name is rendered."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert _resolve_manifest_option(opt, toml_data={}) == (
        LANGSMITH_PROJECT_DEFAULT,
        "default",
    )


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
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert (value, source) == ("bare", "env (LANGSMITH_PROJECT)")


def test_langsmith_project_empty_bare_is_default(monkeypatch) -> None:
    """An empty bare `LANGSMITH_PROJECT` is unset, so the default applies."""
    from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

    opt = get_option("tracing.langsmith_project")
    assert opt is not None
    monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGSMITH_PROJECT", "")
    assert _resolve_manifest_option(opt, toml_data={}) == (
        LANGSMITH_PROJECT_DEFAULT,
        "default",
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


def test_run_get_json_redacts_credential_bearing_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`config get --json` reports a credential-bearing table's presence only.

    `[async_subagents]` headers, `[models.providers]`, and
    `[sandboxes.providers]` can hold tokens, so their rows carry `redacted`
    and a `None` value instead of the raw table.
    """
    from deepagents_code import model_config

    secret = "Bearer sk-secret"
    config = tmp_path / "config.toml"
    config.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'graph_id = "agent"\n'
        "headers = { Authorization = '" + secret + "' }\n"
        "[models.providers.acme]\n"
        'class_path = "acme.Chat:AcmeChat"\n'
        'api_key = "sk-secret"\n'
        "[sandboxes.providers.acme]\n"
        'class_path = "acme.Sandbox:AcmeSandbox"\n'
        "params = { token = 'sk-secret' }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)

    for key, source in (
        ("agents.async_subagents", "config.toml"),
        ("models.providers", "config.toml"),
        ("sandboxes.providers", "config.toml"),
    ):
        data = _get_json_object(key, capsys)
        assert data["redacted"] is True
        assert data["set"] is True
        assert data["source"] == source
        assert data["value"] is None
    assert "sk-secret" not in capsys.readouterr().out


def test_resolve_cursor_style_from_env(monkeypatch, caplog) -> None:
    """A valid env value wins; an invalid value falls through to config.toml."""
    import logging

    opt = get_option("display.cursor_style")
    assert opt is not None
    toml_data = {"ui": {"cursor_style": "underline"}}

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "block")
    assert _resolve_manifest_option(opt, toml_data=toml_data) == (
        "block",
        f"env ({_env_vars.CURSOR_STYLE})",
    )

    monkeypatch.setenv(_env_vars.CURSOR_STYLE, "bar")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(opt, toml_data=toml_data) == (
            "underline",
            "config.toml",
        )
    assert any(
        _env_vars.CURSOR_STYLE in record.getMessage() for record in caplog.records
    )


def test_auto_update_resolves_persisted_config() -> None:
    """`set_auto_update()` writes the TOML path surfaced by the manifest."""
    opt = get_option("update.auto_update")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"update": {"auto_update": True}}
    ) == (
        True,
        "config.toml",
    )


def test_no_update_check_resolves_inverted_persisted_check() -> None:
    """`[update].check = false` means the effective no-check flag is enabled."""
    opt = get_option("update.no_update_check")
    assert opt is not None
    assert _resolve_manifest_option(opt, toml_data={"update": {"check": False}}) == (
        True,
        "config.toml",
    )
    assert _resolve_manifest_option(opt, toml_data={"update": {"check": True}}) == (
        False,
        "config.toml",
    )


def test_prices_auto_update_persists_in_toml(monkeypatch) -> None:
    """The dotted key resolves from `config.toml`, not env vars only."""
    opt = get_option("update.prices_auto_update")
    assert opt is not None
    monkeypatch.delenv(_env_vars.PRICES_AUTO_UPDATE, raising=False)
    assert opt.toml_keys == ("update", "prices_auto_update")
    assert _resolve_manifest_option(
        opt, toml_data={"update": {"prices_auto_update": False}}
    ) == (
        False,
        "config.toml",
    )


def test_resolve_ptc_delegates_to_parser() -> None:
    """The PTC kind routes through the dedicated allowlist parser."""
    opt = get_option("interpreter.ptc")
    assert opt is not None
    assert _resolve_manifest_option(
        opt, toml_data={"interpreter": {"ptc": "safe"}}
    ) == (
        "safe",
        "config.toml",
    )
    # Invalid PTC value is rejected by the parser and falls back to default.
    value, source = _resolve_manifest_option(
        opt, toml_data={"interpreter": {"ptc": "bogus"}}
    )
    assert (value, source) == (opt.default, "default")


# --- Misc -------------------------------------------------------------------


def test_missing_key_example_is_a_real_option() -> None:
    """The hint's example key must stay a resolvable manifest key."""
    from deepagents_code.client.commands.config import _GET_KEY_EXAMPLE

    assert get_option(_GET_KEY_EXAMPLE) is not None


# --- ConfigOption validation ------------------------------------------------


# --- Coercion matrix --------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false"])
def test_debug_dep_floor_uses_boolean_semantics(monkeypatch, value: str) -> None:
    """Config inspection agrees with the runtime for explicitly falsy values."""
    opt = get_option("debug.dep_floor")
    assert opt is not None
    assert opt.kind is OptionKind.BOOL
    monkeypatch.setenv(_env_vars.DEBUG_DEP_FLOOR, value)
    assert _resolve_manifest_option(opt, toml_data={})[0] is False


def test_structured_fallback_preserves_invalid_tier(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A defaultless table fallback retains the rejected provider metadata."""
    from deepagents_code.configuration.resolver import (
        USER_RANK,
        resolver_from_snapshots,
    )
    from deepagents_code.configuration.types import (
        Invalid,
        ProviderHealth,
        ProviderStatus,
        TomlSnapshot,
    )

    option = get_option("display.terminal_themes")
    assert option is not None
    toml_data = {"ui": "dark"}

    resolved = resolver_from_snapshots(
        managed=TomlSnapshot(
            {}, ProviderStatus("managed config", None, ProviderHealth.OK)
        ),
        user=TomlSnapshot(
            toml_data, ProviderStatus("config.toml", None, ProviderHealth.OK)
        ),
    ).get(option)
    assert resolved.value is None
    assert isinstance(resolved.tier_health[USER_RANK], Invalid)

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(
            option,
            toml_data=toml_data,
            managed_toml_data={},
        ) == (None, "default")
    assert any("expected a table" in record.message for record in caplog.records)


# --- load_config_toml -------------------------------------------------------


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


# --- Display rendering ------------------------------------------------------


def test_display_value_unset_renders_placeholder() -> None:
    """A non-secret option with no value renders the unset placeholder."""
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)
    assert _display_value(opt, is_set=False, value=None) == "(unset)"


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


@pytest.mark.parametrize(
    "argv",
    [
        ["dcode", "config", "get", "credentials", "--verbose", "--json"],
        ["dcode", "config", "--verbose", "--json", "get", "credentials"],
    ],
)
def test_config_get_parser_accepts_verbose_on_either_side(monkeypatch, argv) -> None:
    """`--verbose`/`--json` reach `config get` before or after the subcommand.

    The `get` subparser suppresses its own defaults, so a flag set on the parent
    `config` parser survives the subparser namespace merge.
    """
    import sys

    from deepagents_code.main import parse_args

    monkeypatch.setattr(sys, "argv", argv)
    ns = parse_args()
    assert ns.config_command == "get"
    assert ns.key == "credentials"
    assert ns.verbose is True
    assert ns.output_format == "json"


# --- `config get` sections --------------------------------------------------


def _get_args(
    key: str, output_format: str = "text", *, verbose: bool = False
) -> argparse.Namespace:
    """Build a parsed `config get <key>` namespace."""
    return argparse.Namespace(
        config_command="get", key=key, output_format=output_format, verbose=verbose
    )


def _get_json_data(
    key: str, capsys: pytest.CaptureFixture[str], *, verbose: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run `config get <key> --json` and return the decoded `data` payload."""
    import json

    assert run_config_command(_get_args(key, "json", verbose=verbose)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config get"
    return payload["data"]


def _get_json_object(
    key: str, capsys: pytest.CaptureFixture[str], *, verbose: bool = False
) -> dict[str, Any]:
    """Return the single-option `config get --json` payload for `key`."""
    data = _get_json_data(key, capsys, verbose=verbose)
    assert isinstance(data, dict)
    return data


def test_options_with_key_prefix_matches_whole_segments_only() -> None:
    """A key prefix matches on the dot boundary, never a truncated guess."""
    matched = options_with_key_prefix("credentials")
    assert len(matched) > 1
    assert all(opt.key.startswith("credentials.") for opt in matched)
    assert options_with_key_prefix("credential") == ()
    assert options_with_key_prefix("") == ()
    # No `tools.*` key exists, so the `Tools` heading has no prefix section.
    assert options_with_key_prefix("tools") == ()


def test_run_get_recursion_limit_reports_malformed_upstream_env(
    monkeypatch, capsys
) -> None:
    """A non-numeric upstream value surfaces as data, not a traceback.

    The user sets the malformed variable and then runs `dcode config` to find
    it, so the inspection path must not be the thing that crashes.
    """
    monkeypatch.setenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "not-an-int")

    data = _get_json_object("runtime.recursion_limit", capsys)
    assert data["value"] == "not-an-int"
    assert data["source"] == "env (LANGGRAPH_DEFAULT_RECURSION_LIMIT); invalid"
    assert data["set"] is True

    assert run_config_command(_get_args("runtime.recursion_limit")) == 0
    rendered = " ".join(capsys.readouterr().out.split())
    assert rendered == (
        "runtime.recursion_limit = not-an-int "
        "(env (LANGGRAPH_DEFAULT_RECURSION_LIMIT); invalid)"
    )


def test_run_get_section_json_flags_unreadable_store(stored_auth_dir, capsys) -> None:
    """A corrupt store is reported in-band on every credential section row.

    Without this a corrupt `auth.json` is indistinguishable from one holding no
    keys — the silent failure `store_error` exists to prevent.
    """
    import json

    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    assert run_config_command(_get_args("credentials", "json")) == 0
    rows = json.loads(capsys.readouterr().out)["data"]
    assert rows
    assert all("store_error" in row for row in rows)
    # Redaction holds even when the store could not be read.
    assert all(row["value"] is None for row in rows if row["redacted"])


def test_run_get_section_text_warns_on_unreadable_store(
    stored_auth_dir, capsys
) -> None:
    """Both section text renderers surface the unreadable-store banner."""
    stored_auth_dir.mkdir(parents=True, exist_ok=True)
    (stored_auth_dir / "auth.json").write_text("{ not json", encoding="utf-8")
    for verbose in (False, True):
        assert run_config_command(_get_args("credentials", verbose=verbose)) == 0
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "unreadable" in out


@pytest.mark.usefixtures("stored_auth_dir")
def test_run_get_section_skips_store_when_no_credentials(monkeypatch) -> None:
    """A credential-free section never reads the store, and reads it once if not.

    Pins the documented skip in `_run_get_section`: always loading would emit a
    spurious store warning for sections that cannot consult it, and the JSON
    would look identical either way.

    Bootstrap is marked done first: `_ensure_bootstrap` applies stored
    LangSmith auth (two store reads) only on the first call in the process, so
    without the pin this test's counts depend on which tests ran earlier in
    the same worker.
    """
    from deepagents_code import auth_store, config as config_mod

    monkeypatch.setattr(config_mod._bootstrap_state, "done", True)

    calls = 0
    real_load = auth_store.load_credentials

    def _counting_load() -> dict:
        nonlocal calls
        calls += 1
        return real_load()

    monkeypatch.setattr(auth_store, "load_credentials", _counting_load)

    assert run_config_command(_get_args("display", "json")) == 0
    assert calls == 0

    assert run_config_command(_get_args("credentials", "json")) == 0
    assert calls == 1


# --- BOOL env coercion ------------------------------------------------------


# --- FLOAT / shell-list env coercion ---------------------------------------


def test_resolve_shell_list_env_happy_and_invalid(monkeypatch, caplog) -> None:
    """The shell-list env delegate parses a valid list and rejects bad input."""
    import logging

    opt = get_option("shell.allow_list")
    assert opt is not None
    monkeypatch.setenv(opt.env_var, "git status,ls")
    value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == f"env ({opt.env_var})"
    assert isinstance(value, list)
    assert "ls" in value

    # `'all'` cannot be combined with other commands; the parser raises and the
    # resolver logs + falls back rather than crashing.
    monkeypatch.setenv(opt.env_var, "all,ls")
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(opt, toml_data={})
    assert source == "default"
    assert any("Ignoring invalid" in r.getMessage() for r in caplog.records)


def test_environment_coercion_delegate_returns_invalid_not_raw() -> None:
    """A delegate kind reaching env coercion returns `Invalid`, never raw.

    PTC/STRUCTURED options declare no env var, so this branch is unreachable in
    the live manifest. The guard exists so that if one ever gains an env var,
    an uncoerced raw string cannot leak into a typed credentials field.
    """
    from deepagents_code.configuration.providers import coerce_environment_value
    from deepagents_code.configuration.types import Invalid

    opt = get_option("interpreter.ptc")
    assert opt is not None
    result = coerce_environment_value(opt, "safe", "DEEPAGENTS_CODE_FAKE")
    assert isinstance(result, Invalid)
    assert "not env-backed" in result.reason


# --- TOML coercion (success + mismatch) ------------------------------------


def test_resolve_startup_mode_from_toml(caplog) -> None:
    """`startup.mode` resolves only valid runtime modes from config.toml."""
    import logging

    opt = get_option("startup.mode")
    assert opt is not None
    for mode in ("auto", "yolo"):
        assert _resolve_manifest_option(opt, toml_data={"startup": {"mode": mode}}) == (
            mode,
            "config.toml",
        )
    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            opt, toml_data={"startup": {"mode": "dangerously-auto"}}
        )
    assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
    assert any(
        "[startup].mode='dangerously-auto'" in r.getMessage() for r in caplog.records
    )

    for raw in (["manual"], {"name": "manual"}):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
            value, source = _resolve_manifest_option(
                opt, toml_data={"startup": {"mode": raw}}
            )
        assert (value, source) == (DEFAULT_STARTUP_MODE, "default")
        assert any("[startup].mode" in r.getMessage() for r in caplog.records)

    assert _resolve_manifest_option(opt, toml_data={}) == (
        DEFAULT_STARTUP_MODE,
        "default",
    )


@pytest.mark.parametrize(
    ("toml_data", "managed_toml_data", "expected"),
    [
        # Only `recent` set: a bare launch runs Auto, so the display must agree.
        ({"startup": {"recent": "auto"}}, {}, ("auto", "config.toml")),
        # An explicit mode outranks `recent`.
        (
            {"startup": {"mode": "manual", "recent": "auto"}},
            {},
            ("manual", "config.toml"),
        ),
        # Nothing configured: the typed default stands.
        ({}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # An unsafe `recent` fails closed rather than crediting config.toml.
        ({"startup": {"recent": "yolo"}}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # A non-scalar `recent` cannot reach the membership test.
        ({"startup": {"recent": ["auto"]}}, {}, (DEFAULT_STARTUP_MODE, "default")),
        # An invalid explicit mode is fail-closed: `load_startup_mode` returns
        # Manual without consulting `recent`, so the display must too.
        (
            {"startup": {"mode": "hands-off", "recent": "auto"}},
            {},
            (DEFAULT_STARTUP_MODE, "default"),
        ),
        # Managed `recent` participates in the same precedence as runtime loading.
        ({}, {"startup": {"recent": "auto"}}, ("auto", "managed config")),
        (
            {"startup": {"recent": "auto"}},
            {"startup": {"recent": "manual"}},
            ("manual", "managed config"),
        ),
    ],
    ids=[
        "recent-only",
        "explicit-outranks-recent",
        "nothing-configured",
        "unsafe-recent",
        "non-scalar-recent",
        "invalid-explicit-mode",
        "managed-recent",
        "managed-recent-outranks-user",
    ],
)
def test_resolve_startup_mode_with_source_reports_recent_fallback(
    monkeypatch: pytest.MonkeyPatch,
    toml_data: dict,
    managed_toml_data: dict,
    expected: tuple[str, str],
) -> None:
    """`startup.mode` display reflects the `[startup].recent` restore."""
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)

    assert (
        resolve_startup_mode_with_source(
            toml_data=toml_data,
            managed_toml_data=managed_toml_data,
        )
        == expected
    )


@pytest.mark.parametrize(
    "config_text",
    [
        "",
        "[startup]\n",
        "[startup]\nrecent = 'auto'\n",
        "[startup]\nrecent = 'manual'\n",
        "[startup]\nrecent = 'yolo'\n",
        "[startup]\nrecent = ['auto']\n",
        # Whitespace and blanks: the display must not accept a value the
        # loader's exact match rejects.
        "[startup]\nrecent = ' auto '\n",
        "[startup]\nrecent = 'AUTO'\n",
        "[startup]\nrecent = ''\n",
        "[startup]\nmode = 'auto'\n",
        "[startup]\nmode = 'yolo'\nrecent = 'manual'\n",
        "[startup]\nmode = 'hands-off'\nrecent = 'auto'\n",
        "[startup]\nmode = ['auto']\nrecent = 'auto'\n",
        "startup = 'nonsense'\n",
    ],
)
def test_resolve_startup_mode_with_source_agrees_with_loader(
    tmp_path, monkeypatch: pytest.MonkeyPatch, config_text: str
) -> None:
    """The display resolver and the runtime loader must never disagree.

    Both consult `[startup].mode`, `[startup].recent`, and the Auto notice, in
    two separate implementations held together only by a docstring. Per-case
    assertions cannot catch drift between them; this can.
    """
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source
    from deepagents_code.model_config import load_startup_mode

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)
    config = tmp_path / "config.toml"
    config.write_text(config_text)
    with config.open("rb") as file:
        toml_data = tomllib.load(file)

    displayed, _ = resolve_startup_mode_with_source(
        toml_data=toml_data,
        managed_toml_data={},
    )
    assert displayed == load_startup_mode(config)


def test_resolve_startup_mode_with_source_gates_recent_auto_on_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The displayed fallback matches a launch blocked by a stale notice."""
    from deepagents_code import approval_mode
    from deepagents_code.config_manifest import resolve_startup_mode_with_source

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: False)

    assert resolve_startup_mode_with_source(
        toml_data={},
        managed_toml_data={"startup": {"recent": "auto"}},
    ) == (DEFAULT_STARTUP_MODE, "default")


# --- Theme resolution warnings ----------------------------------------------


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
    from deepagents_code.configuration import service

    patched = {
        **model_config.PROVIDER_API_KEY_ENV,
        "synthetic_xyz": "SYNTHETIC_XYZ_API_KEY",
    }
    monkeypatch.setattr(model_config, "PROVIDER_API_KEY_ENV", patched)
    config_manifest.get_config_options.cache_clear()
    service._managed_table_paths.cache_clear()
    config_manifest._options_by_key.cache_clear()
    config_manifest._options_by_toml_path.cache_clear()
    try:
        opt = config_manifest.get_option("credentials.synthetic_xyz")
        assert opt is not None
        assert opt.env_var == "SYNTHETIC_XYZ_API_KEY"
        # A *_API_KEY env var is treated as secret material.
        assert opt.redacted is True
    finally:
        # Restore the cache so later tests rebuild against the real registry.
        config_manifest.get_config_options.cache_clear()
        service._managed_table_paths.cache_clear()
        config_manifest._options_by_key.cache_clear()
        config_manifest._options_by_toml_path.cache_clear()


# --- Auto classifier timeout -----------------------------------------------


@pytest.mark.parametrize(
    ("toml_data", "expected"),
    [
        # A file with no `mode` key still reports the mode the launch will use.
        ({"startup": {"recent": "auto"}}, (True, "config.toml", "auto")),
        ({}, (False, "default", DEFAULT_STARTUP_MODE)),
        (
            {"startup": {"mode": "hands-off", "recent": "auto"}},
            (False, "default", DEFAULT_STARTUP_MODE),
        ),
    ],
    ids=["recent-only", "nothing-configured", "invalid-explicit-mode"],
)
def test_config_resolve_reports_effective_startup_mode(
    monkeypatch: pytest.MonkeyPatch,
    toml_data: dict,
    expected: tuple[bool, str, str],
) -> None:
    """`config get startup.mode` must route through the recent-aware resolver."""
    from deepagents_code import approval_mode

    monkeypatch.setattr(approval_mode, "has_auto_mode_notice", lambda: True)
    option = get_option("startup.mode")
    assert option is not None

    assert _resolve(option, toml_data, managed_toml_data={}) == expected


# --- Recursion limit -------------------------------------------------------


def test_resolve_recursion_limit_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override, the resolver leaves the limit to LangGraph."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.delenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", raising=False)
    assert resolve_recursion_limit(toml_data={}) is None


def test_resolve_recursion_limit_inherits_langgraph_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upstream environment default becomes the effective agent limit."""
    from deepagents_code.config_manifest import resolve_recursion_limit

    monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)
    monkeypatch.setenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "12000")
    assert resolve_recursion_limit(toml_data={}) == 12_000


def test_whitespace_env_is_ignored_with_a_warning(monkeypatch, caplog) -> None:
    """A whitespace-only env value falls through and says so.

    Empty is a normal "unset" idiom, but whitespace-only is nearly always an
    accident (`export X="$UNSET "`). Discarding it without a word was the only
    unlogged rejection path in resolution, so a user could lose an
    override with no evidence anywhere.
    """
    option = get_option("models.auto_classifier")
    assert option is not None
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "   ")

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(
            option, toml_data={"models": {"auto_classifier": "openai:gpt-5"}}
        )

    assert (value, source) == ("openai:gpt-5", "config.toml")
    assert any("whitespace-only" in r.getMessage() for r in caplog.records)


def test_whitespace_env_opts_out_when_empty_means_false(monkeypatch) -> None:
    """`empty_env_is_false` treats whitespace like empty, outranking config.toml.

    The asymmetry with the test above is the resolver's contract, so pin both
    sides: a blank value is an opt-out only where the option declares one.
    """
    option = get_option("display.cursor_blink")
    assert option is not None
    assert option.empty_env_is_false is True
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "  \t ")

    assert _resolve_manifest_option(
        option, toml_data={"ui": {"cursor_blink": True}}
    ) == (
        False,
        f"env ({option.env_var})",
    )


def test_invalid_primary_env_warns_before_valid_fallback(monkeypatch, caplog) -> None:
    """Alias fall-through retains diagnostics from earlier env candidates."""
    option = get_option("tracing.langsmith_project")
    assert option is not None
    assert option.env_var is not None
    fallback = option.fallback_env_vars[0]
    monkeypatch.setenv(option.env_var, "  ")
    monkeypatch.setenv(fallback, "fallback-project")

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        value, source = _resolve_manifest_option(option, toml_data={})

    assert (value, source) == ("fallback-project", f"env ({fallback})")
    assert [
        record.getMessage()
        for record in caplog.records
        if "whitespace-only" in record.getMessage()
    ] == [f"Ignoring {option.env_var}='  ' (whitespace-only; treated as unset)"]


def test_non_table_toml_section_is_reported_once(caplog) -> None:
    """A scalar shadowing a whole table is logged, not silently defaulted.

    `ui = "dark"` defaults every `[ui]` option at once; the pre-manifest loaders
    each warned about it, and dropping that left the user's edited value absent
    from the output with no explanation. Logged once per path per process
    because `config` resolves the whole manifest in one pass.
    """
    from deepagents_code import config_manifest

    config_manifest._warned_non_table_paths.clear()
    scrollbar = get_option("display.show_scrollbar")
    blink = get_option("display.cursor_blink")
    assert scrollbar is not None
    assert blink is not None

    with caplog.at_level(logging.WARNING, logger="deepagents_code.config_manifest"):
        assert _resolve_manifest_option(scrollbar, toml_data={"ui": "dark"}) == (
            False,
            "default",
        )
        assert _resolve_manifest_option(blink, toml_data={"ui": "dark"}) == (
            True,
            "default",
        )

    warnings = [r for r in caplog.records if "expected a table" in r.getMessage()]
    assert len(warnings) == 1
    assert "[ui]" in warnings[0].getMessage()
    config_manifest._warned_non_table_paths.clear()


def test_verbose_provenance_distinguishes_quoted_dotted_keys() -> None:
    """Display labels retain the resolver's tuple-path distinction."""
    from deepagents_code.client.commands.config import _option_provenance

    option = get_option("display.themes")
    assert option is not None

    assert _option_provenance(
        option,
        source="managed config + config.toml",
        toml_data={"themes": {"a": {"b": "user"}, "sibling": 1}},
        managed_toml_data={"themes": {"a.b": "managed"}},
    ) == {
        '"a.b"': "managed config",
        "a.b": "config.toml",
        "sibling": "config.toml",
    }


def test_blank_env_auto_classifier_reports_a_problem(
    monkeypatch, tmp_path: Path
) -> None:
    """A blank env classifier must be described, not just logged.

    The blank value reverts authorization review to the main agent model — the
    agent grading its own actions. A `logger.warning` lands in the debug log,
    which is not a surface the user reads, so the caller needs the description.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5"\n', encoding="utf-8"
    )
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "   ")

    spec, problem = resolve_auto_classifier_model_with_problem()

    assert spec is None
    assert problem is not None
    assert _env_vars.AUTO_CLASSIFIER_MODEL in problem
    # The message must name the value it overrode; without it the user checks
    # config.toml, still sees their setting, and learns nothing.
    assert "openai:gpt-5" in problem


@pytest.mark.parametrize("blank", ["", "   "])
def test_config_surface_agrees_with_runtime_on_blank_env_classifier(
    blank: str, monkeypatch, tmp_path: Path
) -> None:
    """`config` must not credit a classifier the runtime refuses to use.

    A blank env var vetoes `config.toml` for this option only, so resolving it
    with the generic scalar path would report the config.toml model while the
    runtime inherits the main agent model.
    """
    from deepagents_code.config import resolve_auto_classifier_model_with_problem

    toml_data = {"models": {"auto_classifier": "openai:gpt-5"}}
    (tmp_path / "config.toml").write_text(
        '[models]\nauto_classifier = "openai:gpt-5"\n', encoding="utf-8"
    )
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, blank)

    option = get_option("models.auto_classifier")
    assert option is not None
    runtime_spec, _ = resolve_auto_classifier_model_with_problem()
    is_set, source, displayed = _resolve(
        option, toml_data=toml_data, managed_toml_data={}
    )

    assert runtime_spec is None
    assert displayed == runtime_spec
    assert source == f"env ({_env_vars.AUTO_CLASSIFIER_MODEL})"
    assert is_set is True
    assert _display_value(option, is_set=is_set, value=displayed) == "(unset)"


def test_usable_env_classifier_is_still_reported(monkeypatch) -> None:
    """The veto must not swallow a real env value."""
    from deepagents_code import config_manifest

    toml_data = {"models": {"auto_classifier": "openai:gpt-5"}}
    monkeypatch.setattr(config_manifest, "load_config_toml", lambda: toml_data)
    monkeypatch.setenv(_env_vars.AUTO_CLASSIFIER_MODEL, "anthropic:claude-haiku-4-5")

    option = get_option("models.auto_classifier")
    assert option is not None
    is_set, source, value = _resolve(option, toml_data=toml_data, managed_toml_data={})

    assert (is_set, value) == (True, "anthropic:claude-haiku-4-5")
    assert source == f"env ({_env_vars.AUTO_CLASSIFIER_MODEL})"


def test_config_text_output_redacts_credential_bearing_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The human-readable surfaces must not print a credential-bearing table.

    The JSON path redacts through `_config_json_row`; the table, `--verbose`,
    and `config get` text paths go through `_display_value` instead, so they
    need their own guard against a renderer that forgets.
    """
    from deepagents_code import model_config

    config = tmp_path / "config.toml"
    config.write_text(
        "[async_subagents.researcher]\n"
        'description = "Research agent"\n'
        'graph_id = "agent"\n'
        "headers = { Authorization = 'Bearer sk-secret' }\n"
        "[models.providers.acme]\n"
        'class_path = "acme.Chat:AcmeChat"\n'
        'api_key = "sk-secret"\n'
        "[sandboxes.providers.acme]\n"
        'class_path = "acme.Sandbox:AcmeSandbox"\n'
        "params = { token = 'sk-secret' }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config)

    for args in (
        argparse.Namespace(config_command=None, output_format="text", verbose=False),
        argparse.Namespace(config_command=None, output_format="text", verbose=True),
        argparse.Namespace(
            config_command="get",
            key="agents.async_subagents",
            output_format="text",
            verbose=False,
        ),
    ):
        assert run_config_command(args) == 0
        out = capsys.readouterr().out
        assert "sk-secret" not in out
        assert "Authorization" not in out
    assert "configured" in out


def test_empty_redacted_table_reads_as_unset() -> None:
    """A present-but-empty table has nothing configured, so say so."""
    option = get_option("agents.async_subagents")
    assert option is not None
    assert option.redacted is True
    assert _display_value(option, is_set=True, value={}) == "(unset)"
    assert _display_value(option, is_set=True, value={"a": {}}) == "configured"


def test_display_value_ascii_truncation_respects_limit(monkeypatch) -> None:
    from deepagents_code.config import ASCII_GLYPHS

    monkeypatch.setattr("deepagents_code.config.get_glyphs", lambda: ASCII_GLYPHS)
    opt = ConfigOption(key="x", group="g", summary="s", kind=OptionKind.STR)

    rendered = _display_value(opt, is_set=True, value="a" * 100)

    assert len(rendered) == 60
    assert rendered.endswith(ASCII_GLYPHS.ellipsis)
    assert rendered.isascii()


def test_legacy_debug_file_remains_discoverable(monkeypatch) -> None:
    option = get_option("debug.file")
    assert option is not None
    monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG_FILE", "/tmp/legacy.log")
    assert _resolve_manifest_option(option, toml_data={}) == (
        "/tmp/legacy.log",
        "env (DEEPAGENTS_CODE_DEBUG_FILE)",
    )
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_FILE")
    assert _resolve_manifest_option(
        option, toml_data={"debug": {"file": "/tmp/configured.log"}}
    ) == ("/tmp/configured.log", "config.toml")
