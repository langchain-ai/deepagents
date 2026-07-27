"""Tests for the `dcode config set` write command.

Covers the generic writer (`config_manifest.parse_config_write_value`,
`model_config.set_config_toml_value`) and the `config set` CLI handler:
creating a missing file, setting/updating values, preserving unrelated tables,
scalar parsing, invalid input leaving the file untouched, unknown/non-writable/
credential rejection, higher-precedence override reporting, JSON output, and
argparse wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib

import pytest

from deepagents_code import _env_vars
from deepagents_code.client.commands.config import run_config_command
from deepagents_code.config_manifest import (
    ConfigWriteError,
    get_option,
    parse_config_write_value,
    writable_rejection,
)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    """Redirect `config.toml` reads and writes to a temp file."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr(
        "deepagents_code.model_config.DEFAULT_CONFIG_PATH", path, raising=True
    )
    # The default startup.mode option has no env var, but neutralize any that
    # could shadow override-free assertions.
    for name in (
        "LANGSMITH_REDACT",
        f"{_env_vars.SERVER_ENV_PREFIX}LANGSMITH_REDACT",
    ):
        monkeypatch.delenv(name, raising=False)
    return path


def _set_args(
    key: str | None, value: str | None, output_format: str = "text"
) -> argparse.Namespace:
    return argparse.Namespace(
        command="config",
        config_command="set",
        key=key,
        value=value,
        output_format=output_format,
    )


# --- Writer / parser units --------------------------------------------------


def test_parse_startup_mode_accepts_yolo() -> None:
    """`startup.mode` uses the runtime allowlist; `yolo` is accepted."""
    option = get_option("startup.mode")
    assert option is not None
    parsed = parse_config_write_value(option, "yolo")
    assert parsed.toml_value == "yolo"
    assert parsed.effective_value == "yolo"


@pytest.mark.parametrize("bad", ["dangerously-auto", "MANUAL", "on", ""])
def test_parse_startup_mode_rejects_invalid(bad: str) -> None:
    """Invalid startup modes (including the legacy spelling) are rejected."""
    option = get_option("startup.mode")
    assert option is not None
    with pytest.raises(ConfigWriteError):
        parse_config_write_value(option, bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("yes", True), ("0", False), ("on", True)],
)
def test_parse_bool_tokens(raw: str, expected: bool) -> None:
    """Boolean options accept the standard truthy/falsy tokens."""
    option = get_option("tracing.langsmith_redact")
    assert option is not None
    parsed = parse_config_write_value(option, raw)
    assert parsed.toml_value is expected


def test_parse_int_and_invalid() -> None:
    """Integer options parse digits and reject non-numeric input."""
    option = get_option("runtime.recursion_limit")
    assert option is not None
    assert parse_config_write_value(option, "5000").toml_value == 5000
    with pytest.raises(ConfigWriteError):
        parse_config_write_value(option, "not-a-number")


def test_parse_invert_toml_bool_stores_negation() -> None:
    """An `invert_toml_bool` option stores the negated literal."""
    option = get_option("update.no_update_check")
    assert option is not None
    assert option.invert_toml_bool is True
    parsed = parse_config_write_value(option, "true")
    # Logical value (no_update_check) is True; the stored `[update].check`
    # literal is its negation so a later read resolves back to True.
    assert parsed.toml_value is False
    assert parsed.effective_value is True


def test_writable_rejection_flags_credentials() -> None:
    """Credential options are rejected with an `auth set` hint."""
    option = get_option("credentials.anthropic")
    assert option is not None
    reason = writable_rejection(option)
    assert reason is not None
    assert "dcode auth set" in reason


def test_writable_rejection_flags_non_toml_option() -> None:
    """An env-only option (no toml_keys) is rejected as non-persistable."""
    option = get_option("features.experimental")
    assert option is not None
    assert option.toml_keys is None
    reason = writable_rejection(option)
    assert reason is not None
    assert "config.toml" in reason


def test_writable_rejection_allows_startup_mode() -> None:
    """A safe, TOML-backed, supported option is writable."""
    option = get_option("startup.mode")
    assert option is not None
    assert writable_rejection(option) is None


# --- Command: success paths -------------------------------------------------


def test_set_creates_missing_config_file(config_path) -> None:
    """Setting a value creates config.toml when it does not yet exist."""
    assert not config_path.exists()
    assert run_config_command(_set_args("startup.mode", "yolo")) == 0
    assert config_path.exists()
    data = tomllib.loads(config_path.read_text())
    assert data == {"startup": {"mode": "yolo"}}


def test_set_startup_mode_yolo_persists(config_path, capsys) -> None:
    """`config set startup.mode yolo` writes the documented TOML block."""
    assert run_config_command(_set_args("startup.mode", "yolo")) == 0
    data = tomllib.loads(config_path.read_text())
    assert data["startup"]["mode"] == "yolo"
    out = capsys.readouterr().out
    assert "startup.mode" in out
    assert "yolo" in out


def test_set_updates_existing_value(config_path) -> None:
    """Re-setting an option overwrites the previous value."""
    config_path.write_text('[startup]\nmode = "auto"\n')
    assert run_config_command(_set_args("startup.mode", "yolo")) == 0
    data = tomllib.loads(config_path.read_text())
    assert data["startup"]["mode"] == "yolo"


def test_set_preserves_unrelated_settings(config_path) -> None:
    """Unrelated keys and nested tables survive a write."""
    config_path.write_text(
        "[models]\n"
        'default = "openai:gpt-5"\n\n'
        "[ui]\n"
        'theme = "langchain"\n\n'
        "[interpreter]\n"
        "memory_limit_mb = 128\n"
    )
    assert run_config_command(_set_args("startup.mode", "yolo")) == 0
    data = tomllib.loads(config_path.read_text())
    assert data["startup"]["mode"] == "yolo"
    assert data["models"]["default"] == "openai:gpt-5"
    assert data["ui"]["theme"] == "langchain"
    assert data["interpreter"]["memory_limit_mb"] == 128


def test_set_bool_option_persists(config_path) -> None:
    """A boolean option is stored as a TOML bool."""
    assert run_config_command(_set_args("tracing.langsmith_redact", "true")) == 0
    data = tomllib.loads(config_path.read_text())
    assert data["tracing"]["langsmith_redact"] is True


def test_set_int_option_persists(config_path) -> None:
    """An integer option is stored as a TOML integer."""
    assert run_config_command(_set_args("runtime.recursion_limit", "5000")) == 0
    data = tomllib.loads(config_path.read_text())
    assert data["runtime"]["recursion_limit"] == 5000


# --- Command: rejection paths (no file mutation) ----------------------------


def test_set_invalid_value_does_not_write(config_path, capsys) -> None:
    """An invalid value is rejected and never creates or mutates the file."""
    assert run_config_command(_set_args("startup.mode", "dangerously-auto")) == 1
    assert not config_path.exists()
    assert "invalid value" in capsys.readouterr().err


def test_set_invalid_value_preserves_existing_file(config_path) -> None:
    """A rejected write leaves a pre-existing file byte-for-byte unchanged."""
    original = '[startup]\nmode = "manual"\n'
    config_path.write_text(original)
    assert run_config_command(_set_args("startup.mode", "bogus")) == 1
    assert config_path.read_text() == original


def test_set_unknown_key_rejected(config_path, capsys) -> None:
    """An unknown key exits 1 and points at `config --verbose`."""
    assert run_config_command(_set_args("nope.nope", "x")) == 1
    assert not config_path.exists()
    assert "Unknown config option" in capsys.readouterr().err


def test_set_non_writable_key_rejected(config_path, capsys) -> None:
    """An env-only option is rejected without writing."""
    assert run_config_command(_set_args("features.experimental", "true")) == 1
    assert not config_path.exists()
    assert "config.toml" in capsys.readouterr().err


def test_set_credential_key_rejected(config_path, capsys) -> None:
    """A credential option is rejected and never persisted or echoed."""
    assert run_config_command(_set_args("credentials.anthropic", "sk-secret")) == 1
    assert not config_path.exists()
    err = capsys.readouterr().err
    assert "dcode auth set" in err
    # The provided secret value must never be echoed back.
    assert "sk-secret" not in err


@pytest.mark.usefixtures("config_path")
def test_set_missing_args_hints(capsys) -> None:
    """A bare `config set` explains it needs a key and value."""
    assert run_config_command(_set_args(None, None)) == 2
    assert "needs a key and value" in capsys.readouterr().err


# --- Command: override reporting --------------------------------------------


def test_set_reports_higher_precedence_override(
    config_path, capsys, monkeypatch
) -> None:
    """A write shadowed by an env var reports the override, but still persists."""
    option = get_option("tracing.langsmith_redact")
    assert option is not None
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "false")

    assert run_config_command(_set_args("tracing.langsmith_redact", "true")) == 0
    # The TOML value is still written even though env wins at resolution.
    data = tomllib.loads(config_path.read_text())
    assert data["tracing"]["langsmith_redact"] is True
    out = capsys.readouterr().out
    assert "Note:" in out
    assert option.env_var in out


# --- Command: JSON output ---------------------------------------------------


@pytest.mark.usefixtures("config_path")
def test_set_json_output(capsys) -> None:
    """JSON output reports the stored value, source, and override flag."""
    assert (
        run_config_command(_set_args("startup.mode", "yolo", output_format="json")) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "config set"
    data = payload["data"]
    assert data["key"] == "startup.mode"
    assert data["stored_value"] == "yolo"
    assert data["source"] == "config.toml"
    assert data["overridden"] is False


@pytest.mark.usefixtures("config_path")
def test_set_json_unknown_key(capsys) -> None:
    """JSON output surfaces an error field for an unknown key."""
    assert run_config_command(_set_args("nope.nope", "x", output_format="json")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["key"] == "nope.nope"
    assert "error" in payload["data"]


@pytest.mark.usefixtures("config_path")
def test_set_json_override(capsys, monkeypatch) -> None:
    """JSON output flags a higher-precedence override."""
    option = get_option("tracing.langsmith_redact")
    assert option is not None
    assert option.env_var is not None
    monkeypatch.setenv(option.env_var, "false")

    assert (
        run_config_command(
            _set_args("tracing.langsmith_redact", "true", output_format="json")
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["overridden"] is True
    assert data["stored_value"] is True
    assert data["effective_value"] is False


# --- CLI help & argparse wiring ---------------------------------------------


def test_config_set_argparse_wiring(monkeypatch) -> None:
    """`dcode config set startup.mode yolo --json` parses into the namespace."""
    from deepagents_code.main import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["dcode", "config", "set", "startup.mode", "yolo", "--json"],
    )
    ns = parse_args()
    assert ns.config_command == "set"
    assert ns.key == "startup.mode"
    assert ns.value == "yolo"
    assert ns.output_format == "json"


def test_config_set_appears_in_help(capsys) -> None:
    """The `config set` command is documented on the config help screen."""
    from deepagents_code.ui import show_config_help

    show_config_help()
    rendered = " ".join(capsys.readouterr().out.split())
    assert "config set" in rendered
    assert "startup.mode yolo" in rendered
