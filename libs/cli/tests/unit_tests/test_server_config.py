"""Tests for _server_config helpers and ServerConfig invariants."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_cli._env_vars import SERVER_ENV_PREFIX
from deepagents_cli._server_config import (
    ServerConfig,
    _normalize_path,
    _read_env_bool,
    _read_env_json,
    _read_env_optional_bool,
    _read_env_str,
)

# ------------------------------------------------------------------
# _read_env_bool
# ------------------------------------------------------------------


class TestReadEnvBool:
    def test_true_lowercase(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "true"}):
            assert _read_env_bool("FOO") is True

    def test_true_uppercase(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "TRUE"}):
            assert _read_env_bool("FOO") is True

    def test_true_mixed_case(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "True"}):
            assert _read_env_bool("FOO") is True

    def test_false_lowercase(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "false"}):
            assert _read_env_bool("FOO") is False

    def test_false_uppercase(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "FALSE"}):
            assert _read_env_bool("FOO") is False

    def test_arbitrary_string_is_false(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}FOO": "yes"}):
            assert _read_env_bool("FOO") is False

    def test_missing_returns_default_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_bool("MISSING") is False

    def test_missing_returns_custom_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_bool("MISSING", default=True) is True


# ------------------------------------------------------------------
# _read_env_json
# ------------------------------------------------------------------


class TestReadEnvJson:
    def test_valid_json(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}DATA": '{"a": 1}'}):
            assert _read_env_json("DATA") == {"a": 1}

    def test_missing_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_json("MISSING") is None

    def test_malformed_json_raises(self) -> None:
        with (
            patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}DATA": "{bad json"}),
            pytest.raises(ValueError, match="Failed to parse"),
        ):
            _read_env_json("DATA")

    def test_malformed_json_includes_value_snippet(self) -> None:
        with (
            patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}DATA": "{bad"}),
            pytest.raises(ValueError, match=r"\{bad"),
        ):
            _read_env_json("DATA")


# ------------------------------------------------------------------
# _read_env_str
# ------------------------------------------------------------------


class TestReadEnvStr:
    def test_present(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}X": "val"}):
            assert _read_env_str("X") == "val"

    def test_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_str("X") is None


# ------------------------------------------------------------------
# _read_env_optional_bool
# ------------------------------------------------------------------


class TestReadEnvOptionalBool:
    def test_true(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}X": "true"}):
            assert _read_env_optional_bool("X") is True

    def test_false(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}X": "false"}):
            assert _read_env_optional_bool("X") is False

    def test_missing_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_optional_bool("X") is None

    def test_false_distinct_from_none(self) -> None:
        """False and None must not be conflated."""
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}X": "false"}):
            result = _read_env_optional_bool("X")
            assert result is not None
            assert result is False


# ------------------------------------------------------------------
# _normalize_path
# ------------------------------------------------------------------


class TestNormalizePath:
    def test_none_returns_none(self) -> None:
        assert _normalize_path(None, None, "test") is None

    def test_empty_string_returns_none(self) -> None:
        assert _normalize_path("", None, "test") is None

    def test_absolute_path_without_context(self, tmp_path: Path) -> None:
        p = tmp_path / "mcp.json"
        p.touch()
        result = _normalize_path(str(p), None, "MCP config")
        assert result is not None
        assert Path(result).is_absolute()

    def test_raises_on_unresolvable_path(self) -> None:
        with (
            patch(
                "deepagents_cli._server_config.Path.expanduser",
                side_effect=OSError("perm"),
            ),
            pytest.raises(ValueError, match="Could not resolve"),
        ):
            _normalize_path("/some/path/mcp.json", None, "MCP config")

    def test_label_appears_in_error_message(self) -> None:
        with (
            patch(
                "deepagents_cli._server_config.Path.expanduser",
                side_effect=OSError("perm"),
            ),
            pytest.raises(ValueError, match="sandbox setup"),
        ):
            _normalize_path("/some/path/setup.sh", None, "sandbox setup")


# ------------------------------------------------------------------
# ServerConfig.__post_init__
# ------------------------------------------------------------------


class TestServerConfigPostInit:
    def test_sandbox_type_none_string_normalized(self) -> None:
        config = ServerConfig(sandbox_type="none")
        assert config.sandbox_type is None

    def test_sandbox_type_valid_preserved(self) -> None:
        config = ServerConfig(sandbox_type="modal")
        assert config.sandbox_type == "modal"

    def test_sandbox_type_none_value_preserved(self) -> None:
        config = ServerConfig(sandbox_type=None)
        assert config.sandbox_type is None


# ------------------------------------------------------------------
# ServerConfig round-trip edge cases
# ------------------------------------------------------------------


class TestServerConfigEdgeCases:
    def test_trust_project_mcp_false_round_trips(self) -> None:
        """False must survive round-trip (not collapse to None)."""
        original = ServerConfig(trust_project_mcp=False)
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.trust_project_mcp is False

    def test_sandbox_type_none_string_round_trips(self) -> None:
        """sandbox_type='none' normalizes to None and survives round-trip."""
        original = ServerConfig(sandbox_type="none")
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.sandbox_type is None

    def test_repl_runtime_round_trips(self) -> None:
        """Configured REPL runtime should survive env serialization."""
        original = ServerConfig(repl_runtime="quickjs")
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.repl_runtime == "quickjs"
