"""Tests for _server_config helpers and ServerConfig invariants."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import (
    ServerConfig,
    _interpreter_suppressed_by_sandbox,
    _normalize_path,
    _read_env_bool,
    _read_env_int,
    _read_env_json,
    _read_env_optional_bool,
    _read_env_str,
)
from deepagents_code.config import settings

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
# _read_env_int
# ------------------------------------------------------------------


class TestReadEnvInt:
    def test_valid_int(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}COUNT": "5"}):
            assert _read_env_int("COUNT", default=3) == 5

    def test_missing_returns_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _read_env_int("COUNT", default=3) == 3

    def test_malformed_returns_default(self) -> None:
        with patch.dict(os.environ, {f"{SERVER_ENV_PREFIX}COUNT": "abc"}):
            assert _read_env_int("COUNT", default=3) == 3


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
                "deepagents_code._server_config.Path.expanduser",
                side_effect=OSError("perm"),
            ),
            pytest.raises(ValueError, match="Could not resolve"),
        ):
            _normalize_path("/some/path/mcp.json", None, "MCP config")

    def test_label_appears_in_error_message(self) -> None:
        with (
            patch(
                "deepagents_code._server_config.Path.expanduser",
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

    def test_recursion_limit_none_allowed(self) -> None:
        assert ServerConfig(recursion_limit=None).recursion_limit is None

    def test_recursion_limit_positive_allowed(self) -> None:
        assert ServerConfig(recursion_limit=2000).recursion_limit == 2000

    def test_recursion_limit_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="recursion_limit"):
            ServerConfig(recursion_limit=0)

    def test_recursion_limit_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="recursion_limit"):
            ServerConfig(recursion_limit=-1)

    def test_recursion_limit_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="recursion_limit"):
            ServerConfig(recursion_limit=True)


class TestServerConfigInterpreterDefault:
    """Tests for sandbox-aware interpreter default resolution."""

    def test_bare_server_config_keeps_interpreter_disabled(self) -> None:
        config = ServerConfig()

        assert config.enable_interpreter is False

    @staticmethod
    def _build(*, sandbox_type: str, enable_interpreter: bool | None) -> ServerConfig:
        """Build a `ServerConfig` exercising only the interpreter resolution."""
        return ServerConfig.from_cli_args(
            project_context=None,
            model_name=None,
            model_params=None,
            assistant_id="agent",
            auto_approve=False,
            sandbox_type=sandbox_type,
            sandbox_id=None,
            sandbox_snapshot_name=None,
            sandbox_setup=None,
            enable_shell=True,
            enable_ask_user=False,
            enable_interpreter=enable_interpreter,
            mcp_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            interactive=True,
        )

    def test_local_none_false_uses_settings_default(self) -> None:
        with patch.object(settings, "enable_interpreter", False):
            config = self._build(sandbox_type="none", enable_interpreter=None)

        assert config.enable_interpreter is False

    def test_local_none_true_uses_settings_default(self) -> None:
        with patch.object(settings, "enable_interpreter", True):
            config = self._build(sandbox_type="none", enable_interpreter=None)

        assert config.enable_interpreter is True

    def test_local_explicit_false_is_preserved(self) -> None:
        # An explicit `False` must win over a `True` config default rather than
        # falling through to the settings lookup.
        with patch.object(settings, "enable_interpreter", True):
            config = self._build(sandbox_type="none", enable_interpreter=False)

        assert config.enable_interpreter is False

    def test_empty_sandbox_is_treated_as_local(self) -> None:
        # An empty-string sandbox is falsy and must not be mistaken for a remote
        # backend, which would silently disable the interpreter.
        with patch.object(settings, "enable_interpreter", True):
            config = self._build(sandbox_type="", enable_interpreter=None)

        assert config.enable_interpreter is True

    def test_remote_none_disables_interpreter(self) -> None:
        with patch.object(settings, "enable_interpreter", True):
            config = self._build(sandbox_type="daytona", enable_interpreter=None)

        assert config.enable_interpreter is False

    def test_remote_explicit_true_is_preserved_for_validation(self) -> None:
        config = self._build(sandbox_type="daytona", enable_interpreter=True)

        assert config.enable_interpreter is True


class TestInterpreterSuppressedBySandbox:
    """Tests for the `_interpreter_suppressed_by_sandbox` advisory predicate.

    The predicate takes the *raw* tri-state intent: only the unset default
    (`None`) can be silently suppressed by a sandbox.
    """

    def test_suppressed_when_remote_and_default_on(self) -> None:
        # Unset intent + remote sandbox + default-on = a silent drop worth a heads-up.
        assert _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="daytona", local_default=True
        )

    def test_not_suppressed_on_explicit_enable(self) -> None:
        # `--interpreter` on a sandbox is the user's choice; the server raises a
        # clear error instead of a silent drop.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=True, sandbox_type="daytona", local_default=True
        )

    def test_not_suppressed_on_explicit_opt_out(self) -> None:
        # `--no-interpreter` is an explicit opt-out, not a sandbox-imposed drop.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=False, sandbox_type="daytona", local_default=True
        )

    def test_not_suppressed_when_local(self) -> None:
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type=None, local_default=True
        )

    def test_not_suppressed_when_sandbox_none_string(self) -> None:
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="none", local_default=True
        )

    def test_empty_sandbox_treated_as_local(self) -> None:
        # An empty-string sandbox is falsy and must count as local, so the
        # advisory does not fire spuriously.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="", local_default=True
        )

    def test_not_suppressed_when_default_off(self) -> None:
        # A user who disabled the interpreter in config should not be nagged.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="daytona", local_default=False
        )


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

    def test_enable_interpreter_true_round_trips(self) -> None:
        """A resolved-`True` interpreter must survive the env boundary.

        The "on by default" intent rides entirely on this round-trip: the
        dataclass and `from_env` defaults are both `False`, so if `to_env` ever
        dropped the key the subprocess would silently disable `js_eval`.
        """
        original = ServerConfig(enable_interpreter=True)
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.enable_interpreter is True

    def test_enable_interpreter_false_round_trips(self) -> None:
        """A resolved-`False` interpreter must survive (not flip to the default)."""
        original = ServerConfig(enable_interpreter=False)
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.enable_interpreter is False

    def test_malformed_rubric_max_iterations_env_uses_sdk_default(self) -> None:
        """Bad optional rubric iteration config must fall back to SDK default."""
        with patch.dict(
            os.environ,
            {f"{SERVER_ENV_PREFIX}RUBRIC_MAX_ITERATIONS": "abc"},
            clear=True,
        ):
            restored = ServerConfig.from_env()

        assert restored.rubric_max_iterations is None

    def test_rubric_max_iterations_env_parses_int(self) -> None:
        """Valid rubric iteration config survives the server env boundary."""
        with patch.dict(
            os.environ,
            {f"{SERVER_ENV_PREFIX}RUBRIC_MAX_ITERATIONS": "7"},
            clear=True,
        ):
            restored = ServerConfig.from_env()

        assert restored.rubric_max_iterations == 7

    def test_profile_overrides_round_trip_to_server(self) -> None:
        """Profile metadata must reach server-side model construction."""
        original = ServerConfig(profile_overrides={"max_input_tokens": 32000})
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.profile_overrides == {"max_input_tokens": 32000}

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

    def test_sandbox_snapshot_name_round_trips(self) -> None:
        """Snapshot/blueprint names survive server env serialization."""
        original = ServerConfig(
            sandbox_type="langsmith",
            sandbox_snapshot_name="customer-image",
        )
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.sandbox_type == "langsmith"
        assert restored.sandbox_snapshot_name == "customer-image"

    def test_sandbox_snapshot_name_empty_env_normalizes_to_none(self) -> None:
        """An empty `SANDBOX_SNAPSHOT_NAME` env var must not trip the validator."""
        with patch.dict(
            os.environ,
            {f"{SERVER_ENV_PREFIX}SANDBOX_SNAPSHOT_NAME": ""},
            clear=True,
        ):
            restored = ServerConfig.from_env()

        assert restored.sandbox_snapshot_name is None

    def test_recursion_limit_round_trips(self) -> None:
        """An explicit recursion limit survives the server env boundary."""
        original = ServerConfig(recursion_limit=3000)
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.recursion_limit == 3000

    def test_recursion_limit_unset_defaults_to_none(self) -> None:
        """No recursion-limit env var restores as `None` (resolve at build time)."""
        original = ServerConfig()
        env_dict = original.to_env()
        assert env_dict["RECURSION_LIMIT"] is None
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.recursion_limit is None

    def test_malformed_recursion_limit_env_uses_none(self) -> None:
        """A malformed recursion-limit env var falls back to `None`."""
        with patch.dict(
            os.environ,
            {f"{SERVER_ENV_PREFIX}RECURSION_LIMIT": "abc"},
            clear=True,
        ):
            restored = ServerConfig.from_env()

        assert restored.recursion_limit is None
