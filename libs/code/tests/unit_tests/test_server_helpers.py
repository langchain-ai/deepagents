"""Tests for extracted helper functions in server.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_code.agent import _apply_inherited_pythonpath
from deepagents_code.client.launch.server import (
    _SERVER_ENV_DENYLIST,
    _build_server_cmd,
    _build_server_env,
    _scoped_env_overrides,
)
from deepagents_code.config import _DOTENV_DENIED_ENV_KEYS, _INHERITED_PYTHONPATH_ENV


class TestBuildServerCmd:
    def test_contains_host_and_port(self) -> None:
        cmd = _build_server_cmd(Path("/tmp/lg.json"), host="0.0.0.0", port=3000)
        assert "--host" in cmd
        assert "0.0.0.0" in cmd
        assert "--port" in cmd
        assert "3000" in cmd

    def test_contains_config_path(self) -> None:
        p = Path("/work/langgraph.json")
        cmd = _build_server_cmd(p, host="127.0.0.1", port=2024)
        assert str(p) in cmd

    def test_includes_no_browser_and_no_reload(self) -> None:
        cmd = _build_server_cmd(Path("/tmp/lg.json"), host="127.0.0.1", port=2024)
        assert "--no-browser" in cmd
        assert "--no-reload" in cmd


class TestBuildServerEnv:
    def test_sets_auth_noop(self) -> None:
        env = _build_server_env()
        assert env["LANGGRAPH_AUTH_TYPE"] == "noop"

    def test_strips_auth_variables(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LANGGRAPH_AUTH": "secret",
                "LANGGRAPH_CLOUD_LICENSE_KEY": "key",
                "LANGSMITH_CONTROL_PLANE_API_KEY": "cpkey",
                "LANGSMITH_TENANT_ID": "tid",
            },
        ):
            env = _build_server_env()
        assert "LANGGRAPH_AUTH" not in env
        assert "LANGGRAPH_CLOUD_LICENSE_KEY" not in env
        assert "LANGSMITH_CONTROL_PLANE_API_KEY" not in env
        assert "LANGSMITH_TENANT_ID" not in env

    def test_sets_pythondontwritebytecode(self) -> None:
        env = _build_server_env()
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"

    def test_strips_subprocess_hijack_variables(self) -> None:
        injected = {key: f"/tmp/evil-{key}" for key in _SERVER_ENV_DENYLIST}
        with patch.dict(
            os.environ,
            {**injected, "PATH": os.environ.get("PATH", "")},
        ):
            env = _build_server_env()
        for key in _SERVER_ENV_DENYLIST:
            assert key not in env
        assert "PATH" in env

    def test_relays_pythonpath_off_server_interpreter(self) -> None:
        """A launch `PYTHONPATH` is kept off the server but carried for `execute`."""
        with patch.dict(os.environ, {"PYTHONPATH": "src"}):
            env = _build_server_env()
        assert "PYTHONPATH" not in env
        assert env[_INHERITED_PYTHONPATH_ENV] == "src"

    def test_no_carrier_when_pythonpath_absent(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env = _build_server_env()
        assert _INHERITED_PYTHONPATH_ENV not in env

    def test_inherited_carrier_var_is_dropped(self) -> None:
        """A pre-existing carrier var is never trusted as a PYTHONPATH source."""
        with patch.dict(
            os.environ,
            {_INHERITED_PYTHONPATH_ENV: "smuggled", "KEEP_ME": "1"},
            clear=True,
        ):
            env = _build_server_env()
        assert _INHERITED_PYTHONPATH_ENV not in env
        assert env["KEEP_ME"] == "1"

    def test_relays_empty_pythonpath_as_empty(self) -> None:
        """An empty launch `PYTHONPATH` relays as `""` (distinct from absent)."""
        with patch.dict(os.environ, {"PYTHONPATH": ""}):
            env = _build_server_env()
        assert env[_INHERITED_PYTHONPATH_ENV] == ""

    def test_startup_hijack_keys_blocked_from_dotenv(self) -> None:
        """Project `.env` files must not inject interpreter startup hooks."""
        assert "PYTHONPATH" in _SERVER_ENV_DENYLIST
        for key in (
            "BASH_ENV",
            "BASHOPTS",
            "CDPATH",
            "COMSPEC",
            "ENV",
            "GLOBIGNORE",
            "PYTHONPATH",
            "SHELLOPTS",
            "SYSTEMROOT",
            "WINDIR",
            _INHERITED_PYTHONPATH_ENV,
        ):
            assert key in _DOTENV_DENIED_ENV_KEYS


class TestPythonpathRelayRoundTrip:
    def test_launch_pythonpath_round_trips_to_execute_env(self) -> None:
        """A launch `PYTHONPATH` survives the server-env relay to `execute`.

        Composes the two halves (`_build_server_env` strips + carries; the agent
        helper re-applies) to pin the end-to-end contract that the carrier var
        name agrees across modules.
        """
        with patch.dict(os.environ, {"PYTHONPATH": "src"}):
            server_env = _build_server_env()
        assert "PYTHONPATH" not in server_env

        # The shell backend re-applies the relayed value for `execute` commands.
        shell_env = dict(server_env)
        _apply_inherited_pythonpath(shell_env)
        assert shell_env["PYTHONPATH"] == "src"
        assert _INHERITED_PYTHONPATH_ENV not in shell_env


class TestScopedEnvOverrides:
    def test_overrides_applied_inside_context(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            _scoped_env_overrides({"TEST_SCOPED_VAR": "val"}),
        ):
            assert os.environ.get("TEST_SCOPED_VAR") == "val"

    def test_overrides_kept_on_success(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            with _scoped_env_overrides({"TEST_SCOPED_KEEP": "val"}):
                pass
            assert os.environ.get("TEST_SCOPED_KEEP") == "val"

    def test_overrides_rolled_back_on_exception(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            msg = "boom"
            with (
                pytest.raises(RuntimeError),
                _scoped_env_overrides({"TEST_SCOPED_ROLL": "new"}),
            ):
                raise RuntimeError(msg)
            assert os.environ.get("TEST_SCOPED_ROLL") is None

    def test_previous_value_restored_on_exception(self) -> None:
        msg = "boom"
        with patch.dict(os.environ, {"TEST_SCOPED_PREV": "original"}, clear=False):
            with (
                pytest.raises(RuntimeError),
                _scoped_env_overrides({"TEST_SCOPED_PREV": "new"}),
            ):
                raise RuntimeError(msg)
            assert os.environ["TEST_SCOPED_PREV"] == "original"
