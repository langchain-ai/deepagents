"""Tests for extracted helper functions in server.py."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deepagents_code._paths import PATHS
from deepagents_code.agent import _apply_inherited_pythonpath
from deepagents_code.client.launch.server import (
    _SERVER_ENV_DENYLIST,
    _build_server_cmd,
    _build_server_env,
    _server_env_with_overrides,
)
from deepagents_code.config import (
    _INHERITED_PYTHONPATH_ENV,
    _USER_LANGSMITH_ENV_CARRIER,
)


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

    def test_strips_values_injected_by_client_dotenv_loader(self) -> None:
        """A client project value does not become server launch state."""
        import deepagents_code.config as config_mod

        config_mod._dotenv_loaded_values["WORKSPACE_VALUE"] = "first"
        try:
            with patch.dict(os.environ, {"WORKSPACE_VALUE": "first"}, clear=False):
                env = _build_server_env()
            assert "WORKSPACE_VALUE" not in env

            with patch.dict(os.environ, {"WORKSPACE_VALUE": "changed"}, clear=False):
                env = _build_server_env()
            assert env["WORKSPACE_VALUE"] == "changed"
        finally:
            config_mod._dotenv_loaded_values.pop("WORKSPACE_VALUE", None)

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

    def test_requires_captured_langsmith_environment(self) -> None:
        import deepagents_code.config as config_mod

        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = {}
        config_mod._bootstrap_state.user_langsmith_env = {}
        try:
            with pytest.raises(
                RuntimeError,
                match="were not captured at startup",
            ):
                _build_server_env()
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user

    def test_names_the_swallowed_bootstrap_failure(self) -> None:
        """Bootstrap tolerates its own errors, so this must name the cause."""
        import deepagents_code.config as config_mod

        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        original_error = config_mod._bootstrap_state.error
        config_mod._bootstrap_state.launch_langsmith_env = {}
        config_mod._bootstrap_state.user_langsmith_env = {}
        config_mod._bootstrap_state.error = OSError("profile config unreadable")
        try:
            with pytest.raises(RuntimeError, match="profile config unreadable") as exc:
                _build_server_env()
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user
            config_mod._bootstrap_state.error = original_error

        assert isinstance(exc.value.__cause__, OSError)

    def test_overwrites_untrusted_langsmith_carrier(self) -> None:
        import json

        import deepagents_code.config as config_mod

        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env["LANGSMITH_PROFILE"] = "oauth"
        try:
            with patch.dict(
                os.environ,
                {_USER_LANGSMITH_ENV_CARRIER: '{"LANGSMITH_API_KEY":"evil"}'},
            ):
                env = _build_server_env()
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user

        assert json.loads(env[_USER_LANGSMITH_ENV_CARRIER])["user"] == {
            **dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS),
            "LANGSMITH_PROFILE": "oauth",
        }


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


class TestLangSmithCarrierRoundTrip:
    @pytest.mark.parametrize("user_key", ["user-launch-key", None])
    def test_launch_langsmith_env_round_trips_to_execute_env(
        self, monkeypatch: pytest.MonkeyPatch, user_key: str | None
    ) -> None:
        """The real server handoff restores user settings and removes agent ones."""
        import deepagents_code.config as config_mod
        from deepagents_code.config import restore_user_langsmith_env

        launch = dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS)
        launch["LANGSMITH_API_KEY"] = user_key
        launch["LANGSMITH_PROJECT"] = "user-launch-project"
        user = dict(
            launch, LANGSMITH_PROFILE="oauth", LANGSMITH_CONFIG_FILE="/tmp/ls.json"
        )
        monkeypatch.setattr(config_mod._bootstrap_state, "launch_langsmith_env", launch)
        monkeypatch.setattr(config_mod._bootstrap_state, "user_langsmith_env", user)
        server_env = _build_server_env()

        # The server hands the agent an environment holding its own key.
        shell_env = dict(server_env)
        shell_env.update(
            LANGSMITH_API_KEY="agent-session-key",
            LANGSMITH_TRACING="true",
            DEEPAGENTS_CODE_LANGSMITH_TRACING="true",
        )
        restore_user_langsmith_env(shell_env)

        if user_key is None:
            assert "LANGSMITH_API_KEY" not in shell_env
        else:
            assert shell_env["LANGSMITH_API_KEY"] == user_key
        assert shell_env["LANGSMITH_PROJECT"] == "user-launch-project"
        assert shell_env["LANGSMITH_PROFILE"] == "oauth"
        assert shell_env["LANGSMITH_CONFIG_FILE"] == "/tmp/ls.json"
        assert "LANGSMITH_TRACING" not in shell_env
        assert "DEEPAGENTS_CODE_LANGSMITH_TRACING" not in shell_env
        assert _USER_LANGSMITH_ENV_CARRIER not in shell_env


class TestServerEnvProfilePinning:
    """The server must always inherit the client's profile selection.

    `persist_env` validates its keys, but `update_env` accepts any key. Without
    the final re-pin a caller could point the server at a different profile
    than the client, splitting the trust root across the two processes.
    """

    def test_persistent_override_cannot_move_the_profile(self) -> None:
        env = _server_env_with_overrides({"DEEPAGENTS_HOME": "/tmp/evil"}, {})
        assert env["DEEPAGENTS_HOME"] == str(PATHS.profile.root)

    def test_scoped_override_cannot_replace_langsmith_carrier(self) -> None:
        import json

        import deepagents_code.config as config_mod

        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env["LANGSMITH_PROFILE"] = "oauth"
        try:
            env = _server_env_with_overrides(
                {},
                {_USER_LANGSMITH_ENV_CARRIER: '{"LANGSMITH_API_KEY":"evil"}'},
            )
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user

        user = json.loads(env[_USER_LANGSMITH_ENV_CARRIER])["user"]
        assert user["LANGSMITH_PROFILE"] == "oauth"
        assert user["LANGSMITH_API_KEY"] is None
