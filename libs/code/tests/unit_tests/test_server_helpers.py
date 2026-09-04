"""Tests for extracted helper functions in server.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

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
    _INHERITED_USER_TRACING_ENV,
    apply_inherited_user_tracing,
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


class TestServerEnvProfilePinning:
    """The server must always inherit the client's profile selection.

    `persist_env` validates its keys, but `update_env` accepts any key. Without
    the final re-pin a caller could point the server at a different profile
    than the client, splitting the trust root across the two processes.
    """

    def test_persistent_override_cannot_move_the_profile(self) -> None:
        env = _server_env_with_overrides({"DEEPAGENTS_HOME": "/tmp/evil"}, {})
        assert env["DEEPAGENTS_HOME"] == str(PATHS.profile.root)


class TestUserTracingRelay:
    """The caller's tracing identity must survive the client/server boundary.

    The server inherits an environment where bootstrap has already replaced the
    canonical LangSmith flags and key, so its own `_bootstrap_state` capture
    holds the *agent's* values. Restoring from it would be a no-op that leaks
    the agent session key into user `execute` commands.
    """

    def test_relays_caller_tracing_values_for_execute(self) -> None:
        """`execute` receives the caller's key, not the agent's override."""
        import deepagents_code.config as config_mod

        state = config_mod._bootstrap_state
        with (
            patch.object(state, "done", True),
            patch.object(state, "original_tracing_env", {"LANGSMITH_TRACING": None}),
            patch.object(
                state,
                "original_tracing_api_keys",
                {"LANGSMITH_API_KEY": "caller-key"},
            ),
            patch.dict(
                os.environ,
                {
                    "LANGSMITH_API_KEY": "agent-session-key",
                    "LANGSMITH_TRACING": "true",
                },
                clear=False,
            ),
        ):
            server_env = _build_server_env()

        # The agent's own values still reach the server process itself.
        assert server_env["LANGSMITH_API_KEY"] == "agent-session-key"

        shell_env = dict(server_env)
        assert apply_inherited_user_tracing(shell_env) is True
        assert shell_env["LANGSMITH_API_KEY"] == "caller-key"
        # The caller had no tracing flag, so the agent's must not survive.
        assert "LANGSMITH_TRACING" not in shell_env
        assert _INHERITED_USER_TRACING_ENV not in shell_env

    def test_no_relay_reports_absence_so_local_capture_wins(self) -> None:
        """Without a carrier the local `_bootstrap_state` stays authoritative."""
        shell_env = {"LANGSMITH_API_KEY": "agent-session-key"}
        assert apply_inherited_user_tracing(shell_env) is False
        assert shell_env["LANGSMITH_API_KEY"] == "agent-session-key"

    def test_unparsable_relay_fails_closed(self) -> None:
        """A corrupt carrier drops the agent's key rather than passing it on."""
        shell_env = {
            _INHERITED_USER_TRACING_ENV: "not json",
            "LANGSMITH_API_KEY": "agent-session-key",
            "LANGSMITH_TRACING": "true",
        }
        assert apply_inherited_user_tracing(shell_env) is True
        assert "LANGSMITH_API_KEY" not in shell_env
        assert "LANGSMITH_TRACING" not in shell_env

    def test_relay_applies_only_tracing_keys(self) -> None:
        """A relayed blob cannot write arbitrary shell environment keys."""
        shell_env = {
            _INHERITED_USER_TRACING_ENV: json.dumps(
                {
                    "LANGSMITH_API_KEY": "caller-key",
                    "PATH": "/attacker/bin",
                    "LD_PRELOAD": "/attacker/evil.so",
                }
            ),
            "PATH": "/usr/bin",
        }

        assert apply_inherited_user_tracing(shell_env) is True

        assert shell_env["LANGSMITH_API_KEY"] == "caller-key"
        assert shell_env["PATH"] == "/usr/bin"
        assert "LD_PRELOAD" not in shell_env

    def test_inherited_carrier_var_is_never_trusted(self) -> None:
        """A smuggled carrier cannot choose the tracing identity."""
        import deepagents_code.config as config_mod

        state = config_mod._bootstrap_state
        with (
            patch.object(state, "done", False),
            patch.dict(
                os.environ,
                {_INHERITED_USER_TRACING_ENV: '{"LANGSMITH_API_KEY": "smuggled"}'},
                clear=False,
            ),
        ):
            server_env = _build_server_env()
        assert _INHERITED_USER_TRACING_ENV not in server_env
