"""Tests for config module including project discovery utilities."""

import json
import logging
import subprocess
import sys
import warnings
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import pytest

from deepagents_code import _git as git_module, config as config_module, model_config
from deepagents_code._paths import get_agent_dir, get_user_agent_md_path
from deepagents_code.config import (
    _MCP_SHUTDOWN_RACE_MESSAGES,
    _MCP_SSE_LOGGER_NAME,
    _MCP_STREAMABLE_HTTP_LOGGER_NAME,
    _QUIET_SDK_LOGGER_NAMES,
    LANGSMITH_EU_ENDPOINT,
    LANGSMITH_US_ENDPOINT,
    RECOMMENDED_SAFE_SHELL_COMMANDS,
    SHELL_ALLOW_ALL,
    Credentials,
    LangsmithShadowResult,
    ModelResult,
    _apply_default_langsmith_project,
    _apply_stored_langsmith_tracing,
    _create_model_via_init,
    _disable_orphaned_tracing,
    _get_provider_kwargs,
    _McpShutdownRaceFilter,
    _quiet_sdk_logging,
    _read_retry_config,
    _resolve_model_retries_from_section,
    apply_stored_langsmith_auth,
    configure_langsmith_secret_redaction,
    consume_orphaned_tracing_disabled_notice,
    create_model,
    credentials,
    detect_provider,
    get_langsmith_project_name,
    is_http_url,
    is_langsmith_redaction_enabled,
    langsmith_key_shadowed_by_empty_override,
    parse_shell_allow_list,
    reset_langsmith_url_cache,
    runtime_state,
)
from deepagents_code.configuration.interpreter import InterpreterConfig
from deepagents_code.model_config import (
    ModelConfig,
    ModelConfigError,
    ModelNotAllowedError,
    clear_caches,
)
from deepagents_code.project_utils import (
    find_project_agent_md as _find_project_agent_md,
)


class TestRuntimeDotenvReload:
    """Tests for project-scoped dotenv refresh behavior."""

    def test_direct_reload_preserves_user_settings_for_server_commands(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Direct reload preserves user settings through the server handoff."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.client.launch.server import _build_server_env

        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        monkeypatch.setattr(config_mod._bootstrap_state, "launch_langsmith_env", {})
        monkeypatch.setattr(config_mod._bootstrap_state, "user_langsmith_env", {})
        monkeypatch.setattr(config_mod, "_dotenv_loaded_values", {})
        monkeypatch.delenv(config_mod._USER_LANGSMITH_ENV_CARRIER, raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "user-launch-key")
        (tmp_path / ".env").write_text("LANGSMITH_PROJECT=user-project\n")
        runtime = Credentials.from_environment(start_path=tmp_path)

        runtime.reload_from_environment(start_path=tmp_path)
        shell_env = _build_server_env()
        assert config_mod._USER_LANGSMITH_ENV_CARRIER not in os.environ
        shell_env.update(
            LANGSMITH_API_KEY="agent-key", LANGSMITH_PROJECT="agent-project"
        )
        config_mod.restore_user_langsmith_env(shell_env)

        assert shell_env["LANGSMITH_API_KEY"] == "user-launch-key"
        assert shell_env["LANGSMITH_PROJECT"] == "user-project"
        assert config_mod._USER_LANGSMITH_ENV_CARRIER not in shell_env

    def test_reload_restores_the_launch_value_over_an_agent_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reload un-does the agent's in-process override from the carrier."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", raising=False)
        # What the agent put on the canonical var in this process.
        monkeypatch.setenv("LANGSMITH_API_KEY", "agent-override")
        monkeypatch.setenv("LANGSMITH_PROFILE", "agent-profile")

        # The user's launch environment: a different key, and no profile at all.
        launch = dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS)
        launch["LANGSMITH_API_KEY"] = "user-key"
        monkeypatch.setenv(
            config_mod._USER_LANGSMITH_ENV_CARRIER,
            json.dumps({"launch": launch, "user": dict(launch)}),
        )
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._dotenv_loaded_values.clear()

        try:
            runtime = Credentials.from_environment(start_path=tmp_path)

            runtime.reload_from_environment(start_path=tmp_path)

            assert os.environ["LANGSMITH_API_KEY"] == "user-key"
            # `None` in the carrier means the user had none: remove it.
            assert "LANGSMITH_PROFILE" not in os.environ
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user
            config_mod._dotenv_loaded_values.clear()

    def test_reload_keeps_settings_when_the_carrier_is_unusable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An undecodable carrier leaves LangSmith settings alone, and reports."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        monkeypatch.setenv(config_mod._USER_LANGSMITH_ENV_CARRIER, "{not json")
        monkeypatch.setenv("LANGSMITH_API_KEY", "in-process-key")
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        # A stale mapping that would otherwise be written over `os.environ`.
        config_mod._bootstrap_state.launch_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            runtime = Credentials.from_environment(start_path=tmp_path)

            changes = runtime.reload_from_environment(start_path=tmp_path)

            assert os.environ["LANGSMITH_API_KEY"] == "in-process-key"
            assert any("could not be read" in change for change in changes)
            # The refusal must not be laundered: republishing here would encode
            # the in-process agent key into a well-formed carrier that every
            # later restore accepts without warning.
            assert os.environ[config_mod._USER_LANGSMITH_ENV_CARRIER] == "{not json"
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user
            config_mod._dotenv_loaded_values.clear()

    def test_reload_from_environment_refreshes_loaded_project_dotenv_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runtime reload replaces managed `.env` values after a cwd switch."""
        import os

        import deepagents_code.config as config_mod

        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()
        (current / ".env").write_text(
            "DEEPAGENTS_CODE_OPENAI_API_KEY=sk-current\n",
        )
        (target / ".env").write_text(
            "DEEPAGENTS_CODE_OPENAI_API_KEY=sk-target\n"
            "DEEPAGENTS_CODE_ANTHROPIC_API_KEY=sk-target-anthropic\n",
        )

        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_ANTHROPIC_API_KEY", "sk-shell")
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=current)
            runtime = Credentials.from_environment(start_path=current)
            assert runtime.openai_api_key == "sk-current"

            changes = runtime.reload_from_environment(start_path=target)

            assert runtime.openai_api_key == "sk-target"
            assert os.environ["DEEPAGENTS_CODE_OPENAI_API_KEY"] == "sk-target"
            assert runtime.anthropic_api_key == "sk-shell"
            assert os.environ["DEEPAGENTS_CODE_ANTHROPIC_API_KEY"] == "sk-shell"
            assert "openai_api_key: set -> set" in changes
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_reload_resets_prefixed_resolution_logging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Runtime reload starts a new generation of resolution diagnostics."""
        import deepagents_code.config as config_mod
        from deepagents_code.model_config import (
            reset_env_resolution_log,
            resolve_env_var,
        )

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-prefixed")
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        caplog.set_level(logging.DEBUG, logger="deepagents_code.model_config")
        reset_env_resolution_log()
        try:
            runtime = Credentials.from_environment(start_path=tmp_path)
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            runtime.reload_from_environment(start_path=tmp_path)
            assert resolve_env_var("OPENAI_API_KEY") == "sk-prefixed"
            assert (
                caplog.messages.count(
                    "Resolved OPENAI_API_KEY from DEEPAGENTS_CODE_OPENAI_API_KEY"
                )
                == 2
            )
        finally:
            reset_env_resolution_log()

    def test_reload_reapplies_prefixed_langsmith_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reload keeps the session key on the canonical SDK variable."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "prefixed-key")
        monkeypatch.setenv("LANGSMITH_API_KEY", "prefixed-key")
        auth_store.set_stored_key("langsmith", "stored-key")
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        config_mod._bootstrap_state.user_langsmith_env = dict(
            config_mod._bootstrap_state.launch_langsmith_env
        )

        try:
            runtime = Credentials.from_environment(start_path=tmp_path)

            runtime.reload_from_environment(start_path=tmp_path)

            assert os.environ["LANGSMITH_API_KEY"] == "prefixed-key"
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user

    def test_reload_redefaults_project_when_override_cleared_and_tracing_on(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clearing the agent project on reload re-applies the default.

        Regression: when a cwd switch unsets `DEEPAGENTS_CODE_LANGSMITH_PROJECT`
        and the user has no original `LANGSMITH_PROJECT`, the reload must fall
        back to `deepagents-code` (not leave the var unset) so trace ingestion
        keeps matching the name `get_langsmith_project_name` displays.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        current = tmp_path / "current"
        target = tmp_path / "target"
        current.mkdir()
        target.mkdir()

        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)

        try:
            # User never set LANGSMITH_PROJECT; tracing is active with a key.
            config_mod._bootstrap_state.original_langsmith_project = None
            config_mod._bootstrap_state.launch_langsmith_env = dict.fromkeys(
                config_mod._USER_LANGSMITH_ENV_VARS
            )
            config_mod._bootstrap_state.launch_langsmith_env["LANGSMITH_API_KEY"] = (
                "lsv2_test"
            )
            config_mod._bootstrap_state.launch_langsmith_env["LANGSMITH_TRACING"] = (
                "true"
            )
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            # Agent-project override is active before the reload, cleared after.
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "agent-project")

            runtime = Credentials.from_environment(start_path=current)
            assert runtime.deepagents_langchain_project == "agent-project"

            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            runtime.reload_from_environment(start_path=target)

            assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        finally:
            config_mod._bootstrap_state.original_langsmith_project = original_ls
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._dotenv_loaded_values.clear()


class TestWorkspaceDotenvEnvironment:
    """Workspace previews stay isolated without replacing the process environment."""

    def test_conflicting_workspaces_resolve_independently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Construction-time consumers read each workspace's immutable snapshot."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config_manifest import get_option
        from deepagents_code.configuration.providers import EnvProvider
        from deepagents_code.configuration.types import Found
        from deepagents_code.mcp_config import resolve_mcp_server_env
        from deepagents_code.model_config import resolve_env_var

        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / ".env").write_text(
            "OPENAI_API_KEY=first-key\nDEEPAGENTS_CODE_TEST_VALUE=first\nGOOGLE_CLOUD_LOCATION=first-region\n",
            encoding="utf-8",
        )
        (second / ".env").write_text(
            "OPENAI_API_KEY=second-key\nDEEPAGENTS_CODE_TEST_VALUE=second\nGOOGLE_CLOUD_LOCATION=second-region\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_VALUE", raising=False)
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )
        option = get_option("credentials.google_cloud_location")
        assert option is not None

        for workspace, expected in ((first, "first"), (second, "second")):
            env = config_mod._preview_dotenv_environ(start_path=workspace)
            with config_mod.use_environment(env):
                assert resolve_env_var("TEST_VALUE") == expected
                assert (
                    resolve_mcp_server_env(
                        "srv", {"command": "${DEEPAGENTS_CODE_TEST_VALUE}"}
                    )["command"]
                    == expected
                )
                snapshot = config_mod.Credentials.snapshot_from_environment(
                    start_path=workspace
                )
                assert snapshot.openai_api_key == f"{expected}-key"
                assert EnvProvider().get(option).result == Found(f"{expected}-region")

        assert "OPENAI_API_KEY" not in os.environ
        assert "DEEPAGENTS_CODE_TEST_VALUE" not in os.environ

    def test_workspace_only_tracing_is_active(self) -> None:
        """Tracing flags and project resolve from the workspace snapshot."""
        import deepagents_code.config as config_mod

        with config_mod.use_environment(
            {
                "LANGSMITH_API_KEY": "lsv2-workspace",
                "LANGSMITH_TRACING": "true",
                "LANGSMITH_PROJECT": "workspace-traces",
            }
        ):
            assert config_mod.get_langsmith_project_name() == "workspace-traces"

    def test_snapshot_preserves_project_replaced_during_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shells retain the caller's project after the agent override is applied."""
        import deepagents_code.config as config_mod

        monkeypatch.setattr(config_mod._bootstrap_state, "done", True)
        monkeypatch.setattr(
            config_mod._bootstrap_state,
            "original_langsmith_project",
            "caller-traces",
        )
        snapshot = config_mod.Credentials.snapshot_from_environment(
            environ={
                "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "agent-traces",
                "LANGSMITH_PROJECT": "agent-traces",
            }
        )

        assert snapshot.deepagents_langchain_project == "agent-traces"
        assert snapshot.user_langchain_project == "caller-traces"

    def test_environment_binding_is_immutable_and_restored(self) -> None:
        """Bindings copy inputs and reset after exceptions."""
        import os

        from deepagents_code.config import active_environment, use_environment

        source = {"VALUE": "workspace"}
        error = RuntimeError("boom")

        def fail() -> None:
            with use_environment(source):
                source["VALUE"] = "changed"
                assert active_environment()["VALUE"] == "workspace"
                with pytest.raises(TypeError):
                    active_environment()["VALUE"] = "forbidden"  # ty: ignore[invalid-assignment]
                raise error

        with pytest.raises(RuntimeError):
            fail()
        assert active_environment() is os.environ

    def test_windows_dotenv_keys_follow_case_insensitive_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lowercase dotenv names normalize and cannot replace shell values."""
        import deepagents_code.config as config_mod
        import deepagents_code.config_manifest as manifest

        (tmp_path / ".env").write_text("openai_api_key=dotenv-key\n")
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        monkeypatch.setattr(manifest, "resolve_read_project_dotenv", lambda **_: True)

        from_dotenv = config_mod._dotenv_environment(
            start_path=tmp_path,
            environ={},
        )
        from_shell = config_mod._dotenv_environment(
            start_path=tmp_path,
            environ={"OPENAI_API_KEY": "shell-key"},
        )

        assert from_dotenv["OPENAI_API_KEY"] == "dotenv-key"
        assert "openai_api_key" not in from_dotenv
        assert from_shell["OPENAI_API_KEY"] == "shell-key"

    def test_windows_interpolation_resolves_normalized_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lowercase `${...}` reference finds the normalized key."""
        import deepagents_code.config as config_mod
        import deepagents_code.config_manifest as manifest

        (tmp_path / ".env").write_text(
            "proxy_url=http://proxy:8080\nHTTPS_PROXY=${proxy_url}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )
        monkeypatch.setattr(manifest, "resolve_read_project_dotenv", lambda **_: True)

        env = config_mod._dotenv_environment(start_path=tmp_path, environ={})

        assert env["HTTPS_PROXY"] == "http://proxy:8080"

    def test_preview_interpolates_prior_dotenv_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Later values interpolate earlier values from the same file."""
        import deepagents_code.config as config_mod

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_text(
            "BASE=project\nCOMPOSED=${BASE}-value\n", encoding="utf-8"
        )
        monkeypatch.delenv("BASE", raising=False)
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )

        env = config_mod._preview_dotenv_environ(start_path=workspace)

        assert env["BASE"] == "project"
        assert env["COMPOSED"] == "project-value"

    def test_preview_interpolates_the_value_that_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reference resolves to the effective value, not the shadowed one."""
        import deepagents_code.config as config_mod

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_text(
            "BASE=project\nCOMPOSED=${BASE}-value\n", encoding="utf-8"
        )
        monkeypatch.setenv("BASE", "shell")
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )

        env = config_mod._preview_dotenv_environ(start_path=workspace)

        # The shell value outranks the file, so `${BASE}` must not expand to the
        # file's losing value: the environment stays self-consistent.
        assert env["BASE"] == "shell"
        assert env["COMPOSED"] == "shell-value"

    def test_preview_preserves_shell_project_global_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Immutable snapshots retain existing first-write-wins behavior."""
        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "SHELL_VALUE=project\nPROJECT_VALUE=project\n",
            encoding="utf-8",
        )
        global_dotenv = tmp_path / "global.env"
        global_dotenv.write_text(
            "SHELL_VALUE=global\nPROJECT_VALUE=global\nGLOBAL_VALUE=global\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("SHELL_VALUE", "shell")
        monkeypatch.delenv("PROJECT_VALUE", raising=False)
        monkeypatch.delenv("GLOBAL_VALUE", raising=False)
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", global_dotenv)

        env = config_mod._preview_dotenv_environ(start_path=project)

        assert env["SHELL_VALUE"] == "shell"
        assert env["PROJECT_VALUE"] == "project"
        assert env["GLOBAL_VALUE"] == "global"


class TestProjectDotenvDeniedKeys:
    """A cloned repo must not set user-level environment values."""

    def test_project_values_do_not_interpolate_into_the_global_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A project `.env` cannot reach a denied key through the global file."""
        import deepagents_code.config as config_mod
        import deepagents_code.config_manifest as manifest

        denied = "DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS"
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text("MY_SERVERS=evil-server\n", encoding="utf-8")
        global_dotenv = tmp_path / "global.env"
        global_dotenv.write_text(f"{denied}=${{MY_SERVERS}}\n", encoding="utf-8")
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", global_dotenv)
        monkeypatch.setattr(manifest, "resolve_read_project_dotenv", lambda **_: True)

        env = config_mod._dotenv_environment(start_path=project, environ={})

        # The project file may set its own name, but the trusted global file
        # must not expand it into a key the project is denied.
        assert env["MY_SERVERS"] == "evil-server"
        assert env[denied] == ""

    def test_profile_dotenv_inside_project_keeps_project_provenance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One file cannot become trusted by also owning the active profile."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code._env_vars import (
            DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
            READ_PROJECT_DOTENV,
        )

        dotenv = tmp_path / ".env"
        dotenv.write_text(
            f"{READ_PROJECT_DOTENV}=0\n"
            f"{DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}=attacker\n"
            "DEEPAGENTS_CODE_TEST_PROJECT_VALUE=allowed\n",
            encoding="utf-8",
        )
        monkeypatch.delenv(DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS, raising=False)
        monkeypatch.delenv(READ_PROJECT_DOTENV, raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_PROJECT_VALUE", raising=False)
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", dotenv)
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=tmp_path)
            preview = config_mod._preview_dotenv_environ(start_path=tmp_path)

            assert DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in os.environ
            assert DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in preview
            assert os.environ["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
            assert preview["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_profile_dotenv_identity_error_keeps_project_provenance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An identity error cannot promote a project file to trusted input."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code._env_vars import (
            DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        )

        project = tmp_path / "cloned-repo"
        alias_parent = project / "alias"
        alias_parent.mkdir(parents=True)
        dotenv = project / ".env"
        dotenv.write_text(
            f"{DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS}=attacker\n"
            "DEEPAGENTS_CODE_TEST_PROJECT_VALUE=allowed\n",
            encoding="utf-8",
        )
        profile_alias = alias_parent / ".." / ".env"
        monkeypatch.delenv(DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS, raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_PROJECT_VALUE", raising=False)
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", profile_alias)
        monkeypatch.setattr(
            Path,
            "samefile",
            Mock(side_effect=PermissionError("identity unavailable")),
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=project)
            preview = config_mod._preview_dotenv_environ(start_path=project)

            assert DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in os.environ
            assert DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS not in preview
            assert os.environ["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
            assert preview["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_project_dotenv_cannot_configure_forked_subagents(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A repo cannot alter whether the built-in subagent inherits state."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code._env_vars import FORKED_SUBAGENTS

        project = tmp_path / "cloned-repo"
        project.mkdir()
        (project / ".env").write_text(
            f"{FORKED_SUBAGENTS}=false\nDEEPAGENTS_CODE_TEST_PROJECT_VALUE=allowed\n",
            encoding="utf-8",
        )

        monkeypatch.delenv(FORKED_SUBAGENTS, raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_PROJECT_VALUE", raising=False)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=project)
            preview = config_mod._preview_dotenv_environ(start_path=project)

            assert FORKED_SUBAGENTS not in os.environ
            assert FORKED_SUBAGENTS not in preview
            assert os.environ["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
            assert preview["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_project_dotenv_cannot_set_langgraph_recursion_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A repo cannot bypass the bounded Deep Agents recursion setting."""
        import os

        import deepagents_code.config as config_mod

        upstream_env = "LANGGRAPH_DEFAULT_RECURSION_LIMIT"
        project = tmp_path / "cloned-repo"
        project.mkdir()
        (project / ".env").write_text(
            f"{upstream_env}=0\nDEEPAGENTS_CODE_TEST_PROJECT_VALUE=allowed\n",
            encoding="utf-8",
        )

        monkeypatch.delenv(upstream_env, raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_PROJECT_VALUE", raising=False)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=project)
            preview = config_mod._preview_dotenv_environ(start_path=project)

            assert upstream_env not in os.environ
            assert upstream_env not in preview
            assert os.environ["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
            assert preview["DEEPAGENTS_CODE_TEST_PROJECT_VALUE"] == "allowed"
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_project_dotenv_cannot_set_langgraph_recursion_default_lowercase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lowercase spelling cannot bypass the project denial on Windows."""
        import os

        import deepagents_code.config as config_mod

        upstream_env = "LANGGRAPH_DEFAULT_RECURSION_LIMIT"
        project = tmp_path / "cloned-repo"
        project.mkdir()
        (project / ".env").write_text(
            "langgraph_default_recursion_limit=0\n"
            "deepagents_code_test_project_value=allowed\n",
            encoding="utf-8",
        )

        monkeypatch.delenv(upstream_env, raising=False)
        monkeypatch.delenv("deepagents_code_test_project_value", raising=False)
        monkeypatch.delenv("DEEPAGENTS_CODE_TEST_PROJECT_VALUE", raising=False)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=project)
            preview = config_mod._preview_dotenv_environ(start_path=project)

            # On POSIX the lowercase key is a distinct, inert variable; on
            # Windows the same assignment would activate the real one. Either
            # way the denied spelling must not reach the environment.
            assert upstream_env not in os.environ
            assert upstream_env not in preview
            assert "langgraph_default_recursion_limit" not in os.environ
            assert "langgraph_default_recursion_limit" not in preview
        finally:
            config_mod._dotenv_loaded_values.clear()

    def test_project_dotenv_cannot_set_term_program(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Trace terminal metadata must come from the launch environment."""
        import os

        import deepagents_code.config as config_mod

        project = tmp_path / "cloned-repo"
        project.mkdir()
        (project / ".env").write_text(
            "TERM_PROGRAM=${LANGSMITH_API_KEY}\n"
            "DEEPAGENTS_CODE_OPENAI_API_KEY=sk-from-project\n",
        )

        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-value-not-for-tracing")
        monkeypatch.delenv("DEEPAGENTS_CODE_OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        try:
            config_mod._load_dotenv(start_path=project)

            assert "TERM_PROGRAM" not in os.environ
            metadata = config_mod.build_stream_config("thread-123", assistant_id=None)[
                "metadata"
            ]
            assert "dcode_term_program" not in metadata
            assert os.environ["DEEPAGENTS_CODE_OPENAI_API_KEY"] == "sk-from-project"
        finally:
            config_mod._dotenv_loaded_values.clear()


class TestEditableInstallInfo:
    """Editable-install detection degrades safely without a usable home."""


class TestResolveReadProjectDotenv:
    """`startup.read_project_dotenv` resolution across config layers."""


class TestProjectRootDetection:
    """Test project root detection via .git directory."""


class TestGitMetadataLookup:
    """Tests for shared git metadata helpers."""

    def setup_method(self) -> None:
        """Clear git metadata caches between tests."""
        git_module._git_dir_cache.clear()


class TestProjectContext:
    """Tests for explicit project context handling."""


class TestProjectAgentMdFinding:
    """Test finding project-specific AGENTS.md files."""

    def test_skips_paths_with_permission_errors(self, tmp_path: Path) -> None:
        """`OSError` from `Path.resolve()` is caught and the candidate is skipped."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        real_md = project_root / "AGENTS.md"
        real_md.write_text("root instructions")

        original_resolve = Path.resolve

        def patched_resolve(self: Path, *args: object, **kwargs: object) -> Path:
            if self.name == "AGENTS.md" and ".deepagents" in str(self):
                msg = "Permission denied"
                raise PermissionError(msg)
            return original_resolve(self, *args, **kwargs)  # ty: ignore

        with patch.object(Path, "resolve", patched_resolve):
            result = _find_project_agent_md(project_root)

        assert len(result) == 1
        assert result[0].samefile(real_md)
        assert not result[0].is_symlink()

    def test_in_tree_symlink_resolves_to_target(self, tmp_path: Path) -> None:
        """`AGENTS.md -> CLAUDE.md` returns a non-symlink path same-file as target."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        target = project_root / "CLAUDE.md"
        target.write_text("real instructions")

        link = project_root / "AGENTS.md"
        link.symlink_to(target)

        result = _find_project_agent_md(project_root)

        assert len(result) == 1
        # Returned path must be the resolved target — not the symlink — so
        # `FilesystemBackend.download_files` opens the regular file rather
        # than tripping `O_NOFOLLOW` on the link itself.
        assert not result[0].is_symlink()
        assert result[0].samefile(target)
        assert result[0].is_relative_to(project_root.resolve())

    def test_out_of_tree_symlink_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symlink pointing outside project root is skipped with a warning."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        outside_target = tmp_path / "outside.md"
        outside_target.write_text("attacker-controlled")

        link = project_root / "AGENTS.md"
        link.symlink_to(outside_target)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any("outside the project root" in r.getMessage() for r in caplog.records)

    def test_out_of_tree_parent_symlink_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parent directory symlink cannot bypass project-root containment."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        outside_agent_md = outside / "AGENTS.md"
        outside_agent_md.write_text("attacker-controlled")

        (project_root / ".deepagents").symlink_to(outside, target_is_directory=True)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any("outside the project root" in r.getMessage() for r in caplog.records)

    def test_broken_symlink_skipped(self, tmp_path: Path) -> None:
        """Symlink whose target does not exist is skipped without crashing."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        link = project_root / "AGENTS.md"
        link.symlink_to(project_root / "missing.md")

        result = _find_project_agent_md(project_root)

        # `Path.exists()` returns False for broken symlinks, so the candidate
        # is silently skipped — matches pre-existing behavior for absent files.
        assert result == []

    def test_symlink_loop_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symlink loop is skipped with a warning instead of crashing the agent."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        a = project_root / "AGENTS.md"
        b = project_root / "loop.md"
        a.symlink_to(b)
        b.symlink_to(a)

        with caplog.at_level(logging.WARNING, logger="deepagents_code.project_utils"):
            result = _find_project_agent_md(project_root)

        assert result == []
        assert any(
            "Skipping AGENTS.md candidate" in r.getMessage() for r in caplog.records
        )

    def test_regular_file_unchanged_by_resolution(self, tmp_path: Path) -> None:
        """Regular (non-symlink) AGENTS.md returns a non-symlink, in-tree path."""
        project_root = tmp_path / "project"
        project_root.mkdir()

        agent_md = project_root / "AGENTS.md"
        agent_md.write_text("plain file")

        result = _find_project_agent_md(project_root)

        assert len(result) == 1
        assert not result[0].is_symlink()
        assert result[0].samefile(agent_md)
        assert result[0].is_relative_to(project_root.resolve())

    def test_non_canonical_project_root_handled(self, tmp_path: Path) -> None:
        """Non-canonical `project_root` (symlinked ancestor) still locates AGENTS.md.

        Regression test: a symlinked `project_root` previously caused the
        regular-file candidate to fail the absolute-vs-resolved equality check
        and be returned as the canonical target rather than reported as missing.
        Pin behavior so that callers passing an uncanonicalized root (common
        when `Credentials.project_root` originates from an unresolved cwd) still
        find a regular AGENTS.md.
        """
        real_root = tmp_path / "real"
        real_root.mkdir()
        agent_md = real_root / "AGENTS.md"
        agent_md.write_text("instructions")

        link_root = tmp_path / "link"
        link_root.symlink_to(real_root, target_is_directory=True)

        result = _find_project_agent_md(link_root)

        assert len(result) == 1
        assert not result[0].is_symlink()
        assert result[0].samefile(agent_md)
        assert result[0].is_relative_to(link_root.resolve())


class TestUserDeepagentsDir:
    """Test user-level paths derived from `DEEPAGENTS_HOME`."""


class TestGetProjectAgentMdPath:
    """Test `get_project_agent_md_path` integration."""


class TestNewlineShortcut:
    """Tests for newline shortcut labels.

    The label depends on both the platform and whether the attached
    terminal advertises kitty-keyboard-protocol support. Each test
    patches the cached capability probe so the platform-fallback logic
    is exercised in isolation.
    """


class TestValidateModelCapabilities:
    """Tests for model capability validation."""


class TestAgentsAliasDirectories:
    """Tests for `.agents` directory path helpers."""


class TestClaudeSkillsDirs:
    """Tests for `.claude/skills` path helpers."""


class TestDefaultModelSpecAllowlist:
    """Default resolution under an active `models.allowed` policy."""

    def _pin(self, monkeypatch: pytest.MonkeyPatch, policy: ModelConfig) -> None:
        monkeypatch.setattr(
            model_config.ModelConfig,
            "load",
            classmethod(lambda _cls, _path=None: policy),
        )

    def test_empty_policy_error_does_not_invent_a_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deny-all policy reports itself, not a placeholder model name.

        The message reaches the terminal verbatim, and `<default>` reads like a
        bug in dcode rather than a policy the user configured.
        """
        from deepagents_code.config import _get_default_model_spec

        self._pin(
            monkeypatch,
            ModelConfig(allowed_models=(), allowed_models_source="config.toml"),
        )

        with pytest.raises(ModelNotAllowedError) as excinfo:
            _get_default_model_spec()

        assert "<default>" not in str(excinfo.value)
        assert "No model can be used" in str(excinfo.value)

    def test_all_allowed_providers_missing_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every candidate definitively lacking auth is a distinct failure.

        Distinguishable from the bare no-credentials case because the remedy
        differs: only a credential for an allowlisted provider helps.
        """
        from deepagents_code.config import _get_default_model_spec

        self._pin(
            monkeypatch,
            ModelConfig(
                allowed_models=("openai:gpt-5.6-terra", "anthropic:claude-opus-5"),
                allowed_models_source="config.toml",
            ),
        )
        monkeypatch.setattr(
            model_config,
            "get_provider_auth_status",
            lambda provider: model_config.ProviderAuthStatus(
                state=model_config.ProviderAuthState.MISSING,
                source=None,
                provider=provider,
            ),
        )

        with pytest.raises(model_config.NoAllowedModelCredentialsError) as excinfo:
            _get_default_model_spec()

        # Names what would fix it, and stays catchable as the base class so the
        # existing deferred-start recovery keeps working.
        assert "openai:gpt-5.6-terra" in str(excinfo.value)
        assert isinstance(excinfo.value, model_config.NoCredentialsConfiguredError)

    def test_disallowed_stored_default_is_skipped_for_an_allowed_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale `[models].default` outside policy does not win."""
        from deepagents_code.config import _get_default_model_spec

        self._pin(
            monkeypatch,
            ModelConfig(
                default_model="openai:blocked",
                allowed_models=("anthropic:claude-opus-5",),
                allowed_models_source="config.toml",
            ),
        )
        monkeypatch.setattr(
            model_config,
            "get_provider_auth_status",
            lambda provider: model_config.ProviderAuthStatus(
                state=model_config.ProviderAuthState.CONFIGURED,
                source=model_config.ProviderAuthSource.ENV,
                provider=provider,
            ),
        )

        assert _get_default_model_spec() == "anthropic:claude-opus-5"

    def test_wildcard_expands_to_discovered_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `provider:*` wildcard yields the discovered lineup, not just config.

        Built-in providers like `openai` have their models discovered from the
        installed provider package, not from `[models.providers.openai].models`,
        so a wildcard policy with valid credentials must not fall through to
        the "No discoverable models" error.
        """
        from deepagents_code.config import _get_default_model_spec

        self._pin(
            monkeypatch,
            ModelConfig(
                allowed_models=("openai:*",),
                allowed_models_source="config.toml",
            ),
        )
        monkeypatch.setattr(
            model_config,
            "get_discovered_models",
            lambda provider: (
                ["gpt-5.6-terra", "gpt-5.5"] if provider == "openai" else []
            ),
        )
        monkeypatch.setattr(
            model_config,
            "get_provider_auth_status",
            lambda provider: model_config.ProviderAuthStatus(
                state=model_config.ProviderAuthState.CONFIGURED,
                source=model_config.ProviderAuthSource.ENV,
                provider=provider,
            ),
        )

        assert _get_default_model_spec() == "openai:gpt-5.6-terra"

    def test_wildcard_with_no_discovered_models_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wildcard for a provider with no known models has no candidates."""
        from deepagents_code.config import _get_default_model_spec

        self._pin(
            monkeypatch,
            ModelConfig(
                allowed_models=("openai:*",),
                allowed_models_source="config.toml",
            ),
        )
        monkeypatch.setattr(
            model_config,
            "get_discovered_models",
            lambda _provider: [],
        )

        with pytest.raises(
            model_config.NoAllowedModelCredentialsError, match="No discoverable"
        ):
            _get_default_model_spec()


class TestWorkspaceStoredCredentials:
    """Stored auth remains workspace-local during server model construction."""

    @patch("langchain.chat_models.init_chat_model")
    def test_azure_sdk_environment_is_forwarded_explicitly(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Azure SDK-only workspace settings reach the model constructor."""
        import os

        from deepagents_code.config import create_model, use_environment

        mock_model = Mock(profile=None)
        mock_init_chat_model.return_value = mock_model
        monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_AD_TOKEN", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

        with use_environment(
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://workspace.openai.azure.com/",
                "OPENAI_API_VERSION": "2026-01-01",
                "AZURE_OPENAI_AD_TOKEN": "test-token",
            }
        ):
            create_model("azure_openai:deployment")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["azure_endpoint"] == "https://workspace.openai.azure.com/"
        assert "base_url" not in kwargs
        assert kwargs["api_version"] == "2026-01-01"
        assert kwargs["azure_ad_token"] == "test-token"
        assert "OPENAI_API_VERSION" not in os.environ
        assert "AZURE_OPENAI_AD_TOKEN" not in os.environ
        assert "AZURE_OPENAI_ENDPOINT" not in os.environ

    @patch("langchain.chat_models.init_chat_model")
    def test_bedrock_sdk_environment_is_forwarded_explicitly(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AWS workspace settings reach Bedrock without process mutation."""
        import os

        from deepagents_code.config import create_model, use_environment

        mock_model = Mock(profile=None)
        mock_init_chat_model.return_value = mock_model
        aws_environment = {
            "AWS_DEFAULT_REGION": "us-test-1",
            "AWS_DEFAULT_PROFILE": "workspace-profile",
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_SESSION_TOKEN": "test-session-token",
        }
        for name in aws_environment:
            monkeypatch.delenv(name, raising=False)

        with use_environment(aws_environment):
            create_model("bedrock:amazon.test-model")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["region_name"] == "us-test-1"
        assert kwargs["credentials_profile_name"] == "workspace-profile"
        assert kwargs["aws_access_key_id"] == "test-access-key"
        assert kwargs["aws_secret_access_key"] == "test-secret-key"
        assert kwargs["aws_session_token"] == "test-session-token"
        assert all(name not in os.environ for name in aws_environment)

    @patch("langchain.chat_models.init_chat_model")
    def test_stored_key_and_endpoint_do_not_mutate_process_environment(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scoped model creation passes stored auth explicitly."""
        import os

        from deepagents_code.config import create_model, use_environment

        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_key",
            lambda provider: "stored-key" if provider == "openai" else None,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_base_url",
            lambda provider: (
                "https://stored.example/v1" if provider == "openai" else None
            ),
        )
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        with use_environment({}):
            create_model("openai:gpt-5.5")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["api_key"] == "stored-key"
        assert kwargs["base_url"] == "https://stored.example/v1"
        assert "OPENAI_API_KEY" not in os.environ
        assert "OPENAI_BASE_URL" not in os.environ

    @patch("langchain.chat_models.init_chat_model")
    def test_stored_native_key_clears_workspace_endpoint(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored native key is not sent to a workspace gateway URL."""
        from deepagents_code.config import create_model, use_environment

        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_key",
            lambda provider: "stored-key" if provider == "openai" else None,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_base_url",
            lambda _provider: None,
        )

        with use_environment({"OPENAI_BASE_URL": "https://workspace.example/v1"}):
            create_model("openai:gpt-5.5")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["api_key"] == "stored-key"
        assert "base_url" not in kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_explicit_key_does_not_use_stored_endpoint(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller-supplied key is not paired with another key's endpoint."""
        from deepagents_code.config import create_model, use_environment

        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_key",
            lambda provider: "stored-key" if provider == "openai" else None,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_base_url",
            lambda _provider: "https://stored.example/v1",
        )
        with use_environment({}):
            create_model(
                "openai:gpt-5.5",
                extra_kwargs={"api_key": "caller-key"},
            )

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["api_key"] == "caller-key"
        assert "base_url" not in kwargs

    @patch("langchain.chat_models.init_chat_model")
    def test_stored_anthropic_key_clears_endpoint_and_headers(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A native Anthropic key gets explicit native transport settings."""
        from deepagents_code.config import create_model, use_environment

        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_key",
            lambda provider: "stored-key" if provider == "anthropic" else None,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_base_url",
            lambda _provider: None,
        )
        with use_environment(
            {
                "ANTHROPIC_BASE_URL": "https://workspace.example/v1",
                "ANTHROPIC_CUSTOM_HEADERS": "X-Api-Key: gateway-key",
            }
        ):
            create_model("anthropic:claude-sonnet-4-6")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["api_key"] == "stored-key"
        assert kwargs["base_url"] == "https://api.anthropic.com"
        assert kwargs["default_headers"] == {}

    def test_bare_claude_provider_uses_workspace_credentials(self) -> None:
        """Bare model inference reads the active workspace snapshot."""
        from deepagents_code.config import detect_provider, use_environment

        with use_environment(
            {
                "GOOGLE_CLOUD_PROJECT": "workspace-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        ):
            assert detect_provider("claude-sonnet-4-6") == "google_anthropic_vertex"

    @patch("langchain.chat_models.init_chat_model")
    def test_vertex_uses_active_workspace_project_and_not_api_key(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Vertex project settings come from the workspace snapshot."""
        from deepagents_code.config import create_model, use_environment

        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model
        monkeypatch.setattr(
            "deepagents_code.model_config.auth_store.get_stored_key",
            lambda _provider: None,
        )

        with use_environment(
            {
                "GOOGLE_CLOUD_PROJECT": "workspace-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        ):
            create_model("google_anthropic_vertex:claude-sonnet-4-6")

        kwargs = mock_init_chat_model.call_args.kwargs
        assert kwargs["project"] == "workspace-project"
        assert kwargs["location"] == "us-central1"
        assert "api_key" not in kwargs


class TestCreateModelProfileExtraction:
    """Tests for profile extraction in create_model.

    These tests verify that create_model correctly extracts the context_limit
    from the model's profile attribute. We mock init_chat_model since create_model
    now uses it internally.
    """

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_handles_missing_profile_gracefully(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Test that missing profile attribute leaves context_limit as None."""
        mock_model = Mock(spec=["invoke"])  # No profile attribute
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")
        assert result.context_limit is None


class TestCreateModelSplitCredentialWiring:
    """`create_model` wires the split-credential diagnostic in correctly."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @pytest.fixture(autouse=True)
    def _isolate_openai_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "OPENAI_API_KEY",
            "DEEPAGENTS_CODE_OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE",
            "DEEPAGENTS_CODE_OPENAI_BASE_URL",
            "DEEPAGENTS_CODE_OPENAI_API_BASE",
        ):
            monkeypatch.delenv(var, raising=False)

    @patch("langchain.chat_models.init_chat_model")
    def test_create_model_emits_split_credential_warning(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A prefixed key + plain base URL surfaces the DEBUG diagnostic.

        Guards the call site itself: `TestSplitCredentialSource` only exercises
        the helper in isolation, so without this a dropped call would go unnoticed.
        """
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        monkeypatch.setenv("DEEPAGENTS_CODE_OPENAI_API_KEY", "sk-secret-value")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

        with caplog.at_level(logging.DEBUG, logger="deepagents_code.model_config"):
            create_model("openai:gpt-5.5")

        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "DEEPAGENTS_CODE_OPENAI_API_KEY" in m and "OPENAI_BASE_URL" in m
            for m in messages
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_diagnostic_runs_before_apply_stored_credentials(
        self,
        mock_init_chat_model: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The diagnostic must observe raw env intent, i.e. run before the bridge.

        `apply_stored_credentials` rewrites the unprefixed base-URL env vars, so
        the ordering claimed by the call-site comment is load-bearing. Pin it by
        asserting the relative call order, which a reorder/removal would break.
        """
        mock_model = Mock()
        mock_model.profile = {"max_input_tokens": 128000, "tool_calling": True}
        mock_init_chat_model.return_value = mock_model

        manager = Mock()
        monkeypatch.setattr(
            "deepagents_code.model_config.warn_on_split_credential_source",
            manager.warn,
        )
        monkeypatch.setattr(
            "deepagents_code.model_config.apply_stored_credentials",
            manager.apply,
        )

        create_model("openai:gpt-5.5")

        ordered = [name for name, _args, _kwargs in manager.mock_calls]
        assert ordered == ["warn", "apply"]


class TestModelResultApplyToRuntimeState:
    """Tests for `ModelResult.apply_to_runtime_state` propagation."""

    def test_propagates_unsupported_modalities(self) -> None:
        """Test model results update all process-wide runtime metadata."""
        model_result = ModelResult(
            model=Mock(),
            model_name="deepseek-r1",
            provider="deepseek",
            context_limit=64000,
            unsupported_modalities=frozenset({"image", "audio"}),
        )
        # The method writes four fields to process-global runtime state;
        # restore all of them or the values leak into every later test.
        original_name = runtime_state.model_name
        original_provider = runtime_state.model_provider
        original_limit = runtime_state.model_context_limit
        original_modalities = runtime_state.model_unsupported_modalities
        try:
            model_result.apply_to_runtime_state()
            expected = frozenset({"image", "audio"})
            assert runtime_state.model_name == "deepseek-r1"
            assert runtime_state.model_provider == "deepseek"
            assert runtime_state.model_context_limit == 64000
            assert runtime_state.model_unsupported_modalities == expected
        finally:
            runtime_state.model_name = original_name
            runtime_state.model_provider = original_provider
            runtime_state.model_context_limit = original_limit
            runtime_state.model_unsupported_modalities = original_modalities


class TestRetriesConfig:
    """Tests for `[retries]` config.toml support."""

    def test_read_retries_returns_none_when_unreadable(self, tmp_path: Path) -> None:
        """Unreadable config returns `None` with a warning."""
        config_path = tmp_path / "config.toml"

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.object(Path, "open", side_effect=PermissionError("denied")),
        ):
            config = _read_retry_config()

        assert any("Could not read retries config" in text for text in config.warnings)


class TestResolveModelRetries:
    """The retry resolver resolves the model-node retry budget."""

    @staticmethod
    def _resolve_retries(provider: str, *, cli_max_retries: int | None = None) -> int:
        """Resolve a budget the way `create_model` does.

        Returns:
            The effective retry count.
        """
        return _resolve_model_retries_from_section(
            _read_retry_config(), provider, cli_max_retries
        )


class TestCreateModelMaxRetries:
    """`create_model` keeps CLI retries separate from provider kwargs."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )


class TestCreateModelProfileOverrides:
    """Tests for profile overrides from config.toml in create_model."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_override_logs_warning_on_frozen_model(
        self,
        mock_init_chat_model: Mock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Graceful warning when model rejects attribute assignment."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.anthropic.profile]
max_input_tokens = 4096
""")
        mock_model = Mock()
        # Make .profile read return a dict but assignment raises
        type(mock_model).profile = property(
            fget=lambda _: {"max_input_tokens": 200000},
            fset=lambda _, __: (_ for _ in ()).throw(AttributeError("frozen")),
        )
        mock_init_chat_model.return_value = mock_model

        clear_caches()
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            result = create_model("anthropic:claude-sonnet-4-5")

        assert any(
            "Could not apply" in r.message and "profile overrides" in r.message
            for r in caplog.records
        )
        # Falls back to original profile extraction
        assert result.context_limit == 200000


class TestCreateModelCLIProfileOverrides:
    """Tests for CLI --profile-override in create_model."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )


class TestParseShellAllowList:
    """Test parsing shell allow-list strings."""

    def test_none_input_returns_none(self) -> None:
        """Test that None input returns None."""
        result = parse_shell_allow_list(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        result = parse_shell_allow_list("")
        assert result is None

    def test_recommended_only(self) -> None:
        """Test that 'recommended' returns the full recommended list."""
        result = parse_shell_allow_list("recommended")
        assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_recommended_case_insensitive(self) -> None:
        """Test that 'RECOMMENDED', 'Recommended', etc. all work."""
        for variant in ["RECOMMENDED", "Recommended", "ReCoMmEnDeD", "  recommended  "]:
            result = parse_shell_allow_list(variant)
            assert result == list(RECOMMENDED_SAFE_SHELL_COMMANDS)

    def test_custom_commands_only(self) -> None:
        """Test parsing custom commands without 'recommended'."""
        result = parse_shell_allow_list("ls,cat,grep")
        assert result == ["ls", "cat", "grep"]

    def test_custom_commands_with_whitespace(self) -> None:
        """Test parsing custom commands with whitespace."""
        result = parse_shell_allow_list("ls , cat , grep")
        assert result == ["ls", "cat", "grep"]

    def test_recommended_merged_with_custom_commands(self) -> None:
        """Test that 'recommended' in list merges with custom commands."""
        result = parse_shell_allow_list("recommended,mycmd,myothercmd")
        expected = [*list(RECOMMENDED_SAFE_SHELL_COMMANDS), "mycmd", "myothercmd"]
        assert result == expected

    def test_custom_commands_before_recommended(self) -> None:
        """Test custom commands before 'recommended' keyword."""
        result = parse_shell_allow_list("mycmd,recommended,myothercmd")
        # mycmd first, then all recommended, then myothercmd
        expected = ["mycmd", *list(RECOMMENDED_SAFE_SHELL_COMMANDS), "myothercmd"]
        assert result == expected

    def test_duplicate_removal(self) -> None:
        """Test that duplicates are removed while preserving order."""
        result = parse_shell_allow_list("ls,cat,ls,grep,cat")
        assert result == ["ls", "cat", "grep"]

    def test_duplicate_removal_with_recommended(self) -> None:
        """Test that duplicates from recommended are removed."""
        # 'ls' is in RECOMMENDED_SAFE_SHELL_COMMANDS
        result = parse_shell_allow_list("ls,recommended,mycmd")
        # Should have ls once (first occurrence), then all recommended commands
        # except ls (since it's already in), then mycmd
        assert result is not None
        assert result[0] == "ls"
        # ls should not appear again
        assert result.count("ls") == 1
        # mycmd should appear once at the end
        assert result[-1] == "mycmd"
        # Total should be: 1 (ls) + len(recommended) - 1 (duplicate ls) + 1 (mycmd)
        # Which simplifies to: len(recommended) + 1
        assert len(result) == len(RECOMMENDED_SAFE_SHELL_COMMANDS) + 1

    def test_all_returns_sentinel(self) -> None:
        """Test that 'all' returns SHELL_ALLOW_ALL sentinel."""
        result = parse_shell_allow_list("all")
        assert result is SHELL_ALLOW_ALL

    def test_all_case_insensitive(self) -> None:
        """Test that 'ALL', 'All', etc. all return sentinel."""
        for variant in ["ALL", "All", "aLl", "  all  "]:
            result = parse_shell_allow_list(variant)
            assert result is SHELL_ALLOW_ALL

    def test_all_mixed_with_commands_raises(self) -> None:
        """Combining 'all' with other commands should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot combine 'all'"):
            parse_shell_allow_list("all,ls")

    def test_all_mixed_case_insensitive_raises(self) -> None:
        """Combining 'ALL' with other commands should also raise."""
        with pytest.raises(ValueError, match="Cannot combine 'all'"):
            parse_shell_allow_list("ls,ALL,cat")

    def test_empty_commands_ignored(self) -> None:
        """Test that empty strings from split are ignored."""
        result = parse_shell_allow_list("ls,,cat,,,grep,")
        assert result == ["ls", "cat", "grep"]


class TestGetLangsmithProjectName:
    """Tests for get_langsmith_project_name()."""

    @pytest.mark.parametrize(
        "flag",
        ["LANGSMITH_TRACING", "DEEPAGENTS_CODE_LANGSMITH_TRACING"],
    )
    def test_returns_none_when_tracing_explicitly_disabled(self, flag: str) -> None:
        """Recognized tracing opt-outs should disable project resolution."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_PROJECT": "configured-project",
            flag: "false",
        }
        with patch.dict("os.environ", env, clear=True):
            assert get_langsmith_project_name() is None

    def test_returns_project_from_credentials(self) -> None:
        """Should prefer `credentials.deepagents_langchain_project`."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config._credentials_instance") as mock_credentials,
        ):
            mock_credentials.deepagents_langchain_project = "credentials-project"
            assert get_langsmith_project_name() == "credentials-project"

    def test_falls_back_to_env_project(self) -> None:
        """Should fall back to LANGSMITH_PROJECT env var."""
        env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "env-project",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config._credentials_instance") as mock_credentials,
        ):
            mock_credentials.deepagents_langchain_project = None
            assert get_langsmith_project_name() == "env-project"

    def test_accepts_langchain_api_key(self) -> None:
        """Should accept LANGCHAIN_API_KEY as alternative to LANGSMITH_API_KEY."""
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = {
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config._credentials_instance") as mock_credentials,
        ):
            mock_credentials.deepagents_langchain_project = None
            assert get_langsmith_project_name() == LANGSMITH_PROJECT_DEFAULT

    def test_agrees_with_config_manifest_resolution(self) -> None:
        """`get_langsmith_project_name` and the resolver agree on the project.

        The `fallback_env_vars` mechanism exists so `config`/`config get` report
        the project agent traces actually route to. This pins that parity for
        the bare-env and unset cases, catching future drift between the two
        resolution paths.
        """
        from deepagents_code.config_manifest import (
            LANGSMITH_PROJECT_DEFAULT,
            get_option,
        )
        from deepagents_code.configuration.resolver import resolver_from_snapshots
        from deepagents_code.configuration.types import (
            ProviderHealth,
            ProviderStatus,
            TomlSnapshot,
        )

        opt = get_option("tracing.langsmith_project")
        assert opt is not None

        def resolve() -> object:
            return (
                resolver_from_snapshots(
                    managed=TomlSnapshot(
                        {},
                        ProviderStatus("managed config", None, ProviderHealth.OK),
                    ),
                    user=TomlSnapshot(
                        {},
                        ProviderStatus("config.toml", None, ProviderHealth.OK),
                    ),
                )
                .get(opt)
                .value
            )

        # Bare `LANGSMITH_PROJECT` set, no prefixed override, no credential value.
        bare_env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "parity-bare",
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        }
        with (
            patch.dict("os.environ", bare_env, clear=False),
            patch("deepagents_code.config._credentials_instance") as mock_credentials,
        ):
            mock_credentials.deepagents_langchain_project = None
            manifest_value = resolve()
            assert get_langsmith_project_name() == manifest_value == "parity-bare"

        # Nothing configured: both fall back to the shared default.
        default_env = {
            "LANGSMITH_API_KEY": "lsv2_test",
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": "",
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        }
        with (
            patch.dict("os.environ", default_env, clear=False),
            patch("deepagents_code.config._credentials_instance") as mock_credentials,
        ):
            mock_credentials.deepagents_langchain_project = None
            manifest_value = resolve()
            assert (
                get_langsmith_project_name()
                == manifest_value
                == LANGSMITH_PROJECT_DEFAULT
            )


class TestLangsmithKeyShadowedByEmptyOverride:
    """Tests for langsmith_key_shadowed_by_empty_override()."""

    @pytest.fixture(autouse=True)
    def _clear_tracing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start each test from a clean slate for the four tracing key vars.

        Every case below sets only the vars it cares about; clearing the rest
        up front keeps results independent of whatever the test runner happens
        to have exported (e.g. a developer's own empty override).
        """
        for var in (
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
            "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def fake_state_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect the credential store into a temp directory."""
        state = tmp_path / ".state"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
        return state

    def test_returns_none_without_empty_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No empty prefixed override means nothing is being shadowed."""
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_returns_none_when_override_shadows_nothing(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty override with no underlying key is not a shadow."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_detects_shadowed_canonical_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty override shadowing a canonical env key is reported."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_detects_shadowed_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty override shadowing a `/auth`-stored key is reported."""
        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_langchain_override_does_not_consult_the_stored_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The stored-key bridge is `LANGSMITH`-only, so `LANGCHAIN` ignores it.

        `/auth` only bridges its stored key onto `LANGSMITH_API_KEY`, never
        `LANGCHAIN_API_KEY`. An empty `LANGCHAIN` override with no canonical
        `LANGCHAIN_API_KEY` therefore shadows nothing even when a key is stored,
        so no shadow is reported. Pins the asymmetry against a future change that
        wrongly makes the `LANGCHAIN` path consult the store.
        """
        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_returns_none_when_override_carries_a_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty prefixed override resolves normally, so no shadow."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_detects_shadowed_langchain_canonical_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy `LANGCHAIN_API_KEY` override path is reported too."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
        )

    def test_reports_the_override_that_actually_shadows_the_key(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With both overrides empty, name the one hiding the only real key.

        The `LANGSMITH` override is empty but shadows nothing (no canonical
        value, no stored key); only `LANGCHAIN_API_KEY` carries a key, so
        unsetting the `LANGCHAIN` override -- not the `LANGSMITH` one -- is what
        restores tracing. The hint must name the `LANGCHAIN` override.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
        )

    def test_prefers_langsmith_when_both_overrides_shadow_a_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both overrides genuinely shadow a key, `LANGSMITH` wins."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
            shadowing_var="DEEPAGENTS_CODE_LANGSMITH_API_KEY"
        )

    def test_ignores_empty_override_when_a_key_already_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty override is not reported when a key resolves anyway.

        `LANGSMITH_API_KEY` resolves fine; the empty `LANGCHAIN` override hides
        no key (no canonical `LANGCHAIN_API_KEY`). Tracing may still be off for
        an unrelated reason (e.g. a missing tracing flag), but this override is
        not the cause, so the generic hint -- not a false shadow claim -- is
        correct.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_ignores_lower_precedence_override_when_langsmith_key_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolving `LANGSMITH` key silences a genuine `LANGCHAIN` shadow.

        `LANGSMITH_API_KEY` resolves and wins under precedence, so the effective
        key is present and unsetting the empty `LANGCHAIN` override (which does
        shadow `LANGCHAIN_API_KEY`) would change nothing. Reporting it would send
        the user to unset the wrong variable, so nothing is reported.
        """
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_langsmith")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_langchain")
        assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult()

    def test_reports_store_unreadable_when_no_other_shadow_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A corrupt store with no other shadow surfaces `store_unreadable`.

        The warning must carry the underlying exception text (not a static
        guess) so logs point at the real fault.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        with (
            patch(
                "deepagents_code.auth_store.get_stored_key",
                side_effect=RuntimeError("bad json at line 3"),
            ),
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
                store_unreadable=True
            )
        messages = [r.getMessage() for r in caplog.records]
        assert any("empty-override shadow" in m for m in messages)
        assert any("bad json at line 3" in m for m in messages)

    def test_unreadable_store_does_not_abort_scan_of_later_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store error on `LANGSMITH` must not stop the `LANGCHAIN` check.

        `LANGSMITH`'s stored-key read raises, but an empty `LANGCHAIN` override
        genuinely shadows a canonical `LANGCHAIN_API_KEY`. That concrete shadow
        is the actionable answer and must win over the store uncertainty, which
        proves the loop continued past the exception rather than bailing.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lsv2_test")
        with patch(
            "deepagents_code.auth_store.get_stored_key",
            side_effect=RuntimeError("corrupt"),
        ):
            assert langsmith_key_shadowed_by_empty_override() == LangsmithShadowResult(
                shadowing_var="DEEPAGENTS_CODE_LANGCHAIN_API_KEY"
            )


class TestLangsmithSecretRedaction:
    """Tests for LangSmith trace secret redaction configuration."""

    def test_redaction_can_be_disabled_by_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The redaction env var can opt out for local debugging."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", "false")
        with patch("deepagents_code.config_manifest.load_config_toml", return_value={}):
            assert is_langsmith_redaction_enabled() is False

    def test_configures_langsmith_client_with_secret_anonymizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Active tracing installs a client whose anonymizer scrubs secrets."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        # Exercise the real `create_secret_anonymizer` (network-free) so the test
        # fails if the installed anonymizer does not actually redact secrets.
        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert kwargs["api_key"] == "lsv2_test"
        assert "api_url" not in kwargs
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl"
        redacted = str(kwargs["anonymizer"]([{"text": f"key={secret}"}]))
        assert secret not in redacted
        assert "[SECRET_DETECTED]" in redacted

    def test_workspace_environment_configures_redaction(self) -> None:
        """Workspace-only tracing settings configure the SDK client."""
        from deepagents_code.config import use_environment

        client = object()
        with (
            use_environment(
                {
                    "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "lsv2_workspace",
                    "DEEPAGENTS_CODE_LANGSMITH_TRACING": "true",
                }
            ),
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        assert client_cls.call_args.kwargs["api_key"] == "lsv2_workspace"

    def test_skips_client_configuration_when_redaction_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opting out leaves the LangSmith client untouched."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", "false")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_skips_when_tracing_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redaction is skipped when tracing is inactive, even if credentialed."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_skips_when_no_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redaction is skipped when tracing is active but uncredentialed."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            # No env key is set; ignore any LangSmith profile on the dev machine
            # so the no-credentials branch is exercised hermetically.
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_falls_back_to_langchain_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The legacy LANGCHAIN_API_KEY is used when LANGSMITH_API_KEY is absent."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGCHAIN_API_KEY", "lsv2_legacy")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert kwargs["api_key"] == "lsv2_legacy"

    def test_forwards_custom_endpoint_as_api_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A custom tracing endpoint is forwarded to the client as `api_url`."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://eu.smith.example.com")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert kwargs["api_url"] == "https://eu.smith.example.com"

    def test_does_not_forward_default_us_endpoint_as_api_url(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The SDK default US SaaS URL is not forwarded as a custom `api_url`."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", LANGSMITH_US_ENDPOINT)

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert "api_url" not in kwargs

    def test_default_us_env_does_not_forward_profile_custom_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A populated default env endpoint wins over a profile custom `api_url`.

        The SDK resolves `env_api_url or profile_config.api_url`, so an explicit
        default US env value must not fall through to the profile when building
        the redacting client.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", LANGSMITH_US_ENDPOINT)

        profile = Mock()
        profile.api_url = "https://eu.smith.example.com"
        profile.api_key = None
        profile.oauth_access_token = None
        profile.oauth_refresh_token = None

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._load_langsmith_profile_config",
                return_value=profile,
            ),
            patch("langsmith.Client", return_value=object()) as client_cls,
            patch("langsmith.configure"),
        ):
            assert configure_langsmith_secret_redaction() is True

        _, kwargs = client_cls.call_args
        assert "api_url" not in kwargs

    def test_skips_keyless_default_us_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyless default US endpoint is not treated as an upload target."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", f"{LANGSMITH_US_ENDPOINT}/")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_configures_client_for_keyless_custom_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keyless custom endpoints still get the secret anonymizer installed."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://localhost:1984")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert kwargs["api_url"] == "http://localhost:1984"
        assert "anonymizer" in kwargs

    def test_configures_client_for_runs_endpoints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replica trace endpoints still get the secret anonymizer installed."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv(
            "LANGSMITH_RUNS_ENDPOINTS",
            '[{"api_url":"https://replica.example.com","api_key":"lsv2_replica"}]',
        )

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert "api_url" not in kwargs
        assert "anonymizer" in kwargs

    @pytest.mark.parametrize(
        "value",
        ["[]", '[{"api_url":"https://replica.example.com"}]', "not json"],
    )
    def test_skips_invalid_runs_endpoints(
        self,
        value: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only valid replica endpoint configs count as upload targets."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_RUNS_ENDPOINTS", value)

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=False,
            ),
            patch("langsmith.Client") as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        client_cls.assert_not_called()
        configure.assert_not_called()

    def test_fails_closed_by_disabling_tracing_on_setup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redaction setup failure disables tracing to avoid leaking secrets."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", side_effect=RuntimeError("boom")),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        configure.assert_called_once_with(enabled=False)

    def test_configures_client_with_profile_only_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Profile-only credentials (no env key) still install the anonymizer."""
        client = object()
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            # Credentials come only from an active LangSmith profile, not the env.
            # No endpoint either, so the profile credentials are the sole reason
            # the upload gate passes. The SDK client self-resolves that profile
            # auth when no key is forwarded.
            patch(
                "deepagents_code.config._has_langsmith_profile_credentials",
                return_value=True,
            ),
            patch("deepagents_code.config._tracing_endpoint_from", return_value=None),
            patch("langsmith.Client", return_value=client) as client_cls,
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True

        configure.assert_called_once_with(client=client)
        _, kwargs = client_cls.call_args
        assert "api_key" not in kwargs
        assert "anonymizer" in kwargs

    def test_redaction_can_be_disabled_by_toml(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A `[tracing] langsmith_redact = false` in config.toml opts out."""
        monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", raising=False)
        (tmp_path / "config.toml").write_text(
            "[tracing]\nlangsmith_redact = false\n", encoding="utf-8"
        )
        assert is_langsmith_redaction_enabled() is False

    def test_env_redaction_toggle_overrides_toml(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """The redaction env var takes precedence over a conflicting config.toml."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_REDACT", "true")
        (tmp_path / "config.toml").write_text(
            "[tracing]\nlangsmith_redact = false\n", encoding="utf-8"
        )
        assert is_langsmith_redaction_enabled() is True

    def test_fail_closed_clears_env_when_sdk_disable_also_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the SDK cannot disable tracing, all enable env vars are cleared.

        This is the worst-case fail-closed path: redaction setup raised and the
        SDK's `configure(enabled=False)` raised too, so the only remaining
        barrier is removing every env var the LangChain tracer falls back to —
        both the canonical names and their `DEEPAGENTS_CODE_`-prefixed forms.
        """
        import os

        from deepagents_code.config import _TRACING_ENABLE_ENV_VARS
        from deepagents_code.model_config import _ENV_PREFIX

        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
        enable_vars = [
            *_TRACING_ENABLE_ENV_VARS,
            *(f"{_ENV_PREFIX}{var}" for var in _TRACING_ENABLE_ENV_VARS),
        ]
        for var in enable_vars:
            monkeypatch.setenv(var, "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", side_effect=RuntimeError("boom")),
            patch("langsmith.configure", side_effect=RuntimeError("nope")),
        ):
            assert configure_langsmith_secret_redaction() is False

        for var in enable_vars:
            assert var not in os.environ, f"{var} should have been cleared"

    def test_fails_closed_when_redaction_toggle_lookup_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unexpected error from the redaction-toggle lookup fails closed.

        `is_langsmith_redaction_enabled()` runs inside the fail-closed boundary,
        so even an unexpected exception there disables tracing rather than
        escaping the function and leaving tracing live but unredacted.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch(
                "deepagents_code.config.is_langsmith_redaction_enabled",
                side_effect=RuntimeError("boom"),
            ),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is False

        configure.assert_called_once_with(enabled=False)

    def test_reconfigures_on_each_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each call reinstalls the redacting client (no fail-open caching)."""
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_test")
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")

        with (
            patch("deepagents_code.config_manifest.load_config_toml", return_value={}),
            patch("langsmith.Client", return_value=object()),
            patch("langsmith.configure") as configure,
        ):
            assert configure_langsmith_secret_redaction() is True
            assert configure_langsmith_secret_redaction() is True

        assert configure.call_count == 2


class TestDisableOrphanedTracing:
    """Tests for _disable_orphaned_tracing()."""

    _ALL_TRACING_VARS = (
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_RUNS_ENDPOINTS",
        "LANGCHAIN_RUNS_ENDPOINTS",
        "LANGSMITH_CONFIG_FILE",
        "LANGSMITH_PROFILE",
    )

    def _clean_env(self) -> dict[str, str]:
        consume_orphaned_tracing_disabled_notice()
        env = dict.fromkeys(self._ALL_TRACING_VARS, "")
        env["LANGSMITH_CONFIG_FILE"] = "/__deepagents_missing_langsmith_config__.json"
        return env

    def test_disables_tracing_when_no_key(self) -> None:
        """Tracing flag on with empty key should be turned off."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert consume_orphaned_tracing_disabled_notice() is not None
            # One-shot: the notice clears on read, so a second read is empty.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_notice_mentions_langsmith_auth_login_when_cli_available(self) -> None:
        """The startup notice gives the CLI login command only when available."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.shutil.which", return_value="/bin/langsmith"),
        ):
            _disable_orphaned_tracing()

        notice = consume_orphaned_tracing_disabled_notice()
        assert notice is not None
        assert "langsmith auth login" in notice

    def test_notice_omits_langsmith_auth_login_when_cli_unavailable(self) -> None:
        """The startup notice avoids unavailable CLI commands."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        with (
            patch.dict("os.environ", env, clear=False),
            patch("deepagents_code.config.shutil.which", return_value=None),
        ):
            _disable_orphaned_tracing()

        notice = consume_orphaned_tracing_disabled_notice()
        assert notice is not None
        assert "langsmith auth login" not in notice
        assert "LANGSMITH_API_KEY" in notice

    def test_preserves_tracing_when_custom_endpoint_set(self) -> None:
        """A custom endpoint (self-hosted/proxied) is trusted even without a key.

        Keyless ingestion is valid against a self-hosted LangSmith, so an
        explicitly configured endpoint must not trip the orphaned-tracing guard.
        """
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_ENDPOINT"] = "http://localhost:1984"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # Nothing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_disables_tracing_when_only_default_us_endpoint_set(self) -> None:
        """SDK default US endpoint alone is not a keyless custom upload target."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_ENDPOINT"] = LANGSMITH_US_ENDPOINT
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert consume_orphaned_tracing_disabled_notice() is not None

    def test_preserves_tracing_when_runs_endpoints_set(self) -> None:
        """Replica endpoints are trusted upload targets even without a top-level key."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_RUNS_ENDPOINTS"] = (
            '[{"api_url":"https://replica.example.com","api_key":"lsv2_replica"}]'
        )
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_preserves_tracing_when_profile_custom_endpoint_set(
        self, tmp_path: Path
    ) -> None:
        """Profile api_url is a custom endpoint and is trusted without a key."""
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{"default":{"api_url":"http://localhost:1984"}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # Nothing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_disables_tracing_when_profile_only_has_default_us_endpoint(
        self, tmp_path: Path
    ) -> None:
        """Profile default US api_url alone does not count as a custom endpoint."""
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            f'"profiles":{{"default":{{"api_url":"{LANGSMITH_US_ENDPOINT}"}}}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert consume_orphaned_tracing_disabled_notice() is not None

    def test_disables_tracing_when_default_us_env_overrides_profile_custom(
        self, tmp_path: Path
    ) -> None:
        """Populated default US env wins over profile custom for orphaned disable.

        The SDK would still target keyless US in that case, so the profile's
        custom api_url must not be trusted as a keyless upload target.
        """
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{"default":{"api_url":"http://localhost:1984"}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_ENDPOINT"] = LANGSMITH_US_ENDPOINT
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert consume_orphaned_tracing_disabled_notice() is not None

    def test_preserves_tracing_when_key_present(self) -> None:
        """Tracing stays enabled when a usable API key is set."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_API_KEY"] = "lsv2_test"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
            # No tracing was disabled, so no startup notice should be staged.
            assert consume_orphaned_tracing_disabled_notice() is None

    def test_accepts_langchain_api_key(self) -> None:
        """LANGCHAIN_API_KEY also counts as a usable key."""
        env = self._clean_env()
        env["LANGSMITH_TRACING"] = "true"
        env["LANGCHAIN_API_KEY"] = "lsv2_test"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_preserves_tracing_when_profile_api_key_present(
        self, tmp_path: Path
    ) -> None:
        """LangSmith profile API keys count as usable credentials."""
        config = tmp_path / "config.json"
        config.write_text(
            '{"current_profile":"default","profiles":{"default":{"api_key":"lsv2_profile"}}}',
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"

    def test_preserves_tracing_when_profile_oauth_present(self, tmp_path: Path) -> None:
        """LangSmith profile OAuth credentials count as usable credentials."""
        config = tmp_path / "config.json"
        config.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{"default":{"oauth":{"refresh_token":"refresh"}}}'
            "}",
            encoding="utf-8",
        )
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "true"

    def test_noop_when_tracing_disabled(self) -> None:
        """Does nothing when no tracing flag is enabled."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "false"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_disables_all_set_tracing_flags(self) -> None:
        """Every set tracing flag is turned off, not just one."""
        env = self._clean_env()
        env["LANGCHAIN_TRACING_V2"] = "true"
        env["LANGSMITH_TRACING"] = "1"
        with patch.dict("os.environ", env, clear=False):
            _disable_orphaned_tracing()
            import os

            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
            assert os.environ["LANGSMITH_TRACING"] == "false"


class TestApplyStoredLangSmithTracing:
    """Tests for _apply_stored_langsmith_tracing()."""

    @pytest.fixture
    def fake_state_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect the credential store into a temp directory."""
        state = tmp_path / ".state"
        monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state)
        return state

    def test_enables_tracing_when_key_stored(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored LangSmith key turns tracing on by default."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "true"

    def test_respects_explicit_opt_out(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit falsy tracing flag is honored as a temporary opt-out."""
        import os

        from deepagents_code import auth_store

        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "false"

    def test_scoped_opt_out_disables_sibling_tracing_flags(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The documented scoped opt-out wins over other truthy tracing flags."""
        import os

        from deepagents_code import auth_store

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

    def test_leaves_explicit_enable_untouched(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An already-truthy tracing flag is left as-is."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert "LANGSMITH_TRACING" not in os.environ

    def test_applies_stored_custom_project(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored custom project is applied when none is already set."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGSMITH_PROJECT"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_PROJECT"] == "my-app"

    def test_stored_project_does_not_override_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit LANGSMITH_PROJECT wins over the stored project."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_PROJECT"] == "from-env"

    def test_replace_project_applies_stored_project_over_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save applies the latest stored project."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        auth_store.set_stored_key("langsmith", "lsv2_test", project="my-app")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert os.environ["LANGSMITH_PROJECT"] == "my-app"

    def test_replace_project_clears_existing_env_when_project_removed(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save clears the old project when the field is blank."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert "LANGSMITH_PROJECT" not in os.environ

    def test_immediate_auth_clear_restores_default_project(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Clearing `/auth` project updates the active traced session."""
        import os

        from deepagents_code import auth_store
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        redaction_env: list[dict[str, str]] = []

        def capture_redaction_env() -> bool:
            redaction_env.append(dict(os.environ))
            return True

        monkeypatch.setenv("LANGSMITH_PROJECT", "old-project")
        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        auth_store.set_stored_key("langsmith", "lsv2_test")
        with patch(
            "deepagents_code.config.configure_langsmith_secret_redaction",
            side_effect=capture_redaction_env,
        ) as configure_redaction:
            apply_stored_langsmith_auth(replace_project=True)
        assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        assert os.environ["LANGSMITH_TRACING"] == "true"
        configure_redaction.assert_called_once_with()
        assert redaction_env[0]["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        assert redaction_env[0]["LANGSMITH_TRACING"] == "true"

    def test_corrupt_store_warns_and_leaves_env_untouched(
        self,
        fake_state_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A corrupt credential file is logged and tracing is left untouched."""
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        fake_state_dir.mkdir(parents=True, exist_ok=True)
        (fake_state_dir / "auth.json").write_text("{ not json", encoding="utf-8")
        with caplog.at_level("WARNING", logger="deepagents_code.config"):
            _apply_stored_langsmith_tracing()
        assert "LANGSMITH_TRACING" not in os.environ
        assert any("may be corrupt" in r.getMessage() for r in caplog.records)

    def test_applies_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored endpoint is applied to LANGSMITH_ENDPOINT at startup."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_stored_endpoint_does_not_override_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An explicit LANGSMITH_ENDPOINT wins over the stored endpoint at startup."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://from-env.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://from-env.example.com"

    def test_no_stored_endpoint_leaves_env_endpoint_untouched(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored key without an endpoint never clears an existing env endpoint."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://self-hosted.example.com")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://self-hosted.example.com"

    def test_alternate_env_endpoint_blocks_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lone LANGCHAIN_ENDPOINT (the alternate) also wins over a stored one."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://alt-env.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        # The stored endpoint must not be applied, and the alternate is preserved.
        assert "LANGSMITH_ENDPOINT" not in os.environ
        assert os.environ["LANGCHAIN_ENDPOINT"] == "https://alt-env.example.com"

    def test_replace_applies_stored_endpoint_over_existing_env(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save replaces the endpoint and clears the alternate."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://old.example.com")
        monkeypatch.setenv("LANGCHAIN_ENDPOINT", "https://old-alt.example.com")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing(replace_project=True)
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_replace_clears_endpoint_when_blank(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Immediate `/auth` save clears the endpoint when no base URL is stored."""
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://old.example.com")
        auth_store.set_stored_key("langsmith", "lsv2_test")
        _apply_stored_langsmith_tracing(replace_project=True)
        assert "LANGSMITH_ENDPOINT" not in os.environ

    def test_disabled_tracing_leaves_endpoint_unset(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A paused key (tracing off) never applies its stored endpoint.

        The endpoint is applied only after the enable decision, so a deliberate
        opt-out must not leak a stored endpoint into the process env.
        """
        import os

        from deepagents_code import auth_store

        monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
        monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        auth_store.set_stored_key(
            "langsmith", "lsv2_test", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert "LANGSMITH_ENDPOINT" not in os.environ

    def test_prefixed_key_override_leaves_stored_endpoint_unset(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A session-scoped key is not paired with a stale stored endpoint."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
        auth_store.set_stored_key(
            "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert "LANGSMITH_ENDPOINT" not in os.environ
        assert "LANGCHAIN_ENDPOINT" not in os.environ

    def test_matching_prefixed_key_allows_stored_endpoint(
        self,
        fake_state_dir: Path,  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scoped key with the stored value still uses the stored endpoint."""
        import os

        from deepagents_code import auth_store

        for var in ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_stored")
        auth_store.set_stored_key(
            "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
        )
        _apply_stored_langsmith_tracing()
        assert os.environ["LANGSMITH_ENDPOINT"] == LANGSMITH_EU_ENDPOINT
        assert "LANGCHAIN_ENDPOINT" not in os.environ


class TestNormalizeLangSmithEndpoint:
    """Tests for normalize_langsmith_endpoint() and is_http_url()."""

    def test_is_http_url(self) -> None:
        """Only http(s) URLs with a host pass validation."""
        assert is_http_url(LANGSMITH_EU_ENDPOINT) is True
        assert is_http_url("http://localhost:1984") is True
        assert is_http_url("ftp://example.com") is False
        assert is_http_url("not-a-url") is False
        assert is_http_url("https://") is False
        assert is_http_url("http://[::1") is False
        # A host with internal whitespace is malformed: rejecting it at save time
        # stops a wrong endpoint from being stored and silently dropping traces.
        assert is_http_url("https://exa mple.com") is False


class TestGetTracingStatus:
    """Tests for get_tracing_status()."""

    _CLEAN: ClassVar[dict[str, str]] = {
        "LANGSMITH_TRACING_V2": "",
        "LANGCHAIN_TRACING_V2": "",
        "LANGSMITH_TRACING": "",
        "LANGCHAIN_TRACING": "",
        "LANGSMITH_API_KEY": "",
        "LANGCHAIN_API_KEY": "",
        "LANGSMITH_ENDPOINT": "",
        "LANGCHAIN_ENDPOINT": "",
        "LANGSMITH_PROJECT": "",
        "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "",
        "DEEPAGENTS_CODE_LANGSMITH_TRACING": "",
        "DEEPAGENTS_CODE_LANGCHAIN_TRACING_V2": "",
        "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "",
        "DEEPAGENTS_CODE_LANGCHAIN_API_KEY": "",
        "LANGSMITH_REPLICA_PROJECTS": "",
        "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS": "",
        "LANGSMITH_PROFILE": "",
        "LANGSMITH_CONFIG_FILE": "/__deepagents_missing_langsmith_config__.json",
    }

    def test_disabled_when_no_flags(self) -> None:
        """A clean environment reports tracing off with configured project metadata."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        with patch.dict("os.environ", self._CLEAN, clear=False):
            status = get_tracing_status()
        assert status.enabled is False
        assert status.explicitly_disabled is False
        assert status.has_credentials is False
        assert status.endpoint is None
        assert status.project == LANGSMITH_PROJECT_DEFAULT
        assert status.project_is_default is True
        assert status.replica_project is None

    @pytest.mark.parametrize(
        ("var", "token", "expected_enabled", "expected_disabled"),
        [
            # Non-bridged flags (bare `env.get` path) set to each falsy token.
            ("LANGSMITH_TRACING_V2", "false", False, True),
            ("LANGSMITH_TRACING_V2", "0", False, True),
            ("LANGSMITH_TRACING_V2", "no", False, True),
            ("LANGSMITH_TRACING_V2", "off", False, True),
            ("LANGCHAIN_TRACING", "false", False, True),
            # Prefixed bridged falsy flag — exercises the prefix-aware
            # `_resolve_env_var_from` off path; doctor runs pre-bootstrap.
            ("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false", False, True),
            # An empty flag reads as not configured, not an explicit opt-out.
            ("LANGSMITH_TRACING_V2", "", False, False),
            # An unrecognized token is neither enabled nor an explicit opt-out.
            ("LANGSMITH_TRACING_V2", "maybe", False, False),
            # A truthy flag is enabled and never reported as explicitly disabled.
            ("LANGSMITH_TRACING_V2", "true", True, False),
        ],
    )
    def test_explicit_disable_matrix(
        self,
        var: str,
        token: str,
        expected_enabled: bool,
        expected_disabled: bool,
    ) -> None:
        """Each flag/token combination resolves to the right tri-state.

        Exercises both the bare-`env.get` path (non-bridged vars) and the
        prefix-aware `_resolve_env_var_from` path (the prefixed bridged flag),
        across every recognized falsy token plus the empty and unrecognized
        cases.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env[var] = token
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is expected_enabled
        assert status.explicitly_disabled is expected_disabled

    def test_prefixed_flag_and_key_are_detected(self) -> None:
        """`DEEPAGENTS_CODE_`-prefixed tracing/key vars resolve like the runtime.

        `dcode doctor` runs before bootstrap bridges these to canonical names,
        so a user with only the supported prefixed vars must still read as
        enabled/configured with the prefixed project resolved.
        """
        from deepagents_code.config import get_tracing_status

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["DEEPAGENTS_CODE_LANGSMITH_API_KEY"] = "lsv2_test"
        env["DEEPAGENTS_CODE_LANGSMITH_PROJECT"] = "prefixed-proj"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == "prefixed-proj"
        assert status.project_is_default is False

    def test_dotenv_values_are_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doctor tracing status sees the same dotenv values as bootstrap."""
        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "DEEPAGENTS_CODE_LANGSMITH_TRACING=true\n"
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY=lsv2_dotenv\n"
            "DEEPAGENTS_CODE_LANGSMITH_PROJECT=dotenv-proj\n"
            "DEEPAGENTS_CODE_LANGSMITH_REPLICA_PROJECTS=replica\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(project)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        with patch.dict("os.environ", {}, clear=True):
            status = config_mod.get_tracing_status()

        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == "dotenv-proj"
        assert status.replica_project == "replica"

    def test_dotenv_profile_credentials_are_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Doctor tracing status uses dotenv profile selectors for credentials."""
        import deepagents_code.config as config_mod

        langsmith = tmp_path / "langsmith.json"
        langsmith.write_text(
            "{"
            '"current_profile":"default",'
            '"profiles":{'
            '"default":{},'
            '"dotenv":{"api_key":"lsv2_profile","api_url":"http://localhost:1984"}'
            "}"
            "}",
            encoding="utf-8",
        )
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "DEEPAGENTS_CODE_LANGSMITH_TRACING=true\n"
            f"LANGSMITH_CONFIG_FILE={langsmith}\n"
            "LANGSMITH_PROFILE=dotenv\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(project)
        monkeypatch.setattr(
            config_mod,
            "_GLOBAL_DOTENV_PATH",
            tmp_path / "missing-global.env",
        )
        config_mod._dotenv_loaded_values.clear()

        with patch.dict("os.environ", {}, clear=True):
            status = config_mod.get_tracing_status()

        assert status.enabled is True
        assert status.has_credentials is True
        assert status.endpoint == "http://localhost:1984"

    def test_keyless_custom_endpoint_resolves_project(self) -> None:
        """A keyless custom endpoint counts as active and resolves the project."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_ENDPOINT"] = "http://localhost:1984"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is False
        assert status.endpoint == "http://localhost:1984"
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_profile_credentials_are_detected(self, tmp_path: Path) -> None:
        """A LangSmith profile API key counts as credentials (no env key needed)."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        config = tmp_path / "config.json"
        config.write_text(
            '{"current_profile":"default","profiles":{"default":{"api_key":"lsv2_profile"}}}',
            encoding="utf-8",
        )
        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        env["LANGSMITH_CONFIG_FILE"] = str(config)
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is True
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_project_resolved_when_enabled_without_auth(self) -> None:
        """Tracing auth state does not hide configured project metadata."""
        from deepagents_code.config import get_tracing_status
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        env = dict(self._CLEAN)
        env["DEEPAGENTS_CODE_LANGSMITH_TRACING"] = "true"
        with patch.dict("os.environ", env, clear=False):
            status = get_tracing_status()
        assert status.enabled is True
        assert status.has_credentials is False
        assert status.project == LANGSMITH_PROJECT_DEFAULT

    def test_enabled_and_explicitly_disabled_is_rejected(self) -> None:
        """The contradictory enabled/disabled pair fails loud at construction."""
        from deepagents_code.config import TracingStatus

        with pytest.raises(ValueError, match="both enabled and explicitly disabled"):
            TracingStatus(
                enabled=True,
                explicitly_disabled=True,
                has_credentials=False,
                endpoint=None,
                project=None,
                project_is_default=False,
                replica_project=None,
            )


class TestQuietSdkLogging:
    """Tests for _quiet_sdk_logging()."""

    def test_attaches_null_handler_without_debug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without debug, SDK loggers get a NullHandler so logs stay off stderr."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        for name in _QUIET_SDK_LOGGER_NAMES:
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
            monkeypatch.setattr(logger, "propagate", True)

        _quiet_sdk_logging()

        for name in _QUIET_SDK_LOGGER_NAMES:
            logger = logging.getLogger(name)
            handlers = logger.handlers
            assert any(isinstance(h, logging.NullHandler) for h in handlers)
            assert logger.level == logging.NOTSET
            # Propagation is left intact so a deliberately configured handler
            # (an embedding app's root handler, pytest's caplog) still receives
            # real SDK errors; the NullHandler alone keeps routine noise off the
            # last-resort stderr handler.
            assert logger.propagate is True

    @staticmethod
    def _mcp_record(
        logger_name: str,
        message: str,
        exc: BaseException,
    ) -> logging.LogRecord:
        """Build the record `logger.exception(message)` would emit for *exc*."""
        return logging.LogRecord(
            logger_name, logging.ERROR, __file__, 0, message, (), (type(exc), exc, None)
        )

    @staticmethod
    def _filter_for(logger_name: str) -> _McpShutdownRaceFilter:
        """Build the filter that gets installed on *logger_name*."""
        return _McpShutdownRaceFilter(_MCP_SHUTDOWN_RACE_MESSAGES[logger_name])

    def test_covers_the_loggers_mcp_actually_uses(self) -> None:
        """The shutdown-race filters target the transports' real loggers.

        When the app quits while an MCP response is in flight, the transport's
        reader task finds the session stream already closed and logs the
        resulting `anyio.ClosedResourceError` with a full traceback. Each filter
        must sit on the exact logger that emits it — logger-level filters do not
        run for records propagated from children, so filtering the `mcp` root
        would miss these records. Reading the names off the modules pins the
        coupling, so an upstream rename fails here rather than in a user's
        terminal.
        """
        import mcp.client.sse
        import mcp.client.streamable_http

        assert (
            mcp.client.streamable_http.logger.name == _MCP_STREAMABLE_HTTP_LOGGER_NAME
        )
        assert mcp.client.sse.logger.name == _MCP_SSE_LOGGER_NAME
        assert set(_MCP_SHUTDOWN_RACE_MESSAGES) == {
            _MCP_STREAMABLE_HTTP_LOGGER_NAME,
            _MCP_SSE_LOGGER_NAME,
        }

    def test_mcp_hierarchy_is_not_quieted(self) -> None:
        """The `mcp` loggers must never get a `NullHandler`.

        The whole point of the targeted filter is that everything else the
        transports report — OAuth failures, unexpected content types, real
        parse errors — keeps reaching the terminal. Quieting `mcp` (or any
        ancestor) would attach a handler to the hierarchy, so `lastResort` is
        skipped and those diagnostics vanish instead. Without this guard,
        someone chasing more MCP noise can add `mcp` to the tuple and every
        other test here still passes.
        """
        for name in _QUIET_SDK_LOGGER_NAMES:
            assert name != "mcp"
            assert not name.startswith("mcp.")

    @pytest.mark.parametrize(
        ("logger_name", "message"),
        [
            (_MCP_STREAMABLE_HTTP_LOGGER_NAME, "Error parsing JSON response"),
            (_MCP_STREAMABLE_HTTP_LOGGER_NAME, "Error parsing SSE message"),
            (_MCP_SSE_LOGGER_NAME, "Error in sse_reader"),
        ],
    )
    def test_mcp_shutdown_race_record_is_dropped(
        self, logger_name: str, message: str
    ) -> None:
        """Every send-into-a-closed-stream record is filtered out.

        A server answering with one JSON body races in `_handle_json_response`,
        one streaming its answer races in `_handle_sse_event`, and the plain SSE
        transport races in `sse_reader`. All three wrap the read-stream `send`
        in the `try` that logs, so all three must be covered.
        """
        import anyio

        f = self._filter_for(logger_name)
        for exc in (anyio.ClosedResourceError(), anyio.BrokenResourceError()):
            assert f.filter(self._mcp_record(logger_name, message, exc)) is False

    def test_mcp_shutdown_race_filter_unwraps_exception_groups(self) -> None:
        """A group of nothing but stream teardowns is dropped, at any depth."""
        import anyio

        f = self._filter_for(_MCP_STREAMABLE_HTTP_LOGGER_NAME)
        grouped = BaseExceptionGroup(
            "unhandled errors in a TaskGroup",
            [
                anyio.BrokenResourceError(),
                BaseExceptionGroup("nested", [anyio.ClosedResourceError()]),
            ],
        )
        record = self._mcp_record(
            _MCP_STREAMABLE_HTTP_LOGGER_NAME, "Error parsing JSON response", grouped
        )

        assert f.filter(record) is False

    def test_mcp_shutdown_race_filter_keeps_mixed_exception_groups(self) -> None:
        """A group carrying a real fault alongside the teardown stays visible.

        Suppressing on *any* matching leaf would hide the `ValueError` here,
        which is the only thing in the group a user could act on.
        """
        import anyio

        f = self._filter_for(_MCP_STREAMABLE_HTTP_LOGGER_NAME)
        grouped = BaseExceptionGroup(
            "unhandled errors in a TaskGroup",
            [ValueError("other"), anyio.ClosedResourceError()],
        )
        record = self._mcp_record(
            _MCP_STREAMABLE_HTTP_LOGGER_NAME, "Error parsing JSON response", grouped
        )

        assert f.filter(record) is True

    def test_mcp_shutdown_race_filter_keeps_unlisted_messages(self) -> None:
        """A closed stream does not hide records outside the listed messages.

        `Error in post_writer` wraps the whole write loop, so a closed stream
        there can mean the transport was orphaned while the session was still
        live — a real fault, not the quit-time race.
        """
        import anyio

        for logger_name in (_MCP_STREAMABLE_HTTP_LOGGER_NAME, _MCP_SSE_LOGGER_NAME):
            f = self._filter_for(logger_name)
            record = self._mcp_record(
                logger_name, "Error in post_writer", anyio.ClosedResourceError()
            )
            assert f.filter(record) is True

    def test_mcp_shutdown_race_filter_keeps_other_records(self) -> None:
        """Records with no exception, or an unrelated one, pass through."""
        f = self._filter_for(_MCP_STREAMABLE_HTTP_LOGGER_NAME)
        no_exc = logging.LogRecord(
            _MCP_STREAMABLE_HTTP_LOGGER_NAME,
            logging.WARNING,
            __file__,
            0,
            "Unknown SSE event: ping",
            (),
            None,
        )
        assert f.filter(no_exc) is True

        record = self._mcp_record(
            _MCP_STREAMABLE_HTTP_LOGGER_NAME, "Error parsing SSE message", ValueError()
        )
        assert f.filter(record) is True

    def test_mcp_shutdown_race_filter_keeps_bare_exception_calls(self) -> None:
        """`logger.exception()` with no active exception has nothing to classify.

        That call yields `exc_info=(None, None, None)` rather than `None`, so
        the record must survive on the strength of having no exception at all.
        `filter`'s `exc_info[1] is None` check is what keeps
        `_is_closed_resource`'s `BaseException` parameter honest; the kept-record
        outcome here holds either way.
        """
        f = self._filter_for(_MCP_STREAMABLE_HTTP_LOGGER_NAME)
        record = logging.LogRecord(
            _MCP_STREAMABLE_HTTP_LOGGER_NAME,
            logging.ERROR,
            __file__,
            0,
            "Error parsing JSON response",
            (),
            (None, None, None),
        )

        assert f.filter(record) is True

    def test_mcp_shutdown_race_is_suppressed_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Logging the real race through the real logger emits nothing.

        The unit tests above hand-build records; this one drives the actual
        `logger.exception` path so a filter installed on the wrong logger, or a
        mismatch in how `exc_info` is shaped, still fails.
        """
        import anyio
        import mcp.client.streamable_http

        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        transport_logger = mcp.client.streamable_http.logger
        monkeypatch.setattr(transport_logger, "filters", [])
        monkeypatch.setattr(transport_logger, "handlers", [])
        captured: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record.getMessage())

        transport_logger.addHandler(_Capture())
        _quiet_sdk_logging()

        def _raise(exc: BaseException) -> None:
            raise exc

        # The race is dropped; the same message carrying a real parse failure
        # is kept, so exactly one record reaches the handler.
        for exc in (anyio.ClosedResourceError(), ValueError("bad json")):
            try:
                _raise(exc)
            except (anyio.ClosedResourceError, ValueError):
                transport_logger.exception("Error parsing JSON response")

        assert captured == ["Error parsing JSON response"]

    def test_routes_harness_diagnostics_to_debug_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Debug mode configures a file handler for harness diagnostics."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.setenv(DEBUG, "1")
        harness_logger = logging.getLogger(
            "deepagents.profiles.harness.harness_profiles"
        )
        harness_logger.handlers.clear()
        monkeypatch.setattr(harness_logger, "propagate", True)

        with patch("deepagents_code._debug.configure_debug_logging") as configure:
            _quiet_sdk_logging()

        assert any(call.args == (harness_logger,) for call in configure.call_args_list)
        assert harness_logger.propagate is True

    def test_routes_mcp_diagnostics_to_debug_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Debug mode configures a file handler for MCP transport diagnostics.

        Without this, removing the filter under debug would only move the
        traceback back to `logging.lastResort` — i.e. straight over the
        alternate-screen TUI — instead of into the debug log.
        """
        from deepagents_code._env_vars import DEBUG

        monkeypatch.setenv(DEBUG, "1")
        transport_loggers = [
            logging.getLogger(name) for name in _MCP_SHUTDOWN_RACE_MESSAGES
        ]
        for transport_logger in transport_loggers:
            monkeypatch.setattr(transport_logger, "handlers", [])
            monkeypatch.setattr(transport_logger, "propagate", True)

        with patch("deepagents_code._debug.configure_debug_logging") as configure:
            _quiet_sdk_logging()

        for transport_logger in transport_loggers:
            assert any(
                call.args == (transport_logger,) for call in configure.call_args_list
            )
            assert transport_logger.propagate is True

    def test_leaves_other_deepagents_loggers_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Actionable Deep Agents runtime warnings keep their normal routing."""
        from deepagents_code._env_vars import DEBUG

        monkeypatch.delenv(DEBUG, raising=False)
        runtime_logger = logging.getLogger("deepagents.backends.filesystem")
        runtime_logger.handlers.clear()
        monkeypatch.setattr(runtime_logger, "propagate", True)

        _quiet_sdk_logging()

        assert runtime_logger.handlers == []
        assert runtime_logger.propagate is True


class TestFetchLangsmithProjectUrl:
    """Tests for fetch_langsmith_project_url()."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()


class TestBuildLangsmithThreadUrl:
    """Tests for build_langsmith_thread_url()."""

    def setup_method(self) -> None:
        """Clear LangSmith URL cache before each test."""
        reset_langsmith_url_cache()


class TestGetProviderKwargsConfigFallback:
    """Tests for _get_provider_kwargs() config-file fallback."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_stored_auth_base_url_reaches_kwargs_without_env_var(
        self, tmp_path: Path
    ) -> None:
        """A `/auth` endpoint reaches the `base_url` kwarg for a non-mapped provider.

        `baseten` has an API-key env var but no base-URL env var, so the stored
        endpoint resolves only through `get_base_url`'s store fallback. This is
        the end-to-end path that makes a saved base URL reach the model as the
        `base_url` constructor kwarg (which `ChatBaseten` accepts via its
        `base_url` alias) rather than being silently dropped.
        """
        from deepagents_code import auth_store

        state_dir = tmp_path / ".state"
        config_path = tmp_path / "config.toml"
        config_path.write_text("")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.object(model_config, "DEFAULT_STATE_DIR", state_dir),
            patch.dict("os.environ", {"BASETEN_API_KEY": "tk"}, clear=True),
        ):
            clear_caches()
            auth_store.set_stored_key(
                "baseten", "tk", base_url="https://proxy.example/v1"
            )
            kwargs = _get_provider_kwargs("baseten")

        assert kwargs["base_url"] == "https://proxy.example/v1"

    def test_unconfigured_providers_return_empty(self) -> None:
        """Providers without config or env credentials return empty kwargs."""
        with patch.dict("os.environ", {}, clear=True):
            kwargs = _get_provider_kwargs("anthropic")
            assert kwargs == {}

            kwargs = _get_provider_kwargs("google_genai")
            assert kwargs == {}

    def test_merges_config_params(self, tmp_path: Path) -> None:
        """Merges params from config with base_url and api_key."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://my-endpoint.example.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
temperature = 0
max_tokens = 4096
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("custom")

        assert kwargs["temperature"] == 0
        assert kwargs["max_tokens"] == 4096
        assert kwargs["base_url"] == "https://my-endpoint.example.com"
        assert kwargs["api_key"] == "secret"

    def test_ollama_optional_api_key_sets_authorization_header(
        self,
        tmp_path: Path,
    ) -> None:
        """OLLAMA_API_KEY is forwarded through client_kwargs for cloud use."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["client_kwargs"]["headers"]["Authorization"] == (
            "Bearer test-key"
        )

    def test_ollama_prefixed_optional_api_key_overrides_canonical(
        self,
        tmp_path: Path,
    ) -> None:
        """The CLI-scoped Ollama key follows normal env override behavior."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict(
                "os.environ",
                {
                    "OLLAMA_API_KEY": "canonical",
                    "DEEPAGENTS_CODE_OLLAMA_API_KEY": "prefixed",
                },
                clear=True,
            ),
        ):
            kwargs = _get_provider_kwargs("ollama")

        assert kwargs["client_kwargs"]["headers"]["Authorization"] == (
            "Bearer prefixed"
        )

    def test_ollama_preserves_user_authorization_header(
        self,
        tmp_path: Path,
    ) -> None:
        """Existing Authorization header (any case) is not overwritten."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"

[models.providers.ollama.params.llama3]
client_kwargs = { headers = { authorization = "Bearer user-supplied" } }
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama", model_name="llama3")

        headers = kwargs["client_kwargs"]["headers"]
        assert headers["authorization"] == "Bearer user-supplied"
        assert "Authorization" not in headers

    def test_ollama_preserves_unrelated_headers_and_client_kwargs(
        self,
        tmp_path: Path,
    ) -> None:
        """Sibling client_kwargs and headers entries survive Authorization injection."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
base_url = "https://ollama.example.com"

[models.providers.ollama.params.llama3]
client_kwargs = { timeout = 30, headers = { "X-Trace-Id" = "abc" } }
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"OLLAMA_API_KEY": "env-key"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama", model_name="llama3")

        client_kwargs = kwargs["client_kwargs"]
        assert client_kwargs["timeout"] == 30
        assert client_kwargs["headers"]["X-Trace-Id"] == "abc"
        assert client_kwargs["headers"]["Authorization"] == "Bearer env-key"

    def test_ollama_local_endpoint_does_not_inject_header(
        self,
        tmp_path: Path,
    ) -> None:
        """Without OLLAMA_API_KEY, no Authorization header is injected."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.ollama]
models = ["llama3"]
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {}, clear=True),
        ):
            kwargs = _get_provider_kwargs("ollama")

        client_kwargs = kwargs.get("client_kwargs", {})
        headers = (
            client_kwargs.get("headers", {}) if isinstance(client_kwargs, dict) else {}
        )
        assert "Authorization" not in headers
        assert "authorization" not in headers

    def test_base_url_and_api_key_override_config_params(self, tmp_path: Path) -> None:
        """base_url/api_key from config fields override same keys in params."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.custom]
models = ["my-model"]
base_url = "https://correct-url.com"
api_key_env = "CUSTOM_KEY"

[models.providers.custom.params]
base_url = "https://wrong-url.com"
""")
        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch.dict("os.environ", {"CUSTOM_KEY": "secret"}, clear=True),
        ):
            kwargs = _get_provider_kwargs("custom")

        # Explicit base_url field should win over kwargs.base_url
        assert kwargs["base_url"] == "https://correct-url.com"


def _make_init_chat_model_mock() -> Mock:
    """Return a `Mock` shaped like `init_chat_model`'s return value.

    Each `TestCreateModel*` test patches `langchain.chat_models.init_chat_model`
    and inspects `call_args`; the returned model needs `profile = None` so the
    downstream context-limit/modality extraction in `create_model` is a no-op.
    """
    mock_model = Mock()
    mock_model.profile = None
    return mock_model


@pytest.fixture
def _isolate_provider_profiles() -> Iterator[None]:
    """Snapshot/restore SDK `_PROVIDER_PROFILES` and CLI registration sentinel.

    The provider-profile registry is process-global. Tests that register
    custom profiles (or that exercise the CLI's lazy OpenRouter registration)
    must not leak state into other tests in the same session.
    """
    from deepagents.profiles.provider import provider_profiles

    from deepagents_code import config as cli_config

    saved_profiles = dict(provider_profiles._PROVIDER_PROFILES)
    saved_cli_flag = cli_config._cli_openrouter_profile_registered
    try:
        yield
    finally:
        provider_profiles._PROVIDER_PROFILES.clear()
        provider_profiles._PROVIDER_PROFILES.update(saved_profiles)
        cli_config._cli_openrouter_profile_registered = saved_cli_flag


class TestOpenRouterVersionCheck:
    """Tests for OpenRouter version enforcement via the SDK profile."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )


class TestOpenRouterHeaders:
    """Tests for OpenRouter default attribution headers."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_injects_attribution_kwargs(self, mock_init: Mock) -> None:
        """`create_model` injects `app_url`, `app_title`, `app_categories`."""
        mock_init.return_value = _make_init_chat_model_mock()

        create_model("openrouter:deepseek/deepseek-chat")

        _, call_kwargs = mock_init.call_args
        assert call_kwargs["app_url"] == "https://pypi.org/project/deepagents-code/"
        assert call_kwargs["app_title"] == "Deep Agents Code"
        assert call_kwargs["app_categories"] == ["cli-agent"]


class TestCreateModelForwardsProviderProfile:
    """Tests that `create_model` forwards profile kwargs to `init_chat_model`.

    Regression coverage for #2959: env-default and explicit OpenAI selections
    both need `use_responses_api=True` so the CLI's PDF-attachment path (which
    emits `type: "file"` content blocks) is routed through the Responses API
    instead of 400'ing against Chat Completions.
    """

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_profile_pre_init_failure_wrapped_in_model_config_error(
        self,
        mock_init: Mock,
        _isolate_provider_profiles: None,  # noqa: PT019
    ) -> None:
        """Arbitrary `pre_init` exceptions surface as `ModelConfigError`.

        Without wrapping, a profile's `pre_init` failure bubbles up as a raw
        traceback to the user; the CLI's error path expects `ModelConfigError`
        for actionable rendering.
        """
        from deepagents.profiles.provider import (
            ProviderProfile,
            register_provider_profile,
        )

        def _broken_pre_init(_spec: str) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        register_provider_profile(
            "anthropic",
            ProviderProfile(pre_init=_broken_pre_init),
        )
        mock_init.return_value = _make_init_chat_model_mock()

        with pytest.raises(ModelConfigError, match="provider profile"):
            create_model("anthropic:claude-sonnet-4-5")


class TestCreateModelFromClass:
    """Tests for _create_model_from_class() custom class factory."""


class TestCreateModelWithCustomClass:
    """Tests for create_model() using custom class_path from config."""

    def setup_method(self) -> None:
        """Clear model config cache before each test."""
        clear_caches()

    def test_configured_provider_takes_precedence_over_bedrock_inference(
        self, tmp_path: Path
    ) -> None:
        """A configured explicit provider is not treated as a bare Bedrock ID."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers."meta.custom"]
class_path = "my_pkg.models:MyChatModel"
models = ["my-model"]
""")
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_instance.profile = None

        with (
            patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path),
            patch(
                "deepagents_code.config._create_model_from_class",
                return_value=mock_instance,
            ) as mock_factory,
        ):
            result = create_model("meta.custom:my-model")

        mock_factory.assert_called_once()
        assert mock_factory.call_args.args[1:3] == ("my-model", "meta.custom")
        assert result.model is mock_instance
        assert result.model_name == "my-model"
        assert result.provider == "meta.custom"


class TestCreateModelExtraKwargs:
    """Tests for create_model() with extra_kwargs from --model-params."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    def test_google_anthropic_vertex_requires_location(self) -> None:
        """Missing Claude-on-Vertex location produces an actionable error."""
        owner = config_module._get_credentials()
        replacement = replace(
            owner.active,
            google_cloud_project="test-project",
            google_cloud_location=None,
        )
        with (
            patch.object(owner, "_active", replacement),
            pytest.raises(
                ModelConfigError,
                match=r"GOOGLE_CLOUD_LOCATION.*DEEPAGENTS_CODE_GOOGLE_CLOUD_LOCATION",
            ),
        ):
            create_model("google_anthropic_vertex:claude-sonnet-4-6")


class TestCreateModelEdgeCaseParsing:
    """Tests for create_model() edge-case spec parsing."""

    @pytest.fixture(autouse=True)
    def _bypass_credential_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "deepagents_code.model_config.has_provider_credentials", lambda _: True
        )

    @patch("langchain.chat_models.init_chat_model")
    def test_leading_colon_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """Leading colon (e.g., ':claude-opus-4-6') is treated as bare model name."""
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        credentials.anthropic_api_key = "test"
        try:
            result = create_model(":claude-opus-4-6")
        finally:
            credentials.anthropic_api_key = None

        # Should have detected 'anthropic' provider and used 'claude-opus-4-6'
        assert result.model_name == "claude-opus-4-6"

    @patch("langchain.chat_models.init_chat_model")
    def test_versioned_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock version suffix is not parsed as a provider separator."""
        model_id = "meta.llama3-70b-instruct-v1:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_versioned_bedrock_vendor_id_not_misparsed(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock vendor namespace is not split on its `:version` suffix.

        Regression: `mistral.` collides with the bare `mistral` prefix, so
        without dotted-namespace detection this ID parses to the garbage pair
        `provider='mistral.mistral-large-2402-v1', model='0'`.
        """
        model_id = "mistral.mistral-large-2402-v1:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_cross_region_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A cross-region inference-profile ID resolves to Bedrock intact."""
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.args == (model_id,)
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_non_versioned_bedrock_id_treated_as_bare_model(
        self, mock_init_chat_model: Mock
    ) -> None:
        """A Bedrock ID without a `:version` suffix still routes to Bedrock."""
        model_id = "amazon.titan-text-express-v1"
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model(model_id)

        assert result.provider == "bedrock"
        assert result.model_name == model_id
        assert mock_init_chat_model.call_args.kwargs["model_provider"] == "bedrock"

    @patch("langchain.chat_models.init_chat_model")
    def test_explicit_provider_not_hijacked_by_bedrock(
        self, mock_init_chat_model: Mock
    ) -> None:
        """An explicit `provider:model` spec wins over Bedrock inference.

        `anthropic.` (dot) is a Bedrock namespace, but `anthropic:` (colon) is
        the explicit-provider syntax and must resolve to Anthropic, not Bedrock.
        """
        mock_model = Mock()
        mock_model.profile = None
        mock_init_chat_model.return_value = mock_model

        result = create_model("anthropic:claude-sonnet-4-5")

        assert result.provider == "anthropic"
        assert result.model_name == "claude-sonnet-4-5"


class TestCreateModelViaInitImportError:
    """Tests for _create_model_via_init() ImportError handling."""

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_provider_receipt_failure_falls_back_to_manual(
        self, mock_init: Mock, tmp_path, monkeypatch
    ) -> None:
        """An unreadable uv receipt degrades to the manual-install hint.

        Exercises the `ToolRequirementIntrospectionError` arm of the fallback:
        `install_package_command` reads the uv tool receipt, and a missing
        receipt must surface an actionable message instead of letting the error
        leak out of hint construction.
        """
        # tmp_path has no uv-receipt.toml, so the receipt read raises.
        monkeypatch.setattr("sys.prefix", str(tmp_path))
        mock_init.side_effect = ImportError("no module")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                return_value=set(),
            ),
            pytest.raises(
                ModelConfigError,
                match="Install the 'langchain-custom_provider' package manually",
            ),
        ):
            _create_model_via_init("some-model", "custom_provider", {})

    @patch("langchain.chat_models.init_chat_model")
    def test_unknown_provider_introspection_failure_falls_back_to_manual(
        self, mock_init: Mock
    ) -> None:
        """Unreadable extras metadata degrades to the manual-install hint.

        Exercises the `ExtrasIntrospectionError` arm of the fallback so the
        user still gets an actionable message instead of an unhandled error
        leaking out of hint construction.
        """
        from deepagents_code.extras_info import ExtrasIntrospectionError

        mock_init.side_effect = ImportError("no module")
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch(
                "deepagents_code.extras_info.installed_extra_names",
                side_effect=ExtrasIntrospectionError("metadata unreadable"),
            ),
            pytest.raises(
                ModelConfigError,
                match="Install the 'langchain-custom_provider' package manually",
            ),
        ):
            _create_model_via_init("some-model", "custom_provider", {})


class TestCreateModelViaInitUnknownProvider:
    """Tests for `UnknownProviderError` translation of langchain inference."""


class TestDetectProvider:
    """Tests for detect_provider() auto-detection from model names."""

    @pytest.mark.parametrize(
        ("model_name", "expected"),
        [
            ("gpt-5.5", "openai"),
            ("gpt-5.2", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("o4-mini", "openai"),
            ("text-davinci-003", "openai"),
            ("command-r-plus", "cohere"),
            ("amazon.titan-text-express-v1", "bedrock"),
            ("anthropic.claude-3-sonnet", "bedrock"),
            ("meta.llama3-70b-instruct-v1:0", "bedrock"),
            # Bedrock vendor namespaces that collide with the bare direct-API
            # prefixes below: the dotted form must win so the `:version` suffix
            # is not misparsed as a `provider:model` separator.
            ("mistral.mistral-large-2402-v1:0", "bedrock"),
            ("deepseek.r1-v1:0", "bedrock"),
            ("cohere.command-r-v1:0", "bedrock"),
            ("ai21.jamba-1-5-large-v1:0", "bedrock"),
            ("writer.palmyra-x5-v1:0", "bedrock"),
            # Structural detection covers vendors with no hardcoded entry.
            ("qwen.qwen3-32b-v1:0", "bedrock"),
            ("google.gemma-3-27b-v1:0", "bedrock"),
            # Cross-region inference-profile IDs front the vendor with a region.
            ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
            ("eu.meta.llama3-2-3b-instruct-v1:0", "bedrock"),
            ("apac.anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock"),
            ("US.Anthropic.Claude-3-5-Sonnet-20241022-v2:0", "bedrock"),
            # A bare name that merely starts with a region token is not Bedrock.
            ("useful-model", None),
            ("mistral-large", "mistralai"),
            ("mixtral-8x7b-instruct", "mistralai"),
            ("deepseek-chat", "deepseek"),
            ("grok-4", "xai"),
            ("sonar-pro", "perplexity"),
            ("claude-sonnet-4-5", "anthropic"),
            ("claude-opus-4-5", "anthropic"),
            ("gemini-3.1-pro-preview", "google_genai"),
            ("nemotron-3-nano-30b-a3b", "nvidia"),
            ("nvidia/nemotron-3-nano-30b-a3b", "nvidia"),
            ("accounts/fireworks/models/kimi-k2p7-code", "fireworks"),
            ("accounts/fireworks/routers/kimi-k2p7-code", "fireworks"),
            ("Accounts/Fireworks/Models/Kimi-K2P7-Code", "fireworks"),
            ("accounts/openai/models/gpt-5.5", None),
            # A different account whose name merely starts with "fireworks"
            # must not resolve to Fireworks; the trailing slash in the prefix
            # is what anchors the match to the exact account namespace.
            ("accounts/fireworks-enterprise/models/kimi-k2p7-code", None),
            ("llama3", None),
            ("solar-pro", None),
            ("some-unknown-model", None),
        ],
    )
    def test_detect_known_patterns(self, model_name: str, expected: str | None) -> None:
        """detect_provider returns the correct provider for known patterns."""
        # Ensure both Anthropic and Google credentials are "available" so the
        # default paths are taken (not the Vertex AI fallbacks).
        credentials.anthropic_api_key = "test"
        credentials.google_api_key = "test"
        try:
            assert detect_provider(model_name) == expected
        finally:
            credentials.anthropic_api_key = None
            credentials.google_api_key = None

    def test_claude_falls_back_to_vertex_when_no_anthropic(self) -> None:
        """Claude models route to Anthropic Vertex when only Vertex is configured."""
        credentials.anthropic_api_key = None
        credentials.google_cloud_project = "my-project"
        credentials.google_api_key = None
        try:
            assert detect_provider("claude-sonnet-4-5") == "google_anthropic_vertex"
        finally:
            credentials.google_cloud_project = None

    def test_gemini_falls_back_to_vertex_when_no_google(self) -> None:
        """Gemini models route to google_vertexai when only Vertex AI is configured."""
        credentials.google_api_key = None
        credentials.google_cloud_project = "my-project"
        try:
            assert detect_provider("gemini-3-pro") == "google_vertexai"
        finally:
            credentials.google_cloud_project = None

    def test_gemini_prefers_google_genai_when_both_available(self) -> None:
        """Gemini prefers google_genai when both Google and Vertex AI are configured."""
        credentials.google_api_key = "test"
        credentials.google_cloud_project = "my-project"
        try:
            # has_vertex_ai is False when google_api_key is set, so this
            # tests the google_genai path which is preferred.
            assert detect_provider("gemini-3-pro") == "google_genai"
        finally:
            credentials.google_api_key = None
            credentials.google_cloud_project = None

    def test_case_insensitive(self) -> None:
        """detect_provider is case-insensitive."""
        credentials.anthropic_api_key = "test"
        try:
            assert detect_provider("Claude-Sonnet-4-5") == "anthropic"
            assert detect_provider("gpt-5.5") == "openai"
        finally:
            credentials.anthropic_api_key = None


class TestPrefixedLangsmithBridge:
    """Bridging a prefixed override onto the canonical SDK name."""

    def test_an_empty_prefixed_value_still_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`FOO=""` is an explicit disable, not an absent override."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "")
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        config_mod._apply_prefixed_langsmith_env()

        assert os.environ["LANGSMITH_TRACING"] == ""


class TestTracingEnvironmentReconcile:
    """The LangSmith SDK reads `os.environ`, so the snapshot is published."""

    @pytest.mark.parametrize("prefixed", [False, True])
    @pytest.mark.parametrize("load_first", [False, True])
    def test_publishing_does_not_contaminate_later_workspaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        prefixed: bool,
        load_first: bool,
    ) -> None:
        """Repeated publication preserves each workspace's identity and opt-out."""
        import deepagents_code.config as config_mod

        monkeypatch.setattr(config_mod, "_dotenv_loaded_values", {})
        for var in config_mod._TRACING_RECONCILED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        first = tmp_path / "first"
        second = tmp_path / "second"
        empty = tmp_path / "empty"
        for workspace in (first, second, empty):
            workspace.mkdir()
        prefix = "DEEPAGENTS_CODE_" if prefixed else ""
        (first / ".env").write_text(
            f"{prefix}LANGSMITH_TRACING=true\n"
            f"{prefix}LANGSMITH_API_KEY=first-key\n"
            f"{prefix}LANGSMITH_PROJECT=first-project\n"
            "LANGSMITH_ENDPOINT=https://first.example.com\n"
        )
        (second / ".env").write_text(
            "LANGSMITH_TRACING=false\n"
            "LANGSMITH_API_KEY=second-key\n"
            "LANGSMITH_PROJECT=second-project\n"
        )
        if load_first:
            config_mod._load_dotenv(start_path=first)
        for workspace in (first, second, empty, first, second):
            snapshot = config_mod._preview_dotenv_environ(start_path=workspace)
            config_mod.reconcile_tracing_environment(snapshot)
            if workspace == second:
                assert snapshot["LANGSMITH_TRACING"] == "false"
                assert snapshot["LANGSMITH_API_KEY"] == "second-key"
                assert snapshot["LANGSMITH_PROJECT"] == "second-project"
                assert "LANGSMITH_ENDPOINT" not in snapshot
                with config_mod.use_environment(snapshot):
                    assert config_mod.get_langsmith_project_name() is None
            elif workspace == empty:
                assert not snapshot.keys() & set(
                    config_mod._TRACING_RECONCILED_ENV_VARS
                )

    @pytest.mark.parametrize("published", ["workspace-value", None])
    def test_preview_preserves_overwritten_shell_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, published: str | None
    ) -> None:
        """Publishing or removing a selector does not erase its shell baseline."""
        import deepagents_code.config as config_mod

        monkeypatch.setenv("LANGSMITH_PROJECT", "shell-project")
        (tmp_path / ".env").write_text("LANGSMITH_PROJECT=dotenv-project\n")
        snapshot = {"LANGSMITH_PROJECT": published} if published is not None else {}
        for _ in range(2):
            config_mod.reconcile_tracing_environment(snapshot)
            preview = config_mod._preview_dotenv_environ(start_path=tmp_path)
            assert preview["LANGSMITH_PROJECT"] == "shell-project"

        monkeypatch.setenv("LANGSMITH_PROJECT", "updated-shell-project")
        preview = config_mod._preview_dotenv_environ(start_path=tmp_path)
        assert preview["LANGSMITH_PROJECT"] == "updated-shell-project"

    def test_workspace_settings_reach_the_process_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot values are written under their canonical SDK names."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "stale-key")

        config_mod.reconcile_tracing_environment(
            {
                "LANGSMITH_TRACING": "true",
                "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "workspace-key",
                "DEEPAGENTS_CODE_LANGSMITH_PROJECT": "agent-project",
            }
        )

        assert os.environ["LANGSMITH_TRACING"] == "true"
        # A prefixed override resolves onto the canonical name the SDK reads.
        assert os.environ["LANGSMITH_API_KEY"] == "workspace-key"
        assert os.environ["LANGSMITH_PROJECT"] == "agent-project"

    def test_unsupported_prefixed_endpoint_does_not_reach_sdk_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prefixed endpoint ignored by redaction must also be ignored here."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://stale.example.com")

        config_mod.reconcile_tracing_environment(
            {
                "DEEPAGENTS_CODE_LANGSMITH_TRACING": "true",
                "DEEPAGENTS_CODE_LANGSMITH_ENDPOINT": "https://upload.example.com",
            }
        )

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert "LANGSMITH_ENDPOINT" not in os.environ

    def test_previous_workspace_values_do_not_linger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A var the workspace does not set is removed, not left behind."""
        import os

        import deepagents_code.config as config_mod

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_API_KEY", "workspace-a-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "workspace-a")
        # The profile pair decides the key and endpoint when no canonical var
        # does, so it lingers just as consequentially as the rest.
        monkeypatch.setenv("LANGSMITH_PROFILE", "workspace-a-profile")
        monkeypatch.setenv("LANGSMITH_CONFIG_FILE", "/tmp/workspace-a.json")

        config_mod.reconcile_tracing_environment({})

        for var in (
            "LANGSMITH_TRACING",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_PROFILE",
            "LANGSMITH_CONFIG_FILE",
        ):
            assert var not in os.environ

    def test_sdk_env_caches_are_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reconcile must not leave the SDK answering from a stale cache."""
        from langsmith import utils as ls_utils

        import deepagents_code.config as config_mod

        monkeypatch.setenv("LANGSMITH_PROJECT", "workspace-a")
        assert ls_utils.get_env_var("PROJECT") == "workspace-a"

        config_mod.reconcile_tracing_environment({"LANGSMITH_PROJECT": "workspace-b"})

        assert ls_utils.get_env_var("PROJECT") == "workspace-b"


class TestUserLangsmithEnvironment:
    """LangSmith credentials for the agent stay out of user commands."""

    def test_dotenv_snapshot_excludes_global_and_preserves_project(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "LANGSMITH_PROJECT=project-value\n"
            "LANGCHAIN_PROJECT=legacy-project-value\n"
            "LANGSMITH_SESSION=legacy-smith-session\n"
            "LANGCHAIN_SESSION=legacy-chain-session\n"
            "LANGSMITH_WORKSPACE_ID=project-workspace\n"
        )
        global_dotenv = tmp_path / "global.env"
        global_dotenv.write_text(
            "LANGSMITH_API_KEY=global-key\n"
            "LANGSMITH_PROJECT=global-value\n"
            "LANGSMITH_PROFILE=global-profile\n"
        )
        for var in config_mod._USER_LANGSMITH_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://launch.example.com")
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", global_dotenv)
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = {
            var: os.environ.get(var) for var in config_mod._USER_LANGSMITH_ENV_VARS
        }
        config_mod._dotenv_loaded_values.clear()

        try:
            with patch.dict(os.environ, os.environ.copy(), clear=True):
                config_mod._load_dotenv(
                    start_path=project,
                    capture_user_langsmith=True,
                )

                assert os.environ["LANGSMITH_API_KEY"] == "global-key"
            assert config_mod._bootstrap_state.user_langsmith_env == {
                "LANGSMITH_API_KEY": None,
                "LANGCHAIN_API_KEY": None,
                "LANGSMITH_PROJECT": "project-value",
                "LANGCHAIN_PROJECT": "legacy-project-value",
                "LANGSMITH_SESSION": "legacy-smith-session",
                "LANGCHAIN_SESSION": "legacy-chain-session",
                "LANGSMITH_ENDPOINT": "https://launch.example.com",
                "LANGCHAIN_ENDPOINT": None,
                "LANGSMITH_WORKSPACE_ID": "project-workspace",
                "LANGSMITH_PROFILE": None,
                "LANGSMITH_CONFIG_FILE": None,
                "LANGSMITH_TRACING_V2": None,
                "LANGCHAIN_TRACING_V2": None,
                "LANGSMITH_TRACING": None,
                "LANGCHAIN_TRACING": None,
                "LANGSMITH_RUNS_ENDPOINTS": None,
                "LANGCHAIN_RUNS_ENDPOINTS": None,
            }
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user
            config_mod._dotenv_loaded_values.clear()

    @pytest.mark.parametrize(
        "encoded",
        [
            "not json at all",
            '{"launch": {}}',
            '{"launch": {}, "user": {}}',
            '{"launch": {"LANGSMITH_API_KEY": "k"}, "user": {}}',
            '{"launch": {"LANGSMITH_API_KEY": 1}, "user": {}}',
        ],
        ids=["malformed", "missing-half", "empty", "partial-keys", "wrong-type"],
    )
    def test_unusable_carrier_strips_rather_than_using_agent_credentials(
        self, encoded: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An undecodable carrier must not fall back to the agent's own key."""
        import deepagents_code.config as config_mod

        agent_state = dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS, "agent-value")
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.launch_langsmith_env = dict(agent_state)
        config_mod._bootstrap_state.user_langsmith_env = dict(agent_state)
        env = {
            config_mod._USER_LANGSMITH_ENV_CARRIER: encoded,
            "LANGSMITH_API_KEY": "agent-session-key",
            "LANGSMITH_PROJECT": "deepagents-code",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "prefixed-agent-key",
            "PATH": "/usr/bin",
        }

        try:
            config_mod.restore_user_langsmith_env(env)
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user

        for var in config_mod._USER_LANGSMITH_ENV_VARS:
            assert var not in env
            assert f"DEEPAGENTS_CODE_{var}" not in env
        assert config_mod._USER_LANGSMITH_ENV_CARRIER not in env
        # Unrelated variables survive.
        assert env["PATH"] == "/usr/bin"
        # The user is told, on stderr, not only in the buffered debug log.
        assert "without LangSmith credentials" in capsys.readouterr().err

    def test_launch_shell_outranks_the_project_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Launch beats project `.env`, and the global `.env` is excluded."""
        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "LANGSMITH_API_KEY=project-key\nLANGSMITH_PROJECT=project-only\n"
        )
        global_dotenv = tmp_path / "global.env"
        global_dotenv.write_text("LANGSMITH_TRACING=true\n")
        monkeypatch.setattr(config_mod, "_GLOBAL_DOTENV_PATH", global_dotenv)

        launch = dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS)
        launch["LANGSMITH_API_KEY"] = "launch-key"
        carrier = json.dumps({"launch": launch, "user": dict(launch)})

        env = {config_mod._USER_LANGSMITH_ENV_CARRIER: carrier}
        config_mod.restore_user_langsmith_env(env, start_path=project)

        # Both sources set the key; the launch shell wins.
        assert env["LANGSMITH_API_KEY"] == "launch-key"
        # Only the project file sets this one, so it fills in.
        assert env["LANGSMITH_PROJECT"] == "project-only"
        # The global profile `.env` configures the agent, not user commands.
        assert "LANGSMITH_TRACING" not in env

    def test_unreadable_project_dotenv_keeps_the_carried_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `.env` that cannot be read must not read as an empty one."""
        import deepagents_code.config as config_mod

        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text("LANGSMITH_PROJECT=from-dotenv\n")
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )

        # The realistic shape: the launch shell did not set the project, the
        # project `.env` did. So only the carried `user` half can supply it.
        launch = dict.fromkeys(config_mod._USER_LANGSMITH_ENV_VARS)
        user = dict(launch, LANGSMITH_PROJECT="carried-project")
        carrier = json.dumps({"launch": launch, "user": user})

        def _unreadable(*_args: object, **_kwargs: object) -> dict[str, str | None]:
            raise OSError(5, "Input/output error")

        env = {config_mod._USER_LANGSMITH_ENV_CARRIER: carrier}
        with patch.object(config_mod, "_dotenv_values_from", _unreadable):
            config_mod.restore_user_langsmith_env(env, start_path=project)

        assert env["LANGSMITH_PROJECT"] == "carried-project"

    def test_restore_drops_agent_values_and_prefixed_settings(self) -> None:
        import deepagents_code.config as config_mod

        original = dict(config_mod._bootstrap_state.user_langsmith_env)
        config_mod._bootstrap_state.user_langsmith_env = dict.fromkeys(
            config_mod._USER_LANGSMITH_ENV_VARS
        )
        env = {
            "LANGSMITH_API_KEY": "stored-key",
            "LANGSMITH_ENDPOINT": "https://stored.example.com",
            "LANGSMITH_PROJECT": "stored-project",
            "DEEPAGENTS_CODE_LANGSMITH_API_KEY": "prefixed-key",
            "DEEPAGENTS_CODE_LANGSMITH_PROFILE": "prefixed-profile",
        }

        try:
            config_mod.restore_user_langsmith_env(env)
        finally:
            config_mod._bootstrap_state.user_langsmith_env = original

        assert not any(
            var in env or f"DEEPAGENTS_CODE_{var}" in env
            for var in config_mod._USER_LANGSMITH_ENV_VARS
        )


class TestLazySingletons:
    """Tests for lazy process-wide state and console resolution."""

    def test_getattr_returns_credentials(self) -> None:
        """The credentials accessor returns the typed singleton."""
        from deepagents_code.config import _get_credentials

        result = _get_credentials()
        assert isinstance(result, Credentials)
        assert result is _get_credentials()
        assert result.active is result.active

    def test_ensure_bootstrap_is_idempotent(self) -> None:
        """_ensure_bootstrap is a no-op on second call."""
        from deepagents_code.config import _ensure_bootstrap

        # First call already ran (credentials were used above).
        # Calling again should be a harmless no-op.
        _ensure_bootstrap()
        from deepagents_code.config import _get_credentials

        assert isinstance(_get_credentials(), Credentials)

    def test_bootstrap_captures_langsmith_before_project_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing project lookup still leaves an encodable launch snapshot."""
        import deepagents_code.config as config_mod
        from deepagents_code import project_utils

        def _boom() -> None:
            msg = "project context unavailable"
            raise RuntimeError(msg)

        monkeypatch.setattr(project_utils, "get_server_project_context", _boom)
        monkeypatch.setenv("LANGSMITH_API_KEY", "launch-key")
        original_launch = dict(config_mod._bootstrap_state.launch_langsmith_env)
        original_user = dict(config_mod._bootstrap_state.user_langsmith_env)
        original_done = config_mod._bootstrap_state.done
        original_error = config_mod._bootstrap_state.error
        config_mod._bootstrap_state.launch_langsmith_env = {}
        config_mod._bootstrap_state.user_langsmith_env = {}
        config_mod._bootstrap_state.done = False
        config_mod._bootstrap_state.error = None

        try:
            config_mod._ensure_bootstrap()

            # Bootstrap swallows the failure by contract. The snapshot needs
            # only `os.environ`, so it must survive -- otherwise the server
            # cannot start at all.
            assert config_mod._bootstrap_state.error is not None
            assert (
                config_mod._bootstrap_state.launch_langsmith_env["LANGSMITH_API_KEY"]
                == "launch-key"
            )
            config_mod._encode_user_langsmith_env()
        finally:
            config_mod._bootstrap_state.launch_langsmith_env = original_launch
            config_mod._bootstrap_state.user_langsmith_env = original_user
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.error = original_error

    def test_bootstrap_warns_on_conflicting_override(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A conflicting prefixed override logs a single explanatory warning."""
        import logging

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_original")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
            monkeypatch.delenv(
                "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", raising=False
            )
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            warnings = [
                r.getMessage()
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
            assert len(warnings) == 1
            assert "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING=1" in warnings[0]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_suppresses_override_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The suppression flag silences the override warning but keeps the override."""
        import logging
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_original")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_override")
            monkeypatch.setenv("DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", "1")
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            # Override still applies; only the warning is silenced.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_override"
            assert not [
                r
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_no_warning_when_values_match(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Matching canonical and prefixed values propagate without warning."""
        import logging
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_same")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_same")
            monkeypatch.delenv(
                "DEEPAGENTS_CODE_SUPPRESS_ENV_OVERRIDE_WARNING", raising=False
            )
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
            ):
                _ensure_bootstrap()

            # No conflict, so no warning; the shared value stays in place.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_same"
            assert not [
                r
                for r in caplog.records
                if "DEEPAGENTS_CODE_LANGSMITH_API_KEY" in r.getMessage()
            ]
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_defaults_project_when_tracing_and_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing on with a key but no project defaults to deepagents-code.

        Exercises `_apply_default_langsmith_project` wired into the real
        `_ensure_bootstrap` flow (after the override and orphaned-tracing
        steps) — coverage the helper-level tests cannot provide.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap
        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_keyless_tracing_leaves_project_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keyless tracing is disabled first, so no default project is applied.

        Regression guard for the ordering between `_disable_orphaned_tracing`
        and `_apply_default_langsmith_project`: a tracing flag with no
        resolvable key must be flipped off *before* the default runs, so
        `LANGSMITH_PROJECT` is left unset (tracing never starts) rather than
        pointed at `deepagents-code`.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code.config import _ensure_bootstrap

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("LANGSMITH_TRACING", "true")
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
            monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_credentials",
                    return_value=False,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_custom_endpoint",
                    return_value=False,
                ),
            ):
                _ensure_bootstrap()

            assert "LANGSMITH_PROJECT" not in os.environ
            assert os.environ["LANGSMITH_TRACING"] == "false"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_stored_langsmith_key_keeps_tracing_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A `/auth`-stored LangSmith key survives `_disable_orphaned_tracing`.

        End-to-end regression guard for the bootstrap ordering: a key stored via
        `/auth` (never exported to the env) must be bridged onto
        `LANGSMITH_API_KEY` and auto-enable tracing *before*
        `_disable_orphaned_tracing` runs, so the orphan guard sees the key and
        leaves tracing on. The helper-level `_apply_stored_langsmith_tracing`
        tests cannot catch a regression that reorders the two bootstrap steps.
        """
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import _ensure_bootstrap

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            for var in (
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "LANGSMITH_PROJECT",
                "DEEPAGENTS_CODE_LANGSMITH_TRACING",
                "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
            ):
                monkeypatch.delenv(var, raising=False)
            auth_store.set_stored_key("langsmith", "lsv2_stored")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_credentials",
                    return_value=False,
                ),
                patch(
                    "deepagents_code.config._has_langsmith_profile_custom_endpoint",
                    return_value=False,
                ),
            ):
                _ensure_bootstrap()

            # The stored key was bridged onto the canonical env var, and tracing
            # stayed on instead of being disabled as orphaned.
            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_stored"
            assert os.environ["LANGSMITH_TRACING"] == "true"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_bootstrap_prefixed_langsmith_key_wins_over_stored_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session-scoped LangSmith key remains authoritative at bootstrap."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import _ensure_bootstrap

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "lsv2_prefixed")
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "true")
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
            monkeypatch.delenv("LANGCHAIN_ENDPOINT", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            auth_store.set_stored_key(
                "langsmith", "lsv2_stored", base_url=LANGSMITH_EU_ENDPOINT
            )

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_API_KEY"] == "lsv2_prefixed"
            assert os.environ["LANGSMITH_TRACING"] == "true"
            assert "LANGSMITH_ENDPOINT" not in os.environ
            assert "LANGCHAIN_ENDPOINT" not in os.environ
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls

    def test_scoped_tracing_opt_out_restores_user_tracing_for_shell_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deep Agents Code opt-out does not leak into child command envs."""
        import os

        import deepagents_code.config as config_mod
        from deepagents_code import auth_store
        from deepagents_code.config import (
            _ensure_bootstrap,
            restore_user_langsmith_env,
        )

        monkeypatch.setattr(
            "deepagents_code.model_config.DEFAULT_STATE_DIR", tmp_path / ".state"
        )

        original_done = config_mod._bootstrap_state.done
        original_ls = config_mod._bootstrap_state.original_langsmith_project
        config_mod._bootstrap_state.done = False

        try:
            monkeypatch.setenv("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false")
            monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
            monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
            monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
            monkeypatch.delenv("DEEPAGENTS_CODE_LANGSMITH_PROJECT", raising=False)
            auth_store.set_stored_key("langsmith", "lsv2_stored")

            with (
                patch("deepagents_code.config._load_dotenv"),
                patch(
                    "deepagents_code.project_utils.get_server_project_context",
                    return_value=None,
                ),
            ):
                _ensure_bootstrap()

            assert os.environ["LANGSMITH_TRACING"] == "false"
            assert os.environ["LANGCHAIN_TRACING_V2"] == "false"

            shell_env = os.environ.copy()
            restore_user_langsmith_env(shell_env)

            assert "LANGSMITH_TRACING" not in shell_env
            assert shell_env["LANGCHAIN_TRACING_V2"] == "true"
        finally:
            config_mod._bootstrap_state.done = original_done
            config_mod._bootstrap_state.original_langsmith_project = original_ls


class TestApplyDefaultLangsmithProject:
    """Tests for _apply_default_langsmith_project()."""

    def test_defaults_when_tracing_on_and_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tracing on with no project set routes to the default project."""
        import os

        from deepagents_code.config_manifest import LANGSMITH_PROJECT_DEFAULT

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        _apply_default_langsmith_project()

        assert os.environ["LANGSMITH_PROJECT"] == LANGSMITH_PROJECT_DEFAULT

    def test_noop_when_project_already_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing LANGSMITH_PROJECT is never overwritten."""
        import os

        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_PROJECT", "user-project")

        _apply_default_langsmith_project()

        assert os.environ["LANGSMITH_PROJECT"] == "user-project"

    def test_noop_when_tracing_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No default is applied when tracing is not enabled."""
        import os

        monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

        _apply_default_langsmith_project()

        assert "LANGSMITH_PROJECT" not in os.environ


class TestFindDotenvFromStartPath:
    """Tests for _find_dotenv_from_start_path."""

    def test_continues_past_oserror_on_intermediate_dir(self, tmp_path: Path) -> None:
        """OSError on an intermediate .env candidate doesn't abort search."""
        from deepagents_code.config import _find_dotenv_from_start_path

        # Create .env in the grandparent
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=val")

        child = tmp_path / "sub"
        child.mkdir()

        # Patch is_file to raise OSError for the child's .env candidate
        original_is_file = Path.is_file

        def patched_is_file(self: Path) -> bool:
            if self == child / ".env":
                msg = "Permission denied"
                raise OSError(msg)
            return original_is_file(self)

        with patch.object(Path, "is_file", patched_is_file):
            result = _find_dotenv_from_start_path(child)

        # Should continue past the OSError and find .env in parent
        assert result == env_file


class TestDetectModePrefix:
    """Tests for `detect_mode_prefix`.

    This helper is the linchpin for routing typed prefixes to the correct
    mode. The longest-prefix-first invariant is critical: if `!!` ever loses
    to `!`, every `!!` command would silently route as a single-bang shell
    command and leak content to the model.
    """


class TestInterpreterSettings:
    """Tests for `[interpreter]` config.toml loading and validation."""

    @staticmethod
    def _resolve() -> tuple[bool, InterpreterConfig]:
        from deepagents_code.config_manifest import get_option
        from deepagents_code.configuration.resolver import get_config_resolver

        option = get_option("interpreter.enable_interpreter")
        assert option is not None
        enabled = bool(get_config_resolver().get(option).value)
        return enabled, InterpreterConfig.from_resolver()

    def test_ptc_list_with_safe_preset_round_trip(self, tmp_path: Path) -> None:
        """`"safe"` is preserved as a list entry until agent-build expansion."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = ["safe", "task"]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            _, interpreter = self._resolve()

        assert interpreter.ptc == ["safe", "task"]

    def test_ptc_list_with_all_falls_back(self, tmp_path: Path) -> None:
        """`"all"` inside a list is rejected, falling back to the default."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            """
[interpreter]
ptc = ["all", "task"]
"""
        )
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            _, interpreter = self._resolve()

        assert interpreter.ptc == "safe"


class TestCreateModelCodex:
    """`create_model` dispatch for the ChatGPT-OAuth `openai_codex` provider.

    Covers the runtime path that turns a stored token into a working model —
    untested before this suite. All cases isolate the token store to a temp
    path and never touch the network.
    """

    def _plant_token(self, path: Path) -> None:
        """Write a valid (unexpired) token bundle at `path` with 0600 perms."""
        import json as _json
        from datetime import UTC, datetime, timedelta

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _json.dumps(
                {
                    "access_token": "fake_access",
                    "refresh_token": "fake_refresh",
                    "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                    "account_id": "acct_abc",
                    "plan_type": "pro",
                    "user_id": "user_xyz",
                    "id_token": None,
                }
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def test_missing_token_raises_missing_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No stored token → `MissingCredentialsError` pointing at `/auth`."""
        from deepagents_code.integrations import openai_codex
        from deepagents_code.model_config import MissingCredentialsError

        monkeypatch.setattr(
            openai_codex, "default_store_path", lambda: tmp_path / "missing.json"
        )
        clear_caches()
        with pytest.raises(MissingCredentialsError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        # No env var to set; the recovery hint must route through `/auth`.
        assert exc_info.value.env_var is None
        assert "ChatGPT" in str(exc_info.value)

    def test_success_builds_codex_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored token → a `_ChatOpenAICodex` under the codex provider."""
        from langchain_openai.chat_models.codex import _ChatOpenAICodex

        from deepagents_code.integrations import openai_codex

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(openai_codex, "default_store_path", lambda: path)
        clear_caches()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`_ChatOpenAICodex` is experimental",
                category=UserWarning,
            )
            result = create_model(
                "openai_codex:gpt-5.2-codex",
                extra_kwargs={"http_socket_options": []},
            )
        assert isinstance(result.model, _ChatOpenAICodex)
        assert result.provider == "openai_codex"
        assert result.model_name == "gpt-5.2-codex"

    def test_reasoning_effort_composes_with_configured_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from deepagents_code.integrations import openai_codex as codex_mod

        token_path = tmp_path / "auth.json"
        self._plant_token(token_path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: token_path)
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[models.providers.openai_codex]
models = ["gpt-5.5"]
[models.providers.openai_codex.params."gpt-5.5".reasoning]
summary = "auto"
effort = "low"
""")
        captured: dict[str, Any] = {}
        model = _make_init_chat_model_mock()

        def _capture(_model_name: str, /, **kwargs: Any) -> Any:  # noqa: ANN401
            captured.update(kwargs)
            return model

        monkeypatch.setattr(codex_mod, "build_chat_model", _capture)
        clear_caches()
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            create_model(
                "openai_codex:gpt-5.5",
                extra_kwargs={"reasoning_effort": "high"},
            )

        assert captured["reasoning"] == {"summary": "auto", "effort": "high"}
        assert "reasoning_effort" not in captured

    def test_api_key_kwarg_is_stripped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A passed `api_key` must not reach the model; bearer is OAuth-only.

        `_ChatOpenAICodex` wires the OAuth token into `openai_api_key` as a
        callable, so the cleanest check is that the codex branch drops the
        `api_key` kwarg before it reaches `build_chat_model`.
        """
        from deepagents_code.integrations import openai_codex as codex_mod

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        captured: dict[str, Any] = {}
        real_build = codex_mod.build_chat_model

        def _capture(model_name: str, /, **kwargs: Any) -> Any:  # noqa: ANN401  # passthrough capture
            captured.update(kwargs)
            return real_build(model_name, **kwargs)

        monkeypatch.setattr(codex_mod, "build_chat_model", _capture)
        clear_caches()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`_ChatOpenAICodex` is experimental",
                category=UserWarning,
            )
            create_model(
                "openai_codex:gpt-5.2-codex",
                extra_kwargs={
                    "api_key": "sk-should-be-stripped",
                    "http_socket_options": [],
                },
            )
        assert "api_key" not in captured

    def test_expired_session_routes_to_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A revoked refresh token → `MissingCredentialsError`, not generic.

        The codex branch must route `CodexAuthExpiredError` to the same
        sign-in recovery path as a missing token so the retry flow offers
        `/auth`, rather than wrapping it in a generic `ModelConfigError`.
        """
        from deepagents_code.integrations import openai_codex as codex_mod
        from deepagents_code.model_config import MissingCredentialsError

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        def _raise_expired(_model_name: str, /, **_kwargs: Any) -> Any:  # noqa: ANN401  # passthrough stub
            msg = "session expired"
            raise codex_mod.CodexAuthExpiredError(msg)

        monkeypatch.setattr(codex_mod, "build_chat_model", _raise_expired)
        clear_caches()
        with pytest.raises(MissingCredentialsError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        assert exc_info.value.env_var is None
        assert "expired" in str(exc_info.value).lower()

    def test_unexpected_build_error_wraps_as_model_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely unexpected build failure → `ModelConfigError` with spec.

        The broad catch-all is the last resort for construction failures that
        are neither missing nor expired credentials; it must name the spec
        rather than leak a raw traceback.
        """
        from deepagents_code.integrations import openai_codex as codex_mod
        from deepagents_code.model_config import ModelConfigError

        path = tmp_path / "auth.json"
        self._plant_token(path)
        monkeypatch.setattr(codex_mod, "default_store_path", lambda: path)

        def _boom(_model_name: str, /, **_kwargs: Any) -> Any:  # noqa: ANN401  # passthrough stub
            msg = "unexpected constructor failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(codex_mod, "build_chat_model", _boom)
        clear_caches()
        with pytest.raises(ModelConfigError) as exc_info:
            create_model("openai_codex:gpt-5.2-codex")
        assert "openai_codex:gpt-5.2-codex" in str(exc_info.value)


class TestResolveGoalAutoAcceptCriteria:
    """Coverage for the config.py wrapper over the goals manifest option."""

    def test_missing_manifest_option_fails_closed_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A missing manifest entry disables auto-accept and logs why.

        The `option is None` branch only fires on a manifest regression (a
        renamed or dropped key), so it must fail closed to review *and* leave a
        trail rather than silently ignoring a user's saved preference.
        """
        from deepagents_code.config import resolve_goal_auto_accept_criteria

        with (
            patch(
                "deepagents_code.config_manifest.get_option",
                return_value=None,
            ) as get_option,
            caplog.at_level(logging.WARNING, logger="deepagents_code.config"),
        ):
            result = resolve_goal_auto_accept_criteria()

        assert result == (False, "default")
        get_option.assert_called_once_with("goals.auto_accept_criteria")
        assert any(
            "goals.auto_accept_criteria" in record.getMessage()
            for record in caplog.records
        )


class TestCollectRetryConfigWarnings:
    """Retry config problems must reach the user, not just the debug buffer."""

    @staticmethod
    def _warnings(tmp_path: Path, toml: str) -> list[str]:
        from deepagents_code.config import collect_retry_config_startup

        config_path = tmp_path / "config.toml"
        config_path.write_text(toml)
        with patch.object(model_config, "DEFAULT_CONFIG_PATH", config_path):
            warnings, _ = collect_retry_config_startup()
        return warnings


class TestDeniedHomeKeyReporting:
    """A denied `DEEPAGENTS_HOME` must be loud in the user's own dotenv.

    Denying it from a project `.env` is expected and stays at debug level. The
    global `.env` is the user's own trusted file, so silently dropping a key
    they deliberately wrote leaves a setting that never takes effect and no way
    to discover why — the constraint (it selects the profile owning that very
    file) is not guessable.
    """

    @pytest.mark.parametrize(
        "key", ["BASH_ENV", "GIT_SSH_COMMAND", "PYTHONPATH", "NODE_OPTIONS"]
    )
    def test_every_denied_key_in_the_users_own_dotenv_is_reported(
        self, key: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The trusted-file rule applies to all denied keys, not just one.

        A user who writes `PYTHONPATH` into their own `~/.deepagents/.env`
        otherwise gets a setting that never takes effect, with the drop
        recorded only at debug level.
        """
        from deepagents_code.config import _report_denied_env_key

        with caplog.at_level(logging.WARNING, logger="deepagents_code.config"):
            _report_denied_env_key(key, tmp_path / ".env", is_project=False)

        assert key in caplog.text
        assert "shell environment" in caplog.text


class TestReservedAgentNames:
    """An agent must not be able to resolve onto app-owned profile state.

    Agent profiles live directly under the profile root, so `dcode -a plugins`
    would otherwise stamp an `AGENTS.md` marker into the plugin store.
    """

    def test_windows_trailing_space_alias_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject a Windows trailing-space alias of a reserved name.

        `"plugins "` passes the character allowlist (whitespace is allowed)
        but Windows strips the trailing space and resolves it onto `plugins/`.
        The strip is Windows filesystem semantics, so the guard only applies
        it there; on POSIX `plugins ` is a genuinely different directory.
        """
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(ValueError, match="reserved"):
            get_agent_dir("plugins ")

    def test_trailing_dot_never_reaches_the_reserved_check(self) -> None:
        """The character allowlist already rejects `.` on every platform.

        A trailing-dot alias such as `plugins.` is refused as an invalid name
        before the reserved-name comparison runs.
        """
        with pytest.raises(ValueError, match="Invalid agent name"):
            get_agent_dir("plugins.")

    def test_the_agents_md_accessor_rejects_invalid_characters(self) -> None:
        """It skipped the character check as well, not only reserved names."""
        with pytest.raises(ValueError, match="Invalid agent name"):
            get_user_agent_md_path("../escape")


class TestAgentDirStaysOffTheHeavyImportPath:
    """`get_agent_dir` is a path join reached by client-side CLI commands.

    `AGENTS.md` § Startup performance forbids importing `deepagents` or
    LangChain on that path. The reserved-name check briefly imported
    `deepagents_code.agent`, which pulls in both at module level and cost
    roughly 0.8s on commands that touch no model code.
    """


class TestBuildStreamConfigRecursionLimit:
    """`build_stream_config` carries the resolved graph step budget."""

    def test_omitted_when_nothing_is_configured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No configured limit leaves the server default in force.

        Sending an explicit value here would override `langgraph_api`'s own
        default, so the key must be absent rather than `None`.
        """
        import deepagents_code.config as config_mod
        from deepagents_code import _env_vars, model_config
        from deepagents_code.configuration import service

        monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)
        monkeypatch.delenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", raising=False)
        empty = tmp_path / "config.toml"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", empty)
        service.invalidate_config_sources()
        model_config.clear_caches()
        try:
            assert "recursion_limit" not in config_mod.build_stream_config(
                "thread-123", assistant_id=None
            )
        finally:
            service.invalidate_config_sources()
            model_config.clear_caches()

    def test_resolved_limit_reaches_the_run_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A configured limit rides in the run config, not just the binding.

        The graph runs inside the `langgraph dev` server, which stamps its own
        `recursion_limit` over whatever `create_cli_agent` bound onto the
        compiled graph. The run config is the only channel that survives, so a
        regression here silently restores the server default.
        """
        import deepagents_code.config as config_mod
        from deepagents_code import _env_vars, model_config
        from deepagents_code.configuration import service

        monkeypatch.setenv(_env_vars.RECURSION_LIMIT, "3000")
        empty = tmp_path / "config.toml"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", empty)
        service.invalidate_config_sources()
        model_config.clear_caches()
        try:
            config = config_mod.build_stream_config("thread-123", assistant_id=None)
        finally:
            service.invalidate_config_sources()
            model_config.clear_caches()

        assert config["recursion_limit"] == 3000

    def test_inherited_langgraph_limit_reaches_the_run_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The upstream environment fallback is sent with each run."""
        import deepagents_code.config as config_mod
        from deepagents_code import _env_vars, model_config
        from deepagents_code.configuration import service

        monkeypatch.delenv(_env_vars.RECURSION_LIMIT, raising=False)
        monkeypatch.setenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "12000")
        empty = tmp_path / "config.toml"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", empty)
        service.invalidate_config_sources()
        model_config.clear_caches()
        try:
            config = config_mod.build_stream_config("thread-123", assistant_id=None)
        finally:
            service.invalidate_config_sources()
            model_config.clear_caches()

        assert config["recursion_limit"] == 12_000
