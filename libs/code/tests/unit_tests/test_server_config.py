"""Tests for _server_config helpers and ServerConfig invariants."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import (
    ServerConfig,
    _interpreter_suppressed_by_sandbox,
    _normalize_path,
)

if TYPE_CHECKING:
    from pathlib import Path

# ------------------------------------------------------------------
# _read_env_bool
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# _read_env_json
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# _read_env_int
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# _read_env_str
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# _read_env_optional_bool
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# _normalize_path
# ------------------------------------------------------------------


class TestNormalizePath:
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
    def test_sandbox_type_valid_preserved(self) -> None:
        config = ServerConfig(sandbox_type="modal")
        assert config.sandbox_type == "modal"


class TestServerConfigInterpreterDefault:
    """Tests for sandbox-aware interpreter default resolution."""

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

    @staticmethod
    def _write_default(tmp_path: Path, *, enabled: bool) -> None:
        (tmp_path / "config.toml").write_text(
            f"[interpreter]\nenable_interpreter = {str(enabled).lower()}\n",
            encoding="utf-8",
        )

    def test_local_none_false_uses_resolver_default(self, tmp_path: Path) -> None:
        self._write_default(tmp_path, enabled=False)
        config = self._build(sandbox_type="none", enable_interpreter=None)

        assert config.enable_interpreter is False

    def test_local_none_true_uses_resolver_default(self, tmp_path: Path) -> None:
        self._write_default(tmp_path, enabled=True)
        config = self._build(sandbox_type="none", enable_interpreter=None)

        assert config.enable_interpreter is True

    def test_local_explicit_false_is_preserved(self, tmp_path: Path) -> None:
        # An explicit `False` must win over a `True` config default rather than
        # falling through to the settings lookup.
        self._write_default(tmp_path, enabled=True)
        config = self._build(sandbox_type="none", enable_interpreter=False)

        assert config.enable_interpreter is False

    def test_empty_sandbox_is_treated_as_local(self, tmp_path: Path) -> None:
        # An empty-string sandbox is falsy and must not be mistaken for a remote
        # backend, which would silently disable the interpreter.
        self._write_default(tmp_path, enabled=True)
        config = self._build(sandbox_type="", enable_interpreter=None)

        assert config.enable_interpreter is True

    def test_remote_none_disables_interpreter(self, tmp_path: Path) -> None:
        self._write_default(tmp_path, enabled=True)
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


# ------------------------------------------------------------------
# ServerConfig round-trip edge cases
# ------------------------------------------------------------------


class TestServerConfigEdgeCases:
    def test_empty_sandbox_treated_as_local(self) -> None:
        # An empty-string sandbox is falsy and must count as local, so the
        # advisory does not fire spuriously.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="", local_default=True
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

    def test_not_suppressed_when_default_off(self) -> None:
        # A user who disabled the interpreter in config should not be nagged.
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="daytona", local_default=False
        )

    def test_not_suppressed_when_local(self) -> None:
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type=None, local_default=True
        )

    def test_not_suppressed_when_sandbox_none_string(self) -> None:
        assert not _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="none", local_default=True
        )

    def test_suppressed_when_remote_and_default_on(self) -> None:
        # Unset intent + remote sandbox + default-on = a silent drop worth a heads-up.
        assert _interpreter_suppressed_by_sandbox(
            enable_interpreter=None, sandbox_type="daytona", local_default=True
        )

    def test_workspace_claim_partitions_every_policy_field(self) -> None:
        config = ServerConfig()

        assert set(config.to_workspace_payload()) == set(
            config.to_session_workspace_claim()
        ) | set(config.to_project_workspace_policy())
        assert not (
            set(config.to_session_workspace_claim())
            & set(config.to_project_workspace_policy())
        )
        assert {"no_mcp", "allow_fs_tools"} <= set(config.to_session_workspace_claim())

    def test_session_fingerprint_excludes_every_project_field(self) -> None:
        baseline = ServerConfig()
        project_changed = ServerConfig(
            extension_paths=("/tmp/extension.py",),
            mcp_config_path="/tmp/mcp.json",
            sandbox_setup="/tmp/setup.sh",
            trust_project_extensions=True,
            trust_project_mcp=True,
        )

        assert (
            baseline.session_workspace_fingerprint()
            == project_changed.session_workspace_fingerprint()
        )
        assert (
            baseline.workspace_fingerprint() != project_changed.workspace_fingerprint()
        )
        assert (
            baseline.session_workspace_fingerprint()
            != ServerConfig(no_mcp=True).session_workspace_fingerprint()
        )
        assert (
            baseline.session_workspace_fingerprint()
            != ServerConfig(
                allow_fs_tools=["read_file"]
            ).session_workspace_fingerprint()
        )

    def test_second_project_resolves_its_persisted_extension_trust(
        self, tmp_path: Path
    ) -> None:
        launch = tmp_path / "launch"
        other = tmp_path / "other"
        launch.mkdir()
        other.mkdir()
        config = ServerConfig(
            cwd=str(launch),
            project_root=str(launch),
            trust_project_extensions=False,
        )

        with patch(
            "deepagents_code.extensions.trust.is_project_extensions_trusted",
            return_value=True,
        ) as trusted:
            resolved = config.resolve_workspace(str(other), str(other))

        assert resolved.trust_project_extensions is True
        trusted.assert_called_once_with(str(other))

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
