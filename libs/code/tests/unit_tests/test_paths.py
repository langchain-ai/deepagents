"""Unit and subprocess tests for the immutable launch path snapshot."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepagents_code._paths import (
    PATHS,
    DeepAgentsHomeError,
    PathState,
    _capture_paths,
    classify_path,
    get_deepagents_home,
)


class TestClassifyPath:
    """Tests for the shared path classifier."""

    def test_existing_path(self, tmp_path: Path) -> None:
        """A path that exists classifies as EXISTS."""
        target = tmp_path / "present"
        target.write_text("x")
        assert classify_path(target) is PathState.EXISTS

    def test_existing_directory(self, tmp_path: Path) -> None:
        """A directory that exists classifies as EXISTS."""
        assert classify_path(tmp_path) is PathState.EXISTS

    def test_missing_path(self, tmp_path: Path) -> None:
        """A path that does not exist classifies as MISSING."""
        assert classify_path(tmp_path / "absent") is PathState.MISSING

    def test_unreadable_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An OSError from `Path.stat()` classifies as UNREADABLE.

        Simulates EACCES on a parent directory rather than relying on chmod,
        which is ignored when running as root and varies by platform.
        """

        def _raise(self: Path) -> object:  # noqa: ARG001  # must match Path.stat signature
            msg = "permission denied"
            raise PermissionError(msg)

        monkeypatch.setattr(Path, "stat", _raise)
        assert classify_path(Path("/anything")) is PathState.UNREADABLE

    def test_state_value_is_json_friendly(self) -> None:
        """`PathState` is a str enum, so its value serializes directly."""
        assert PathState.UNREADABLE == "unreadable"
        assert PathState.EXISTS.value == "exists"


class TestGetDeepagentsHome:
    """Tests for launch-time profile-root validation and immutability."""

    def test_defaults_to_home(self, tmp_path: Path) -> None:
        """The default remains `~/.deepagents`."""
        snapshot = _capture_paths(None, launch_home=tmp_path)
        assert snapshot.profile.root == tmp_path / ".deepagents"
        assert snapshot.uses_default_profile

    def test_uses_absolute_override(self, tmp_path: Path) -> None:
        """`DEEPAGENTS_HOME` replaces the default root."""
        configured = tmp_path / "data" / ".." / "deepagents"
        snapshot = _capture_paths(str(configured), launch_home=tmp_path)
        assert snapshot.profile.root == tmp_path / "deepagents"
        assert not snapshot.uses_default_profile

    def test_expands_leading_tilde(self, tmp_path: Path) -> None:
        """A leading `~/` expands against the captured launch-user home."""
        snapshot = _capture_paths("~/profiles/dcode", launch_home=tmp_path)
        assert snapshot.profile.root == tmp_path / "profiles" / "dcode"

    def test_empty_override_is_unset(self, tmp_path: Path) -> None:
        """An empty override preserves the default root."""
        snapshot = _capture_paths("", launch_home=tmp_path)
        assert snapshot.profile.root == tmp_path / ".deepagents"
        assert snapshot.uses_default_profile

    def test_rejects_relative_override(self, tmp_path: Path) -> None:
        """Other relative values fail instead of depending on cwd."""
        with pytest.raises(DeepAgentsHomeError, match="absolute path"):
            _capture_paths("profiles/dcode", launch_home=tmp_path)

    @pytest.mark.parametrize("configured", ["~other/dcode", "~other", "~"])
    def test_rejects_other_tilde_forms(self, configured: str, tmp_path: Path) -> None:
        """`~user` and other unsupported tilde forms fail actionably."""
        with pytest.raises(DeepAgentsHomeError, match="~user"):
            _capture_paths(configured, launch_home=tmp_path)

    def test_current_snapshot_ignores_env_and_cwd_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The process snapshot cannot move after import."""
        captured = get_deepagents_home()
        monkeypatch.setenv("DEEPAGENTS_HOME", str(tmp_path / "other"))
        monkeypatch.chdir(tmp_path)
        assert get_deepagents_home() == captured == PATHS.profile.root

    def test_snapshot_models_profile_install_and_project_paths(
        self, tmp_path: Path
    ) -> None:
        """The frozen model keeps ownership scopes explicit."""
        snapshot = _capture_paths(str(tmp_path / "profile"), launch_home=tmp_path)
        project = snapshot.for_project(tmp_path / "repo")

        assert snapshot.profile.config_file == tmp_path / "profile" / "config.toml"
        assert snapshot.profile.mcp_tokens_dir == (
            tmp_path / "profile" / ".state" / "mcp-tokens"
        )
        assert snapshot.installation.managed_bin_dir.is_absolute()
        assert project.root_mcp_config_file == tmp_path / "repo" / ".mcp.json"
        assert project.config_mcp_config_file == (
            tmp_path / "repo" / ".deepagents" / ".mcp.json"
        )


def _subprocess_env(*, home: Path, configured: str | None) -> dict[str, str]:
    """Return a synthetic launch environment without reading secret files."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    if configured is None:
        env.pop("DEEPAGENTS_HOME", None)
    else:
        env["DEEPAGENTS_HOME"] = configured
    return env


class TestLaunchSnapshotSubprocess:
    """Regressions that must exercise a clean import generation."""

    def test_import_time_consumers_and_server_agree_across_cwd(
        self, tmp_path: Path
    ) -> None:
        """Client constants and a server env use the same launch snapshot."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        configured = tmp_path / "profiles" / "main"
        code = """
import os
from pathlib import Path
from deepagents_code._paths import PATHS, get_deepagents_home
from deepagents_code import config, model_config
from deepagents_code.client.launch.server import _build_server_env
captured = get_deepagents_home()
os.environ["DEEPAGENTS_HOME"] = "/tmp/attacker-change"
os.chdir(os.environ["TEST_SECOND_CWD"])
assert get_deepagents_home() == captured
assert model_config.DEFAULT_CONFIG_DIR == captured
assert config._GLOBAL_DOTENV_PATH == captured / ".env"
assert _build_server_env()["DEEPAGENTS_HOME"] == str(captured)
assert PATHS.profile.root == captured
print(captured)
"""
        env = _subprocess_env(home=tmp_path, configured=str(configured))
        env["TEST_SECOND_CWD"] = str(second)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=first,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(configured)

    def test_deleted_cwd_does_not_break_profile_lookup(self, tmp_path: Path) -> None:
        """Profile lookup never calls `Path.cwd()` after launch."""
        configured = tmp_path / "profile"
        code = """
import os
import tempfile
from pathlib import Path
from deepagents_code._paths import get_deepagents_home
captured = get_deepagents_home()
working = Path(tempfile.mkdtemp())
os.chdir(working)
working.rmdir()
assert get_deepagents_home() == captured
print(captured)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=_subprocess_env(home=tmp_path, configured=str(configured)),
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(configured)

    @pytest.mark.parametrize("configured", ["relative/profile", "~other/profile"])
    def test_invalid_launch_value_is_actionable(
        self, configured: str, tmp_path: Path
    ) -> None:
        """The CLI reports invalid inherited values without a traceback."""
        proc = subprocess.run(
            [sys.executable, "-m", "deepagents_code", "--version"],
            env=_subprocess_env(home=tmp_path, configured=configured),
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "Invalid DEEPAGENTS_HOME" in proc.stderr
        assert "Traceback" not in proc.stderr
