"""Unit and subprocess tests for the immutable launch path snapshot."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol
from unittest import mock

import pytest

from deepagents_code._paths import (
    PATHS,
    DeepAgentsHomeError,
    DeepAgentsPathSnapshot,
    PathState,
    _capture_paths,
    classify_path,
    get_deepagents_home,
    project_paths,
)


class InstallProfileSnapshot(Protocol):
    """Shape of the `install_profile_snapshot` fixture from `conftest`.

    Declared locally because `tests.unit_tests` is not an importable package,
    so the fixture's own Protocol cannot be imported here.
    """

    def __call__(
        self, root: Path | str | None, *, launch_home: Path
    ) -> DeepAgentsPathSnapshot:
        """Install the snapshot and return it."""
        ...


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

    def test_default_display_keeps_tilde_readable(self, tmp_path: Path) -> None:
        """Default profile diagnostics use the familiar compact spelling."""
        snapshot = _capture_paths(None, launch_home=tmp_path)

        assert snapshot.display(snapshot.profile.config_file) == (
            "~/.deepagents/config.toml"
        )

    def test_configured_display_uses_effective_absolute_path(
        self, tmp_path: Path
    ) -> None:
        """Configured profile diagnostics show the normalized effective path."""
        snapshot = _capture_paths(
            str(tmp_path / "profiles" / ".." / "custom"), launch_home=tmp_path
        )

        assert snapshot.display(snapshot.profile.config_file) == str(
            tmp_path / "custom" / "config.toml"
        )

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
        project = project_paths(tmp_path / "repo")

        assert snapshot.profile.config_file == tmp_path / "profile" / "config.toml"
        assert snapshot.profile.mcp_tokens_dir == (
            tmp_path / "profile" / ".state" / "mcp-tokens"
        )
        assert snapshot.installation.managed_bin_dir.is_absolute()
        assert project.root_mcp_config_file == tmp_path / "repo" / ".mcp.json"
        assert project.config_mcp_config_file == (
            tmp_path / "repo" / ".deepagents" / ".mcp.json"
        )

    def test_profiles_share_installation_paths(self, tmp_path: Path) -> None:
        """Changing profile selection cannot move shared tool resources or locks."""
        first = _capture_paths(str(tmp_path / "first"), launch_home=tmp_path)
        second = _capture_paths(str(tmp_path / "second"), launch_home=tmp_path)

        assert first.profile.root != second.profile.root
        assert first.installation == second.installation
        assert first.installation.managed_bin_dir.is_relative_to(
            first.installation.root
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

    def test_absolute_override_launches_when_home_lookup_fails(
        self, tmp_path: Path
    ) -> None:
        """An absolute profile does not require a resolvable OS home."""
        configured = tmp_path / "profile"
        code = """
import os
from pathlib import Path
from unittest.mock import patch
with patch.object(Path, "home", side_effect=RuntimeError("home unavailable")):
    from deepagents_code._paths import PATHS
assert PATHS.launch_home is None
print(PATHS.profile.root)
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

    def test_project_dotenv_cannot_reclassify_project_mcp(self, tmp_path: Path) -> None:
        """A committed dotenv plus MCP config cannot self-approve the server."""
        project = tmp_path / "repo"
        project.mkdir()
        (project / ".mcp.json").write_text("{}")
        (project / ".env").write_text(f"DEEPAGENTS_HOME={project}\n")
        configured = tmp_path / "safe-profile"
        code = """
import os
from pathlib import Path
from deepagents_code._paths import PATHS
from deepagents_code.config import _load_dotenv
from deepagents_code.mcp_tools import MCPConfigScope, discover_mcp_config_sources
from deepagents_code.project_utils import ProjectContext
project = Path(os.environ["TEST_PROJECT"])
os.environ.pop("DEEPAGENTS_HOME", None)
_load_dotenv(start_path=project)
context = ProjectContext(user_cwd=project, project_root=project)
sources = discover_mcp_config_sources(project_context=context)
assert PATHS.profile.root != project
assert "DEEPAGENTS_HOME" not in os.environ
assert sources and all(source.scope is MCPConfigScope.PROJECT for source in sources)
print(PATHS.profile.root)
"""
        env = _subprocess_env(home=tmp_path, configured=str(configured))
        env["TEST_PROJECT"] = str(project)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
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


class TestDegenerateProfileRoots:
    """Reject resolved roots that would scatter profile state.

    None of these are typos the user can see in the resulting behavior: each
    one silently produces a "working" profile whose files land somewhere
    surprising, so they must fail at launch instead.
    """

    def test_rejects_home_directory_itself(self, tmp_path: Path) -> None:
        """`DEEPAGENTS_HOME=~/` must not make `~/.env` the trusted dotenv."""
        with pytest.raises(DeepAgentsHomeError, match="home directory itself"):
            _capture_paths("~/", launch_home=tmp_path)

    def test_rejects_filesystem_root(self, tmp_path: Path) -> None:
        """A root of `/` would place credentials in `/.state/auth.json`."""
        with pytest.raises(DeepAgentsHomeError, match="filesystem root"):
            _capture_paths("/", launch_home=tmp_path)

    def test_rejects_dotdot_chain_escaping_to_root(self, tmp_path: Path) -> None:
        """A `..` chain that normalizes to `/` is rejected like a literal `/`."""
        with pytest.raises(DeepAgentsHomeError, match="filesystem root"):
            _capture_paths("/tmp/../..", launch_home=tmp_path)

    def test_rejects_existing_file(self, tmp_path: Path) -> None:
        """An existing regular file cannot hold a profile."""
        target = tmp_path / "not-a-dir"
        target.write_text("")
        with pytest.raises(DeepAgentsHomeError, match="not a"):
            _capture_paths(str(target), launch_home=tmp_path)

    def test_accepts_nested_directory_under_home(self, tmp_path: Path) -> None:
        """The rejections above do not catch an ordinary nested profile."""
        snapshot = _capture_paths("~/profiles/work", launch_home=tmp_path)
        assert snapshot.profile.root == tmp_path / "profiles" / "work"


class TestUnresolvableHome:
    """A missing home directory must stay actionable, not raise `RuntimeError`.

    `Path.home()` raises when `$HOME` is unset and the uid has no passwd entry
    — a bare container, or a cleared-environment service unit. `_paths` is
    imported by many modules, so an unguarded raise there makes all of them
    unimportable with a traceback rather than a message.
    """

    def test_unresolvable_home_is_reported_as_home_error(self) -> None:
        """The `RuntimeError` is translated and names both escape hatches."""
        with (
            mock.patch.object(Path, "home", side_effect=RuntimeError("no home")),
            pytest.raises(DeepAgentsHomeError) as excinfo,
        ):
            _capture_paths(None)

        message = str(excinfo.value)
        assert "$HOME" in message
        assert "DEEPAGENTS_HOME" in message

    def test_absolute_profile_survives_an_unresolvable_home(
        self, tmp_path: Path
    ) -> None:
        """An absolute profile is the documented way out of a missing home.

        The home is still looked up *best-effort*, to reject a profile that is
        the home directory itself, but a failure there must not fail the launch
        and must not end up in the snapshot.
        """
        configured = tmp_path / "profile"
        with mock.patch.object(Path, "home", side_effect=RuntimeError("no home")):
            snapshot = _capture_paths(str(configured))

        assert snapshot.profile.root == configured
        assert snapshot.launch_home is None

    def test_relative_home_names_home_not_just_the_path(self) -> None:
        """A relative `$HOME` says which variable to fix."""
        with pytest.raises(DeepAgentsHomeError) as excinfo:
            _capture_paths(None, launch_home=Path("relative/dir"))

        assert "$HOME" in str(excinfo.value)


class TestDefaultProfileDisplayThroughConsumers:
    """The `~`-abbreviating branch must be exercised through real call sites.

    The suite runs with `DEEPAGENTS_HOME` set (see `conftest`), so
    `uses_default_profile` is `False` and `display()` is the identity function
    everywhere. An assertion comparing a message against `PATHS.display(...)`
    therefore passes even if the call site forgot to call `display` at all.
    These tests install a *default* snapshot so the abbreviation is real.
    """

    def test_config_suppress_hint_abbreviates_default_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """A default-profile hint shows `~/.deepagents`, not an absolute path."""
        from deepagents_code import main

        install_profile_snapshot(None, launch_home=tmp_path)

        hint = main._suppress_hint_cli("some_key")

        assert "~/.deepagents/config.toml" in hint
        assert str(tmp_path) not in hint

    def test_agent_skills_dir_hint_abbreviates_default_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """The same holds for a path rendered from a different module."""
        from deepagents_code import ui

        install_profile_snapshot(None, launch_home=tmp_path)

        rendered = ui.PATHS.display(ui.PATHS.profile.agent_skills_dir("<agent>"))

        assert rendered == "~/.deepagents/<agent>/skills"

    def test_configured_profile_still_renders_literally(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """The other branch keeps the absolute path so the profile is visible."""
        from deepagents_code import main

        configured = tmp_path / "custom"
        install_profile_snapshot(configured, launch_home=tmp_path)

        assert str(configured) in main._suppress_hint_cli("some_key")


class TestConfiguredProfileNotice:
    """Launch must say which profile it selected when it is not the default."""

    def test_no_notice_for_default_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """The common case stays silent."""
        from deepagents_code import main

        install_profile_snapshot(None, launch_home=tmp_path)

        assert main._configured_profile_notice() is None

    def test_existing_configured_profile_is_named(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """An existing profile is reported quietly."""
        from deepagents_code import main

        configured = tmp_path / "custom"
        configured.mkdir()
        install_profile_snapshot(configured, launch_home=tmp_path)

        notice = main._configured_profile_notice()

        assert notice is not None
        assert str(configured) in notice

    def test_missing_configured_profile_warns_about_empty_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """A typo must not look like lost credentials."""
        from deepagents_code import main

        configured = tmp_path / "typoed"
        install_profile_snapshot(configured, launch_home=tmp_path)

        notice = main._configured_profile_notice()

        assert notice is not None
        assert "new empty profile" in notice
        assert str(configured) in notice


class TestHomeDirectoryProfileSpellings:
    """Every spelling of "the profile is my home directory" must be rejected.

    `~/` and an absolute `/Users/me` name the same directory, so rejecting only
    the tilde form would leave the `~/.env`-as-trusted-dotenv hazard reachable.
    """

    def test_rejects_absolute_spelling_of_the_home_directory(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DeepAgentsHomeError, match="home directory itself"):
            _capture_paths(str(tmp_path), launch_home=tmp_path)

    def test_rejects_absolute_home_when_launch_home_is_implicit(
        self, tmp_path: Path
    ) -> None:
        """The check still applies when the home is only looked up to validate."""
        with (
            mock.patch.object(Path, "home", return_value=tmp_path),
            pytest.raises(DeepAgentsHomeError, match="home directory itself"),
        ):
            _capture_paths(str(tmp_path))

    def test_rejects_dot_normalized_spelling(self, tmp_path: Path) -> None:
        """A `.`/`..` chain that lands on the home directory is also rejected."""
        with pytest.raises(DeepAgentsHomeError, match="home directory itself"):
            _capture_paths(str(tmp_path / "sub" / ".."), launch_home=tmp_path)

    def test_unresolvable_home_does_not_block_an_absolute_profile(
        self, tmp_path: Path
    ) -> None:
        """Validation degrades to "cannot check" rather than failing the launch."""
        configured = tmp_path / "profile"
        with mock.patch.object(Path, "home", side_effect=RuntimeError("no home")):
            snapshot = _capture_paths(str(configured))

        assert snapshot.profile.root == configured
