"""Unit and subprocess tests for the immutable launch path snapshot."""

import ast
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol
from unittest import mock
from unittest.mock import patch

import pytest

from deepagents_code import _paths as _paths_module
from deepagents_code._paths import (
    PATHS,
    DeepAgentsHomeError,
    DeepAgentsPathSnapshot,
    PathState,
    _capture_paths,
    classify_path,
    get_deepagents_home,
    harden_state_dir,
    probe_writable,
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

    def test_repeated_tilde_separators_stay_under_home(self, tmp_path: Path) -> None:
        """Extra separators cannot turn the suffix into an absolute path."""
        snapshot = _capture_paths("~//profiles/dcode", launch_home=tmp_path)
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
    env.pop("DEEPAGENTS_HOME_IS_DEFAULT", None)
    if configured is None:
        env.pop("DEEPAGENTS_HOME", None)
    else:
        env["DEEPAGENTS_HOME"] = configured
    return env


class TestHomeCheckSkipped:
    """An unresolvable home silently disables a security check.

    `_paths` is imported before any log handler exists, so a log line about it
    would be invisible even under `--debug`. The snapshot records it instead so
    `dcode doctor` can report it.
    """

    def test_not_skipped_when_the_home_resolves(self, tmp_path: Path) -> None:
        """The ordinary case records nothing."""
        snapshot = _capture_paths(str(tmp_path / "profile"), launch_home=tmp_path)

        assert snapshot.home_check_skipped is False

    def test_recorded_when_the_home_cannot_be_resolved(self, tmp_path: Path) -> None:
        """An absolute profile still launches, but the skip is recorded."""
        configured = tmp_path / "profile"
        with mock.patch.object(
            Path, "home", side_effect=RuntimeError("home unavailable")
        ):
            snapshot = _capture_paths(str(configured))

        assert snapshot.profile.root == configured
        assert snapshot.home_check_skipped is True

    def test_update_check_import_uses_profile_cache_without_home(
        self, tmp_path: Path
    ) -> None:
        """Startup consumers cannot reintroduce a required home lookup."""
        configured = tmp_path / "profile"
        env = _subprocess_env(home=tmp_path, configured=str(configured))
        env.pop("LOCALAPPDATA", None)
        env.pop("XDG_CACHE_HOME", None)
        code = """
from pathlib import Path

def unavailable_home(cls):
    raise RuntimeError("home unavailable")

Path.home = classmethod(unavailable_home)
from deepagents_code.update_check import UPDATE_LOG_DIR
print(UPDATE_LOG_DIR)
"""

        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == str(
            configured / ".state" / "cache" / "deepagents-code" / "update_logs"
        )


class TestDisplayOutsideTheProfileRoot:
    """A path outside the profile must not be given a bogus `~` prefix."""

    def test_installation_path_renders_literally_under_a_default_profile(
        self, tmp_path: Path
    ) -> None:
        """This is what keeps doctor from mangling installation paths."""
        snapshot = _capture_paths(None, launch_home=tmp_path)

        rendered = snapshot.display(snapshot.installation.managed_bin_dir)

        assert rendered == str(snapshot.installation.managed_bin_dir)
        assert not rendered.startswith("~")


class TestProbeWritable:
    """`probe_writable` decides which shared directory a process may use."""

    def test_reports_a_writable_directory(self, tmp_path: Path) -> None:
        """A usable directory is created and leaves no probe behind."""
        directory = tmp_path / "fresh"

        probe_writable(directory)

        assert directory.is_dir()
        assert list(directory.iterdir()) == []

    def test_a_stale_probe_does_not_make_a_directory_look_unusable(
        self, tmp_path: Path
    ) -> None:
        """A leftover probe must not demote a writable shared directory.

        These directories are shared across profiles and processes, so a
        PID-named probe could collide with a crashed peer's leftover file. The
        probe would then fail on a directory whose very contents prove it is
        writable, and the caller would fall back — splitting the lock it was
        trying to share.
        """
        directory = tmp_path / "shared"
        directory.mkdir()
        (directory / f".deepagents-probe-{os.getpid()}").touch()

        probe_writable(directory)

    def test_does_not_delete_a_peer_probe(self, tmp_path: Path) -> None:
        """Probing must not remove another process's in-flight probe file."""
        directory = tmp_path / "shared"
        directory.mkdir()
        peer = directory / f".deepagents-probe-{os.getpid()}"
        peer.touch()

        probe_writable(directory)

        assert peer.exists()

    def test_reports_an_unwritable_directory(self, tmp_path: Path) -> None:
        """A directory that cannot accept files raises for the caller."""
        directory = tmp_path / "locked"
        directory.mkdir(mode=0o500)
        try:
            with pytest.raises(OSError, match="Permission denied"):
                probe_writable(directory)
        finally:
            directory.chmod(0o700)

    def test_existing_dir_that_rejects_unlink_is_still_writable(
        self, tmp_path: Path
    ) -> None:
        """A failed cleanup must not be reported as an unwritable directory."""
        directory = tmp_path / "sticky"
        directory.mkdir()
        try:
            with patch.object(Path, "unlink", side_effect=OSError("cannot unlink")):
                probe_writable(directory)
        finally:
            _paths_module._LEAKED_PROBE_DIRS.discard(str(directory))
            for probe in directory.glob(".deepagents-probe-*"):
                probe.unlink()

    def test_warns_on_first_rejected_unlink_only(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every process makes its first stranded probe visible to the user."""
        directory = tmp_path / "sticky"
        directory.mkdir()
        try:
            with (
                caplog.at_level(logging.WARNING, logger="deepagents_code._paths"),
                patch.object(Path, "unlink", side_effect=OSError("cannot unlink")),
            ):
                probe_writable(directory)
                probe_writable(directory)

            messages = [
                record.message
                for record in caplog.records
                if record.message.startswith("Cannot remove write probes")
            ]
            assert len(messages) == 1
        finally:
            _paths_module._LEAKED_PROBE_DIRS.discard(str(directory))
            for probe in directory.glob(".deepagents-probe-*"):
                probe.unlink()


class TestDefaultProfileMarkerSurvivesChildProcesses:
    """`uses_default_profile` must not be destroyed at a process boundary.

    The parent re-exports `DEEPAGENTS_HOME` for every descendant, so without a
    separate marker a child re-derives the value through the absolute-path
    branch and concludes the profile was configured. That renders absolute
    paths in the server's system prompt (leaking the OS username) and makes a
    post-upgrade re-exec announce a profile the user never set.
    """

    _CHILD = (
        "from deepagents_code._paths import PATHS; "
        "print(PATHS.uses_default_profile, PATHS.display(PATHS.profile.config_file))"
    )

    def _run(self, env: dict[str, str]) -> str:
        code = f"""
import subprocess, sys
from deepagents_code._paths import PATHS
print(PATHS.uses_default_profile, PATHS.display(PATHS.profile.config_file))
proc = subprocess.run([sys.executable, "-c", {self._CHILD!r}],
                      check=True, capture_output=True, text=True)
print(proc.stdout.strip())
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    def test_default_profile_stays_default_in_a_child(self, tmp_path: Path) -> None:
        """A defaulted profile still abbreviates to `~` one process down."""
        out = self._run(_subprocess_env(home=tmp_path, configured=None))

        parent, child = out.strip().splitlines()
        assert parent == child
        assert child == "True ~/.deepagents/config.toml"

    def test_configured_profile_stays_configured_in_a_child(
        self, tmp_path: Path
    ) -> None:
        """A configured profile keeps rendering literally, marker never set."""
        configured = tmp_path / "profile"
        out = self._run(_subprocess_env(home=tmp_path, configured=str(configured)))

        parent, child = out.strip().splitlines()
        assert parent == child
        assert child == f"False {configured / 'config.toml'}"

    def test_marker_cannot_relabel_a_configured_profile(self, tmp_path: Path) -> None:
        """A forged marker is ignored: it is a display hint, not a trust input."""
        configured = tmp_path / "profile"
        env = _subprocess_env(home=tmp_path, configured=str(configured))
        env["DEEPAGENTS_HOME_IS_DEFAULT"] = "1"

        out = self._run(env)

        parent, _child = out.strip().splitlines()
        assert parent == f"False {configured / 'config.toml'}"

    def test_stale_marker_is_cleared_for_descendants(self, tmp_path: Path) -> None:
        """An inherited marker that no longer applies is removed, not passed on."""
        env = _subprocess_env(home=tmp_path, configured=str(tmp_path / "profile"))
        env["DEEPAGENTS_HOME_IS_DEFAULT"] = "1"
        code = (
            "import os; import deepagents_code._paths; "
            "print(os.environ.get('DEEPAGENTS_HOME_IS_DEFAULT'))"
        )

        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "None"

    def test_explicit_default_location_is_honored_via_marker(
        self, tmp_path: Path
    ) -> None:
        """The marker is honored when the root really is the default location."""
        env = _subprocess_env(home=tmp_path, configured=str(tmp_path / ".deepagents"))
        env["DEEPAGENTS_HOME_IS_DEFAULT"] = "1"

        out = self._run(env)

        parent, child = out.strip().splitlines()
        assert parent == child == "True ~/.deepagents/config.toml"


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
        (project / ".mcp.json").write_text(
            '{"mcpServers": {"evil": {"command": "/bin/echo", "args": []}}}'
        )
        (project / ".env").write_text(f"DEEPAGENTS_HOME={project}\n")
        configured = tmp_path / "safe-profile"
        # Assert on the outcome, not only the labelling: every other trust test
        # supplies the provenance itself, so this is the one place that proves
        # project scope actually reaches the filter and withholds the server.
        code = """
import asyncio, os
from pathlib import Path
from deepagents_code._paths import PATHS
from deepagents_code.config import _load_dotenv
from deepagents_code.mcp_tools import (
    MCPConfigScope,
    discover_mcp_config_sources,
    resolve_and_load_mcp_tools,
)
from deepagents_code.project_utils import ProjectContext
project = Path(os.environ["TEST_PROJECT"])
os.environ.pop("DEEPAGENTS_HOME", None)
_load_dotenv(start_path=project)
context = ProjectContext(user_cwd=project, project_root=project)
sources = discover_mcp_config_sources(project_context=context)
assert PATHS.profile.root != project
assert "DEEPAGENTS_HOME" not in os.environ
assert sources and all(source.scope is MCPConfigScope.PROJECT for source in sources)
tools, manager, servers = asyncio.run(
    resolve_and_load_mcp_tools(trust_project_mcp=None, project_context=context)
)
assert not servers, f"untrusted project server loaded: {servers!r}"
assert not tools, f"untrusted project tools loaded: {tools!r}"
assert manager is None
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

    def test_rejects_dangling_symlink(self, tmp_path: Path) -> None:
        """A profile symlink must resolve before launch treats it as creatable."""
        target = tmp_path / "missing-profile"
        link = tmp_path / "profile-link"
        link.symlink_to(target, target_is_directory=True)

        with pytest.raises(DeepAgentsHomeError, match="symlink whose target"):
            _capture_paths(str(link), launch_home=tmp_path)

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

    def test_system_prompt_abbreviates_the_default_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
    ) -> None:
        """The skills path in the system prompt is the highest-stakes call site.

        It goes to the model on every request, so an unabbreviated path there
        puts the operator's real home directory — and OS username — in front of
        the model. Call the real prompt builder: asserting against
        `PATHS.display(...)` would re-test `display` rather than the call site.
        """
        from deepagents_code.agent import get_system_prompt

        install_profile_snapshot(None, launch_home=tmp_path)

        prompt = get_system_prompt("<agent>")

        assert "~/.deepagents/<agent>/skills" in prompt
        assert str(tmp_path) not in prompt

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


class TestConfiguredProfileNoticeReachesEveryLaunchPath:
    """The notice only helps if the launch path a user takes actually prints it.

    It was wired into the non-interactive branch only, so the interactive TUI
    launch — how most users start dcode — showed nothing, leaving the exact
    "where did my settings go?" case the notice exists to prevent.
    """

    def test_source_prints_the_notice_on_both_session_launches(self) -> None:
        """Both session-launch branches call the printer."""
        import inspect

        from deepagents_code import main

        source = inspect.getsource(main.cli_main)

        assert source.count("_print_configured_profile_notice()") == 2

    def test_printer_is_not_inside_the_optional_tools_guard(self) -> None:
        """A markup failure here must not be reported as a tool-check failure."""
        import inspect

        from deepagents_code import main

        source = inspect.getsource(main.cli_main)
        notice_at = source.index("_print_configured_profile_notice()")
        guard_at = source.index("Tool availability check skipped")

        assert notice_at < guard_at

    def test_printer_writes_to_stderr(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Piped stdout stays clean; the notice goes to stderr."""
        from deepagents_code import main

        configured = tmp_path / "custom"
        configured.mkdir()
        install_profile_snapshot(configured, launch_home=tmp_path)

        main._print_configured_profile_notice()

        captured = capsys.readouterr()
        # Rich hard-wraps to the console width, and will break mid-path, so
        # compare with every space removed.
        err = "".join(captured.err.split())
        assert str(configured) in err
        assert captured.out == ""

    def test_printer_stays_silent_for_the_default_profile(
        self,
        tmp_path: Path,
        install_profile_snapshot: InstallProfileSnapshot,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The common case prints nothing at all."""
        from deepagents_code import main

        install_profile_snapshot(None, launch_home=tmp_path)

        main._print_configured_profile_notice()

        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""


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

    def test_rejects_a_symlinked_spelling_of_the_home_directory(
        self, tmp_path: Path
    ) -> None:
        """A symlink to the home directory is the same trust hazard.

        Path construction is lexical on purpose, so a `==` against the home
        misses this spelling entirely and `profile.dotenv_file` becomes the
        user's generic `~/.env`.
        """
        home = tmp_path / "home"
        home.mkdir()
        link = tmp_path / "link"
        link.symlink_to(home, target_is_directory=True)

        with pytest.raises(DeepAgentsHomeError, match="home directory itself"):
            _capture_paths(str(link), launch_home=home)

    def test_rejects_a_case_variant_of_the_home_directory(self, tmp_path: Path) -> None:
        """On a case-insensitive filesystem `/Users/Me` is `/Users/me`."""
        home = tmp_path / "home"
        home.mkdir()
        variant = tmp_path / "HOME"
        if not variant.exists():
            pytest.skip("case-sensitive filesystem")

        with pytest.raises(DeepAgentsHomeError, match="home directory itself"):
            _capture_paths(str(variant), launch_home=home)

    def test_rejects_a_symlinked_spelling_of_the_filesystem_root(
        self, tmp_path: Path
    ) -> None:
        """A link to `/` would put credentials in `/.state/auth.json`."""
        link = tmp_path / "slash"
        link.symlink_to(Path("/"), target_is_directory=True)

        with pytest.raises(DeepAgentsHomeError, match="filesystem root"):
            _capture_paths(str(link), launch_home=tmp_path / "home")

    def test_accepts_a_subdirectory_reached_through_a_symlink(
        self, tmp_path: Path
    ) -> None:
        """Only identity with the home is rejected, not symlinks in general."""
        home = tmp_path / "home"
        (home / "profiles" / "main").mkdir(parents=True)
        link = tmp_path / "link"
        link.symlink_to(home / "profiles", target_is_directory=True)

        snapshot = _capture_paths(str(link / "main"), launch_home=home)

        assert snapshot.profile.root == link / "main"

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


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
class TestHardenStateDir:
    """The state directory holds conversation content in mode-0644 files.

    Its permission bits are the only thing keeping another local user out, so
    these tests pin the directory itself rather than any single file under it.
    """

    def test_creates_the_state_directory_owner_only(
        self, install_profile_snapshot: InstallProfileSnapshot, tmp_path: Path
    ) -> None:
        """A fresh profile never passes through a world-readable state."""
        snapshot = install_profile_snapshot(tmp_path / "profile", launch_home=tmp_path)

        harden_state_dir()

        assert snapshot.profile.state_dir.stat().st_mode & 0o777 == 0o700

    def test_restricts_an_existing_permissive_state_directory(
        self, install_profile_snapshot: InstallProfileSnapshot, tmp_path: Path
    ) -> None:
        """`mkdir(mode=...)` is a no-op on an existing directory.

        A profile created before the state directory was hardened must be
        repaired, so the explicit `chmod` has to run on the exist_ok path too.
        """
        snapshot = install_profile_snapshot(tmp_path / "profile", launch_home=tmp_path)
        snapshot.profile.state_dir.mkdir(parents=True)
        snapshot.profile.state_dir.chmod(0o755)

        harden_state_dir()

        assert snapshot.profile.state_dir.stat().st_mode & 0o777 == 0o700

    def test_a_refused_chmod_is_not_fatal(
        self, install_profile_snapshot: InstallProfileSnapshot, tmp_path: Path
    ) -> None:
        """CIFS/exFAT mounts refuse `chmod`; the directory is still usable."""
        install_profile_snapshot(tmp_path / "profile", launch_home=tmp_path)

        with mock.patch.object(Path, "chmod", side_effect=OSError("read-only")):
            harden_state_dir()

    def test_the_sessions_database_lands_in_a_hardened_directory(
        self, tmp_path: Path
    ) -> None:
        """The regression was that only the update lock hardened the directory.

        `sessions.py` creates it on its own path, so it must harden it too. It
        locates the database from `model_config.DEFAULT_STATE_DIR`, not from
        `PATHS.profile.state_dir`, so this patches the name it actually reads.
        """
        from deepagents_code import model_config, sessions as sessions_module

        state_dir = tmp_path / "profile" / ".state"
        with (
            mock.patch.object(model_config, "DEFAULT_STATE_DIR", state_dir),
            mock.patch.object(sessions_module, "_db_path", None),
        ):
            assert sessions_module.get_db_path() == state_dir / "sessions.db"

        assert state_dir.stat().st_mode & 0o777 == 0o700


class TestPathsBindingModulesDrift:
    """Guards `conftest._PATHS_BINDING_MODULES` against silent drift.

    `install_profile_snapshot` patches `PATHS` on each module in that tuple.
    A module that binds `PATHS` at import time but is absent from the tuple is
    not patched, so a test using the fixture reads the developer's real
    profile and still passes. That failure is invisible, which is why it needs
    a check rather than a documented grep.

    Only module-level imports count. A `from ... import PATHS` inside a
    function re-binds on each call and therefore already follows a patch of
    `deepagents_code._paths.PATHS`.
    """

    @staticmethod
    def _module_level_imports(tree: ast.Module) -> list[ast.ImportFrom]:
        """Return the `ImportFrom` nodes that execute at import time.

        Descends through module-level `if`/`try`/`with` blocks but not into
        functions or classes, whose imports run per call instead.

        Returns:
            The module-level `ImportFrom` nodes.
        """
        found: list[ast.ImportFrom] = []
        stack: list[ast.AST] = list(tree.body)
        while stack:
            node = stack.pop()
            if isinstance(node, ast.ImportFrom):
                found.append(node)
                continue
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            for field in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(getattr(node, field, []) or [])
        return found

    @classmethod
    def _modules_binding_paths(cls, package_root: Path, prefix: str) -> set[str]:
        """Return modules under `package_root` that bind `PATHS` at import time.

        Uses `ast` rather than a regex so a parenthesized or multi-line import
        is found too — exactly the spelling a line-oriented grep misses.

        Args:
            package_root: Directory holding the package's modules.
            prefix: Dotted module path the imports must name.

        Returns:
            Dotted module names relative to `package_root`.
        """
        found: set[str] = set()
        for source_file in sorted(package_root.rglob("*.py")):
            try:
                tree = ast.parse(source_file.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):  # pragma: no cover - defensive
                continue
            for node in cls._module_level_imports(tree):
                if node.module != prefix:
                    continue
                if not any(alias.name == "PATHS" for alias in node.names):
                    continue
                relative = source_file.relative_to(package_root).with_suffix("")
                parts = [p for p in relative.parts if p != "__init__"]
                found.add(".".join(parts))
        return found

    def test_the_tuple_matches_the_source(self) -> None:
        """Every import-time `PATHS` binding is listed, and nothing extra is."""
        from unit_tests.conftest import _PATHS_BINDING_MODULES

        discovered = self._modules_binding_paths(
            Path(_paths_module.__file__).parent, "deepagents_code._paths"
        )
        listed = set(_PATHS_BINDING_MODULES)

        assert discovered - listed == set(), (
            "These modules bind PATHS at import time but are missing from "
            "_PATHS_BINDING_MODULES, so install_profile_snapshot will not "
            "patch them and tests will silently read the real profile."
        )
        assert listed - discovered == set(), (
            "These entries in _PATHS_BINDING_MODULES no longer bind PATHS at "
            "import time; patching them creates an unused module attribute."
        )

    def test_a_parenthesized_import_is_detected(self, tmp_path: Path) -> None:
        """The scan catches the spelling the previous grep recipe missed.

        The documented recipe was `grep '^from deepagents_code._paths
        import.*PATHS'`, which cannot match a multi-line import. A module
        spelled this way would have been left unpatched with no failure.
        """
        (tmp_path / "wide.py").write_text(
            "from pkg._paths import (\n    PATHS,\n    classify_path,\n)\n"
        )

        assert self._modules_binding_paths(tmp_path, "pkg._paths") == {"wide"}

    def test_a_function_level_import_is_ignored(self, tmp_path: Path) -> None:
        """Deferred imports follow a patch already, so they must not be listed."""
        (tmp_path / "deferred.py").write_text(
            "def go():\n    from pkg._paths import PATHS\n    return PATHS\n"
        )

        assert self._modules_binding_paths(tmp_path, "pkg._paths") == set()


class TestUnreadableProfileRoot:
    """An unreadable root must fail once, with its real cause.

    `Path.is_symlink` swallows `OSError` and reports `False` under EACCES, and
    `classify_path` reports `UNREADABLE` rather than `EXISTS`. Before the
    explicit branch, such a root passed every guard and the launch continued.
    Each later access then failed on its own, and each failure looked like a
    first run.
    """

    def test_an_unreadable_root_is_rejected(self, tmp_path: Path) -> None:
        configured = tmp_path / "profile"
        configured.mkdir()

        with (
            mock.patch(
                "deepagents_code._paths.classify_path",
                return_value=PathState.UNREADABLE,
            ),
            pytest.raises(DeepAgentsHomeError, match="cannot be read"),
        ):
            _capture_paths(str(configured), launch_home=tmp_path)

    def test_the_message_names_permissions_not_a_broken_symlink(
        self, tmp_path: Path
    ) -> None:
        """The old symlink wording sent users after the wrong cause."""
        configured = tmp_path / "profile"
        configured.mkdir()

        with (
            mock.patch(
                "deepagents_code._paths.classify_path",
                return_value=PathState.UNREADABLE,
            ),
            pytest.raises(DeepAgentsHomeError) as exc_info,
        ):
            _capture_paths(str(configured), launch_home=tmp_path)

        assert "symlink" not in str(exc_info.value)
        assert "permissions" in str(exc_info.value)

    def test_a_statable_but_inaccessible_root_is_rejected(self, tmp_path: Path) -> None:
        """A directory still needs read and search access after `stat` succeeds."""
        configured = tmp_path / "profile"
        configured.mkdir()

        with (
            mock.patch("deepagents_code._paths.os.access", return_value=False),
            pytest.raises(DeepAgentsHomeError, match="cannot be read or searched"),
        ):
            _capture_paths(str(configured), launch_home=tmp_path)

    def test_a_missing_root_is_still_accepted(self, tmp_path: Path) -> None:
        """The profile root is created lazily, so absent is normal."""
        configured = tmp_path / "not-created-yet"

        snapshot = _capture_paths(str(configured), launch_home=tmp_path)

        assert snapshot.profile.root == configured


class TestHomeComparisonFailsClosed:
    """An indeterminate home comparison must not be read as "different".

    `_same_directory` exists to catch the non-lexical spellings of the home
    directory — a symlink, a case difference. Those are exactly the spellings
    that reach `samefile`, so answering `False` when it raises would accept the
    alias the guard exists to reject, and load `~/.env` as trusted config.
    """

    def test_a_permission_error_is_not_treated_as_different(
        self, tmp_path: Path
    ) -> None:
        configured = tmp_path / "profile"
        configured.mkdir()

        with (
            mock.patch.object(Path, "samefile", side_effect=PermissionError("denied")),
            pytest.raises(DeepAgentsHomeError, match="Cannot determine"),
        ):
            _capture_paths(str(configured), launch_home=tmp_path)

    def test_a_missing_path_is_still_a_real_answer(self, tmp_path: Path) -> None:
        """`FileNotFoundError` means "not the same", not "cannot tell"."""
        configured = tmp_path / "absent"

        with mock.patch.object(Path, "samefile", side_effect=FileNotFoundError("gone")):
            snapshot = _capture_paths(str(configured), launch_home=tmp_path)

        assert snapshot.profile.root == configured


class TestHomeResolvedOncePerLaunch:
    """An unresolvable home must warn once, not twice.

    `_resolve_profile_root` needs a home for the degenerate-root comparison,
    and the default-profile marker check needs the same value. Looking it up
    twice emitted the multi-line warning twice on one launch.
    """

    def test_the_warning_is_not_repeated(self, tmp_path: Path) -> None:
        configured = tmp_path / "profile"
        calls: list[None] = []
        real = _paths_module._resolve_launch_home

        def counting(launch_home: Path | None) -> Path:
            if launch_home is None:
                calls.append(None)
                msg = "no home"
                raise DeepAgentsHomeError(msg)
            return real(launch_home)

        with mock.patch.object(
            _paths_module, "_resolve_launch_home", side_effect=counting
        ):
            snapshot = _capture_paths(str(configured), default_marker=True)

        assert snapshot.home_check_skipped is True
        assert len(calls) == 1


class TestUserDeepagentsDirHelpers:
    """Free functions for user-level paths derived from `DEEPAGENTS_HOME`."""

    def test_uses_deepagents_home(
        self, tmp_path: Path, install_profile_snapshot: InstallProfileSnapshot
    ) -> None:
        """Agent profiles and instructions use the configured root."""
        from deepagents_code import _paths

        configured = tmp_path / "custom-home"
        install_profile_snapshot(str(configured), launch_home=tmp_path)

        assert _paths.user_deepagents_dir() == configured
        assert _paths.get_agent_dir("coder") == configured / "coder"
        assert _paths.get_user_agent_md_path("coder") == (
            configured / "coder" / "AGENTS.md"
        )


class TestGetProjectAgentMdPath:
    """Test `_paths.get_project_agent_md_path()`."""

    def test_returns_empty_list_when_no_project_root(self) -> None:
        """No project root has no project instructions."""
        from deepagents_code import _paths

        assert _paths.get_project_agent_md_path(None) == []

    def test_returns_existing_paths(self, tmp_path: Path) -> None:
        """Both supported project instruction files are returned in order."""
        from deepagents_code import _paths

        deepagents_dir = tmp_path / ".deepagents"
        deepagents_dir.mkdir()
        deepagents_md = deepagents_dir / "AGENTS.md"
        deepagents_md.write_text("inner")
        root_md = tmp_path / "AGENTS.md"
        root_md.write_text("root")

        assert _paths.get_project_agent_md_path(tmp_path) == [
            deepagents_md,
            root_md,
        ]

    def test_returns_empty_when_no_agents_md_files(self, tmp_path: Path) -> None:
        """A project without instruction files returns an empty list."""
        from deepagents_code import _paths

        assert _paths.get_project_agent_md_path(tmp_path) == []


class TestAgentsAliasDirectories:
    """Tests for `.agents` directory alias helpers."""

    def test_user_aliases_use_launch_home(self) -> None:
        """User aliases derive from the immutable launch home."""
        from deepagents_code import _paths

        assert _paths.user_agents_dir() == Path.home() / ".agents"
        assert _paths.get_user_agent_skills_dir() == (
            Path.home() / ".agents" / "skills"
        )

    def test_home_aliases_are_skipped_when_home_is_unresolvable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absolute profile stays usable without optional home aliases."""
        from deepagents_code import _paths
        from deepagents_code._paths import _capture_paths

        with patch.object(Path, "home", side_effect=RuntimeError("no home")):
            snapshot = _capture_paths(str(tmp_path / "profile"))
        monkeypatch.setattr(_paths, "PATHS", snapshot)

        assert _paths.user_agents_dir() is None
        assert _paths.get_user_agent_skills_dir() is None
        assert _paths.get_user_claude_skills_dir() is None

    def test_project_alias_uses_explicit_root(self, tmp_path: Path) -> None:
        """The project alias derives from the supplied root."""
        from deepagents_code import _paths

        assert _paths.get_project_agent_skills_dir(tmp_path) == (
            tmp_path / ".agents" / "skills"
        )

    def test_project_alias_without_project(self) -> None:
        """No project root has no project alias directory."""
        from deepagents_code import _paths

        assert _paths.get_project_agent_skills_dir(None) is None


class TestClaudeSkillsDirs:
    """Tests for `.claude/skills` directory helpers."""

    def test_user_alias_uses_launch_home(self) -> None:
        """The user Claude alias derives from the launch home."""
        from deepagents_code import _paths

        assert _paths.get_user_claude_skills_dir() == (
            Path.home() / ".claude" / "skills"
        )

    def test_project_alias_uses_explicit_root(self, tmp_path: Path) -> None:
        """The project Claude alias derives from the supplied root."""
        from deepagents_code import _paths

        assert _paths.get_project_claude_skills_dir(tmp_path) == (
            tmp_path / ".claude" / "skills"
        )

    def test_project_alias_without_project(self) -> None:
        """No project root has no project Claude alias directory."""
        from deepagents_code import _paths

        assert _paths.get_project_claude_skills_dir(None) is None


class TestReservedAgentNameGuards:
    """Agent profiles cannot overlap directories owned by dcode."""

    @pytest.mark.parametrize("name", ["bin", "plugins", "conversation_history"])
    def test_reserved_names_are_rejected(self, name: str) -> None:
        """Exact app-owned directory names are reserved."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="reserved"):
            _paths.get_agent_dir(name)

    def test_ordinary_names_still_resolve(self) -> None:
        """Non-reserved profile names continue to derive a path."""
        from deepagents_code import _paths

        assert _paths.get_agent_dir("coder").name == "coder"

    @pytest.mark.parametrize(
        "name", ["BIN", "Plugins", "CONVERSATION_HISTORY", "pLuGiNs"]
    )
    def test_case_aliases_are_rejected(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case aliases are reserved on case-insensitive filesystems."""
        from deepagents_code import _paths

        monkeypatch.setattr(sys, "platform", "darwin")

        with pytest.raises(ValueError, match="reserved"):
            _paths.get_agent_dir(name)

    def test_case_alias_is_allowed_on_case_sensitive_linux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A differently cased name is distinct on Linux."""
        from deepagents_code import _paths

        monkeypatch.setattr(sys, "platform", "linux")

        assert _paths.get_agent_dir("Plugins").name == "Plugins"

    def test_windows_trailing_space_alias_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows strips a trailing space before resolving a path."""
        from deepagents_code import _paths

        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(ValueError, match="reserved"):
            _paths.get_agent_dir("plugins ")

    def test_trailing_dot_is_invalid(self) -> None:
        """A trailing-dot alias is rejected by the character allowlist."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="Invalid agent name"):
            _paths.get_agent_dir("plugins.")

    def test_trailing_space_is_allowed_off_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trailing space is a distinct directory on POSIX."""
        from deepagents_code import _paths

        monkeypatch.setattr(sys, "platform", "linux")

        assert _paths.get_agent_dir("plugins ").name == "plugins "

    @pytest.mark.parametrize("name", ["bin", "plugins", "conversation_history"])
    def test_agent_md_accessor_rejects_reserved_names(self, name: str) -> None:
        """The instruction path accessor enforces the same reserved names."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="reserved"):
            _paths.get_user_agent_md_path(name)

    def test_agent_md_accessor_rejects_invalid_characters(self) -> None:
        """The instruction path accessor cannot escape the profile root."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="Invalid agent name"):
            _paths.get_user_agent_md_path("../escape")


class TestEnsureDirHelpers:
    """The `ensure_*` helpers create the directory they return."""

    def test_ensure_agent_dir_creates_it(
        self, tmp_path: Path, install_profile_snapshot: InstallProfileSnapshot
    ) -> None:
        """The agent profile directory is created under the profile root."""
        from deepagents_code import _paths

        configured = tmp_path / "custom-home"
        install_profile_snapshot(str(configured), launch_home=tmp_path)

        agent_dir = _paths.ensure_agent_dir("coder")
        assert agent_dir == configured / "coder"
        assert agent_dir.is_dir()

    def test_ensure_user_skills_dir_creates_it(
        self, tmp_path: Path, install_profile_snapshot: InstallProfileSnapshot
    ) -> None:
        """The user skills directory is created under the agent profile."""
        from deepagents_code import _paths

        configured = tmp_path / "custom-home"
        install_profile_snapshot(str(configured), launch_home=tmp_path)

        skills_dir = _paths.ensure_user_skills_dir("coder")
        assert skills_dir == configured / "coder" / "skills"
        assert skills_dir.is_dir()

    def test_ensure_project_skills_dir_creates_it(self, tmp_path: Path) -> None:
        """The project skills directory is created under `.deepagents`."""
        from deepagents_code import _paths

        skills_dir = _paths.ensure_project_skills_dir(tmp_path)
        assert skills_dir == tmp_path / ".deepagents" / "skills"
        assert skills_dir is not None
        assert skills_dir.is_dir()

    def test_ensure_project_skills_dir_without_project(self) -> None:
        """No project root means there is no directory to create."""
        from deepagents_code import _paths

        assert _paths.ensure_project_skills_dir(None) is None

    def test_get_built_in_skills_dir_is_packaged(self) -> None:
        """The built-in skills directory is part of the package tree."""
        from deepagents_code import _paths

        built_in = _paths.get_built_in_skills_dir()
        assert built_in.is_dir()
        assert built_in.name == "built_in_skills"
