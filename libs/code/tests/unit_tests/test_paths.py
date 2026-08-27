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


class TestGetDeepagentsHome:
    """Tests for launch-time profile-root validation and immutability."""

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


class TestDisplayOutsideTheProfileRoot:
    """A path outside the profile must not be given a bogus `~` prefix."""


class TestProbeWritable:
    """`probe_writable` decides which shared directory a process may use."""

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

    def test_marker_cannot_relabel_a_configured_profile(self, tmp_path: Path) -> None:
        """A forged marker is ignored: it is a display hint, not a trust input."""
        configured = tmp_path / "profile"
        env = _subprocess_env(home=tmp_path, configured=str(configured))
        env["DEEPAGENTS_HOME_IS_DEFAULT"] = "1"

        out = self._run(env)

        parent, _child = out.strip().splitlines()
        assert parent == f"False {configured / 'config.toml'}"


class TestLaunchSnapshotSubprocess:
    """Regressions that must exercise a clean import generation."""

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

    def test_rejects_dangling_symlink(self, tmp_path: Path) -> None:
        """A profile symlink must resolve before launch treats it as creatable."""
        target = tmp_path / "missing-profile"
        link = tmp_path / "profile-link"
        link.symlink_to(target, target_is_directory=True)

        with pytest.raises(DeepAgentsHomeError, match="symlink whose target"):
            _capture_paths(str(link), launch_home=tmp_path)


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


class TestDefaultProfileDisplayThroughConsumers:
    """The `~`-abbreviating branch must be exercised through real call sites.

    The suite runs with `DEEPAGENTS_HOME` set (see `conftest`), so
    `uses_default_profile` is `False` and `display()` is the identity function
    everywhere. An assertion comparing a message against `PATHS.display(...)`
    therefore passes even if the call site forgot to call `display` at all.
    These tests install a *default* snapshot so the abbreviation is real.
    """


class TestConfiguredProfileNotice:
    """Launch must say which profile it selected when it is not the default."""

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


class TestHomeDirectoryProfileSpellings:
    """Every spelling of "the profile is my home directory" must be rejected.

    `~/` and an absolute `/Users/me` name the same directory, so rejecting only
    the tilde form would leave the `~/.env`-as-trusted-dotenv hazard reachable.
    """

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


class TestUnreadableProfileRoot:
    """An unreadable root must fail once, with its real cause.

    `Path.is_symlink` swallows `OSError` and reports `False` under EACCES, and
    `classify_path` reports `UNREADABLE` rather than `EXISTS`. Before the
    explicit branch, such a root passed every guard and the launch continued.
    Each later access then failed on its own, and each failure looked like a
    first run.
    """

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


class TestUserDeepagentsDirHelpers:
    """Free functions for user-level paths derived from `DEEPAGENTS_HOME`."""


class TestGetProjectAgentMdPath:
    """Test `_paths.get_project_agent_md_path()`."""


class TestAgentsAliasDirectories:
    """Tests for `.agents` directory alias helpers."""


class TestClaudeSkillsDirs:
    """Tests for `.claude/skills` directory helpers."""


class TestReservedAgentNameGuards:
    """Agent profiles cannot overlap directories owned by dcode."""

    def test_trailing_dot_is_invalid(self) -> None:
        """A trailing-dot alias is rejected by the character allowlist."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="Invalid agent name"):
            _paths.get_agent_dir("plugins.")

    def test_agent_md_accessor_rejects_invalid_characters(self) -> None:
        """The instruction path accessor cannot escape the profile root."""
        from deepagents_code import _paths

        with pytest.raises(ValueError, match="Invalid agent name"):
            _paths.get_user_agent_md_path("../escape")


class TestEnsureDirHelpers:
    """The `ensure_*` helpers create the directory they return."""
