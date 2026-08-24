"""Unit tests for `deepagents_code._paths`."""

from pathlib import Path

import pytest

from deepagents_code._paths import PathState, classify_path, get_deepagents_home


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
    """Tests for the user-level Deep Agents directory resolver."""

    def test_defaults_to_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default remains `~/.deepagents`."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        assert get_deepagents_home() == tmp_path / ".deepagents"

    def test_uses_absolute_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`DEEPAGENTS_HOME` replaces the default root."""
        configured = tmp_path / "data" / "deepagents"
        monkeypatch.setenv("DEEPAGENTS_HOME", str(configured))

        assert get_deepagents_home() == configured

    def test_expands_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tilde in `DEEPAGENTS_HOME` expands to the user home."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("DEEPAGENTS_HOME", "~/profiles/dcode")

        assert get_deepagents_home() == tmp_path / "profiles" / "dcode"

    def test_makes_relative_override_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative override is anchored to the launch directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DEEPAGENTS_HOME", "profiles/dcode")

        assert get_deepagents_home() == tmp_path / "profiles" / "dcode"

    def test_empty_override_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty override preserves the default root."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("DEEPAGENTS_HOME", "")

        assert get_deepagents_home() == tmp_path / ".deepagents"
