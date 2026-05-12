"""Tests for first-run onboarding state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_cli._env_vars import DEBUG_ONBOARDING
from deepagents_cli.onboarding import (
    ONBOARDING_MARKER_FILENAME,
    ONBOARDING_NAME_MEMORY_END,
    ONBOARDING_NAME_MEMORY_START,
    has_completed_onboarding,
    mark_onboarding_complete,
    onboarding_marker_path,
    should_run_onboarding,
    write_onboarding_name_memory,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestOnboardingState:
    """Tests for the onboarding completion marker and debug override."""

    def test_missing_marker_runs_onboarding(self, tmp_path) -> None:
        """Onboarding should run before the marker exists."""
        assert should_run_onboarding(tmp_path) is True

    def test_existing_marker_skips_onboarding(self, tmp_path) -> None:
        """Onboarding should not run after completion is marked."""
        onboarding_marker_path(tmp_path).write_text("1\n", encoding="utf-8")

        assert has_completed_onboarding(tmp_path) is True
        assert should_run_onboarding(tmp_path) is False

    def test_debug_override_runs_even_with_marker(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Debug override should force onboarding every startup."""
        onboarding_marker_path(tmp_path).write_text("1\n", encoding="utf-8")
        monkeypatch.setenv(DEBUG_ONBOARDING, "1")

        assert should_run_onboarding(tmp_path) is True

    def test_mark_onboarding_complete_creates_marker(self, tmp_path) -> None:
        """Completion should create the marker under the state directory."""
        assert mark_onboarding_complete(tmp_path) is True

        assert onboarding_marker_path(tmp_path).read_text(encoding="utf-8") == "1\n"
        assert should_run_onboarding(tmp_path) is False

    def test_write_onboarding_name_memory_creates_managed_block(self, tmp_path) -> None:
        """Submitted names should be written to user agent memory."""
        memory_path = tmp_path / "agent" / "AGENTS.md"

        assert (
            write_onboarding_name_memory(
                "Ada Lovelace",
                "agent",
                memory_path=memory_path,
            )
            is True
        )

        content = memory_path.read_text(encoding="utf-8")
        assert "## User Preferences" in content
        assert ONBOARDING_NAME_MEMORY_START in content
        assert '- The user\'s preferred name is "Ada Lovelace".' in content
        assert ONBOARDING_NAME_MEMORY_END in content

    def test_write_onboarding_name_memory_replaces_managed_block(
        self,
        tmp_path,
    ) -> None:
        """Repeated onboarding runs should update the name instead of duplicating it."""
        memory_path = tmp_path / "agent" / "AGENTS.md"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text(
            "Existing notes\n\n"
            "## User Preferences\n\n"
            f"{ONBOARDING_NAME_MEMORY_START}\n"
            "- The user's preferred name is Ada.\n"
            f"{ONBOARDING_NAME_MEMORY_END}\n\n"
            "Keep this note.\n",
            encoding="utf-8",
        )

        assert (
            write_onboarding_name_memory(
                "Grace Hopper",
                "agent",
                memory_path=memory_path,
            )
            is True
        )

        content = memory_path.read_text(encoding="utf-8")
        assert content.count(ONBOARDING_NAME_MEMORY_START) == 1
        assert '- The user\'s preferred name is "Grace Hopper".' in content
        assert "Ada." not in content
        assert "Existing notes" in content
        assert "Keep this note." in content

    def test_write_onboarding_name_memory_skips_empty_name(self, tmp_path) -> None:
        """Empty optional names should not create memory files."""
        memory_path = tmp_path / "agent" / "AGENTS.md"

        assert (
            write_onboarding_name_memory("", "agent", memory_path=memory_path) is False
        )

        assert not memory_path.exists()

    def test_default_marker_path_lives_under_state_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default marker path resolves under `~/.deepagents/.state/`.

        Pins the convention introduced when CLI internal state was moved out
        of the user-facing config directory. A regression that pointed the
        marker back at `~/.deepagents/` would silently re-pollute the agent
        listing surface.
        """
        from deepagents_cli import onboarding as onboarding_module

        fake_state_dir = tmp_path / ".deepagents" / ".state"
        monkeypatch.setattr(onboarding_module, "DEFAULT_STATE_DIR", fake_state_dir)

        path = onboarding_marker_path()

        assert path == fake_state_dir / ONBOARDING_MARKER_FILENAME
        assert path.parent.name == ".state"

    def test_mark_onboarding_complete_returns_false_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A write failure should return `False` rather than raise."""
        from pathlib import Path as _Path

        original_write_text = _Path.write_text

        def boom(self: _Path, *args: object, **kwargs: object) -> int:
            if self.name == ONBOARDING_MARKER_FILENAME:
                msg = "simulated read-only filesystem"
                raise PermissionError(msg)
            return original_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_Path, "write_text", boom)

        assert mark_onboarding_complete(tmp_path) is False
        assert not onboarding_marker_path(tmp_path).exists()

    def test_has_completed_onboarding_returns_false_on_oserror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An `exists()` failure should be swallowed and reported as not done."""
        from pathlib import Path as _Path

        def boom(self: _Path) -> bool:  # noqa: ARG001  # required by Path.exists signature
            msg = "simulated permission denied"
            raise PermissionError(msg)

        monkeypatch.setattr(_Path, "exists", boom)

        assert has_completed_onboarding(tmp_path) is False

    def test_write_onboarding_name_memory_returns_false_on_decode_error(
        self,
        tmp_path: Path,
    ) -> None:
        """A non-UTF-8 existing memory file should not be clobbered."""
        memory_path = tmp_path / "agent" / "AGENTS.md"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_bytes(b"\xff\xfe garbage \x00\x01")

        assert (
            write_onboarding_name_memory("Ada", "agent", memory_path=memory_path)
            is False
        )

        # Existing bytes are preserved — write was aborted.
        assert memory_path.read_bytes() == b"\xff\xfe garbage \x00\x01"

    def test_write_onboarding_name_memory_returns_false_on_oserror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A write failure on the memory file should return `False`."""
        from pathlib import Path as _Path

        memory_path = tmp_path / "agent" / "AGENTS.md"
        original_write_text = _Path.write_text

        def boom(self: _Path, *args: object, **kwargs: object) -> int:
            if self == memory_path:
                msg = "simulated full disk"
                raise OSError(msg)
            return original_write_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_Path, "write_text", boom)

        assert (
            write_onboarding_name_memory("Ada", "agent", memory_path=memory_path)
            is False
        )

    def test_write_onboarding_name_memory_appends_heading_when_absent(
        self,
        tmp_path: Path,
    ) -> None:
        """Pre-existing memory without `## User Preferences` should keep its content.

        Existing notes must be preserved and the managed block gets appended
        under a freshly created heading rather than wiping or overwriting.
        """
        memory_path = tmp_path / "agent" / "AGENTS.md"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text(
            "Existing freeform notes about the user.\n", encoding="utf-8"
        )

        assert (
            write_onboarding_name_memory(
                "Grace Hopper",
                "agent",
                memory_path=memory_path,
            )
            is True
        )

        content = memory_path.read_text(encoding="utf-8")
        assert "Existing freeform notes about the user." in content
        assert content.count("## User Preferences") == 1
        assert ONBOARDING_NAME_MEMORY_START in content
        assert '- The user\'s preferred name is "Grace Hopper".' in content
