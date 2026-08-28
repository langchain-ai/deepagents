"""Tests for first-run onboarding state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from deepagents_code._env_vars import ONBOARDING
from deepagents_code.onboarding import (
    GOAL_AUTO_ACCEPT_PROMPT_MARKER_FILENAME,
    ONBOARDING_MARKER_FILENAME,
    ONBOARDING_NAME_MEMORY_END,
    ONBOARDING_NAME_MEMORY_START,
    extract_onboarding_name_block,
    goal_auto_accept_prompt_marker_path,
    has_completed_onboarding,
    has_shown_goal_auto_accept_prompt,
    mark_goal_auto_accept_prompt_shown,
    mark_onboarding_complete,
    onboarding_marker_path,
    should_run_onboarding,
    write_onboarding_name_memory,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestOnboardingState:
    """Tests for the onboarding completion marker and env override."""

    def test_goal_preference_prompt_marker_write_failure_returns_false(
        self,
        tmp_path,
    ) -> None:
        """A marker I/O failure should not escape the onboarding helper."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("blocked", encoding="utf-8")

        assert mark_goal_auto_accept_prompt_shown(blocker / ".state") is False

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

    def test_default_marker_path_lives_under_state_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default marker path resolves under `~/.deepagents/.state/`.

        Pins the convention introduced when CLI internal state was moved out
        of the user-facing config directory. A regression that pointed the
        marker back at `~/.deepagents/` would silently re-pollute the agent
        listing surface.
        """
        from deepagents_code import onboarding as onboarding_module

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
            return original_write_text(self, *args, **kwargs)  # ty: ignore

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


class TestExtractOnboardingNameBlock:
    """Tests for `extract_onboarding_name_block`."""

    def test_well_formed_block_returned_with_markers(self) -> None:
        """A well-formed block is returned inclusive of both markers."""
        block = (
            f"{ONBOARDING_NAME_MEMORY_START}\n"
            '- The user\'s preferred name is "Ada".\n'
            f"{ONBOARDING_NAME_MEMORY_END}"
        )
        text = f"## User Preferences\n\n{block}\n"

        assert extract_onboarding_name_block(text) == block

    def test_trailing_content_after_end_marker_excluded(self) -> None:
        """Extraction stops at the end marker and drops trailing content."""
        block = (
            f"{ONBOARDING_NAME_MEMORY_START}\n"
            '- The user\'s preferred name is "Ada".\n'
            f"{ONBOARDING_NAME_MEMORY_END}"
        )
        text = f"{block}\n\nUnrelated note after the block.\n"

        assert extract_onboarding_name_block(text) == block


class TestOnboardingSkipsReservedAgents:
    """The marker write is the path that would stamp `AGENTS.md` into app state."""
