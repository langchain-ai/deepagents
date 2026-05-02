"""Tests for first-run onboarding state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_cli._env_vars import DEBUG_ONBOARDING
from deepagents_cli.onboarding import (
    ONBOARDING_NAME_MEMORY_END,
    ONBOARDING_NAME_MEMORY_START,
    has_completed_onboarding,
    mark_onboarding_complete,
    onboarding_marker_path,
    should_run_onboarding,
    write_onboarding_name_memory,
)

if TYPE_CHECKING:
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
        """Completion should create the marker under the config directory."""
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
