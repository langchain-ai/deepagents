"""Unit tests for /skill:<name> command parsing and skill content loading."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.skills.load import load_skill_content

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadSkillContent:
    """Test load_skill_content() reads SKILL.md files correctly."""

    def test_allowed_roots_blocks_outside_path(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside" / "SKILL.md"
        outside.parent.mkdir()
        outside.write_text("secret", encoding="utf-8")

        allowed = tmp_path / "skills"
        allowed.mkdir()

        with pytest.raises(PermissionError, match="resolves outside all allowed"):
            load_skill_content(str(outside), allowed_roots=[allowed])

    def test_allowed_roots_blocks_symlink_escape(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("ssh key", encoding="utf-8")

        skills_dir = tmp_path / "skills" / "evil"
        skills_dir.mkdir(parents=True)
        symlink = skills_dir / "SKILL.md"
        symlink.symlink_to(secret)

        # Symlink resolves to secret.txt which is outside skills/
        with pytest.raises(PermissionError, match="resolves outside all allowed"):
            load_skill_content(str(symlink), allowed_roots=[tmp_path / "skills"])


class TestBuildSkillCommands:
    """Test build_skill_commands() produces correct autocomplete tuples."""


class TestSkillCommandParsing:
    """Test parse_skill_command() from command_registry."""


def _make_app() -> MagicMock:
    """Create a mock app with the methods `_handle_skill_command` needs."""
    from deepagents_code.app import DeepAgentsApp

    app = MagicMock(spec=DeepAgentsApp)
    app._skill_trust_denied = set()
    app._pending_skill_row = None
    mounted_messages: list[object] = []
    app._mounted_messages = mounted_messages

    def capture_mount(msg: object) -> None:
        mounted_messages.append(msg)

    app._mount_message = AsyncMock(side_effect=capture_mount)
    app._send_to_agent = AsyncMock()
    app._push_screen_wait = AsyncMock(return_value=False)
    app._invoke_skill = DeepAgentsApp._invoke_skill.__get__(app)
    app._handle_skill_command = DeepAgentsApp._handle_skill_command.__get__(app)
    app._on_skill_pinned = DeepAgentsApp._on_skill_pinned.__get__(app)
    app._request_skill_trust = DeepAgentsApp._request_skill_trust.__get__(app)
    return app


class TestHandleSkillCommand:
    """The client sends the invocation without reading the skill."""

    async def test_sends_without_reading_the_skill(self) -> None:
        from deepagents_code.tui.widgets.messages import SkillMessage

        app = _make_app()
        with patch("deepagents_code.skills.load.load_skill_content") as load:
            await app._handle_skill_command("/skill:Test-Skill find quantum")

        load.assert_not_called()
        prompt = app._send_to_agent.call_args[0][0]
        assert "`test-skill`" in prompt
        assert "**User request:** find quantum" in prompt
        assert app._send_to_agent.call_args.kwargs["skill_name"] == "test-skill"
        [row] = [m for m in app._mounted_messages if isinstance(m, SkillMessage)]
        assert row._args == "find quantum"
        assert app._pending_skill_row is row

    async def test_pin_fills_the_matching_row(self) -> None:
        app = _make_app()
        await app._handle_skill_command("/skill:test-skill")
        row = app._pending_skill_row
        app._on_skill_pinned("other", "Other", "Not this one")
        assert row._body == ""

        await app._handle_skill_command("/skill:test-skill")
        row = app._pending_skill_row
        app._on_skill_pinned("test-skill", "A test skill", "Do stuff")
        assert row._description == "A test skill"
        assert row._body == "Do stuff"

    def test_pin_survives_rebuilding_row_from_store(self) -> None:
        from deepagents_code.tui.widgets.message_store import MessageData, MessageStore
        from deepagents_code.tui.widgets.messages import SkillMessage

        app = _make_app()
        row = SkillMessage(skill_name="test-skill", id="skill-test")
        store = MessageStore()
        store.append(MessageData.from_widget(row))
        app._message_store = store
        app._pending_skill_row = row

        app._on_skill_pinned("test-skill", "A test skill", "Do stuff")

        data = store.get_message("skill-test")
        assert data is not None
        assert data.skill_description == "A test skill"
        assert data.skill_body == "Do stuff"
        rebuilt = data.to_widget()
        assert isinstance(rebuilt, SkillMessage)
        assert rebuilt._description == "A test skill"
        assert rebuilt._body == "Do stuff"


class TestRequestSkillTrust:
    """The TUI answers the server's trust interrupt and fails closed."""

    async def test_deny_is_remembered_for_the_session(self) -> None:
        app = _make_app()
        assert await app._request_skill_trust("review", "/outside") is False
        app._push_screen_wait = AsyncMock(return_value=True)
        assert await app._request_skill_trust("review", "/outside") is False
        app._push_screen_wait.assert_not_awaited()

    async def test_prompt_failure_denies(self) -> None:
        app = _make_app()
        app._push_screen_wait = AsyncMock(side_effect=RuntimeError("no screen"))
        assert await app._request_skill_trust("review", "/outside") is False
        assert "/outside" in app._skill_trust_denied


class TestDiscoverSkillsAndRoots:
    """The containment roots must include persisted trust and extra dirs.

    This is the join that makes an in-session approval survive a relaunch. The
    real `discover_skills_and_roots` is exercised (not mocked wholesale) so a
    regression that drops the trust join, or re-resolves it instead of adding it
    as-is, is caught.
    """

    def test_persisted_trust_added_as_is_while_extra_dirs_resolved(
        self, tmp_path: Path
    ) -> None:
        """Trusted dirs join roots verbatim; `extra_allowed_dirs` are resolved.

        A symlinked entry makes the two code paths distinguishable: re-resolving
        the trusted entry (the regression this guards) would follow the symlink
        to a directory the user never approved, so it must be added as-is. The
        declarative `extra_allowed_dirs` allowlist is intentionally resolved. A
        real (non-symlink) dir could not tell these apart because `resolve()`
        would be idempotent.
        """
        from deepagents_code.skills.invocation import discover_skills_and_roots

        real_trusted = tmp_path / "real_trusted"
        real_trusted.mkdir()
        link_trusted = tmp_path / "link_trusted"
        link_trusted.symlink_to(real_trusted, target_is_directory=True)
        real_extra = tmp_path / "real_extra"
        real_extra.mkdir()
        link_extra = tmp_path / "link_extra"
        link_extra.symlink_to(real_extra, target_is_directory=True)
        (tmp_path / "config.toml").write_text(
            f'[skills]\nextra_allowed_dirs = ["{link_extra}"]\n',
            encoding="utf-8",
        )

        with (
            patch(
                "deepagents_code.config.credentials",
                SimpleNamespace(project_root=tmp_path),
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=[]),
            patch(
                "deepagents_code.skills.trust.load_trusted_skill_dirs",
                return_value=[link_trusted],
            ),
        ):
            _skills, roots = discover_skills_and_roots("agent")

        # Trusted entry added verbatim — not followed to its symlink target.
        assert link_trusted in roots
        assert real_trusted.resolve() not in roots
        # `extra_allowed_dirs` is the declarative allowlist and is resolved.
        assert real_extra.resolve() in roots

    def test_relative_extra_dirs_use_user_cwd_not_project_root(
        self, tmp_path: Path
    ) -> None:
        """Relative allowlist entries keep the cwd used by config loading."""
        from deepagents_code.skills.invocation import discover_skills_and_roots

        project_root = tmp_path / "project"
        user_cwd = project_root / "subdir"
        configured = user_cwd / "shared"
        wrong = project_root / "shared"
        configured.mkdir(parents=True)
        wrong.mkdir()
        (tmp_path / "config.toml").write_text(
            '[skills]\nextra_allowed_dirs = ["shared"]\n',
            encoding="utf-8",
        )

        with (
            patch(
                "deepagents_code.config.credentials",
                SimpleNamespace(project_root=project_root),
            ),
            patch("deepagents_code.skills.load.list_skills", return_value=[]),
            patch(
                "deepagents_code.skills.trust.load_trusted_skill_dirs",
                return_value=[],
            ),
        ):
            _skills, roots = discover_skills_and_roots(
                "agent",
                path_base=user_cwd,
            )

        assert configured.resolve() in roots
        assert wrong.resolve() not in roots
