"""Unit tests for /skill:<name> command parsing and skill content loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from deepagents_cli.command_registry import build_skill_commands, parse_skill_command
from deepagents_cli.skills.load import load_skill_content

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadSkillContent:
    """Test load_skill_content() reads SKILL.md files correctly."""

    def test_valid_skill_file(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        content = "---\nname: test\ndescription: A test\n---\n\n# Test Skill\n"
        skill_md.write_text(content, encoding="utf-8")

        result = load_skill_content(str(skill_md))
        assert result == content

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_skill_content(str(tmp_path / "nonexistent" / "SKILL.md"))
        assert result is None

    def test_encoding_error_returns_none(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_bytes(b"\x80\x81\x82\xff\xfe")

        result = load_skill_content(str(skill_md))
        assert result is None

    def test_empty_file_returns_empty_string(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("", encoding="utf-8")

        result = load_skill_content(str(skill_md))
        assert result == ""

    def test_allowed_roots_permits_valid_path(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "skills" / "SKILL.md"
        skill_md.parent.mkdir()
        skill_md.write_text("content", encoding="utf-8")

        result = load_skill_content(str(skill_md), allowed_roots=[tmp_path / "skills"])
        assert result == "content"

    def test_allowed_roots_blocks_outside_path(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside" / "SKILL.md"
        outside.parent.mkdir()
        outside.write_text("secret", encoding="utf-8")

        allowed = tmp_path / "skills"
        allowed.mkdir()

        result = load_skill_content(str(outside), allowed_roots=[allowed])
        assert result is None

    def test_allowed_roots_blocks_symlink_escape(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("ssh key", encoding="utf-8")

        skills_dir = tmp_path / "skills" / "evil"
        skills_dir.mkdir(parents=True)
        symlink = skills_dir / "SKILL.md"
        symlink.symlink_to(secret)

        result = load_skill_content(str(symlink), allowed_roots=[tmp_path / "skills"])
        # Symlink resolves to secret.txt which is outside skills/
        assert result is None

    def test_empty_allowed_roots_skips_check(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "anywhere" / "SKILL.md"
        skill_md.parent.mkdir()
        skill_md.write_text("ok", encoding="utf-8")

        result = load_skill_content(str(skill_md), allowed_roots=[])
        assert result == "ok"


class TestBuildSkillCommands:
    """Test build_skill_commands() produces correct autocomplete tuples."""

    def test_empty_list(self) -> None:
        assert build_skill_commands([]) == []

    def test_single_skill(self) -> None:
        skills = [
            {
                "name": "web-research",
                "description": "Research topics on the web",
                "path": "/some/path/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            }
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        assert len(result) == 1
        name, desc, keywords = result[0]
        assert name == "/skill:web-research"
        assert desc == "Research topics on the web"
        assert keywords == "web-research"

    def test_multiple_skills(self) -> None:
        skills = [
            {
                "name": "skill-a",
                "description": "Skill A",
                "path": "/a/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            },
            {
                "name": "skill-b",
                "description": "Skill B",
                "path": "/b/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "project",
            },
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        assert len(result) == 2
        assert result[0][0] == "/skill:skill-a"
        assert result[1][0] == "/skill:skill-b"

    def test_tuple_format(self) -> None:
        """Each entry is a 3-tuple of strings."""
        skills = [
            {
                "name": "test",
                "description": "Test skill",
                "path": "/test/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "built-in",
            }
        ]
        result = build_skill_commands(skills)  # type: ignore[arg-type]
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            assert all(isinstance(s, str) for s in entry)


class TestSkillCommandParsing:
    """Test parse_skill_command() from command_registry."""

    def test_name_only(self) -> None:
        name, args = parse_skill_command("/skill:web-research")
        assert name == "web-research"
        assert args == ""

    def test_name_with_args(self) -> None:
        name, args = parse_skill_command("/skill:web-research find quantum computing")
        assert name == "web-research"
        assert args == "find quantum computing"

    def test_empty_skill_prefix(self) -> None:
        name, args = parse_skill_command("/skill:")
        assert name == ""
        assert args == ""

    def test_name_with_spaces(self) -> None:
        name, args = parse_skill_command("/skill:  web-research  some args ")
        assert name == "web-research"
        assert args == "some args"

    def test_case_normalization(self) -> None:
        name, args = parse_skill_command("/skill:Web-Research")
        assert name == "web-research"
        assert args == ""

    def test_whitespace_only_after_prefix(self) -> None:
        name, args = parse_skill_command("/skill:   ")
        assert name == ""
        assert args == ""


def _make_app() -> MagicMock:
    """Create a mock app with the methods _handle_skill_command needs."""
    from deepagents_cli.app import DeepAgentsApp

    app = MagicMock(spec=DeepAgentsApp)
    app._assistant_id = "agent"
    app._discovered_skills = []
    app._skill_allowed_roots = []
    app._mounted_messages: list[object] = []

    def capture_mount(msg: object) -> None:
        app._mounted_messages.append(msg)

    app._mount_message = AsyncMock(side_effect=capture_mount)
    app._handle_user_message = AsyncMock()
    app._send_to_agent = AsyncMock()
    app._handle_skill_command = DeepAgentsApp._handle_skill_command.__get__(app)
    app._discover_skills_and_roots = DeepAgentsApp._discover_skills_and_roots.__get__(
        app
    )
    return app


def _app_message_texts(app: MagicMock) -> list[str]:
    """Extract plain text from AppMessage widgets mounted by the mock app."""
    from deepagents_cli.widgets.messages import AppMessage

    return [str(m.content) for m in app._mounted_messages if isinstance(m, AppMessage)]


def _fake_skill(
    name: str = "test-skill",
    desc: str = "A test skill",
    path: str = "/skills/test-skill/SKILL.md",
) -> dict[str, object]:
    return {
        "name": name,
        "description": desc,
        "path": path,
        "license": None,
        "compatibility": None,
        "metadata": {},
        "allowed_tools": [],
        "source": "user",
    }


class TestHandleSkillCommand:
    """Test _handle_skill_command orchestration paths.

    Most tests leave `_discovered_skills` empty so the fallback (fresh
    discovery) path is exercised. Cache-hit tests populate the cache
    directly.
    """

    async def test_empty_name_shows_usage(self) -> None:
        app = _make_app()
        await app._handle_skill_command("/skill:")

        texts = _app_message_texts(app)
        assert any("Usage:" in t for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_skill_not_found(self) -> None:
        app = _make_app()
        with (
            patch("deepagents_cli.skills.load.list_skills", return_value=[]),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:nonexistent")

        texts = _app_message_texts(app)
        assert any("not found" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_content_none_shows_error(self) -> None:
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_cli.skills.load.list_skills", return_value=[skill]),
            patch("deepagents_cli.skills.load.load_skill_content", return_value=None),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("could not read" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_empty_content_shows_error(self) -> None:
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_cli.skills.load.list_skills", return_value=[skill]),
            patch("deepagents_cli.skills.load.load_skill_content", return_value=""),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("empty" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_happy_path_sends_prompt(self) -> None:
        from deepagents_cli.widgets.messages import SkillMessage

        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_cli.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_cli.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        app._send_to_agent.assert_awaited_once()
        prompt = app._send_to_agent.call_args[0][0]
        assert "test-skill" in prompt
        assert "# Instructions" in prompt
        # Verify SkillMessage was mounted instead of UserMessage
        skill_msgs = [m for m in app._mounted_messages if isinstance(m, SkillMessage)]
        assert len(skill_msgs) == 1
        assert skill_msgs[0]._skill_name == "test-skill"

    async def test_happy_path_with_args(self) -> None:
        from deepagents_cli.widgets.messages import SkillMessage

        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_cli.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_cli.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill find quantum")

        prompt = app._send_to_agent.call_args[0][0]
        assert "find quantum" in prompt
        assert "**User request:**" in prompt
        skill_msgs = [m for m in app._mounted_messages if isinstance(m, SkillMessage)]
        assert len(skill_msgs) == 1
        assert skill_msgs[0]._args == "find quantum"

    async def test_filesystem_error_shows_specific_message(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_cli.skills.load.list_skills",
                side_effect=PermissionError("access denied"),
            ),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("filesystem error" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_unexpected_error_includes_exception_type(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_cli.skills.load.list_skills",
                side_effect=TypeError("bad argument"),
            ),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("TypeError" in t for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_cache_hit_skips_list_skills(self) -> None:
        """When the skill is in the cache, list_skills should not be called."""
        from pathlib import Path

        app = _make_app()
        skill = _fake_skill()
        app._discovered_skills = [skill]
        sentinel_root = Path("/sentinel/root")
        app._skill_allowed_roots = [sentinel_root]

        with (
            patch(
                "deepagents_cli.skills.load.load_skill_content",
                return_value="# Cached\nDo cached stuff",
            ) as mock_load,
            patch("deepagents_cli.skills.load.list_skills") as mock_list,
        ):
            await app._handle_skill_command("/skill:test-skill")

        mock_list.assert_not_called()
        # Verify cached allowed_roots flow through to load_skill_content
        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs["allowed_roots"] == [sentinel_root]
        app._send_to_agent.assert_awaited_once()
        prompt = app._send_to_agent.call_args[0][0]
        assert "test-skill" in prompt
        assert "# Cached" in prompt

    async def test_cache_miss_falls_back_to_discovery(self) -> None:
        """When skill is not in cache, fresh discovery is used and cache backfilled."""
        app = _make_app()
        skill = _fake_skill(name="new-skill")
        # Cache has a different skill
        app._discovered_skills = [_fake_skill(name="other-skill")]

        with (
            patch(
                "deepagents_cli.skills.load.list_skills",
                return_value=[skill],
            ) as mock_list,
            patch(
                "deepagents_cli.skills.load.load_skill_content",
                return_value="# Fresh\nContent",
            ),
            patch("deepagents_cli.config.settings"),
        ):
            await app._handle_skill_command("/skill:new-skill")

        mock_list.assert_called_once()
        app._send_to_agent.assert_awaited_once()
        prompt = app._send_to_agent.call_args[0][0]
        assert "new-skill" in prompt
        assert "# Fresh" in prompt
        # Cache should be backfilled with fresh discovery results
        assert len(app._discovered_skills) == 1
        assert app._discovered_skills[0]["name"] == "new-skill"
