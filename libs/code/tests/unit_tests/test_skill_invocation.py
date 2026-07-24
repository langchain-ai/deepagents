"""Unit tests for /skill:<name> command parsing and skill content loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.command_registry import (
    _STATIC_SKILL_ALIASES,
    CommandEntry,
    build_skill_commands,
    parse_skill_command,
)
from deepagents_code.skills.load import load_skill_content


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
        result = build_skill_commands(skills)  # ty: ignore
        assert len(result) == 1
        entry = result[0]
        assert isinstance(entry, CommandEntry)
        assert entry.name == "/skill:web-research"
        assert entry.description == "Research topics on the web"
        assert entry.hidden_keywords == "web-research"
        assert entry.argument_hint == ""

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
        result = build_skill_commands(skills)  # ty: ignore
        assert len(result) == 2
        assert result[0].name == "/skill:skill-a"
        assert result[1].name == "/skill:skill-b"

    def test_entry_format(self) -> None:
        """Each entry is a `CommandEntry`."""
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
        result = build_skill_commands(skills)  # ty: ignore
        for entry in result:
            assert isinstance(entry, CommandEntry)

    def test_excludes_static_skill_aliases(self) -> None:
        """Skills with names matching static aliases are excluded."""
        skills = [
            {
                "name": "remember",
                "description": "Update memory",
                "path": "/built-in/SKILL.md",
                "license": "MIT",
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "built-in",
            },
            {
                "name": "skill-creator",
                "description": "Create skills",
                "path": "/built-in/SKILL.md",
                "license": "MIT",
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "built-in",
            },
            {
                "name": "custom-skill",
                "description": "A custom skill",
                "path": "/user/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            },
        ]
        result = build_skill_commands(skills)  # ty: ignore
        names = [r[0] for r in result]
        assert "/skill:remember" not in names
        assert "/skill:skill-creator" not in names
        assert "/skill:custom-skill" in names
        assert len(result) == 1

    def test_non_alias_command_names_not_suppressed(self) -> None:
        """Skills named after non-alias commands are NOT excluded."""
        skills = [
            {
                "name": "model",
                "description": "A model management skill",
                "path": "/user/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            },
        ]
        result = build_skill_commands(skills)  # ty: ignore
        assert len(result) == 1
        assert result[0][0] == "/skill:model"

    def test_static_skill_aliases_contains_expected_entries(self) -> None:
        """Verify the alias set only contains actual skill-backed commands."""
        assert {"remember", "skill-creator"} == _STATIC_SKILL_ALIASES

    def test_plugin_skill_separates_machine_name_from_display(self) -> None:
        """Plugin skills insert the namespaced name but show a short label."""
        skills = [
            {
                "name": "my-plugin@market:foo:review",
                "description": "Review code",
                "path": "/plugin/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "plugin",
            }
        ]
        result = build_skill_commands(skills)  # ty: ignore
        assert len(result) == 1
        entry = result[0]
        # Machine name (matched + inserted) keeps the full namespace.
        assert entry.name == "/skill:my-plugin@market:foo:review"
        # Popup label is the short terminal segment.
        assert entry.label() == "/skill:review"
        # Description is tagged with the plugin id for source clarity.
        assert entry.description == "(my-plugin@market) Review code"
        # Both the full name and terminal segment are fuzzy-matchable.
        assert "review" in entry.hidden_keywords
        assert "my-plugin@market" in entry.hidden_keywords

    def test_non_plugin_skill_label_equals_name(self) -> None:
        """Non-plugin skills keep label == name (no display override)."""
        skills = [
            {
                "name": "web-research",
                "description": "Research topics",
                "path": "/user/SKILL.md",
                "license": None,
                "compatibility": None,
                "metadata": {},
                "allowed_tools": [],
                "source": "user",
            }
        ]
        result = build_skill_commands(skills)  # ty: ignore
        entry = result[0]
        assert entry.name == "/skill:web-research"
        assert entry.label() == "/skill:web-research"
        assert entry.description == "Research topics"


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
    from deepagents_code.app import DeepAgentsApp

    app = MagicMock(spec=DeepAgentsApp)
    app._assistant_id = "agent"
    app._discovered_skills = []
    app._skill_allowed_roots = []
    app._skill_trust_denied = set()
    mounted_messages: list[object] = []
    app._mounted_messages = mounted_messages

    def capture_mount(msg: object) -> None:
        mounted_messages.append(msg)

    app._mount_message = AsyncMock(side_effect=capture_mount)
    app._handle_user_message = AsyncMock()
    app._send_to_agent = AsyncMock()
    # Default to deny so the containment error surfaces unless a test opts into
    # approval by overriding the return value.
    app._push_screen_wait = AsyncMock(return_value=False)
    app.notify = MagicMock()
    app._invoke_skill = DeepAgentsApp._invoke_skill.__get__(app)
    app._prompt_skill_trust_and_retry = (
        DeepAgentsApp._prompt_skill_trust_and_retry.__get__(app)
    )
    app._handle_skill_command = DeepAgentsApp._handle_skill_command.__get__(app)
    app._discover_skills_and_roots = DeepAgentsApp._discover_skills_and_roots.__get__(
        app
    )
    app._discover_skills_and_roots_with_import_lock = (
        DeepAgentsApp._discover_skills_and_roots_with_import_lock.__get__(app)
    )
    return app


def _app_message_texts(app: MagicMock) -> list[str]:
    """Extract plain text from AppMessage widgets mounted by the mock app."""
    from deepagents_code.tui.widgets.messages import AppMessage

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


class TestBuildSkillInvocationEnvelope:
    """Direct unit tests for `build_skill_invocation_envelope`."""

    def test_happy_path_with_args(self) -> None:
        """Envelope should contain wrapped prompt and full metadata."""
        from deepagents_code.skills.invocation import build_skill_invocation_envelope

        skill = {
            "name": "code-review",
            "description": "Review code changes",
            "source": "user",
            "path": "/skills/code-review/SKILL.md",
        }
        envelope = build_skill_invocation_envelope(
            skill,  # ty: ignore
            "# Instructions\nDo stuff",
            "review this patch",
        )
        assert "I'm invoking the skill `code-review`." in envelope.prompt
        assert "---\n# Instructions\nDo stuff\n---" in envelope.prompt
        assert "**User request:** review this patch" in envelope.prompt
        meta = envelope.message_kwargs["additional_kwargs"]["__skill"]
        assert meta["name"] == "code-review"
        assert meta["description"] == "Review code changes"
        assert meta["source"] == "user"
        assert meta["args"] == "review this patch"

    def test_empty_args_omits_user_request(self) -> None:
        """No `**User request:**` line when args is empty."""
        from deepagents_code.skills.invocation import build_skill_invocation_envelope

        skill = {"name": "test", "description": "", "source": "built-in", "path": "/x"}
        envelope = build_skill_invocation_envelope(
            skill,  # ty: ignore
            "body",
            "",
        )
        assert "**User request:**" not in envelope.prompt
        assert envelope.message_kwargs["additional_kwargs"]["__skill"]["args"] == ""

    def test_missing_optional_fields_default_to_empty(self) -> None:
        """Skill dicts without `description`/`source` should default to ''."""
        from deepagents_code.skills.invocation import build_skill_invocation_envelope

        skill = {"name": "minimal", "path": "/x"}
        envelope = build_skill_invocation_envelope(
            skill,  # ty: ignore
            "body",
        )
        meta = envelope.message_kwargs["additional_kwargs"]["__skill"]
        assert meta["description"] == ""
        assert meta["source"] == ""


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
            patch("deepagents_code.skills.load.list_skills", return_value=[]),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:nonexistent")

        texts = _app_message_texts(app)
        assert any("not found" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_content_none_shows_error(self) -> None:
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch("deepagents_code.skills.load.load_skill_content", return_value=None),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("could not read" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_containment_violation_shows_specific_message(self) -> None:
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=PermissionError(
                    "Skill path /tmp/evil resolves outside "
                    "all allowed skill directories."
                ),
            ),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("resolves outside" in t for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_empty_content_shows_error(self) -> None:
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch("deepagents_code.skills.load.load_skill_content", return_value=""),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("empty" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_happy_path_sends_prompt(self) -> None:
        from deepagents_code.tui.widgets.messages import SkillMessage

        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_code.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch("deepagents_code.config.settings"),
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
        from deepagents_code.tui.widgets.messages import SkillMessage

        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_code.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill find quantum")

        prompt = app._send_to_agent.call_args[0][0]
        assert "find quantum" in prompt
        assert "**User request:**" in prompt
        skill_msgs = [m for m in app._mounted_messages if isinstance(m, SkillMessage)]
        assert len(skill_msgs) == 1
        assert skill_msgs[0]._args == "find quantum"

    async def test_direct_invoke_preserves_exact_args(self) -> None:
        """Startup skill invocation should preserve the original prompt text."""
        app = _make_app()
        skill = _fake_skill()
        with (
            patch("deepagents_code.skills.load.list_skills", return_value=[skill]),
            patch(
                "deepagents_code.skills.load.load_skill_content",
                return_value="# Instructions\nDo stuff",
            ),
            patch("deepagents_code.config.settings"),
        ):
            await app._invoke_skill("test-skill", "  keep leading whitespace")

        prompt = app._send_to_agent.call_args[0][0]
        assert "**User request:**   keep leading whitespace" in prompt
        metadata = app._send_to_agent.call_args.kwargs["message_kwargs"]
        assert (
            metadata["additional_kwargs"]["__skill"]["args"]
            == "  keep leading whitespace"
        )

    async def test_filesystem_error_shows_specific_message(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_code.skills.load.list_skills",
                side_effect=PermissionError("access denied"),
            ),
            patch("deepagents_code.config.settings"),
        ):
            await app._handle_skill_command("/skill:test-skill")

        texts = _app_message_texts(app)
        assert any("filesystem error" in t.lower() for t in texts)
        app._send_to_agent.assert_not_awaited()

    async def test_unexpected_error_includes_exception_type(self) -> None:
        app = _make_app()
        with (
            patch(
                "deepagents_code.skills.load.list_skills",
                side_effect=TypeError("bad argument"),
            ),
            patch("deepagents_code.config.settings"),
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
                "deepagents_code.skills.load.load_skill_content",
                return_value="# Cached\nDo cached stuff",
            ) as mock_load,
            patch("deepagents_code.skills.load.list_skills") as mock_list,
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
                "deepagents_code.skills.load.list_skills",
                return_value=[skill],
            ) as mock_list,
            patch(
                "deepagents_code.skills.load.load_skill_content",
                return_value="# Fresh\nContent",
            ),
            patch("deepagents_code.config.settings"),
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


class TestPromptSkillTrustAndRetry:
    """Cover the in-the-moment trust prompt routed to on containment failure.

    All tests use a cache-hit (`_discovered_skills` pre-populated) so only the
    load + trust-prompt path runs, and drive it end-to-end through
    `_invoke_skill`. `_push_screen_wait` defaults to deny in `_make_app`; tests
    that approve override its return value. The first `load_skill_content` call
    raises the containment `PermissionError`; the second is the post-approval
    retry.
    """

    _CONTAINMENT_ERROR = PermissionError(
        "Skill path /tmp/evil resolves outside all allowed skill directories."
    )

    @staticmethod
    def _target_dir() -> str:
        """Resolved parent dir of the fake skill's SKILL.md (the trust key)."""
        return str(Path(_fake_skill()["path"]).resolve().parent)  # ty: ignore

    def _cache_hit_app(self) -> MagicMock:
        app = _make_app()
        app._discovered_skills = [_fake_skill()]
        return app

    async def test_allow_persists_and_retries(self) -> None:
        """Approving trusts the dir, extends the allowlist, and reads the skill."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, "# Instructions\nDo stuff"],
            ),
            patch(
                "deepagents_code.skills.trust.trust_skill_dir", return_value=True
            ) as mock_trust,
        ):
            await app._handle_skill_command("/skill:test-skill")

        mock_trust.assert_called_once_with(self._target_dir())
        app._send_to_agent.assert_awaited_once()
        assert "# Instructions" in app._send_to_agent.call_args[0][0]
        # The approved directory joins the in-session containment allowlist.
        assert Path(self._target_dir()) in app._skill_allowed_roots

    async def test_allow_extends_allowlist_exactly_once(self) -> None:
        """The approved dir is appended once, not duplicated across the two lists."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, "# Body"],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True),
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert app._skill_allowed_roots.count(Path(self._target_dir())) == 1

    async def test_deny_shows_error_and_remembers(self) -> None:
        """Denying surfaces the original error and suppresses re-prompts."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=False)
        with patch(
            "deepagents_code.skills.load.load_skill_content",
            side_effect=self._CONTAINMENT_ERROR,
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("resolves outside" in t for t in _app_message_texts(app))
        assert self._target_dir() in app._skill_trust_denied
        app._send_to_agent.assert_not_awaited()

    async def test_prior_deny_does_not_reprompt(self) -> None:
        """A dir denied earlier this session errors without a second prompt."""
        app = self._cache_hit_app()
        app._skill_trust_denied.add(self._target_dir())
        app._push_screen_wait = AsyncMock(return_value=True)
        with patch(
            "deepagents_code.skills.load.load_skill_content",
            side_effect=self._CONTAINMENT_ERROR,
        ):
            await app._handle_skill_command("/skill:test-skill")

        app._push_screen_wait.assert_not_awaited()
        assert any("resolves outside" in t for t in _app_message_texts(app))
        app._send_to_agent.assert_not_awaited()

    async def test_retry_permission_error_flags_location_change(self) -> None:
        """A retry containment failure (symlink swap) shows a distinct message."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, self._CONTAINMENT_ERROR],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True),
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("location changed" in t.lower() for t in _app_message_texts(app))
        app._send_to_agent.assert_not_awaited()

    async def test_retry_returns_none_shows_read_error(self) -> None:
        """A readable-but-empty retry result shows the read-failure message."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, None],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True),
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("could not read" in t.lower() for t in _app_message_texts(app))
        app._send_to_agent.assert_not_awaited()

    async def test_persist_failure_still_allows_but_notifies(self) -> None:
        """When the trust store can't be written, we read this session and warn."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, "# Body"],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=False),
        ):
            await app._handle_skill_command("/skill:test-skill")

        app.notify.assert_called_once()
        assert app.notify.call_args.kwargs.get("severity") == "warning"
        app._send_to_agent.assert_awaited_once()

    async def test_retry_os_error_shows_generic_read_failure(self) -> None:
        """A transient FS error on the post-approval retry surfaces distinctly."""
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=[self._CONTAINMENT_ERROR, OSError("disk gone")],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True),
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("after granting trust" in t.lower() for t in _app_message_texts(app))
        app._send_to_agent.assert_not_awaited()

    async def test_resolve_failure_fails_closed_without_prompting(self) -> None:
        """If resolving the skill path errors, refuse without prompting.

        A symlink loop introduced after the first containment check makes
        `_resolve_parent_dir` raise; the flow must mount the original error and
        never reach the trust prompt rather than crashing the worker.
        """
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                side_effect=self._CONTAINMENT_ERROR,
            ),
            patch(
                "deepagents_code.app._resolve_parent_dir",
                side_effect=OSError("ELOOP"),
            ),
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("resolves outside" in t for t in _app_message_texts(app))
        app._push_screen_wait.assert_not_awaited()
        app._send_to_agent.assert_not_awaited()

    async def test_modal_push_failure_fails_closed(self) -> None:
        """If the trust modal itself fails to display, treat it as a deny.

        The prompt runs inside `_invoke_skill`'s `except PermissionError` block,
        so an error escaping the push would not be caught by that method's own
        handlers and would surface as an unhandled worker crash. It must fail
        closed: mount the original error, remember the deny, and not proceed.
        """
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(side_effect=RuntimeError("screen stack"))
        with patch(
            "deepagents_code.skills.load.load_skill_content",
            side_effect=self._CONTAINMENT_ERROR,
        ):
            await app._handle_skill_command("/skill:test-skill")

        assert any("resolves outside" in t for t in _app_message_texts(app))
        assert self._target_dir() in app._skill_trust_denied
        app._send_to_agent.assert_not_awaited()

    async def test_allowlisted_dir_not_reprompted_same_session(self) -> None:
        """Once approved, a later invocation loads without a second prompt.

        The approved directory joins the in-session containment allowlist, so a
        subsequent read that no longer trips containment must not re-nag.
        """
        app = self._cache_hit_app()
        app._push_screen_wait = AsyncMock(return_value=True)
        with (
            patch(
                "deepagents_code.skills.load.load_skill_content",
                # 1st invoke: containment fails then the retry succeeds.
                # 2nd invoke: loads directly (dir already allowlisted).
                side_effect=[self._CONTAINMENT_ERROR, "# Body", "# Body"],
            ),
            patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True),
        ):
            await app._handle_skill_command("/skill:test-skill")
            await app._handle_skill_command("/skill:test-skill")

        assert app._push_screen_wait.await_count == 1
        assert app._send_to_agent.await_count == 2


class TestDiscoverSkillsAndRoots:
    """The containment roots must include persisted trust and extra dirs.

    This is the join that makes an in-session approval survive a relaunch. The
    real `discover_skills_and_roots` is exercised (not mocked wholesale) so a
    regression that drops the trust join, or re-resolves it instead of adding it
    as-is, is caught.
    """

    def _settings_with_no_builtin_roots(self, extra: list[Path]) -> MagicMock:
        settings = MagicMock()
        for getter in (
            "get_built_in_skills_dir",
            "get_user_skills_dir",
            "get_project_skills_dir",
            "get_user_agent_skills_dir",
            "get_project_agent_skills_dir",
            "get_user_claude_skills_dir",
            "get_project_claude_skills_dir",
        ):
            getattr(settings, getter).return_value = None
        settings.get_extra_skills_dirs.return_value = extra
        return settings

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

        with (
            patch(
                "deepagents_code.config.settings",
                self._settings_with_no_builtin_roots([link_extra]),
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


class TestResolveParentDir:
    """`_resolve_parent_dir` keys trust on the resolved target, not the link."""

    def test_resolves_then_takes_parent(self, tmp_path: Path) -> None:
        """A symlinked SKILL.md keys on the real target's parent directory.

        `resolve().parent` (what the code does) and `parent.resolve()` diverge
        when the SKILL.md file is itself a symlink and its discovery directory
        is a *real* directory: the former yields the symlink target's parent,
        the latter the discovery directory. The trust key must match what
        `load_skill_content` enforces (it resolves the file), i.e. the resolved
        target's parent — so the retry's containment check can pass.
        """
        from deepagents_code.app import _resolve_parent_dir

        disc = tmp_path / "disc"
        disc.mkdir()
        real = tmp_path / "real"
        real.mkdir()
        (real / "SKILL.md").write_text("# body", encoding="utf-8")
        link = disc / "SKILL.md"
        link.symlink_to(real / "SKILL.md")

        assert _resolve_parent_dir(link) == str(real.resolve())
        # Not the (real) discovery directory the link lexically sits in.
        assert _resolve_parent_dir(link) != str(disc.resolve())


class TestSkillTrustRealContainment:
    """Drive `_invoke_skill` through the REAL `load_skill_content` gate.

    Unlike `TestPromptSkillTrustAndRetry` (which mocks `load_skill_content`),
    these build real directories and symlinks so the actual containment check
    runs on both the initial load and the post-approval retry — the path a mock
    cannot exercise, where a wrong/empty `allowed_roots` would silently read a
    swapped target. `trust_skill_dir` is patched so no real state dir is written.
    """

    @staticmethod
    def _outside_skill(tmp_path: Path) -> tuple[MagicMock, Path, Path]:
        """Return (app, allowed root, outside target) for an escaping skill.

        The discovered skill's SKILL.md lives under `root` (the sole allowed
        root) but symlinks out to `outside/SKILL.md`, so the initial real
        containment check refuses it.
        """
        root = tmp_path / "skills"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text(
            "# Real body\nDo real stuff", encoding="utf-8"
        )
        link = root / "linked-skill"
        link.symlink_to(outside, target_is_directory=True)

        app = _make_app()
        app._skill_allowed_roots = [root.resolve()]
        app._discovered_skills = [
            _fake_skill(name="linked-skill", path=str(link / "SKILL.md"))
        ]
        return app, root, outside

    async def test_real_retry_reads_after_approval(self, tmp_path: Path) -> None:
        """Approving an outside skill reads it through the real gate on retry."""
        app, _root, outside = self._outside_skill(tmp_path)
        app._push_screen_wait = AsyncMock(return_value=True)
        with patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True):
            await app._handle_skill_command("/skill:linked-skill")

        app._send_to_agent.assert_awaited_once()
        assert "Do real stuff" in app._send_to_agent.call_args[0][0]
        # The resolved outside dir was actually admitted to the allowlist by the
        # real retry, not just asserted via a mock.
        assert outside.resolve() in app._skill_allowed_roots

    async def test_real_retry_refuses_symlink_swap_during_prompt(
        self, tmp_path: Path
    ) -> None:
        """Re-pointing the discovery symlink during the prompt is refused.

        The user approves the originally-resolved target dir, but the skill's
        SKILL.md is re-pointed to an unapproved location before the retry runs.
        The real containment check must refuse the swapped target rather than
        read it, even though a directory *was* just approved.
        """
        app, root, _outside = self._outside_skill(tmp_path)
        link = root / "linked-skill"
        evil = tmp_path / "evil"
        evil.mkdir()
        (evil / "SKILL.md").write_text("# stolen", encoding="utf-8")

        def approve_then_swap(_screen: object) -> bool:
            # Swap the discovery symlink to an unapproved target inside the
            # prompt window, then approve the (original) resolved dir.
            link.unlink()
            link.symlink_to(evil, target_is_directory=True)
            return True

        app._push_screen_wait = AsyncMock(side_effect=approve_then_swap)
        with patch("deepagents_code.skills.trust.trust_skill_dir", return_value=True):
            await app._handle_skill_command("/skill:linked-skill")

        assert any("location changed" in t.lower() for t in _app_message_texts(app))
        app._send_to_agent.assert_not_awaited()
