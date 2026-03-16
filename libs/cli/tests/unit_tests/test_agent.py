"""Unit tests for agent formatting functions."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState
    from langchain.messages import ToolCall
    from langgraph.runtime import Runtime

from deepagents_cli.agent import (
    DEFAULT_AGENT_NAME,
    _format_edit_file_description,
    _format_execute_description,
    _format_fetch_url_description,
    _format_task_description,
    _format_web_search_description,
    _format_write_file_description,
    create_cli_agent,
    get_system_prompt,
    list_agents,
    load_async_subagents,
)
from deepagents_cli.config import Settings, get_glyphs
from deepagents_cli.project_utils import ProjectContext


def _make_fake_chat_model() -> GenericFakeChatModel:
    """Create a fake chat model compatible with summarization middleware."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}
    return model


def test_format_write_file_description_create_new_file(tmp_path: Path) -> None:
    """Test write_file description for creating a new file."""
    new_file = tmp_path / "new_file.py"
    tool_call = cast(
        "ToolCall",
        {
            "name": "write_file",
            "args": {
                "file_path": str(new_file),
                "content": "def hello():\n    return 'world'\n",
            },
            "id": "call-1",
        },
    )

    description = _format_write_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert f"File: {new_file}" in description
    assert "Action: Create file" in description
    assert "Lines: 2" in description


def test_format_write_file_description_overwrite_existing_file(tmp_path: Path) -> None:
    """Test write_file description for overwriting an existing file."""
    existing_file = tmp_path / "existing.py"
    existing_file.write_text("old content")

    tool_call = cast(
        "ToolCall",
        {
            "name": "write_file",
            "args": {
                "file_path": str(existing_file),
                "content": "line1\nline2\nline3\n",
            },
            "id": "call-2",
        },
    )

    description = _format_write_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert f"File: {existing_file}" in description
    assert "Action: Overwrite file" in description
    assert "Lines: 3" in description


def test_format_edit_file_description_single_occurrence():
    """Test edit_file description for single occurrence replacement."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "edit_file",
            "args": {
                "file_path": "/path/to/file.py",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": False,
            },
            "id": "call-3",
        },
    )

    description = _format_edit_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "File: /path/to/file.py" in description
    assert "Action: Replace text (single occurrence)" in description


def test_format_edit_file_description_all_occurrences():
    """Test edit_file description for replacing all occurrences."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "edit_file",
            "args": {
                "file_path": "/path/to/file.py",
                "old_string": "foo",
                "new_string": "bar",
                "replace_all": True,
            },
            "id": "call-4",
        },
    )

    description = _format_edit_file_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "File: /path/to/file.py" in description
    assert "Action: Replace text (all occurrences)" in description


def test_format_web_search_description():
    """Test web_search description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "web_search",
            "args": {
                "query": "python async programming",
                "max_results": 10,
            },
            "id": "call-5",
        },
    )

    description = _format_web_search_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Query: python async programming" in description
    assert "Max results: 10" in description
    assert f"{get_glyphs().warning}  This will use Tavily API credits" in description


def test_format_web_search_description_default_max_results():
    """Test web_search description with default max_results."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "web_search",
            "args": {
                "query": "langchain tutorial",
            },
            "id": "call-6",
        },
    )

    description = _format_web_search_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Query: langchain tutorial" in description
    assert "Max results: 5" in description


def test_format_fetch_url_description():
    """Test fetch_url description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {
                "url": "https://example.com/docs",
                "timeout": 60,
            },
            "id": "call-7",
        },
    )

    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "URL: https://example.com/docs" in description
    assert "Timeout: 60s" in description
    warning = get_glyphs().warning
    assert f"{warning}  Will fetch and convert web content to markdown" in description


def test_format_fetch_url_description_default_timeout():
    """Test fetch_url description with default timeout."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {
                "url": "https://api.example.com",
            },
            "id": "call-8",
        },
    )

    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "URL: https://api.example.com" in description
    assert "Timeout: 30s" in description


def test_format_task_description():
    """Test task (subagent) description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "task",
            "args": {
                "description": "Analyze code structure and identify main components.",
                "subagent_type": "general-purpose",
            },
            "id": "call-9",
        },
    )

    description = _format_task_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Subagent Type: general-purpose" in description
    assert "Task Instructions:" in description
    assert "Analyze code structure and identify main components." in description
    warning = get_glyphs().warning
    assert (
        f"{warning}  Subagent will have access to file operations and shell commands"
        in description
    )


def test_format_task_description_truncates_long_description():
    """Test task description truncates long descriptions."""
    long_description = "x" * 600  # 600 characters
    tool_call = cast(
        "ToolCall",
        {
            "name": "task",
            "args": {
                "description": long_description,
                "subagent_type": "general-purpose",
            },
            "id": "call-10",
        },
    )

    description = _format_task_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Subagent Type: general-purpose" in description
    assert "..." in description
    # Description should be truncated to 500 chars + "..."
    assert len(description) < len(long_description) + 300


def test_format_execute_description():
    """Test execute command description formatting."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "execute",
            "args": {
                "command": "python script.py",
            },
            "id": "call-12",
        },
    )

    description = _format_execute_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Execute Command: python script.py" in description
    assert "Working Directory:" in description


def test_format_execute_description_with_hidden_unicode():
    """Hidden Unicode in command should trigger warning and marker display."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "execute",
            "args": {"command": "echo a\u202eb"},
            "id": "call-13",
        },
    )
    description = _format_execute_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "Execute Command: echo ab" in description
    assert "Hidden Unicode detected" in description
    assert "U+202E" in description
    assert "Raw:" in description


def test_format_fetch_url_description_with_suspicious_url():
    """Suspicious URL should trigger warning lines in fetch_url description."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {"url": "https://аpple.com"},
            "id": "call-14",
        },
    )
    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "URL warning" in description


def test_format_fetch_url_description_with_hidden_unicode_in_url():
    """Hidden Unicode in URL should be stripped from display."""
    tool_call = cast(
        "ToolCall",
        {
            "name": "fetch_url",
            "args": {"url": "https://exa\u200bmple.com"},
            "id": "call-15",
        },
    )
    description = _format_fetch_url_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )
    assert "URL: https://example.com" in description
    assert "\u200b" not in description


class TestGetSystemPromptModelIdentity:
    """Tests for model identity section in get_system_prompt."""

    def test_includes_model_identity_when_all_settings_present(self) -> None:
        """Test that model identity section is included when all settings are set."""
        mock_settings = Mock()
        mock_settings.model_name = "claude-sonnet-4-6"
        mock_settings.model_provider = "anthropic"
        mock_settings.model_context_limit = 200000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "claude-sonnet-4-6" in prompt
        assert "(provider: anthropic)" in prompt
        assert "Your context window is 200,000 tokens." in prompt

    def test_excludes_model_identity_when_model_name_is_none(self) -> None:
        """Test that model identity section is excluded when model_name is None."""
        mock_settings = Mock()
        mock_settings.model_name = None
        mock_settings.model_provider = "anthropic"
        mock_settings.model_context_limit = 200000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" not in prompt

    def test_excludes_provider_when_not_set(self) -> None:
        """Test that provider is excluded when model_provider is None."""
        mock_settings = Mock()
        mock_settings.model_name = "gpt-4"
        mock_settings.model_provider = None
        mock_settings.model_context_limit = 128000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "gpt-4" in prompt
        assert "(provider:" not in prompt
        assert "Your context window is 128,000 tokens." in prompt

    def test_excludes_context_limit_when_not_set(self) -> None:
        """Test that context limit is excluded when model_context_limit is None."""
        mock_settings = Mock()
        mock_settings.model_name = "gemini-3-pro"
        mock_settings.model_provider = "google"
        mock_settings.model_context_limit = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "gemini-3-pro" in prompt
        assert "(provider: google)" in prompt
        assert "context window" not in prompt

    def test_model_identity_with_only_model_name(self) -> None:
        """Test model identity section with only model_name set."""
        mock_settings = Mock()
        mock_settings.model_name = "test-model"
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "You are running as model `test-model`." in prompt
        assert "(provider:" not in prompt
        assert "context window" not in prompt


class TestGetSystemPromptNonInteractive:
    """Tests for interactive vs non-interactive system prompt."""

    def test_interactive_prompt_mentions_interactive_cli(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        assert "interactive CLI" in prompt
        assert "ask questions before acting" in prompt

    def test_non_interactive_prompt_mentions_headless(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "non-interactive" in prompt
        assert "no human" in prompt.lower()

    def test_non_interactive_prompt_does_not_ask_questions(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "ask questions before acting" not in prompt

    def test_non_interactive_prompt_instructs_autonomous_execution(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "Do NOT ask clarifying questions" in prompt
        assert "reasonable assumptions" in prompt

    def test_non_interactive_prompt_requires_non_interactive_commands(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "non-interactive command variants" in prompt
        assert "npm init -y" in prompt

    def test_default_is_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "interactive CLI" in prompt


class TestGetSystemPromptCwdOSError:
    """Tests for Path.cwd() OSError handling in get_system_prompt."""

    def test_falls_back_on_cwd_oserror(self) -> None:
        """get_system_prompt should not crash when Path.cwd() raises OSError."""
        mock_settings = Mock()
        mock_settings.model_name = None

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.Path.cwd", side_effect=OSError("deleted")),
        ):
            prompt = get_system_prompt("test-agent")

        assert "Current Working Directory" in prompt


class TestGetSystemPromptPlaceholderValidation:
    """Tests for unreplaced placeholder detection."""

    def test_no_unreplaced_placeholders_in_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        # No raw {placeholder} patterns should remain
        import re

        assert not re.findall(r"\{[a-z_]+\}", prompt)

    def test_no_unreplaced_placeholders_in_non_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        import re

        assert not re.findall(r"\{[a-z_]+\}", prompt)


class TestCreateCliAgentInteractiveForwarding:
    """Tests for interactive parameter forwarding in create_cli_agent."""

    def test_forwards_interactive_false_to_get_system_prompt(
        self, tmp_path: Path
    ) -> None:
        """create_cli_agent should forward interactive=False to get_system_prompt."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            patch("deepagents_cli.agent.get_system_prompt") as mock_get_prompt,
        ):
            mock_get_prompt.return_value = "mocked prompt"
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                interactive=False,
            )

        mock_get_prompt.assert_called_once()
        _, kwargs = mock_get_prompt.call_args
        assert kwargs["interactive"] is False

    def test_explicit_system_prompt_ignores_interactive(self, tmp_path: Path) -> None:
        """Explicit system_prompt should be used verbatim, ignoring interactive."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            patch("deepagents_cli.agent.get_system_prompt") as mock_get_prompt,
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                system_prompt="custom prompt",
                interactive=False,
            )

        # get_system_prompt should NOT be called when system_prompt is provided
        mock_get_prompt.assert_not_called()


class TestDefaultAgentName:
    """Tests for the DEFAULT_AGENT_NAME constant."""

    def test_default_agent_name_value(self) -> None:
        """Guard against accidental renames of the default agent identifier.

        Other modules (main.py, commands.py) rely on this value matching
        the directory name under `~/.deepagents/`.
        """
        assert DEFAULT_AGENT_NAME == "agent"


class TestListAgents:
    """Tests for list_agents output."""

    def test_default_agent_marked(self, tmp_path: Path) -> None:
        """Test that the default agent is labeled as (default) in list output."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create the default agent directory with AGENTS.md
        default_dir = agents_dir / DEFAULT_AGENT_NAME
        default_dir.mkdir()
        (default_dir / "AGENTS.md").touch()

        # Create a non-default agent
        other_dir = agents_dir / "researcher"
        other_dir.mkdir()
        (other_dir / "AGENTS.md").touch()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        output: list[str] = []

        def capture_print(*args: Any, **_: Any) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.console") as mock_console,
        ):
            mock_console.print = capture_print
            list_agents()

        joined = "\n".join(output)
        assert "(default)" in joined
        # Only the default agent should be marked
        assert joined.count("(default)") == 1
        # The default agent name should appear with the (default) label
        assert DEFAULT_AGENT_NAME in joined
        # The other agent should NOT be marked as default
        for line in output:
            if "researcher" in line and "(default)" in line:
                msg = "Non-default agent should not be marked as (default)"
                raise AssertionError(msg)

    def test_non_default_agent_not_marked(self, tmp_path: Path) -> None:
        """Test that non-default agents are not labeled as (default)."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Only create a non-default agent
        custom_dir = agents_dir / "researcher"
        custom_dir.mkdir()
        (custom_dir / "AGENTS.md").touch()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        output: list[str] = []

        def capture_print(*args: Any, **_: Any) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.console") as mock_console,
        ):
            mock_console.print = capture_print
            list_agents()

        joined = "\n".join(output)
        assert "(default)" not in joined


class TestListAgentsJson:
    """Tests for list_agents JSON output."""

    def test_json_output_with_agents(self, tmp_path: Path) -> None:
        """JSON output returns array of agent dicts."""
        import json
        from io import StringIO

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        default_dir = agents_dir / DEFAULT_AGENT_NAME
        default_dir.mkdir()
        (default_dir / "AGENTS.md").touch()

        other_dir = agents_dir / "researcher"
        other_dir.mkdir()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        buf = StringIO()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            list_agents(output_format="json")

        result = json.loads(buf.getvalue())
        assert result["schema_version"] == 1
        assert result["command"] == "list"
        agents = result["data"]
        assert len(agents) == 2

        default = next(a for a in agents if a["name"] == DEFAULT_AGENT_NAME)
        assert default["is_default"] is True
        assert default["has_agents_md"] is True

        researcher = next(a for a in agents if a["name"] == "researcher")
        assert researcher["is_default"] is False
        assert researcher["has_agents_md"] is False

    def test_json_output_empty(self, tmp_path: Path) -> None:
        """JSON output returns empty array when no agents exist."""
        import json
        from io import StringIO

        agents_dir = tmp_path / "empty"
        agents_dir.mkdir()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        buf = StringIO()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            list_agents(output_format="json")

        result = json.loads(buf.getvalue())
        assert result["data"] == []


class TestResetAgentJson:
    """Tests for reset_agent JSON output."""

    def test_json_output_default_reset(self, tmp_path: Path) -> None:
        """JSON output after resetting to default."""
        import json
        from io import StringIO

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        buf = StringIO()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            from deepagents_cli.agent import reset_agent

            reset_agent("coder", output_format="json")

        result = json.loads(buf.getvalue())
        assert result["command"] == "reset"
        assert result["data"]["agent"] == "coder"
        assert result["data"]["reset_to"] == "default"
        assert "path" in result["data"]


class TestCreateCliAgentSkillsSources:
    """Test that `create_cli_agent` wires skills sources in precedence order."""

    def test_skills_source_precedence_order(self, tmp_path: Path) -> None:
        """Skills sources should be wired from lowest to highest precedence.

        SkillsMiddleware uses last-one-wins dedup, so source order matters.
        """
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        user_agent_skills_dir = tmp_path / "user-agent-skills"
        user_agent_skills_dir.mkdir()
        project_skills_dir = tmp_path / "project-skills"
        project_skills_dir.mkdir()
        project_agent_skills_dir = tmp_path / "project-agent-skills"
        project_agent_skills_dir.mkdir()
        built_in_dir = Settings.get_built_in_skills_dir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = user_agent_skills_dir
        mock_settings.get_project_skills_dir.return_value = project_skills_dir
        mock_settings.get_project_agent_skills_dir.return_value = (
            project_agent_skills_dir
        )
        mock_settings.get_built_in_skills_dir.return_value = built_in_dir
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        # Needed by get_system_prompt() which formats model identity
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        captured_sources: list[list[str]] = []

        class FakeSkillsMiddleware:
            """Capture the sources arg passed to SkillsMiddleware."""

            def __init__(self, **kwargs: Any) -> None:
                captured_sources.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=True,
                enable_shell=False,
            )

        assert len(captured_sources) == 1
        sources = captured_sources[0]
        assert sources == [
            str(built_in_dir),
            str(skills_dir),
            str(user_agent_skills_dir),
            str(project_skills_dir),
            str(project_agent_skills_dir),
        ]


class TestCreateCliAgentMemorySources:
    """Test that `create_cli_agent` wires project AGENTS.md into memory sources."""

    def test_project_agent_md_paths_in_memory_sources(self, tmp_path: Path) -> None:
        """Project AGENTS.md paths should be passed to MemoryMiddleware sources."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        project_inner = tmp_path / ".deepagents" / "AGENTS.md"
        project_root = tmp_path / "AGENTS.md"

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = [
            project_inner,
            project_root,
        ]
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = tmp_path

        captured: list[list[str]] = []

        class FakeMemoryMiddleware:
            """Capture the sources arg passed to MemoryMiddleware."""

            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_cli.agent.FilesystemBackend"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=True,
                enable_skills=False,
                enable_shell=False,
            )

        assert len(captured) == 1
        sources = captured[0]
        # User AGENTS.md is always first
        assert sources[0] == str(agent_dir / "AGENTS.md")
        # Both project paths follow
        assert sources[1] == str(project_inner)
        assert sources[2] == str(project_root)
        assert len(sources) == 3

    def test_empty_project_paths_no_extra_sources(self, tmp_path: Path) -> None:
        """Empty project path list should not add extra memory sources."""
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        captured: list[list[str]] = []

        class FakeMemoryMiddleware:
            """Capture the sources arg passed to MemoryMiddleware."""

            def __init__(self, **kwargs: Any) -> None:
                captured.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_cli.agent.FilesystemBackend"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=True,
                enable_skills=False,
                enable_shell=False,
            )

        assert len(captured) == 1
        sources = captured[0]
        # Only user AGENTS.md, no project paths
        assert sources == [str(agent_dir / "AGENTS.md")]


class TestCreateCliAgentProjectContext:
    """Tests for explicit project context in `create_cli_agent`."""

    def test_project_context_drives_project_skills_and_subagents(
        self, tmp_path: Path
    ) -> None:
        """Project-sensitive paths should come from explicit project context."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()

        project_skills_dir = project_root / ".deepagents" / "skills"
        project_skills_dir.mkdir(parents=True)
        project_agent_skills_dir = project_root / ".agents" / "skills"
        project_agent_skills_dir.mkdir(parents=True)
        project_agents_dir = project_root / ".deepagents" / "agents"
        project_agents_dir.mkdir(parents=True)
        project_context = ProjectContext.from_user_cwd(user_cwd)

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        user_skills_dir = tmp_path / "user-skills"
        user_skills_dir.mkdir()
        user_agent_skills_dir = tmp_path / "user-agent-skills"
        user_agent_skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = user_agent_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None
        mock_settings.user_langchain_project = None

        captured_sources: list[list[str]] = []

        class FakeSkillsMiddleware:
            """Capture the sources argument passed to SkillsMiddleware."""

            def __init__(self, **kwargs: Any) -> None:
                captured_sources.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.list_subagents", return_value=[]) as mock_list,
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch("deepagents._models.init_chat_model", return_value=fake_model),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=True,
                enable_shell=False,
                project_context=project_context,
            )

        assert len(captured_sources) == 1
        sources = captured_sources[0]
        assert str(project_skills_dir) in sources
        assert str(project_agent_skills_dir) in sources
        mock_list.assert_called_once_with(
            user_agents_dir=tmp_path / "agents",
            project_agents_dir=project_agents_dir,
        )

    def test_project_context_drives_project_agents_md_paths(
        self, tmp_path: Path
    ) -> None:
        """Memory sources should use project AGENTS from explicit context."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()

        deepagents_md = project_root / ".deepagents" / "AGENTS.md"
        deepagents_md.parent.mkdir(parents=True)
        deepagents_md.write_text("deepagents instructions")
        root_md = project_root / "AGENTS.md"
        root_md.write_text("root instructions")
        project_context = ProjectContext.from_user_cwd(user_cwd)

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        user_skills_dir = tmp_path / "skills"
        user_skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None
        mock_settings.user_langchain_project = None

        captured_sources: list[list[str]] = []

        class FakeMemoryMiddleware:
            """Capture the sources argument passed to MemoryMiddleware."""

            def __init__(self, **kwargs: Any) -> None:
                captured_sources.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_cli.agent.FilesystemBackend"),
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch("deepagents._models.init_chat_model", return_value=fake_model),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=True,
                enable_skills=False,
                enable_shell=False,
                project_context=project_context,
            )

        assert len(captured_sources) == 1
        sources = captured_sources[0]
        assert sources[0] == str(agent_dir / "AGENTS.md")
        assert sources[1:] == [str(deepagents_md), str(root_md)]

    def test_project_context_sets_local_shell_root_dir(self, tmp_path: Path) -> None:
        """Shell backend root should follow the explicit user working directory."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()
        project_context = ProjectContext.from_user_cwd(user_cwd)

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        user_skills_dir = tmp_path / "skills"
        user_skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None
        mock_settings.user_langchain_project = None

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_backend = Mock()

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch(
                "deepagents_cli.agent.LocalShellBackend", return_value=mock_backend
            ) as mock_shell,
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch("deepagents._models.init_chat_model", return_value=fake_model),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
                project_context=project_context,
            )

        assert mock_shell.call_args.kwargs["root_dir"] == user_cwd

    def test_cwd_sets_local_filesystem_root_dir_without_shell(
        self, tmp_path: Path
    ) -> None:
        """Filesystem backend root should follow the explicit working directory."""
        user_cwd = tmp_path / "project" / "src"
        user_cwd.mkdir(parents=True)

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        user_skills_dir = tmp_path / "skills"
        user_skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = user_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.FilesystemBackend") as mock_filesystem,
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
            patch("deepagents._models.init_chat_model", return_value=fake_model),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                cwd=user_cwd,
            )

        assert mock_filesystem.call_args_list[0].kwargs["root_dir"] == user_cwd


class TestMiddlewareStackConformance:
    """Verify all middleware passed to create_deep_agent inherits AgentMiddleware."""

    def test_all_middleware_inherit_agent_middleware(self, tmp_path: Path) -> None:
        """Every middleware in the stack must be an AgentMiddleware subclass.

        This prevents runtime errors like 'has no attribute wrap_tool_call'
        when the agent framework iterates over the middleware list.
        """
        from langchain.agents.middleware.types import AgentMiddleware

        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = (
            Settings.get_built_in_skills_dir()
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        captured_middleware: list[list[Any]] = []

        def capture_create_agent(**kwargs: Any) -> Mock:
            captured_middleware.append(kwargs.get("middleware", []))
            agent = Mock()
            agent.with_config.return_value = agent
            return agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                side_effect=capture_create_agent,
            ),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=True,
                enable_skills=True,
                enable_shell=False,
            )

        assert len(captured_middleware) == 1
        middleware_list = captured_middleware[0]
        assert len(middleware_list) > 0, "Expected at least one middleware"

        for mw in middleware_list:
            assert isinstance(mw, AgentMiddleware), (
                f"{type(mw).__name__} does not inherit from AgentMiddleware"
            )


class TestLoadAsyncSubagents:
    def test_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        result = load_async_subagents(tmp_path / "nonexistent.toml")
        assert result == []

    def test_returns_empty_when_no_section(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text('[models]\ndefault = "gpt-4"\n')
        result = load_async_subagents(config)
        assert result == []

    def test_loads_valid_async_subagent(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            "[async_subagents.researcher]\n"
            'description = "Research agent"\n'
            'url = "https://my-deployment.langsmith.dev"\n'
            'graph_id = "agent"\n'
        )
        result = load_async_subagents(config)
        assert len(result) == 1
        assert result[0]["name"] == "researcher"
        assert result[0]["description"] == "Research agent"
        assert result[0]["url"] == "https://my-deployment.langsmith.dev"
        assert result[0]["graph_id"] == "agent"

    def test_loads_multiple_subagents(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            "[async_subagents.researcher]\n"
            'description = "Research agent"\n'
            'url = "https://research.langsmith.dev"\n'
            'graph_id = "agent"\n'
            "\n"
            "[async_subagents.coder]\n"
            'description = "Coding agent"\n'
            'url = "https://coder.langsmith.dev"\n'
            'graph_id = "coder"\n'
        )
        result = load_async_subagents(config)
        assert len(result) == 2
        names = {a["name"] for a in result}
        assert names == {"researcher", "coder"}

    def test_skips_entry_missing_required_fields(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            '[async_subagents.incomplete]\ndescription = "Missing url and graph_id"\n'
        )
        result = load_async_subagents(config)
        assert result == []

    def test_includes_optional_headers(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            "[async_subagents.custom]\n"
            'description = "Custom agent"\n'
            'url = "https://custom.langsmith.dev"\n'
            'graph_id = "agent"\n'
            "\n"
            "[async_subagents.custom.headers]\n"
            'x-custom = "value"\n'
        )
        result = load_async_subagents(config)
        assert len(result) == 1
        assert result[0]["headers"] == {"x-custom": "value"}

    def test_handles_invalid_toml(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[")
        result = load_async_subagents(config)
        assert result == []
