"""Unit tests for agent formatting functions."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

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
)
from deepagents_cli.config import Settings, get_glyphs


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


class TestGetSystemPromptModelIdentity:
    """Tests for model identity section in get_system_prompt."""

    def test_includes_model_identity_when_all_settings_present(self) -> None:
        """Test that model identity section is included when all settings are set."""
        mock_settings = Mock()
        mock_settings.model_name = "claude-sonnet-4-5-20250929"
        mock_settings.model_provider = "anthropic"
        mock_settings.model_context_limit = 200000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "claude-sonnet-4-5-20250929" in prompt
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


class TestCreateCliAgentSkillsSources:
    """Test that `create_cli_agent` wires built-in skills as first source."""

    def test_built_in_dir_is_first_source(self, tmp_path: Path) -> None:
        """Built-in skills dir should be the first (lowest-precedence) source.

        SkillsMiddleware uses last-one-wins dedup, so first = lowest precedence.
        """
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        built_in_dir = Settings.get_built_in_skills_dir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_project_skills_dir.return_value = None
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

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch("deepagents_cli.agent.create_deep_agent", return_value=mock_agent),
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
        # Built-in dir should be the first source
        assert sources[0] == str(built_in_dir)
        # User skills dir should follow
        assert sources[1] == str(skills_dir)


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

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_cli.agent.FilesystemBackend"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
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

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_cli.agent.FilesystemBackend"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
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
