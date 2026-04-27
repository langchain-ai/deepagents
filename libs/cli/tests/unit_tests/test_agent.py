"""Unit tests for agent formatting functions."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

import pytest
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
    build_model_identity_section,
    create_cli_agent,
    get_available_agent_names,
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

    assert "Action: Create file" in description
    assert "File:" not in description


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

    assert "Action: Overwrite file" in description
    assert "File:" not in description


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

    assert "Action: Replace text (single occurrence)" in description
    assert "File:" not in description


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

    assert "Action: Replace text (all occurrences)" in description
    assert "File:" not in description


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
    msg = "Subagent will have access to file operations and shell commands"
    assert f"{warning} {msg} {warning}" in description
    assert description.index(warning) < description.index("Task Instructions:")


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


class TestBuildModelIdentitySection:
    """Direct tests for build_model_identity_section."""

    def test_empty_when_no_name(self) -> None:
        assert build_model_identity_section(None) == ""

    def test_basic_name_only(self) -> None:
        result = build_model_identity_section("gpt-4o")
        assert "You are running as model `gpt-4o`." in result
        assert "may not be available" not in result

    def test_unsupported_single(self) -> None:
        result = build_model_identity_section(
            "test-model", unsupported_modalities=frozenset({"audio"})
        )
        assert "Audio input may not be available for this model." in result
        assert "Do not attempt to read or process" in result

    def test_unsupported_two_uses_and(self) -> None:
        result = build_model_identity_section(
            "test-model",
            unsupported_modalities=frozenset({"video", "audio"}),
        )
        assert "Audio and video input may not be available" in result

    def test_unsupported_multiple_uses_oxford_comma(self) -> None:
        result = build_model_identity_section(
            "test-model",
            unsupported_modalities=frozenset({"video", "audio", "image"}),
        )
        assert "Audio, image, and video input may not be available" in result

    def test_unsupported_empty_frozenset_no_warning(self) -> None:
        result = build_model_identity_section(
            "test-model", unsupported_modalities=frozenset()
        )
        assert "may not be available" not in result

    def test_all_fields(self) -> None:
        result = build_model_identity_section(
            "deepseek-r1",
            provider="deepseek",
            context_limit=64000,
            unsupported_modalities=frozenset({"image", "pdf"}),
        )
        assert "deepseek-r1" in result
        assert "(provider: deepseek)" in result
        assert "64,000 tokens" in result
        assert "Image and pdf input may not be available" in result


class TestGetSystemPromptModelIdentity:
    """Tests for model identity section in get_system_prompt."""

    def test_includes_model_identity_when_all_settings_present(self) -> None:
        """Test that model identity section is included when all settings are set."""
        mock_settings = Mock()
        mock_settings.model_name = "claude-sonnet-4-6"
        mock_settings.model_provider = "anthropic"
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = 200000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" not in prompt

    def test_excludes_provider_when_not_set(self) -> None:
        """Test that provider is excluded when model_provider is None."""
        mock_settings = Mock()
        mock_settings.model_name = "gpt-4"
        mock_settings.model_provider = None
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" in prompt
        assert "You are running as model `test-model`." in prompt
        assert "(provider:" not in prompt
        assert "context window" not in prompt

    def test_includes_unsupported_modalities_warning(self) -> None:
        """Test that unsupported modalities are surfaced in the prompt."""
        mock_settings = Mock()
        mock_settings.model_name = "deepseek-r1"
        mock_settings.model_provider = "deepseek"
        mock_settings.model_unsupported_modalities = frozenset(
            {"image", "audio", "video", "pdf"}
        )
        mock_settings.model_context_limit = 64000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "Audio, image, pdf, and video input may not be available" in prompt

    def test_single_unsupported_modality(self) -> None:
        """Test warning with a single unsupported modality."""
        mock_settings = Mock()
        mock_settings.model_name = "test-model"
        mock_settings.model_provider = "test"
        mock_settings.model_unsupported_modalities = frozenset({"audio"})
        mock_settings.model_context_limit = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "Audio input may not be available" in prompt

    def test_no_modality_warning_when_all_supported(self) -> None:
        """Test that no modality warning appears when all modalities supported."""
        mock_settings = Mock()
        mock_settings.model_name = "claude-opus-4-6"
        mock_settings.model_provider = "anthropic"
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = 200000

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "may not be available" not in prompt


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

    def test_interactive_todo_section_asks_user_before_starting(self) -> None:
        """Interactive mode should require plan approval before first in_progress."""
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        assert "Wait for the user's response before marking the first todo" in prompt

    def test_non_interactive_todo_section_does_not_wait_for_user(self) -> None:
        """Headless mode must not contradict 'no human' guidance in todo rules."""
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        wait_for_user = "Wait for the user's response before marking the first todo"
        assert wait_for_user not in prompt
        assert "do NOT ask the user to approve your plan" in prompt
        assert "mark the first item `in_progress` immediately" in prompt


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


class TestGetSystemPromptSandbox:
    """Tests for sandbox-specific system prompt content."""

    def test_sandbox_includes_no_local_filesystem_warning(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="modal")

        assert "do NOT have access to the user's local filesystem" in prompt

    def test_sandbox_includes_working_dir_constraint(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="modal")

        assert "/workspace" in prompt
        assert "remote Linux sandbox" in prompt

    def test_sandbox_warns_about_subagent_paths(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="daytona")

        assert "subagents" in prompt
        assert "/home/daytona" in prompt

    def test_local_mode_omits_sandbox_warnings(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_cli.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "do NOT have access to the user's local filesystem" not in prompt
        assert "remote Linux sandbox" not in prompt


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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        user_claude_skills_dir = tmp_path / "user-claude-skills"
        user_claude_skills_dir.mkdir()
        project_claude_skills_dir = tmp_path / "project-claude-skills"
        project_claude_skills_dir.mkdir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = user_agent_skills_dir
        mock_settings.get_project_skills_dir.return_value = project_skills_dir
        mock_settings.get_project_agent_skills_dir.return_value = (
            project_agent_skills_dir
        )
        mock_settings.get_built_in_skills_dir.return_value = built_in_dir
        mock_settings.get_user_claude_skills_dir.return_value = user_claude_skills_dir
        mock_settings.get_project_claude_skills_dir.return_value = (
            project_claude_skills_dir
        )
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        # Needed by get_system_prompt() which formats model identity
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_unsupported_modalities = frozenset()
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
            (str(built_in_dir), "Built-in"),
            (str(skills_dir), "User Deepagents"),
            (str(user_agent_skills_dir), "User Agents"),
            (str(project_skills_dir), "Project Deepagents"),
            (str(project_agent_skills_dir), "Project Agents"),
            (str(tmp_path / "user-claude-skills"), "User Claude"),
            (str(tmp_path / "project-claude-skills"), "Project Claude"),
        ]

        # End-to-end: the captured tuple list should produce distinct
        # labels when formatted by the real middleware. Guards against
        # a regression that drops labels back to leaf-only derivation
        # (which would collapse user- vs project-scoped `.claude/skills`
        # and `.agents/skills` / `.deepagents/skills` directories).
        from deepagents.middleware.skills import (
            SkillsMiddleware as RealSkillsMiddleware,
        )

        real_middleware = RealSkillsMiddleware(
            backend=None,  # type: ignore[arg-type]
            sources=sources,
        )
        rendered = real_middleware._format_skills_locations()
        for expected in (
            "**Built-in Skills**:",
            "**User Deepagents Skills**:",
            "**User Agents Skills**:",
            "**Project Deepagents Skills**:",
            "**Project Agents Skills**:",
            "**User Claude Skills**:",
            "**Project Claude Skills**:",
        ):
            assert expected in rendered, f"missing {expected!r} in:\n{rendered}"
        assert rendered.rstrip().endswith("(higher priority)")

    def test_skills_sources_fallback_to_bare_paths_on_old_sdk(
        self, tmp_path: Path
    ) -> None:
        """If the installed SDK lacks `SkillSource`, CLI passes bare paths.

        Backwards-compat: SDKs < 0.5.4 only accept `list[str]`. The CLI
        detects the missing alias at import time and strips labels
        before handing sources to `SkillsMiddleware`, so the middleware
        never receives an unsupported tuple.
        """
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        user_agent_skills_dir = tmp_path / "user-agent-skills"
        user_agent_skills_dir.mkdir()
        built_in_dir = Settings.get_built_in_skills_dir()

        mock_settings = Mock()
        mock_settings.ensure_agent_dir.return_value = agent_dir
        mock_settings.ensure_user_skills_dir.return_value = skills_dir
        mock_settings.get_user_agent_skills_dir.return_value = user_agent_skills_dir
        mock_settings.get_project_skills_dir.return_value = None
        mock_settings.get_project_agent_skills_dir.return_value = None
        mock_settings.get_built_in_skills_dir.return_value = built_in_dir
        mock_settings.get_user_claude_skills_dir.return_value = tmp_path / "nonexistent"
        mock_settings.get_project_claude_skills_dir.return_value = None
        mock_settings.get_user_agent_md_path.return_value = agent_dir / "AGENTS.md"
        mock_settings.get_project_agent_md_path.return_value = []
        mock_settings.get_user_agents_dir.return_value = tmp_path / "agents"
        mock_settings.get_project_agents_dir.return_value = None
        mock_settings.model_name = None
        mock_settings.model_provider = None
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        captured_sources: list[list[Any]] = []

        class FakeSkillsMiddleware:
            def __init__(self, **kwargs: Any) -> None:
                captured_sources.append(kwargs.get("sources", []))

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent._SUPPORTS_SKILL_SOURCE_TUPLES", False),
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
        # Fallback stripped all labels; middleware receives bare strings.
        assert sources == [
            str(built_in_dir),
            str(skills_dir),
            str(user_agent_skills_dir),
        ]
        for source in sources:
            assert isinstance(source, str), f"expected str, got {type(source)!r}"


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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        # Sources are (path, label) tuples; assert the project paths are wired.
        source_paths = [s[0] if isinstance(s, tuple) else s for s in sources]
        assert str(project_skills_dir) in source_paths
        assert str(project_agent_skills_dir) in source_paths
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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
        mock_settings.model_unsupported_modalities = frozenset()
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


class TestEnableAskUser:
    """Verify enable_ask_user controls AskUserMiddleware inclusion."""

    def _capture_middleware(
        self, tmp_path: Path, *, enable_ask_user: bool
    ) -> list[Any]:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(exist_ok=True)
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(exist_ok=True)

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
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = None
        mock_settings.project_root = None

        captured: list[list[Any]] = []

        def capture(**kwargs: Any) -> Mock:
            captured.append(kwargs.get("middleware", []))
            agent = Mock()
            agent.with_config.return_value = agent
            return agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                side_effect=capture,
            ),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_ask_user=enable_ask_user,
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        return captured[0]

    def test_ask_user_included_when_enabled(self, tmp_path: Path) -> None:
        from deepagents_cli.ask_user import AskUserMiddleware

        middleware = self._capture_middleware(tmp_path, enable_ask_user=True)
        assert any(isinstance(mw, AskUserMiddleware) for mw in middleware)

    def test_ask_user_excluded_when_disabled(self, tmp_path: Path) -> None:
        from deepagents_cli.ask_user import AskUserMiddleware

        middleware = self._capture_middleware(tmp_path, enable_ask_user=False)
        assert not any(isinstance(mw, AskUserMiddleware) for mw in middleware)


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


class TestShellAllowListMiddleware:
    """Tests for inline shell command validation middleware."""

    def test_allows_approved_shell_command_sync(self) -> None:
        """Approved shell commands pass through in synchronous contexts."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {
            "name": "execute",
            "args": {"command": "ls -la"},
            "id": "tc-sync-1",
        }
        handler = Mock(return_value="output")

        result = middleware.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result == "output"

    def test_allows_non_shell_tools_sync(self) -> None:
        """Non-shell tools pass through unconditionally in synchronous contexts."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "write_file", "args": {}, "id": "tc-sync-ns"}
        handler = Mock(return_value="ok")

        result = middleware.wrap_tool_call(request, handler)
        handler.assert_called_once_with(request)
        assert result == "ok"

    async def test_allows_non_shell_tools(self) -> None:
        """Non-shell tools pass through unconditionally."""
        from unittest.mock import AsyncMock

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "write_file", "args": {}, "id": "tc1"}
        handler = AsyncMock(return_value="ok")

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result == "ok"

    @pytest.mark.parametrize(
        ("tool_name", "command"),
        [
            pytest.param("execute", "ls -la", id="execute-tool"),
            pytest.param("bash", "cat README.md", id="bash-tool"),
        ],
    )
    async def test_allows_approved_shell_command(
        self, tool_name: str, command: str
    ) -> None:
        """Shell commands in the allow-list pass through for all SHELL_TOOL_NAMES."""
        from unittest.mock import AsyncMock

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls", "cat"])
        request = Mock()
        request.tool_call = {
            "name": tool_name,
            "args": {"command": command},
            "id": "tc2",
        }
        handler = AsyncMock(return_value="output")

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result == "output"

    async def test_rejects_disallowed_shell_command(self) -> None:
        """Shell commands not in the allow-list get rejected as error ToolMessage."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import ToolMessage

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls", "cat"])
        request = Mock()
        request.tool_call = {
            "name": "execute",
            "args": {"command": "rm -rf /"},
            "id": "tc3",
        }
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "rejected" in result.content
        assert result.tool_call_id == "tc3"
        assert result.name == "execute"

    def test_rejects_disallowed_shell_command_sync(self) -> None:
        """Disallowed shell commands are rejected in synchronous contexts."""
        from langchain_core.messages import ToolMessage

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls", "cat"])
        request = Mock()
        request.tool_call = {
            "name": "execute",
            "args": {"command": "rm -rf /"},
            "id": "tc-sync-2",
        }
        handler = Mock()

        result = middleware.wrap_tool_call(request, handler)
        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "tc-sync-2"

    async def test_rejects_missing_command(self) -> None:
        """Shell tool call with no command arg is rejected, not an exception."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import ToolMessage

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "execute", "args": {}, "id": "tc4"}
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    async def test_rejects_empty_command_string(self) -> None:
        """Shell tool call with empty command string is rejected."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import ToolMessage

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "execute", "args": {"command": ""}, "id": "tc5"}
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    async def test_handles_none_args(self) -> None:
        """Shell tool call with args=None is rejected, not an exception."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import ToolMessage

        from deepagents_cli.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "execute", "args": None, "id": "tc6"}
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    def test_rejects_empty_allow_list(self) -> None:
        """Constructor rejects empty allow-list."""
        import pytest

        from deepagents_cli.agent import ShellAllowListMiddleware

        with pytest.raises(ValueError, match="must not be empty"):
            ShellAllowListMiddleware(allow_list=[])

    def test_rejects_shell_allow_all(self) -> None:
        """Constructor rejects SHELL_ALLOW_ALL sentinel."""
        import pytest

        from deepagents_cli.agent import ShellAllowListMiddleware
        from deepagents_cli.config import SHELL_ALLOW_ALL

        with pytest.raises(TypeError, match="SHELL_ALLOW_ALL"):
            ShellAllowListMiddleware(allow_list=SHELL_ALLOW_ALL)


class TestCreateCliAgentShellMiddlewareWiring:
    """Verify `create_cli_agent` wires `ShellAllowListMiddleware` correctly."""

    @staticmethod
    def _build_mock_settings(tmp_path: Path) -> Mock:
        """Create a settings mock suitable for `create_cli_agent` wiring tests."""
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
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = None
        mock_settings.project_root = None
        mock_settings.shell_allow_list = ["ls", "cat"]
        return mock_settings

    def test_interrupt_shell_only_adds_middleware_and_disables_interrupts(
        self, tmp_path: Path
    ) -> None:
        """Middleware is added and `interrupt_on={}` with interrupt_shell_only."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                interrupt_shell_only=True,
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        assert kwargs["interrupt_on"] == {}
        middleware_types = [type(m) for m in kwargs["middleware"]]
        assert ShellAllowListMiddleware in middleware_types

    def test_interrupt_shell_only_skipped_when_auto_approve(
        self, tmp_path: Path
    ) -> None:
        """When `auto_approve=True`, `interrupt_shell_only` has no effect."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                auto_approve=True,
                interrupt_shell_only=True,
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        assert kwargs["interrupt_on"] == {}
        middleware_types = [type(m) for m in kwargs["middleware"]]
        assert ShellAllowListMiddleware not in middleware_types

    def test_interrupt_shell_only_adds_middleware_to_subagents(
        self, tmp_path: Path
    ) -> None:
        """Restrictive shell mode must cover delegated subagents too."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_meta = {
            "name": "researcher",
            "description": "Researches things",
            "system_prompt": "Investigate the task thoroughly.",
            "model": None,
        }

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch(
                "deepagents_cli.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                interrupt_shell_only=True,
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents = kwargs["subagents"]
        assert subagents is not None

        subagents_by_name = {subagent["name"]: subagent for subagent in subagents}
        assert "researcher" in subagents_by_name
        assert "general-purpose" in subagents_by_name

        for name in ("researcher", "general-purpose"):
            middleware = subagents_by_name[name]["middleware"]
            assert any(isinstance(mw, ShellAllowListMiddleware) for mw in middleware), (
                f"Expected shell middleware on subagent {name!r}"
            )

    def test_no_duplicate_general_purpose_when_user_defined(
        self, tmp_path: Path
    ) -> None:
        """User-defined general-purpose subagent is not duplicated."""
        from deepagents_cli.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_meta = {
            "name": "general-purpose",
            "description": "User-defined general-purpose agent",
            "system_prompt": "You are helpful.",
            "model": None,
        }

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch(
                "deepagents_cli.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                interrupt_shell_only=True,
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents = kwargs["subagents"]
        gp_subagents = [s for s in subagents if s["name"] == "general-purpose"]
        assert len(gp_subagents) == 1, "Should not duplicate general-purpose subagent"
        assert any(
            isinstance(mw, ShellAllowListMiddleware)
            for mw in gp_subagents[0]["middleware"]
        )

    def test_shell_allow_all_skips_subagent_middleware(self, tmp_path: Path) -> None:
        """`SHELL_ALLOW_ALL` sentinel should not inject middleware on subagents."""
        from deepagents_cli.agent import ShellAllowListMiddleware
        from deepagents_cli.config import SHELL_ALLOW_ALL

        mock_settings = self._build_mock_settings(tmp_path)
        mock_settings.shell_allow_list = SHELL_ALLOW_ALL
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_meta = {
            "name": "researcher",
            "description": "Researches things",
            "system_prompt": "Investigate the task thoroughly.",
            "model": None,
        }

        with (
            patch("deepagents_cli.agent.settings", mock_settings),
            patch("deepagents_cli.agent.SkillsMiddleware"),
            patch("deepagents_cli.agent.MemoryMiddleware"),
            patch(
                "deepagents_cli.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_cli.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                interrupt_shell_only=True,
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents = kwargs["subagents"]
        for subagent in subagents:
            middleware = subagent.get("middleware", [])
            assert not any(
                isinstance(mw, ShellAllowListMiddleware) for mw in middleware
            ), f"Subagent {subagent['name']!r} should not have shell middleware"


def _mock_agents_dir(agents_dir: Path) -> Mock:
    mock_settings = Mock()
    mock_settings.user_deepagents_dir = agents_dir
    return mock_settings


class TestGetAvailableAgentNames:
    """Tests for `get_available_agent_names`."""

    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        """No ~/.deepagents directory → empty list, no error."""
        missing = tmp_path / "does_not_exist"
        with patch("deepagents_cli.agent.settings", _mock_agents_dir(missing)):
            assert get_available_agent_names() == []

    def test_returns_sorted_agent_names(self, tmp_path: Path) -> None:
        """Subdirectories are returned sorted."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        for name in ("zebra", "alpha", "mango"):
            (agents_dir / name).mkdir()

        with patch("deepagents_cli.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["alpha", "mango", "zebra"]

    def test_ignores_files_and_non_dirs(self, tmp_path: Path) -> None:
        """Files sitting next to agent directories are excluded."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent").mkdir()
        (agents_dir / "config.toml").write_text("")
        (agents_dir / ".DS_Store").write_text("")

        with patch("deepagents_cli.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["agent"]

    def test_ignores_symlinks(self, tmp_path: Path) -> None:
        """Symlinked directories are excluded — a dangling link must not show up."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "real").mkdir()
        # Symlink to a real dir — still excluded because we only want files
        # that live inside `~/.deepagents/` directly.
        real_target = tmp_path / "outside"
        real_target.mkdir()
        (agents_dir / "linked").symlink_to(real_target, target_is_directory=True)
        # Dangling symlink (target doesn't exist).
        (agents_dir / "broken").symlink_to(tmp_path / "ghost")

        with patch("deepagents_cli.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["real"]

    def test_permission_error_returns_empty(self, tmp_path: Path) -> None:
        """PermissionError on iterdir → logged + empty list, not raised."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        def boom(_self: Path) -> list[Path]:
            msg = "forbidden"
            raise PermissionError(msg)

        with (
            patch("deepagents_cli.agent.settings", _mock_agents_dir(agents_dir)),
            patch.object(Path, "iterdir", boom),
        ):
            assert get_available_agent_names() == []
