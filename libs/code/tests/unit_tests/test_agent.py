"""Unit tests for agent formatting functions."""

import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState
    from langchain.messages import ToolCall
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime

from deepagents_code._cli_context import CLIContext, CLIContextSchema
from deepagents_code.agent import (
    DEFAULT_AGENT_NAME,
    _add_interrupt_on,
    _apply_inherited_pythonpath,
    _create_rubric_grader_tools,
    _format_delete_description,
    _format_edit_file_description,
    _format_execute_description,
    _format_fetch_url_description,
    _format_task_description,
    _format_web_search_description,
    _format_write_file_description,
    _reserved_agent_dir_names,
    _sanitize_agent_message_name,
    _should_interrupt_tool_call,
    build_model_identity_section,
    create_cli_agent,
    get_available_agent_names,
    get_system_prompt,
    list_agents,
    load_async_subagents,
)
from deepagents_code.config import Settings, get_glyphs
from deepagents_code.managed_tools import BIN_DIR
from deepagents_code.project_utils import ProjectContext


@dataclass
class _StoreItem:
    value: dict[str, Any]


class _FakeStore:
    def __init__(self) -> None:
        self.items: dict[tuple[tuple[str, ...], str], _StoreItem] = {}

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: Mapping[str, Any],
    ) -> None:
        self.items[namespace, key] = _StoreItem(dict(value))

    def get(self, namespace: tuple[str, ...], key: str) -> _StoreItem | None:
        return self.items.get((namespace, key))


def _make_fake_chat_model() -> GenericFakeChatModel:
    """Create a fake chat model compatible with summarization middleware."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}
    return model


@contextmanager
def _ignore_interpreter_beta_warning() -> Iterator[None]:
    """Suppress the dependency's expected beta middleware warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The class `CodeInterpreterMiddleware` is in beta",
            category=Warning,
        )
        yield


def test_add_interrupt_on_gates_async_task_tools() -> None:
    """Async subagent tools should use their actual tool names in HITL config."""
    interrupt_on = _add_interrupt_on()

    for tool_name in ("start_async_task", "update_async_task", "cancel_async_task"):
        assert tool_name in interrupt_on


def test_add_interrupt_on_attaches_auto_approve_predicate() -> None:
    """Every gated tool carries the `when` predicate that honors auto-approve."""
    interrupt_on = _add_interrupt_on()

    assert interrupt_on
    for config in interrupt_on.values():
        assert config.get("when") is _should_interrupt_tool_call


def _request_with_context(
    context: object,
    *,
    store: object | None = None,
) -> "ToolCallRequest":
    return cast(
        "ToolCallRequest",
        SimpleNamespace(runtime=SimpleNamespace(context=context, store=store)),
    )


def test_should_interrupt_tool_call_respects_auto_approve_context() -> None:
    """The predicate suppresses interrupts once auto-approve is in run context."""
    # Dataclass-shaped context (in-process path).
    assert _should_interrupt_tool_call(
        _request_with_context(CLIContextSchema(auto_approve=False))
    )
    assert not _should_interrupt_tool_call(
        _request_with_context(CLIContextSchema(auto_approve=True))
    )
    # Dict-shaped context (LangGraph API / RemoteGraph path).
    assert _should_interrupt_tool_call(_request_with_context({"auto_approve": False}))
    assert not _should_interrupt_tool_call(
        _request_with_context({"auto_approve": True})
    )


def test_should_interrupt_tool_call_prefers_live_approval_mode() -> None:
    """A live manual toggle overrides an auto-approve run-context snapshot."""
    from deepagents_code.approval_mode import (
        APPROVAL_MODE_NAMESPACE,
        approval_mode_key,
        approval_mode_payload,
    )

    store = _FakeStore()
    key = approval_mode_key("thread-1")
    store.put(
        APPROVAL_MODE_NAMESPACE,
        key,
        approval_mode_payload(auto_approve=False),
    )
    assert _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": True, "approval_mode_key": key},
            store=store,
        )
    )
    assert _should_interrupt_tool_call(
        _request_with_context(
            CLIContextSchema(auto_approve=True, approval_mode_key=key),
            store=store,
        )
    )

    store.put(
        APPROVAL_MODE_NAMESPACE,
        key,
        approval_mode_payload(auto_approve=True),
    )
    assert not _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": False, "approval_mode_key": key},
            store=store,
        )
    )


async def test_live_approval_round_trip_flips_interrupt_decision() -> None:
    """A mode written via `awrite_approval_mode` is read back by the predicate.

    Exercises the full writer -> store -> reader contract across the shared
    `approval_mode_key` seam. The isolated write- and read-side tests would both
    stay green even if the two ever derived the key differently; only crossing
    the seam catches that — a key mismatch would surface here as an unexpected
    fail-closed interrupt.
    """
    from deepagents_code.approval_mode import approval_mode_key, awrite_approval_mode

    store = _FakeStore()

    class _StoreWriter:
        """Agent double whose store writer feeds the same store the reader uses."""

        async def aput_store_item(
            self,
            namespace: tuple[str, ...],
            key: str,
            value: Mapping[str, Any],
        ) -> None:
            store.put(namespace, key, value)

    agent = _StoreWriter()
    key = approval_mode_key("thread-1")

    written = await awrite_approval_mode(agent, "thread-1", auto_approve=True)
    assert written == key
    # Live auto-approve suppresses the interrupt even though the context
    # snapshot still says manual.
    assert not _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": False, "approval_mode_key": key},
            store=store,
        )
    )

    await awrite_approval_mode(agent, "thread-1", auto_approve=False)
    # Flipping the stored mode to manual interrupts despite an auto context.
    assert _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": True, "approval_mode_key": key},
            store=store,
        )
    )


def test_cli_context_schema_fields_mirror_typed_dict() -> None:
    """`CLIContextSchema` and `CLIContext` must stay structurally identical.

    The two shapes carry the same payload across the API boundary (dataclass
    in-process, dict over RemoteGraph). A field added to one but not the other
    would silently drop across that boundary; this pins the documented mirror.
    """
    from deepagents_code._cli_context import CLIContext

    assert {f.name for f in fields(CLIContextSchema)} == set(CLIContext.__annotations__)


def test_should_interrupt_tool_call_fails_closed_when_live_mode_missing() -> None:
    """A configured but missing live mode should interrupt for safety."""
    from deepagents_code.approval_mode import approval_mode_key

    assert _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": True, "approval_mode_key": approval_mode_key("thread-1")},
            store=_FakeStore(),
        )
    )


def test_should_interrupt_tool_call_fails_closed_without_live_mode_store() -> None:
    """A configured live-mode key with no runtime store should interrupt."""
    from deepagents_code.approval_mode import approval_mode_key

    assert _should_interrupt_tool_call(
        _request_with_context(
            {"auto_approve": True, "approval_mode_key": approval_mode_key("thread-1")}
        )
    )


def test_should_interrupt_tool_call_defaults_to_interrupting() -> None:
    """Missing or malformed context must not auto-approve."""
    assert _should_interrupt_tool_call(_request_with_context({}))
    assert _should_interrupt_tool_call(_request_with_context(None))
    assert _should_interrupt_tool_call(
        cast("ToolCallRequest", SimpleNamespace(runtime=None))
    )
    assert _should_interrupt_tool_call(cast("ToolCallRequest", SimpleNamespace()))


def test_should_interrupt_tool_call_truthy_non_bool_fails_closed() -> None:
    """A truthy non-bool context value must interrupt, not auto-approve.

    Context can arrive as a dataclass after in-process `context_schema`
    coercion, or as a dict from the JSON/RemoteGraph boundary. A malformed
    value in either shape must not slip past the gate on mere truthiness.
    """
    assert _should_interrupt_tool_call(
        _request_with_context(CLIContextSchema(auto_approve=cast("Any", 1)))
    )
    assert _should_interrupt_tool_call(
        _request_with_context(CLIContextSchema(auto_approve=cast("Any", "yes")))
    )
    assert _should_interrupt_tool_call(_request_with_context({"auto_approve": 1}))
    assert _should_interrupt_tool_call(_request_with_context({"auto_approve": "yes"}))


def test_should_interrupt_tool_call_warns_on_unexpected_context_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-None context that is neither shape interrupts and logs a warning.

    Reaching this branch means the `context_schema` coercion contract broke
    (likely an SDK change); fail closed but surface it so a silently-degraded
    auto-approve is observable instead of looking like a feature that "broke".
    """
    with caplog.at_level("WARNING", logger="deepagents_code.agent"):
        assert _should_interrupt_tool_call(_request_with_context("garbage"))
        assert _should_interrupt_tool_call(
            _request_with_context(SimpleNamespace(auto_approve=True))
        )

    assert "unexpected context type" in caplog.text
    # A legitimate absent context must stay silent — not every default is an anomaly.
    caplog.clear()
    with caplog.at_level("WARNING", logger="deepagents_code.agent"):
        assert _should_interrupt_tool_call(_request_with_context(None))
    assert "unexpected context type" not in caplog.text


def test_cli_context_field_parity() -> None:
    """`CLIContext` and `CLIContextSchema` must declare the same field set.

    The two types model the same run-context payload on opposite sides of the
    API boundary; the docstrings note "fields mirror" but nothing structural
    enforces it. This locks in parity so a field added to one is added to both.
    """
    typed_dict_keys = set(CLIContext.__annotations__)
    dataclass_keys = {f.name for f in fields(CLIContextSchema)}
    assert typed_dict_keys == dataclass_keys


def test_sanitize_agent_message_name_replaces_provider_unsafe_chars() -> None:
    """Agent display names with spaces must become valid message names."""
    assert _sanitize_agent_message_name("my agent") == "my_agent"
    assert _sanitize_agent_message_name("  my\tagent  ") == "my_agent"
    assert _sanitize_agent_message_name("my-agent_2") == "my-agent_2"
    assert _sanitize_agent_message_name("  ") == DEFAULT_AGENT_NAME


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


def test_format_delete_description() -> None:
    """Test delete description for approval prompts."""
    tool_call = cast(
        "ToolCall",
        {"name": "delete", "args": {"file_path": "/path/to/file.py"}, "id": "call-5"},
    )

    description = _format_delete_description(
        tool_call, cast("AgentState[Any]", None), cast("Runtime[Any]", None)
    )

    assert "Action: Delete file or directory" in description


def test_add_interrupt_on_gates_delete() -> None:
    """The destructive delete tool is approval-gated like other write tools."""
    interrupt_map = _add_interrupt_on()

    assert "delete" in interrupt_map
    assert interrupt_map["delete"]["allowed_decisions"] == ["approve", "reject"]
    assert interrupt_map["delete"]["description"] is _format_delete_description
    assert interrupt_map["delete"]["when"] is _should_interrupt_tool_call


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
        result = build_model_identity_section("gpt-5.5")
        assert "You are running as model `gpt-5.5`." in result
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
    """Model identity is rendered outside get_system_prompt.

    The section's content is covered by `TestBuildModelIdentitySection`; here we
    lock that (a) the base template no longer carries it and (b) `_make_prompt_for`
    assembles it back in for a given model.
    """

    def test_get_system_prompt_omits_model_identity(self) -> None:
        """Identity moved to a config `suffix`, so the base template omits it."""
        mock_settings = Mock()
        mock_settings.model_name = "claude-sonnet-4-6"
        mock_settings.model_provider = "anthropic"
        mock_settings.model_unsupported_modalities = frozenset()
        mock_settings.model_context_limit = 200000

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "### Model Identity" not in prompt
        assert "You are running as model" not in prompt

    def test_make_prompt_for_renders_base_and_identity(self) -> None:
        """`_make_prompt_for` assembles base -> identity (-> profile suffix)."""
        from langchain_anthropic import ChatAnthropic

        from deepagents_code.agent import _make_prompt_for
        from deepagents_code.config import ModelResult

        model = ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="x")
        model_result = ModelResult(
            model=model,
            model_name="claude-sonnet-4-6",
            provider="anthropic",
            context_limit=200_000,
        )
        prompt = _make_prompt_for("BASE PROMPT")(model_result)

        assert isinstance(prompt, str)
        assert prompt.startswith("BASE PROMPT")
        assert "You are running as model `claude-sonnet-4-6`" in prompt
        assert "(provider: anthropic)" in prompt
        assert "200,000 tokens" in prompt

    def test_make_prompt_for_honors_profile_base(self) -> None:
        """A target model whose profile sets a custom base uses it, not BASE."""
        from deepagents import HarnessProfile
        from deepagents.graph import BASE_AGENT_PROMPT
        from langchain_anthropic import ChatAnthropic

        from deepagents_code.agent import _make_prompt_for
        from deepagents_code.config import ModelResult

        model = ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="x")
        model_result = ModelResult(
            model=model, model_name="claude-sonnet-4-6", provider="anthropic"
        )
        profile = HarnessProfile(
            base_system_prompt="CUSTOM PROFILE BASE", system_prompt_suffix="PROFILE SFX"
        )
        with patch(
            "deepagents_code.agent._harness_profile_for_model", return_value=profile
        ):
            prompt = _make_prompt_for("CLI PROMPT")(model_result)

        assert isinstance(prompt, str)
        assert "CLI PROMPT" in prompt
        assert "CUSTOM PROFILE BASE" in prompt
        assert "PROFILE SFX" in prompt
        assert BASE_AGENT_PROMPT not in prompt


class TestGetSystemPromptNonInteractive:
    """Tests for interactive vs non-interactive system prompt."""

    def test_interactive_prompt_mentions_interactive_tui(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        assert "interactive TUI" in prompt
        assert "ask questions before acting" in prompt

    def test_non_interactive_prompt_mentions_headless(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "non-interactive" in prompt
        assert "no human" in prompt.lower()

    def test_non_interactive_prompt_does_not_ask_questions(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "ask questions before acting" not in prompt

    def test_non_interactive_prompt_instructs_autonomous_execution(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "Do NOT ask clarifying questions" in prompt
        assert "reasonable assumptions" in prompt

    def test_non_interactive_prompt_requires_non_interactive_commands(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=False)

        assert "non-interactive command variants" in prompt
        assert "npm init -y" in prompt

    def test_default_is_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "interactive TUI" in prompt

    def test_interactive_todo_section_asks_user_before_starting(self) -> None:
        """Interactive mode should require plan approval before first in_progress."""
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        assert "Wait for the user's response before marking the first todo" in prompt

    def test_non_interactive_todo_section_does_not_wait_for_user(self) -> None:
        """Headless mode must not contradict 'no human' guidance in todo rules."""
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.Path.cwd", side_effect=OSError("deleted")),
        ):
            prompt = get_system_prompt("test-agent")

        assert "Current Working Directory" in prompt


class TestGetSystemPromptSandbox:
    """Tests for sandbox-specific system prompt content."""

    def test_sandbox_includes_no_local_filesystem_warning(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="modal")

        assert "do NOT have access to the user's local filesystem" in prompt

    def test_sandbox_includes_working_dir_constraint(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="modal")

        assert "/workspace" in prompt
        assert "remote Linux sandbox" in prompt

    def test_sandbox_warns_about_subagent_paths(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", sandbox_type="daytona")

        assert "subagents" in prompt
        assert "/home/daytona" in prompt

    def test_local_mode_omits_sandbox_warnings(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent")

        assert "do NOT have access to the user's local filesystem" not in prompt
        assert "remote Linux sandbox" not in prompt


class TestGetSystemPromptPlaceholderValidation:
    """Tests for unreplaced placeholder detection."""

    def test_no_unreplaced_placeholders_in_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
            prompt = get_system_prompt("test-agent", interactive=True)

        # No raw {placeholder} patterns should remain
        import re

        assert not re.findall(r"\{[a-z_]+\}", prompt)

    def test_no_unreplaced_placeholders_in_non_interactive(self) -> None:
        mock_settings = Mock()
        mock_settings.model_name = None

        with patch("deepagents_code.agent.settings", mock_settings):
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent", return_value=mock_agent
            ) as mock_create_deep_agent,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            patch("deepagents_code.agent.get_system_prompt") as mock_get_prompt,
        ):
            mock_get_prompt.return_value = "mocked prompt"
            create_cli_agent(
                model="fake-model",
                assistant_id="my agent",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                interactive=False,
            )

        mock_get_prompt.assert_called_once()
        _, kwargs = mock_get_prompt.call_args
        assert kwargs["interactive"] is False
        assert mock_create_deep_agent.call_args.kwargs["name"] == "my_agent"
        assert (
            mock_create_deep_agent.call_args.kwargs["context_schema"]
            is CLIContextSchema
        )

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            patch("deepagents_code.agent.get_system_prompt") as mock_get_prompt,
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.console") as mock_console,
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.console") as mock_console,
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
            patch("deepagents_code.agent.settings", mock_settings),
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            list_agents(output_format="json")

        result = json.loads(buf.getvalue())
        assert result["data"] == []

    def test_json_output_excludes_state_dir(self, tmp_path: Path) -> None:
        """`.state/` is never surfaced as an agent in JSON output."""
        import json
        from io import StringIO

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / DEFAULT_AGENT_NAME).mkdir()
        (agents_dir / DEFAULT_AGENT_NAME / "AGENTS.md").touch()
        (agents_dir / ".state").mkdir()
        (agents_dir / ".state" / "sessions.db").touch()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        buf = StringIO()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            list_agents(output_format="json")

        result = json.loads(buf.getvalue())
        names = [a["name"] for a in result["data"]]
        assert names == [DEFAULT_AGENT_NAME]
        assert ".state" not in names

    def test_text_output_excludes_state_dir(self, tmp_path: Path) -> None:
        """`.state/` is never surfaced as an agent in Rich output."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / DEFAULT_AGENT_NAME).mkdir()
        (agents_dir / DEFAULT_AGENT_NAME / "AGENTS.md").touch()
        (agents_dir / ".state").mkdir()
        (agents_dir / ".state" / "sessions.db").touch()

        mock_settings = Mock()
        mock_settings.user_deepagents_dir = agents_dir

        output: list[str] = []

        def capture_print(*args: Any, **_: Any) -> None:
            output.append(" ".join(str(a) for a in args))

        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.console") as mock_console,
        ):
            mock_console.print = capture_print
            list_agents()

        joined = "\n".join(output)
        assert ".state" not in joined


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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("sys.stdout", buf),
        ):
            from deepagents_code.agent import reset_agent

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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
            backend=None,  # ty: ignore
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
            patch("deepagents_code.agent._SUPPORTS_SKILL_SOURCE_TUPLES", False),
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_code.agent.FilesystemBackend"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_code.agent.FilesystemBackend"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware", FakeSkillsMiddleware),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.list_subagents", return_value=[]) as mock_list,
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware", FakeMemoryMiddleware),
            patch("deepagents_code.agent.FilesystemBackend"),
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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

    @staticmethod
    def _build_shell_agent(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        user_langchain_project: str | None,
    ) -> tuple[Mock, Path]:
        """Build a shell-enabled CLI agent and return the `LocalShellBackend` mock.

        The agent's `deepagents-code` override is placed in `os.environ` so the
        returned `call_args` reflect how the user's original `LANGSMITH_PROJECT`
        is restored or dropped for shell commands.

        Returns:
            The `LocalShellBackend` mock (for `call_args` assertions) and the
            resolved user working directory.
        """
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
        mock_settings.user_langchain_project = user_langchain_project

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_backend = Mock()
        monkeypatch.setenv("LANGSMITH_PROJECT", "deepagents-code")

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch(
                "deepagents_code.agent.LocalShellBackend", return_value=mock_backend
            ) as mock_shell,
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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

        return mock_shell, user_cwd

    def test_project_context_sets_local_shell_root_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Shell backend root follows the cwd; agent override is dropped.

        With no user `LANGSMITH_PROJECT` (`user_langchain_project is None`),
        the agent's `deepagents-code` override must not leak into the shell
        env — it is popped so the user's code does not trace into the agent's
        project.
        """
        mock_shell, user_cwd = self._build_shell_agent(
            monkeypatch, tmp_path, user_langchain_project=None
        )

        assert mock_shell.call_args.kwargs["root_dir"] == user_cwd
        assert "LANGSMITH_PROJECT" not in mock_shell.call_args.kwargs["env"]

    @pytest.mark.parametrize("user_project", ["user-project", ""])
    def test_project_context_restores_user_shell_langchain_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, user_project: str
    ) -> None:
        """A non-None original project is restored into the shell env verbatim.

        The guard is `is not None`, so an empty-string original is restored as
        `""` (not popped) — the user explicitly cleared their project and that
        intent is preserved for shell commands.
        """
        mock_shell, _ = self._build_shell_agent(
            monkeypatch, tmp_path, user_langchain_project=user_project
        )

        assert mock_shell.call_args.kwargs["env"]["LANGSMITH_PROJECT"] == user_project

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.FilesystemBackend") as mock_filesystem,
            patch("deepagents_code.agent.create_deep_agent", return_value=mock_agent),
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
            patch("deepagents_code.agent.settings", mock_settings),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        from deepagents_code.ask_user import AskUserMiddleware

        middleware = self._capture_middleware(tmp_path, enable_ask_user=True)
        assert any(isinstance(mw, AskUserMiddleware) for mw in middleware)

    def test_ask_user_excluded_when_disabled(self, tmp_path: Path) -> None:
        from deepagents_code.ask_user import AskUserMiddleware

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
        from deepagents_code.agent import ShellAllowListMiddleware

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
        from deepagents_code.agent import ShellAllowListMiddleware

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

        from deepagents_code.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls"])
        request = Mock()
        request.tool_call = {"name": "write_file", "args": {}, "id": "tc1"}
        handler = AsyncMock(return_value="ok")

        result = await middleware.awrap_tool_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result == "ok"

    async def test_allows_approved_shell_command(self) -> None:
        """Shell commands in the allow-list pass through to the handler."""
        from unittest.mock import AsyncMock

        from deepagents_code.agent import ShellAllowListMiddleware

        middleware = ShellAllowListMiddleware(allow_list=["ls", "cat"])
        request = Mock()
        request.tool_call = {
            "name": "execute",
            "args": {"command": "ls -la"},
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

        from deepagents_code.agent import ShellAllowListMiddleware

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

        from deepagents_code.agent import ShellAllowListMiddleware

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

        from deepagents_code.agent import ShellAllowListMiddleware

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

        from deepagents_code.agent import ShellAllowListMiddleware

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

        from deepagents_code.agent import ShellAllowListMiddleware

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
        from deepagents_code.agent import ShellAllowListMiddleware

        with pytest.raises(ValueError, match="must not be empty"):
            ShellAllowListMiddleware(allow_list=[])

    def test_rejects_shell_allow_all(self) -> None:
        """Constructor rejects SHELL_ALLOW_ALL sentinel."""
        from deepagents_code.agent import ShellAllowListMiddleware
        from deepagents_code.config import SHELL_ALLOW_ALL

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
        from deepagents_code.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        from deepagents_code.agent import ShellAllowListMiddleware

        mock_settings = self._build_mock_settings(tmp_path)

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent

        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        from deepagents_code.agent import ShellAllowListMiddleware

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        from deepagents_code.agent import ShellAllowListMiddleware

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        from deepagents_code.agent import ShellAllowListMiddleware
        from deepagents_code.config import SHELL_ALLOW_ALL

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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

    def test_adds_configurable_model_middleware_to_implicit_model_subagents(
        self, tmp_path: Path
    ) -> None:
        """Runtime model switches should reach subagents without explicit models."""
        from deepagents_code.agent import ShellAllowListMiddleware
        from deepagents_code.configurable_model import ConfigurableModelMiddleware

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents = kwargs["subagents"]
        subagents_by_name = {subagent["name"]: subagent for subagent in subagents}
        assert "researcher" in subagents_by_name
        assert "general-purpose" in subagents_by_name

        for name in ("researcher", "general-purpose"):
            middleware = subagents_by_name[name]["middleware"]
            assert any(
                isinstance(mw, ConfigurableModelMiddleware) for mw in middleware
            ), f"Expected configurable model middleware on subagent {name!r}"
            # Without a restrictive allow-list, no shell middleware should be added
            # (the implicit `general-purpose` fallback must not be over-restricted).
            assert not any(
                isinstance(mw, ShellAllowListMiddleware) for mw in middleware
            ), f"Unexpected shell middleware on subagent {name!r}"

    def test_subagent_middleware_combines_shell_and_configurable_model(
        self, tmp_path: Path
    ) -> None:
        """Restrictive shell + implicit model should yield both middlewares.

        Explicitly pinned subagents keep shell restriction but must not gain
        `ConfigurableModelMiddleware`, which would let a runtime `/model` switch
        clobber the pinned model.
        """
        from deepagents_code.agent import ShellAllowListMiddleware
        from deepagents_code.configurable_model import ConfigurableModelMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_metas = [
            {
                "name": "researcher",
                "description": "Researches things",
                "system_prompt": "Investigate the task thoroughly.",
                "model": None,
            },
            {
                "name": "pinned",
                "description": "Runs on a fixed model",
                "system_prompt": "Stay on your assigned model.",
                "model": "anthropic:claude-haiku-4-5",
            },
        ]

        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=subagent_metas,
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
        subagents_by_name = {
            subagent["name"]: subagent for subagent in kwargs["subagents"]
        }

        # Implicit-model subagents (and the general-purpose fallback) get
        # configurable-model and shell middlewares, with the configurable-model
        # swap ordered before the shell gate so a runtime `/model` switch applies
        # before tools are filtered.
        for name in ("researcher", "general-purpose"):
            middleware_types = [
                type(mw) for mw in subagents_by_name[name]["middleware"]
            ]
            assert middleware_types == [
                ConfigurableModelMiddleware,
                ShellAllowListMiddleware,
            ], f"Unexpected middleware on subagent {name!r}: {middleware_types}"

        # The pinned subagent keeps shell restriction but is NOT given the
        # configurable-model middleware, so its model stays fixed.
        pinned = subagents_by_name["pinned"]
        assert pinned["model"] == "anthropic:claude-haiku-4-5"
        pinned_middleware = pinned["middleware"]
        assert any(
            isinstance(mw, ShellAllowListMiddleware) for mw in pinned_middleware
        ), "Pinned subagent should retain shell middleware"
        assert not any(
            isinstance(mw, ConfigurableModelMiddleware) for mw in pinned_middleware
        ), "Pinned subagent must not gain configurable model middleware"

    def test_subagents_get_managed_memory_guard_when_memory_enabled(
        self, tmp_path: Path
    ) -> None:
        """Subagents share the disk backend, so they get the managed-block guard."""
        from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=True,
                enable_skills=False,
                enable_shell=False,
            )

        _, kwargs = mock_create.call_args
        subagents_by_name = {
            subagent["name"]: subagent for subagent in kwargs["subagents"]
        }
        for name in ("researcher", "general-purpose"):
            middleware = subagents_by_name[name]["middleware"]
            assert any(
                isinstance(mw, ManagedMemoryGuardMiddleware) for mw in middleware
            ), f"Expected managed memory guard on subagent {name!r}"

    def test_subagents_skip_managed_memory_guard_when_memory_disabled(
        self, tmp_path: Path
    ) -> None:
        """With memory off there is no managed block, so no guard is added."""
        from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware

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
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        _, kwargs = mock_create.call_args
        for subagent in kwargs["subagents"]:
            assert not any(
                isinstance(mw, ManagedMemoryGuardMiddleware)
                for mw in subagent["middleware"]
            ), f"Subagent {subagent['name']!r} should not have the memory guard"

    def test_empty_string_subagent_model_treated_as_implicit(
        self, tmp_path: Path
    ) -> None:
        """An empty `model:` spec should inherit the runtime model, not pin `""`."""
        from deepagents_code.configurable_model import ConfigurableModelMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_meta = {
            "name": "researcher",
            "description": "Researches things",
            "system_prompt": "Investigate the task thoroughly.",
            "model": "",
        }

        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents_by_name = {
            subagent["name"]: subagent for subagent in kwargs["subagents"]
        }
        researcher = subagents_by_name["researcher"]
        assert "model" not in researcher, "Empty model spec must not be forwarded"
        assert any(
            isinstance(mw, ConfigurableModelMiddleware)
            for mw in researcher["middleware"]
        ), "Implicit-model subagent should receive configurable model middleware"

    def test_preserves_explicit_subagent_model_without_configurable_middleware(
        self, tmp_path: Path
    ) -> None:
        """Explicit subagent models should not be replaced by runtime switches."""
        from deepagents_code.configurable_model import ConfigurableModelMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()

        subagent_meta = {
            "name": "researcher",
            "description": "Researches things",
            "system_prompt": "Investigate the task thoroughly.",
            "model": "anthropic:claude-haiku-4-5",
        }

        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.list_subagents",
                return_value=[subagent_meta],
            ),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=True,
            )

        _, kwargs = mock_create.call_args
        subagents = kwargs["subagents"]
        subagents_by_name = {subagent["name"]: subagent for subagent in subagents}
        researcher = subagents_by_name["researcher"]
        assert researcher["model"] == "anthropic:claude-haiku-4-5"
        assert not any(
            isinstance(mw, ConfigurableModelMiddleware)
            for mw in researcher.get("middleware", [])
        )
        assert any(
            isinstance(mw, ConfigurableModelMiddleware)
            for mw in subagents_by_name["general-purpose"]["middleware"]
        )


def _mock_agents_dir(agents_dir: Path) -> Mock:
    mock_settings = Mock()
    mock_settings.user_deepagents_dir = agents_dir
    return mock_settings


class TestGetAvailableAgentNames:
    """Tests for `get_available_agent_names`."""

    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        """No ~/.deepagents directory → empty list, no error."""
        missing = tmp_path / "does_not_exist"
        with patch("deepagents_code.agent.settings", _mock_agents_dir(missing)):
            assert get_available_agent_names() == []

    def test_returns_sorted_agent_names(self, tmp_path: Path) -> None:
        """Subdirectories are returned sorted."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        for name in ("zebra", "alpha", "mango"):
            (agents_dir / name).mkdir()

        with patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["alpha", "mango", "zebra"]

    def test_ignores_files_and_non_dirs(self, tmp_path: Path) -> None:
        """Files sitting next to agent directories are excluded."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent").mkdir()
        (agents_dir / "config.toml").write_text("")
        (agents_dir / ".DS_Store").write_text("")

        with patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)):
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

        with patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["real"]

    def test_ignores_dot_prefixed_dirs(self, tmp_path: Path) -> None:
        """`.state/` and other hidden dirs are excluded from the agent list."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent").mkdir()
        (agents_dir / ".state").mkdir()
        (agents_dir / ".cache").mkdir()

        with patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["agent"]

    def test_ignores_reserved_bin_dir(self, tmp_path: Path) -> None:
        """The managed-binary install dir is excluded from the agent list.

        The reserved dir name is derived from `BIN_DIR.name` rather than the
        literal `bin` so a future rename of `BIN_DIR` keeps this test exercising
        the actual reserved name instead of a stale string.
        """
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "agent").mkdir()
        (agents_dir / BIN_DIR.name).mkdir()

        with patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)):
            assert get_available_agent_names() == ["agent"]

    def test_reserved_agent_dir_names_includes_bin_dir(self) -> None:
        """The reserved-name set is sourced from `BIN_DIR.name` (single source)."""
        assert _reserved_agent_dir_names() == frozenset({BIN_DIR.name})

    def test_permission_error_returns_empty(self, tmp_path: Path) -> None:
        """PermissionError on iterdir → logged + empty list, not raised."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        def boom(_self: Path) -> list[Path]:
            msg = "forbidden"
            raise PermissionError(msg)

        with (
            patch("deepagents_code.agent.settings", _mock_agents_dir(agents_dir)),
            patch.object(Path, "iterdir", boom),
        ):
            assert get_available_agent_names() == []


class TestCreateCliAgentInterpreterWiring:
    """Tests for `create_cli_agent` interpreter middleware wiring."""

    @staticmethod
    def _build_mock_settings(tmp_path: Path) -> Mock:
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
        mock_settings.shell_allow_list = None
        mock_settings.user_langchain_project = None
        mock_settings.interpreter_timeout_seconds = 5.0
        mock_settings.interpreter_memory_limit_mb = 64
        mock_settings.interpreter_max_ptc_calls = 256
        mock_settings.interpreter_max_result_chars = 4000
        mock_settings.interpreter_ptc = False
        mock_settings.interpreter_ptc_acknowledge_unsafe = False
        return mock_settings

    def test_appends_rubric_middleware(self, tmp_path: Path) -> None:
        from deepagents.middleware.rubric import RubricMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                rubric_model="custom-grader-model",
                rubric_max_iterations=5,
            )

        _, kwargs = mock_create.call_args
        rubrics = [
            mw for mw in kwargs["middleware"] if isinstance(mw, RubricMiddleware)
        ]
        assert len(rubrics) == 1
        assert rubrics[0]._model == "custom-grader-model"
        assert rubrics[0].max_iterations == 5
        assert "use the `read_file` tool" in rubrics[0]._system_prompt
        assert [tool.name for tool in rubrics[0]._tools] == ["read_file"]

    def test_omits_default_rubric_max_iterations(self, tmp_path: Path) -> None:
        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch("deepagents_code.agent.RubricMiddleware") as mock_rubric,
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
            )

        _, kwargs = mock_rubric.call_args
        assert "max_iterations" not in kwargs

    def test_rubric_grader_read_tool_only_reads_large_results(
        self, tmp_path: Path
    ) -> None:
        from deepagents.backends import CompositeBackend
        from deepagents.backends.filesystem import FilesystemBackend

        large_results = FilesystemBackend(
            root_dir=tmp_path / "large",
            virtual_mode=True,
        )
        project = FilesystemBackend(
            root_dir=tmp_path / "project",
            virtual_mode=False,
        )
        backend = CompositeBackend(
            default=project,
            routes={"/large_tool_results/": large_results},
        )
        backend.upload_files(
            [("/large_tool_results/tool-call-id", b"first\nsecond\nthird")]
        )
        read_tool = cast("Any", _create_rubric_grader_tools(backend)[0])

        runtime = SimpleNamespace(tool_call_id="grader-read")
        allowed = read_tool.func(
            file_path="/large_tool_results/tool-call-id",
            runtime=runtime,
            limit=2,
        )
        denied = read_tool.func(
            file_path="/Users/mason/.ssh/id_rsa",
            runtime=runtime,
        )

        assert "1\tfirst" in allowed.content
        assert "2\tsecond" in allowed.content
        assert "can only read" in denied

    def test_appends_interpreter_middleware_when_enabled(self, tmp_path: Path) -> None:
        from langchain_quickjs import CodeInterpreterMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            _ignore_interpreter_beta_warning(),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=True,
            )

        _, kwargs = mock_create.call_args
        middleware_types = [type(m) for m in kwargs["middleware"]]
        assert CodeInterpreterMiddleware in middleware_types

    def test_no_interpreter_middleware_when_disabled(self, tmp_path: Path) -> None:
        from langchain_quickjs import CodeInterpreterMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        fake_model = _make_fake_chat_model()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
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
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=False,
            )

        _, kwargs = mock_create.call_args
        middleware_types = [type(m) for m in kwargs["middleware"]]
        assert CodeInterpreterMiddleware not in middleware_types

    def test_raises_when_sandbox_present(self, tmp_path: Path) -> None:
        mock_settings = self._build_mock_settings(tmp_path)
        fake_model = _make_fake_chat_model()
        fake_sandbox = Mock()
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            pytest.raises(ValueError, match="remote sandbox"),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=True,
                sandbox=fake_sandbox,
            )

    def test_unknown_ptc_names_pass_through_to_middleware(self, tmp_path: Path) -> None:
        """Names absent from `tools` are forwarded, not rejected.

        The middleware matches `ptc` names against the live runtime registry and
        silently drops unmatched ones, so an unrecognized name (a typo, or a
        runtime-injected built-in) is passed through rather than raising at
        build time.
        """
        from langchain_core.tools import tool
        from langchain_quickjs import CodeInterpreterMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_settings.interpreter_ptc = ["nope", "grep"]
        fake_model = _make_fake_chat_model()

        @tool
        def grep(pattern: str) -> str:  # noqa: ARG001
            """Search for a pattern."""
            return ""

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            _ignore_interpreter_beta_warning(),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=True,
                tools=[grep],
            )

        _, kwargs = mock_create.call_args
        middlewares = [
            m for m in kwargs["middleware"] if isinstance(m, CodeInterpreterMiddleware)
        ]
        assert len(middlewares) == 1
        assert middlewares[0]._ptc == ["nope", "grep"]

    def test_raises_on_ptc_all_without_acknowledge(self, tmp_path: Path) -> None:
        from langchain_core.tools import tool

        mock_settings = self._build_mock_settings(tmp_path)
        mock_settings.interpreter_ptc = "all"
        mock_settings.interpreter_ptc_acknowledge_unsafe = False
        fake_model = _make_fake_chat_model()

        @tool
        def grep(pattern: str) -> str:  # noqa: ARG001
            """Search."""
            return ""

        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            pytest.raises(ValueError, match="acknowledge_unsafe"),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                auto_approve=False,
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=True,
                tools=[grep],
            )

    def test_safe_preset_includes_runtime_builtins(self, tmp_path: Path) -> None:
        """`'safe'` resolves to the full preset including SDK-injected built-ins.

        `glob` is not in the passed `tools`, but `create_deep_agent` injects it
        at runtime, so the `ptc` list handed to `CodeInterpreterMiddleware` must
        include all three preset members — not just the ones in `tools`. This is
        the regression guard for the server/non-interactive path, where the
        filesystem tools are never members of the `tools` sequence.
        """
        from langchain_core.tools import tool
        from langchain_quickjs import CodeInterpreterMiddleware

        mock_settings = self._build_mock_settings(tmp_path)
        mock_settings.interpreter_ptc = "safe"
        fake_model = _make_fake_chat_model()

        @tool
        def grep(pattern: str) -> str:  # noqa: ARG001
            """Search."""
            return ""

        @tool
        def read_file(path: str) -> str:  # noqa: ARG001
            """Read."""
            return ""

        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        with (
            patch("deepagents_code.agent.settings", mock_settings),
            patch("deepagents_code.agent.SkillsMiddleware"),
            patch("deepagents_code.agent.MemoryMiddleware"),
            patch(
                "deepagents_code.agent.create_deep_agent",
                return_value=mock_agent,
            ) as mock_create,
            patch(
                "deepagents._models.init_chat_model",
                return_value=fake_model,
            ),
            _ignore_interpreter_beta_warning(),
        ):
            create_cli_agent(
                model="fake-model",
                assistant_id="test",
                enable_memory=False,
                enable_skills=False,
                enable_shell=False,
                enable_interpreter=True,
                tools=[grep, read_file],
            )

        _, kwargs = mock_create.call_args
        middlewares = [
            m for m in kwargs["middleware"] if isinstance(m, CodeInterpreterMiddleware)
        ]
        assert len(middlewares) == 1
        # `glob` is absent from `tools` but is a runtime built-in, so the safe
        # preset resolves to all three members rather than dropping it.
        assert sorted(middlewares[0]._ptc) == ["glob", "grep", "read_file"]


class TestResolvePtcOption:
    """Direct tests for the `_resolve_ptc_option` helper."""

    @staticmethod
    def _tools() -> list:
        from langchain_core.tools import tool

        @tool
        def read_file(path: str) -> str:  # noqa: ARG001
            """Read."""
            return ""

        @tool
        def write_file(path: str, content: str) -> str:  # noqa: ARG001
            """Write."""
            return ""

        @tool
        def delete(path: str) -> str:  # noqa: ARG001
            """Delete."""
            return ""

        @tool
        def grep(pattern: str) -> str:  # noqa: ARG001
            """Search."""
            return ""

        return [read_file, write_file, delete, grep]

    def test_false_returns_none(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        assert (
            _resolve_ptc_option(
                False,
                tools=self._tools(),
                acknowledge_unsafe=False,
                auto_approve=False,
            )
            is None
        )

    def test_empty_list_returns_none(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        assert (
            _resolve_ptc_option(
                [],
                tools=self._tools(),
                acknowledge_unsafe=False,
                auto_approve=False,
            )
            is None
        )

    def test_safe_includes_builtin_preset_members(self) -> None:
        """`"safe"` resolves to the full preset even when members are SDK built-ins.

        `glob` is not in the passed `tools` here, but it is a Deep Agents
        built-in injected at runtime, so the resolved allowlist must still
        include it — the middleware bridges it against the live registry.
        """
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            "safe",
            tools=self._tools(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file"]

    def test_all_with_auto_approve_skips_ack_check(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            "all",
            tools=self._tools(),
            acknowledge_unsafe=False,
            auto_approve=True,
        )
        assert result is not None
        # `all` enumerates only the tools passed to `create_cli_agent`; SDK
        # runtime built-ins are injected later and are not enumerable here.
        assert sorted(result) == ["delete", "grep", "read_file", "write_file"]

    @staticmethod
    def _tools_with_task() -> list:
        from langchain_core.tools import tool

        @tool
        def read_file(path: str) -> str:  # noqa: ARG001
            """Read."""
            return ""

        @tool
        def glob(pattern: str) -> str:  # noqa: ARG001
            """Glob."""
            return ""

        @tool
        def grep(pattern: str) -> str:  # noqa: ARG001
            """Search."""
            return ""

        @tool
        def task(prompt: str) -> str:  # noqa: ARG001
            """Dispatch a subagent."""
            return ""

        return [read_file, glob, grep, task]

    def test_safe_in_list_expands_with_explicit_tool(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["safe", "task"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "task"]

    def test_safe_in_list_dedupes_preserving_order(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["grep", "safe", "task", "grep"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["grep", "glob", "read_file", "task"]

    def test_all_in_list_raises(self) -> None:
        from deepagents_code.agent import _resolve_ptc_option

        with pytest.raises(ValueError, match="cannot include 'all'"):
            _resolve_ptc_option(
                ["all", "task"],
                tools=self._tools_with_task(),
                acknowledge_unsafe=False,
                auto_approve=False,
            )

    def test_unknown_name_in_list_passes_through(self) -> None:
        """Unrecognized names are forwarded, not rejected.

        A name absent from `tools` may still match an SDK built-in injected at
        runtime (or be a genuine typo the middleware drops), so the resolver
        passes it through after expanding `"safe"`.
        """
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["safe", "nope"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "nope"]

    def test_safe_alone_in_list_equals_standalone(self) -> None:
        """`["safe"]` must resolve identically to the standalone `"safe"`."""
        from deepagents_code.agent import _resolve_ptc_option

        tools = self._tools_with_task()
        kwargs = {"acknowledge_unsafe": False, "auto_approve": False}
        as_list = _resolve_ptc_option(["safe"], tools=tools, **kwargs)
        standalone = _resolve_ptc_option("safe", tools=tools, **kwargs)
        assert as_list == standalone == ["glob", "grep", "read_file"]

    def test_safe_in_list_is_case_insensitive(self) -> None:
        """The `"safe"` sentinel is matched case-insensitively inside a list."""
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["SAFE", "task"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "task"]

    def test_safe_sentinel_is_whitespace_tolerant(self) -> None:
        """Whitespace around the `"safe"` sentinel is tolerated and expanded."""
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            [" safe ", "task"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "task"]

    def test_explicit_names_are_not_normalized(self) -> None:
        """Only the `"safe"`/`"all"` sentinels are normalized; names pass verbatim.

        The CLI and config layers strip whitespace before this layer, so a
        padded explicit name should never reach here in practice. If one does,
        it is forwarded verbatim (not trimmed) and the middleware resolves it
        against the runtime registry.
        """
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["safe", " task "],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", " task "]

    def test_resolves_builtins_absent_from_passed_tools(self) -> None:
        """Reproduce the server path: built-in names resolve without being in `tools`.

        In server/non-interactive mode `create_cli_agent` only receives custom
        tools (e.g. `fetch_url` + MCP); the filesystem and `task` tools are
        injected by `create_deep_agent` middleware. The PTC allowlist must
        still resolve `safe`/`task` against those runtime built-ins rather than
        raising "Unknown tool names".
        """
        from langchain_core.tools import tool

        from deepagents_code.agent import _resolve_ptc_option

        @tool
        def fetch_url(url: str) -> str:  # noqa: ARG001
            """Fetch a URL (a custom, non-built-in tool)."""
            return ""

        result = _resolve_ptc_option(
            ["safe", "task"],
            tools=[fetch_url],
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "task"]

    def test_duplicate_safe_tokens_dedupe(self) -> None:
        """Repeated `"safe"` tokens expand once; members are not duplicated."""
        from deepagents_code.agent import _resolve_ptc_option

        result = _resolve_ptc_option(
            ["safe", "safe", "task"],
            tools=self._tools_with_task(),
            acknowledge_unsafe=False,
            auto_approve=False,
        )
        assert result == ["glob", "grep", "read_file", "task"]

    def test_safe_excludes_hitl_gated_tools(self) -> None:
        """`"safe"` must never expose tools that are HITL-gated outside the REPL.

        Including network or subagent tools in the preset would silently
        bypass `_add_interrupt_on()` gating via PTC. Locking the contents
        of `INTERPRETER_PTC_SAFE_PRESET` against the live HITL map here is
        the forcing function for that invariant.
        """
        from deepagents_code.agent import _add_interrupt_on
        from deepagents_code.config import INTERPRETER_PTC_SAFE_PRESET

        gated = set(_add_interrupt_on().keys())
        overlap = INTERPRETER_PTC_SAFE_PRESET & gated
        assert not overlap, (
            f"INTERPRETER_PTC_SAFE_PRESET must not include HITL-gated tools; "
            f"found: {sorted(overlap)}"
        )

    def test_safe_preset_contents_are_locked(self) -> None:
        """Lock the literal contents of the `"safe"` preset.

        A reviewer flagged the original `"safe"` choice (network + subagent
        tools) as a silent HITL bypass. The current preset is intentionally
        restricted to non-gated, read-only file inspection; widening it
        without re-auditing the HITL surface should fail this test.
        """
        from deepagents_code.config import INTERPRETER_PTC_SAFE_PRESET

        assert frozenset({"read_file", "glob", "grep"}) == INTERPRETER_PTC_SAFE_PRESET


class TestApplyInheritedPythonpath:
    def test_relays_carrier_to_pythonpath(self) -> None:
        env = {"DEEPAGENTS_INHERITED_PYTHONPATH": "src", "PATH": "/usr/bin"}
        _apply_inherited_pythonpath(env)
        assert env["PYTHONPATH"] == "src"
        assert "DEEPAGENTS_INHERITED_PYTHONPATH" not in env

    def test_noop_without_carrier(self) -> None:
        env = {"PATH": "/usr/bin"}
        _apply_inherited_pythonpath(env)
        assert "PYTHONPATH" not in env
