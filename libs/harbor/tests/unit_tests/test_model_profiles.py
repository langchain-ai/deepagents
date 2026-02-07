"""Unit tests for Harbor model profile resolution."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from deepagents_harbor.deepagents_wrapper import (
    MODEL_PROFILES,
    _infer_model_profile_name,
    _parse_csv_list,
    _resolve_anthropic_betas,
    _resolve_model_profile,
)
from deepagents_harbor.middleware import (
    APIErrorRecoveryMiddleware,
    ContextBudgetMiddleware,
    LoopDetectionMiddleware,
    PreCompletionCheckMiddleware,
)


class FakeAPIError(RuntimeError):
    """Minimal exception shape used to emulate provider API failures."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def test_infer_profile_for_codex() -> None:
    """Infer OpenAI reasoning profile for Codex/GPT-5 models."""
    assert _infer_model_profile_name("openai:gpt-5.2-codex") == "openai_reasoning"


def test_infer_profile_for_opus() -> None:
    """Infer Anthropic Opus profile for Claude Opus models."""
    assert _infer_model_profile_name("anthropic:claude-opus-4-6") == "anthropic_opus"


def test_resolve_explicit_profile() -> None:
    """Resolve explicit profile names directly."""
    profile = _resolve_model_profile(
        model_name="openai:gpt-5.2-codex",
        profile_name="default",
    )
    assert profile == MODEL_PROFILES["default"]


def test_resolve_invalid_profile_raises() -> None:
    """Reject unknown profile names."""
    with pytest.raises(ValueError):
        _resolve_model_profile(
            model_name="openai:gpt-5.2-codex",
            profile_name="does-not-exist",
        )


def test_loop_detection_rejects_invalid_thresholds() -> None:
    """Validate middleware threshold guardrails."""
    with pytest.raises(ValueError):
        LoopDetectionMiddleware(soft_warning_threshold=0, hard_reflection_threshold=1)
    with pytest.raises(ValueError):
        LoopDetectionMiddleware(soft_warning_threshold=5, hard_reflection_threshold=5)


def test_context_budget_rejects_invalid_thresholds() -> None:
    """Validate context budget guardrails."""
    with pytest.raises(ValueError):
        ContextBudgetMiddleware(max_output_lines=0)
    with pytest.raises(ValueError):
        ContextBudgetMiddleware(warn_threshold_percent=0)
    with pytest.raises(ValueError):
        ContextBudgetMiddleware(warn_threshold_percent=101)


def test_openai_reasoning_profile_uses_400k_input_budget() -> None:
    """OpenAI reasoning profile should align with the 400k model context."""
    profile = MODEL_PROFILES["openai_reasoning"]
    assert profile.max_input_tokens == 400000
    assert profile.max_context_tokens == 300000
    assert profile.max_context_tokens < profile.max_input_tokens


def test_anthropic_opus_profile_uses_safe_default_budget() -> None:
    """Opus profile should default to a safe budget unless beta is explicitly enabled."""
    profile = MODEL_PROFILES["anthropic_opus"]
    assert profile.max_input_tokens == 200000
    assert profile.max_context_tokens < profile.max_input_tokens


def test_parse_csv_list() -> None:
    """Parse comma-separated values and drop empty entries."""
    assert _parse_csv_list(None) == []
    assert _parse_csv_list("") == []
    assert _parse_csv_list(" a, b ,, c ") == ["a", "b", "c"]


def test_resolve_anthropic_betas_defaults_for_opus46() -> None:
    """Opus 4.6 should default to the long-context beta when unspecified."""
    betas = _resolve_anthropic_betas("anthropic:claude-opus-4-6", None)
    assert betas == ["context-1m-2025-08-07"]


def test_resolve_anthropic_betas_respects_explicit_override() -> None:
    """Explicit betas should override defaults."""
    betas = _resolve_anthropic_betas(
        "anthropic:claude-opus-4-6",
        "foo-beta,bar-beta",
    )
    assert betas == ["foo-beta", "bar-beta"]


def test_resolve_anthropic_betas_allows_explicit_disable() -> None:
    """Passing an empty string should disable default beta injection."""
    betas = _resolve_anthropic_betas("anthropic:claude-opus-4-6", "")
    assert betas == []


@pytest.mark.asyncio
async def test_context_budget_handles_command_tool_messages() -> None:
    """Context budget middleware should process ToolMessages wrapped in Command updates."""
    middleware = ContextBudgetMiddleware(max_context_tokens=1000, max_output_lines=6)
    tool_message = ToolMessage(
        content="\n".join(f"line {i}" for i in range(40)),
        tool_call_id="tool-1",
    )
    command = Command(update={"messages": [tool_message]})

    async def handler(_request):
        return command

    result = await middleware.awrap_tool_call(SimpleNamespace(), handler)

    assert isinstance(result, Command)
    processed = result.update["messages"][0]
    assert isinstance(processed, ToolMessage)
    assert "OUTPUT TRUNCATED" in str(processed.content)
    middleware.after_model({"messages": [processed]}, SimpleNamespace())
    assert middleware._estimated_tokens > 0


@pytest.mark.asyncio
async def test_context_budget_counts_command_image_tool_messages() -> None:
    """Image tool results in Commands should count toward context estimates."""
    middleware = ContextBudgetMiddleware(max_context_tokens=1000, max_output_lines=6)
    tool_message = ToolMessage(
        content=[
            {
                "type": "image",
                "base64": "A" * 8000,
                "mime_type": "image/jpeg",
            }
        ],
        tool_call_id="tool-img",
    )
    command = Command(update={"messages": [tool_message]})

    async def handler(_request):
        return command

    result = await middleware.awrap_tool_call(SimpleNamespace(), handler)

    assert isinstance(result, Command)
    processed = result.update["messages"][0]
    assert isinstance(processed, ToolMessage)
    assert isinstance(processed.content, list)
    middleware.after_model({"messages": [processed]}, SimpleNamespace())
    assert middleware._estimated_tokens > 0


def test_precompletion_defers_when_ai_content_contains_tool_use_block() -> None:
    """Checklist injection must not break Anthropic tool_use/tool_result ordering."""
    middleware = PreCompletionCheckMiddleware()
    state = {
        "messages": [
            AIMessage(content=[{"type": "tool_use", "id": "tu_1", "name": "read_file"}])
        ]
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is None


def test_loop_detection_defers_reflection_for_tool_use_content_block() -> None:
    """Loop reflection should wait until pending tool results are returned."""
    middleware = LoopDetectionMiddleware(soft_warning_threshold=2, hard_reflection_threshold=3)
    middleware._pending_reflection = ("/app/main.py", 3)
    state = {
        "messages": [
            AIMessage(content=[{"type": "tool_use", "id": "tu_2", "name": "edit_file"}])
        ]
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is None
    assert middleware._pending_reflection == ("/app/main.py", 3)


def test_context_budget_defers_warning_for_tool_use_content_block() -> None:
    """Context warnings should not be inserted between tool_use and tool_result."""
    middleware = ContextBudgetMiddleware(max_context_tokens=10, warn_threshold_percent=10)
    state = {
        "messages": [
            AIMessage(content=[{"type": "tool_use", "id": "tu_3", "name": "execute"}])
        ]
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is None
    assert not middleware._warning_shown


def test_context_budget_recomputes_from_live_state() -> None:
    """Usage estimate should track current message state, not cumulative history."""
    middleware = ContextBudgetMiddleware(max_context_tokens=100000, warn_threshold_percent=90)

    middleware.after_model({"messages": [HumanMessage(content="A" * 4000)]}, SimpleNamespace())
    large_estimate = middleware._estimated_tokens

    middleware.after_model({"messages": [HumanMessage(content="small state only")]}, SimpleNamespace())
    small_estimate = middleware._estimated_tokens

    assert large_estimate > 0
    assert small_estimate > 0
    assert small_estimate < large_estimate


@pytest.mark.asyncio
async def test_api_error_recovery_bubbles_context_overflow() -> None:
    """Context overflow should bubble for orchestrator-level retry instead of looping."""
    middleware = APIErrorRecoveryMiddleware()

    async def handler(_request):
        raise FakeAPIError("input exceeds the context window", code="context_length_exceeded")

    with pytest.raises(FakeAPIError):
        await middleware.awrap_model_call(SimpleNamespace(), handler)

    # No synthetic continuation should be injected for non-recoverable overflow.
    assert middleware.after_model({"messages": [AIMessage(content="done")]}, SimpleNamespace()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "error_message", "expected_text"),
    [
        ("content_filter", "blocked by usage policy", "blocked by the model's content policy"),
        ("invalid_image_format", "invalid image payload", "image data you tried to send was invalid"),
    ],
)
async def test_api_error_recovery_keeps_recoverable_paths(
    error_code: str,
    error_message: str,
    expected_text: str,
) -> None:
    """Recoverable error classes should still produce guidance and continuation jump."""
    middleware = APIErrorRecoveryMiddleware()

    async def handler(_request):
        raise FakeAPIError(error_message, code=error_code)

    response = await middleware.awrap_model_call(SimpleNamespace(), handler)

    assert isinstance(response, AIMessage)
    assert expected_text in str(response.content).lower()

    follow_up = middleware.after_model({"messages": [response]}, SimpleNamespace())
    assert follow_up is not None
    assert follow_up["jump_to"] == "model"


def test_precompletion_enforces_test_run_after_checklist() -> None:
    """After checklist reminder, finishing without tests should be blocked once."""
    middleware = PreCompletionCheckMiddleware()
    state = {
        "checklist_reminder_shown": True,
        "activity_enforcement_shown": True,
        "messages": [
            HumanMessage(content="**Run Tests**: `pytest`"),
            AIMessage(content="Done."),
        ],
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is not None
    assert result["jump_to"] == "model"
    assert result["test_enforcement_shown"] is True
    assert "without running verifier/tests" in result["messages"][0].content


@pytest.mark.asyncio
async def test_precompletion_enforces_non_readonly_finish_after_checklist() -> None:
    """After checklist, finishing without execute/write activity should be blocked."""
    middleware = PreCompletionCheckMiddleware()
    middleware.before_agent({}, SimpleNamespace())
    state = {
        "checklist_reminder_shown": True,
        "messages": [
            HumanMessage(content="Environment context"),
            AIMessage(content="Done."),
        ],
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is not None
    assert result["jump_to"] == "model"
    assert result["activity_enforcement_shown"] is True
    assert "read-only exploration" in result["messages"][0].content


@pytest.mark.asyncio
async def test_precompletion_allows_finish_after_test_execution() -> None:
    """If tests were executed, no extra finish-blocking message should be injected."""
    middleware = PreCompletionCheckMiddleware()
    middleware.before_agent({}, SimpleNamespace())

    request = SimpleNamespace(
        tool_call={
            "name": "execute",
            "args": {"command": "pytest -q"},
        }
    )

    async def handler(_request):
        return ToolMessage(content="ok", tool_call_id="tool-verify")

    await middleware.awrap_tool_call(request, handler)

    state = {
        "checklist_reminder_shown": True,
        "messages": [
            HumanMessage(content="**Run Tests**: `pytest`"),
            AIMessage(content="Done."),
        ],
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is None


@pytest.mark.asyncio
async def test_precompletion_allows_finish_after_file_write_when_no_test_hint() -> None:
    """A concrete write action should satisfy non-read-only enforcement."""
    middleware = PreCompletionCheckMiddleware()
    middleware.before_agent({}, SimpleNamespace())

    request = SimpleNamespace(
        tool_call={
            "name": "write_file",
            "args": {"file_path": "/app/output.txt", "content": "ok"},
        }
    )

    async def handler(_request):
        return ToolMessage(content="written", tool_call_id="tool-write")

    await middleware.awrap_tool_call(request, handler)

    state = {
        "checklist_reminder_shown": True,
        "messages": [
            HumanMessage(content="No explicit test command here"),
            AIMessage(content="Done."),
        ],
    }

    result = middleware.after_model(state, SimpleNamespace())

    assert result is None
