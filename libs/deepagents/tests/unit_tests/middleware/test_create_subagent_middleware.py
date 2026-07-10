"""Unit tests for `create_subagent_middleware` and its default middleware stack."""

from langchain.agents.middleware import TodoListMiddleware

from deepagents.backends.state import StateBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    SubAgentMiddleware,
    _default_subagent_middleware,
    create_subagent_middleware,
)
from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def test_default_subagent_middleware_includes_summarization_for_chat_model() -> None:
    backend = StateBackend()
    model = GenericFakeChatModel(messages=iter([]))

    stack = _default_subagent_middleware(model, backend)

    assert isinstance(stack[0], TodoListMiddleware)
    assert isinstance(stack[1], FilesystemMiddleware)
    assert isinstance(stack[2], SummarizationMiddleware)
    assert isinstance(stack[-1], PatchToolCallsMiddleware)


def test_default_subagent_middleware_skips_summarization_for_string_model() -> None:
    stack = _default_subagent_middleware("anthropic:claude-sonnet-5", StateBackend())

    assert not any(isinstance(m, SummarizationMiddleware) for m in stack)
    assert isinstance(stack[-1], PatchToolCallsMiddleware)


def test_create_subagent_middleware_fills_in_defaults_for_user_subagents() -> None:
    backend = StateBackend()
    model = GenericFakeChatModel(messages=iter([]))
    gp_subagent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": [],
    }

    middleware = create_subagent_middleware(
        backend=backend,
        gp_subagent=gp_subagent,
        subagents=[
            {
                "name": "researcher",
                "description": "Research agent",
                "system_prompt": "You are a researcher.",
                "tools": [],
            },
            {"name": "async-agent", "description": "Remote agent", "graph_id": "remote"},
        ],
        task_description="Custom delegation guidance for {available_agents}.",
        system_prompt="Custom system prompt.",
    )

    assert isinstance(middleware, SubAgentMiddleware)
    assert middleware.subagent_names == {"general-purpose", "researcher"}
    general_purpose = next(s for s in middleware._subagents if s["name"] == "general-purpose")
    assert isinstance(general_purpose["middleware"][0], TodoListMiddleware)
    researcher = next(s for s in middleware._subagents if s["name"] == "researcher")
    assert researcher["model"] is model
    assert isinstance(researcher["middleware"][0], TodoListMiddleware)


def test_create_subagent_middleware_respects_explicit_overrides() -> None:
    backend = StateBackend()
    custom_middleware = [PatchToolCallsMiddleware()]
    default_model = GenericFakeChatModel(messages=iter([]))
    researcher_model = GenericFakeChatModel(messages=iter([]))
    gp_subagent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": default_model,
        "tools": [],
    }

    middleware = create_subagent_middleware(
        backend=backend,
        gp_subagent=gp_subagent,
        subagents=[
            {
                "name": "researcher",
                "description": "Research agent",
                "system_prompt": "You are a researcher.",
                "tools": [],
                "model": researcher_model,
                "middleware": custom_middleware,
            }
        ],
    )

    researcher = next(s for s in middleware._subagents if s["name"] == "researcher")
    assert researcher["model"] is researcher_model
    assert researcher["middleware"] is custom_middleware
