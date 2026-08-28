"""Tests for server-side goal-criteria drafting helpers."""

from __future__ import annotations

import ast
import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import FileInfo, LsResult
from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import (
    ApproveDecision,
    RejectDecision,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import ValidationError

from deepagents_code._repository_bounds import (
    REPOSITORY_DIRECTORY_ENTRY_LIMIT as _REPOSITORY_DIRECTORY_ENTRY_LIMIT,
    REPOSITORY_GLOB_MATCH_LIMIT as _REPOSITORY_GLOB_MATCH_LIMIT,
    REPOSITORY_READ_BYTE_LIMIT as _REPOSITORY_READ_BYTE_LIMIT,
    REPOSITORY_READ_LINE_LIMIT as _REPOSITORY_READ_LINE_LIMIT,
    REPOSITORY_TOOL_RESULT_LIMIT as _REPOSITORY_TOOL_RESULT_LIMIT,
)
from deepagents_code._testing_models import (
    GoalCriteriaIntegrationChatModel,
    _tool_call_result,
)
from deepagents_code.goal_rubric import (
    _CONVERSATION_CONTEXT_MESSAGE_LIMIT,
    _CONVERSATION_CONTEXT_SERIALIZED_LIMIT,
    _CRITERIA_CONTEXT_TOTAL_TEXT_LIMIT,
    _CRITERIA_OBJECTIVE_DISPLAY_LIMIT,
    _CRITERIA_RESULT_LOG_LIMIT,
    _REPOSITORY_GREP_MATCH_LIMIT,
    _REPOSITORY_OPERATION_BUDGET_CACHE_LIMIT,
    _REPOSITORY_TOOL_CALL_LIMIT,
    _WEB_SEARCH_CALL_LIMIT,
    GOAL_RUBRIC_SYSTEM_PROMPT,
    GoalCriteriaAgentState,
    GoalCriteriaMiddleware,
    GoalCriteriaRequest,
    GoalCriteriaState,
    GoalProposal,
    _coerce_goal_proposal,
    _ContextToolCallBudgetMiddleware,
    _conversation_context,
    _create_goal_criteria_agent,
    _criteria_interrupt_on,
    _CriteriaContextBudgetMiddleware,
    _goal_amendment_human_prompt,
    _goal_criteria_request,
    _goal_proposal_from_text,
    _goal_rubric_human_prompt,
    _GoalContextFallbackMiddleware,
    _prompt_with_conversation_context,
    _proposal_from_result,
    _raise_terminal_goal_state_size_error,
    _RepositoryToolBudgetMiddleware,
    _rubric_interrupt_on,
    _summarize_criteria_result,
    _WebSearchBudgetMiddleware,
    create_goal_criteria_agent,
    create_goal_criteria_fallback_agent,
)
from deepagents_code.goal_state_limits import (
    GOAL_APPLICATION_CHAR_LIMIT,
    GOAL_OBJECTIVE_CHAR_LIMIT,
    RUBRIC_CHAR_LIMIT,
    GoalStateSizeError,
)
from deepagents_code.goal_tools import GoalToolState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.outputs import ChatResult
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents_code.agent import AsyncApprovalHITLMiddleware


class _OversizedThenValidCriteriaModel(GoalCriteriaIntegrationChatModel):
    """Return invalid long criteria once, then honor structured-output feedback."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Retry with concise criteria after receiving the validation error.

        Returns:
            An oversized proposal first, then a valid proposal.
        """
        saw_error = any(isinstance(message, ToolMessage) for message in messages)
        criteria = "- concise result" if saw_error else "x" * (RUBRIC_CHAR_LIMIT + 1)
        call_id = "valid-proposal" if saw_error else "oversized-proposal"
        return _tool_call_result(
            "GoalProposal",
            {"objective": "ship it", "criteria": criteria},
            call_id,
        )


class _LoopBoundAsyncStore:
    """Async server Store whose sync API is invalid on the event loop."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.aget_calls = 0
        self.get_calls = 0

    async def aget(self, namespace: tuple[str, ...], key: str) -> object:
        from deepagents_code.approval_mode import APPROVAL_MODE_NAMESPACE

        assert namespace == APPROVAL_MODE_NAMESPACE
        assert key
        self.aget_calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(value=self.value)

    def get(self, namespace: tuple[str, ...], key: str) -> object:
        _ = (namespace, key)
        self.get_calls += 1
        msg = "synchronous Store access is forbidden on the event loop"
        raise asyncio.InvalidStateError(msg)


class TestGoalPrompts:
    """Prompt construction preserves user input and fallback guidance."""

    def test_objective_only(self) -> None:
        prompt = _goal_rubric_human_prompt("add OAuth refresh")

        assert "<operation>draft</operation>" in prompt
        assert "<goal>\nadd OAuth refresh\n</goal>" in prompt
        assert "<user_feedback>" not in prompt

    def test_rejection_feedback_includes_previous_criteria(self) -> None:
        prompt = _goal_rubric_human_prompt(
            "add OAuth refresh",
            feedback="be stricter",
            previous_criteria="- old criterion",
        )

        assert "Regenerate" in prompt
        assert "<previous_criteria>\n- old criterion\n</previous_criteria>" in prompt
        assert "<user_feedback>\nbe stricter\n</user_feedback>" in prompt

    def test_amendment_contains_current_state_and_feedback(self) -> None:
        prompt = _goal_amendment_human_prompt(
            "ship login",
            "- password login works",
            "add passkeys",
        )

        assert "<operation>amend</operation>" in prompt
        assert "<current_goal>\nship login\n</current_goal>" in prompt
        assert (
            "<current_criteria>\n- password login works\n</current_criteria>" in prompt
        )
        assert "<user_feedback>\nadd passkeys\n</user_feedback>" in prompt

    def test_system_prompt_resolves_underspecified_objectives(self) -> None:
        """A deictic objective is resolved from context, never restated."""
        normalized = " ".join(GOAL_RUBRIC_SYSTEM_PROMPT.split())

        assert (
            "Resolving what the objective refers to is not inventing requirements"
            in normalized
        )
        assert 'a bare "do it", "fix it"' in normalized
        assert "naming the files, commands, behavior, or deliverables" in normalized
        assert "the requested work is completed as specified" in normalized
        assert "is never acceptable" in normalized


class TestConversationContext:
    """Parent context is recent, text-only, bounded, and safely serialized."""

    @staticmethod
    def _request() -> GoalCriteriaRequest:
        return {
            "request_id": "context-request",
            "kind": "create",
            "objective": "ship the explicit goal",
        }

    def test_internal_messages_blocks_calls_and_media_are_excluded(self) -> None:
        context = _conversation_context(
            [
                SystemMessage(content="SYSTEM_SECRET"),
                FunctionMessage(content="FUNCTION_SECRET", name="internal"),
                ToolMessage(content="TOOL_SECRET", tool_call_id="tool"),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "visible human text"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/MEDIA_SECRET"},
                        },
                    ]
                ),
                AIMessage(
                    content=[
                        {"type": "text", "text": "visible assistant text"},
                        {"type": "reasoning", "reasoning": "REASONING_SECRET"},
                        {"type": "image", "url": "https://example.com/IMAGE_SECRET"},
                    ],
                    tool_calls=[
                        {
                            "name": "internal_search",
                            "args": {"query": "TOOL_ARGUMENT_SECRET"},
                            "id": "call",
                            "type": "tool_call",
                        }
                    ],
                    additional_kwargs={"private": "METADATA_SECRET"},
                ),
                HumanMessage(content="   "),
            ]
        )

        assert "visible human text" in context
        assert "visible assistant text" in context
        for secret in (
            "SYSTEM_SECRET",
            "FUNCTION_SECRET",
            "TOOL_SECRET",
            "MEDIA_SECRET",
            "REASONING_SECRET",
            "IMAGE_SECRET",
            "internal_search",
            "TOOL_ARGUMENT_SECRET",
            "METADATA_SECRET",
        ):
            assert secret not in context

    def test_control_messages_are_excluded_but_summary_is_retained(self) -> None:
        context = _conversation_context(
            [
                HumanMessage(content="visible user text"),
                HumanMessage(
                    content="STATE_SECRET",
                    additional_kwargs={"lc_source": "goal_state"},
                ),
                HumanMessage(
                    content="CONTINUATION_SECRET",
                    additional_kwargs={"lc_source": "goal_control"},
                ),
                HumanMessage(
                    content="visible summary",
                    additional_kwargs={"lc_source": "summarization"},
                ),
                HumanMessage(content="[SYSTEM] Goal set by the user. LEGACY_SECRET"),
                AIMessage(content="visible assistant text"),
            ]
        )

        assert "visible user text" in context
        assert "visible summary" in context
        assert "visible assistant text" in context
        assert "STATE_SECRET" not in context
        assert "CONTINUATION_SECRET" not in context
        assert "LEGACY_SECRET" not in context

    def test_context_resolves_referents_without_adding_requirements(self) -> None:
        """Context may disambiguate the objective but never extend its scope."""
        prompt = _prompt_with_conversation_context(
            self._request(),
            [HumanMessage(content="rewrite the optimizer starter prompt")],
        )

        assert "resolve what an underspecified objective refers to" in prompt
        assert "Do not treat the context as a source of additional requirements" in (
            prompt
        )
        assert "do not infer additional requirements" not in prompt


class TestGoalContextFallbackMiddleware:
    """Context failures retry within the same criteria graph operation."""


class TestCriteriaContextBudgetMiddleware:
    """All gathered tool context shares one bounded operation budget."""


class TestContextToolCallBudgetMiddleware:
    """Rubric verification tools share an atomic per-evaluation call budget."""

    @staticmethod
    def _request(call_id: str, operation_id: str) -> ToolCallRequest:
        return ToolCallRequest(
            tool_call={
                "name": "notion_fetch",
                "args": {"page_id": "page"},
                "id": call_id,
                "type": "tool_call",
            },
            tool=None,
            state={"rubric_grading_operation_id": operation_id},
            runtime=MagicMock(),
        )


class TestWebSearchBudgetMiddleware:
    """Repeated searches are bounded per criteria operation."""

    @staticmethod
    def _request(
        call_id: str, operation_id: str = "search-operation"
    ) -> ToolCallRequest:
        return TestRepositoryToolBudgetMiddleware._request(
            call_id=call_id,
            name="web_search",
            operation_id=operation_id,
        )


class TestRepositoryToolBudgetMiddleware:
    """Repository reads remain server-backed, read-only, and bounded."""

    @staticmethod
    def _backend(*, size: int = 10) -> MagicMock:
        backend = MagicMock()
        backend.ls.return_value = LsResult(
            entries=[{"path": "/src.py", "is_dir": False, "size": size}]
        )
        return backend

    @staticmethod
    def _request(
        *,
        call_id: str,
        name: str = "read_file",
        limit: object = 999,
        path: str | None = "/src.py",
        operation_id: str = "operation-1",
        max_count: object = 999,
        search_glob: object = None,
    ) -> ToolCallRequest:
        key = "file_path" if name == "read_file" else "path"
        args = {key: path}
        if name == "read_file":
            args["limit"] = limit
        elif name == "glob":
            args["pattern"] = "**/*.py"
        elif name == "grep":
            args.update({"pattern": "needle", "max_count": max_count})
            if search_glob is not None:
                args["glob"] = search_glob
        runtime = MagicMock()
        return ToolCallRequest(
            tool_call={
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            },
            tool=None,
            state={"criteria_operation_id": operation_id},
            runtime=runtime,
        )


class TestCriteriaHitlPolicy:
    """External context tools keep normal HITL predicates and criteria context."""

    @staticmethod
    def _tool(name: str, description: str) -> StructuredTool:
        def invoke(query: str) -> str:
            return query

        return StructuredTool.from_function(
            func=invoke,
            name=name,
            description=description,
        )

    def test_fetch_and_mcp_tools_are_individually_gated(self) -> None:
        fetch = self._tool("fetch_url", "Fetch the requested URL.")
        mcp = self._tool("docs_search", "Search the documentation server.")

        policy = _criteria_interrupt_on([fetch, mcp])

        assert set(policy) == {"fetch_url", "docs_search"}
        assert policy["fetch_url"]["when"] is policy["docs_search"]["when"]
        assert policy["docs_search"]["allowed_decisions"] == ["approve", "reject"]

    def test_manual_prompt_prefix_is_bounded_and_preserves_details(self) -> None:
        tool = self._tool("docs_search", "Search the documentation server.")
        policy = _criteria_interrupt_on([tool])
        description = policy["docs_search"]["description"]
        assert callable(description)
        render_description = cast("Callable[..., str]", description)
        objective = "word " * _CRITERIA_OBJECTIVE_DISPLAY_LIMIT
        runtime = SimpleNamespace(context={})

        rendered = render_description(
            {"name": "docs_search", "args": {}, "id": "call"},
            {"criteria_objective": objective},
            runtime,
        )

        assert rendered.startswith(
            "Deep Agents Code wants to use docs_search while gathering context "
            "to propose acceptance criteria for: \u201c"
        )
        assert "Search the documentation server." in rendered
        displayed = rendered.split("\u201c", 1)[1].split("\u201d", 1)[0]
        assert len(displayed) <= _CRITERIA_OBJECTIVE_DISPLAY_LIMIT

    def test_rubric_context_tool_uses_grading_approval_description(self) -> None:
        tool = self._tool("notion_fetch", "Read the current Notion page.")
        policy = _rubric_interrupt_on([tool])
        description = cast("Callable[..., str]", policy["notion_fetch"]["description"])

        rendered = description(
            {"name": "notion_fetch", "args": {}, "id": "call"},
            {},
            SimpleNamespace(context={}),
        )

        assert "while verifying the completed work" in rendered
        assert "Read the current Notion page." in rendered


_RUBRIC_PROPOSAL_RESULT = {
    "structured_response": {
        "objective": "ship login with passkeys",
        "criteria": "- passkeys work",
    }
}


class _RubricProbeMiddleware(AgentMiddleware[GoalToolState, Any]):
    """Record the rubric visible to end-of-turn middleware.

    Declares the production `GoalToolState` schema so the public `rubric`
    channel is registered exactly as in the real app (via
    `GoalToolsMiddleware`), rather than a test-only redeclaration that could
    silently drift from production.
    """

    state_schema = GoalToolState

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []

    def after_agent(
        self,
        state: GoalToolState,
        runtime: Runtime[Any],
    ) -> None:
        _ = runtime
        # Record the raw value rather than coercing to None, so a regression
        # that swapped the cleared sentinel for a non-str would be caught here
        # instead of masked.
        self.seen.append(state.get("rubric"))


class TestGoalCriteriaMiddleware:
    """The main graph owns criteria execution and pending-state persistence."""

    @staticmethod
    def _runtime() -> Runtime[Any]:
        return cast(
            "Runtime[Any]",
            SimpleNamespace(context={"model": "openai:gpt-5.5"}),
        )

    @staticmethod
    def _assert_rubric_suppressed(
        probe: _RubricProbeMiddleware,
        state_values: dict[str, Any],
        *,
        kind: str,
        expected_objective: str,
    ) -> None:
        """The stale rubric is cleared before exit hooks and the proposal persists."""
        assert probe.seen == [None]
        assert state_values["rubric"] is None
        assert state_values["_pending_goal_objective"] == expected_objective
        assert state_values["_pending_goal_rubric"] == "- passkeys work"
        assert state_values["_pending_goal_kind"] == kind

    async def test_amendment_request_is_built_and_persisted_server_side(self) -> None:
        criteria = MagicMock()
        criteria.ainvoke = AsyncMock(
            return_value={
                "structured_response": {
                    "objective": "ship login with passkeys",
                    "criteria": "- passkeys work",
                }
            }
        )
        middleware = GoalCriteriaMiddleware(criteria)

        update = await middleware.abefore_agent(
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-2",
                    "kind": "amend",
                    "objective": "ship login",
                    "criteria": "- passwords work",
                    "feedback": "add passkeys",
                },
            },
            self._runtime(),
        )

        assert update is not None
        assert update["_pending_goal_objective"] == "ship login with passkeys"
        assert update["_pending_goal_rubric"] == "- passkeys work"
        assert update["_pending_goal_kind"] == "amend"
        assert update["_pending_goal_request_id"] == "request-2"
        awaited = criteria.ainvoke.await_args
        assert awaited is not None
        prompt = awaited.args[0]["messages"][0]["content"]
        assert "<current_goal>\nship login\n</current_goal>" in prompt
        assert "<user_feedback>\nadd passkeys\n</user_feedback>" in prompt

    @pytest.mark.parametrize(
        ("kind", "request_extra", "expected_objective"),
        [
            # create keeps the caller's objective; amend adopts the model's.
            ("create", {}, "ship login"),
            (
                "amend",
                {"criteria": "- passwords work", "feedback": "add passkeys"},
                "ship login with passkeys",
            ),
        ],
    )
    async def test_proposal_suppresses_persisted_rubric_before_exit_hooks(
        self,
        kind: str,
        request_extra: dict[str, str],
        expected_objective: str,
    ) -> None:
        criteria = MagicMock()
        criteria.ainvoke = AsyncMock(return_value=_RUBRIC_PROPOSAL_RESULT)
        probe = _RubricProbeMiddleware()
        middleware: list[AgentMiddleware[Any, Any]] = [
            GoalCriteriaMiddleware(criteria),
            probe,
        ]
        parent = create_agent(
            model=GoalCriteriaIntegrationChatModel(),
            tools=[],
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": f"criteria-{kind}-no-grade"}
        }
        await parent.aupdate_state(
            config,
            {
                "messages": [AIMessage(content="Interrupted work")],
                "rubric": "- passwords work",
            },
        )

        await parent.ainvoke(
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-proposal",
                    "kind": kind,
                    "objective": "ship login",
                    **request_extra,
                },
            },
            config=config,
            context={},
        )

        state = await parent.aget_state(config)
        self._assert_rubric_suppressed(
            probe, state.values, kind=kind, expected_objective=expected_objective
        )

    def test_proposal_suppresses_persisted_rubric_before_exit_hooks_sync(
        self,
    ) -> None:
        # Mirror the async path over the synchronous before_agent → invoke driver.
        criteria = MagicMock()
        criteria.invoke.return_value = _RUBRIC_PROPOSAL_RESULT
        probe = _RubricProbeMiddleware()
        middleware: list[AgentMiddleware[Any, Any]] = [
            GoalCriteriaMiddleware(criteria),
            probe,
        ]
        parent = create_agent(
            model=GoalCriteriaIntegrationChatModel(),
            tools=[],
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": "criteria-amend-no-grade-sync"}
        }
        parent.update_state(
            config,
            {
                "messages": [AIMessage(content="Interrupted work")],
                "rubric": "- passwords work",
            },
        )

        parent.invoke(
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-proposal-sync",
                    "kind": "amend",
                    "objective": "ship login",
                    "criteria": "- passwords work",
                    "feedback": "add passkeys",
                },
            },
            config=config,
            context={},
        )

        state = parent.get_state(config)
        self._assert_rubric_suppressed(
            probe,
            state.values,
            kind="amend",
            expected_objective="ship login with passkeys",
        )

    async def test_nested_hitl_resumes_through_parent_graph(self) -> None:
        def read_file(file_path: str, limit: int = 20) -> str:
            return f"{file_path}:{limit}"

        context_tool = StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read server context.",
        )
        model = GoalCriteriaIntegrationChatModel()
        criteria = create_goal_criteria_agent(
            model=model,
            repository_backend=None,
            context_tools=[context_tool],
        )
        parent = create_agent(
            model=model,
            tools=[],
            middleware=[GoalCriteriaMiddleware(criteria)],
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "criteria-hitl"}}
        request = {
            "messages": [],
            "goal_criteria_request": {
                "request_id": "request-hitl",
                "kind": "create",
                "objective": "verify server-side criteria generation",
                "feedback": "DCA_TEST_GOAL_CRITERIA=/context.txt",
            },
        }

        first = await parent.ainvoke(request, config=config, context={})

        interrupts = first["__interrupt__"]
        assert len(interrupts) == 1
        interrupt = interrupts[0]
        resumed = await parent.ainvoke(
            Command(
                resume={interrupt.id: {"decisions": [ApproveDecision(type="approve")]}}
            ),
            config=config,
            context={},
        )

        assert resumed["messages"] == []
        state = await parent.aget_state(config)
        assert state.values["_pending_goal_objective"] == (
            "verify server-side criteria generation"
        )
        assert state.values["_pending_goal_rubric"] == (
            "- server repository context is available"
        )

    async def test_nested_hitl_reject_still_finishes_with_a_proposal(self) -> None:
        """Rejecting a context tool skips it; the nested agent still proposes."""

        def read_file(file_path: str, limit: int = 20) -> str:
            return f"{file_path}:{limit}"

        context_tool = StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read server context.",
        )
        model = GoalCriteriaIntegrationChatModel()
        criteria = create_goal_criteria_agent(
            model=model,
            repository_backend=None,
            context_tools=[context_tool],
        )
        parent = create_agent(
            model=model,
            tools=[],
            middleware=[GoalCriteriaMiddleware(criteria)],
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": "criteria-reject"}}
        request = {
            "messages": [],
            "goal_criteria_request": {
                "request_id": "request-reject",
                "kind": "create",
                "objective": "verify server-side criteria generation",
                "feedback": "DCA_TEST_GOAL_CRITERIA=/context.txt",
            },
        }

        first = await parent.ainvoke(request, config=config, context={})
        interrupt = first["__interrupt__"][0]

        resumed = await parent.ainvoke(
            Command(
                resume={interrupt.id: {"decisions": [RejectDecision(type="reject")]}}
            ),
            config=config,
            context={},
        )

        # A bare reject skips the tool rather than aborting, so the nested agent
        # completes and the parent still persists a proposal.
        assert resumed["messages"] == []
        state = await parent.aget_state(config)
        assert state.values["_pending_goal_rubric"] == (
            "- server repository context is available"
        )
        assert state.values["goal_criteria_request"] is None


class TestCreateGoalCriteriaAgent:
    """The criteria graph is dedicated and uses server-provided resources."""

    def test_parent_allowlist_restricts_repository_tools(self) -> None:
        """Nested criteria generation cannot bypass the parent fs allowlist."""
        backend = MagicMock()
        filesystem = MagicMock()
        graph = MagicMock()
        graph.with_config.return_value = graph

        with (
            patch(
                "deepagents.middleware.FilesystemMiddleware",
                return_value=filesystem,
            ) as filesystem_type,
            patch("langchain.agents.create_agent", return_value=graph),
        ):
            _create_goal_criteria_agent(
                model=MagicMock(),
                repository_backend=backend,
                repository_root="/workspace",
                context_tools=[],
                auto_mode_enabled=True,
                fs_tools=["read_file"],
            )

        filesystem_type.assert_called_once_with(
            backend=backend,
            tools=["read_file"],
            grep_max_count=_REPOSITORY_GREP_MATCH_LIMIT,
            tool_token_limit_before_evict=None,
        )

    def test_parent_allowlist_intersects_with_repository_tools(self) -> None:
        """The criteria agent keeps only allowed *repository* tools.

        The repository tool set is `["ls", "read_file", "glob", "grep"]`. Given
        an allowlist that overlaps it partially and also names a non-repository
        tool (`write_file`), the result must be the intersection in repository
        order — multiple repository names survive, the disallowed repository
        names (`glob`, `grep`) drop, and the non-repository name never appears.
        """
        backend = MagicMock()
        filesystem = MagicMock()
        graph = MagicMock()
        graph.with_config.return_value = graph

        with (
            patch(
                "deepagents.middleware.FilesystemMiddleware",
                return_value=filesystem,
            ) as filesystem_type,
            patch("langchain.agents.create_agent", return_value=graph),
        ):
            _create_goal_criteria_agent(
                model=MagicMock(),
                repository_backend=backend,
                repository_root="/workspace",
                context_tools=[],
                auto_mode_enabled=True,
                fs_tools=["ls", "read_file", "write_file"],
            )

        filesystem_type.assert_called_once_with(
            backend=backend,
            tools=["ls", "read_file"],
            grep_max_count=_REPOSITORY_GREP_MATCH_LIMIT,
            tool_token_limit_before_evict=None,
        )

    @staticmethod
    def _async_hitl(*, auto_mode_enabled: bool = True) -> AsyncApprovalHITLMiddleware:
        from deepagents_code.agent import AsyncApprovalHITLMiddleware

        fetch = StructuredTool.from_function(
            func=lambda url: url,
            name="fetch_url",
            description="Fetch a URL.",
        )
        graph = MagicMock()
        graph.with_config.return_value = graph
        with patch("langchain.agents.create_agent", return_value=graph) as make_agent:
            _create_goal_criteria_agent(
                model=MagicMock(),
                repository_backend=None,
                repository_root="/",
                context_tools=[fetch],
                auto_mode_enabled=auto_mode_enabled,
            )

        return next(
            item
            for item in make_agent.call_args.kwargs["middleware"]
            if isinstance(item, AsyncApprovalHITLMiddleware)
        )

    @staticmethod
    def _async_runtime(store: _LoopBoundAsyncStore) -> SimpleNamespace:
        from deepagents_code.approval_mode import approval_mode_key

        thread_id = "criteria-thread"
        return SimpleNamespace(
            context={
                "thread_id": thread_id,
                "approval_mode_key": approval_mode_key(thread_id),
                "approval_mode": "auto",
            },
            store=store,
            stream_writer=lambda _event: None,
            execution_info=None,
            server_info=None,
        )

    @staticmethod
    def _fetch_state() -> dict[str, object]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "fetch_url",
                            "args": {"url": "https://example.com/context"},
                            "id": "call-fetch",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    async def test_context_tool_still_interrupts_in_manual(self) -> None:
        """Goal-criteria external context retains its Manual approval gate."""
        middleware = self._async_hitl()
        store = _LoopBoundAsyncStore({"mode": "manual"})

        with (
            patch(
                "langchain.agents.middleware.human_in_the_loop.interrupt",
                side_effect=GraphInterrupt(()),
            ),
            pytest.raises(GraphInterrupt),
        ):
            await middleware.aafter_model(
                cast("Any", self._fetch_state()),
                cast("Any", self._async_runtime(store)),
            )

        assert store.aget_calls == 1
        assert store.get_calls == 0

    async def test_context_tool_auto_is_ineligible_when_classifier_is_off(
        self,
    ) -> None:
        """Goal-criteria Auto cannot bypass an ineligible parent runtime."""
        middleware = self._async_hitl(auto_mode_enabled=False)
        store = _LoopBoundAsyncStore({"mode": "auto"})

        with (
            patch(
                "langchain.agents.middleware.human_in_the_loop.interrupt",
                side_effect=GraphInterrupt(()),
            ),
            pytest.raises(GraphInterrupt),
        ):
            await middleware.aafter_model(
                cast("Any", self._fetch_state()),
                cast("Any", self._async_runtime(store)),
            )


class TestGoalCriteriaRequestValidation:
    """`_goal_criteria_request` is the trust boundary for graph input."""


class TestProposalParsing:
    """Parsing helpers stay robust to messy nested criteria output."""


class TestNoCompleteProposalFailure:
    """A nested run that yields no proposal fails loudly and logs the output."""

    def test_before_agent_validates_the_objective_it_actually_applies(self) -> None:
        """A `create` applies the user's objective, not the model's paraphrase.

        The model is told to preserve the objective verbatim and nothing enforces
        that. A shortened paraphrase can satisfy `GoalProposal._fit_notice_budget`
        while the pair that actually gets persisted exceeds the budget, so the
        check has to run against the applied objective.
        """
        user_objective = "u" * (GOAL_APPLICATION_CHAR_LIMIT // 2)
        criteria_text = "c" * (GOAL_APPLICATION_CHAR_LIMIT // 2 + 1)
        criteria = MagicMock()
        criteria.invoke.return_value = {
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {"objective": "short", "criteria": criteria_text}
                    )
                )
            ]
        }
        middleware = GoalCriteriaMiddleware(criteria)
        state = cast(
            "GoalCriteriaState",
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "r",
                    "kind": "create",
                    "objective": user_objective,
                },
            },
        )

        # The proposal the model returned fits on its own; the applied pair does not.
        GoalProposal.model_validate({"objective": "short", "criteria": criteria_text})

        with pytest.raises(GoalStateSizeError, match="combined"):
            middleware.before_agent(state, TestGoalCriteriaMiddleware._runtime())


class TestRepositoryPathGuards:
    """Path guards reject traversal and non-absolute paths on both paths."""

    @staticmethod
    def _backend() -> MagicMock:
        backend = MagicMock()
        backend.ls.return_value = LsResult(entries=[])
        backend.als = AsyncMock(return_value=LsResult(entries=[]))
        return backend

    @pytest.mark.parametrize("name", ["read_file", "ls", "glob", "grep"])
    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "~/secrets", "relative/x", "/a/../b", "/a/~user/b"]
    )
    def test_sync_rejects_unsafe_paths(self, name: str, path: str) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="p", name=name, path=path
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()
        backend.ls.assert_not_called()

    @pytest.mark.parametrize("name", ["read_file", "ls", "glob", "grep"])
    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "~/secrets", "relative/x", "/a/../b", "/a/~user/b"]
    )
    async def test_async_rejects_unsafe_paths(self, name: str, path: str) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="p", name=name, path=path
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_awaited()
        backend.als.assert_not_awaited()

    def test_sync_rejects_sandbox_symlink_escape(self, tmp_path: Path) -> None:
        root = tmp_path / "repository"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")
        (root / "escape").symlink_to(outside, target_is_directory=True)
        backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=False)
        middleware = _RepositoryToolBudgetMiddleware(backend, root=str(root))
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="symlink",
                path=str(root / "escape" / secret.name),
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()

    async def test_async_rejects_sandbox_symlink_escape(self, tmp_path: Path) -> None:
        root = tmp_path / "repository"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")
        (root / "escape").symlink_to(outside, target_is_directory=True)
        backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=False)
        middleware = _RepositoryToolBudgetMiddleware(backend, root=str(root))
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="symlink",
                path=str(root / "escape" / secret.name),
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_awaited()

    @pytest.mark.parametrize("name", ["glob", "grep"])
    @pytest.mark.parametrize("pattern", ["../*.py", "~/secrets/*", "a/../b/*"])
    def test_sync_rejects_traversing_search_patterns(
        self, name: str, pattern: str
    ) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        handler = MagicMock()
        request = TestRepositoryToolBudgetMiddleware._request(
            call_id="pattern",
            name=name,
            path="/",
            search_glob=pattern if name == "grep" else None,
        )
        if name == "glob":
            request.tool_call["args"]["pattern"] = pattern

        result = middleware.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()


class TestAsyncRepositoryBudget:
    """The async budget path mirrors the sync rejections and bounding."""

    @staticmethod
    def _backend(*, entries: list[FileInfo] | None = None) -> MagicMock:
        backend = MagicMock()
        backend.als = AsyncMock(
            return_value=LsResult(
                entries=entries
                if entries is not None
                else [{"path": "/src.py", "is_dir": False, "size": 10}]
            )
        )
        return backend


class TestGoalContextFallbackDoubleFailure:
    """When the goal-only retry also fails, the original error is surfaced."""


class TestGoalCriteriaFallback:
    """Graph-level context-agent failures degrade to goal-only generation."""

    @staticmethod
    def _state() -> GoalCriteriaState:
        return cast(
            "GoalCriteriaState",
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "fallback-op",
                    "kind": "create",
                    "objective": "ship it",
                },
            },
        )

    @staticmethod
    def _fallback(criteria: str = "- goal-only criteria") -> MagicMock:
        agent = MagicMock()
        agent.invoke.return_value = {
            "structured_response": {"objective": "ship it", "criteria": criteria}
        }
        agent.ainvoke = AsyncMock(
            return_value={
                "structured_response": {"objective": "ship it", "criteria": criteria}
            }
        )
        return agent

    def test_hitl_interrupt_is_never_swallowed_by_the_fallback(self) -> None:
        criteria = MagicMock()
        criteria.invoke.side_effect = GraphInterrupt(())
        fallback = self._fallback()
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        with pytest.raises(GraphInterrupt):
            middleware.before_agent(
                self._state(), TestGoalCriteriaMiddleware._runtime()
            )
        fallback.invoke.assert_not_called()

    async def test_async_hitl_interrupt_is_never_swallowed(self) -> None:
        criteria = MagicMock()
        criteria.ainvoke = AsyncMock(side_effect=GraphInterrupt(()))
        fallback = self._fallback()
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        with pytest.raises(GraphInterrupt):
            await middleware.abefore_agent(
                self._state(), TestGoalCriteriaMiddleware._runtime()
            )
        fallback.ainvoke.assert_not_awaited()

    @staticmethod
    def _rejection(objective: str, criteria: str) -> BaseException:
        """Wrap an invalid proposal exactly as the structured-output loop does.

        The layering matters and cannot be faked: pydantic converts the
        validator's `ValueError` into a `ValidationError` without chaining the
        original, the parser rewraps that as a plain `ValueError`, and the agent
        builds a `StructuredOutputValidationError` that has not been raised yet,
        so it has neither `__cause__` nor `__context__`. Anything that recovers
        the size error by walking the exception chain passes a hand-built double
        and fails here.

        Returns:
            The exception the agent hands to its `handle_errors` callable.
        """
        from langchain.agents.structured_output import (
            StructuredOutputValidationError,
            _parse_with_schema,
        )

        data = {"objective": objective, "criteria": criteria}
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "GoalProposal", "args": data, "id": "call-1"}],
        )
        try:
            _parse_with_schema(GoalProposal, "pydantic", data)
        except Exception as exc:  # noqa: BLE001
            return StructuredOutputValidationError("GoalProposal", exc, ai_message)
        pytest.fail("proposal was expected to be invalid")


class TestPreflightBackendErrors:
    """Backend faults during preflight degrade to a bounded, logged error."""


class TestRepositoryBudgetEdgeCases:
    """Read/grep argument clamps and the per-operation budget cache are bounded."""
