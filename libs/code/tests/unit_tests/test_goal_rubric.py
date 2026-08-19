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

    def test_system_prompt_preserves_limits_and_fallback(self) -> None:
        normalized = " ".join(GOAL_RUBRIC_SYSTEM_PROMPT.split())

        assert "usually 2-5 bullets" in normalized
        assert "Do not start implementing the goal" in normalized
        assert f"no more than {_REPOSITORY_TOOL_CALL_LIMIT} repository" in normalized
        assert "rejected" in normalized
        assert "draft criteria from the goal alone" in normalized
        assert "`fetch_url`" in normalized
        assert "`web_search`" in normalized
        assert "never use search to invent additional requirements" in normalized
        assert "configured MCP tools" in normalized

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

    def test_recent_human_and_assistant_text_is_xml_serialized(self) -> None:
        context = _conversation_context(
            [
                HumanMessage(content="use <Widget> & keep it stable"),
                AIMessage(content="I found x > y"),
            ]
        )

        assert (
            '<message type="human">use &lt;Widget&gt; &amp; keep it stable</message>'
            in context
        )
        assert '<message type="ai">I found x &gt; y</message>' in context

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

    def test_context_is_bounded_and_favors_recent_messages(self) -> None:
        messages = [
            HumanMessage(content=f"message-{index} " + "&" * 2_000)
            for index in range(_CONVERSATION_CONTEXT_MESSAGE_LIMIT + 5)
        ]

        context = _conversation_context(messages)

        assert len(context) <= _CONVERSATION_CONTEXT_SERIALIZED_LIMIT
        assert f"message-{len(messages) - 1}" in context
        assert "message-0" not in context

    def test_no_usable_history_preserves_the_existing_prompt(self) -> None:
        request = self._request()

        assert _conversation_context([]) == ""
        assert _prompt_with_conversation_context(request, []) == (
            _goal_rubric_human_prompt("ship the explicit goal")
        )

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

    def test_sync_failure_retries_without_context_tools(self) -> None:
        middleware = _GoalContextFallbackMiddleware()
        request = MagicMock()
        goal = HumanMessage(content="ship login")
        request.messages = [
            goal,
            AIMessage(
                content="",
                tool_calls=[{"name": "fetch_url", "args": {}, "id": "call"}],
            ),
            ToolMessage(content="oversized context", tool_call_id="call"),
        ]
        fallback = MagicMock()
        request.override.return_value = fallback
        handler = MagicMock(side_effect=[RuntimeError("context failed"), "response"])

        result = middleware.wrap_model_call(request, handler)

        assert result == "response"
        request.override.assert_called_once_with(messages=[goal], tools=[])
        assert handler.call_args_list == [call(request), call(fallback)]

    async def test_async_failure_retries_without_context_tools(self) -> None:
        middleware = _GoalContextFallbackMiddleware()
        request = MagicMock()
        goal = HumanMessage(content="ship login")
        request.messages = [
            goal,
            AIMessage(
                content="",
                tool_calls=[{"name": "docs_search", "args": {}, "id": "call"}],
            ),
            ToolMessage(content="malformed context", tool_call_id="call"),
        ]
        fallback = MagicMock()
        request.override.return_value = fallback
        handler = AsyncMock(side_effect=[RuntimeError("context failed"), "response"])

        result = await middleware.awrap_model_call(request, handler)

        assert result == "response"
        request.override.assert_called_once_with(messages=[goal], tools=[])
        assert handler.await_args_list == [call(request), call(fallback)]


class TestCriteriaContextBudgetMiddleware:
    """All gathered tool context shares one bounded operation budget."""

    def test_sync_results_share_budget_across_context_tools(self) -> None:
        middleware = _CriteriaContextBudgetMiddleware()
        first = TestRepositoryToolBudgetMiddleware._request(
            call_id="fetch",
            name="fetch_url",
        )
        second = TestRepositoryToolBudgetMiddleware._request(
            call_id="mcp",
            name="docs_search",
        )

        fetch_result = middleware.wrap_tool_call(
            first,
            lambda _: ToolMessage(
                content="f" * (_CRITERIA_CONTEXT_TOTAL_TEXT_LIMIT - 100),
                tool_call_id="fetch",
            ),
        )
        mcp_result = middleware.wrap_tool_call(
            second,
            lambda _: ToolMessage(content="m" * 1_000, tool_call_id="mcp"),
        )

        assert isinstance(fetch_result, ToolMessage)
        assert isinstance(mcp_result, ToolMessage)
        assert len(fetch_result.text) + len(mcp_result.text) == (
            _CRITERIA_CONTEXT_TOTAL_TEXT_LIMIT
        )
        assert "Criteria context limit reached" in mcp_result.text

    async def test_async_single_result_is_bounded_and_text_only(self) -> None:
        middleware = _CriteriaContextBudgetMiddleware()
        request = TestRepositoryToolBudgetMiddleware._request(
            call_id="mcp",
            name="docs_search",
        )
        result = await middleware.awrap_tool_call(
            request,
            AsyncMock(
                return_value=ToolMessage(
                    content=[
                        {"type": "text", "text": "x" * 40_000},
                        {"type": "image", "base64": "y" * 40_000},
                    ],
                    tool_call_id="mcp",
                )
            ),
        )

        assert isinstance(result, ToolMessage)
        assert isinstance(result.content, str)
        assert len(result.text) == _CRITERIA_CONTEXT_TOTAL_TEXT_LIMIT
        assert "Criteria context limit reached" in result.text


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

    def test_budget_is_scoped_to_rubric_evaluation(self) -> None:
        middleware = _ContextToolCallBudgetMiddleware(
            {"notion_fetch"},
            limit=2,
        )
        handler = MagicMock(
            return_value=ToolMessage(content="page", tool_call_id="call")
        )

        first = middleware.wrap_tool_call(self._request("1", "run:0"), handler)
        second = middleware.wrap_tool_call(self._request("2", "run:0"), handler)
        exhausted = middleware.wrap_tool_call(self._request("3", "run:0"), handler)
        next_iteration = middleware.wrap_tool_call(self._request("4", "run:1"), handler)

        assert isinstance(first, ToolMessage)
        assert isinstance(second, ToolMessage)
        assert isinstance(exhausted, ToolMessage)
        assert "Verification context limit reached" in exhausted.text
        assert isinstance(next_iteration, ToolMessage)
        assert next_iteration.text == "page"
        assert handler.call_count == 3


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

    def test_sync_search_budget_is_per_operation(self) -> None:
        middleware = _WebSearchBudgetMiddleware()
        handler = MagicMock(
            return_value=ToolMessage(content="result", tool_call_id="search")
        )
        for index in range(_WEB_SEARCH_CALL_LIMIT):
            middleware.wrap_tool_call(self._request(str(index)), handler)

        exhausted = middleware.wrap_tool_call(self._request("over"), handler)
        independent = middleware.wrap_tool_call(
            self._request("new", operation_id="another-operation"), handler
        )

        assert isinstance(exhausted, ToolMessage)
        assert exhausted.text == (
            "Web search limit reached. Continue using the available evidence "
            "and context already gathered."
        )
        assert "acceptance criteria" not in exhausted.text
        assert isinstance(independent, ToolMessage)
        assert independent.text == "result"

    async def test_async_search_budget_is_enforced(self) -> None:
        middleware = _WebSearchBudgetMiddleware()
        handler = AsyncMock(
            return_value=ToolMessage(content="result", tool_call_id="search")
        )
        for index in range(_WEB_SEARCH_CALL_LIMIT):
            await middleware.awrap_tool_call(self._request(str(index)), handler)

        result = await middleware.awrap_tool_call(self._request("over"), handler)

        assert isinstance(result, ToolMessage)
        assert "Web search limit reached" in result.text


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

    def test_uses_server_backend_and_clamps_reads(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="read"))

        middleware.wrap_tool_call(self._request(call_id="read"), handler)

        backend.ls.assert_called_once_with("/")
        request = handler.call_args.args[0]
        assert request.tool_call["args"]["limit"] == _REPOSITORY_READ_LINE_LIMIT

    def test_rejects_large_server_backend_file_before_read(self) -> None:
        backend = self._backend(size=_REPOSITORY_READ_BYTE_LIMIT + 1)
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            self._request(call_id="large"),
            handler,
        )

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "size limit" in result.text

    def test_rejects_large_server_backend_directory(self) -> None:
        backend = MagicMock()
        backend.ls.return_value = LsResult(
            entries=[
                {"path": f"/{index}", "is_dir": False}
                for index in range(_REPOSITORY_DIRECTORY_ENTRY_LIMIT + 1)
            ]
        )
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            self._request(call_id="ls", name="ls", path="/"),
            handler,
        )

        handler.assert_not_called()
        assert isinstance(result, ToolMessage)
        assert "listing limit" in result.text

    def test_total_repository_calls_and_output_are_bounded(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)

        def handler(request: ToolCallRequest) -> ToolMessage:
            return ToolMessage(
                content="x" * (_REPOSITORY_TOOL_RESULT_LIMIT + 500),
                tool_call_id=request.tool_call["id"],
            )

        results = [
            middleware.wrap_tool_call(self._request(call_id=str(index)), handler)
            for index in range(_REPOSITORY_TOOL_CALL_LIMIT + 1)
        ]

        assert all(
            isinstance(result, ToolMessage)
            and len(result.text) <= _REPOSITORY_TOOL_RESULT_LIMIT
            for result in results[:-1]
        )
        assert isinstance(results[-1], ToolMessage)
        assert "context limit reached" in results[-1].text

    def test_repository_budget_is_independent_per_operation(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="read"))
        for index in range(_REPOSITORY_TOOL_CALL_LIMIT):
            middleware.wrap_tool_call(
                self._request(call_id=str(index), operation_id="operation-1"),
                handler,
            )

        result = middleware.wrap_tool_call(
            self._request(call_id="new", operation_id="operation-2"),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.text == "ok"

    def test_glob_and_grep_share_the_repository_call_budget(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="x"))
        for index in range(_REPOSITORY_TOOL_CALL_LIMIT - 2):
            middleware.wrap_tool_call(self._request(call_id=str(index)), handler)
        middleware.wrap_tool_call(
            self._request(call_id="glob", name="glob", path="/"), handler
        )
        middleware.wrap_tool_call(
            self._request(call_id="grep", name="grep", path="/"), handler
        )

        result = middleware.wrap_tool_call(
            self._request(call_id="over", name="glob", path="/"), handler
        )

        assert isinstance(result, ToolMessage)
        assert "context limit reached" in result.text

    def test_grep_match_count_is_clamped(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="g"))

        middleware.wrap_tool_call(
            self._request(call_id="g", name="grep", path="/"), handler
        )

        bounded = handler.call_args.args[0]
        assert bounded.tool_call["args"]["max_count"] == _REPOSITORY_GREP_MATCH_LIMIT

    @pytest.mark.parametrize("name", ["glob", "grep"])
    def test_search_without_path_defaults_to_repository_root(self, name: str) -> None:
        middleware = _RepositoryToolBudgetMiddleware(
            self._backend(),
            root="/workspace",
        )
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="s"))
        request = self._request(call_id="s", name=name, path="/workspace")
        request.tool_call["args"].pop("path")

        middleware.wrap_tool_call(request, handler)

        bounded = handler.call_args.args[0]
        assert bounded.tool_call["args"]["path"] == "/workspace"

    def test_glob_match_count_and_output_are_bounded(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        paths = [f"/{index}.py" for index in range(_REPOSITORY_GLOB_MATCH_LIMIT + 5)]
        handler = MagicMock(
            return_value=ToolMessage(content=str(paths), tool_call_id="glob")
        )

        result = middleware.wrap_tool_call(
            self._request(call_id="glob", name="glob", path="/"), handler
        )

        assert isinstance(result, ToolMessage)
        body = result.text.partition("\n\n")[0]
        assert len(cast("list[str]", ast.literal_eval(body))) == (
            _REPOSITORY_GLOB_MATCH_LIMIT
        )
        assert len(result.text) <= _REPOSITORY_TOOL_RESULT_LIMIT
        assert "Glob results limited" in result.text

    def test_external_tools_do_not_consume_repository_budget(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())
        request = self._request(call_id="mcp", name="docs_search", path="/")
        result = ToolMessage(content="external", tool_call_id="mcp")
        handler = MagicMock(return_value=result)

        assert middleware.wrap_tool_call(request, handler) is result
        handler.assert_called_once_with(request)

    def test_non_text_repository_result_is_omitted(self) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend())

        result = middleware.wrap_tool_call(
            self._request(call_id="media"),
            lambda _request: Command(update={}),
        )

        assert isinstance(result, ToolMessage)
        assert "Non-text repository content omitted" in result.text


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

    def test_normal_agent_run_is_unchanged(self) -> None:
        criteria = MagicMock()
        middleware = GoalCriteriaMiddleware(criteria)

        state = cast("GoalCriteriaState", {"messages": []})
        assert middleware.before_agent(state, self._runtime()) is None
        criteria.invoke.assert_not_called()

    def test_create_request_runs_nested_agent_and_preserves_objective(self) -> None:
        criteria = MagicMock()
        criteria.invoke.return_value = {
            "structured_response": {
                "objective": "model changed it",
                "criteria": "- observable result",
            }
        }
        middleware = GoalCriteriaMiddleware(criteria)
        request: GoalCriteriaRequest = {
            "request_id": "request-1",
            "kind": "create",
            "objective": "ship it",
        }

        messages = [
            HumanMessage(content="Earlier the user named src/auth.py."),
            AIMessage(content="I found the login handler."),
        ]
        original_messages = list(messages)
        state = cast(
            "GoalCriteriaState",
            {"messages": messages, "goal_criteria_request": request},
        )
        update = middleware.before_agent(state, self._runtime())

        assert update is not None
        assert update == {
            "goal_criteria_request": None,
            "rubric": None,
            "_pending_goal_objective": "ship it",
            "_pending_goal_rubric": "- observable result",
            "_pending_goal_kind": "create",
            "_pending_goal_request_id": "request-1",
            "jump_to": "end",
        }
        child_input = criteria.invoke.call_args.args[0]
        assert child_input["criteria_objective"] == "ship it"
        assert child_input["criteria_operation_id"] == "request-1"
        assert child_input["messages"][0]["content"].startswith(
            "<operation>draft</operation>"
        )
        prompt = child_input["messages"][0]["content"]
        assert "<conversation_context>" in prompt
        assert "Earlier the user named src/auth.py." in prompt
        assert "I found the login handler." in prompt
        assert criteria.invoke.call_args.kwargs["context"] == {
            "model": "openai:gpt-5.5"
        }
        assert "messages" not in update
        assert state["messages"] == original_messages
        assert all(
            current is original
            for current, original in zip(
                state["messages"], original_messages, strict=True
            )
        )

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

    def test_json_fallback_is_parsed_on_server(self) -> None:
        criteria = MagicMock()
        criteria.invoke.return_value = {
            "messages": [
                {
                    "type": "ai",
                    "content": '{"objective":"ship it","criteria":"- fallback"}',
                }
            ]
        }
        middleware = GoalCriteriaMiddleware(criteria)

        update = middleware.before_agent(
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "request-3",
                    "kind": "create",
                    "objective": "ship it",
                },
            },
            self._runtime(),
        )

        assert update is not None
        assert update["_pending_goal_rubric"] == "- fallback"

    def test_rejection_regeneration_preserves_supplied_objective(self) -> None:
        criteria = MagicMock()
        criteria.invoke.return_value = {
            "structured_response": {
                "objective": "model rewrote it",
                "criteria": "- regenerated",
            }
        }
        middleware = GoalCriteriaMiddleware(criteria)

        update = middleware.before_agent(
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "regenerate",
                    "kind": "create",
                    "objective": "keep this objective exactly",
                    "previous_criteria": "- old",
                    "feedback": "make it concrete",
                },
            },
            self._runtime(),
        )

        assert update is not None
        assert update["_pending_goal_objective"] == "keep this objective exactly"
        prompt = criteria.invoke.call_args.args[0]["messages"][0]["content"]
        assert "<previous_criteria>\n- old\n</previous_criteria>" in prompt
        assert "<user_feedback>\nmake it concrete\n</user_feedback>" in prompt

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

    def test_wires_only_read_repository_tools_plus_external_context(self) -> None:
        model = MagicMock()
        backend = MagicMock()
        fetch = TestCriteriaHitlPolicy._tool("fetch_url", "Fetch URL")
        web = TestCriteriaHitlPolicy._tool("web_search", "Search the web")
        mcp = TestCriteriaHitlPolicy._tool("docs_search", "Search docs")
        filesystem = MagicMock()
        graph = MagicMock()
        graph.with_config.return_value = "configured-graph"

        with (
            patch(
                "deepagents.middleware.FilesystemMiddleware",
                return_value=filesystem,
            ) as filesystem_type,
            patch("langchain.agents.create_agent", return_value=graph) as create_agent,
            patch(
                "langchain.agents.middleware.HumanInTheLoopMiddleware",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        ):
            result = create_goal_criteria_agent(
                model=model,
                repository_backend=backend,
                repository_root="/workspace",
                context_tools=[fetch, web, mcp],
            )

        assert result == "configured-graph"
        filesystem_type.assert_called_once_with(
            backend=backend,
            tools=["ls", "read_file", "glob", "grep"],
            grep_max_count=_REPOSITORY_GREP_MATCH_LIMIT,
            tool_token_limit_before_evict=None,
        )
        kwargs = create_agent.call_args.kwargs
        assert kwargs["model"] is model
        assert kwargs["tools"] == [fetch, web, mcp]
        assert kwargs["name"] == "goal_criteria_agent"
        assert kwargs["state_schema"] is GoalCriteriaAgentState
        assert "repository root `/workspace`" in kwargs["system_prompt"]
        assert all(
            name not in {"write_file", "edit_file", "delete", "execute"}
            for name in (tool.name for tool in kwargs["tools"])
        )
        budget = next(
            item
            for item in kwargs["middleware"]
            if isinstance(item, _RepositoryToolBudgetMiddleware)
        )
        assert budget._bounds.root == "/workspace"
        assert any(
            isinstance(item, _CriteriaContextBudgetMiddleware)
            for item in kwargs["middleware"]
        )

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

    async def test_context_tool_honors_auto_from_async_store(self) -> None:
        """Goal-criteria external context bypasses HITL in eligible Auto."""
        middleware = self._async_hitl()
        store = _LoopBoundAsyncStore({"mode": "auto"})

        update = await middleware.aafter_model(
            cast("Any", self._fetch_state()),
            cast("Any", self._async_runtime(store)),
        )

        assert update is None
        assert store.aget_calls == 1
        assert store.get_calls == 0

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

    def test_client_generation_symbols_are_removed(self) -> None:
        import deepagents_code.goal_rubric as module

        assert not hasattr(module, "generate_goal_rubric")
        assert not hasattr(module, "generate_goal_amendment")

    async def test_preserves_async_callable_as_coroutine_tool(self) -> None:
        async def fetch_context(query: str) -> str:
            """Fetch async context."""
            await asyncio.sleep(0)
            return f"context for {query}"

        graph = MagicMock()
        graph.with_config.return_value = "configured-graph"
        with (
            patch("langchain.agents.create_agent", return_value=graph) as create_agent,
            patch(
                "langchain.agents.middleware.HumanInTheLoopMiddleware",
                side_effect=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
        ):
            create_goal_criteria_agent(
                model=MagicMock(),
                repository_backend=None,
                context_tools=[fetch_context],
            )

        context_tool = create_agent.call_args.kwargs["tools"][0]
        assert context_tool.coroutine is fetch_context
        assert context_tool.func is None
        assert await context_tool.ainvoke({"query": "login"}) == "context for login"

    @pytest.mark.parametrize(
        ("name", "has_repository"),
        [
            ("GoalProposal", False),
            ("ls", True),
            ("read_file", True),
            ("glob", True),
            ("grep", True),
        ],
    )
    def test_rejects_context_tool_names_reserved_by_criteria_agent(
        self,
        name: str,
        has_repository: bool,
    ) -> None:
        context_tool = TestCriteriaHitlPolicy._tool(name, "Conflicting tool")
        backend = MagicMock() if has_repository else None

        with pytest.raises(ValueError, match=name):
            create_goal_criteria_agent(
                model=MagicMock(),
                repository_backend=backend,
                context_tools=[context_tool],
            )


class TestGoalCriteriaRequestValidation:
    """`_goal_criteria_request` is the trust boundary for graph input."""

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError):
            _goal_criteria_request("not a dict")

    def test_rejects_missing_request_id(self) -> None:
        with pytest.raises(ValueError, match="request_id"):
            _goal_criteria_request({"kind": "create", "objective": "ship it"})

    def test_rejects_blank_request_id(self) -> None:
        with pytest.raises(ValueError, match="request_id"):
            _goal_criteria_request(
                {"request_id": "   ", "kind": "create", "objective": "ship it"}
            )

    def test_rejects_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="create or amend"):
            _goal_criteria_request(
                {"request_id": "r", "kind": "delete", "objective": "ship it"}
            )

    def test_rejects_missing_objective(self) -> None:
        with pytest.raises(ValueError, match="objective"):
            _goal_criteria_request({"request_id": "r", "kind": "create"})

    def test_rejects_non_string_field(self) -> None:
        with pytest.raises(TypeError, match="feedback"):
            _goal_criteria_request(
                {
                    "request_id": "r",
                    "kind": "create",
                    "objective": "ship it",
                    "feedback": 5,
                }
            )

    def test_rejects_amend_without_criteria(self) -> None:
        with pytest.raises(ValueError, match="criteria and feedback"):
            _goal_criteria_request(
                {
                    "request_id": "r",
                    "kind": "amend",
                    "objective": "ship it",
                    "feedback": "add tests",
                }
            )

    def test_rejects_amend_without_feedback(self) -> None:
        with pytest.raises(ValueError, match="criteria and feedback"):
            _goal_criteria_request(
                {
                    "request_id": "r",
                    "kind": "amend",
                    "objective": "ship it",
                    "criteria": "- works",
                }
            )

    def test_normalizes_create_and_drops_stray_criteria(self) -> None:
        request = _goal_criteria_request(
            {
                "request_id": "r",
                "kind": "create",
                "objective": "ship it",
                "criteria": "- ignored on create",
                "previous_criteria": "- old",
            }
        )

        assert request == {
            "request_id": "r",
            "kind": "create",
            "objective": "ship it",
            "previous_criteria": "- old",
        }

    def test_normalizes_amend_and_drops_previous_criteria(self) -> None:
        request = _goal_criteria_request(
            {
                "request_id": "r",
                "kind": "amend",
                "objective": "ship it",
                "criteria": "- works",
                "feedback": "add passkeys",
                "previous_criteria": "- dropped on amend",
            }
        )

        assert request == {
            "request_id": "r",
            "kind": "amend",
            "objective": "ship it",
            "criteria": "- works",
            "feedback": "add passkeys",
        }


class TestProposalParsing:
    """Parsing helpers stay robust to messy nested criteria output."""

    def test_coerce_rejects_non_dict(self) -> None:
        assert _coerce_goal_proposal("nope") is None

    def test_coerce_rejects_empty_strings(self) -> None:
        assert _coerce_goal_proposal({"objective": "  ", "criteria": "c"}) is None

    def test_coerce_reads_direct_fields(self) -> None:
        assert _coerce_goal_proposal({"objective": "o", "criteria": "- c"}) == (
            "o",
            "- c",
        )

    def test_coerce_reads_structured_response(self) -> None:
        assert _coerce_goal_proposal(
            {"structured_response": {"objective": "o", "criteria": "- c"}}
        ) == ("o", "- c")

    def test_coerce_skips_incomplete_structured_and_walks_siblings(self) -> None:
        assert _coerce_goal_proposal(
            {
                "structured_response": {"objective": "", "criteria": "- c"},
                "other": {"objective": "o2", "criteria": "- c2"},
            }
        ) == ("o2", "- c2")

    def test_text_parses_code_fenced_json(self) -> None:
        assert _goal_proposal_from_text(
            '```json\n{"objective": "o", "criteria": "- c"}\n```'
        ) == ("o", "- c")

    def test_text_returns_none_on_invalid_json(self) -> None:
        assert _goal_proposal_from_text("not json at all") is None

    def test_result_parses_ai_message_json(self) -> None:
        result = {
            "messages": [AIMessage(content='{"objective": "o", "criteria": "- c"}')]
        }

        assert _proposal_from_result(result) == ("o", "- c")

    def test_result_skips_non_string_content(self) -> None:
        assert (
            _proposal_from_result({"messages": [{"content": {"not": "text"}}]}) is None
        )

    def test_result_returns_none_when_messages_not_list(self) -> None:
        assert _proposal_from_result({"messages": "nope"}) is None


class TestNoCompleteProposalFailure:
    """A nested run that yields no proposal fails loudly and logs the output."""

    def test_before_agent_raises_when_no_proposal(self) -> None:
        criteria = MagicMock()
        criteria.invoke.return_value = {"messages": []}
        middleware = GoalCriteriaMiddleware(criteria)
        state = cast(
            "GoalCriteriaState",
            {
                "messages": [],
                "goal_criteria_request": {
                    "request_id": "r",
                    "kind": "create",
                    "objective": "ship it",
                },
            },
        )

        with pytest.raises(RuntimeError, match="no complete proposal"):
            middleware.before_agent(state, TestGoalCriteriaMiddleware._runtime())

    def test_before_agent_rejects_oversized_pair_from_raw_json(self) -> None:
        """The text-fallback path cannot smuggle oversized state past the schema.

        `_proposal_from_result` parses raw JSON out of message content when
        structured output is absent, so `GoalProposal`'s validators never run.
        `_update`'s own check is the only thing standing between that path and a
        goal too large to include in the notice the model reads every turn.
        """
        criteria = MagicMock()
        criteria.invoke.return_value = {
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {
                            "objective": "ship it",
                            "criteria": "x" * GOAL_APPLICATION_CHAR_LIMIT,
                        }
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
                    "objective": "ship it",
                },
            },
        )

        with pytest.raises(GoalStateSizeError, match="combined"):
            middleware.before_agent(state, TestGoalCriteriaMiddleware._runtime())

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

    def test_summarize_result_is_bounded(self) -> None:
        big = "x" * (_CRITERIA_RESULT_LOG_LIMIT + 100)

        summary = _summarize_criteria_result({"messages": [AIMessage(content=big)]})

        assert "last_message_text=" in summary
        assert len(summary) < _CRITERIA_RESULT_LOG_LIMIT + 200
        assert _summarize_criteria_result("plain") == "'plain'"


class TestRepositoryPathGuards:
    """Path guards reject traversal and non-absolute paths on both paths."""

    @staticmethod
    def _backend() -> MagicMock:
        backend = MagicMock()
        backend.ls.return_value = LsResult(entries=[])
        backend.als = AsyncMock(return_value=LsResult(entries=[]))
        return backend

    @pytest.mark.parametrize("name", ["glob", "grep"])
    def test_sync_replaces_none_search_path_with_root(self, name: str) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend(), root="/workspace")
        handler = MagicMock(
            return_value=ToolMessage(content="ok", tool_call_id="search")
        )

        middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="search", name=name, path=None
            ),
            handler,
        )

        request = handler.call_args.args[0]
        assert request.tool_call["args"]["path"] == "/workspace"

    @pytest.mark.parametrize("name", ["glob", "grep"])
    async def test_async_replaces_none_search_path_with_root(self, name: str) -> None:
        middleware = _RepositoryToolBudgetMiddleware(self._backend(), root="/workspace")
        handler = AsyncMock(
            return_value=ToolMessage(content="ok", tool_call_id="search")
        )

        await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="search", name=name, path=None
            ),
            handler,
        )

        request = handler.call_args.args[0]
        assert request.tool_call["args"]["path"] == "/workspace"

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

    @pytest.mark.parametrize("name", ["read_file", "ls", "glob", "grep"])
    def test_sync_rejects_absolute_paths_outside_configured_root(
        self, name: str
    ) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend, root="/workspace")
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="outside",
                name=name,
                path="/etc/passwd",
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()

    @pytest.mark.parametrize("name", ["read_file", "ls", "glob", "grep"])
    async def test_async_rejects_absolute_paths_outside_configured_root(
        self, name: str
    ) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend, root="/workspace")
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="outside",
                name=name,
                path="/etc/passwd",
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_awaited()

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

    async def test_async_clamps_read_limit(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="r"))

        await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="r"),
            handler,
        )

        backend.als.assert_awaited_once_with("/")
        await_args = handler.await_args
        assert await_args is not None
        request = await_args.args[0]
        assert request.tool_call["args"]["limit"] == _REPOSITORY_READ_LINE_LIMIT

    async def test_async_rejects_large_file(self) -> None:
        backend = self._backend(
            entries=[
                {
                    "path": "/src.py",
                    "is_dir": False,
                    "size": _REPOSITORY_READ_BYTE_LIMIT + 1,
                }
            ]
        )
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="big"),
            handler,
        )

        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert "size limit" in result.text

    async def test_async_rejects_large_directory(self) -> None:
        backend = self._backend(
            entries=[
                {"path": f"/{index}", "is_dir": False}
                for index in range(_REPOSITORY_DIRECTORY_ENTRY_LIMIT + 1)
            ]
        )
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="ls", name="ls", path="/"
            ),
            handler,
        )

        handler.assert_not_awaited()
        assert isinstance(result, ToolMessage)
        assert "listing limit" in result.text

    async def test_async_truncates_large_result(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock(
            return_value=ToolMessage(
                content="x" * (_REPOSITORY_TOOL_RESULT_LIMIT + 500),
                tool_call_id="r",
            )
        )

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="r"),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert len(result.text) <= _REPOSITORY_TOOL_RESULT_LIMIT

    async def test_async_clamps_grep_matches(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="g"))

        await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="g", name="grep", path="/"
            ),
            handler,
        )

        await_args = handler.await_args
        assert await_args is not None
        request = await_args.args[0]
        assert request.tool_call["args"]["max_count"] == _REPOSITORY_GREP_MATCH_LIMIT

    async def test_async_bounds_glob_matches(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        paths = [f"/{index}.py" for index in range(_REPOSITORY_GLOB_MATCH_LIMIT + 1)]
        handler = AsyncMock(
            return_value=ToolMessage(content=str(paths), tool_call_id="glob")
        )

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="glob", name="glob", path="/"
            ),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert "Glob results limited" in result.text

    async def test_async_budget_exhaustion_is_reported(self) -> None:
        backend = self._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="r"))
        for index in range(_REPOSITORY_TOOL_CALL_LIMIT):
            await middleware.awrap_tool_call(
                TestRepositoryToolBudgetMiddleware._request(call_id=str(index)),
                handler,
            )

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="over"),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert "context limit reached" in result.text


class TestGoalContextFallbackDoubleFailure:
    """When the goal-only retry also fails, the original error is surfaced."""

    def test_sync_double_failure_raises_original(self) -> None:
        middleware = _GoalContextFallbackMiddleware()
        request = MagicMock()
        request.override.return_value = MagicMock()
        handler = MagicMock(
            side_effect=[RuntimeError("invalid api key"), ValueError("second")]
        )

        with pytest.raises(RuntimeError, match="invalid api key"):
            middleware.wrap_model_call(request, handler)

    async def test_async_double_failure_raises_original(self) -> None:
        middleware = _GoalContextFallbackMiddleware()
        request = MagicMock()
        request.override.return_value = MagicMock()
        handler = AsyncMock(
            side_effect=[RuntimeError("invalid api key"), ValueError("second")]
        )

        with pytest.raises(RuntimeError, match="invalid api key"):
            await middleware.awrap_model_call(request, handler)


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

    def test_recursion_failure_falls_back_to_goal_only(self) -> None:
        criteria = MagicMock()
        criteria.invoke.side_effect = GraphRecursionError("Recursion limit reached")
        fallback = self._fallback()
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        update = middleware.before_agent(
            self._state(), TestGoalCriteriaMiddleware._runtime()
        )

        assert update is not None
        assert update["_pending_goal_rubric"] == "- goal-only criteria"
        fallback.invoke.assert_called_once()

    def test_empty_proposal_falls_back_to_goal_only(self) -> None:
        criteria = MagicMock()
        criteria.invoke.return_value = {"messages": []}
        fallback = self._fallback("- salvaged criteria")
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        update = middleware.before_agent(
            self._state(), TestGoalCriteriaMiddleware._runtime()
        )

        assert update is not None
        assert update["_pending_goal_rubric"] == "- salvaged criteria"
        fallback.invoke.assert_called_once()

    def test_size_rejection_is_never_swallowed_by_the_fallback(self) -> None:
        """A size rejection is deterministic, so the fallback cannot help.

        `GoalStateSizeError` is a `ValueError`, which `_CRITERIA_FALLBACK_ERRORS`
        would otherwise catch — logging a character-limit problem as a context
        fault and re-asking with less context, which fails the same way while
        the limit message never reaches the caller.
        """
        criteria = MagicMock()
        criteria.invoke.side_effect = GoalStateSizeError(
            label="Goal objective and criteria combined",
            actual=12_500,
            limit=12_000,
        )
        fallback = self._fallback()
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        with pytest.raises(GoalStateSizeError, match="Remove at least 500"):
            middleware.before_agent(
                self._state(), TestGoalCriteriaMiddleware._runtime()
            )
        fallback.invoke.assert_not_called()

    async def test_async_size_rejection_is_never_swallowed_by_the_fallback(
        self,
    ) -> None:
        criteria = MagicMock()
        criteria.ainvoke = AsyncMock(
            side_effect=GoalStateSizeError(
                label="Goal objective and criteria combined",
                actual=12_500,
                limit=12_000,
            )
        )
        fallback = self._fallback()
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        with pytest.raises(GoalStateSizeError, match="Remove at least 500"):
            await middleware.abefore_agent(
                self._state(), TestGoalCriteriaMiddleware._runtime()
            )
        fallback.ainvoke.assert_not_awaited()

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

    def test_failure_without_fallback_agent_surfaces(self) -> None:
        criteria = MagicMock()
        criteria.invoke.side_effect = GraphRecursionError("Recursion limit reached")
        middleware = GoalCriteriaMiddleware(criteria)

        with pytest.raises(GraphRecursionError):
            middleware.before_agent(
                self._state(), TestGoalCriteriaMiddleware._runtime()
            )

    async def test_async_recursion_failure_falls_back_to_goal_only(self) -> None:
        criteria = MagicMock()
        criteria.ainvoke = AsyncMock(
            side_effect=GraphRecursionError("Recursion limit reached")
        )
        fallback = self._fallback("- async goal-only")
        middleware = GoalCriteriaMiddleware(criteria, fallback)

        update = await middleware.abefore_agent(
            self._state(), TestGoalCriteriaMiddleware._runtime()
        )

        assert update is not None
        assert update["_pending_goal_rubric"] == "- async goal-only"
        fallback.ainvoke.assert_awaited_once()

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

    def test_structured_output_validation_retries_oversized_criteria(self) -> None:
        """The nested model self-corrects before an error reaches the user."""
        agent = create_goal_criteria_fallback_agent(
            model=_OversizedThenValidCriteriaModel()
        )

        result = agent.invoke(
            {
                "messages": [HumanMessage(content="ship it")],
                "criteria_objective": "ship it",
                "criteria_operation_id": "retry-op",
            },
            context={},
        )

        assert _proposal_from_result(result) == ("ship it", "- concise result")
        messages = result["messages"]
        assert any(
            isinstance(message, ToolMessage) and "12000" in message.text
            for message in messages
        )

    def test_goal_proposal_schema_rejects_oversized_combination(self) -> None:
        """The structured schema reports pair limits through ToolStrategy."""
        with pytest.raises(ValueError, match="combined"):
            GoalProposal.model_validate(
                {
                    "objective": "goal",
                    "criteria": "x" * RUBRIC_CHAR_LIMIT,
                }
            )

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

    def test_combined_budget_rejection_ends_the_turn(self) -> None:
        """The limit message must reach the user, not a retry loop.

        The model is never told the combined budget, so retrying it is blind. The
        old default retried until `GraphRecursionError`, which the fallback tuple
        logged as a context fault, spent the fallback agent on the same request,
        and finally reported as "could not generate acceptance criteria" — the
        character count the user has to act on never arrived.
        """
        rejection = self._rejection("o" * 7_000, "c" * 6_000)

        with pytest.raises(GoalStateSizeError, match="Remove at least 1,000"):
            _raise_terminal_goal_state_size_error(rejection)

    def test_combined_budget_rejection_names_the_real_excess(self) -> None:
        """The error is rebuilt from the arguments, so its numbers are the real ones."""
        rejection = self._rejection("o" * 7_000, "c" * 6_000)

        with pytest.raises(GoalStateSizeError) as caught:
            _raise_terminal_goal_state_size_error(rejection)

        assert caught.value.label == "Goal objective and criteria combined"
        assert caught.value.actual == 13_000
        assert caught.value.limit == GOAL_APPLICATION_CHAR_LIMIT

    def test_oversized_single_field_still_retries(self) -> None:
        """A field over its own `max_length` is feedback the model can act on.

        The schema publishes both per-field limits, and shortening the overlong
        field usually brings the combined total inside its budget too — so this
        must stay a retry even though the combined total also overshoots here.
        """
        rejection = self._rejection("ship it", "x" * (RUBRIC_CHAR_LIMIT + 1))

        assert "Please fix your mistakes" in _raise_terminal_goal_state_size_error(
            rejection
        )

    def test_oversized_objective_still_retries(self) -> None:
        """Symmetric to the criteria case: the objective limit is in the schema."""
        rejection = self._rejection("o" * (GOAL_OBJECTIVE_CHAR_LIMIT + 1), "- ok")

        assert "Please fix your mistakes" in _raise_terminal_goal_state_size_error(
            rejection
        )

    def test_whitespace_only_field_still_retries(self) -> None:
        """Whitespace-only output is a model mistake with no user action attached."""
        rejection = self._rejection("   ", "- ok")

        assert "Please fix your mistakes" in _raise_terminal_goal_state_size_error(
            rejection
        )

    def test_error_without_tool_call_arguments_still_retries(self) -> None:
        """An error carrying no proposal must not crash the handler."""
        assert "Please fix your mistakes" in _raise_terminal_goal_state_size_error(
            ValueError("malformed tool call")
        )

    @pytest.mark.parametrize(
        "factory_kwargs",
        [{}, {"repository_backend": None, "context_tools": []}],
        ids=["fallback_agent", "context_agent"],
    )
    def test_both_criteria_agents_install_the_terminal_size_handler(
        self, factory_kwargs: dict[str, Any]
    ) -> None:
        """A missing handler silently restores the retry-to-exhaustion behavior."""
        factory = (
            create_goal_criteria_agent
            if factory_kwargs
            else create_goal_criteria_fallback_agent
        )
        graph = MagicMock()
        graph.with_config.return_value = "configured-graph"

        with patch("langchain.agents.create_agent", return_value=graph) as create:
            factory(model=MagicMock(), **factory_kwargs)

        response_format = create.call_args.kwargs["response_format"]
        assert response_format.schema is GoalProposal
        assert response_format.handle_errors is _raise_terminal_goal_state_size_error

    @pytest.mark.parametrize("field", ["objective", "criteria"])
    def test_goal_proposal_schema_rejects_whitespace_only_field(
        self, field: str
    ) -> None:
        """Whitespace-only structured output becomes retry feedback, not a goal."""
        payload = {"objective": "goal", "criteria": "- tests pass"}
        payload[field] = "   \n\t "

        with pytest.raises(ValueError, match="non-whitespace"):
            GoalProposal.model_validate(payload)

    def test_goal_proposal_schema_forbids_extra_keys(self) -> None:
        """Unknown keys fail rather than being silently dropped downstream."""
        with pytest.raises(ValueError, match="extra"):
            GoalProposal.model_validate(
                {
                    "objective": "goal",
                    "criteria": "- tests pass",
                    "unexpected": "value",
                }
            )

    def test_goal_proposal_is_frozen_after_validation(self) -> None:
        """Validated text cannot be swapped for oversized text after the fact.

        Pydantic does not validate assignment by default, so without `frozen`
        a plain attribute write would bypass `_fit_notice_budget`.
        """
        proposal = GoalProposal.model_validate(
            {"objective": "goal", "criteria": "- tests pass"}
        )

        with pytest.raises(ValidationError):
            proposal.criteria = "x" * (RUBRIC_CHAR_LIMIT + 1)  # ty: ignore[invalid-assignment]

    def test_fallback_agent_can_be_created(self) -> None:
        agent = create_goal_criteria_fallback_agent(
            model=GoalCriteriaIntegrationChatModel()
        )

        assert agent is not None


class TestPreflightBackendErrors:
    """Backend faults during preflight degrade to a bounded, logged error."""

    def test_sync_backend_error_is_treated_as_unavailable(self) -> None:
        backend = MagicMock()
        backend.ls.side_effect = RuntimeError("backend outage")
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock()

        result = middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="err"),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_called()

    async def test_async_backend_error_is_treated_as_unavailable(self) -> None:
        backend = MagicMock()
        backend.als = AsyncMock(side_effect=OSError("backend outage"))
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = AsyncMock()

        result = await middleware.awrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="err"),
            handler,
        )

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        handler.assert_not_awaited()

    def test_malformed_entry_missing_path_does_not_crash(self) -> None:
        backend = MagicMock()
        # This intentionally violates FileInfo to exercise defensive parsing.
        malformed_entry = cast("FileInfo", {"is_dir": False, "size": 5})
        backend.ls.return_value = LsResult(entries=[malformed_entry])
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="read"))

        result = middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="read"),
            handler,
        )

        # No size known for the target => read proceeds to the handler.
        handler.assert_called_once()
        assert isinstance(result, ToolMessage)
        assert result.status != "error"


class TestRepositoryBudgetEdgeCases:
    """Read/grep argument clamps and the per-operation budget cache are bounded."""

    @pytest.mark.parametrize(
        ("limit", "expected"),
        [(True, _REPOSITORY_READ_LINE_LIMIT), (-5, 1), (0, 1)],
    )
    def test_read_limit_is_clamped(self, limit: object, expected: int) -> None:
        backend = TestRepositoryToolBudgetMiddleware._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="r"))

        middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(call_id="r", limit=limit),
            handler,
        )

        request = handler.call_args.args[0]
        assert request.tool_call["args"]["limit"] == expected

    @pytest.mark.parametrize("count", [True, -5, 0, "x"])
    def test_grep_max_count_resets_to_default(self, count: object) -> None:
        backend = TestRepositoryToolBudgetMiddleware._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="g"))

        middleware.wrap_tool_call(
            TestRepositoryToolBudgetMiddleware._request(
                call_id="g", name="grep", max_count=count
            ),
            handler,
        )

        request = handler.call_args.args[0]
        assert request.tool_call["args"]["max_count"] == _REPOSITORY_GREP_MATCH_LIMIT

    def test_operation_budget_cache_is_bounded(self) -> None:
        backend = TestRepositoryToolBudgetMiddleware._backend()
        middleware = _RepositoryToolBudgetMiddleware(backend)

        for index in range(_REPOSITORY_OPERATION_BUDGET_CACHE_LIMIT + 50):
            middleware._reserve_call(
                TestRepositoryToolBudgetMiddleware._request(
                    call_id=f"c{index}", operation_id=f"op-{index}"
                )
            )

        assert len(middleware._calls) <= _REPOSITORY_OPERATION_BUDGET_CACHE_LIMIT
