"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.protocol import FileDownloadResponse, WriteResult
from langchain.agents.middleware.types import ModelRequest
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.offload_middleware import (
    CLICompactionMiddleware,
    _ArchiveReadGuard,
    _runtime_model_config,
)
from deepagents_code.tool_display import format_tool_display

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.messages import AnyMessage


class TestHITLGating:
    """Test that compact_conversation HITL gating respects the constant."""

    def test_hitl_gating_when_enabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=True, tool should be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", True):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" in result

    def test_hitl_gating_when_disabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=False, tool should NOT be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", False):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" not in result


class TestDisplayFormatting:
    """Test tool display formatting for compact_conversation."""

    def test_display_formatting(self) -> None:
        """format_tool_display should return the expected string."""
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result


class TestArchiveReadGuard:
    """Cover fail-closed archive writes after backend read errors."""

    def test_sync_error_response_blocks_write(self) -> None:
        """A synchronous error response must not permit a truncating write."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="permission_denied",
        )
        backend = MagicMock()
        backend.download_files.return_value = [response]
        backend.write.return_value = WriteResult(path=response.path)
        guard = _ArchiveReadGuard(backend)

        assert guard.download_files([response.path]) == [response]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            guard.write(response.path, "new history")

        backend.write.assert_not_called()

    async def test_async_error_response_blocks_write(self) -> None:
        """An asynchronous error response must not permit a truncating write."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="transient backend error",
        )
        backend = MagicMock()
        backend.adownload_files = AsyncMock(return_value=[response])
        backend.awrite = AsyncMock(return_value=WriteResult(path=response.path))
        guard = _ArchiveReadGuard(backend)

        assert await guard.adownload_files([response.path]) == [response]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            await guard.awrite(response.path, "new history")

        backend.awrite.assert_not_awaited()

    def test_missing_archive_allows_create(self) -> None:
        """A missing archive remains the expected first-write path."""
        response = FileDownloadResponse(
            path="/conversation_history/thread.md",
            error="file_not_found",
        )
        backend = MagicMock()
        backend.download_files.return_value = [response]
        expected = WriteResult(path=response.path)
        backend.write.return_value = expected
        guard = _ArchiveReadGuard(backend)

        assert guard.download_files([response.path]) == [response]
        assert guard.write(response.path, "new history") == expected


class TestCLICompactionMiddleware:
    """Cover dcode's explicit `/offload` behavior layered over the SDK tool."""

    @staticmethod
    def _summarization() -> MagicMock:
        summarization = MagicMock()
        backend = MagicMock()
        backend.adownload_files = AsyncMock(
            return_value=[
                FileDownloadResponse(
                    path="/conversation_history/thread.md",
                    error="file_not_found",
                )
            ]
        )
        summarization._backend = backend
        summarization._get_history_path.return_value = "/conversation_history/thread.md"
        summarization._apply_event_to_messages.side_effect = lambda messages, _event: (
            messages
        )
        summarization._determine_cutoff_index.return_value = 2
        summarization._partition_messages.side_effect = lambda messages, cutoff: (
            messages[:cutoff],
            messages[cutoff:],
        )
        summarization._acreate_summary = AsyncMock(return_value="Summary")
        summarization._aoffload_to_backend = AsyncMock(
            return_value="/conversation_history/thread.md"
        )
        summarization._build_new_messages_with_path.return_value = [
            HumanMessage(content="Summary")
        ]
        summarization._compute_state_cutoff.return_value = 2
        return summarization

    @pytest.mark.parametrize("is_async", [False, True])
    @pytest.mark.parametrize("overflow", [False, True])
    async def test_auto_compaction_runs_precompact_hook(
        self, overflow: bool, is_async: bool
    ) -> None:
        """Every automatic compaction must ask `PreCompact` before summarizing.

        `wrap_model_call` and `awrap_model_call` are separately written parallel
        implementations, so both are driven here to keep the two from drifting.
        """
        from deepagents.middleware.summarization import SummarizationMiddleware

        from deepagents_code.hooks.models.domain import (
            HookEvent,
            PreCompactDecision,
        )

        messages: list[AnyMessage] = [HumanMessage("one"), HumanMessage("two")]
        summarization = self._summarization()
        summarization._get_effective_messages.return_value = messages
        summarization._count_tokens.return_value = 2
        summarization._truncate_args.return_value = (messages, False)
        summarization._should_summarize.return_value = not overflow
        summarization._determine_cutoff_index.return_value = 1
        wrapper_name = "awrap_model_call" if is_async else "wrap_model_call"
        if overflow:
            setattr(
                summarization,
                wrapper_name,
                MethodType(
                    getattr(SummarizationMiddleware, wrapper_name), summarization
                ),
            )

        middleware = CLICompactionMiddleware(summarization)
        request: ModelRequest[None] = ModelRequest(
            model=MagicMock(),
            messages=messages,
            state={"messages": messages},
            runtime=Runtime(context=None),
        )
        error = ContextOverflowError("too large") if overflow else None
        handler = (
            AsyncMock(side_effect=error) if is_async else MagicMock(side_effect=error)
        )
        invoke = MagicMock(
            return_value=PreCompactDecision(
                event=HookEvent.PRE_COMPACT,
                continue_processing=False,
                stop_reason="preserve context",
            )
        )

        with (
            patch(
                "deepagents_code.offload_middleware._event_enabled", return_value=True
            ),
            patch("deepagents_code.offload_middleware._invoke_hook", invoke),
        ):

            async def run_middleware() -> None:
                if is_async:
                    await middleware.awrap_model_call(request, handler)
                else:
                    middleware.wrap_model_call(request, handler)

            if overflow:
                with pytest.raises(ContextOverflowError, match="too large"):
                    await run_middleware()
            else:
                await run_middleware()

        event = invoke.call_args.args[1]
        assert event.trigger.value == "auto"
        logical_id = invoke.call_args.kwargs["logical_event_id"]
        assert logical_id == middleware._auto_compaction_id(request)
        assert logical_id != middleware._auto_compaction_id(
            request.override(messages=[*messages, HumanMessage("three")])
        )
        invoke.assert_called_once()
        if is_async:
            handler.assert_awaited_once()
            summarization._aoffload_to_backend.assert_not_awaited()
            summarization._acreate_summary.assert_not_awaited()
        else:
            handler.assert_called_once()
            summarization._offload_to_backend.assert_not_called()
            summarization._create_summary.assert_not_called()

    async def test_operation_path_writes_through_the_archive_guard(self) -> None:
        """The server `/offload` operation's write path has the same invariant.

        The guard is applied per write site rather than by the backend's type, so
        the server operation entry point does not inherit it from the tool paths
        — it has to apply it itself, and nothing but a test says so.
        """
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        write_backend = summarization._aoffload_to_backend.await_args.args[0]
        assert isinstance(write_backend, _ArchiveReadGuard)
        assert write_backend._backend is summarization._backend

    async def test_operation_plan_defers_archive_until_checkpoint_reservation(
        self,
    ) -> None:
        """Planning may spend on a summary but cannot mutate archive storage."""
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        plan = await middleware._aplan_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        assert plan is not None
        assert plan.update(None)["_summarization_event"]["file_path"] is None
        summarization._aoffload_to_backend.assert_not_awaited()

    async def test_operation_path_returns_an_absolute_cutoff(self) -> None:
        """The committed event must carry the absolute cutoff, not the relative one.

        `_determine_cutoff_index` is relative to the *effective* conversation
        (post-previous-summary), while the persisted `cutoff_index` indexes the
        full message list — `_compute_state_cutoff` converts between them. The
        two coincide on a thread's first `/offload`, so returning the relative
        value passes every other test here and only corrupts the *second*
        `/offload`, which reads this back as its base.
        """
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 2
        summarization._compute_state_cutoff.return_value = 9
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        prior = {"cutoff_index": 7, "summary_message": None, "file_path": None}

        result = await middleware.arun_forced_compaction_update(
            cast(
                "Any",
                {
                    "messages": [HumanMessage("one"), HumanMessage("two")],
                    "_summarization_event": prior,
                },
            ),
            runtime,
        )

        assert result is not None
        event = result["_summarization_event"]
        summarization._compute_state_cutoff.assert_called_once_with(prior, 2)
        assert event["cutoff_index"] == 9
        assert event["file_path"] == "/conversation_history/thread.md"
        assert isinstance(event["summary_message"], HumanMessage)

    async def test_operation_path_threads_and_persists_the_session_id(self) -> None:
        """`/offload` must reuse and re-commit the SDK's archive-file id.

        The SDK's `_offload_to_backend` names the archive by `session_id`, and
        the committed `_summarization_session_id` is what makes a later
        compaction append to the same file instead of starting a new one. The
        server operation bypasses the SDK's own state update, so it has to
        thread the id through and write it back itself.
        """
        summarization = self._summarization()
        summarization._get_session_id.return_value = "session_abc"
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        result = await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        assert result is not None
        assert result["_summarization_session_id"] == "session_abc"
        assert summarization._aoffload_to_backend.await_args.args[2] == "session_abc"

    async def test_operation_path_refuses_a_chained_no_advance_compaction(
        self,
    ) -> None:
        """A compaction that would not advance the cutoff must not commit.

        The degenerate chained case: everything eligible already sits behind the
        prior event, so the only thing left to summarize is the previous summary
        itself. `_compute_state_cutoff` returns the prior absolute cutoff
        unchanged, and the client — which keys its report on that value moving —
        reports "nothing to offload". Committing anyway would spend a model
        call, replace the in-context summary with a summary-of-a-summary, and
        drop the prior archive's `file_path`, all while telling the user nothing
        happened. Stop before the model call so the report and the state agree.
        """
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 1
        summarization._compute_state_cutoff.return_value = 7
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        prior = {
            "cutoff_index": 7,
            "summary_message": None,
            "file_path": "/conversation_history/thread.md",
        }

        result = await middleware.arun_forced_compaction_update(
            cast(
                "Any",
                {
                    "messages": [HumanMessage("summary"), HumanMessage("recent")],
                    "_summarization_event": prior,
                },
            ),
            runtime,
        )

        assert result is None
        # Neither the billable step nor the archive write may happen.
        summarization._acreate_summary.assert_not_awaited()
        summarization._aoffload_to_backend.assert_not_awaited()

    async def test_operation_path_rejects_an_empty_conversation(self) -> None:
        """An empty `messages` must raise rather than report a clean no-op.

        The server operation normally handles an empty thread before invoking
        compaction. A direct caller that bypasses that service check still gets
        an explicit error instead of a misleading successful no-op.
        """
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        with pytest.raises(ValueError, match="checkpointed conversation"):
            await middleware.arun_forced_compaction_update(
                cast("Any", {"messages": [], "_summarization_event": None}), runtime
            )

        summarization._acreate_summary.assert_not_awaited()

    async def test_operation_path_logs_a_failed_archive_write(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A `None` archive path must leave a trace naming this call site.

        `_aoffload_to_backend` catches every write failure and returns `None` —
        including `_ArchiveReadGuard`'s deliberate "refusing to overwrite
        existing history" `RuntimeError`. The compaction still commits (the
        client reports the missing archive to the user), but without this the
        only server-side record is a warning inside the SDK that names neither
        the thread nor `/offload`.
        """
        summarization = self._summarization()
        summarization._aoffload_to_backend = AsyncMock(return_value=None)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        with caplog.at_level("ERROR"):
            result = await middleware.arun_forced_compaction_update(
                {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
            )

        assert result is not None
        assert result["_summarization_event"]["file_path"] is None
        assert "archive write failed" in caplog.text

    async def test_operation_path_returns_none_when_nothing_to_compact(self) -> None:
        """A cutoff of 0 must be `None`, not an event pinning cutoff 0.

        The caller distinguishes "nothing old enough" from a real compaction by
        this return value; an empty-but-present event would advance nothing while
        still reading as success.
        """
        summarization = self._summarization()
        summarization._determine_cutoff_index = MagicMock(return_value=0)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        result = await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one")]}, runtime
        )

        assert result is None
        summarization._aoffload_to_backend.assert_not_awaited()

    def test_runtime_model_builds_matching_summarizer(self) -> None:
        """A `/model` override selects the summarizer used by `/offload`."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:active-model",
            "model_params": {"temperature": 0},
        }
        active_model = object()
        result = SimpleNamespace(model=active_model)
        selected = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model", return_value=result
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        create_model.assert_called_once_with(
            "provider:active-model",
            extra_kwargs={"temperature": 0},
            profile_overrides=None,
        )
        create_summarization.assert_called_once()
        assert create_summarization.call_args.args[0] is active_model
        # The summarizer gets the composite backend itself, not the
        # `_ArchiveReadGuard` wrapper: it reads `artifacts_root` to prefix the
        # archive path, and the guard exposes no such attribute. The server
        # operation applies the guard separately at the write site.
        assert create_summarization.call_args.args[1] is startup._backend

    def test_runtime_profile_overrides_and_context_limit_are_applied(self) -> None:
        """Server-side offload uses the CLI's effective model profile."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:active-model",
            "model_params": {},
            "profile_overrides": {"max_input_tokens": 32_000},
            "model_context_limit": 24_000,
        }
        active_model = SimpleNamespace(profile={"max_input_tokens": 200_000})
        result = SimpleNamespace(model=active_model)
        selected = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model", return_value=result
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        create_model.assert_called_once_with(
            "provider:active-model",
            extra_kwargs=None,
            profile_overrides={"max_input_tokens": 32_000},
        )
        assert active_model.profile["max_input_tokens"] == 24_000
        create_summarization.assert_called_once()
        assert create_summarization.call_args.args[0] is active_model
        assert create_summarization.call_args.args[1] is startup._backend

    def test_model_initiated_tool_delegates_to_gated_path(self) -> None:
        """The public tool keeps using the SDK's eligibility-gated sync path."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()

        with patch.object(middleware, "_run_compact", return_value="gated") as gated:
            assert tool.func(runtime) == "gated"

        gated.assert_called_once_with(runtime)

    async def test_model_initiated_tool_delegates_to_gated_path_async(self) -> None:
        """The public tool keeps using the SDK's eligibility-gated async path."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()

        with patch.object(
            middleware,
            "_arun_compact",
            new=AsyncMock(return_value="gated"),
        ) as gated:
            assert await tool.coroutine(runtime) == "gated"

        gated.assert_awaited_once_with(runtime)

    async def test_operation_read_failure_never_truncates_archive(self) -> None:
        """A transient archive read failure blocks the server operation's write."""
        from deepagents.middleware.summarization import SummarizationMiddleware

        summarization = self._summarization()
        backend = MagicMock()
        backend.adownload_files = AsyncMock(side_effect=RuntimeError("read failed"))
        backend.awrite = AsyncMock()
        backend.aedit = AsyncMock()
        summarization._backend = backend
        summarization._get_history_path.return_value = "/conversation_history/thread.md"
        summarization._filter_summary_messages.side_effect = lambda messages: messages

        async def sdk_offload(
            guarded: BackendProtocol, messages: list[AnyMessage], session_id: str
        ) -> str | None:
            return await SummarizationMiddleware._aoffload_to_backend(
                summarization, guarded, messages, session_id
            )

        summarization._aoffload_to_backend = AsyncMock(side_effect=sdk_offload)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None

        result = await middleware.arun_forced_compaction_update(
            {"messages": [HumanMessage("one"), HumanMessage("two")]}, runtime
        )

        backend.awrite.assert_not_awaited()
        backend.aedit.assert_not_awaited()
        assert result is not None
        assert result["_summarization_event"]["file_path"] is None

    def test_factory_builds_cli_middleware_threading_system_prompt(self) -> None:
        """The factory returns a CLI middleware carrying the SDK's config."""
        from deepagents_code import offload_middleware as om

        sdk = MagicMock()
        sdk._summarization = MagicMock()
        sdk._summarization.name = "SummarizationMiddleware"
        sdk.system_prompt = "SYSTEM PROMPT"
        backend: Any = object()
        with patch.object(
            om, "create_summarization_tool_middleware", return_value=sdk
        ) as factory:
            result = om._create_cli_compaction_middleware("provider:model", backend)

        factory.assert_called_once()
        assert isinstance(result, om.CLICompactionMiddleware)
        assert result.name == "SummarizationMiddleware"
        assert result.system_prompt == "SYSTEM PROMPT"
        assert result._summarization is sdk._summarization


class TestRuntimeModelConfig:
    """Cover the three context shapes `_runtime_model_config` accepts."""

    @staticmethod
    def _runtime(context: object) -> MagicMock:
        runtime = MagicMock()
        runtime.context = context
        return runtime

    def test_schema_instance(self) -> None:
        ctx = CLIContextSchema(model="p:m", model_params={"temperature": 0})
        assert _runtime_model_config(self._runtime(ctx)) == (
            "p:m",
            {"temperature": 0},
            {},
            None,
        )

    def test_serialized_dict(self) -> None:
        ctx = {"model": "p:m2", "model_params": {"x": 1}}
        assert _runtime_model_config(self._runtime(ctx)) == (
            "p:m2",
            {"x": 1},
            {},
            None,
        )

    def test_dict_with_bad_types_normalizes(self) -> None:
        ctx = {"model": 123, "model_params": None}
        assert _runtime_model_config(self._runtime(ctx)) == (None, {}, {}, None)

    def test_unknown_shape(self) -> None:
        assert _runtime_model_config(self._runtime(object())) == (None, {}, {}, None)

    def test_named_fields_disambiguate_the_two_dict_slots(self) -> None:
        """The two `dict` slots are addressable by name, not just position.

        `model_params` and `profile_overrides` are structurally identical, so a
        positional swap would be invisible; named-field access pins each to the
        right source value.
        """
        ctx = CLIContextSchema(
            model="p:m",
            model_params={"temperature": 0},
            profile_overrides={"max_input_tokens": 99},
            model_context_limit=7,
        )
        config = _runtime_model_config(self._runtime(ctx))
        assert config.model_params == {"temperature": 0}
        assert config.profile_overrides == {"max_input_tokens": 99}
        assert config.context_limit == 7


class TestSdkContractGuards:
    """Guard summarization-event assumptions shared with the SDK."""

    def test_summarization_cutoff_is_an_absolute_index(self) -> None:
        """`cutoff_index` must index unfiltered persisted messages.

        `goal_state_notice.validated_summarization_cutoff` and
        `GoalToolsMiddleware._request_with_goal_notice` both depend on this: the
        goal middleware wraps the summarizer, so anything it removes from the
        request below the cutoff shifts the indices this slice uses. If the SDK
        ever stored an effective-list index instead, the bounds check would keep
        returning a plausible integer and the notice logic would degrade
        silently rather than fail.
        """
        from deepagents.middleware.summarization import SummarizationMiddleware

        messages = ["m0", "m1", "m2", "m3"]
        event = {"summary_message": "S", "cutoff_index": 2}

        applied = SummarizationMiddleware._apply_event_to_messages(
            messages,  # ty: ignore[invalid-argument-type]
            event,  # ty: ignore[invalid-argument-type]
        )

        assert applied == ["S", "m2", "m3"]
