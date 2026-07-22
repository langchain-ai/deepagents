"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagents.backends.protocol import FileDownloadResponse, WriteResult
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.offload_middleware import (
    COMPACTION_FAILURE_PREFIX,
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
        summarization._backend = object()
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

    async def test_force_bypasses_sdk_eligibility_gate(self) -> None:
        """Forced compaction partitions directly even below the proactive gate."""
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        summarization._is_eligible_for_compaction.assert_not_called()
        summarization._acreate_summary.assert_awaited_once()
        assert result.update is not None
        assert result.update["_summarization_event"]["cutoff_index"] == 2

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
        guarded_backend = create_summarization.call_args.args[1]
        assert guarded_backend._backend is startup._backend

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
        guarded_backend = create_summarization.call_args.args[1]
        assert guarded_backend._backend is startup._backend

    async def test_force_noops_when_nothing_old_enough(self) -> None:
        """Forced compaction still no-ops at cutoff 0 (bypasses only the gate)."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 0
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._acreate_summary.assert_not_awaited()
        assert "Nothing to compact" in result.update["messages"][0].content

    async def test_async_force_excludes_seed_from_retention_cutoff(self) -> None:
        """The async cutoff is calculated from the pre-seed conversation."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.side_effect = lambda messages: (
            0 if len(messages) == 6 else 1
        )
        middleware = CLICompactionMiddleware(summarization)
        conversation = [HumanMessage(str(index)) for index in range(6)]
        seed = AIMessage(
            content="",
            id="offload-seed-tool-call",
            tool_calls=[
                {
                    "name": "compact_conversation",
                    "args": {"force": True},
                    "id": "tool-call",
                }
            ],
        )
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [*conversation, seed]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._determine_cutoff_index.assert_called_once_with(conversation)
        summarization._partition_messages.assert_not_called()

    def test_sync_force_excludes_serialized_seed_from_retention_cutoff(self) -> None:
        """The sync cutoff also ignores a serialized synthetic seed."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.side_effect = lambda messages: (
            0 if len(messages) == 6 else 1
        )
        middleware = CLICompactionMiddleware(summarization)
        conversation = [HumanMessage(str(index)) for index in range(6)]
        seed = {"id": "offload-seed-tool-call", "type": "ai", "content": ""}
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [*conversation, seed]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._determine_cutoff_index.assert_called_once_with(conversation)
        summarization._partition_messages.assert_not_called()

    async def test_forced_compact_error_when_summary_fails(self) -> None:
        """A summary failure returns the failure prefix and does not compact."""
        summarization = self._summarization()
        summarization._acreate_summary = AsyncMock(side_effect=RuntimeError("boom"))
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        # The failure must NOT persist an event, and must carry the stable
        # prefix the `/offload` client keys on.
        assert result.update is not None
        assert "_summarization_event" not in result.update
        content = result.update["messages"][0].content
        assert content.startswith(COMPACTION_FAILURE_PREFIX)
        assert "RuntimeError" in content

    def test_sync_forced_compact_compacts(self) -> None:
        """The synchronous forced path mirrors the async one."""
        summarization = self._summarization()
        summarization._create_summary.return_value = "Summary"
        summarization._offload_to_backend.return_value = (
            "/conversation_history/thread.md"
        )
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        summarization._create_summary.assert_called_once()
        assert result.update is not None
        assert result.update["_summarization_event"]["cutoff_index"] == 2

    def test_force_is_hidden_from_model_schema(self) -> None:
        """`force` must not appear in the schema the model sees."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool = middleware.tools[0]
        # `tool_call_schema` is a pydantic model (or, rarely, a dict); either
        # way the model-facing property set must not expose `force`.
        schema: Any = tool.tool_call_schema
        props = (
            schema.get("properties", {})
            if isinstance(schema, dict)
            else schema.model_json_schema().get("properties", {})
        )
        assert "force" not in props

    def test_ordinary_context_delegates_to_gated_path(self) -> None:
        """Caller-supplied `force` cannot bypass the trusted runtime context."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()
        runtime.context = {}
        runtime.tool_call_id = "model-call"
        with (
            patch.object(middleware, "_run_compact", return_value="gated") as gated,
            patch.object(
                middleware, "_run_forced_compact", return_value="forced"
            ) as forced,
        ):
            assert tool.func(runtime, force=False) == "gated"
            assert tool.func(runtime, force=True) == "gated"
            assert gated.call_count == 2
            forced.assert_not_called()

    async def test_offload_context_delegates_to_forced_path_async(self) -> None:
        """The authorized call ID in runtime context selects forced mode."""
        middleware = CLICompactionMiddleware(self._summarization())
        tool: Any = middleware.tools[0]
        runtime = MagicMock()
        runtime.context = {"offload_tool_call_id": "offload-call"}
        runtime.tool_call_id = "offload-call"
        with (
            patch.object(
                middleware,
                "_arun_compact",
                new_callable=AsyncMock,
                return_value="gated",
            ) as gated,
            patch.object(
                middleware,
                "_arun_forced_compact",
                new_callable=AsyncMock,
                return_value="forced",
            ) as forced,
        ):
            # ToolNode replaces the seeded `force=True` with this default.
            assert await tool.coroutine(runtime, force=False) == "forced"
            gated.assert_not_awaited()
            forced.assert_awaited_once_with(runtime)

    async def test_tool_node_preserves_forced_mode_via_runtime_context(self) -> None:
        """A real ToolNode strips `force` but still reaches forced compaction."""
        from langchain_core.messages import ToolMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ToolNode
        from langgraph.types import Command
        from typing_extensions import TypedDict

        class ToolState(TypedDict):
            messages: list[object]

        middleware = CLICompactionMiddleware(self._summarization())
        # LangGraph accepts these runtime schemas, but its generic bound is not
        # recognized by ty on Python 3.14.
        builder = StateGraph(
            ToolState,  # ty: ignore[invalid-argument-type]
            context_schema=CLIContextSchema,
        )
        builder.add_node("tools", ToolNode(middleware.tools))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        tool_call_id = "offload-call"
        seed = AIMessage(
            content="",
            id=f"offload-seed-{tool_call_id}",
            tool_calls=[
                {
                    "name": "compact_conversation",
                    "args": {"force": True},
                    "id": tool_call_id,
                }
            ],
        )
        command = Command(
            update={
                "messages": [
                    ToolMessage(content="compacted", tool_call_id=tool_call_id)
                ]
            }
        )

        with (
            patch.object(
                middleware,
                "_arun_compact",
                new_callable=AsyncMock,
                return_value=command,
            ) as gated,
            patch.object(
                middleware,
                "_arun_forced_compact",
                new_callable=AsyncMock,
                return_value=command,
            ) as forced,
            warnings.catch_warnings(),
        ):
            warnings.filterwarnings(
                "error", message="Pydantic serializer warnings", category=UserWarning
            )
            await graph.ainvoke(
                ToolState(messages=[seed]),  # ty: ignore[invalid-argument-type]
                context=CLIContextSchema(  # ty: ignore[invalid-argument-type]
                    offload_tool_call_id=tool_call_id
                ),
            )

        gated.assert_not_awaited()
        forced.assert_awaited_once()

    async def test_read_failure_never_reaches_truncating_archive_write(self) -> None:
        """A transient archive read failure aborts the SDK write fallback."""
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
            guarded: BackendProtocol, messages: list[AnyMessage]
        ) -> str | None:
            return await SummarizationMiddleware._aoffload_to_backend(
                summarization, guarded, messages
            )

        summarization._aoffload_to_backend = AsyncMock(side_effect=sdk_offload)
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = {"offload_tool_call_id": "tool-call"}
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        backend.awrite.assert_not_awaited()
        backend.aedit.assert_not_awaited()
        assert result.update is not None
        assert result.update["_summarization_event"]["file_path"] is None

    def test_sync_forced_compact_noops_when_nothing_old_enough(self) -> None:
        """The sync forced path also no-ops at cutoff 0 (mirrors the async one)."""
        summarization = self._summarization()
        summarization._determine_cutoff_index.return_value = 0
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        summarization._create_summary.assert_not_called()
        assert "Nothing to compact" in result.update["messages"][0].content

    def test_sync_forced_compact_error_when_summary_fails(self) -> None:
        """A sync summary failure returns the failure prefix and does not compact."""
        summarization = self._summarization()
        summarization._create_summary = MagicMock(side_effect=RuntimeError("boom"))
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = middleware._run_forced_compact(runtime)

        assert result.update is not None
        assert "_summarization_event" not in result.update
        content = result.update["messages"][0].content
        assert content.startswith(COMPACTION_FAILURE_PREFIX)
        assert "RuntimeError" in content

    def test_forced_compact_error_starts_with_prefix(self) -> None:
        """The prefix position is the load-bearing failure-detection contract."""
        command = CLICompactionMiddleware._forced_compact_error(
            "call-1", RuntimeError("boom")
        )
        assert command.update is not None
        (message,) = command.update["messages"]
        assert message.content.startswith(COMPACTION_FAILURE_PREFIX)
        assert message.tool_call_id == "call-1"
        assert "RuntimeError" in message.content

    def test_factory_builds_cli_middleware_threading_system_prompt(self) -> None:
        """The factory returns a CLI middleware carrying the SDK's config."""
        from deepagents_code import offload_middleware as om

        sdk = MagicMock()
        sdk._summarization = MagicMock()
        sdk.system_prompt = "SYSTEM PROMPT"
        backend: Any = object()
        with patch.object(
            om, "create_summarization_tool_middleware", return_value=sdk
        ) as factory:
            result = om._create_cli_compaction_middleware("provider:model", backend)

        factory.assert_called_once()
        assert isinstance(result, om.CLICompactionMiddleware)
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
    """Guard the SDK seams the forced-compaction fork depends on.

    `CLICompactionMiddleware` forks the SDK's gated compaction flow and keys
    failure detection on a shared message prefix. These tests fail loudly in CI
    if a coordinated SDK bump renames a depended-on private method or changes
    the failure wording, instead of the fork silently drifting out of parity.
    """

    def test_forced_compact_matches_sdk_summarizer_calls(self) -> None:
        """Every SDK method the fork invokes must still exist."""
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            SummarizationToolMiddleware,
        )

        # Called on `self._summarization` (a SummarizationMiddleware).
        for name in (
            "_apply_event_to_messages",
            "_determine_cutoff_index",
            "_partition_messages",
            "_create_summary",
            "_acreate_summary",
            "_offload_to_backend",
            "_aoffload_to_backend",
        ):
            assert callable(getattr(SummarizationMiddleware, name, None)), name

        # Inherited SDK helpers called on the tool-middleware subclass.
        for name in (
            "_build_compact_result",
            "_nothing_to_compact",
        ):
            assert callable(getattr(SummarizationToolMiddleware, name, None)), name

    def test_failure_prefix_matches_sdk_failure_message(self) -> None:
        """Dcode's prefix must match the SDK's own compaction-failure wording.

        `/offload` detects failures from either path by this prefix, so the
        SDK's `_compact_error` message must keep starting with it.
        """
        from deepagents.middleware.summarization import SummarizationToolMiddleware

        command = SummarizationToolMiddleware._compact_error(
            "call-1", RuntimeError("boom")
        )
        assert command.update is not None
        (message,) = command.update["messages"]
        assert message.content.startswith(COMPACTION_FAILURE_PREFIX)
