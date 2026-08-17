"""Tests for the server-owned offload HTTP boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.offload_middleware import OffloadExecution


class TestOperationPayload:
    """Malformed client requests fail with a field-naming 422 at the boundary."""

    def test_valid_payload_passes_with_unknown_keys(self) -> None:
        from deepagents_code.offload_api import _operation_payload

        operation_id, context, responses = _operation_payload(
            {
                "operation_id": "op-1",
                "context": {
                    "model": "openai:gpt-5",
                    "model_params": {"temperature": 0.2},
                    "profile_overrides": {"max_input_tokens": 1000},
                    "model_context_limit": 32000,
                    "auto_approve": True,
                    "hooks_server_events": ["PreCompact"],
                    "thread_id": "thread-1",
                    "some_future_field": {"ignored": True},
                },
            }
        )

        assert operation_id == "op-1"
        assert context["model"] == "openai:gpt-5"
        assert context["some_future_field"] == {"ignored": True}
        assert responses == {}

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model", 123),
            ("classifier_model", ["openai:gpt-5"]),
            ("approval_mode", 1),
            ("thread_id", {"id": "t"}),
            ("hooks_snapshot_id", 0.5),
            ("prompt_id", True),
            ("model_params", "temperature=0.2"),
            ("profile_overrides", [("max_input_tokens", 1000)]),
            ("model_context_limit", "32000"),
            ("model_context_limit", True),
            ("auto_approve", "yes"),
            ("hooks_server_events", "PreCompact"),
            ("hooks_server_events", ["PreCompact", 42]),
        ],
    )
    def test_bad_context_field_names_the_field(self, field: str, value: object) -> None:
        from deepagents_code.offload_api import _operation_payload

        with pytest.raises(TypeError, match=f"context.{field}"):
            _operation_payload(
                {
                    "operation_id": "op-1",
                    "context": {field: value},
                }
            )

    def test_null_context_fields_pass(self) -> None:
        from deepagents_code.offload_api import _operation_payload

        _, context, _ = _operation_payload(
            {
                "operation_id": "op-1",
                "context": {
                    "model": None,
                    "model_params": None,
                    "model_context_limit": None,
                    "auto_approve": None,
                    "hooks_server_events": None,
                },
            }
        )
        assert context["model"] is None


def _thread_state(checkpoint_id: str = "checkpoint-1") -> dict[str, object]:
    """Build an idle LangGraph thread-state response."""
    return {
        "values": {
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "id": "message-1",
                }
            ],
            "_session_cost_usd": 1.0,
        },
        "next": [],
        "tasks": [],
        "interrupts": [],
        "checkpoint": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
            "checkpoint_map": {},
        },
    }


def _result() -> dict[str, object]:
    """Build a complete operation result."""
    return {
        "status": "compacted",
        "messages_offloaded": 1,
        "messages_kept": 1,
        "tokens_before": 20,
        "tokens_after": 10,
        "archive_path": "/conversation_history/thread-1.md",
        "archive_ephemeral": False,
        "error": None,
    }


class TestExecuteOffload:
    """The route owns state hydration, validation, and atomic persistence."""

    def test_hydrates_persisted_summary_message(self) -> None:
        """A subsequent offload receives a message object in its prior event."""
        from langchain_core.messages import HumanMessage

        from deepagents_code.offload_api import _hydrate_state

        state = _hydrate_state(
            {
                "messages": [
                    {"role": "user", "content": "new message", "id": "message-1"}
                ],
                "_summarization_event": {
                    "cutoff_index": 1,
                    "summary_message": {
                        "type": "human",
                        "content": "Prior summary.",
                        "id": "summary-1",
                    },
                },
            }
        )

        event = state["_summarization_event"]
        assert isinstance(event, dict)
        assert isinstance(event["summary_message"], HumanMessage)
        assert event["summary_message"].content == "Prior summary."

    async def test_commits_event_and_cost_without_messages(self) -> None:
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25}, rollback=MagicMock()
        )

        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        agent=SimpleNamespace(store=None), offload=operation
                    )
                ),
            ),
            patch.object(offload_api, "prepare_operation_cost", return_value=prepared),
        ):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={"model": "test:model"},
                hook_responses={},
            )

        assert response == {"status": "complete", "result": _result()}
        state = operation.execute.await_args.args[0]
        assert state["messages"][0].id == "message-1"
        runtime = operation.execute.await_args.args[1]
        assert runtime.context["thread_id"] == "thread-1"
        threads.update_state.assert_awaited_once()
        args = threads.update_state.await_args
        assert args.args[:2] == (
            "thread-1",
            {
                "_summarization_event": {"cutoff_index": 1},
                "_session_cost_usd": 0.25,
            },
        )
        assert "messages" not in args.args[1]
        assert "checkpoint" not in args.kwargs
        prepared.rollback.assert_not_called()

    async def test_checkpoint_change_fails_without_state_commit(self) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[_thread_state("before"), _thread_state("changed")]
            ),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )

        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        agent=SimpleNamespace(store=None), offload=operation
                    )
                ),
            ),
            patch.object(offload_api, "prepare_operation_cost") as prepare,
            pytest.raises(offload_api._OffloadConflictError, match="thread changed"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()
        prepare.assert_not_called()

    async def test_busy_thread_is_rejected_before_operation(self) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "busy"}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
            pytest.raises(offload_api._OffloadConflictError, match="active"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.get_state.assert_not_awaited()
        runtime.assert_not_awaited()
