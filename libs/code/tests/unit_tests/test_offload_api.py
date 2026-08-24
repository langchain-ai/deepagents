"""Tests for the server-owned offload HTTP boundary."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.offload_middleware import OffloadExecution, OffloadResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents_code.offload_middleware import _PendingArchive


@pytest.fixture(autouse=True)
def _reset_offload_globals() -> Iterator[None]:
    """Clear cached clients and operation state between tests.

    `offload_api` caches one `httpx`-backed client per process (building one per
    request leaks a connection pool). Tests patch `get_client`, so the cache has
    to be dropped or the first test's mock would serve every later one.
    """
    from deepagents_code import offload_api

    offload_api._client = None
    offload_api._active_operations.clear()
    offload_api._operation_outcomes.clear()
    try:
        yield
    finally:
        offload_api._client = None
        offload_api._active_operations.clear()
        offload_api._operation_outcomes.clear()


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

    @pytest.mark.parametrize(
        "key",
        [
            "base_url",
            "api_base",
            "openai_api_base",
            "anthropic_api_url",
            "azure_endpoint",
            "azure_openai_api_base",
            "api_endpoint",
            "openai_proxy",
            "anthropic_proxy",
            "proxy",
            "proxies",
            "http_client",
            "http_async_client",
            "transport",
            "default_headers",
            "custom_headers",
        ],
    )
    def test_transport_model_params_are_stripped(self, key: str) -> None:
        """The boundary drops endpoint/transport keys from `model_params`.

        These keys would route the server's credentialed provider calls to a
        client-chosen destination.
        """
        from deepagents_code.offload_api import _operation_payload

        _, context, _ = _operation_payload(
            {
                "operation_id": "op-1",
                "context": {
                    "model": "openai:gpt-5",
                    "model_params": {
                        key: "http://attacker.example/",
                        "temperature": 0.2,
                    },
                },
            }
        )

        assert context["model_params"] == {"temperature": 0.2}

    def test_stripping_is_logged_with_key_names_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dropped transport key must leave a trace naming the key.

        Silently ignoring it leaves a user whose gateway config is being skipped
        with nothing to find. The value is an endpoint or header, so only the
        key name is logged.
        """
        import logging

        from deepagents_code.offload_api import _operation_payload

        with caplog.at_level(logging.WARNING):
            _operation_payload(
                {
                    "operation_id": "op-1",
                    "context": {
                        "model_params": {
                            "base_url": "http://gateway.internal/v1",
                            "temperature": 0.2,
                        }
                    },
                }
            )

        assert "base_url" in caplog.text
        assert "gateway.internal" not in caplog.text

    def test_clean_model_params_dict_is_untouched(self) -> None:
        from deepagents_code.offload_api import _operation_payload

        _, context, _ = _operation_payload(
            {
                "operation_id": "op-1",
                "context": {"model_params": {"temperature": 0.2, "max_tokens": 64}},
            }
        )

        assert context["model_params"] == {"temperature": 0.2, "max_tokens": 64}


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
            "_model_spec": "provider:checkpointed-model",
            "_model_params": {
                "base_url": "https://trusted.example/v1",
                "temperature": 0.1,
            },
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


def _result(
    archive_path: str | None = "/conversation_history/thread-1.md",
) -> OffloadResult:
    """Build a complete operation result."""
    return {
        "status": "compacted",
        "messages_offloaded": 1,
        "messages_kept": 1,
        "tokens_before": 20,
        "tokens_after": 10,
        "archive_path": archive_path,
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
        calls: list[str] = []

        async def update_state(  # noqa: RUF029  # AsyncMock side effect contract
            *_args: object, **_kwargs: object
        ) -> None:
            calls.append("checkpoint")

        append = SimpleNamespace(
            path="/conversation_history/thread-1.md", rollback=AsyncMock()
        )
        archive = SimpleNamespace(
            session_id="archive-1",
            write=AsyncMock(side_effect=lambda: calls.append("archive") or append),
            update=lambda path: {
                "_summarization_event": {"cutoff_index": 1, "file_path": path}
            },
        )
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(side_effect=update_state),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {
                        "_summarization_event": {
                            "cutoff_index": 1,
                            "file_path": None,
                        },
                        "_summarization_session_id": "archive-1",
                    },
                    _result(archive_path=None),
                    cast("_PendingArchive", archive),
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
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
        assert runtime.context["model"] == "provider:checkpointed-model"
        assert runtime.context["model_params"] == {
            "base_url": "https://trusted.example/v1",
            "temperature": 0.1,
        }
        assert threads.update_state.await_count == 2
        args = threads.update_state.await_args_list[0]
        assert args.args[:2] == (
            "thread-1",
            {
                "_summarization_event": {
                    "cutoff_index": 1,
                    "file_path": None,
                },
                "_summarization_session_id": "archive-1",
                "_session_cost_usd": 0.25,
            },
        )
        assert "messages" not in args.args[1]
        assert "checkpoint" not in args.kwargs
        assert threads.update_state.await_args_list[1].args == (
            "thread-1",
            {
                "_summarization_event": {
                    "cutoff_index": 1,
                    "file_path": "/conversation_history/thread-1.md",
                }
            },
        )
        assert calls == ["checkpoint", "archive", "checkpoint"]
        prepared.rollback.assert_not_called()

    async def test_failed_archive_link_restores_the_append(self) -> None:
        """A failed follow-up checkpoint cannot leave duplicate history."""
        from deepagents_code import offload_api

        before = _thread_state()
        unlinked = _thread_state("reserved")
        unlinked_values = cast("dict[str, object]", unlinked["values"])
        unlinked_values["_summarization_event"] = {
            "cutoff_index": 1,
            "file_path": None,
        }
        append = SimpleNamespace(
            path="/conversation_history/thread-1.md", rollback=AsyncMock()
        )
        archive = SimpleNamespace(
            session_id="archive-1",
            write=AsyncMock(return_value=append),
            update=lambda path: {
                "_summarization_event": {"cutoff_index": 1, "file_path": path}
            },
        )
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before, unlinked]),
            update_state=AsyncMock(
                side_effect=[None, RuntimeError("archive link unavailable")]
            ),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {
                        "_summarization_event": {
                            "cutoff_index": 1,
                            "file_path": None,
                        },
                        "_summarization_session_id": "archive-1",
                    },
                    _result(archive_path=None),
                    cast("_PendingArchive", archive),
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "complete"
        assert response["result"]["archive_path"] is None
        append.rollback.assert_awaited_once()
        prepared.rollback.assert_not_called()

    async def test_cancellation_waits_for_checkpoint_archive_settlement(self) -> None:
        """A reserved commit must settle before cancellation becomes terminal."""
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
        )
        settlement_started = asyncio.Event()
        finish_settlement = asyncio.Event()

        async def settle(*_args: object, **_kwargs: object) -> None:
            settlement_started.set()
            await finish_settlement.wait()

        with (
            self._patched(offload_api, threads, operation, prepared),
            patch.object(
                offload_api,
                "_commit_deferred_archive",
                new=AsyncMock(side_effect=settle),
            ) as commit,
        ):
            task = asyncio.create_task(
                offload_api._execute_offload(
                    "thread-1",
                    operation_id="operation-1",
                    context={},
                    hook_responses={},
                )
            )
            await asyncio.wait_for(settlement_started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            finish_settlement.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        commit.assert_awaited_once()

    async def test_request_transport_cannot_replace_checkpointed_model(self) -> None:
        """Offload uses the model settings from the target thread checkpoint."""
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
                    {},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={
                    "model": "attacker:model",
                    "model_params": {"base_url": "https://attacker.example"},
                },
                hook_responses={},
            )

        runtime = operation.execute.await_args.args[1]
        assert runtime.context["model"] == "provider:checkpointed-model"
        assert runtime.context["model_params"]["base_url"] == (
            "https://trusted.example/v1"
        )

    async def test_legacy_thread_reuses_startup_summarizer(self) -> None:
        """A thread without model metadata ignores request model selection."""
        from deepagents_code import offload_api

        before = _thread_state()
        values = cast("dict[str, object]", before["values"])
        assert isinstance(values, dict)
        values.pop("_model_spec")
        values.pop("_model_params")
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={"model": "request:model", "model_params": {"x": 1}},
                hook_responses={},
            )

        runtime = operation.execute.await_args.args[1]
        assert "model" not in runtime.context
        assert "model_params" not in runtime.context

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
                    cast(
                        "_PendingArchive",
                        SimpleNamespace(session_id="archive-1", write=AsyncMock()),
                    ),
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
        operation.execute.return_value.archive.write.assert_not_awaited()
        prepare.assert_not_called()

    @pytest.mark.parametrize("status", ["busy", "interrupted"])
    async def test_thread_with_work_in_flight_is_rejected_before_operation(
        self, status: str
    ) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": status}),
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

    async def test_errored_thread_is_still_offloadable(self) -> None:
        """A failed turn must not lock the user out of `/offload`.

        A run that raises leaves the thread row on `error` until the next run
        completes, which is exactly when a user reaches for `/offload` to
        recover. Reaching the state read proves the status gate let it past;
        in-flight work is caught separately by the `next`/`tasks`/`interrupts`
        check against the checkpoint.
        """
        from deepagents_code import offload_api

        class _ReachedStateReadError(Exception):
            """Sentinel proving control passed the thread-status gate."""

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "error"}),
            get_state=AsyncMock(side_effect=_ReachedStateReadError),
            update_state=AsyncMock(),
        )
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=AsyncMock()),
            pytest.raises(_ReachedStateReadError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()

    @staticmethod
    @contextlib.contextmanager
    def _patched(
        offload_api: object,
        threads: SimpleNamespace,
        operation: SimpleNamespace,
        prepared: object,
    ) -> Iterator[None]:
        """Patch the client, runtime, and cost seams `_execute_offload` uses."""
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
            yield

    @pytest.mark.parametrize("channel", ["messages", "todos"])
    async def test_unpermitted_update_is_refused_and_cost_rolled_back(
        self, channel: str
    ) -> None:
        """A channel outside `OffloadStateUpdate` cannot reach the checkpoint.

        Unlike asserting on a mocked update that never contains the channel
        (which passes whether the guard exists or not), this drives an update
        that actually carries one. `todos` covers the allowlist itself: a
        `messages`-only check would let every other channel through.
        """
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
                    # Deliberately violates `OffloadStateUpdate` -- that is the
                    # point of the test: the runtime guard is the backstop for
                    # `Any`-typed values the SDK hands back.
                    {channel: ["smuggled"]},  # ty: ignore[invalid-key,invalid-argument-type]
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(RuntimeError, match=f"may not write .*{channel}"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()
        prepared.rollback.assert_called_once()

    async def test_empty_update_still_releases_claimed_cost_records(self) -> None:
        """A prepare with nothing to write must not silently eat its records."""
        from deepagents_code import offload_api

        before = _thread_state()
        noop_result = {**_result(), "status": "noop"}
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution({}, noop_result)  # ty: ignore[invalid-argument-type]
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response == {"status": "complete", "result": noop_result}
        threads.update_state.assert_not_awaited()
        prepared.rollback.assert_called_once()

    async def test_write_failure_without_advance_restores_cost(self) -> None:
        """A write that provably did not land returns its records to the recorder."""
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            # Third read is `_write_landed`: same checkpoint => did not land.
            get_state=AsyncMock(side_effect=[before, before, before]),
            update_state=AsyncMock(side_effect=RuntimeError("boom")),
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_called_once()

    async def test_write_failure_after_advance_keeps_cost_claimed(self) -> None:
        """An indeterminate write must not re-queue records and double-charge."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[
                    _thread_state("before"),
                    _thread_state("before"),
                    # `_write_landed`: the thread advanced, so the write likely
                    # applied despite the transport error.
                    _thread_state("after"),
                ]
            ),
            update_state=AsyncMock(side_effect=RuntimeError("connection reset")),
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(
                offload_api._OffloadIndeterminateError, match="could not confirm"
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_not_called()

    async def test_unreadable_thread_does_not_claim_the_thread_advanced(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unreadable readback must not be logged as an observed advance.

        Both outcomes keep the records claimed, but only `advanced` has evidence
        the write landed. Reporting a thread advance that was never observed
        would tell anyone auditing a missing charge that the spend was
        accounted for.
        """
        import logging

        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[before, before, RuntimeError("thread store offline")]
            ),
            update_state=AsyncMock(side_effect=RuntimeError("connection reset")),
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            caplog.at_level(logging.ERROR),
            pytest.raises(offload_api._OffloadIndeterminateError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert "could not be read back" in caplog.text
        assert "may be lost from the thread total" in caplog.text
        assert "advanced past checkpoint" not in caplog.text
        prepared.rollback.assert_not_called()
        prepared.commit.assert_called_once()

    async def test_cancelled_probe_still_settles_the_cost_records(self) -> None:
        """A cancel inside the write-landed probe must not skip settlement.

        `prepare_operation_cost` drains the recorder destructively, so a prepare
        that is neither committed nor rolled back deletes that spend from the
        thread's lifetime total permanently. The probe runs inside the
        settlement handler, so an escape from it -- a disconnect or a shutdown
        re-delivering cancellation while the handler unwinds -- would take both
        branches off the table and lose the records with no trace.
        """
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before, asyncio.CancelledError()]),
            update_state=AsyncMock(side_effect=RuntimeError("boom")),
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(offload_api._OffloadIndeterminateError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        # Unreadable means indeterminate, so the records stay claimed rather
        # than being restored -- but a decision was reached either way.
        prepared.rollback.assert_not_called()

    async def test_a_cancelled_write_is_not_converted_to_a_runtime_error(
        self,
    ) -> None:
        """Cancellation must propagate so the task actually observes it."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[
                    _thread_state("before"),
                    _thread_state("before"),
                    _thread_state("after"),
                ]
            ),
            update_state=AsyncMock(side_effect=asyncio.CancelledError()),
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
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(asyncio.CancelledError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_not_called()

    @staticmethod
    def _hook_request() -> object:
        """Build a real server-owned hook invocation request."""
        from datetime import UTC, datetime
        from pathlib import Path
        from uuid import uuid4

        from deepagents_code.hooks.models.domain import (
            ApprovalMode,
            HookContext,
            HookEvent,
            PreToolUseEvent,
            ToolCallData,
        )
        from deepagents_code.hooks.models.transport import (
            HookInvocation,
            HookInvocationRequest,
        )

        return HookInvocationRequest(
            protocol_version=1,
            invocation_id=uuid4(),
            snapshot_id="snapshot-1",
            run_id="run-1",
            invocation=HookInvocation(
                context=HookContext(
                    thread_id="thread-1",
                    cwd=Path("/tmp"),
                    approval_mode=ApprovalMode.MANUAL,
                ),
                event=PreToolUseEvent(
                    event=HookEvent.PRE_TOOL_USE,
                    call=ToolCallData(
                        id="call-1", name="compact_conversation", args={"force": True}
                    ),
                ),
            ),
            deadline=datetime(2026, 7, 23, tzinfo=UTC),
        )

    async def test_a_hook_request_becomes_an_interrupt_response(self) -> None:
        """An unanswered hook must leave the route as a resumable interrupt.

        `HookTransportInterruptError` derives from `BaseException` so the
        compaction chain cannot swallow it -- which also means the route's own
        `except Exception` cannot catch it. Without the dedicated handler it
        escapes to Starlette as a raw 500 for every user with a `PreCompact` or
        `PreToolUse` hook configured, and nothing else in the suite notices.
        """
        from deepagents_code import offload_api
        from deepagents_code.hooks.interrupt import is_hook_interrupt_payload
        from deepagents_code.hooks.server_middleware import (
            HookTransportInterruptError,
        )

        request = self._hook_request()
        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=before),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(side_effect=HookTransportInterruptError(request))  # ty: ignore[invalid-argument-type]
        )
        prepared = SimpleNamespace(
            update={}, rollback=MagicMock(), commit=MagicMock(), records=[]
        )

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "interrupt"
        assert is_hook_interrupt_payload(response["request"])
        assert response["request"]["request"]["invocation_id"] == str(
            request.invocation_id  # ty: ignore[unresolved-attribute]
        )
        # Nothing may be committed while a hook is still unanswered.
        threads.update_state.assert_not_awaited()

    async def test_accumulated_hook_responses_reach_the_operation(self) -> None:
        """The resume round must hand the replies back to the hook transport.

        `operation_hook_responses` is the single line that makes a multi-round
        resume terminate: `_invoke_hook` replays an already-answered invocation
        from that mapping instead of raising again. Passing an empty mapping
        would re-raise the same invocation forever and the client would die at
        its round limit, so assert the mapping is actually installed.
        """
        from deepagents_code import offload_api
        from deepagents_code.hooks.server_middleware import operation_hook_responses

        seen: list[object] = []

        async def execute(  # noqa: RUF029 -- must satisfy the async execute signature
            *_args: object, **_kwargs: object
        ) -> OffloadExecution:
            # Read the context var the way the hook transport does.
            from deepagents_code.hooks import server_middleware

            seen.append(server_middleware._HOOK_RESPONSES.get())
            return OffloadExecution(
                {"_summarization_event": {"cutoff_index": 1}},
                _result(),  # ty: ignore[invalid-argument-type]
            )

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(execute=AsyncMock(side_effect=execute))
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        replies: dict[str, object] = {"hook-1": {"decision": "allow"}}
        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses=replies,
            )

        assert response["status"] == "complete"
        assert seen == [replies]
        # Outside the operation the var is back to graph mode.
        assert operation_hook_responses is not None

    async def test_pending_graph_work_is_rejected(self) -> None:
        from deepagents_code import offload_api

        pending = {**_thread_state(), "next": ["tools"]}
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=pending),
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
            pytest.raises(
                offload_api._OffloadConflictError, match="pending graph work"
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        runtime.assert_not_awaited()

    async def test_unregistered_thread_is_rejected_with_an_actionable_conflict(
        self,
    ) -> None:
        """A 404 from the live thread store must not become an opaque 500.

        The dev server keeps checkpoint persistence and thread registration
        separate, so a resumed thread can 404 here while holding state on disk.
        `NotFoundError` is an ordinary `Exception`, so without this mapping it
        reaches the route's generic handler and the user is told to read the
        server log.
        """
        import httpx
        from langgraph_sdk.errors import NotFoundError

        from deepagents_code import offload_api

        request = httpx.Request("GET", "http://localhost/threads/thread-1")
        not_found = NotFoundError(
            "missing", response=httpx.Response(404, request=request), body=None
        )
        threads = SimpleNamespace(
            get=AsyncMock(side_effect=not_found),
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
            pytest.raises(
                offload_api._OffloadConflictError, match="not registered on the server"
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.get_state.assert_not_awaited()
        threads.update_state.assert_not_awaited()
        runtime.assert_not_awaited()

    async def test_empty_thread_reports_nothing_to_offload(self) -> None:
        """An empty thread is an unchanged outcome, not a failure.

        `_checkpoint_id` rejects a thread with no checkpoint, so answering the
        empty case at the boundary is what keeps `OffloadOperation.execute`'s
        graceful `empty` branch reachable over HTTP. Without it the user is told
        the operation failed for a thread that simply has nothing to compact.
        """
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value={"values": {}, "checkpoint": {}}),
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
        ):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "complete"
        assert response["result"]["status"] == "empty"
        assert response["result"]["messages_kept"] == 0
        # Nothing is compacted, so nothing is written and no agent is built.
        threads.update_state.assert_not_awaited()
        runtime.assert_not_awaited()

    async def test_missing_checkpoint_with_messages_is_rejected(self) -> None:
        """State with messages but no checkpoint cannot be written against."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                return_value={**_thread_state(), "checkpoint": {}},
            ),
            update_state=AsyncMock(),
        )
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=AsyncMock()),
            pytest.raises(offload_api._OffloadConflictError, match="no checkpoint"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )


def test_validated_context_fields_exist_on_the_schema() -> None:
    """The validator's field lists must not drift from `CLIContextSchema`.

    The names are hand-written string tuples, so a rename in the dataclass would
    leave this route validating a key nobody sends -- forever, with no test
    failing. The protocol version is pinned the same way; this closes the other
    hand-maintained list.
    """
    from dataclasses import fields

    from deepagents_code import offload_api
    from deepagents_code._cli_context import CLIContextSchema

    declared = {f.name for f in fields(CLIContextSchema)}
    validated = {
        *offload_api._CONTEXT_STR_OR_NONE_FIELDS,
        *offload_api._CONTEXT_DICT_FIELDS,
    }

    assert validated <= declared, validated - declared


class TestRouteRegistration:
    """The Starlette app exposes the paths and methods the client calls.

    Every other route test fabricates a request with hand-written
    `path_params`, so a path or converter rename -- `{thread_id:str}` to
    `{tid:str}`, say -- would leave the whole unit suite green while the real
    handler raised `KeyError` (neither `TypeError` nor `ValueError`, so it
    escapes the 422 block as a bare 500).
    """

    def test_offload_and_cancel_paths_are_registered(self) -> None:
        from starlette.testclient import TestClient

        from deepagents_code import offload_api
        from deepagents_code.offload_middleware import unchanged_offload_result

        calls: list[tuple[str, str]] = []

        async def fake_execute(  # noqa: RUF029  # replaces an async callee
            thread_id: str,
            *,
            operation_id: str,
            context: dict[str, object],  # noqa: ARG001
            hook_responses: dict[str, object],  # noqa: ARG001
        ) -> dict[str, object]:
            calls.append((thread_id, operation_id))
            return {
                "status": "complete",
                "result": unchanged_offload_result("noop", messages=1, tokens=5),
            }

        with (
            patch.object(offload_api, "_execute_offload", new=fake_execute),
            TestClient(offload_api.app) as client,
        ):
            response = client.post(
                "/dcode/threads/thread-42/offload",
                json={
                    "operation_id": "op-1",
                    "context": {},
                    "hook_responses": {},
                },
            )

        assert response.status_code == 200, response.text
        # The handler read the id out of the real path params, so the route's
        # converter name and the key it indexes agree.
        assert calls == [("thread-42", "op-1")]

    def test_get_on_the_offload_path_is_not_allowed(self) -> None:
        """Only POST is registered; the capability probe was removed."""
        from starlette.testclient import TestClient

        from deepagents_code import offload_api

        with TestClient(offload_api.app) as client:
            response = client.get("/dcode/threads/thread-42/offload")

        assert response.status_code == 405

    def test_cancel_path_is_registered(self) -> None:
        from starlette.testclient import TestClient

        from deepagents_code import offload_api

        with TestClient(offload_api.app) as client:
            response = client.post(
                "/dcode/threads/thread-42/offload/op-1/cancel",
            )

        # No such operation is active, but the route resolved and its handler
        # answered rather than 404-ing on an unmatched path.
        assert response.status_code == 200, response.text


class TestThreadLock:
    """Concurrent offloads of one thread are serialized in-process.

    The whole design is read-check-execute-recheck-write against a checkpoint.
    The per-thread lock is what makes the recheck meaningful for two requests in
    the same process: without it both can pass the status and checkpoint gates
    before either writes.
    """

    def test_each_thread_gets_its_own_lock(self) -> None:
        from deepagents_code import offload_api

        first = offload_api._thread_lock("thread-1")

        assert offload_api._thread_lock("thread-1") is first
        assert offload_api._thread_lock("thread-2") is not first

    async def test_execute_waits_for_the_threads_lock(self) -> None:
        """An offload must not touch thread state while the lock is held.

        Holding the lock externally proves the `async with` is on the path:
        remove it, or key it on `operation_id` instead of `thread_id`, and the
        operation reads state immediately.
        """
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "busy"}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )

        async def run() -> None:
            with contextlib.suppress(offload_api._OffloadConflictError):
                await offload_api._execute_offload(
                    "thread-1",
                    operation_id="op-2",
                    context={},
                    hook_responses={},
                )

        with patch.object(
            offload_api,
            "get_client",
            return_value=SimpleNamespace(threads=threads),
        ):
            async with offload_api._thread_lock("thread-1"):
                blocked = asyncio.create_task(run())
                for _ in range(5):
                    await asyncio.sleep(0)
                threads.get.assert_not_awaited()

            await asyncio.wait_for(blocked, timeout=5)

        threads.get.assert_awaited_once()

    async def test_a_different_thread_is_not_blocked(self) -> None:
        """The lock is per thread, so unrelated threads must not serialize."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "busy"}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )

        with patch.object(
            offload_api,
            "get_client",
            return_value=SimpleNamespace(threads=threads),
        ):
            async with offload_api._thread_lock("thread-1"):
                with pytest.raises(offload_api._OffloadConflictError):
                    await asyncio.wait_for(
                        offload_api._execute_offload(
                            "thread-2",
                            operation_id="op-1",
                            context={},
                            hook_responses={},
                        ),
                        timeout=5,
                    )

        threads.get.assert_awaited_once()


class TestOffloadRoute:
    """The HTTP layer maps operation outcomes onto distinct status codes."""

    @staticmethod
    def _request(payload: object) -> SimpleNamespace:
        """Build a minimal Starlette-like request for the route handler."""
        return SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(return_value=payload),
        )

    @staticmethod
    def _cancel_request(operation_id: str = "op-1") -> SimpleNamespace:
        """Build a minimal request for the cancellation route."""
        return SimpleNamespace(
            path_params={"thread_id": "thread-1", "operation_id": operation_id}
        )

    async def test_malformed_request_is_422(self) -> None:
        import json

        from deepagents_code import offload_api

        response = await offload_api.offload(self._request({"operation_id": ""}))  # ty: ignore[invalid-argument-type]

        assert response.status_code == 422
        assert "operation_id" in json.loads(bytes(response.body))["detail"]

    async def test_cancel_stops_and_joins_an_active_operation(self) -> None:
        """The cancel response is sent only after the operation task exits."""
        import json

        from deepagents_code import offload_api

        started = asyncio.Event()
        stopped = asyncio.Event()

        async def execute(*_args: object, **_kwargs: object) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.object(
            offload_api, "_execute_offload", new=AsyncMock(side_effect=execute)
        ):
            operation = asyncio.create_task(
                offload_api.offload(
                    self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            response = await offload_api.cancel_offload(self._cancel_request())  # ty: ignore[invalid-argument-type]

        assert response.status_code == 200
        assert json.loads(bytes(response.body)) == {"status": "cancelled"}
        assert stopped.is_set()
        with pytest.raises(asyncio.CancelledError):
            await operation

    async def test_cancel_before_request_prevents_operation_start(self) -> None:
        """A reordered cancel closes the disconnect-before-register race."""
        import json

        from deepagents_code import offload_api

        cancel = await offload_api.cancel_offload(self._cancel_request())  # ty: ignore[invalid-argument-type]
        execute = AsyncMock()
        with patch.object(offload_api, "_execute_offload", new=execute):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert json.loads(bytes(cancel.body)) == {"status": "cancelled"}
        assert response.status_code == 409
        assert "cancelled" in json.loads(bytes(response.body))["detail"]
        execute.assert_not_awaited()

    async def test_conflict_is_409(self) -> None:
        import json

        from deepagents_code import offload_api

        with patch.object(
            offload_api,
            "_execute_offload",
            new=AsyncMock(
                side_effect=offload_api._OffloadConflictError("thread is busy")
            ),
        ):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert response.status_code == 409
        assert json.loads(bytes(response.body))["detail"] == "thread is busy"

    async def test_indeterminate_write_is_500_with_its_own_detail(self) -> None:
        import json

        from deepagents_code import offload_api

        with patch.object(
            offload_api,
            "_execute_offload",
            new=AsyncMock(
                side_effect=offload_api._OffloadIndeterminateError("cannot confirm")
            ),
        ):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert response.status_code == 500
        assert json.loads(bytes(response.body))["detail"] == "cannot confirm"

    async def test_unbuildable_runtime_is_503_and_does_not_exit(self) -> None:
        """The startup barrier must not kill the process from a request handler.

        `get_server_runtime` answers a construction failure with `sys.exit(1)`,
        which is correct for the `langgraph.json` graph factory and fatal here:
        `SystemExit` is a `BaseException`, so without containment it escapes the
        route entirely and takes the server down mid-request.
        """
        import json

        from deepagents_code import offload_api

        with patch.object(
            offload_api,
            "_execute_offload",
            new=AsyncMock(
                side_effect=offload_api._OffloadUnavailableError("runtime failed")
            ),
        ):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert response.status_code == 503
        assert json.loads(bytes(response.body))["detail"] == "runtime failed"

    async def test_a_startup_exit_becomes_unavailable_not_a_process_exit(
        self,
    ) -> None:
        """`SystemExit` from the runtime resolves to a typed error, not an exit."""
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=before),
            update_state=AsyncMock(),
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
                new=AsyncMock(side_effect=SystemExit(1)),
            ),
            pytest.raises(offload_api._OffloadUnavailableError, match="unavailable"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()

    async def test_internal_type_error_is_500_not_422(self) -> None:
        """A server-side shape fault must not be reported as a client error."""
        import json

        from deepagents_code import offload_api

        with patch.object(
            offload_api,
            "_execute_offload",
            new=AsyncMock(side_effect=TypeError("LangGraph returned non-object state")),
        ):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert response.status_code == 500
        assert "server log" in json.loads(bytes(response.body))["detail"]
