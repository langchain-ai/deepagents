"""Tests for the mid-turn steering inbox and its delivery middleware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.store.memory import InMemoryStore

from deepagents_code._env_vars import STEERING
from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY
from deepagents_code.steering import (
    STEER_NAMESPACE_ROOT,
    adelete_steers,
    aread_pending_steers,
    awrite_steer,
    coerce_consumed_seq,
    next_steer_seq,
    parse_steer_item,
    steer_key,
    steer_namespace,
    steering_enabled,
)
from deepagents_code.steering_middleware import SteeringMiddleware

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langgraph.store.base import BaseStore

THREAD = "thread-steer-1"
OTHER_THREAD = "thread-steer-2"


@dataclass
class _Context:
    """Minimal stand-in for the run context field the middleware reads."""

    thread_id: str | None = None


class _StoreWritingAgent:
    """Client-side agent double that writes to a real Store like the server would.

    `RemoteAgent.aput_store_item` posts to the server's Store over HTTP; this
    performs the same write locally so the client write path and the server read
    path can be exercised against one another without a server.
    """

    def __init__(self, store: BaseStore) -> None:
        self.store = store
        self.ttls: list[int | None] = []

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        ttl: int | None = None,
    ) -> None:
        """Record the TTL and write the item into the backing Store."""
        self.ttls.append(ttl)
        await self.store.aput(namespace, key, value)


class _ExplodingStore:
    """Store whose search always fails, to prove reads degrade to an empty inbox."""

    async def asearch(
        self,
        *args: Any,  # noqa: ARG002  # signature mirrors `BaseStore.asearch`
        **kwargs: Any,  # noqa: ARG002  # signature mirrors `BaseStore.asearch`
    ) -> list[Any]:
        """Raise as a failing Store would.

        Returns:
            Never returns; the call always raises.
        """
        msg = "store unavailable"
        raise RuntimeError(msg)


def _runtime(
    store: BaseStore | None,
    *,
    thread_id: str | None = THREAD,
    execution_thread_id: str | None = THREAD,
) -> Runtime[Any]:
    """Build a runtime carrying the context, Store, and execution thread id."""
    execution = (
        ExecutionInfo(
            checkpoint_id="cp",
            checkpoint_ns="",
            task_id="task",
            thread_id=execution_thread_id,
        )
        if execution_thread_id is not None
        else None
    )
    return Runtime(
        context=_Context(thread_id=thread_id),
        store=store,
        execution_info=execution,
    )


async def _write(store: BaseStore, thread_id: str, seq: int, text: str) -> None:
    """Write one raw inbox item, bypassing the client helper's sequencing."""
    await store.aput(
        steer_namespace(thread_id),
        steer_key(seq),
        {"seq": seq, "text": text, "literal_user_text": text, "turn_id": "turn-1"},
    )


class TestInboxAddressing:
    """Namespace and key derivation."""

    def test_namespace_is_scoped_per_thread(self) -> None:
        """Each thread gets its own namespace under the shared root."""
        namespace = steer_namespace(THREAD)
        assert namespace[:2] == STEER_NAMESPACE_ROOT
        assert namespace != steer_namespace(OTHER_THREAD)

    def test_namespace_does_not_expose_thread_id(self) -> None:
        """The thread id is hashed, matching the approval-mode key convention."""
        assert THREAD not in steer_namespace(THREAD)[-1]

    def test_keys_sort_in_delivery_order(self) -> None:
        """Zero-padded keys keep lexical order aligned with numeric order."""
        assert sorted([steer_key(2), steer_key(10)]) == [steer_key(2), steer_key(10)]

    def test_sequence_numbers_strictly_increase(self) -> None:
        """Two steers submitted in the same millisecond still order."""
        seqs = [next_steer_seq() for _ in range(5)]
        assert seqs == sorted(set(seqs))


class TestPayloadValidation:
    """`parse_steer_item` must reject anything it cannot fully trust."""

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("k", None),
            ("k", "not-a-mapping"),
            ("k", {"text": "hi"}),
            ("k", {"seq": 0, "text": "hi"}),
            ("k", {"seq": -1, "text": "hi"}),
            ("k", {"seq": True, "text": "hi"}),
            ("k", {"seq": "1", "text": "hi"}),
            ("k", {"seq": 1, "text": "   "}),
            ("k", {"seq": 1, "text": 5}),
            ("", {"seq": 1, "text": "hi"}),
            (None, {"seq": 1, "text": "hi"}),
        ],
    )
    def test_malformed_items_are_rejected(self, key: object, value: object) -> None:
        """A malformed item is dropped rather than partially trusted."""
        assert parse_steer_item(key, value) is None

    def test_valid_item_is_parsed(self) -> None:
        """A well-formed item keeps every validated field."""
        item = parse_steer_item(
            "k",
            {
                "seq": 7,
                "text": "use async",
                "literal_user_text": "use async instead",
                "turn_id": "turn-9",
            },
        )
        assert item is not None
        assert (item.seq, item.text, item.turn_id) == (7, "use async", "turn-9")
        assert item.literal_user_text == "use async instead"

    def test_literal_text_falls_back_to_delivered_text(self) -> None:
        """A missing or non-string literal text falls back to the message text."""
        item = parse_steer_item("k", {"seq": 1, "text": "hi", "literal_user_text": 3})
        assert item is not None
        assert item.literal_user_text == "hi"

    def test_non_string_turn_id_becomes_none(self) -> None:
        """An unusable turn id is dropped instead of reaching prompt metadata."""
        item = parse_steer_item("k", {"seq": 1, "text": "hi", "turn_id": 12})
        assert item is not None
        assert item.turn_id is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(5, 5), (0, 0), (-3, 0), (None, 0), ("4", 0), (True, 0)],
    )
    def test_consumed_watermark_coercion(self, value: object, expected: int) -> None:
        """Only positive integers survive as a watermark."""
        assert coerce_consumed_seq(value) == expected


class TestInboxReads:
    """Reading a thread's pending items."""

    async def test_returns_items_after_watermark_in_order(self) -> None:
        """Only newer items are returned, ordered by sequence."""
        store = InMemoryStore()
        await _write(store, THREAD, 1, "first")
        await _write(store, THREAD, 2, "second")
        await _write(store, THREAD, 3, "third")

        items = await aread_pending_steers(store, THREAD, after_seq=1)

        assert [item.text for item in items] == ["second", "third"]

    async def test_other_threads_are_never_read(self) -> None:
        """A run may only observe its own thread's inbox."""
        store = InMemoryStore()
        await _write(store, OTHER_THREAD, 1, "not mine")

        assert await aread_pending_steers(store, THREAD, after_seq=0) == []

    async def test_malformed_items_are_skipped(self) -> None:
        """A corrupt item cannot block a valid one behind it."""
        store = InMemoryStore()
        await store.aput(steer_namespace(THREAD), steer_key(1), {"text": "no seq"})
        await _write(store, THREAD, 2, "valid")

        items = await aread_pending_steers(store, THREAD, after_seq=0)

        assert [item.text for item in items] == ["valid"]

    async def test_missing_store_reads_empty(self) -> None:
        """No Store means no steering, not an exception."""
        assert await aread_pending_steers(None, THREAD, after_seq=0) == []

    async def test_store_failure_reads_empty(self) -> None:
        """A failing Store degrades to an empty inbox instead of failing the run."""
        assert await aread_pending_steers(_ExplodingStore(), THREAD, after_seq=0) == []

    async def test_delete_removes_delivered_items(self) -> None:
        """Delivered keys are cleaned up so the namespace does not grow."""
        store = InMemoryStore()
        await _write(store, THREAD, 1, "first")

        await adelete_steers(store, THREAD, [steer_key(1)])

        assert await aread_pending_steers(store, THREAD, after_seq=0) == []


class TestClientWritePath:
    """The client write helper must produce items the server can read."""

    async def test_write_then_read_round_trip(self) -> None:
        """Text written by the client is readable by the server-side reader."""
        store = InMemoryStore()
        agent = _StoreWritingAgent(store)

        seq = await awrite_steer(agent, THREAD, text="use async", turn_id="turn-3")

        assert seq is not None
        items = await aread_pending_steers(store, THREAD, after_seq=0)
        assert [(item.text, item.turn_id) for item in items] == [
            ("use async", "turn-3")
        ]
        assert agent.ttls == [pytest.approx(120)]

    async def test_write_is_skipped_without_a_store_writer(self) -> None:
        """A local in-process agent has no Store, so the write is a no-op."""
        assert await awrite_steer(object(), THREAD, text="hi") is None

    async def test_written_items_preserve_submission_order(self) -> None:
        """Successive writes are delivered in the order they were queued."""
        store = InMemoryStore()
        agent = _StoreWritingAgent(store)

        await awrite_steer(agent, THREAD, text="first")
        await awrite_steer(agent, THREAD, text="second")

        items = await aread_pending_steers(store, THREAD, after_seq=0)
        assert [item.text for item in items] == ["first", "second"]


class TestSteeringMiddleware:
    """Delivery at the model boundary."""

    async def test_injects_pending_messages_and_advances_watermark(self) -> None:
        """Pending items become user messages and bump the consumed watermark."""
        store = InMemoryStore()
        await _write(store, THREAD, 4, "use async instead")
        middleware = SteeringMiddleware()

        update = await middleware.abefore_model({"messages": []}, _runtime(store))

        assert update is not None
        assert update["_steer_consumed_seq"] == 4
        (message,) = update["messages"]
        assert isinstance(message, HumanMessage)
        assert message.content == "use async instead"
        metadata = message.additional_kwargs[USER_PROMPT_METADATA_KEY]
        assert metadata["literal_user_text"] == "use async instead"
        assert metadata["turn_id"] == "turn-1"
        assert metadata["referenced_paths"] == []

    async def test_delivered_message_is_not_delivered_twice(self) -> None:
        """The checkpointed watermark makes a second boundary a no-op."""
        store = InMemoryStore()
        await _write(store, THREAD, 4, "use async instead")
        middleware = SteeringMiddleware()
        runtime = _runtime(store)

        first = await middleware.abefore_model({"messages": []}, runtime)
        assert first is not None
        second = await middleware.abefore_model(
            {"messages": [], "_steer_consumed_seq": first["_steer_consumed_seq"]},
            runtime,
        )

        assert second is None

    async def test_uncheckpointed_delivery_is_redelivered(self) -> None:
        """Without the watermark persisted, the message is delivered again.

        This is the deliberate trade: a run that dies before its checkpoint is
        written redelivers rather than silently dropping user input.
        """
        store = InMemoryStore()
        await store.aput(
            steer_namespace(THREAD),
            steer_key(4),
            {"seq": 4, "text": "retry me", "literal_user_text": "retry me"},
        )
        middleware = SteeringMiddleware()

        await middleware.abefore_model({"messages": []}, _runtime(store))
        await store.aput(
            steer_namespace(THREAD),
            steer_key(4),
            {"seq": 4, "text": "retry me", "literal_user_text": "retry me"},
        )
        again = await middleware.abefore_model({"messages": []}, _runtime(store))

        assert again is not None

    async def test_empty_inbox_returns_no_update(self) -> None:
        """Nothing queued means no state update at all."""
        middleware = SteeringMiddleware()

        assert (
            await middleware.abefore_model({"messages": []}, _runtime(InMemoryStore()))
            is None
        )

    async def test_disabled_flag_skips_the_store_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Opting out stops delivery even with items waiting."""
        monkeypatch.setenv(STEERING, "0")
        store = InMemoryStore()
        await _write(store, THREAD, 1, "ignored")

        assert not steering_enabled()
        assert (
            await SteeringMiddleware().abefore_model({"messages": []}, _runtime(store))
            is None
        )

    async def test_thread_id_mismatch_skips_delivery(self) -> None:
        """A context/execution thread disagreement must not pick a namespace."""
        store = InMemoryStore()
        await _write(store, THREAD, 1, "mine")
        runtime = _runtime(store, thread_id=OTHER_THREAD, execution_thread_id=THREAD)

        assert (
            await SteeringMiddleware().abefore_model({"messages": []}, runtime) is None
        )

    async def test_missing_thread_id_skips_delivery(self) -> None:
        """With no thread id there is no inbox to read."""
        store = InMemoryStore()
        await _write(store, THREAD, 1, "mine")
        runtime = _runtime(store, thread_id=None, execution_thread_id=None)

        assert (
            await SteeringMiddleware().abefore_model({"messages": []}, runtime) is None
        )


class TestSteeringInRealAgent:
    """The middleware inside a real `create_agent` loop."""

    async def test_steered_message_reaches_the_next_model_call(self) -> None:
        """A steer written mid-run is visible to the model's next call."""
        from langchain.agents import create_agent
        from langchain_core.tools import tool

        store = InMemoryStore()
        seen_prompts: list[list[str]] = []

        @tool
        def slow_tool(query: str) -> str:
            """Record the call, then steer the still-running turn."""
            store.put(
                steer_namespace(THREAD),
                steer_key(9),
                {"seq": 9, "text": "actually, use async", "turn_id": "turn-1"},
            )
            return f"looked up {query}"

        class _TwoStepModel(_ToolBindingFakeModel):
            """Calls `slow_tool` once, then answers."""

            disable_streaming: bool = True

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: CallbackManagerForLLMRun | None = None,
                **kwargs: Any,
            ) -> ChatResult:
                del stop, run_manager, kwargs
                seen_prompts.append(
                    [
                        str(message.content)
                        for message in messages
                        if isinstance(message, HumanMessage)
                    ]
                )
                if any(isinstance(message, ToolMessage) for message in messages):
                    response = AIMessage(content="done")
                else:
                    response = AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "slow_tool",
                                "args": {"query": "steering"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                return ChatResult(generations=[ChatGeneration(message=response)])

        agent = create_agent(
            model=_TwoStepModel(),
            tools=[slow_tool],
            middleware=[SteeringMiddleware()],
            context_schema=_Context,
            store=store,
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="look something up")]},
            context=_Context(thread_id=THREAD),
        )

        assert seen_prompts[0] == ["look something up"]
        assert "actually, use async" in seen_prompts[1]
        steered = [
            message
            for message in result["messages"]
            if isinstance(message, HumanMessage)
            and message.content == "actually, use async"
        ]
        assert len(steered) == 1
