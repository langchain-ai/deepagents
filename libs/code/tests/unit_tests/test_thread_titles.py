"""Bounded model input, safe output, and failure handling for thread names."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents_code.thread_titles import generate_thread_name


@pytest.fixture
def title_model(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage("Cache invalidation repair")
    factory = MagicMock(return_value=SimpleNamespace(model=model))
    monkeypatch.setattr("deepagents_code.config.create_model", factory)
    return model


async def test_generation_uses_only_visible_conversation(
    title_model: AsyncMock,
) -> None:
    name = await generate_thread_name(
        "provider:rename-model",
        [
            SystemMessage("Private system instructions"),
            HumanMessage("Fix caching"),
            HumanMessage("Goal changed", additional_kwargs={"lc_source": "goal_state"}),
            HumanMessage(
                "Private context", additional_kwargs={"lc_source": "local_context"}
            ),
            ToolMessage("Private tool output", tool_call_id="lookup"),
            AIMessage("Investigating stale cache entries"),
        ],
    )
    assert name == "Cache invalidation repair"
    prompt = title_model.ainvoke.call_args.args[0]
    assert prompt[1] == (
        "human",
        "human: Fix caching\nai: Investigating stale cache entries",
    )
    assert title_model.ainvoke.call_args.kwargs["config"]["callbacks"] == []


async def test_generation_bounds_conversation_sent_to_model(
    title_model: AsyncMock,
) -> None:
    await generate_thread_name(
        "provider:rename-model",
        [HumanMessage("x" * 9000), AIMessage("Must not be included")],
    )
    conversation = title_model.ainvoke.call_args.args[0][1][1]
    assert 0 < len(conversation) <= 8000
    assert conversation.startswith("human: ")
    assert "Must not be included" not in conversation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('  "Cache\n\t invalidation repair."  ', "Cache invalidation repair"),
        ("`Fix\x1b cache\x07 entries\x9c`", "Fix cache entries"),
        (
            "one two three four five six seven eight nine",
            "one two three four five six seven eight",
        ),
        ("x" * 80, "x" * 50),
        (
            "Investigate stale cached configuration values across services",
            "Investigate stale cached configuration values",
        ),
    ],
)
async def test_generated_names_are_normalized(
    title_model: AsyncMock, raw: str, expected: str
) -> None:
    title_model.ainvoke.return_value = AIMessage(raw)
    name = await generate_thread_name(
        "provider:rename-model", [HumanMessage("Fix cache")]
    )
    assert name == expected
    assert name.isprintable()
    assert len(name) <= 50


@pytest.mark.parametrize("raw", ["", "\n\t", "\x1b\x07", "`...`"])
async def test_empty_generated_names_are_rejected(
    title_model: AsyncMock, raw: str
) -> None:
    title_model.ainvoke.return_value = AIMessage(raw)
    with pytest.raises(ValueError, match="between 1 and 50 characters"):
        await generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])


async def test_no_conversation_does_not_call_model(title_model: AsyncMock) -> None:
    with pytest.raises(ValueError, match="Send a message"):
        await generate_thread_name("provider:rename-model", [SystemMessage("Rules")])
    title_model.ainvoke.assert_not_awaited()


async def test_stalled_model_times_out(
    title_model: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeout = asyncio.timeout
    cancelled = asyncio.Event()

    async def stall(*_args: object, **_kwargs: object) -> AIMessage:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return AIMessage("Unreachable")

    monkeypatch.setattr(
        "deepagents_code.thread_titles.asyncio.timeout", lambda _: timeout(0.05)
    )
    title_model.ainvoke.side_effect = stall
    with pytest.raises(TimeoutError):
        await generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])
    assert cancelled.is_set()


@pytest.mark.parametrize("factory_fails", [False, True])
async def test_initialization_timeout_waits_for_factory(
    monkeypatch: pytest.MonkeyPatch, *, factory_fails: bool
) -> None:
    """A naming timeout cannot leave provider-setting mutations running."""
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    deadline = asyncio.timeout(None)
    model = AsyncMock()
    finished = threading.Event()

    def create_model(*_args: object, **_kwargs: object) -> SimpleNamespace:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        finished.set()
        if factory_fails:
            msg = "Model initialization failed"
            raise ValueError(msg)
        return SimpleNamespace(model=model)

    monkeypatch.setattr("deepagents_code.config.create_model", create_model)
    monkeypatch.setattr(
        "deepagents_code.thread_titles.asyncio.timeout", lambda _: deadline
    )
    task = asyncio.create_task(
        generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        deadline.reschedule(loop.time())
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    with pytest.raises(TimeoutError):
        await task
    assert finished.is_set()
    model.ainvoke.assert_not_awaited()
