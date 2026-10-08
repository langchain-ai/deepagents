"""Public graph lifetime of admitted history writes during failure and cancellation."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langgraph.errors import GraphDrained
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware


class _Model(FakeMessagesListChatModel):
    failure: Exception | None = None
    failed: asyncio.Event | None = None
    attempts: int = 0

    def bind_tools(self, tools: Sequence[object], **kwargs: object) -> "_Model":
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if self.failure is not None:
            self.attempts += 1
            if self.failed is not None and self.attempts == 3:
                self.failed.set()
            raise self.failure
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _HeldHistory(FilesystemBackend):
    def __init__(self, root: Path, failure: Exception | None = None) -> None:
        super().__init__(root_dir=root, virtual_mode=True)
        self.failure = failure
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancelled = False

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.started.set()
        try:
            await self.release.wait()
            if self.failure is not None:
                raise self.failure
            return await super().awrite(file_path, content)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.finished.set()


def _graph(backend: FilesystemBackend, model: _Model) -> CompiledStateGraph:
    summary = SummarizationMiddleware(model=model, backend=backend, trigger=("messages", 2), keep=("messages", 1))
    return create_deep_agent(model=_Model(responses=[AIMessage(content="Complete")]), backend=backend, middleware=[summary])


def _messages() -> list[BaseMessage]:
    return [HumanMessage(content="First question"), AIMessage(content="First answer"), HumanMessage(content="Next question")]


@pytest.mark.parametrize("failure", [RuntimeError("summary failed"), GraphDrained("stop")], ids=["summary-error", "graph-drain"])
@pytest.mark.parametrize("archive_fails", [False, True], ids=["archive-success", "archive-failure"])
async def test_summary_failure_waits_for_admitted_history(tmp_path: Path, failure: Exception, *, archive_fails: bool) -> None:
    backend = _HeldHistory(tmp_path, OSError("storage unavailable") if archive_fails else None)
    model = _Model(responses=[AIMessage(content="Summary")], failure=failure, failed=asyncio.Event())
    invocation = asyncio.create_task(_graph(backend, model).ainvoke({"messages": _messages()}))
    try:
        await asyncio.wait_for(backend.started.wait(), timeout=10)
        assert model.failed is not None
        await asyncio.wait_for(model.failed.wait(), timeout=10)
        done, _ = await asyncio.wait({invocation}, timeout=0.1)
        assert not done, "public invocation released its owner while history remained admitted"
        backend.release.set()
        with pytest.raises(type(failure)) as raised:
            await invocation
        assert raised.value is failure
        assert backend.finished.is_set()
        assert not backend.cancelled
        history = await backend.aglob("/conversation_history/*.md")
        assert bool(history.matches) is not archive_fails
    finally:
        backend.release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        await asyncio.wait_for(backend.finished.wait(), timeout=2)


async def test_repeated_cancellation_during_drain_settles_history(tmp_path: Path) -> None:
    backend = _HeldHistory(tmp_path)
    model = _Model(responses=[AIMessage(content="Summary")], failure=RuntimeError("summary failed"), failed=asyncio.Event())
    invocation = asyncio.create_task(_graph(backend, model).ainvoke({"messages": _messages()}))
    try:
        await asyncio.wait_for(backend.started.wait(), timeout=10)
        assert model.failed is not None
        await asyncio.wait_for(model.failed.wait(), timeout=10)
        for _ in range(2):
            invocation.cancel()
            done, _ = await asyncio.wait({invocation}, timeout=0.1)
            assert not done, "cancellation released the public invocation before admitted I/O settled"
        backend.release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert backend.finished.is_set()
        assert not backend.cancelled
        assert (await backend.aglob("/conversation_history/*.md")).matches
    finally:
        backend.release.set()
        await asyncio.gather(invocation, return_exceptions=True)
        await asyncio.wait_for(backend.finished.wait(), timeout=2)


async def test_successful_summary_retains_completed_history(tmp_path: Path) -> None:
    backend = _HeldHistory(tmp_path)
    backend.release.set()
    result = await _graph(backend, _Model(responses=[AIMessage(content="Summary")])).ainvoke({"messages": _messages()})
    assert result["messages"][-1].text == "Complete"
    assert backend.finished.is_set()
    assert not backend.cancelled
    history = await backend.aglob("/conversation_history/*.md")
    assert history.matches is not None
    assert len(history.matches) == 1
    content = (await backend.adownload_files([history.matches[0]["path"]]))[0].content
    assert content is not None
    assert b"First question" in content


async def test_summary_continues_after_history_write_failure(tmp_path: Path) -> None:
    backend = _HeldHistory(tmp_path, OSError("history storage unavailable"))
    backend.release.set()
    with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
        result = await _graph(backend, _Model(responses=[AIMessage(content="Summary")])).ainvoke({"messages": _messages()})
    assert result["messages"][-1].text == "Complete"
    assert backend.finished.is_set()
    assert not backend.cancelled
    assert not (await backend.aglob("/conversation_history/*.md")).matches
