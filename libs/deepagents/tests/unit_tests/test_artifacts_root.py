"""Tests for artifacts_root parameterization."""

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from tests.unit_tests.chat_model import GenericFakeChatModel as FakeChatModel


def _make_store_backend():
    mem_store = InMemoryStore()
    backend = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))
    return backend, mem_store


def _runtime(tool_call_id: str = "tc"):
    return ToolRuntime(
        state={"messages": [], "files": {}},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


class TestCompositeBackendArtifactsRoot:
    def test_default_artifacts_root(self) -> None:
        backend = CompositeBackend(default=StateBackend(), routes={})
        assert backend.artifacts_root == "/"

    def test_custom_artifacts_root(self) -> None:
        backend = CompositeBackend(default=StateBackend(), routes={}, artifacts_root="/workspace")
        assert backend.artifacts_root == "/workspace"


class TestFilesystemMiddlewareArtifactsRoot:
    def test_default_prefixes(self) -> None:
        mw = FilesystemMiddleware()
        assert mw.artifacts_root == "/"
        assert mw._large_tool_results_prefix == "/large_tool_results"
        assert mw._conversation_history_prefix == "/conversation_history"

    def test_custom_artifacts_root_prefixes(self) -> None:
        mw = FilesystemMiddleware(artifacts_root="/workspace")
        assert mw.artifacts_root == "/workspace"
        assert mw._large_tool_results_prefix == "/workspace/large_tool_results"
        assert mw._conversation_history_prefix == "/workspace/conversation_history"

    def test_trailing_slash_normalized(self) -> None:
        mw = FilesystemMiddleware(artifacts_root="/workspace/")
        assert mw._large_tool_results_prefix == "/workspace/large_tool_results"
        assert mw._conversation_history_prefix == "/workspace/conversation_history"

    def test_root_slash_no_double_slash(self) -> None:
        mw = FilesystemMiddleware(artifacts_root="/")
        assert mw._large_tool_results_prefix == "/large_tool_results"
        assert mw._conversation_history_prefix == "/conversation_history"

    def test_large_tool_result_eviction_uses_artifacts_root(self) -> None:
        backend, mem_store = _make_store_backend()
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100, artifacts_root="/workspace")
        runtime = _runtime("evict_123")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_123")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/evict_123" in result.content
        assert mem_store.get(("filesystem",), "/workspace/large_tool_results/evict_123") is not None

    def test_large_tool_result_eviction_default_root(self) -> None:
        backend, mem_store = _make_store_backend()
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100)
        runtime = _runtime("evict_456")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_456")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/large_tool_results/evict_456" in result.content
        assert mem_store.get(("filesystem",), "/large_tool_results/evict_456") is not None


class TestCreateSummarizationMiddlewareArtifactsRoot:
    def test_default_history_path_prefix(self) -> None:
        backend, _ = _make_store_backend()
        model = FakeChatModel(messages=iter([]))
        mw = create_summarization_middleware(model, backend)
        assert mw._history_path_prefix == "/conversation_history"
        assert mw.artifacts_root == "/"

    def test_custom_artifacts_root_history_path_prefix(self) -> None:
        backend, _ = _make_store_backend()
        model = FakeChatModel(messages=iter([]))
        mw = create_summarization_middleware(model, backend, artifacts_root="/workspace")
        assert mw._history_path_prefix == "/workspace/conversation_history"
        assert mw.artifacts_root == "/workspace"

    def test_trailing_slash_normalized(self) -> None:
        backend, _ = _make_store_backend()
        model = FakeChatModel(messages=iter([]))
        mw = create_summarization_middleware(model, backend, artifacts_root="/workspace/")
        assert mw._history_path_prefix == "/workspace/conversation_history"

    def test_root_slash_no_double_slash(self) -> None:
        backend, _ = _make_store_backend()
        model = FakeChatModel(messages=iter([]))
        mw = create_summarization_middleware(model, backend, artifacts_root="/")
        assert mw._history_path_prefix == "/conversation_history"


class TestCompositeBackendEvictionArtifactsRoot:
    """Tests for eviction with CompositeBackend and custom artifacts_root."""

    def test_large_tool_result_eviction(self) -> None:
        """Large tool result eviction writes to the custom artifacts_root path."""
        mem_store = InMemoryStore()
        store_backend = StoreBackend(store=mem_store, namespace=lambda _ctx: ("filesystem",))
        backend = CompositeBackend(
            default=store_backend,
            routes={},
            artifacts_root="/workspace",
        )
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100, artifacts_root="/workspace")
        runtime = _runtime("evict_ws")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="evict_ws")
        result = mw._intercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/evict_ws" in result.content
        assert mem_store.get(("filesystem",), "/workspace/large_tool_results/evict_ws") is not None
        assert mem_store.get(("filesystem",), "/large_tool_results/evict_ws") is None

    def test_summarization_history_prefix(self) -> None:
        """Summarization middleware uses the correct history prefix from artifacts_root."""
        backend, _ = _make_store_backend()
        model = FakeChatModel(messages=iter([]))
        mw = create_summarization_middleware(model, backend, artifacts_root="/workspace")
        assert mw._history_path_prefix == "/workspace/conversation_history"
        assert mw.artifacts_root == "/workspace"


class TestAsyncEvictionArtifactsRoot:
    """Tests for async eviction paths with custom artifacts_root."""

    async def test_async_large_tool_result_eviction_uses_artifacts_root(self) -> None:
        backend, mem_store = _make_store_backend()
        mw = FilesystemMiddleware(backend=backend, tool_token_limit_before_evict=100, artifacts_root="/workspace")
        runtime = _runtime("async_evict_123")

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="async_evict_123")
        result = await mw._aintercept_large_tool_result(msg, runtime)

        assert isinstance(result, ToolMessage)
        assert "/workspace/large_tool_results/async_evict_123" in result.content
        stored = await mem_store.aget(("filesystem",), "/workspace/large_tool_results/async_evict_123")
        assert stored is not None
