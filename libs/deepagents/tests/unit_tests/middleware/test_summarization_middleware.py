"""Unit tests for `SummarizationMiddleware` with backend offloading."""

import asyncio
import base64 as _base64
import hashlib
import inspect
import re
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol, EditResult, FileDownloadResponse, FileUploadResponse, ReadResult, WriteResult
from deepagents.middleware.summarization import (
    _MAX_TRACKED_HISTORY_TAILS,
    SummarizationMiddleware,
    _token_counter_accepts_tools,
    _upload_response_error,
)

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState

# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


def make_conversation_messages(
    num_old: int = 6,
    num_recent: int = 3,
    *,
    include_previous_summary: bool = False,
) -> list:
    """Create a realistic conversation message sequence.

    Args:
        num_old: Number of "old" messages that will be summarized
        num_recent: Number of "recent" messages to preserve
        include_previous_summary: If `True`, start with a summary `HumanMessage`
            containing placeholder text.

    Returns:
        List of messages simulating a conversation
    """
    messages: list[BaseMessage] = []

    if include_previous_summary:
        messages.append(
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nPrevious summary content...",
                additional_kwargs={"lc_source": "summarization"},
                id="summary-msg-0",
            )
        )

    # Add old messages (to be summarized)
    for i in range(num_old):
        if i % 3 == 0:
            messages.append(HumanMessage(content=f"User message {i}", id=f"human-{i}"))
        elif i % 3 == 1:
            messages.append(
                AIMessage(
                    content=f"AI response {i}",
                    id=f"ai-{i}",
                    tool_calls=[{"id": f"tool-call-{i}", "name": "test_tool", "args": {}}],
                )
            )
        else:
            messages.append(
                ToolMessage(
                    content=f"Tool result {i}",
                    tool_call_id=f"tool-call-{i - 1}",
                    id=f"tool-{i}",
                )
            )

    # Add recent messages (to be preserved)
    for i in range(num_recent):
        idx = num_old + i
        messages.append(HumanMessage(content=f"Recent message {idx}", id=f"recent-{idx}"))

    return messages


class MockBackend(BackendProtocol):
    """A mock backend that records read/write calls and can simulate failures."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        error_message: str | None = None,
        existing_content: str | None = None,
        download_raises: bool = False,
        write_raises: bool = False,
    ) -> None:
        """Initialize the mock backend.

        Args:
            should_fail: If `True`, write operations will simulate a failure.
            error_message: The error message to return on failure.
            existing_content: Initialize the backend with existing content for reads.
            download_raises: If `True`, `download_files` will raise an exception.
            write_raises: If `True`, `write`/`edit` will raise an exception.
        """
        self.write_calls: list[tuple[str, str]] = []
        self.edit_calls: list[tuple[str, str, str]] = []
        self.read_calls: list[str] = []
        self.download_files_calls: list[list[str]] = []
        self.should_fail = should_fail
        self.error_message = error_message
        self.existing_content = existing_content
        self.download_raises = download_raises
        self.write_raises = write_raises

    def read(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.read_calls.append(path)
        if self.existing_content is not None:
            return ReadResult(file_data={"content": self.existing_content, "encoding": "utf-8"})
        return ReadResult(file_data={"content": "", "encoding": "utf-8"})

    async def aread(self, path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(path, offset, limit)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files - returns raw content as bytes."""
        self.download_files_calls.append(paths)
        if self.download_raises:
            msg = "Mock download_files exception"
            raise RuntimeError(msg)
        responses = []
        for path in paths:
            if self.existing_content is not None:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=self.existing_content.encode("utf-8"),
                        error=None,
                    )
                )
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=None,
                        error="file_not_found",
                    )
                )
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        if self.download_raises:
            msg = "Mock adownload_files exception"
            raise RuntimeError(msg)
        return self.download_files(paths)

    def write(self, path: str, content: str) -> WriteResult:
        self.write_calls.append((path, content))
        if self.write_raises:
            msg = "Mock write exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return WriteResult(error=self.error_message or "Mock write failure")
        return WriteResult(path=path)

    async def awrite(self, path: str, content: str) -> WriteResult:
        if self.write_raises:
            msg = "Mock awrite exception"
            raise RuntimeError(msg)
        return self.write(path, content)

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Edit a file by replacing string occurrences."""
        self.edit_calls.append((path, old_string, new_string))
        if self.write_raises:
            msg = "Mock edit exception"
            raise RuntimeError(msg)
        if self.should_fail:
            return EditResult(error=self.error_message or "Mock edit failure")
        return EditResult(path=path, occurrences=1)

    async def aedit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        """Async version of edit."""
        if self.write_raises:
            msg = "Mock aedit exception"
            raise RuntimeError(msg)
        return self.edit(path, old_string, new_string, replace_all)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Record binary file uploads."""
        responses = []
        for path, _ in files:
            self.write_calls.append((path, "<binary>"))
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self.upload_files(files)


def make_mock_runtime() -> MagicMock:
    """Create a mock `Runtime`.

    Note: `Runtime` does not have a `config` attribute. Config is accessed
    via `get_config()` from langgraph's contextvar. Use `mock_get_config()`
    to control thread_id in tests.
    """
    runtime = MagicMock()
    runtime.context = {}
    runtime.stream_writer = MagicMock()
    runtime.store = None
    # Explicitly don't set runtime.config - it doesn't exist on real Runtime
    del runtime.config
    return runtime


@contextmanager
def mock_get_config(thread_id: str | None = "test-thread-123") -> Generator[None, None, None]:
    """Context manager to mock `get_config()` with a specific `thread_id`.

    Args:
        thread_id: The `thread_id` to return, or `None` to simulate missing config.

    Yields:
        `None` - use as a context manager around test code.
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id is not None else {"configurable": {}}

    with patch("deepagents.middleware.summarization.get_config", return_value=config):
        yield


def make_mock_model(summary_response: str = "This is a test summary.") -> MagicMock:
    """Create a mock LLM model for summarization.

    Args:
        summary_response: The text to return as the summary for testing purposes.
    """
    model = MagicMock()
    model.invoke.return_value = MagicMock(text=summary_response)
    model._llm_type = "test-model"
    model.profile = {"max_input_tokens": 100000}
    model._get_ls_params.return_value = {"ls_provider": "test"}
    return model


def make_model_request(
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> ModelRequest:
    """Create a ModelRequest from a state dict.

    Args:
        state: The agent state containing messages.
        runtime: The runtime object.

    Returns:
        A ModelRequest suitable for calling wrap_model_call.
    """
    mock_model = make_mock_model()
    return ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=None,
        runtime=runtime,
        state=state,
    )


def call_wrap_model_call(
    middleware: SummarizationMiddleware,
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> tuple["ModelResponse | ExtendedModelResponse", ModelRequest | None]:
    """Helper to call wrap_model_call and capture what was passed to handler.

    Args:
        middleware: The middleware instance to test.
        state: The agent state.
        runtime: The runtime object.

    Returns:
        Tuple of (result, modified_request) where:
        - result is the return value from wrap_model_call
        - modified_request is the request passed to the handler (or None if handler wasn't called)
    """
    request = make_model_request(state, runtime)
    captured_request = None

    def handler(req: ModelRequest) -> "ModelResponse":
        nonlocal captured_request
        captured_request = req
        # Return a mock response
        return AIMessage(content="Mock response")

    result = middleware.wrap_model_call(request, handler)
    return result, captured_request


async def call_awrap_model_call(
    middleware: SummarizationMiddleware,
    state: "AgentState[Any]",
    runtime: Any,  # noqa: ANN401
) -> tuple["ModelResponse | ExtendedModelResponse", ModelRequest | None]:
    """Helper to call awrap_model_call and capture what was passed to handler (async version).

    Args:
        middleware: The middleware instance to test.
        state: The agent state.
        runtime: The runtime object.

    Returns:
        Tuple of (result, modified_request) where:
        - result is the return value from awrap_model_call
        - modified_request is the request passed to the handler (or None if handler wasn't called)
    """
    request = make_model_request(state, runtime)
    captured_request = None

    async def handler(req: ModelRequest) -> "ModelResponse":
        nonlocal captured_request
        captured_request = req
        # Return a mock response
        return AIMessage(content="Mock response")

    result = await middleware.awrap_model_call(request, handler)
    return result, captured_request


# -----------------------------------------------------------------------------


class TestSummarizationMiddlewareInit:
    """Tests for middleware initialization."""

    def test_init_with_backend(self) -> None:
        """Test initialization with a backend instance."""
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert middleware._backend is backend
        assert middleware._history_path_prefix == "/conversation_history"

    def test_init_with_backend_factory(self) -> None:
        """Test initialization with a backend factory function."""
        backend = MockBackend()
        factory = lambda _rt: backend  # noqa: E731

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 3),
        )

        assert callable(middleware._backend)

    def test_deprecated_history_path_prefix_warns_and_applies(self) -> None:
        """Passing history_path_prefix emits a deprecation warning and is used."""
        backend = MockBackend()
        with pytest.warns(match="history_path_prefix"):
            middleware = SummarizationMiddleware(
                model=make_mock_model(),
                backend=backend,
                trigger=("messages", 5),
                keep=("messages", 3),
                history_path_prefix="/custom/history",
            )

        assert middleware._history_path_prefix == "/custom/history"
        assert middleware._media_prefix == "/custom/history/media"

    def test_deprecated_history_path_prefix_overrides_default(self) -> None:
        """Deprecated history_path_prefix takes precedence over the default."""
        backend = MockBackend()
        with pytest.warns(match="history_path_prefix"):
            middleware = SummarizationMiddleware(
                model=make_mock_model(),
                backend=backend,
                trigger=("messages", 5),
                keep=("messages", 3),
                history_path_prefix="/overridden",
            )

        assert middleware._history_path_prefix == "/overridden"
        assert middleware._history_path_prefix != "/conversation_history"


class TestOffloadingBasic:
    """Tests for basic offloading behavior."""

    def test_offload_writes_to_backend(self) -> None:
        """Test that summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should have triggered summarization
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert "_summarization_event" in result.command.update
        assert len(backend.write_calls) == 1

        path, content = backend.write_calls[0]
        assert path == "/conversation_history/test-thread-123.md"

        assert "## Summarized at" in content
        assert '<message type="human">' in content or '<message type="ai">' in content

    def test_offload_preserves_image_urls(self) -> None:
        """Image URLs in evicted messages survive the offload (issue #2873).

        The default `get_buffer_string` drops image content; the XML format keeps
        image URLs, so the archived markdown still references them.
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        image_url = "https://example.com/diagram.png"
        messages: list[BaseMessage] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is the diagram"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
                id="img-msg",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        assert len(backend.write_calls) == 1
        _, content = backend.write_calls[0]
        assert image_url in content

    def test_offload_rewrites_non_base64_data_url(self) -> None:
        """Non-base64 inline `data:` media (e.g. a URL-encoded SVG) is offloaded too.

        `get_buffer_string(format="xml")` drops *any* `data:` URL block, not just
        base64 ones, so a percent-encoded/plaintext `data:` URL must also be
        decoded (here via percent-decoding) to a file and rewritten to a path
        reference -- otherwise the inline media silently disappears from the
        archive (PR #3990 review).
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        svg_url = "data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3C%2Fsvg%3E"
        raw_svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'  # percent-decoded payload
        expected_key = hashlib.sha256(raw_svg).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.svg"

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is the inline SVG"},
                    {"type": "image", "url": svg_url},
                ],
                id="svg-data-url",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        media_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert media_uploads == [(expected_path, "<binary>")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert "Here is the inline SVG" in archive_write[1]
        # Referenced by path; the raw inline data: URL is gone and nothing failed.
        assert f'<image url="{expected_path}" />' in archive_write[1]
        assert svg_url not in archive_write[1]
        assert 'error="failed_to_offload"' not in archive_write[1]

    def test_offload_rewrites_base64_images(self) -> None:
        """Base64 image data is decoded to files and referenced by path in the archive (issue #2873).

        `get_buffer_string(format="xml")` drops base64 blocks, so the middleware
        decodes each one to `/conversation_history/media/{sha256}.{ext}` with
        `upload_files`, then rewrites the block to `<image url="…" />`. This
        covers all three base64 shapes recognized by langchain_core.
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # 1x1 transparent PNG — minimal valid PNG bytes
        raw_png = _base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        b64 = _base64.b64encode(raw_png).decode()
        expected_key = hashlib.sha256(raw_png).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.png"

        messages: list[BaseMessage] = [
            # Shape 1: explicit base64 field
            HumanMessage(
                content=[
                    {"type": "text", "text": "explicit base64 field"},
                    {"type": "image", "base64": b64, "mime_type": "image/png"},
                ],
                id="b64-field",
            ),
            # Shape 2: top-level data: URL
            HumanMessage(
                content=[{"type": "image", "url": f"data:image/png;base64,{b64}"}],
                id="b64-url",
            ),
            # Shape 3: OpenAI-style image_url with data: URL
            HumanMessage(
                content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
                id="b64-openai",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        # Identical images are deduped — only one upload despite three messages.
        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert len(image_uploads) == 1
        assert image_uploads[0][0] == expected_path

        # Archive markdown references the image path three times (once per message).
        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert archive_write[1].count(expected_path) == 3
        # No raw base64 payload in the archive.
        assert b64 not in archive_write[1]

    def test_offload_media_uses_deprecated_history_path_prefix(self) -> None:
        """Media files are written under the final history prefix."""
        backend = MockBackend()
        mock_model = make_mock_model()

        with pytest.warns(match="history_path_prefix"):
            middleware = SummarizationMiddleware(
                model=mock_model,
                backend=backend,
                trigger=("messages", 5),
                keep=("messages", 2),
                history_path_prefix="/custom/history",
            )

        raw_png = _base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        b64 = _base64.b64encode(raw_png).decode()
        expected_key = hashlib.sha256(raw_png).hexdigest()[:16]
        expected_path = f"/custom/history/media/{expected_key}.png"

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[{"type": "image", "url": f"data:image/png;base64,{b64}"}],
                id="custom-prefix-image",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/custom/history/media/")]
        assert len(image_uploads) == 1
        assert image_uploads[0][0] == expected_path

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert archive_write[0] == "/custom/history/test-thread-123.md"
        assert expected_path in archive_write[1]

    def test_offload_per_block_upload_failure(self) -> None:
        """A failed upload writes a placeholder while other images are still rewritten (issue #2873).

        Per-block failure tracking ensures one failed upload does not discard the
        entire batch. The successfully uploaded image gets a path reference in the
        archive; the failed image is replaced with an
        `<image error="failed_to_offload" />` text placeholder.
        """
        raw_a = b"\x89PNG_FAKE_DATA_A"
        raw_b = b"\x89PNG_FAKE_DATA_B"
        b64_a = _base64.b64encode(raw_a).decode()
        b64_b = _base64.b64encode(raw_b).decode()
        key_a = hashlib.sha256(raw_a).hexdigest()[:16]
        key_b = hashlib.sha256(raw_b).hexdigest()[:16]
        expected_path_a = f"/conversation_history/media/{key_a}.png"
        expected_path_b = f"/conversation_history/media/{key_b}.png"

        class PartialUploadBackend(MockBackend):
            """Backend that raises for any upload path containing key_b."""

            def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                responses = []
                for path, _content in files:
                    if key_b in path:
                        msg = f"Simulated upload failure for {path}"
                        raise RuntimeError(msg)
                    self.write_calls.append((path, "<binary>"))
                    responses.append(FileUploadResponse(path=path, error=None))
                return responses

            async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                return self.upload_files(files)

        backend = PartialUploadBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[{"type": "image", "base64": b64_a, "mime_type": "image/png"}],
                id="img-a",
            ),
            HumanMessage(
                content=[{"type": "image", "base64": b64_b, "mime_type": "image/png"}],
                id="img-b",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
            call_wrap_model_call(middleware, state, runtime)

        # Only image A was uploaded — B's upload raised.
        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert len(image_uploads) == 1
        assert image_uploads[0][0] == expected_path_a

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        # Image A is referenced by path in the archive.
        assert expected_path_a in archive_write[1]
        # Image B gets an error placeholder, not a path reference or raw base64.
        assert expected_path_b not in archive_write[1]
        assert b64_b not in archive_write[1]
        assert 'error="failed_to_offload"' in archive_write[1]

    def test_offload_upload_response_error_writes_placeholder(self) -> None:
        """Upload responses with errors are treated as failed image offloads."""
        raw_a = b"\x89PNG_RESPONSE_OK"
        raw_b = b"\x89PNG_RESPONSE_DENIED"
        b64_a = _base64.b64encode(raw_a).decode()
        b64_b = _base64.b64encode(raw_b).decode()
        key_a = hashlib.sha256(raw_a).hexdigest()[:16]
        key_b = hashlib.sha256(raw_b).hexdigest()[:16]
        expected_path_a = f"/conversation_history/media/{key_a}.png"
        expected_path_b = f"/conversation_history/media/{key_b}.png"

        class ResponseErrorBackend(MockBackend):
            """Backend that reports one image upload failure in the response."""

            def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                responses = []
                for path, _content in files:
                    if key_b in path:
                        responses.append(FileUploadResponse(path=path, error="permission_denied"))
                    else:
                        self.write_calls.append((path, "<binary>"))
                        responses.append(FileUploadResponse(path=path, error=None))
                return responses

            async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                return self.upload_files(files)

        backend = ResponseErrorBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages: list[BaseMessage] = [
            HumanMessage(content=[{"type": "image", "base64": b64_a, "mime_type": "image/png"}], id="img-a"),
            HumanMessage(content=[{"type": "image", "base64": b64_b, "mime_type": "image/png"}], id="img-b"),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
            call_wrap_model_call(middleware, state, runtime)

        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert image_uploads == [(expected_path_a, "<binary>")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert expected_path_a in archive_write[1]
        assert expected_path_b not in archive_write[1]
        assert b64_b not in archive_write[1]
        assert 'error="failed_to_offload"' in archive_write[1]

    def test_offload_all_uploads_raise_writes_placeholders(self) -> None:
        """All-raised upload failures still rewrite raw base64 to placeholders."""
        raw_png = b"\x89PNG_ALL_RAISE"
        b64 = _base64.b64encode(raw_png).decode()

        class RaisingUploadBackend(MockBackend):
            """Backend that raises for every image upload."""

            def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                msg = f"Simulated upload failure for {files[0][0]}"
                raise RuntimeError(msg)

            async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
                return self.upload_files(files)

        backend = RaisingUploadBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages: list[BaseMessage] = [
            HumanMessage(content=[{"type": "image", "base64": b64, "mime_type": "image/png"}], id="img"),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
            call_wrap_model_call(middleware, state, runtime)

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert b64 not in archive_write[1]
        assert 'error="failed_to_offload"' in archive_write[1]

    def test_offload_rewrites_non_image_media(self) -> None:
        """Non-image base64 media is offloaded and referenced with its own block type.

        The offload path is media-generic: an audio block is decoded to
        `/conversation_history/media/{sha256}.{ext}` and rewritten so the XML
        archive renders an `<audio url="..." />` reference rather than
        mislabeling it as an image.
        """
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        raw_audio = b"ID3_FAKE_MP3_DATA"
        b64 = _base64.b64encode(raw_audio).decode()
        expected_key = hashlib.sha256(raw_audio).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.mp3"

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[{"type": "audio", "base64": b64, "mime_type": "audio/mpeg"}],
                id="audio-msg",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert image_uploads == [(expected_path, "<binary>")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        # Rendered as an audio reference, not an image, and not raw base64.
        assert f'<audio url="{expected_path}" />' in archive_write[1]
        assert b64 not in archive_write[1]

    def test_offload_decode_failure_writes_placeholder(self) -> None:
        """A detected-but-undecodable base64 block becomes a placeholder, not a silent drop.

        A base64 `data:` URL with an undecodable payload is recognized as media
        but fails to decode. The block must still surface as an
        `<image error="failed_to_offload" />` placeholder so the archive records
        that media was present, and the failure must count toward the user-facing
        warning -- decode failures are as unrecoverable as upload failures.
        """
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # Detected as a base64 data URL (has the comma + "base64" header), but the
        # single-character payload is undecodable -> _decode_data_url returns None.
        malformed = "data:image/png;base64,a"
        messages: list[BaseMessage] = [
            HumanMessage(content=[{"type": "image", "url": malformed}], id="bad-img"),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
            call_wrap_model_call(middleware, state, runtime)

        # Nothing was uploaded (decode failed before any upload).
        assert not [p for p, _ in backend.write_calls if p.startswith("/conversation_history/media/")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert 'error="failed_to_offload"' in archive_write[1]
        assert malformed not in archive_write[1]

    def test_offload_non_av_media_uses_file_reference(self) -> None:
        """Non-(image/audio/video) media falls back to an XML-escaped `<file>` reference.

        A PDF has no typed XML block, so `_media_reference_block` emits a text
        block `<file url="..." />`. The XML renderer escapes text blocks, so the
        archive contains the escaped form `&lt;file url="..." /&gt;` -- the
        reference is preserved (not dropped) but rendered as literal text.
        """
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        raw_pdf = b"%PDF-1.4 fake pdf bytes"
        b64 = _base64.b64encode(raw_pdf).decode()
        expected_key = hashlib.sha256(raw_pdf).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.pdf"

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[{"type": "file", "base64": b64, "mime_type": "application/pdf"}],
                id="pdf-msg",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        media_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert media_uploads == [(expected_path, "<binary>")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        # Text-block fallback renders XML-escaped, and raw base64 is absent.
        assert f'&lt;file url="{expected_path}" /&gt;' in archive_write[1]
        assert b64 not in archive_write[1]

    def test_offload_no_inline_media_returns_messages_unchanged(self) -> None:
        """With no inline media, offload returns the original list and uploads nothing.

        Guards the `saw_inline_media` short-circuit: text-only messages must skip the
        copy/rewrite path entirely (identity return) so no media files are written.
        """
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=5, num_recent=2)
        result, failed = middleware._offload_inline_media(backend, messages)

        assert result is messages  # identity: no copy when there is no inline media
        assert failed == 0
        assert not [p for p, _ in backend.write_calls if p.startswith("/conversation_history/media/")]

    def test_offload_base64_alongside_non_standard_block(self) -> None:
        """Base64 media offloads cleanly even when a non-standard block shares the message.

        A provider-specific `tool_use` block normalizes to a `non_standard`
        content block when rewritten. This must not corrupt the archive: the
        image is still referenced by path and no raw base64 leaks, regardless of
        the sibling block.
        """
        backend = MockBackend()
        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        raw_png = b"\x89PNG_WITH_TOOL_USE"
        b64 = _base64.b64encode(raw_png).decode()
        expected_key = hashlib.sha256(raw_png).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.png"

        messages: list[BaseMessage] = [
            AIMessage(
                content=[
                    {"type": "text", "text": "calling a tool with an image"},
                    {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": 1}},
                    {"type": "image", "base64": b64, "mime_type": "image/png"},
                ],
                id="ai-tool-img",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        media_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert media_uploads == [(expected_path, "<binary>")]

        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert expected_path in archive_write[1]
        assert b64 not in archive_write[1]

    def test_upload_runs_once_before_offload_and_summary(self) -> None:
        """Image upload runs once and the result is shared by offload and summary.

        `_create_summary` uses prefix format internally, so image blocks are
        stripped from the summary prompt regardless. This test verifies that raw
        base64 never reaches the summary model prompt, the archive gets the
        uploaded image path, and each unique image is uploaded only once.
        """
        raw_png = _base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        b64 = _base64.b64encode(raw_png).decode()
        expected_key = hashlib.sha256(raw_png).hexdigest()[:16]
        expected_path = f"/conversation_history/media/{expected_key}.png"

        backend = MockBackend()
        mock_model = make_mock_model()
        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages: list[BaseMessage] = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Here is an image:"},
                    {"type": "image", "base64": b64, "mime_type": "image/png"},
                ],
                id="b64-msg",
            ),
            *make_conversation_messages(num_old=5, num_recent=2),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config():
            call_wrap_model_call(middleware, state, runtime)

        # upload_files is called exactly once even though both offload and
        # summary consume the result.
        image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
        assert len(image_uploads) == 1
        assert image_uploads[0][0] == expected_path

        # Archive has the uploaded image path.
        archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
        assert expected_path in archive_write[1]

        # Raw base64 never reaches the summary model prompt.
        invoke_prompt: str = mock_model.invoke.call_args[0][0]
        assert b64 not in invoke_prompt

    def test_offload_appends_to_existing_content(self) -> None:
        """Test that second summarization appends to existing file."""
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        assert len(backend.edit_calls) == 1
        _, old_string, new_string = backend.edit_calls[0]

        # old_string should be the existing content
        assert old_string == existing

        # new_string (combined content) should contain both old and new sections
        assert "## Summarized at 2024-01-01T00:00:00Z" in new_string
        expected_section_count = 2  # One existing + one new summarization section
        assert new_string.count("## Summarized at") == expected_section_count

    def test_typical_tool_heavy_conversation(self) -> None:
        """Test with a realistic tool-heavy conversation pattern.

        Simulates:

        ```txt
        HumanMessage -> AIMessage(tool_calls) -> ToolMessage -> ToolMessage ->
        ToolMessage -> AIMessage -> HumanMessage -> AIMessage -> ToolMessage (trigger)
        ```
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 8),
            keep=("messages", 3),
        )

        messages = [
            HumanMessage(content="Search for Python tutorials", id="h1"),
            AIMessage(
                content="I'll search for Python tutorials.",
                id="a1",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "python"}}],
            ),
            ToolMessage(content="Result 1: Python basics", tool_call_id="tc1", id="t1"),
            ToolMessage(content="Result 2: Advanced Python", tool_call_id="tc1", id="t2"),
            ToolMessage(content="Result 3: Python projects", tool_call_id="tc1", id="t3"),
            AIMessage(content="Here are some Python tutorials I found...", id="a2"),
            HumanMessage(content="Show me the first one", id="h2"),
            AIMessage(
                content="Let me get that for you.",
                id="a3",
                tool_calls=[{"id": "tc2", "name": "fetch", "args": {"url": "..."}}],
            ),
            ToolMessage(content="Tutorial content...", tool_call_id="tc2", id="t4"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert len(backend.write_calls) == 1

        _, content = backend.write_calls[0]

        # Should have markdown content with summarized messages
        assert "## Summarized at" in content
        assert "Search for Python tutorials" in content

    def test_second_summarization_after_first(self) -> None:
        """Test a second summarization event after an initial one.

        Ensures the chained summarization correctly handles the existing summary message.
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization
        messages = [
            # Previous summary from first summarization
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None

        _, content = backend.write_calls[0]

        # The previous summary message (marked with lc_source: "summarization") should NOT
        # be offloaded—it's a synthetic message, and the original messages it summarized
        # are already stored in the backend file from the first summarization
        assert "First summary" not in content, "Previous summary should be filtered from offload"
        # But the new conversation messages should be offloaded
        assert "New question 1" in content

    def test_filters_previous_summary_messages(self) -> None:
        """Test that previous summary `HumanMessage` objects are NOT included in offload.

        When a second summarization occurs, the previous summary message should be
        filtered out since we already have the original messages stored.
        """
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # Create messages that include a previous summary
        messages = make_conversation_messages(
            num_old=6,
            num_recent=2,
            include_previous_summary=True,
        )
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        _, content = backend.write_calls[0]

        # Check that the offloaded content doesn't include "Previous summary content"
        # (which is the content of the summary message added by include_previous_summary)
        assert "Previous summary content" not in content, "Previous summary message should be filtered from offload"


class TestSummaryMessageFormat:
    """Tests for the summary message format with file path reference."""

    def test_summary_includes_file_path(self) -> None:
        """Test that summary message includes the file path reference."""
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Test summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="test-thread"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # Get the summary message (first in modified messages list)
        summary_msg = modified_request.messages[0]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/test-thread.md" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Test summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

    def test_summary_has_lc_source_marker(self) -> None:
        """Test that summary message has `lc_source=summarization` marker."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert modified_request is not None
        summary_msg = modified_request.messages[0]

        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"

    def test_summarization_aborts_on_backend_failure(self) -> None:
        """Test that summarization warns when backend write fails but still summarizes."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model(summary_response="Unused summary")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # file_path must be None so the summary message does not reference
        # a nonexistent backend path.
        event = result.command.update["_summarization_event"]
        assert event["file_path"] is None

    def test_summary_includes_file_path_after_second_summarization(self) -> None:
        """Test that summary message includes file path reference after multiple summarizations.

        This ensures the path reference is present even when a previous summary message
        exists in the conversation (i.e., chained summarization).
        """
        backend = MockBackend()
        mock_model = make_mock_model(summary_response="Second summary content")

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        # State after first summarization - starts with a summary message
        messages = [
            HumanMessage(
                content="Here is a summary of the conversation to date:\n\nFirst summary...",
                additional_kwargs={"lc_source": "summarization"},
                id="prev-summary",
            ),
            # New messages after first summary that trigger second summarization
            HumanMessage(content="New question 1", id="h1"),
            AIMessage(content="Answer 1", id="a1"),
            HumanMessage(content="New question 2", id="h2"),
            AIMessage(content="Answer 2", id="a2"),
            HumanMessage(content="New question 3", id="h3"),
            AIMessage(content="Answer 3", id="a3"),
        ]

        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="multi-summarize-thread"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # The summary message should be the first message
        summary_msg = modified_request.messages[0]

        # Should include the file path reference
        assert "full conversation history has been saved to" in summary_msg.content
        assert "/conversation_history/multi-summarize-thread.md" in summary_msg.content

        # Should include the summary in XML tags
        assert "<summary>" in summary_msg.content
        assert "Second summary content" in summary_msg.content
        assert "</summary>" in summary_msg.content

        # Should have lc_source marker
        assert summary_msg.additional_kwargs.get("lc_source") == "summarization"


class TestNoSummarizationTriggered:
    """Tests for when summarization threshold is not met."""

    def test_no_offload_when_below_threshold(self) -> None:
        """Test that no offload occurs when message count is below trigger."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization)
        assert not isinstance(result, ExtendedModelResponse)

        # No writes should have occurred
        assert len(backend.write_calls) == 0


def test_system_message_counts_for_trigger_only() -> None:
    """System message should affect token trigger but not be sent in messages."""
    backend = MockBackend()
    seen_system = {"counted": False}

    def token_counter(messages: list[BaseMessage]) -> int:
        if any(isinstance(msg, SystemMessage) for msg in messages):
            seen_system["counted"] = True
        return len(messages)

    middleware = SummarizationMiddleware(
        model=make_mock_model(),
        backend=backend,
        trigger=("tokens", 3),
        keep=("messages", 1),
        token_counter=token_counter,
    )

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = make_model_request(state, runtime).override(system_message=SystemMessage(content="sys"))

    captured_request = None

    def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    with mock_get_config():
        result = middleware.wrap_model_call(request, handler)

    assert isinstance(result, ExtendedModelResponse)
    assert seen_system["counted"] is True
    assert captured_request is not None
    assert captured_request.system_message is not None
    assert all(not isinstance(msg, SystemMessage) for msg in captured_request.messages)
    assert len(backend.write_calls) == 1


@pytest.mark.anyio
async def test_async_tools_passed_to_token_counter_for_summarization() -> None:
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))
    seen = {"tools": False, "system": False}

    def token_counter(messages: list[BaseMessage], *, tools: list[dict[str, Any]] | None = None) -> int:
        if tools:
            seen["tools"] = True
        if any(isinstance(msg, SystemMessage) for msg in messages):
            seen["system"] = True
        return 3 if seen["system"] else 2

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("tokens", 3),
        keep=("messages", 1),
        token_counter=token_counter,
    )

    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=SystemMessage(content="sys"),
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}],
        runtime=runtime,
        state=state,
    )

    captured_request = None

    async def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    with mock_get_config():
        result = await middleware.awrap_model_call(request, handler)

    assert isinstance(result, ExtendedModelResponse)
    assert seen["tools"]
    assert seen["system"] is True
    assert captured_request is not None
    assert all(not isinstance(msg, SystemMessage) for msg in captured_request.messages)


@pytest.mark.anyio
async def test_async_system_message_counts_for_truncate_trigger() -> None:
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

    def token_counter(messages: list[BaseMessage], *, tools: list[dict[str, Any]] | None = None) -> int:
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            msg = "system message not included"
            raise AssertionError(msg)
        assert tools is not None
        return 3

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        keep=("messages", 1),
        truncate_args_settings={
            "trigger": ("tokens", 3),
            "keep": ("messages", 1),
            "max_length": 40,
            "truncation_text": "...(argument truncated)",
        },
        token_counter=token_counter,
    )

    long_content = "x" * 100
    messages = [
        AIMessage(
            content="write",
            tool_calls=[{"id": "call-1", "name": "write_file", "args": {"content": long_content}}],
        ),
        HumanMessage(content="next"),
    ]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()
    request = ModelRequest(
        model=mock_model,
        messages=state["messages"],
        system_message=SystemMessage(content="sys"),
        runtime=runtime,
        state=state,
    )

    captured_request = None

    async def handler(req: ModelRequest) -> ModelResponse:
        nonlocal captured_request
        captured_request = req
        return AIMessage(content="Mock response")

    result = await middleware.awrap_model_call(request, handler)

    assert not isinstance(result, ExtendedModelResponse)
    assert captured_request is not None
    truncated_call = captured_request.messages[0].tool_calls[0]
    assert truncated_call["args"]["content"] == "x" * 20 + "...(argument truncated)"


class TestBackendFailureHandling:
    """Tests for backend failure handling - summarization aborts to prevent data loss."""

    def test_summarization_aborts_on_write_failure(self) -> None:
        """Test that summarization warns when backend write fails but still summarizes."""
        backend = MockBackend(should_fail=True, error_message="Storage unavailable")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # file_path must be None so the summary message does not reference
        # a nonexistent backend path.
        event = result.command.update["_summarization_event"]
        assert event["file_path"] is None

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Test that summarization warns when backend raises exception but still summarizes."""
        backend = MagicMock()
        backend.download_files.return_value = []
        backend.write.side_effect = Exception("Network error")
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

        # file_path must be None so the summary message does not reference
        # a nonexistent backend path.
        event = result.command.update["_summarization_event"]
        assert event["file_path"] is None


class TestThreadIdExtraction:
    """Tests for thread ID extraction via `get_config()`."""

    def test_thread_id_from_config(self) -> None:
        """Test that `thread_id` is correctly extracted from `get_config()`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id="custom-thread-456"):
            call_wrap_model_call(middleware, state, runtime)

        path, _ = backend.write_calls[0]
        assert path == "/conversation_history/custom-thread-456.md"

    def test_fallback_thread_id_when_missing(self) -> None:
        """Test that a fallback ID is generated when `thread_id` is not in config."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with mock_get_config(thread_id=None):
            call_wrap_model_call(middleware, state, runtime)

        path, _ = backend.write_calls[0]

        # Should have a generated session ID in the path
        assert "session_" in path
        assert path.endswith(".md")


class TestAsyncBehavior:
    """Tests for async version of `before_model`."""

    @pytest.mark.anyio
    async def test_async_offload_writes_to_backend(self) -> None:
        """Test that async summarization triggers a write to the backend."""
        backend = MockBackend()
        mock_model = make_mock_model()
        # Mock the async create summary
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_aborts_on_failure(self) -> None:
        """Test that async summarization warns on backend failure but still summarizes."""
        backend = MockBackend(should_fail=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None


class TestBackendFactoryInvocation:
    """Tests for backend factory invocation during summarization."""

    def test_backend_factory_invoked_during_summarization(self) -> None:
        """Test that backend factory is called with `ToolRuntime` during summarization."""
        backend = MockBackend()
        factory_called_with: list = []

        def factory(tool_runtime: object) -> MockBackend:
            factory_called_with.append(tool_runtime)
            return backend

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=factory,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        call_wrap_model_call(middleware, state, runtime)

        # Factory should have been called once
        assert len(factory_called_with) == 1
        # Backend should have received write call
        assert len(backend.write_calls) == 1


class TestMarkdownFormatting:
    """Tests for markdown message formatting using `get_buffer_string`."""

    def test_markdown_format_includes_message_content(self) -> None:
        """Test that markdown format includes message content."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None

        # Verify the offloaded content is markdown formatted
        _, content = backend.write_calls[0]

        # Should contain XML-tagged messages (get_buffer_string format="xml")
        assert '<message type="human">' in content or '<message type="ai">' in content
        # Should contain the actual message content
        assert "User message" in content


class TestDownloadFilesException:
    """Tests for exception handling when download_files raises."""

    def test_summarization_continues_on_download_files_exception(self) -> None:
        """Test that summarization continues when download_files raises an exception."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result, _ = call_wrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        # download_files was called (and raised)
        assert len(backend.download_files_calls) == 1
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1

    @pytest.mark.anyio
    async def test_async_summarization_continues_on_download_files_exception(self) -> None:
        """Test that async summarization continues when adownload_files raises."""
        backend = MockBackend(download_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        # Should not raise - summarization should continue
        result, _ = await call_awrap_model_call(middleware, state, runtime)

        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        # write should still be called (with no existing content)
        assert len(backend.write_calls) == 1


class TestWriteEditException:
    """Tests for exception handling when `write`/`edit` raises - summarization aborts."""

    def test_summarization_aborts_on_write_exception(self) -> None:
        """Test that summarization warns when `write` raises an exception but still summarizes.

        Covers lines 314-322: Exception handler for write in _offload_to_backend.
        """
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_write_exception(self) -> None:
        """Test that async summarization warns when `awrite` raises but still summarizes.

        Covers lines 387-395: Exception handler for awrite in _aoffload_to_backend.
        """
        backend = MockBackend(write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    def test_summarization_aborts_on_edit_exception(self) -> None:
        """Test that summarization warns when `edit` raises an exception but still summarizes (existing content).

        Covers lines 314-322: Exception handler for edit in _offload_to_backend.
        """
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = call_wrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None

    @pytest.mark.anyio
    async def test_async_summarization_aborts_on_edit_exception(self) -> None:
        """Test that async summarization warns when `aedit` raises but still summarizes (existing content).

        Covers lines 387-395: Exception handler for aedit in _aoffload_to_backend.
        """
        existing = "## Summarized at 2024-01-01T00:00:00Z\n\nHuman: Previous message\n\n"
        backend = MockBackend(existing_content=existing, write_raises=True)
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 5),
            keep=("messages", 2),
        )

        messages = make_conversation_messages(num_old=6, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        with pytest.warns(UserWarning, match="Offloading conversation history to backend failed"):
            result, modified_request = await call_awrap_model_call(middleware, state, runtime)

        # Should still produce summarization result despite backend failure
        assert isinstance(result, ExtendedModelResponse)
        assert result.command is not None
        assert result.command.update is not None
        assert modified_request is not None


class TestCutoffIndexEdgeCases:
    """Tests for edge cases where `cutoff_index <= 0`."""

    def test_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that no summarization occurs when `cutoff_index` is `0`."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = call_wrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization) because cutoff_index would be 0 or negative
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should occur
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_not_triggered(self) -> None:
        """Test that async `abefore_model` returns `None` when summarization not triggered."""
        backend = MockBackend()
        mock_model = make_mock_model()

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 100),  # High threshold
            keep=("messages", 3),
        )

        messages = make_conversation_messages(num_old=3, num_recent=2)
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization)
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should have occurred
        assert len(backend.write_calls) == 0

    @pytest.mark.anyio
    async def test_async_no_summarization_when_cutoff_index_zero(self) -> None:
        """Test that async `abefore_model` returns `None` when `cutoff_index <= 0`."""
        backend = MockBackend()
        mock_model = make_mock_model()
        mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

        middleware = SummarizationMiddleware(
            model=mock_model,
            backend=backend,
            trigger=("messages", 3),  # Trigger at 3 messages
            keep=("messages", 10),  # But keep 10 messages (more than we have)
        )

        # Create exactly 3 messages to trigger summarization check
        messages = [
            HumanMessage(content="Message 1", id="h1"),
            AIMessage(content="Reply 1", id="a1"),
            HumanMessage(content="Message 2", id="h2"),
        ]
        state = cast("AgentState[Any]", {"messages": messages})
        runtime = make_mock_runtime()

        result, _ = await call_awrap_model_call(middleware, state, runtime)

        # Should return ModelResponse (no summarization) because cutoff_index would be 0 or negative
        assert not isinstance(result, ExtendedModelResponse)
        # No writes should occur
        assert len(backend.write_calls) == 0


# -----------------------------------------------------------------------------
# Argument truncation tests
# -----------------------------------------------------------------------------


def test_no_truncation_when_trigger_is_none() -> None:
    """Test that no truncation occurs when truncate_args_settings is None."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings=None,  # Truncation disabled
    )

    # Create messages with large tool calls
    large_content = "x" * 200
    messages = [
        HumanMessage(content="Write a file", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should return ModelResponse (no truncation, no summarization)
    assert not isinstance(result, ExtendedModelResponse)


def test_truncate_old_write_file_tool_call() -> None:
    """Test that old write_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message with write_file tool call (will be cleaned)
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    # Check that the old tool call was cleaned
    first_ai_msg = cleaned_messages[0]
    assert isinstance(first_ai_msg, AIMessage)
    assert len(first_ai_msg.tool_calls) == 1
    assert first_ai_msg.tool_calls[0]["name"] == "write_file"
    # Content should be first 20 chars + truncation text
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_old_edit_file_tool_call() -> None:
    """Test that old edit_file tool calls with large arguments get truncated."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_old_string = "a" * 100
    large_new_string = "b" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "edit_file",
                    "args": {
                        "file_path": "/test.py",
                        "old_string": large_old_string,
                        "new_string": large_new_string,
                    },
                }
            ],
        ),
        ToolMessage(content="File edited", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["name"] == "edit_file"
    assert first_ai_msg.tool_calls[0]["args"]["old_string"] == "a" * 20 + "...(argument truncated)"
    assert first_ai_msg.tool_calls[0]["args"]["new_string"] == "b" * 20 + "...(argument truncated)"


def test_truncate_ignores_other_tool_calls() -> None:
    """Test that tool calls other than write_file and edit_file are not affected."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should return AIMessage since read_file is not cleaned (no truncation)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify read_file args are unchanged
    first_msg = modified_request.messages[0]
    assert isinstance(first_msg, AIMessage)
    assert first_msg.tool_calls[0]["args"]["content"] == large_content


def test_truncate_respects_recent_messages() -> None:
    """Test that recent messages are not cleaned."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 4),  # Keep last 4 messages
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a1"),
        # Recent message with write_file (should NOT be cleaned - it's in the last 4)
        AIMessage(
            content="",
            id="a2",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # No truncation should happen since the tool call is in the keep window (last 4 messages)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify the write_file content is unchanged
    write_file_msg = modified_request.messages[2]
    assert isinstance(write_file_msg, AIMessage)
    assert write_file_msg.tool_calls[0]["args"]["content"] == large_content


def test_truncate_with_token_keep_policy() -> None:
    """Test truncation with token-based keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()

    # Custom token counter that returns predictable counts
    def simple_token_counter(msgs: list) -> int:
        return len(msgs) * 100  # 100 tokens per message

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("tokens", 250),  # Keep ~2-3 messages
            "max_length": 100,
        },
        token_counter=simple_token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    # First message should be cleaned since it's outside the token window
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_with_fraction_trigger_and_keep() -> None:
    """Test truncation with fraction-based trigger and keep policy."""
    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.profile = {"max_input_tokens": 1000}

    # Custom token counter: 200 tokens per message
    def token_counter(msgs: list) -> int:
        return len(msgs) * 200

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold for summarization
        truncate_args_settings={
            "trigger": ("fraction", 0.5),  # Trigger at 50% of 1000 = 500 tokens
            "keep": ("fraction", 0.2),  # Keep 20% of 1000 = 200 tokens (~1 message)
            "max_length": 100,
        },
        token_counter=token_counter,
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Message 1", id="h1"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should trigger truncation: 3 messages * 200 = 600 tokens > 500 threshold
    # Should keep only ~200 tokens (1 message) from the end
    # So first 2 messages should be in truncation zone
    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_before_summarization() -> None:
    """Test that truncation happens before summarization."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Test summary")

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 10),  # Trigger summarization
        keep=("messages", 2),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 3),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        # Old message that will be cleaned and summarized
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
    ] + [HumanMessage(content=f"Message {i}", id=f"h{i}") for i in range(10)]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    with mock_get_config(thread_id="test-thread"):
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert modified_request is not None

    # Should have triggered both truncation and summarization
    # Backend should have received a write call for offloading
    assert len(backend.write_calls) == 1

    # Result should contain summary message
    new_messages = modified_request.messages
    assert any("summary" in str(msg.content).lower() for msg in new_messages)


def test_truncate_without_summarization() -> None:
    """Test that truncation can happen independently of summarization."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold, no summarization
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # No backend write (no summarization)
    assert len(backend.write_calls) == 0

    # But truncation should have happened
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages
    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


def test_truncate_preserves_small_arguments() -> None:
    """Test that small arguments are not truncated even in old messages."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    small_content = "short"

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": small_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # No modification should happen since content is small
    assert isinstance(result, AIMessage)
    assert modified_request is not None

    # Verify the content is unchanged
    first_msg = modified_request.messages[0]
    assert isinstance(first_msg, AIMessage)
    assert first_msg.tool_calls[0]["args"]["content"] == small_content


def test_truncate_mixed_tool_calls() -> None:
    """Test that only write_file and edit_file are cleaned in a message with multiple tool calls."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "read_file",
                    "args": {"file_path": "/test.txt"},
                },
                {
                    "id": "tc2",
                    "name": "write_file",
                    "args": {"file_path": "/output.txt", "content": large_content},
                },
                {
                    "id": "tc3",
                    "name": "shell",
                    "args": {"command": "ls -la"},
                },
            ],
        ),
        ToolMessage(content="File content", tool_call_id="tc1", id="t1"),
        ToolMessage(content="File written", tool_call_id="tc2", id="t2"),
        ToolMessage(content="Output", tool_call_id="tc3", id="t3"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert len(first_ai_msg.tool_calls) == 3

    # read_file should be unchanged
    assert first_ai_msg.tool_calls[0]["name"] == "read_file"
    assert first_ai_msg.tool_calls[0]["args"]["file_path"] == "/test.txt"

    # write_file should be cleaned
    assert first_ai_msg.tool_calls[1]["name"] == "write_file"
    assert first_ai_msg.tool_calls[1]["args"]["content"] == "x" * 20 + "...(argument truncated)"

    # shell should be unchanged
    assert first_ai_msg.tool_calls[2]["name"] == "shell"
    assert first_ai_msg.tool_calls[2]["args"]["command"] == "ls -la"


def test_truncate_custom_truncation_text() -> None:
    """Test that custom truncation text is used."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 50,
            "truncation_text": "[TRUNCATED]",
        },
    )

    large_content = "y" * 100

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "y" * 20 + "[TRUNCATED]"


@pytest.mark.anyio
async def test_truncate_async_works() -> None:
    """Test that async argument truncation works correctly."""
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),
        truncate_args_settings={
            "trigger": ("messages", 5),
            "keep": ("messages", 2),
            "max_length": 100,
        },
    )

    large_content = "x" * 200

    messages = [
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "write_file",
                    "args": {"file_path": "/test.txt", "content": large_content},
                }
            ],
        ),
        ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
        HumanMessage(content="Request 1", id="h1"),
        AIMessage(content="Response 1", id="a2"),
        HumanMessage(content="Request 2", id="h2"),
        AIMessage(content="Response 2", id="a3"),
    ]

    state = {"messages": messages}
    runtime = make_mock_runtime()

    result, modified_request = await call_awrap_model_call(middleware, state, runtime)

    # Truncation only - returns AIMessage (ModelResponse)
    assert isinstance(result, AIMessage)
    assert modified_request is not None
    # Truncation modifies messages inline
    cleaned_messages = modified_request.messages

    first_ai_msg = cleaned_messages[0]
    assert first_ai_msg.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


# -----------------------------------------------------------------------------
# Chained summarization cutoff index tests
# -----------------------------------------------------------------------------


def test_chained_summarization_cutoff_index() -> None:
    """Test that state_cutoff_index is computed correctly across three chained summarizations.

    The formula is:
        state_cutoff = old_state_cutoff + effective_cutoff - 1

    The -1 accounts for the synthetic summary message at effective[0] which does not
    correspond to any state message.

    Setup: trigger=("messages", 5), keep=("messages", 2).

    Round 1 (no previous event):
        State: [S0..S7] (8 messages), cutoff = 8 - 2 = 6.
        Preserved: [S6, S7]. Event: cutoff_index=6.

    Round 2 (previous cutoff=6):
        State: [S0..S13] (14 messages).
        effective = [summary_1, S6..S13] (9 messages), effective cutoff = 9 - 2 = 7.
        state_cutoff = 6 + 7 - 1 = 12. Preserved: [S12, S13].

    Round 3 (previous cutoff=12):
        State: [S0..S19] (20 messages).
        effective = [summary_2, S12..S19] (9 messages), effective cutoff = 9 - 2 = 7.
        state_cutoff = 12 + 7 - 1 = 18. Preserved: [S18, S19].
    """
    backend = MockBackend()
    mock_model = make_mock_model()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )
    runtime = make_mock_runtime()

    def make_state_messages(n: int) -> list:
        return [HumanMessage(content=f"S{i}", id=f"s{i}") if i % 2 == 0 else AIMessage(content=f"S{i}", id=f"s{i}") for i in range(n)]

    def offloaded_labels(write_call_content: str) -> list[str]:
        """Extract S-labels from backend write content (e.g. `<message ...>S0</message>` -> "S0")."""
        return re.findall(r"S\d+", write_call_content)

    def appended_labels(edit_call_index: int) -> list[str]:
        """S-labels added by an anchored append.

        The first offload writes the file; later ones append after the previous
        section, so the newly offloaded messages are the part of `new_string`
        beyond the anchor.
        """
        _, old_string, new_string = backend.edit_calls[edit_call_index]
        assert new_string.startswith(old_string)
        return offloaded_labels(new_string[len(old_string) :])

    # --- Round 1: first summarization, no previous event ---
    state = cast("AgentState[Any]", {"messages": make_state_messages(8)})
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_1 = result.command.update["_summarization_event"]
    assert event_1["cutoff_index"] == 6
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S6", "S7"]
    _, content = backend.write_calls[0]
    assert offloaded_labels(content) == ["S0", "S1", "S2", "S3", "S4", "S5"]

    # --- Round 2: second summarization, feed back event from round 1 ---
    state = cast(
        "AgentState[Any]",
        {"messages": make_state_messages(14), "_summarization_event": event_1},
    )
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_2 = result.command.update["_summarization_event"]
    assert event_2["cutoff_index"] == 12
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S12", "S13"]
    assert appended_labels(0) == ["S6", "S7", "S8", "S9", "S10", "S11"]

    # --- Round 3: third summarization, feed back event from round 2 ---
    state = cast(
        "AgentState[Any]",
        {"messages": make_state_messages(20), "_summarization_event": event_2},
    )
    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    event_3 = result.command.update["_summarization_event"]
    assert event_3["cutoff_index"] == 18
    assert modified_request is not None
    assert [m.content for m in modified_request.messages[1:]] == ["S18", "S19"]
    assert appended_labels(1) == ["S12", "S13", "S14", "S15", "S16", "S17"]


# -----------------------------------------------------------------------------
# ContextOverflowError fallback tests
# -----------------------------------------------------------------------------


def test_context_overflow_triggers_summarization() -> None:
    """Test that ContextOverflowError triggers fallback to summarization."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Fallback summary")

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold - won't trigger normally
        keep=("messages", 2),
    )

    messages = make_conversation_messages(num_old=6, num_recent=2)
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    # Create a handler that raises ContextOverflowError on first call
    call_count = {"count": 0}

    def handler_with_overflow(_req: ModelRequest) -> "ModelResponse":
        call_count["count"] += 1
        if call_count["count"] == 1:
            # First call with unsummarized messages throws overflow
            raise ContextOverflowError
        # Second call with summarized messages succeeds
        return AIMessage(content="Success after summarization")

    request = make_model_request(state, runtime)

    with mock_get_config():
        result = middleware.wrap_model_call(request, handler_with_overflow)

    # Should have triggered summarization as fallback
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert result.command.update["_summarization_event"]

    # Should have called handler twice (once failed, once succeeded)
    assert call_count["count"] == 2

    # Backend should have offloaded messages
    assert len(backend.write_calls) == 1


@pytest.mark.anyio
async def test_async_context_overflow_triggers_summarization() -> None:
    """Test that ContextOverflowError triggers fallback to summarization (async)."""
    backend = MockBackend()
    mock_model = make_mock_model(summary_response="Fallback summary")
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async fallback summary"))

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 100),  # High threshold - won't trigger normally
        keep=("messages", 2),
    )

    messages = make_conversation_messages(num_old=6, num_recent=2)
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    # Create a handler that raises ContextOverflowError on first call
    call_count = {"count": 0}

    async def handler_with_overflow(_req: ModelRequest) -> "ModelResponse":
        call_count["count"] += 1
        if call_count["count"] == 1:
            # First call with unsummarized messages throws overflow
            raise ContextOverflowError
        # Second call with summarized messages succeeds
        return AIMessage(content="Success after summarization")

    request = make_model_request(state, runtime)

    with mock_get_config():
        result = await middleware.awrap_model_call(request, handler_with_overflow)

    # Should have triggered summarization as fallback
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update

    # Should have called handler twice (once failed, once succeeded)
    assert call_count["count"] == 2

    # Backend should have offloaded messages
    assert len(backend.write_calls) == 1


def test_profile_inference_triggers_summary() -> None:
    """Ensure automatic profile inference triggers summarization when limits are exceeded."""

    def token_counter(messages: list[BaseMessage], **_kwargs: Any) -> int:
        return len(messages) * 200

    # Create a mock model with profile
    mock_model = make_mock_model()
    mock_model.profile = {"max_input_tokens": 1000}

    backend = MockBackend()

    # Test 1: Don't engage summarization when below threshold
    # total_tokens = 4 * 200 = 800, threshold = 0.81 * 1000 = 810
    # 800 < 810, so no summarization
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.81),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    messages: list[BaseMessage] = [
        HumanMessage(content="Message 1", id="h1"),
        AIMessage(content="Message 2", id="a1"),
        HumanMessage(content="Message 3", id="h2"),
        AIMessage(content="Message 4", id="a2"),
    ]

    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should not trigger summarization
    assert not isinstance(result, ExtendedModelResponse)
    assert len(backend.write_calls) == 0

    # Test 2: Engage summarization when at threshold
    # total_tokens = 4 * 200 = 800, threshold = 0.80 * 1000 = 800
    # 800 >= 800, so summarization triggers
    backend = MockBackend()  # Reset backend
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.5),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    # Should trigger summarization
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update
    assert len(backend.write_calls) == 1

    # Check the modified messages
    assert modified_request is not None
    summary_message = modified_request.messages[0]
    assert isinstance(summary_message, HumanMessage)
    assert "summarized" in summary_message.content.lower()
    assert "<summary>" in summary_message.content

    # Should preserve last 2 messages (keep=0.5 * 1000 = 500 tokens, 500/200 = 2.5 messages)
    preserved_messages = modified_request.messages[1:]
    assert len(preserved_messages) == 2
    assert [msg.content for msg in preserved_messages] == ["Message 3", "Message 4"]

    # Test 3: With keep=("fraction", 0.6), preserve more messages
    # target tokens = 0.6 * 1000 = 600, 600/200 = 3 messages
    backend = MockBackend()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.6),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, modified_request = call_wrap_model_call(middleware, state, runtime)

    assert isinstance(result, ExtendedModelResponse)
    assert modified_request is not None
    preserved_messages = modified_request.messages[1:]
    assert len(preserved_messages) == 3
    assert [msg.content for msg in preserved_messages] == ["Message 2", "Message 3", "Message 4"]

    # Test 4: With keep=("fraction", 0.8), keep everything (no summarization needed)
    # target tokens = 0.8 * 1000 = 800, which equals total tokens, so keep all
    backend = MockBackend()
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("fraction", 0.80),
        keep=("fraction", 0.8),
        token_counter=token_counter,
    )

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should not trigger summarization since we'd keep everything anyway
    assert not isinstance(result, ExtendedModelResponse)
    assert len(backend.write_calls) == 0


def test_usage_metadata_trigger() -> None:
    """Test that usage_metadata from AI messages can trigger summarization.

    This tests advanced triggering based on `usage_metadata` from AI messages,
    particularly for models like Anthropic that report token usage in response metadata.
    """
    backend = MockBackend()
    mock_model = make_mock_model()
    # Mock the model to appear as Anthropic - need to match ls_provider
    mock_model._llm_type = "anthropic-chat"
    mock_model._get_ls_params.return_value = {"ls_provider": "anthropic"}

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("tokens", 10_000),
        keep=("messages", 4),
    )

    messages: list[BaseMessage] = [
        HumanMessage(content="msg1", id="h1"),
        AIMessage(
            content="msg2",
            id="a1",
            tool_calls=[{"name": "tool", "args": {}, "id": "call1"}],
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 5000,
                "output_tokens": 1000,
                "total_tokens": 6000,
            },
        ),
        ToolMessage(content="result", tool_call_id="call1", id="t1"),
        AIMessage(
            content="msg3",
            id="a2",
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 6100,
                "output_tokens": 900,
                "total_tokens": 7000,
            },
        ),
        HumanMessage(content="msg4", id="h2"),
        AIMessage(
            content="msg5",
            id="a3",
            response_metadata={"model_provider": "anthropic"},
            usage_metadata={
                "input_tokens": 7500,
                "output_tokens": 2501,
                "total_tokens": 10_001,
            },
        ),
    ]

    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        result, _ = call_wrap_model_call(middleware, state, runtime)

    # Should trigger summarization because usage_metadata shows we exceeded 10k tokens
    assert isinstance(result, ExtendedModelResponse)
    assert result.command is not None
    assert result.command.update is not None
    assert "_summarization_event" in result.command.update
    assert len(backend.write_calls) == 1


async def test_async_offload_and_summary_run_concurrently() -> None:
    """Verify that _aoffload_to_backend and _acreate_summary run in parallel."""
    delay = 0.1
    backend = MockBackend()
    mock_model = make_mock_model()

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )

    original_offload = middleware._aoffload_to_backend
    original_summary = middleware._acreate_summary

    async def slow_offload(
        be: Any,  # noqa: ANN401
        msgs: Any,  # noqa: ANN401
    ) -> str | None:
        await asyncio.sleep(delay)
        return await original_offload(be, msgs)

    async def slow_summary(
        msgs: Any,  # noqa: ANN401
    ) -> str:
        await asyncio.sleep(delay)
        return await original_summary(msgs)

    middleware._aoffload_to_backend = slow_offload  # type: ignore[assignment]
    middleware._acreate_summary = slow_summary  # type: ignore[assignment]

    messages = make_conversation_messages(num_old=6, num_recent=2)
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        start = time.monotonic()
        result, _ = await call_awrap_model_call(middleware, state, runtime)
        elapsed = time.monotonic() - start

    assert isinstance(result, ExtendedModelResponse)
    # If sequential, elapsed >= 2 * delay (0.2s). If parallel, elapsed ~ delay.
    # Use 2.5x multiplier to allow for CI scheduling jitter.
    assert elapsed < 2.5 * delay, f"Expected parallel execution (<{2.5 * delay}s) but took {elapsed:.2f}s"


class TestTokenCountingEfficiency:
    """The per-call token count is expensive (tool schema conversion), so it must run once."""

    def _make_counting_middleware(self) -> tuple[SummarizationMiddleware, dict[str, int]]:
        calls = {"count": 0}

        def counting_token_counter(_messages: list[BaseMessage], **_kwargs: Any) -> int:
            calls["count"] += 1
            return 10  # Far below every trigger: no truncation, no summarization.

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=MockBackend(),
            trigger=("tokens", 1_000_000),
            token_counter=counting_token_counter,
            truncate_args_settings={
                "trigger": ("tokens", 1_000_000),
                "keep": ("messages", 2),
            },
        )
        return middleware, calls

    def test_token_counter_called_once_per_model_call(self) -> None:
        middleware, calls = self._make_counting_middleware()
        state = cast("AgentState[Any]", {"messages": make_conversation_messages()})

        _, captured_request = call_wrap_model_call(middleware, state, make_mock_runtime())

        assert captured_request is not None  # Handler ran; nothing was summarized.
        assert calls["count"] == 1

    async def test_token_counter_called_once_per_model_call_async(self) -> None:
        middleware, calls = self._make_counting_middleware()
        state = cast("AgentState[Any]", {"messages": make_conversation_messages()})

        _, captured_request = await call_awrap_model_call(middleware, state, make_mock_runtime())

        assert captured_request is not None  # Handler ran; nothing was summarized.
        assert calls["count"] == 1

    def _make_truncating_counting_middleware(
        self,
    ) -> tuple[SummarizationMiddleware, dict[str, int]]:
        """Middleware whose truncation trigger fires but summarization never does.

        Truncating tool-call args shrinks the message set, so the count must be
        refreshed afterward before the summarization check; this middleware
        exercises that recount branch.
        """
        calls = {"count": 0}

        def counting_token_counter(_messages: list[BaseMessage], **_kwargs: Any) -> int:
            calls["count"] += 1
            return 10  # Far below the summarization trigger: nothing is summarized.

        middleware = SummarizationMiddleware(
            model=make_mock_model(),
            backend=MockBackend(),
            trigger=("messages", 100),  # High: no summarization.
            token_counter=counting_token_counter,
            truncate_args_settings={
                "trigger": ("messages", 5),  # Low: truncation fires.
                "keep": ("messages", 2),
                "max_length": 100,
            },
        )
        return middleware, calls

    def _truncatable_state(self) -> "AgentState[Any]":
        """A conversation whose old `write_file` arg is large enough to truncate."""
        messages = [
            AIMessage(
                content="",
                id="a1",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "write_file",
                        "args": {"file_path": "/test.txt", "content": "x" * 200},
                    }
                ],
            ),
            ToolMessage(content="File written", tool_call_id="tc1", id="t1"),
            HumanMessage(content="Request 1", id="h1"),
            AIMessage(content="Response 1", id="a2"),
            HumanMessage(content="Request 2", id="h2"),
            AIMessage(content="Response 2", id="a3"),
        ]
        return cast("AgentState[Any]", {"messages": messages})

    def test_token_counter_recounts_when_truncation_modifies_messages(self) -> None:
        middleware, calls = self._make_truncating_counting_middleware()

        _, captured_request = call_wrap_model_call(middleware, self._truncatable_state(), make_mock_runtime())

        assert captured_request is not None  # Handler ran; nothing was summarized.
        # Truncation changed the message set, so the count is refreshed:
        # once before truncation, once after.
        assert calls["count"] == 2
        # Confirm the modify path was genuinely taken (not a vacuous recount).
        first_ai = captured_request.messages[0]
        assert isinstance(first_ai, AIMessage)
        assert first_ai.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"

    async def test_token_counter_recounts_when_truncation_modifies_messages_async(self) -> None:
        middleware, calls = self._make_truncating_counting_middleware()

        _, captured_request = await call_awrap_model_call(middleware, self._truncatable_state(), make_mock_runtime())

        assert captured_request is not None  # Handler ran; nothing was summarized.
        # Truncation changed the message set, so the count is refreshed:
        # once before truncation, once after.
        assert calls["count"] == 2
        # Confirm the modify path was genuinely taken (not a vacuous recount).
        first_ai = captured_request.messages[0]
        assert isinstance(first_ai, AIMessage)
        assert first_ai.tool_calls[0]["args"]["content"] == "x" * 20 + "...(argument truncated)"


class _AnchorRejectingBackend(MockBackend):
    """Rejects the first `edit`, then behaves normally.

    Mirrors an archive rewritten by another process: the cached anchor no
    longer matches, so the append must fall back to a full rewrite.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rejected = 0

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        if self.rejected == 0:
            self.rejected += 1
            self.edit_calls.append((path, old_string, new_string))
            return EditResult(error="String not found in file")
        return super().edit(path, old_string, new_string, replace_all)


class _AnchorRaisingBackend(MockBackend):
    """Raises on the first `edit`, then behaves normally."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.raised = 0

    def edit(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
        if self.raised == 0:
            self.raised += 1
            msg = "Mock anchored edit exception"
            raise RuntimeError(msg)
        return super().edit(path, old_string, new_string, replace_all)


class TestHistoryArchiveAnchoredAppend:
    """The archive is appended to via an anchor rather than rewritten in full."""

    @staticmethod
    def _make_middleware(backend: MockBackend) -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model=make_mock_model(),
            backend=backend,
            trigger=("tokens", 1_000_000),
        )

    @staticmethod
    def _batch(idx: int) -> list:
        return [HumanMessage(f"question {idx}"), AIMessage(f"answer {idx}")]

    def test_first_offload_downloads_and_writes(self) -> None:
        """With no anchor cached, the original download-and-write path runs."""
        backend = MockBackend()
        middleware = self._make_middleware(backend)

        with mock_get_config():
            assert middleware._offload_to_backend(backend, self._batch(0)) is not None

        assert len(backend.download_files_calls) == 1
        assert len(backend.write_calls) == 1
        assert backend.edit_calls == []

    def test_second_offload_appends_without_downloading(self) -> None:
        """The anchored append skips the download entirely."""
        backend = MockBackend()
        middleware = self._make_middleware(backend)

        with mock_get_config():
            middleware._offload_to_backend(backend, self._batch(0))
            downloads_after_first = len(backend.download_files_calls)
            assert middleware._offload_to_backend(backend, self._batch(1)) is not None

        # No further download: the second offload used the cached anchor.
        assert len(backend.download_files_calls) == downloads_after_first
        assert len(backend.edit_calls) == 1
        _, old_string, new_string = backend.edit_calls[0]
        # The edit appends: the new content is the anchor plus exactly one section.
        assert new_string.startswith(old_string)
        assert "question 1" in new_string[len(old_string) :]
        assert "question 0" not in new_string[len(old_string) :]

    def test_rejected_anchor_falls_back_to_full_rewrite(self) -> None:
        """A stale anchor is non-fatal: the offload still lands via rewrite."""
        backend = _AnchorRejectingBackend()
        middleware = self._make_middleware(backend)

        with mock_get_config():
            middleware._offload_to_backend(backend, self._batch(0))
            downloads_after_first = len(backend.download_files_calls)
            assert middleware._offload_to_backend(backend, self._batch(1)) is not None

        assert backend.rejected == 1
        # The rejection forced the download the fast path had skipped.
        assert len(backend.download_files_calls) == downloads_after_first + 1

    def test_raising_anchor_falls_back_to_full_rewrite(self) -> None:
        """An exception from the anchored edit is caught and rewritten instead."""
        backend = _AnchorRaisingBackend()
        middleware = self._make_middleware(backend)

        with mock_get_config():
            middleware._offload_to_backend(backend, self._batch(0))
            downloads_after_first = len(backend.download_files_calls)
            assert middleware._offload_to_backend(backend, self._batch(1)) is not None

        assert backend.raised == 1
        assert len(backend.download_files_calls) == downloads_after_first + 1

    def test_failed_offload_does_not_cache_an_anchor(self) -> None:
        """A failed write must not leave an anchor that never reached the backend."""
        backend = MockBackend(should_fail=True, error_message="disk full")
        middleware = self._make_middleware(backend)

        with mock_get_config():
            assert middleware._offload_to_backend(backend, self._batch(0)) is None

        assert middleware._history_tails == {}

    def test_tail_cache_is_bounded(self) -> None:
        """Tracking many threads must not grow without limit."""
        middleware = self._make_middleware(MockBackend())

        for i in range(_MAX_TRACKED_HISTORY_TAILS * 3):
            middleware._record_history_tail(f"/history/thread-{i}.md", f"section {i}")

        assert len(middleware._history_tails) == _MAX_TRACKED_HISTORY_TAILS
        # The most recent paths are the ones retained.
        newest = f"/history/thread-{_MAX_TRACKED_HISTORY_TAILS * 3 - 1}.md"
        assert newest in middleware._history_tails

    async def test_second_offload_appends_without_downloading_async(self) -> None:
        """Async twin: same anchored append, same skipped download."""
        backend = MockBackend()
        middleware = self._make_middleware(backend)

        with mock_get_config():
            await middleware._aoffload_to_backend(backend, self._batch(0))
            downloads_after_first = len(backend.download_files_calls)
            assert await middleware._aoffload_to_backend(backend, self._batch(1)) is not None

        assert len(backend.download_files_calls) == downloads_after_first
        assert len(backend.edit_calls) == 1
        _, old_string, new_string = backend.edit_calls[0]
        assert new_string.startswith(old_string)

    def test_archive_matches_full_rewrite_on_real_backend(self, tmp_path: Path) -> None:
        """End to end: the stored archive is byte-identical to the rewrite path.

        Uses a real `FilesystemBackend` so the anchored `edit` is executed
        rather than recorded. Section timestamps are wall-clock, so they are
        normalized before comparison.
        """

        def archive_after(events: int, root: Path, *, anchored: bool) -> str:
            backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
            middleware = self._make_middleware(backend)
            with mock_get_config():
                for i in range(events):
                    if not anchored:
                        # Drop the anchor each time to force the rewrite path.
                        middleware._history_tails.clear()
                    assert middleware._offload_to_backend(backend, self._batch(i)) is not None
                path = middleware._get_history_path()
            content = backend.download_files([path])[0].content
            assert content is not None
            return re.sub(r"## Summarized at [^\n]+", "## Summarized at <TS>", content.decode())

        rewritten = archive_after(6, tmp_path / "rewrite", anchored=False)
        appended = archive_after(6, tmp_path / "append", anchored=True)

        assert appended == rewritten
        assert appended.count("## Summarized at") == 6

    def test_bytes_transferred_do_not_grow_with_archive_size(self, tmp_path: Path) -> None:
        """Guards the quadratic regression this replaced.

        Rewriting sends the whole archive on every event, so bytes transferred
        grow with the square of the event count. Appending sends one anchor plus
        one section regardless of how large the archive has become.
        """

        def bytes_moved(events: int, root: Path) -> int:
            backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)
            middleware = self._make_middleware(backend)
            moved = 0
            real_download, real_edit, real_write = backend.download_files, backend.edit, backend.write

            def download(paths: list[str]) -> list[FileDownloadResponse]:
                nonlocal moved
                responses = real_download(paths)
                moved += sum(len(r.content) for r in responses if r.content)
                return responses

            def edit(path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:  # noqa: FBT001, FBT002
                nonlocal moved
                moved += len(new_string)
                return real_edit(path, old_string, new_string, replace_all)

            def write(path: str, content: str) -> WriteResult:
                nonlocal moved
                moved += len(content)
                return real_write(path, content)

            backend.download_files, backend.edit, backend.write = download, edit, write  # ty: ignore[invalid-assignment]
            with mock_get_config():
                for i in range(events):
                    assert middleware._offload_to_backend(backend, self._batch(i)) is not None
            return moved

        few = bytes_moved(4, tmp_path / "few")
        many = bytes_moved(16, tmp_path / "many")

        # Four times the events must cost close to four times the bytes. Under
        # the rewrite path the ratio grows with the event count instead.
        assert many < few * 6


class _OpaqueCounter:
    """Wraps a counter so its signature cannot be introspected.

    Accessing `__signature__` raises, which is what `inspect.signature` consults
    first. This forces `_token_counter_accepts_tools` onto its `None` result and
    `_count_tokens` onto the runtime-probe fallback -- the only path that still
    swallows a `TypeError`. The wrapped callable is invoked verbatim, so it can
    either accept or reject a `tools` kwarg at call time.
    """

    def __init__(self, fn: Callable[..., int]) -> None:
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        return self._fn(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        msg = "no signature available"
        raise ValueError(msg)


class TestTokenCounterToolsProbe:
    """`_count_tokens` must not mask a `TypeError` raised inside the counter body."""

    def test_accepts_tools_with_explicit_param(self) -> None:
        # The parameter must be named `tools` to exercise the signature probe.
        def counter(_messages: list[BaseMessage], *, tools: Any = None) -> int:  # noqa: ANN401, ARG001
            return 0

        assert _token_counter_accepts_tools(counter) is True

    def test_accepts_tools_with_var_keyword(self) -> None:
        def counter(_messages: list[BaseMessage], **_kwargs: Any) -> int:
            return 0

        assert _token_counter_accepts_tools(counter) is True

    def test_rejects_tools_without_param(self) -> None:
        def counter(_messages: list[BaseMessage]) -> int:
            return 0

        assert _token_counter_accepts_tools(counter) is False

    def test_rejects_var_positional_only(self) -> None:
        # `*args` captures positional arguments only, so `tools=` cannot reach it
        # as a keyword. The counter is treated as not accepting tools, guarding
        # the deliberate omission of `VAR_POSITIONAL` from the kind check.
        def counter(_messages: list[BaseMessage], *_args: Any) -> int:
            return 0

        assert _token_counter_accepts_tools(counter) is False

    def test_returns_none_for_uninspectable_signature(self) -> None:
        # A callable whose signature cannot be read yields `None`, signaling that
        # `_count_tokens` must fall back to runtime probing.
        opaque = _OpaqueCounter(lambda *_args, **_kwargs: 0)
        assert _token_counter_accepts_tools(opaque) is None

    def _make_request_with_tools(self) -> ModelRequest:
        """A request carrying tools so `tools=` reaches the token counter."""
        state = cast("AgentState[Any]", {"messages": make_conversation_messages()})
        return ModelRequest(
            model=make_mock_model(),
            messages=state["messages"],
            system_message=None,
            tools=[{"type": "function", "function": {"name": "noop"}}],
            runtime=make_mock_runtime(),
            state=state,
        )

    def _make_middleware(self, counter: Any) -> SummarizationMiddleware:  # noqa: ANN401
        """Middleware whose triggers never fire, so only `_count_tokens` runs."""
        return SummarizationMiddleware(
            model=make_mock_model(),
            backend=MockBackend(),
            trigger=("tokens", 1_000_000),
            token_counter=counter,
            truncate_args_settings={"trigger": ("tokens", 1_000_000), "keep": ("messages", 2)},
        )

    def test_in_body_typeerror_propagates_sync(self) -> None:
        # Accepts `tools` but its body raises only when tools is supplied, mimicking
        # a counter whose tool-schema handling is broken. The old probe swallowed
        # this and silently recounted without tools; it must now surface loudly.
        def broken_counter(_messages: list[BaseMessage], *, tools: Any = None) -> int:  # noqa: ANN401
            if tools is not None:
                msg = "tool schema conversion failed"
                raise TypeError(msg)
            return 10

        middleware = self._make_middleware(broken_counter)

        def handler(_req: ModelRequest) -> ModelResponse:
            return AIMessage(content="ok")

        with pytest.raises(TypeError, match="tool schema conversion failed"):
            middleware.wrap_model_call(self._make_request_with_tools(), handler)

    async def test_in_body_typeerror_propagates_async(self) -> None:
        # `awrap_model_call` delegates to the synchronous `_count_tokens`, so the
        # counting logic is shared with the sync test above. This guards the
        # async wrapper itself from swallowing the propagated `TypeError`.
        def broken_counter(_messages: list[BaseMessage], *, tools: Any = None) -> int:  # noqa: ANN401
            if tools is not None:
                msg = "tool schema conversion failed"
                raise TypeError(msg)
            return 10

        middleware = self._make_middleware(broken_counter)

        async def handler(_req: ModelRequest) -> ModelResponse:
            return AIMessage(content="ok")

        with pytest.raises(TypeError, match="tool schema conversion failed"):
            await middleware.awrap_model_call(self._make_request_with_tools(), handler)

    def test_counter_without_tools_param_is_called_without_tools(self) -> None:
        # Tools are present on the request, but a counter that does not declare
        # `tools` is invoked without them -- and does not raise.
        def counter(_messages: list[BaseMessage]) -> int:
            return 10

        middleware = self._make_middleware(counter)
        captured: ModelRequest | None = None

        def handler(req: ModelRequest) -> ModelResponse:
            nonlocal captured
            captured = req
            return AIMessage(content="ok")

        result = middleware.wrap_model_call(self._make_request_with_tools(), handler)

        assert isinstance(result, AIMessage)
        assert captured is not None  # Handler ran; counting without tools did not raise.

    def test_uninspectable_counter_probes_with_tools(self) -> None:
        # Signature can't be read, so `_counter_accepts_tools` is `None` and
        # `_count_tokens` probes with `tools=`. The probe succeeds, so the
        # tools-inclusive count is returned -- tools are not silently dropped.
        counter = _OpaqueCounter(lambda _messages, *, tools=None: 99 if tools is not None else 7)
        middleware = self._make_middleware(counter)
        assert middleware._counter_accepts_tools is None

        messages = make_conversation_messages()
        tools = [{"type": "function", "function": {"name": "noop"}}]
        assert middleware._count_tokens(messages, None, tools) == 99

    def test_uninspectable_counter_falls_back_without_tools(self) -> None:
        # The opaque counter rejects `tools=` at call time. This is the sole path
        # that still catches a `TypeError` from the probe, recounting without
        # tools rather than surfacing it.
        counter = _OpaqueCounter(lambda _messages: 7)
        middleware = self._make_middleware(counter)
        assert middleware._counter_accepts_tools is None

        messages = make_conversation_messages()
        tools = [{"type": "function", "function": {"name": "noop"}}]
        assert middleware._count_tokens(messages, None, tools) == 7


@pytest.mark.anyio
async def test_async_offloads_base64_images() -> None:
    """Async path offloads base64 images via aupload_files before gather (issue #2873).

    `awrap_model_call` must upload image data before
    `asyncio.gather(offload, summary)` so both coroutines see path references,
    not raw base64. This mirrors `test_offload_rewrites_base64_images` but
    exercises the async code path end-to-end.
    """
    raw_png = _base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    b64 = _base64.b64encode(raw_png).decode()
    expected_key = hashlib.sha256(raw_png).hexdigest()[:16]
    expected_path = f"/conversation_history/media/{expected_key}.png"

    backend = MockBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))

    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )

    messages: list[BaseMessage] = [
        # Shape 1: explicit base64 field
        HumanMessage(
            content=[{"type": "image", "base64": b64, "mime_type": "image/png"}],
            id="b64-field",
        ),
        # Shape 2: top-level data: URL
        HumanMessage(
            content=[{"type": "image", "url": f"data:image/png;base64,{b64}"}],
            id="b64-url",
        ),
        # Shape 3: OpenAI-style image_url
        HumanMessage(
            content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}],
            id="b64-openai",
        ),
        *make_conversation_messages(num_old=5, num_recent=2),
    ]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config():
        await call_awrap_model_call(middleware, state, runtime)

    # aupload_files delegates to upload_files in MockBackend, so write_calls captures it.
    # Identical images are deduped — only one upload despite three messages.
    image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
    assert len(image_uploads) == 1
    assert image_uploads[0][0] == expected_path

    # Archive markdown references the offloaded path three times.
    archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
    assert archive_write[1].count(expected_path) == 3
    assert b64 not in archive_write[1]

    # Raw base64 never reaches the async summary model prompt (symmetric with the
    # sync `test_upload_runs_once_before_offload_and_summary` guard).
    assert b64 not in str(mock_model.ainvoke.call_args)


@pytest.mark.anyio
async def test_async_upload_response_error_writes_placeholder() -> None:
    """Async upload responses with errors are treated as failed image offloads."""
    raw_a = b"\x89PNG_ASYNC_RESPONSE_OK"
    raw_b = b"\x89PNG_ASYNC_RESPONSE_DENIED"
    b64_a = _base64.b64encode(raw_a).decode()
    b64_b = _base64.b64encode(raw_b).decode()
    key_a = hashlib.sha256(raw_a).hexdigest()[:16]
    key_b = hashlib.sha256(raw_b).hexdigest()[:16]
    expected_path_a = f"/conversation_history/media/{key_a}.png"
    expected_path_b = f"/conversation_history/media/{key_b}.png"

    class ResponseErrorBackend(MockBackend):
        """Backend that reports one async image upload failure in the response."""

        async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
            responses = []
            for path, _content in files:
                if key_b in path:
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                else:
                    self.write_calls.append((path, "<binary>"))
                    responses.append(FileUploadResponse(path=path, error=None))
            return responses

    backend = ResponseErrorBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )

    messages: list[BaseMessage] = [
        HumanMessage(content=[{"type": "image", "base64": b64_a, "mime_type": "image/png"}], id="img-a"),
        HumanMessage(content=[{"type": "image", "base64": b64_b, "mime_type": "image/png"}], id="img-b"),
        *make_conversation_messages(num_old=5, num_recent=2),
    ]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
        await call_awrap_model_call(middleware, state, runtime)

    image_uploads = [(p, c) for p, c in backend.write_calls if p.startswith("/conversation_history/media/")]
    assert image_uploads == [(expected_path_a, "<binary>")]

    archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
    assert expected_path_a in archive_write[1]
    assert expected_path_b not in archive_write[1]
    assert b64_b not in archive_write[1]
    assert 'error="failed_to_offload"' in archive_write[1]


@pytest.mark.anyio
async def test_async_all_uploads_raise_writes_placeholders() -> None:
    """Async all-raised upload failures still rewrite raw base64 to placeholders."""
    raw_png = b"\x89PNG_ASYNC_ALL_RAISE"
    b64 = _base64.b64encode(raw_png).decode()

    class RaisingUploadBackend(MockBackend):
        """Backend that raises for every async image upload."""

        async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
            msg = f"Simulated upload failure for {files[0][0]}"
            raise RuntimeError(msg)

    backend = RaisingUploadBackend()
    mock_model = make_mock_model()
    mock_model.ainvoke = MagicMock(return_value=MagicMock(text="Async summary"))
    middleware = SummarizationMiddleware(
        model=mock_model,
        backend=backend,
        trigger=("messages", 5),
        keep=("messages", 2),
    )

    messages: list[BaseMessage] = [
        HumanMessage(content=[{"type": "image", "base64": b64, "mime_type": "image/png"}], id="img"),
        *make_conversation_messages(num_old=5, num_recent=2),
    ]
    state = cast("AgentState[Any]", {"messages": messages})
    runtime = make_mock_runtime()

    with mock_get_config(), pytest.warns(UserWarning, match="could not be offloaded"):
        await call_awrap_model_call(middleware, state, runtime)

    archive_write = next((p, c) for p, c in backend.write_calls if p.endswith(".md"))
    assert b64 not in archive_write[1]
    assert 'error="failed_to_offload"' in archive_write[1]


def test_upload_response_error_classifies_batch_result() -> None:
    """`_upload_response_error` distinguishes empty, error, and success responses.

    Covers the defensive empty-list branch (a backend that returns no response
    for a requested upload), which the integration tests don't exercise.
    """
    # Empty list: backend returned no response for the single requested file.
    assert _upload_response_error([]) == "missing_upload_response"
    # Populated error is surfaced as a string.
    assert _upload_response_error([FileUploadResponse(path="/m.png", error="permission_denied")]) == "permission_denied"
    # Success (error is None) returns None.
    assert _upload_response_error([FileUploadResponse(path="/m.png", error=None)]) is None
