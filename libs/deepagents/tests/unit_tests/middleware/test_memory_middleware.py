"""Tests for the MemoryMiddleware."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.memory import (
    MemoryMiddleware,
    MemorySource,
    MemoryState,
)


# --- Fixtures ---


@pytest.fixture
def tmp_memory_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with memory files."""
    # User memory
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    (user_dir / "AGENTS.md").write_text("""# User Preferences

- Always use type hints
- Prefer functional patterns
- Be concise
""")

    # Project memory
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "AGENTS.md").write_text("""# Project Guidelines

## Architecture
This is a FastAPI project with SQLAlchemy.

## Testing
Run tests with: pytest tests/

## Code Style
- 4-space indentation
- Google-style docstrings
""")

    return tmp_path


# --- Unit Tests ---


def test_memory_middleware_init() -> None:
    """Test MemoryMiddleware initialization."""
    backend = MagicMock()
    sources: list[MemorySource] = [
        {"path": "/user/AGENTS.md", "name": "user"},
        {"path": "/project/AGENTS.md", "name": "project"},
    ]

    middleware = MemoryMiddleware(backend=backend, sources=sources)

    assert middleware.sources == sources
    assert middleware._backend == backend


def test_memory_middleware_factory_backend() -> None:
    """Test MemoryMiddleware with factory backend."""
    mock_backend = MagicMock()
    factory = MagicMock(return_value=mock_backend)

    middleware = MemoryMiddleware(
        backend=factory,
        sources=[{"path": "/test/AGENTS.md", "name": "test"}],
    )

    # Factory should be stored
    assert middleware._backend == factory

    # Calling _get_backend should invoke factory
    mock_runtime = MagicMock()
    result = middleware._get_backend(mock_runtime)

    factory.assert_called_once_with(mock_runtime)
    assert result == mock_backend


def test_format_memory_locations_empty() -> None:
    """Test formatting with no sources."""
    middleware = MemoryMiddleware(backend=MagicMock(), sources=[])
    result = middleware._format_memory_locations()
    assert "None configured" in result


def test_format_memory_locations_single() -> None:
    """Test formatting with single source."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[{"path": "/home/user/AGENTS.md", "name": "user"}],
    )
    result = middleware._format_memory_locations()

    assert "**User**" in result
    assert "/home/user/AGENTS.md" in result


def test_format_memory_locations_multiple() -> None:
    """Test formatting with multiple sources."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[
            {"path": "/user/AGENTS.md", "name": "user"},
            {"path": "/project/AGENTS.md", "name": "project"},
        ],
    )
    result = middleware._format_memory_locations()

    assert "**User**" in result
    assert "**Project**" in result
    assert "/user/AGENTS.md" in result
    assert "/project/AGENTS.md" in result


def test_format_memory_contents_empty() -> None:
    """Test formatting with no contents."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[{"path": "/test/AGENTS.md", "name": "test"}],
    )
    result = middleware._format_memory_contents({})
    assert "No memory loaded" in result


def test_format_memory_contents_single() -> None:
    """Test formatting with single source content."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[{"path": "/user/AGENTS.md", "name": "user"}],
    )
    contents = {"user": "# User Memory\nBe helpful."}
    result = middleware._format_memory_contents(contents)

    assert "<user_memory>" in result
    assert "# User Memory" in result
    assert "Be helpful." in result
    assert "</user_memory>" in result


def test_format_memory_contents_multiple() -> None:
    """Test formatting with multiple source contents."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[
            {"path": "/user/AGENTS.md", "name": "user"},
            {"path": "/project/AGENTS.md", "name": "project"},
        ],
    )
    contents = {
        "user": "User preferences here",
        "project": "Project guidelines here",
    }
    result = middleware._format_memory_contents(contents)

    assert "<user_memory>" in result
    assert "User preferences here" in result
    assert "</user_memory>" in result
    assert "<project_memory>" in result
    assert "Project guidelines here" in result
    assert "</project_memory>" in result


def test_format_memory_contents_preserves_order() -> None:
    """Test that content order matches sources order."""
    middleware = MemoryMiddleware(
        backend=MagicMock(),
        sources=[
            {"path": "/first/AGENTS.md", "name": "first"},
            {"path": "/second/AGENTS.md", "name": "second"},
        ],
    )
    contents = {"second": "Second content", "first": "First content"}
    result = middleware._format_memory_contents(contents)

    # First should appear before second
    first_pos = result.find("<first_memory>")
    second_pos = result.find("<second_memory>")
    assert first_pos < second_pos


# --- Integration Tests with FilesystemBackend ---


def test_before_agent_loads_memory(tmp_memory_dir: Path) -> None:
    """Test that before_agent loads memory from sources."""
    backend = FilesystemBackend(root_dir=str(tmp_memory_dir), virtual_mode=False)

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[
            {"path": str(tmp_memory_dir / "user" / "AGENTS.md"), "name": "user"},
            {"path": str(tmp_memory_dir / "project" / "AGENTS.md"), "name": "project"},
        ],
    )

    state: MemoryState = {}
    result = middleware.before_agent(state, None)  # type: ignore

    assert result is not None
    assert "memory_contents" in result
    assert "user" in result["memory_contents"]
    assert "project" in result["memory_contents"]
    assert "type hints" in result["memory_contents"]["user"]
    assert "FastAPI" in result["memory_contents"]["project"]


def test_before_agent_skips_if_already_loaded(tmp_memory_dir: Path) -> None:
    """Test that before_agent doesn't reload if already in state."""
    backend = FilesystemBackend(root_dir=str(tmp_memory_dir), virtual_mode=False)

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[{"path": str(tmp_memory_dir / "user" / "AGENTS.md"), "name": "user"}],
    )

    # Pre-populate state
    state: MemoryState = {"memory_contents": {"user": "Already loaded"}}
    result = middleware.before_agent(state, None)  # type: ignore

    # Should return None (no update needed)
    assert result is None


def test_before_agent_handles_missing_file(tmp_memory_dir: Path) -> None:
    """Test that missing files are handled gracefully."""
    backend = FilesystemBackend(root_dir=str(tmp_memory_dir), virtual_mode=False)

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[
            {"path": str(tmp_memory_dir / "nonexistent" / "AGENTS.md"), "name": "missing"},
            {"path": str(tmp_memory_dir / "user" / "AGENTS.md"), "name": "user"},
        ],
    )

    state: MemoryState = {}
    result = middleware.before_agent(state, None)  # type: ignore

    assert result is not None
    # Missing file should not be in contents
    assert "missing" not in result["memory_contents"]
    # Existing file should be loaded
    assert "user" in result["memory_contents"]


def test_modify_request_injects_memory(tmp_memory_dir: Path) -> None:
    """Test that modify_request injects memory into system prompt."""
    backend = FilesystemBackend(root_dir=str(tmp_memory_dir), virtual_mode=False)

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[{"path": str(tmp_memory_dir / "user" / "AGENTS.md"), "name": "user"}],
    )

    # Create mock request with memory in state
    mock_request = MagicMock()
    mock_request.state = {"memory_contents": {"user": "Be helpful and concise."}}
    mock_request.system_prompt = "You are an assistant."
    mock_request.override = lambda **kwargs: type(
        "MockRequest",
        (),
        {**vars(mock_request), **kwargs, "state": mock_request.state},
    )()

    result = middleware.modify_request(mock_request)

    # Memory should be injected
    assert "<user_memory>" in result.system_prompt
    assert "Be helpful and concise." in result.system_prompt
    # Original prompt should still be there
    assert "You are an assistant." in result.system_prompt


def test_wrap_model_call(tmp_memory_dir: Path) -> None:
    """Test that wrap_model_call modifies request and calls handler."""
    backend = FilesystemBackend(root_dir=str(tmp_memory_dir), virtual_mode=False)

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[{"path": str(tmp_memory_dir / "user" / "AGENTS.md"), "name": "user"}],
    )

    # Create mock request
    mock_request = MagicMock()
    mock_request.state = {"memory_contents": {"user": "Test memory"}}
    mock_request.system_prompt = "Base prompt"

    captured_request = None

    def mock_handler(req):
        nonlocal captured_request
        captured_request = req
        return MagicMock()

    mock_request.override = lambda **kwargs: type(
        "MockRequest",
        (),
        {**vars(mock_request), **kwargs, "state": mock_request.state},
    )()

    middleware.wrap_model_call(mock_request, mock_handler)

    assert captured_request is not None
    assert "<user_memory>" in captured_request.system_prompt


def test_memory_state_schema() -> None:
    """Test that MemoryState has correct schema."""
    from typing import get_type_hints

    hints = get_type_hints(MemoryState, include_extras=True)
    assert "memory_contents" in hints


def test_empty_sources() -> None:
    """Test middleware with empty sources list."""
    middleware = MemoryMiddleware(backend=MagicMock(), sources=[])

    state: MemoryState = {}
    result = middleware.before_agent(state, None)  # type: ignore

    assert result is not None
    assert result["memory_contents"] == {}


def test_memory_content_with_special_characters(tmp_path: Path) -> None:
    """Test that special characters in memory are handled."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "AGENTS.md").write_text("""# Special Characters

- Use `backticks` for code
- <xml> tags should work
- "Quotes" and 'apostrophes'
- {braces} and [brackets]
""")

    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    middleware = MemoryMiddleware(
        backend=backend,
        sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
    )

    state: MemoryState = {}
    result = middleware.before_agent(state, None)  # type: ignore

    assert result is not None
    content = result["memory_contents"]["test"]
    assert "`backticks`" in content
    assert "<xml>" in content
    assert '"Quotes"' in content
