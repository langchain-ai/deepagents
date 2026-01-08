"""Async unit tests for memory middleware with FilesystemBackend.

This module contains async versions of memory middleware tests.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.memory import MemoryMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def make_memory_content(title: str, content: str) -> str:
    """Create AGENTS.md content.

    Args:
        title: Title for the memory file
        content: Content body

    Returns:
        Complete AGENTS.md content as string
    """
    return f"""# {title}

{content}
"""


async def test_load_memory_from_backend_single_source_async(tmp_path: Path) -> None:
    """Test loading memory from a single source using filesystem backend (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create memory file using backend's upload_files interface
    memory_dir = tmp_path / "user"
    memory_path = str(memory_dir / "AGENTS.md")
    memory_content = make_memory_content(
        "User Preferences",
        """- Always use type hints
- Prefer functional patterns
- Be concise""",
    )

    responses = backend.upload_files([(memory_path, memory_content.encode("utf-8"))])
    assert responses[0].error is None

    # Create middleware
    sources: list[str] = [memory_path]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Test abefore_agent loads the memory
    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    assert "memory_contents" in result
    assert memory_path in result["memory_contents"]
    assert "type hints" in result["memory_contents"][memory_path]
    assert "functional patterns" in result["memory_contents"][memory_path]


async def test_load_memory_from_backend_multiple_sources_async(tmp_path: Path) -> None:
    """Test loading memory from multiple sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create multiple memory files
    user_path = str(tmp_path / "user" / "AGENTS.md")
    project_path = str(tmp_path / "project" / "AGENTS.md")

    user_content = make_memory_content("User Preferences", "- Use Python 3.11+\n- Follow PEP 8")
    project_content = make_memory_content("Project Guidelines", "## Architecture\nThis is a FastAPI project.")

    responses = backend.upload_files(
        [
            (user_path, user_content.encode("utf-8")),
            (project_path, project_content.encode("utf-8")),
        ]
    )
    assert all(r.error is None for r in responses)

    # Create middleware with multiple sources
    sources: list[str] = [
        user_path,
        project_path,
    ]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Test abefore_agent loads all memory
    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    assert "memory_contents" in result
    assert user_path in result["memory_contents"]
    assert project_path in result["memory_contents"]
    assert "Python 3.11" in result["memory_contents"][user_path]
    assert "FastAPI" in result["memory_contents"][project_path]


async def test_load_memory_handles_missing_file_async(tmp_path: Path) -> None:
    """Test that missing files raise an error (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create only one of two memory files
    user_path = str(tmp_path / "user" / "AGENTS.md")
    missing_path = str(tmp_path / "nonexistent" / "AGENTS.md")

    user_content = make_memory_content("User Preferences", "- Be helpful")
    backend.upload_files([(user_path, user_content.encode("utf-8"))])

    # Create middleware with existing and missing sources
    sources: list[str] = [
        missing_path,
        user_path,
    ]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Test abefore_agent loads only existing memory
    result = await middleware.abefore_agent({}, None, {})  # type: ignore
    assert result is not None
    assert missing_path not in result["memory_contents"]
    assert user_path in result["memory_contents"]


async def test_before_agent_skips_if_already_loaded_async(tmp_path: Path) -> None:
    """Test that abefore_agent doesn't reload if already in state."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    user_path = str(tmp_path / "user" / "AGENTS.md")
    user_content = make_memory_content("User Preferences", "- Some content")
    backend.upload_files([(user_path, user_content.encode("utf-8"))])

    sources: list[str] = [user_path]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Pre-populate state
    state = {"memory_contents": {user_path: "Already loaded content"}}
    result = await middleware.abefore_agent(state, None, {})  # type: ignore

    # Should return None (no update needed)
    assert result is None


async def test_load_memory_with_empty_sources_async(tmp_path: Path) -> None:
    """Test middleware with empty sources list (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    middleware = MemoryMiddleware(backend=backend, sources=[])

    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    assert result["memory_contents"] == {}


async def test_memory_content_with_special_characters_async(tmp_path: Path) -> None:
    """Test that special characters in memory are handled (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "test" / "AGENTS.md")
    memory_content = make_memory_content(
        "Special Characters",
        """- Use `backticks` for code
- <xml> tags should work
- "Quotes" and 'apostrophes'
- {braces} and [brackets]""",
    )

    backend.upload_files([(memory_path, memory_content.encode("utf-8"))])

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[memory_path],
    )

    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    assert "`backticks`" in content
    assert "<xml>" in content
    assert '"Quotes"' in content
    assert "{braces}" in content


async def test_memory_content_with_unicode_async(tmp_path: Path) -> None:
    """Test that unicode characters in memory are handled (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "test" / "AGENTS.md")
    memory_content = make_memory_content(
        "Unicode Content",
        """- 日本語 (Japanese)
- 中文 (Chinese)
- Emoji: 🚀 🎉 ✨
- Math: ∀x∈ℝ, x² ≥ 0""",
    )

    backend.upload_files([(memory_path, memory_content.encode("utf-8"))])

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[memory_path],
    )

    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    assert "日本語" in content
    assert "中文" in content
    assert "🚀" in content
    assert "∀x∈ℝ" in content


async def test_memory_content_with_large_file_async(tmp_path: Path) -> None:
    """Test that large memory files are loaded correctly (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "test" / "AGENTS.md")
    # Create a large memory file (around 10KB)
    large_content = make_memory_content("Large Memory", "Line of content\n" * 500)

    backend.upload_files([(memory_path, large_content.encode("utf-8"))])

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[memory_path],
    )

    result = await middleware.abefore_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    # Verify content was loaded (check for repeated pattern)
    assert content.count("Line of content") == 500


async def test_agent_with_memory_middleware_multiple_sources_async(tmp_path: Path) -> None:
    """Test agent with memory from multiple sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create multiple memory files
    user_path = str(tmp_path / "user" / "AGENTS.md")
    project_path = str(tmp_path / "project" / "AGENTS.md")

    user_content = make_memory_content("User Style", "- Use Python 3.11+")
    project_content = make_memory_content("Project Info", "- FastAPI backend")

    responses = backend.upload_files(
        [
            (user_path, user_content.encode("utf-8")),
            (project_path, project_content.encode("utf-8")),
        ]
    )
    assert all(r.error is None for r in responses)

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="I see both user and project preferences.")]))

    # Create middleware with multiple sources
    sources: list[str] = [
        user_path,
        project_path,
    ]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Help me")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify both memory sources are in system prompt with new format
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "<agent_memory>" in content
    assert user_path in content
    assert project_path in content
    assert "Python 3.11" in content
    assert "FastAPI" in content


async def test_agent_with_memory_middleware_empty_sources_async(tmp_path: Path) -> None:
    """Test that agent works with empty memory sources (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Working without memory.")]))

    # Create middleware with empty sources
    middleware = MemoryMiddleware(backend=backend, sources=[])

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify system prompt still contains Agent Memory section with empty agent_memory
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "<agent_memory>" in content
    assert "No memory loaded" in content


async def test_memory_middleware_order_matters_async(tmp_path: Path) -> None:
    """Test that memory sources are combined in order (async)."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create memory files
    first_path = str(tmp_path / "first" / "AGENTS.md")
    second_path = str(tmp_path / "second" / "AGENTS.md")

    first_content = make_memory_content("First", "First memory content")
    second_content = make_memory_content("Second", "Second memory content")

    backend.upload_files(
        [
            (first_path, first_content.encode("utf-8")),
            (second_path, second_content.encode("utf-8")),
        ]
    )

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Understood.")]))

    # Create middleware with specific order
    sources: list[str] = [
        first_path,
        second_path,
    ]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Test")]})

    # Verify order in system prompt with new format
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "<agent_memory>" in content
    assert first_path in content
    assert second_path in content

    # First should appear before second (both path and content)
    first_pos = content.find("First memory content")
    second_pos = content.find("Second memory content")
    assert first_pos > 0
    assert second_pos > 0
    assert first_pos < second_pos
