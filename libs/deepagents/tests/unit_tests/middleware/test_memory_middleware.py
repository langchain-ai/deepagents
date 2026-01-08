"""Unit tests for memory middleware with FilesystemBackend.

This module tests the memory middleware using end-to-end tests with fake chat models
and temporary directories with the FilesystemBackend in normal (non-virtual) mode.
"""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.graph import create_deep_agent
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


def create_store_memory_item(content: str) -> dict:
    """Create a memory item in StoreBackend FileData format.

    Args:
        content: Memory content string

    Returns:
        Dict with content (as list of lines), created_at, and modified_at
    """
    timestamp = datetime.now(UTC).isoformat()
    return {
        "content": content.split("\n"),
        "created_at": timestamp,
        "modified_at": timestamp,
    }


def test_format_memory_locations_empty() -> None:
    """Test formatting with no sources."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=[],
    )
    result = middleware._format_memory_locations()
    assert "None configured" in result


def test_format_memory_locations_single() -> None:
    """Test formatting with single source."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=["/home/user/.deepagents/AGENTS.md"],
    )
    result = middleware._format_memory_locations()

    assert "**Memory Sources:**" in result
    assert "`/home/user/.deepagents/AGENTS.md`" in result


def test_format_memory_locations_multiple() -> None:
    """Test formatting with multiple sources."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=[
            "~/.deepagents/AGENTS.md",
            "./.deepagents/AGENTS.md",
        ],
    )
    result = middleware._format_memory_locations()

    assert "**Memory Sources:**" in result
    assert "`~/.deepagents/AGENTS.md`" in result
    assert "`./.deepagents/AGENTS.md`" in result


def test_format_memory_contents_empty() -> None:
    """Test formatting with no contents."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=["/test/AGENTS.md"],
    )
    result = middleware._format_memory_contents({})
    assert "No memory loaded" in result


def test_format_memory_contents_single() -> None:
    """Test formatting with single source content."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=["/user/AGENTS.md"],
    )
    contents = {"/user/AGENTS.md": "# User Memory\nBe helpful."}
    result = middleware._format_memory_contents(contents)

    assert "# User Memory" in result
    assert "Be helpful." in result


def test_format_memory_contents_multiple() -> None:
    """Test formatting with multiple source contents."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=[
            "/user/AGENTS.md",
            "/project/AGENTS.md",
        ],
    )
    contents = {
        "/user/AGENTS.md": "User preferences here",
        "/project/AGENTS.md": "Project guidelines here",
    }
    result = middleware._format_memory_contents(contents)

    assert "User preferences here" in result
    assert "Project guidelines here" in result


def test_format_memory_contents_preserves_order() -> None:
    """Test that content order matches sources order."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=[
            "/first/AGENTS.md",
            "/second/AGENTS.md",
        ],
    )
    contents = {"/second/AGENTS.md": "Second content", "/first/AGENTS.md": "First content"}
    result = middleware._format_memory_contents(contents)

    # First should appear before second (based on sources order)
    first_pos = result.find("First content")
    second_pos = result.find("Second content")
    assert first_pos >= 0  # Found in result
    assert second_pos > 0  # Found after start
    assert first_pos < second_pos  # First appears before second


def test_format_memory_contents_skips_missing_sources() -> None:
    """Test that sources without content are skipped."""
    middleware = MemoryMiddleware(
        backend=None,  # type: ignore
        sources=[
            "/user/AGENTS.md",
            "/project/AGENTS.md",
        ],
    )
    # Only provide content for user, not project
    contents = {"/user/AGENTS.md": "User content only"}
    result = middleware._format_memory_contents(contents)

    assert "User content only" in result


def test_load_memory_from_backend_single_source(tmp_path: Path) -> None:
    """Test loading memory from a single source using filesystem backend."""
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

    # Test before_agent loads the memory
    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    assert "memory_contents" in result
    assert memory_path in result["memory_contents"]
    assert "type hints" in result["memory_contents"][memory_path]
    assert "functional patterns" in result["memory_contents"][memory_path]


def test_load_memory_from_backend_multiple_sources(tmp_path: Path) -> None:
    """Test loading memory from multiple sources."""
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

    # Test before_agent loads all memory
    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    assert "memory_contents" in result
    assert user_path in result["memory_contents"]
    assert project_path in result["memory_contents"]
    assert "Python 3.11" in result["memory_contents"][user_path]
    assert "FastAPI" in result["memory_contents"][project_path]


def test_load_memory_handles_missing_file(tmp_path: Path) -> None:
    """Test that missing files raise an error."""
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

    # Test before_agent raises error for missing file
    with pytest.raises(ValueError, match="Failed to download.*file_not_found"):
        middleware.before_agent({}, None, {})  # type: ignore


def test_before_agent_skips_if_already_loaded(tmp_path: Path) -> None:
    """Test that before_agent doesn't reload if already in state."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    user_path = str(tmp_path / "user" / "AGENTS.md")
    user_content = make_memory_content("User Preferences", "- Some content")
    backend.upload_files([(user_path, user_content.encode("utf-8"))])

    sources: list[str] = [user_path]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Pre-populate state
    state = {"memory_contents": {user_path: "Already loaded content"}}
    result = middleware.before_agent(state, None, {})  # type: ignore

    # Should return None (no update needed)
    assert result is None


def test_load_memory_with_empty_sources(tmp_path: Path) -> None:
    """Test middleware with empty sources list."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    middleware = MemoryMiddleware(backend=backend, sources=[])

    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    assert result["memory_contents"] == {}


def test_memory_content_with_special_characters(tmp_path: Path) -> None:
    """Test that special characters in memory are handled."""
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

    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    assert "`backticks`" in content
    assert "<xml>" in content
    assert '"Quotes"' in content
    assert "{braces}" in content


def test_memory_content_with_unicode(tmp_path: Path) -> None:
    """Test that unicode characters in memory are handled."""
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

    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    assert "日本語" in content
    assert "中文" in content
    assert "🚀" in content
    assert "∀x∈ℝ" in content


def test_memory_content_with_large_file(tmp_path: Path) -> None:
    """Test that large memory files are loaded correctly."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "test" / "AGENTS.md")
    # Create a large memory file (around 10KB)
    large_content = make_memory_content("Large Memory", "Line of content\n" * 500)

    backend.upload_files([(memory_path, large_content.encode("utf-8"))])

    middleware = MemoryMiddleware(
        backend=backend,
        sources=[memory_path],
    )

    result = middleware.before_agent({}, None, {})  # type: ignore

    assert result is not None
    content = result["memory_contents"][memory_path]
    # Verify content was loaded (check for repeated pattern)
    assert content.count("Line of content") == 500


def test_agent_with_memory_middleware_system_prompt(tmp_path: Path) -> None:
    """Test that memory middleware injects memory into the system prompt."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "user" / "AGENTS.md")
    memory_content = make_memory_content(
        "User Preferences",
        """- Always use type hints
- Prefer functional programming
- Be concise""",
    )

    responses = backend.upload_files([(memory_path, memory_content.encode("utf-8"))])
    assert responses[0].error is None

    # Create a fake chat model that we can inspect
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="I understand your preferences.")]))

    # Create middleware
    sources: list[str] = [memory_path]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Create agent with middleware
    agent = create_agent(
        model=fake_model,
        middleware=[middleware],
    )

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    # Verify the agent was invoked
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Inspect the call history to verify system prompt was injected
    assert len(fake_model.call_history) > 0, "Model should have been called at least once"

    # Get the first call
    first_call = fake_model.call_history[0]
    messages = first_call["messages"]

    system_message = messages[0]
    assert system_message.type == "system", "First message should be system prompt"
    content = system_message.text
    assert "Agent Memory" in content, "System prompt should contain 'Agent Memory' section"
    assert "type hints" in content, "System prompt should mention memory content"
    assert "functional programming" in content


def test_agent_with_memory_middleware_multiple_sources(tmp_path: Path) -> None:
    """Test agent with memory from multiple sources."""
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

    # Invoke
    result = agent.invoke({"messages": [HumanMessage(content="Help me")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify both memory sources are in system prompt
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "Python 3.11" in content
    assert "FastAPI" in content


def test_agent_with_memory_middleware_empty_sources(tmp_path: Path) -> None:
    """Test that agent works with empty memory sources."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Working without memory.")]))

    # Create middleware with empty sources
    middleware = MemoryMiddleware(backend=backend, sources=[])

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify system prompt still contains Agent Memory section
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "Agent Memory" in content
    assert "No memory loaded" in content or "None configured" in content


async def test_agent_with_memory_middleware_async(tmp_path: Path) -> None:
    """Test that memory middleware works with async agent invocation."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    memory_path = str(tmp_path / "user" / "AGENTS.md")
    memory_content = make_memory_content("Async Test", "- Test async loading")

    responses = backend.upload_files([(memory_path, memory_content.encode("utf-8"))])
    assert responses[0].error is None

    # Create fake model
    fake_model = GenericFakeChatModel(messages=iter([AIMessage(content="Async invocation successful.")]))

    # Create middleware
    sources: list[str] = [memory_path]
    middleware = MemoryMiddleware(backend=backend, sources=sources)

    # Create agent
    agent = create_agent(model=fake_model, middleware=[middleware])

    # Invoke asynchronously
    result = await agent.ainvoke({"messages": [HumanMessage(content="Hello")]})

    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify memory_contents is NOT in final state (it's private)
    assert "memory_contents" not in result

    # Verify memory was injected in system prompt
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    assert "Agent Memory" in content
    assert "Test async loading" in content


def test_memory_middleware_with_state_backend_factory() -> None:
    """Test that MemoryMiddleware can be initialized with StateBackend factory."""
    sources: list[str] = ["/memory/AGENTS.md"]
    middleware = MemoryMiddleware(
        backend=lambda rt: StateBackend(rt),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert callable(middleware._backend)
    assert len(middleware.sources) == 1
    assert middleware.sources[0] == "/memory/AGENTS.md"

    # Create a mock Runtime (simplified for testing)
    state = {"messages": [], "files": {}}
    runtime = SimpleNamespace(
        context=None,
        store=None,
        stream_writer=lambda _: None,
    )

    backend = middleware._get_backend(state, runtime, {})  # type: ignore
    assert isinstance(backend, StateBackend)
    assert backend.runtime is not None


def test_memory_middleware_with_store_backend_factory() -> None:
    """Test that MemoryMiddleware can be initialized with StoreBackend factory."""
    sources: list[str] = ["/memory/AGENTS.md"]
    middleware = MemoryMiddleware(
        backend=lambda rt: StoreBackend(rt),
        sources=sources,
    )

    # Verify the middleware was created successfully
    assert middleware is not None
    assert callable(middleware._backend)

    # Create a mock Runtime with store
    store = InMemoryStore()
    state = {"messages": []}
    runtime = SimpleNamespace(
        context=None,
        store=store,
        stream_writer=lambda _: None,
    )

    backend = middleware._get_backend(state, runtime, {})  # type: ignore
    assert isinstance(backend, StoreBackend)
    assert backend.runtime is not None


def test_memory_middleware_with_store_backend_assistant_id() -> None:
    """Test namespace isolation: each assistant_id gets its own memory namespace."""
    # Setup
    middleware = MemoryMiddleware(
        backend=lambda rt: StoreBackend(rt),
        sources=["/memory/AGENTS.md"],
    )
    store = InMemoryStore()
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add memory for assistant-123 with namespace (assistant-123, filesystem)
    assistant_1_content = make_memory_content("Assistant 1", "- Context for assistant 1")
    store.put(
        ("assistant-123", "filesystem"),
        "/memory/AGENTS.md",
        create_store_memory_item(assistant_1_content),
    )

    # Test: assistant-123 can read its own memory
    config_1 = {"metadata": {"assistant_id": "assistant-123"}}
    result_1 = middleware.before_agent({}, runtime, config_1)  # type: ignore

    assert result_1 is not None
    assert "/memory/AGENTS.md" in result_1["memory_contents"]
    assert "Context for assistant 1" in result_1["memory_contents"]["/memory/AGENTS.md"]

    # Test: assistant-456 cannot see assistant-123's memory (different namespace)
    # and raises error because no memory exists yet
    config_2 = {"metadata": {"assistant_id": "assistant-456"}}
    with pytest.raises(ValueError, match="Failed to download.*file_not_found"):
        middleware.before_agent({}, runtime, config_2)  # type: ignore

    # Add memory for assistant-456 with namespace (assistant-456, filesystem)
    assistant_2_content = make_memory_content("Assistant 2", "- Context for assistant 2")
    store.put(
        ("assistant-456", "filesystem"),
        "/memory/AGENTS.md",
        create_store_memory_item(assistant_2_content),
    )

    # Test: assistant-456 can read its own memory
    result_3 = middleware.before_agent({}, runtime, config_2)  # type: ignore

    assert result_3 is not None
    assert "/memory/AGENTS.md" in result_3["memory_contents"]
    assert "Context for assistant 2" in result_3["memory_contents"]["/memory/AGENTS.md"]
    assert "Context for assistant 1" not in result_3["memory_contents"]["/memory/AGENTS.md"]

    # Test: assistant-123 still only sees its own memory (no cross-contamination)
    result_4 = middleware.before_agent({}, runtime, config_1)  # type: ignore

    assert result_4 is not None
    assert "/memory/AGENTS.md" in result_4["memory_contents"]
    assert "Context for assistant 1" in result_4["memory_contents"]["/memory/AGENTS.md"]
    assert "Context for assistant 2" not in result_4["memory_contents"]["/memory/AGENTS.md"]


def test_memory_middleware_with_store_backend_no_assistant_id() -> None:
    """Test default namespace: when no assistant_id is provided, uses (filesystem,) namespace."""
    # Setup
    middleware = MemoryMiddleware(
        backend=lambda rt: StoreBackend(rt),
        sources=["/memory/AGENTS.md"],
    )
    store = InMemoryStore()
    runtime = SimpleNamespace(context=None, store=store, stream_writer=lambda _: None)

    # Add memory to default namespace (filesystem,) - no assistant_id
    shared_content = make_memory_content("Shared Memory", "- Default namespace context")
    store.put(
        ("filesystem",),
        "/memory/AGENTS.md",
        create_store_memory_item(shared_content),
    )

    # Test: empty config accesses default namespace
    result_1 = middleware.before_agent({}, runtime, {})  # type: ignore

    assert result_1 is not None
    assert "/memory/AGENTS.md" in result_1["memory_contents"]
    assert "Default namespace context" in result_1["memory_contents"]["/memory/AGENTS.md"]

    # Test: config with metadata but no assistant_id also uses default namespace
    config_with_other_metadata = {"metadata": {"some_other_key": "value"}}
    result_2 = middleware.before_agent({}, runtime, config_with_other_metadata)  # type: ignore

    assert result_2 is not None
    assert "/memory/AGENTS.md" in result_2["memory_contents"]
    assert "Default namespace context" in result_2["memory_contents"]["/memory/AGENTS.md"]


def test_create_deep_agent_with_memory_and_filesystem_backend(tmp_path: Path) -> None:
    """Test end-to-end: create_deep_agent with memory parameter and FilesystemBackend."""
    # Create memory on filesystem
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    memory_path = str(tmp_path / "user" / "AGENTS.md")
    memory_content = make_memory_content("Deep Agent Test", "- Use deep agents wisely")

    backend.upload_files([(memory_path, memory_content.encode("utf-8"))])

    # Create agent with memory parameter
    agent = create_deep_agent(
        backend=backend,
        memory=[memory_path],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Memory loaded successfully.")])),
    )

    # Invoke agent
    result = agent.invoke({"messages": [HumanMessage(content="What do you know?")]})

    # Verify invocation succeeded
    assert "messages" in result
    assert len(result["messages"]) > 0


def test_create_deep_agent_with_memory_missing_files(tmp_path: Path) -> None:
    """Test that memory raises an error when files don't exist."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    # Create agent with non-existent memory files
    agent = create_deep_agent(
        backend=backend,
        memory=[str(tmp_path / "nonexistent" / "AGENTS.md")],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="No memory, but that's okay.")])),
    )

    # Invoke agent - should raise error for missing memory file
    with pytest.raises(ValueError, match="Failed to download.*file_not_found"):
        agent.invoke({"messages": [HumanMessage(content="Hello")]})


def test_create_deep_agent_with_memory_default_backend() -> None:
    """Test create_deep_agent with memory parameter using default backend (StateBackend).

    When no backend is specified, StateBackend is used by tools. The MemoryMiddleware
    should receive a StateBackend factory and be able to load memory from state files.
    """
    checkpointer = InMemorySaver()
    agent = create_deep_agent(
        memory=["/user/.deepagents/AGENTS.md"],
        model=GenericFakeChatModel(messages=iter([AIMessage(content="Working with default backend.")])),
        checkpointer=checkpointer,
    )

    # Create memory content
    memory_content = make_memory_content("User Memory", "- Be helpful and concise")
    timestamp = datetime.now(UTC).isoformat()

    # Prepare files dict with FileData format (for StateBackend)
    memory_files = {
        "/user/.deepagents/AGENTS.md": {
            "content": memory_content.split("\n"),
            "created_at": timestamp,
            "modified_at": timestamp,
        }
    }

    config: RunnableConfig = {"configurable": {"thread_id": "123"}}

    # Invoke agent with files parameter
    result = agent.invoke(
        {
            "messages": [HumanMessage(content="What's in your memory?")],
            "files": memory_files,
        },
        config,
    )

    assert len(result["messages"]) > 0

    # Verify memory was loaded from state
    checkpoint = agent.checkpointer.get(config)
    assert "/user/.deepagents/AGENTS.md" in checkpoint["channel_values"]["files"]
    assert "memory_contents" in checkpoint["channel_values"]
    assert "/user/.deepagents/AGENTS.md" in checkpoint["channel_values"]["memory_contents"]


def test_memory_middleware_order_matters(tmp_path: Path) -> None:
    """Test that memory sources are combined in order."""
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

    # Invoke
    result = agent.invoke({"messages": [HumanMessage(content="Test")]})

    # Verify order in system prompt
    first_call = fake_model.call_history[0]
    system_message = first_call["messages"][0]
    content = system_message.text

    # First should appear before second
    first_pos = content.find("First memory content")
    second_pos = content.find("Second memory content")
    assert first_pos > 0
    assert second_pos > 0
    assert first_pos < second_pos
