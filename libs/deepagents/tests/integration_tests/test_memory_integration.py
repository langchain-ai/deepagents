"""Comprehensive integration tests for MemoryMiddleware.

These tests verify end-to-end behavior of the memory system.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.memory import (
    MEMORY_SYSTEM_PROMPT,
    MemoryMiddleware,
    MemorySource,
    MemoryState,
    MemoryStateUpdate,
)


class TestMemoryLoadingFlow:
    """Test the complete memory loading flow."""

    def test_full_loading_flow(self, tmp_path: Path) -> None:
        """Test complete flow: init -> before_agent -> wrap_model_call."""
        # Setup: Create memory files
        user_dir = tmp_path / "user"
        user_dir.mkdir()
        user_memory_content = "# User Preferences\n\n- Be concise\n- Use type hints"
        (user_dir / "AGENTS.md").write_text(user_memory_content)

        project_dir = tmp_path / "project"
        project_dir.mkdir()
        project_memory_content = "# Project Rules\n\n- Use pytest\n- 4-space indent"
        (project_dir / "AGENTS.md").write_text(project_memory_content)

        # Create middleware
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(user_dir / "AGENTS.md"), "name": "user"},
                {"path": str(project_dir / "AGENTS.md"), "name": "project"},
            ],
        )

        # Step 1: before_agent should load memory
        initial_state: MemoryState = {}
        state_update = middleware.before_agent(initial_state, None)  # type: ignore

        assert state_update is not None
        assert "memory_contents" in state_update
        assert "user" in state_update["memory_contents"]
        assert "project" in state_update["memory_contents"]
        assert state_update["memory_contents"]["user"] == user_memory_content
        assert state_update["memory_contents"]["project"] == project_memory_content

        # Step 2: Simulate state being updated (as LangGraph would do)
        updated_state: MemoryState = {
            "memory_contents": state_update["memory_contents"]
        }

        # Step 3: wrap_model_call should inject memory into prompt
        mock_request = MagicMock()
        mock_request.state = updated_state
        mock_request.system_prompt = "You are a helpful assistant."

        captured_prompt = None

        def capture_handler(req: Any) -> MagicMock:
            nonlocal captured_prompt
            captured_prompt = req.system_prompt
            return MagicMock()

        mock_request.override = lambda **kwargs: type(
            "MockRequest",
            (),
            {"system_prompt": kwargs.get("system_prompt"), "state": mock_request.state},
        )()

        middleware.wrap_model_call(mock_request, capture_handler)

        # Verify the prompt contains everything expected
        assert captured_prompt is not None
        assert "<user_memory>" in captured_prompt
        assert user_memory_content in captured_prompt
        assert "</user_memory>" in captured_prompt
        assert "<project_memory>" in captured_prompt
        assert project_memory_content in captured_prompt
        assert "</project_memory>" in captured_prompt
        assert "You are a helpful assistant." in captured_prompt
        assert "Agent Memory" in captured_prompt

    def test_memory_persists_across_calls(self, tmp_path: Path) -> None:
        """Test that memory is only loaded once and persists."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Initial content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        # First call - should load
        state1: MemoryState = {}
        update1 = middleware.before_agent(state1, None)  # type: ignore
        assert update1 is not None
        assert update1["memory_contents"]["test"] == "Initial content"

        # Simulate file change
        (memory_dir / "AGENTS.md").write_text("Changed content")

        # Second call with memory already in state - should NOT reload
        state2: MemoryState = {"memory_contents": update1["memory_contents"]}
        update2 = middleware.before_agent(state2, None)  # type: ignore
        assert update2 is None  # No update needed

        # Third call with fresh state - SHOULD reload (gets new content)
        state3: MemoryState = {}
        update3 = middleware.before_agent(state3, None)  # type: ignore
        assert update3 is not None
        assert update3["memory_contents"]["test"] == "Changed content"


class TestMemorySourceCombination:
    """Test how multiple memory sources are combined."""

    def test_sources_combined_in_order(self, tmp_path: Path) -> None:
        """Test that sources are combined in the order specified."""
        # Create three sources
        for name in ["first", "second", "third"]:
            d = tmp_path / name
            d.mkdir()
            (d / "AGENTS.md").write_text(f"Content from {name}")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(tmp_path / "first" / "AGENTS.md"), "name": "first"},
                {"path": str(tmp_path / "second" / "AGENTS.md"), "name": "second"},
                {"path": str(tmp_path / "third" / "AGENTS.md"), "name": "third"},
            ],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        # All three should be loaded
        assert len(update["memory_contents"]) == 3

        # Format and check order
        formatted = middleware._format_memory_contents(update["memory_contents"])

        first_pos = formatted.find("<first_memory>")
        second_pos = formatted.find("<second_memory>")
        third_pos = formatted.find("<third_memory>")

        assert first_pos < second_pos < third_pos

    def test_partial_sources_loaded(self, tmp_path: Path) -> None:
        """Test that missing sources don't block others from loading."""
        # Only create one of the sources
        existing = tmp_path / "existing"
        existing.mkdir()
        (existing / "AGENTS.md").write_text("I exist")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(tmp_path / "missing" / "AGENTS.md"), "name": "missing"},
                {"path": str(existing / "AGENTS.md"), "name": "existing"},
                {"path": str(tmp_path / "also_missing" / "AGENTS.md"), "name": "also_missing"},
            ],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        # Only existing should be loaded
        assert "existing" in update["memory_contents"]
        assert "missing" not in update["memory_contents"]
        assert "also_missing" not in update["memory_contents"]
        assert update["memory_contents"]["existing"] == "I exist"


class TestSystemPromptInjection:
    """Test how memory is injected into the system prompt."""

    def test_memory_prepended_to_existing_prompt(self, tmp_path: Path) -> None:
        """Test that memory is prepended, not appended."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Memory content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        mock_request = MagicMock()
        mock_request.state = {"memory_contents": {"test": "Memory content"}}
        mock_request.system_prompt = "ORIGINAL PROMPT"
        mock_request.override = lambda **kwargs: type(
            "MockRequest", (), {"system_prompt": kwargs.get("system_prompt")}
        )()

        result = middleware.modify_request(mock_request)

        # Memory should come BEFORE original prompt
        memory_pos = result.system_prompt.find("Memory content")
        original_pos = result.system_prompt.find("ORIGINAL PROMPT")
        assert memory_pos < original_pos

    def test_empty_system_prompt_handled(self, tmp_path: Path) -> None:
        """Test injection when there's no existing system prompt."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Memory content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        mock_request = MagicMock()
        mock_request.state = {"memory_contents": {"test": "Memory content"}}
        mock_request.system_prompt = None  # No existing prompt
        mock_request.override = lambda **kwargs: type(
            "MockRequest", (), {"system_prompt": kwargs.get("system_prompt")}
        )()

        result = middleware.modify_request(mock_request)

        assert "Memory content" in result.system_prompt
        assert "Agent Memory" in result.system_prompt

    def test_no_memory_loaded_message(self, tmp_path: Path) -> None:
        """Test message when no memory files exist."""
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(tmp_path / "nonexistent" / "AGENTS.md"), "name": "test"}],
        )

        mock_request = MagicMock()
        mock_request.state = {"memory_contents": {}}  # Empty - nothing loaded
        mock_request.system_prompt = "Base prompt"
        mock_request.override = lambda **kwargs: type(
            "MockRequest", (), {"system_prompt": kwargs.get("system_prompt")}
        )()

        result = middleware.modify_request(mock_request)

        assert "No memory loaded" in result.system_prompt


class TestBackendCompatibility:
    """Test compatibility with different backend types."""

    def test_filesystem_backend_direct(self, tmp_path: Path) -> None:
        """Test with FilesystemBackend passed directly."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Direct backend test")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        assert update["memory_contents"]["test"] == "Direct backend test"

    def test_factory_backend_pattern(self, tmp_path: Path) -> None:
        """Test with factory function for backend creation."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Factory backend test")

        real_backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

        # Factory that returns the real backend
        factory_called = False

        def backend_factory(runtime: Any) -> FilesystemBackend:
            nonlocal factory_called
            factory_called = True
            return real_backend

        middleware = MemoryMiddleware(
            backend=backend_factory,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        mock_runtime = MagicMock()
        update = middleware.before_agent(state, mock_runtime)

        assert factory_called, "Factory should have been called"
        assert update["memory_contents"]["test"] == "Factory backend test"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_memory_file(self, tmp_path: Path) -> None:
        """Test handling of empty AGENTS.md file."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        # Empty files are not included (nothing to load)
        assert "test" not in update["memory_contents"]

    def test_large_memory_file(self, tmp_path: Path) -> None:
        """Test handling of large memory files."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        # Create a large file (100KB)
        large_content = "# Large Memory\n\n" + ("x" * 100000)
        (memory_dir / "AGENTS.md").write_text(large_content)

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        assert update["memory_contents"]["test"] == large_content

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Test handling of unicode characters."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        unicode_content = "# Unicode Test\n\n- 你好世界\n- مرحبا\n- 🎉🚀💻\n- Ñoño"
        (memory_dir / "AGENTS.md").write_text(unicode_content, encoding="utf-8")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        assert "你好世界" in update["memory_contents"]["test"]
        assert "🎉🚀💻" in update["memory_contents"]["test"]

    def test_multiline_content_preserved(self, tmp_path: Path) -> None:
        """Test that multiline content and formatting is preserved."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()

        content = """# Project Guidelines

## Code Style

```python
def example():
    return "indentation preserved"
```

## Notes

- Item 1
- Item 2
  - Nested item

> Blockquote
"""
        (memory_dir / "AGENTS.md").write_text(content)

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        loaded = update["memory_contents"]["test"]
        assert "```python" in loaded
        assert "def example():" in loaded
        assert "- Nested item" in loaded
        assert "> Blockquote" in loaded


class TestAsyncBehavior:
    """Test async wrapper behavior."""

    @pytest.mark.asyncio
    async def test_awrap_model_call(self, tmp_path: Path) -> None:
        """Test async wrap_model_call."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Async test content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "test"}],
        )

        mock_request = MagicMock()
        mock_request.state = {"memory_contents": {"test": "Async test content"}}
        mock_request.system_prompt = "Base prompt"

        captured_prompt = None

        async def async_handler(req: Any) -> MagicMock:
            nonlocal captured_prompt
            captured_prompt = req.system_prompt
            return MagicMock()

        mock_request.override = lambda **kwargs: type(
            "MockRequest",
            (),
            {"system_prompt": kwargs.get("system_prompt"), "state": mock_request.state},
        )()

        await middleware.awrap_model_call(mock_request, async_handler)

        assert captured_prompt is not None
        assert "Async test content" in captured_prompt


class TestComparisonWithCLI:
    """Test that SDK behavior matches CLI conventions."""

    def test_xml_tag_format_matches_cli(self, tmp_path: Path) -> None:
        """Test that memory is wrapped in XML tags like CLI does."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "AGENTS.md").write_text("Test content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[{"path": str(memory_dir / "AGENTS.md"), "name": "user"}],
        )

        # CLI uses: <user_memory>...</user_memory>
        contents = {"user": "Test content"}
        formatted = middleware._format_memory_contents(contents)

        assert "<user_memory>" in formatted
        assert "</user_memory>" in formatted
        assert "Test content" in formatted

    def test_multiple_sources_format(self, tmp_path: Path) -> None:
        """Test that multiple sources are formatted correctly."""
        for name in ["user", "project"]:
            d = tmp_path / name
            d.mkdir()
            (d / "AGENTS.md").write_text(f"{name} content")

        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        middleware = MemoryMiddleware(
            backend=backend,
            sources=[
                {"path": str(tmp_path / "user" / "AGENTS.md"), "name": "user"},
                {"path": str(tmp_path / "project" / "AGENTS.md"), "name": "project"},
            ],
        )

        state: MemoryState = {}
        update = middleware.before_agent(state, None)  # type: ignore

        formatted = middleware._format_memory_contents(update["memory_contents"])

        # Should have both sections
        assert "<user_memory>" in formatted
        assert "user content" in formatted
        assert "</user_memory>" in formatted
        assert "<project_memory>" in formatted
        assert "project content" in formatted
        assert "</project_memory>" in formatted
