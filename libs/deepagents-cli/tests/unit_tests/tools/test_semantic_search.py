"""Tests for semantic_search tool."""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from deepagents_cli.tools import semantic_search


class TestSemanticSearchTool:
    """Test semantic_search tool function."""

    def test_semantic_search_with_valid_directory(self, tmp_path: Path) -> None:
        """Test semantic_search with a valid directory."""
        (tmp_path / "test.py").write_text("def hello(): pass")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.return_value = [
                {
                    "content": "def hello(): pass",
                    "file_path": str(tmp_path / "test.py"),
                    "relative_path": "test.py",
                    "file_name": "test.py",
                    "score": 0.5,
                }
            ]
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="hello function",
                workspace_root=str(tmp_path),
                max_results=1,
            )

            assert "results" in result
            assert len(result["results"]) == 1
            assert result["query"] == "hello function"
            assert result["workspace_root"] == str(tmp_path.resolve())
            mock_rag.index.assert_called_once()
            mock_rag.search.assert_called_once()

    def test_semantic_search_with_nonexistent_directory(self) -> None:
        """Test semantic_search returns error for nonexistent directory."""
        result = semantic_search(
            query="test query",
            workspace_root="/nonexistent/path/12345",
            max_results=3,
        )

        assert "error" in result
        assert "does not exist" in result["error"]
        assert result["query"] == "test query"

    def test_semantic_search_with_file_instead_of_directory(
        self, tmp_path: Path
    ) -> None:
        """Test semantic_search returns error for file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        result = semantic_search(
            query="test query",
            workspace_root=str(test_file),
            max_results=3,
        )

        assert "error" in result
        assert "not a directory" in result["error"]

    def test_semantic_search_without_api_key(self, tmp_path: Path) -> None:
        """Test semantic_search handles missing API key gracefully."""
        (tmp_path / "test.py").write_text("code")

        # Remove API key if present
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
                # Simulate ValueError during initialization (caught in try/except)
                mock_rag_class.side_effect = ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
                )

                result = semantic_search(
                    query="test",
                    workspace_root=str(tmp_path),
                    max_results=3,
                )

                assert "error" in result
                assert "API key" in result["error"]
                # The error is caught in the general exception handler, so suggestion
                # may not be present - check that error message is helpful
                assert "OPENAI_API_KEY" in result["error"] or "API key" in result["error"]
        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_semantic_search_with_file_type_filter(self, tmp_path: Path) -> None:
        """Test semantic_search with file type filter."""
        (tmp_path / "test.py").write_text("python code")
        (tmp_path / "test.js").write_text("javascript code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.return_value = []
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="test",
                workspace_root=str(tmp_path),
                file_type=".py",
                max_results=5,
            )

            # Verify search was called with filter
            mock_rag.search.assert_called_once()
            call_kwargs = mock_rag.search.call_args[1]
            assert call_kwargs["filter_metadata"] == {"file_type": ".py"}

    def test_semantic_search_force_reindex(self, tmp_path: Path) -> None:
        """Test semantic_search with force_reindex flag."""
        (tmp_path / "test.py").write_text("code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.return_value = []
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="test",
                workspace_root=str(tmp_path),
                force_reindex=True,
            )

            # Verify index was called with force=True
            mock_rag.index.assert_called_once_with(force=True)

    def test_semantic_search_caches_rag_instances(self, tmp_path: Path) -> None:
        """Test that RAG instances are cached per workspace."""
        (tmp_path / "test.py").write_text("code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.return_value = []
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            # First call
            semantic_search(query="test1", workspace_root=str(tmp_path))
            # Second call with same workspace
            semantic_search(query="test2", workspace_root=str(tmp_path))

            # CodeRAG should only be instantiated once
            assert mock_rag_class.call_count == 1

    def test_semantic_search_handles_indexing_errors(self, tmp_path: Path) -> None:
        """Test that indexing errors are handled gracefully."""
        (tmp_path / "test.py").write_text("code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.index.side_effect = Exception("Indexing failed")
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="test",
                workspace_root=str(tmp_path),
            )

            assert "error" in result
            assert "Semantic search error" in result["error"]

    def test_semantic_search_handles_search_errors(self, tmp_path: Path) -> None:
        """Test that search errors are handled gracefully."""
        (tmp_path / "test.py").write_text("code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.side_effect = Exception("Search failed")
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="test",
                workspace_root=str(tmp_path),
            )

            assert "error" in result
            assert "Semantic search error" in result["error"]

    def test_semantic_search_returns_workspace_root(self, tmp_path: Path) -> None:
        """Test that result includes workspace_root information."""
        (tmp_path / "test.py").write_text("code")

        with patch("deepagents_cli.tools.CodeRAG") as mock_rag_class:
            mock_rag = Mock()
            mock_rag.search.return_value = []
            mock_rag_class.return_value = mock_rag

            os.environ["OPENAI_API_KEY"] = "test-key"

            result = semantic_search(
                query="test",
                workspace_root=str(tmp_path),
            )

            assert "workspace_root" in result
            assert result["workspace_root"] == str(tmp_path.resolve())
