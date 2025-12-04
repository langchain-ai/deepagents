"""Unit tests for RAG (Retrieval-Augmented Generation) module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from deepagents_cli.rag import CodeRAG
from deepagents_cli.config import settings as rag_settings


class TestCodeRAGInitialization:
    """Test CodeRAG class initialization."""

    def test_init_with_valid_directory(self, tmp_path: Path) -> None:
        """Test initialization with a valid directory."""
        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)

            assert rag.workspace_root == tmp_path.resolve()
            assert rag.cache_dir.exists()
            mock_embeddings.assert_called_once()

    def test_init_with_nonexistent_directory(self) -> None:
        """Test that initialization raises FileNotFoundError for nonexistent directory."""
        with patch("deepagents_cli.rag.OpenAIEmbeddings"):
            os.environ["OPENAI_API_KEY"] = "test-key"

            with pytest.raises(FileNotFoundError, match="does not exist"):
                CodeRAG(workspace_root="/nonexistent/path/12345")

    def test_init_with_file_instead_of_directory(self, tmp_path: Path) -> None:
        """Test that initialization raises ValueError for file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with patch("deepagents_cli.rag.OpenAIEmbeddings"):
            os.environ["OPENAI_API_KEY"] = "test-key"

            with pytest.raises(ValueError, match="must be a directory"):
                CodeRAG(workspace_root=test_file)

    def test_init_without_api_key(self, tmp_path: Path) -> None:
        """Test that initialization raises ValueError when API key is missing."""
        # Import settings to patch the underlying openai_api_key
        
        # Save original value
        original_key = rag_settings.openai_api_key
        
        # Set openai_api_key to None, which makes has_openai return False
        rag_settings.openai_api_key = None
        
        try:
            with pytest.raises(ValueError, match="API key not found"):
                CodeRAG(workspace_root=tmp_path)
        finally:
            # Restore original value
            rag_settings.openai_api_key = original_key

    def test_init_with_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test initialization with custom cache directory."""
        custom_cache = tmp_path / "custom_cache"
        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path, cache_dir=custom_cache)

            assert rag.cache_dir == custom_cache
            assert custom_cache.exists()


class TestCodeRAGFileDiscovery:
    """Test code file discovery functionality."""

    def test_get_code_files_finds_python_files(self, tmp_path: Path) -> None:
        """Test that Python files are discovered."""
        # Create test files
        (tmp_path / "test.py").write_text("def hello(): pass")
        (tmp_path / "test.txt").write_text("not code")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "test.pyc").write_text("bytecode")

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            files = rag._get_code_files()

            file_paths = [f.name for f in files]
            assert "test.py" in file_paths
            assert "test.txt" in file_paths  # .txt is in code_extensions
            assert "test.pyc" not in file_paths  # Excluded by pattern

    def test_get_code_files_excludes_patterns(self, tmp_path: Path) -> None:
        """Test that excluded patterns are not included."""
        (tmp_path / "test.py").write_text("code")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "package.js").write_text("package")

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            files = rag._get_code_files()

            file_paths = [str(f) for f in files]
            # Should not include files in .git or node_modules
            assert not any(".git" in p for p in file_paths)
            assert not any("node_modules" in p for p in file_paths)

    def test_get_code_files_finds_multiple_extensions(self, tmp_path: Path) -> None:
        """Test that multiple file extensions are discovered."""
        (tmp_path / "test.py").write_text("python")
        (tmp_path / "test.js").write_text("javascript")
        (tmp_path / "test.md").write_text("markdown")
        (tmp_path / "test.yaml").write_text("yaml: config")

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            files = rag._get_code_files()

            file_names = [f.name for f in files]
            assert "test.py" in file_names
            assert "test.js" in file_names
            assert "test.md" in file_names
            assert "test.yaml" in file_names


class TestCodeRAGChunking:
    """Test code file chunking functionality."""

    def test_chunk_code_file_creates_documents(self, tmp_path: Path) -> None:
        """Test that code files are chunked into documents."""
        test_file = tmp_path / "test.py"
        content = "def function1():\n    pass\n\ndef function2():\n    pass\n"
        test_file.write_text(content)

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            documents = rag._chunk_code_file(test_file)

            assert len(documents) > 0
            assert all(doc.metadata["file_path"] == str(test_file) for doc in documents)
            assert all("relative_path" in doc.metadata for doc in documents)
            assert all("file_name" in doc.metadata for doc in documents)

    def test_chunk_code_file_handles_encoding_errors(self, tmp_path: Path) -> None:
        """Test that encoding errors are handled gracefully."""
        test_file = tmp_path / "test.bin"
        # Write binary data that can't be decoded as UTF-8
        test_file.write_bytes(b"\xff\xfe\x00\x01")

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            documents = rag._chunk_code_file(test_file)

            # Should handle encoding errors gracefully (errors="ignore" in read_text)
            # May return documents with partial content or empty content
            assert isinstance(documents, list)
            # The file should still be processed, just with encoding errors ignored


class TestCodeRAGIndexing:
    """Test codebase indexing functionality."""

    @patch("deepagents_cli.rag.FAISS")
    def test_index_creates_vector_store(self, mock_faiss: Mock, tmp_path: Path) -> None:
        """Test that indexing creates a vector store."""
        (tmp_path / "test.py").write_text("def hello(): pass")

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            rag.index()

            # FAISS.from_documents should be called
            mock_faiss.from_documents.assert_called_once()

    def test_index_with_no_files_raises_error(self, tmp_path: Path) -> None:
        """Test that indexing with no code files raises ValueError."""
        # Empty directory
        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)

            with pytest.raises(ValueError, match="No code files found"):
                rag.index()

    @patch("deepagents_cli.rag.FAISS")
    def test_index_loads_existing_cache(self, mock_faiss: Mock, tmp_path: Path) -> None:
        """Test that existing cache is loaded when available."""
        (tmp_path / "test.py").write_text("def hello(): pass")

        cache_path = tmp_path / "cache"
        cache_path.mkdir()

        # Mock existing cache
        mock_vector_store = Mock()
        mock_faiss.load_local.return_value = mock_vector_store

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path, cache_dir=cache_path)
            # Mock the cache path to exist
            with patch.object(rag, "_get_cache_path", return_value=cache_path / "vectorstore"):
                with patch.object(rag, "_should_reindex", return_value=False):
                    with patch.object(Path, "exists", return_value=True):
                        rag.index()

                        # Should load from cache, not create new
                        mock_faiss.load_local.assert_called()


class TestCodeRAGSearch:
    """Test semantic search functionality."""

    def test_search_without_index_raises_error(self, tmp_path: Path) -> None:
        """Test that search raises error if vector store is not initialized."""
        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)

            with pytest.raises(ValueError, match="not initialized"):
                rag.search("test query")

    @patch("deepagents_cli.rag.FAISS")
    def test_search_returns_formatted_results(
        self, mock_faiss: Mock, tmp_path: Path
    ) -> None:
        """Test that search returns properly formatted results."""
        (tmp_path / "test.py").write_text("def hello(): pass")

        # Mock vector store and search results
        mock_doc = Mock()
        mock_doc.page_content = "def hello(): pass"
        mock_doc.metadata = {
            "file_path": str(tmp_path / "test.py"),
            "relative_path": "test.py",
            "file_name": "test.py",
            "chunk_index": 0,
            "file_type": ".py",
        }

        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        mock_faiss.from_documents.return_value = mock_vector_store

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            rag.index()
            results = rag.search("hello function", k=1)

            assert len(results) == 1
            assert results[0]["content"] == "def hello(): pass"
            assert results[0]["file_path"] == str(tmp_path / "test.py")
            assert results[0]["score"] == 0.5

    @patch("deepagents_cli.rag.FAISS")
    def test_search_with_file_type_filter(
        self, mock_faiss: Mock, tmp_path: Path
    ) -> None:
        """Test that search can filter by file type."""
        (tmp_path / "test.py").write_text("python code")
        (tmp_path / "test.js").write_text("javascript code")

        mock_vector_store = Mock()
        mock_vector_store.similarity_search_with_score.return_value = []
        mock_faiss.from_documents.return_value = mock_vector_store

        with patch("deepagents_cli.rag.OpenAIEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = Mock()
            os.environ["OPENAI_API_KEY"] = "test-key"

            rag = CodeRAG(workspace_root=tmp_path)
            rag.index()
            rag.search("test", k=5, filter_metadata={"file_type": ".py"})

            # Verify search was called
            mock_vector_store.similarity_search_with_score.assert_called()
