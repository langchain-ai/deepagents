"""RAG (Retrieval-Augmented Generation) system for semantic code search."""

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from deepagents_cli.config import settings


class CodeRAG:
    """RAG system for indexing and searching code semantically."""

    def __init__(self, workspace_root: Path | str, cache_dir: Path | None = None):
        """Initialize the RAG system.

        Args:
            workspace_root: Root directory of the codebase to index. Can be any folder path.
            cache_dir: Directory to store the vector store cache. Defaults to ~/.deepagents/rag_cache/
        
        Raises:
            ValueError: If OpenAI API key is not set
            FileNotFoundError: If workspace_root does not exist
        """
        workspace_path = Path(workspace_root).resolve()
        
        if not workspace_path.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {workspace_root}")
        
        if not workspace_path.is_dir():
            raise ValueError(f"Workspace root must be a directory: {workspace_root}")
        
        self.workspace_root = workspace_path
        if cache_dir is None:
            cache_dir = Path.home() / ".deepagents" / "rag_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings - use OpenAI by default
        # Check if API key is available before initializing
        from deepagents_cli.config import settings
        
        if not settings.has_openai:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable, "
                "or modify the RAG system to use a different embedding model."
            )
        self.embeddings = OpenAIEmbeddings()

        self.vector_store: FAISS | None = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    def _get_cache_path(self) -> Path:
        """Get the cache path for the vector store based on workspace root."""
        # Create a hash of the workspace root path for cache naming
        workspace_hash = hashlib.md5(str(self.workspace_root).encode()).hexdigest()[:8]
        return self.cache_dir / f"vectorstore_{workspace_hash}"

    def _get_metadata_path(self) -> Path:
        """Get the metadata path for tracking indexed files."""
        workspace_hash = hashlib.md5(str(self.workspace_root).encode()).hexdigest()[:8]
        return self.cache_dir / f"metadata_{workspace_hash}.json"

    def _should_reindex(self) -> bool:
        """Check if reindexing is needed based on file modifications."""
        metadata_path = self._get_metadata_path()
        cache_path = self._get_cache_path()

        if not metadata_path.exists() or not cache_path.exists():
            return True

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check if any indexed files have been modified
            for file_path_str, mtime in metadata.get("files", {}).items():
                file_path = Path(file_path_str)
                if not file_path.exists() or file_path.stat().st_mtime > mtime:
                    return True

            return False
        except Exception:
            return True

    def _get_code_files(self, exclude_patterns: list[str] | None = None) -> list[Path]:
        """Get all code files in the workspace.

        Args:
            exclude_patterns: List of glob patterns to exclude (e.g., ['*.pyc', '__pycache__'])

        Returns:
            List of code file paths
        """
        if exclude_patterns is None:
            exclude_patterns = [
                "*.pyc",
                "__pycache__",
                "*.pyo",
                "*.pyd",
                ".git",
                "node_modules",
                ".venv",
                "venv",
                "env",
                ".env",
                "*.egg-info",
                "dist",
                "build",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "*.lock",
                "uv.lock",
                ".deepagents",
            ]

        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".sh",
            ".bash",
            ".zsh",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".md",
            ".rst",
            ".txt",
        }

        files = []
        for ext in code_extensions:
            for file_path in self.workspace_root.rglob(f"*{ext}"):
                # Check if file should be excluded
                should_exclude = False
                for pattern in exclude_patterns:
                    if pattern in str(file_path) or file_path.match(pattern):
                        should_exclude = True
                        break

                if not should_exclude and file_path.is_file():
                    files.append(file_path)

        return files

    def _chunk_code_file(self, file_path: Path) -> list[Document]:
        """Chunk a code file into documents with metadata.

        Args:
            file_path: Path to the code file

        Returns:
            List of Document objects
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        # Create chunks
        chunks = self.text_splitter.split_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            rel_path = file_path.relative_to(self.workspace_root)
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_path": str(file_path),
                        "relative_path": str(rel_path),
                        "file_name": file_path.name,
                        "chunk_index": i,
                        "file_type": file_path.suffix,
                    }
                )
            )

        return documents

    def index(self, force: bool = False) -> None:
        """Index the codebase.

        Args:
            force: Force reindexing even if cache exists
        """
        if not force and not self._should_reindex():
            # Load existing vector store
            cache_path = self._get_cache_path()
            if cache_path.exists():
                try:
                    self.vector_store = FAISS.load_local(
                        str(cache_path), self.embeddings, allow_dangerous_deserialization=True
                    )
                    return
                except Exception:
                    # If loading fails, reindex
                    pass

        # Get all code files
        code_files = self._get_code_files()

        # Chunk all files
        all_documents = []
        file_metadata = {}

        for file_path in code_files:
            documents = self._chunk_code_file(file_path)
            all_documents.extend(documents)
            file_metadata[str(file_path)] = file_path.stat().st_mtime

        if not all_documents:
            raise ValueError("No code files found to index")

        # Create vector store
        self.vector_store = FAISS.from_documents(all_documents, self.embeddings)

        # Save vector store
        cache_path = self._get_cache_path()
        self.vector_store.save_local(str(cache_path))

        # Save metadata
        metadata_path = self._get_metadata_path()
        with open(metadata_path, "w") as f:
            json.dump({"files": file_metadata}, f, indent=2)

    def search(
        self, query: str, k: int = 5, filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Perform semantic search on the indexed codebase.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"file_type": ".py"})

        Returns:
            List of search results, each containing:
            - content: The code chunk content
            - file_path: Full path to the file
            - relative_path: Relative path from workspace root
            - file_name: Name of the file
            - score: Similarity score
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call index() first.")

        # Perform similarity search
        if filter_metadata:
            # FAISS doesn't support metadata filtering directly, so we'll filter after
            results = self.vector_store.similarity_search_with_score(query, k=k * 2)
            # Filter by metadata
            filtered_results = []
            for doc, score in results:
                matches = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append((doc, score))
                    if len(filtered_results) >= k:
                        break
            results = filtered_results[:k]
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "file_path": doc.metadata.get("file_path", ""),
                    "relative_path": doc.metadata.get("relative_path", ""),
                    "file_name": doc.metadata.get("file_name", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "file_type": doc.metadata.get("file_type", ""),
                    "score": float(score),
                }
            )

        return formatted_results

    def load(self) -> None:
        """Load the vector store from cache if it exists."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(cache_path), self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                raise ValueError(f"Failed to load vector store: {e}") from e
        else:
            raise ValueError("Vector store not found. Call index() first.")
