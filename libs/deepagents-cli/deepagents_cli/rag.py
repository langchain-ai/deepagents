"""RAG (Retrieval-Augmented Generation) system for semantic code search."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

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
        # Default fallback splitter for unknown file types
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # Mapping of file extensions to Language enums for code-aware chunking
        self._extension_to_language: dict[str, Language] = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".jsx": Language.JS,
            ".ts": Language.TS,
            ".tsx": Language.TS,
            ".java": Language.JAVA,
            ".cpp": Language.CPP,
            ".c": Language.C,
            ".go": Language.GO,
            ".rs": Language.RUST,
            ".rb": Language.RUBY,
            ".php": Language.PHP,
            ".swift": Language.SWIFT,
            ".kt": Language.KOTLIN,
            ".scala": Language.SCALA,
        }
        
        # Language-specific splitters (lazy-loaded)
        self._language_splitters: dict[str, RecursiveCharacterTextSplitter] = {}

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

    def _get_code_files(self) -> list[Path]:
        """Get all code files in the workspace.
        
        Only indexes actual source code files, excluding:
        - Config files (.yaml, .json, .toml, etc.)
        - Documentation files (.md, .txt, .rst)
        - Shell scripts (.sh, .bash, .zsh)
        - Build artifacts and dependencies

        Returns:
            List of code file paths
        """
        # Only actual source code file extensions
        code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php", ".swift",
            ".kt", ".scala",
        }
        
        # Comprehensive list of directories to skip (similar to Cursor IDE)
        # Includes: dependencies, build artifacts, caches, IDE configs, OS files
        skip_dirs = {
            # Version control
            ".git", ".svn", ".hg", ".bzr",
            # Dependencies
            "node_modules", "vendor", "bower_components", "jspm_packages",
            ".venv", "venv", "env", ".env", "virtualenv", "ENV",
            # Python
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            ".python-version", "pip-log.txt", "pip-delete-this-directory.txt",
            # Build artifacts
            "dist", "build", ".build", "target", "bin", "obj", "out",
            ".next", ".nuxt", ".turbo", ".vercel", ".netlify",
            # JavaScript/TypeScript
            ".parcel-cache", ".cache", ".eslintcache", ".yarn",
            # Java
            ".gradle", ".mvn", "gradle", "maven",
            # Rust
            "target",
            # IDE and editor configs
            ".vscode", ".idea", ".vs", ".cursor", ".settings",
            # OS files
            ".DS_Store", "Thumbs.db", ".Spotlight-V100", ".Trashes",
            # Test coverage
            "coverage", ".coverage", "htmlcov", ".nyc_output",
            # Logs and temporary
            "logs", "tmp", "temp", ".tmp",
            # Other common
            ".sass-cache", ".terraform", ".serverless", ".dynamodb",
            ".eggs",
        }

        files = []
        for ext in code_extensions:
            for file_path in self.workspace_root.rglob(f"*{ext}"):
                if not file_path.is_file():
                    continue
                
                # Skip files in common non-code directories
                path_parts = file_path.parts
                if any(skip_dir in path_parts for skip_dir in skip_dirs):
                    continue
                
                files.append(file_path)

        return files

    def _get_splitter_for_file(self, file_path: Path) -> RecursiveCharacterTextSplitter:
        """Get the appropriate text splitter for a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Text splitter instance (language-aware if supported, otherwise default)
        """
        suffix = file_path.suffix.lower()
        
        # Check if we have a language mapping for this extension
        if suffix in self._extension_to_language:
            # Lazy-load splitter if not already created
            if suffix not in self._language_splitters:
                language = self._extension_to_language[suffix]
                self._language_splitters[suffix] = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=1000,
                    chunk_overlap=200,
                )
            return self._language_splitters[suffix]
        
        return self.default_splitter

    def _chunk_code_file(self, file_path: Path) -> list[Document]:
        """Chunk a code file into documents with metadata using code-aware splitting.

        Uses language-specific splitters that respect code structure (functions, classes, etc.)
        when available, falling back to character-based splitting for unsupported languages.

        Args:
            file_path: Path to the code file

        Returns:
            List of Document objects
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        # Get appropriate splitter for this file type
        splitter = self._get_splitter_for_file(file_path)
        
        # Create chunks using language-aware splitting
        chunks = splitter.split_text(content)

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

    def _lexical_search(
        self, query: str, k: int = 5, filter_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Perform lexical (grep-like) search on the codebase.
        
        Searches for exact keyword matches in file contents. This complements
        semantic search by finding exact matches that semantic search might miss.
        
        Args:
            query: Search query (keywords)
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"file_type": ".py"})
            
        Returns:
            List of search results with lexical match scores
        """
        # Extract keywords from query (simple tokenization)
        keywords = re.findall(r'\b\w+\b', query.lower())
        if not keywords:
            return []
        
        code_files = self._get_code_files()
        results = []
        
        for file_path in code_files:
            # Apply file type filter if specified
            if filter_metadata and filter_metadata.get("file_type"):
                if file_path.suffix != filter_metadata["file_type"]:
                    continue
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                content_lower = content.lower()
                
                # Count keyword matches
                match_count = sum(1 for keyword in keywords if keyword in content_lower)
                if match_count == 0:
                    continue
                
                # Calculate a simple score based on keyword density
                # More keywords found = higher score
                # Normalize by content length to avoid bias toward large files
                score = match_count / max(len(keywords), 1) / max(len(content) / 1000, 1)
                
                # Find context around matches (first 500 chars with matches)
                lines = content.split('\n')
                matching_lines = []
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in keywords):
                        # Include some context (previous and next line)
                        start = max(0, i - 1)
                        end = min(len(lines), i + 2)
                        context = '\n'.join(lines[start:end])
                        matching_lines.append((i, context))
                        if len(matching_lines) >= 3:  # Limit context snippets
                            break
                
                if matching_lines:
                    # Use first matching context as preview
                    preview_content = matching_lines[0][1]
                    
                    rel_path = file_path.relative_to(self.workspace_root)
                    results.append({
                        "content": preview_content,
                        "file_path": str(file_path),
                        "relative_path": str(rel_path),
                        "file_name": file_path.name,
                        "chunk_index": 0,
                        "file_type": file_path.suffix,
                        "score": score,
                        "match_type": "lexical",
                    })
            except Exception:
                continue
        
        # Sort by score (higher is better for lexical)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_metadata: dict[str, Any] | None = None,
        use_hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search (semantic + lexical) on the indexed codebase.

        Combines semantic search (meaning-based) with lexical search (keyword-based)
        to provide comprehensive results similar to Cursor's approach.

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"file_type": ".py"})
            use_hybrid: If True, combines semantic and lexical search (default: True)

        Returns:
            List of search results, each containing:
            - content: The code chunk content
            - file_path: Full path to the file
            - relative_path: Relative path from workspace root
            - file_name: Name of the file
            - score: Similarity score (lower is better for semantic, higher for lexical)
            - match_type: "semantic" or "lexical"
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call index() first.")

        all_results = []
        
        # Perform semantic search
        if filter_metadata:
            # FAISS doesn't support metadata filtering directly, so we'll filter after
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k * 2)
            # Filter by metadata
            filtered_results = []
            for doc, score in semantic_results:
                matches = True
                for key, value in filter_metadata.items():
                    if doc.metadata.get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append((doc, score))
                    if len(filtered_results) >= k:
                        break
            semantic_results = filtered_results[:k]
        else:
            semantic_results = self.vector_store.similarity_search_with_score(query, k=k)

        # Format semantic results
        for doc, score in semantic_results:
            all_results.append({
                "content": doc.page_content,
                "file_path": doc.metadata.get("file_path", ""),
                "relative_path": doc.metadata.get("relative_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "file_type": doc.metadata.get("file_type", ""),
                "score": float(score),  # Lower is better for semantic
                "match_type": "semantic",
            })
        
        # Add lexical search results if hybrid mode is enabled
        if use_hybrid:
            lexical_results = self._lexical_search(query, k=k, filter_metadata=filter_metadata)
            
            # Merge results, avoiding duplicates by file_path
            seen_files = {r["file_path"] for r in all_results}
            for lex_result in lexical_results:
                # Only add if not already present (prefer semantic over lexical for same file)
                if lex_result["file_path"] not in seen_files:
                    all_results.append(lex_result)
                    seen_files.add(lex_result["file_path"])
        
        # Normalize and sort results
        # For semantic: lower score is better, for lexical: higher score is better
        # We'll normalize semantic scores to be comparable
        normalized_results = []
        semantic_scores = [r["score"] for r in all_results if r["match_type"] == "semantic"]
        lexical_scores = [r["score"] for r in all_results if r["match_type"] == "lexical"]
        
        # Normalize semantic scores (invert so higher = better, like lexical)
        if semantic_scores:
            max_semantic = max(semantic_scores) if semantic_scores else 1.0
            min_semantic = min(semantic_scores) if semantic_scores else 0.0
            semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1.0
        
        # Normalize lexical scores to similar range
        if lexical_scores:
            max_lexical = max(lexical_scores) if lexical_scores else 1.0
            min_lexical = min(lexical_scores) if lexical_scores else 0.0
            lexical_range = max_lexical - min_lexical if max_lexical > min_lexical else 1.0
        
        for result in all_results:
            if result["match_type"] == "semantic":
                # Invert semantic score: lower distance = higher normalized score
                if semantic_range > 0:
                    normalized_score = 1.0 - ((result["score"] - min_semantic) / semantic_range)
                else:
                    normalized_score = 1.0
            else:  # lexical
                # Normalize lexical score to 0-1 range
                if lexical_range > 0:
                    normalized_score = (result["score"] - min_lexical) / lexical_range
                else:
                    normalized_score = result["score"]
            
            result["normalized_score"] = normalized_score
            normalized_results.append(result)
        
        # Sort by normalized score (higher is better)
        normalized_results.sort(key=lambda x: x["normalized_score"], reverse=True)
        
        # Return top k results
        return normalized_results[:k]

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
