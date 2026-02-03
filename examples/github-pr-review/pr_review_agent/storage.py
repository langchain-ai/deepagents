"""Storage abstraction for state files.

Supports:
- Local filesystem (default)
- Google Cloud Storage (if GCLOUD_STORAGE_BUCKET is set)
- AWS S3 (if AWS_S3_BUCKET is set)

Priority: GCS > S3 > Local
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def read(self, path: str) -> str | None:
        """Read a file and return its contents, or None if not found."""
        pass
    
    @abstractmethod
    def write(self, path: str, content: str) -> None:
        """Write content to a file."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file exists."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file."""
        pass
    
    def read_json(self, path: str) -> dict | None:
        """Read and parse a JSON file."""
        content = self.read(path)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return None
        return None
    
    def write_json(self, path: str, data: dict, indent: int = 2) -> None:
        """Write data as JSON to a file."""
        self.write(path, json.dumps(data, indent=indent))


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _full_path(self, path: str) -> Path:
        full = self.base_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        return full
    
    def read(self, path: str) -> str | None:
        full_path = self._full_path(path)
        if full_path.exists():
            try:
                return full_path.read_text()
            except IOError:
                return None
        return None
    
    def write(self, path: str, content: str) -> None:
        full_path = self._full_path(path)
        full_path.write_text(content)
    
    def exists(self, path: str) -> bool:
        return self._full_path(path).exists()
    
    def delete(self, path: str) -> None:
        full_path = self._full_path(path)
        if full_path.exists():
            full_path.unlink()


class GCSStorage(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = ""):
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage is required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            )
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.strip("/")
    
    def _blob_name(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def read(self, path: str) -> str | None:
        blob = self.bucket.blob(self._blob_name(path))
        try:
            return blob.download_as_text()
        except Exception:
            return None
    
    def write(self, path: str, content: str) -> None:
        blob = self.bucket.blob(self._blob_name(path))
        blob.upload_from_string(content, content_type="application/json")
    
    def exists(self, path: str) -> bool:
        blob = self.bucket.blob(self._blob_name(path))
        return blob.exists()
    
    def delete(self, path: str) -> None:
        blob = self.bucket.blob(self._blob_name(path))
        try:
            blob.delete()
        except Exception:
            pass


class S3Storage(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, bucket_name: str, prefix: str = ""):
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 backend. "
                "Install with: pip install boto3"
            )
        
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.strip("/")
    
    def _key(self, path: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{path}"
        return path
    
    def read(self, path: str) -> str | None:
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=self._key(path))
            return response["Body"].read().decode("utf-8")
        except Exception:
            return None
    
    def write(self, path: str, content: str) -> None:
        self.s3.put_object(
            Bucket=self.bucket_name,
            Key=self._key(path),
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )
    
    def exists(self, path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=self._key(path))
            return True
        except Exception:
            return False
    
    def delete(self, path: str) -> None:
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=self._key(path))
        except Exception:
            pass


# Global storage instance - initialized on first use
_storage: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """Get the configured storage backend.
    
    Priority:
    1. GCLOUD_STORAGE_BUCKET -> GCS
    2. AWS_S3_BUCKET -> S3
    3. Local filesystem (STATE_DIR or ./data)
    """
    global _storage
    
    if _storage is not None:
        return _storage
    
    gcs_bucket = os.environ.get("GCLOUD_STORAGE_BUCKET")
    s3_bucket = os.environ.get("AWS_S3_BUCKET")
    storage_prefix = os.environ.get("STORAGE_PREFIX", "")
    
    if gcs_bucket:
        print(f"[storage] Using Google Cloud Storage: gs://{gcs_bucket}/{storage_prefix}")
        _storage = GCSStorage(gcs_bucket, storage_prefix)
    elif s3_bucket:
        print(f"[storage] Using AWS S3: s3://{s3_bucket}/{storage_prefix}")
        _storage = S3Storage(s3_bucket, storage_prefix)
    else:
        local_dir = os.environ.get("STATE_DIR", "./data")
        print(f"[storage] Using local filesystem: {local_dir}")
        _storage = LocalStorage(local_dir)
    
    return _storage


def get_repo_path(owner: str, repo: str, filename: str) -> str:
    """Get the storage path for a repository file."""
    return f"{owner}/{repo}/{filename}"


def get_pr_path(owner: str, repo: str, pr_number: int, filename: str) -> str:
    """Get the storage path for a PR file."""
    return f"{owner}/{repo}/prs/{pr_number}/{filename}"


def get_memory_path(owner: str, repo: str) -> str:
    """Get the storage path for a repository's AGENTS.md memory file."""
    return f"/{owner}/{repo}/AGENTS.md"


# DeepAgents Backend Adapter
# Adapts the existing storage backends to the deepagents BackendProtocol

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.backends.protocol import (
        BackendProtocol,
        FileDownloadResponse,
        FileUploadResponse,
        FileInfo,
        GrepMatch,
        WriteResult,
        EditResult,
    )


class DeepAgentsBackend:
    """Adapter that wraps our storage backends to implement deepagents BackendProtocol.

    This allows using MemoryMiddleware and FilesystemMiddleware with our existing
    storage system (local/GCS/S3).
    """

    def __init__(self, owner: str, repo: str):
        """Initialize the backend for a specific repository.

        Args:
            owner: Repository owner (e.g., 'langchain-ai')
            repo: Repository name (e.g., 'langchain')
        """
        self.owner = owner
        self.repo = repo
        self.storage = get_storage()
        # All paths are relative to the repo root
        self._path_prefix = f"{owner}/{repo}"

    def _to_storage_path(self, path: str) -> str:
        """Convert a virtual path to storage path.

        Virtual paths start with /{owner}/{repo}/, we strip that prefix
        since the storage already knows the repo context.
        """
        # Remove leading slash
        path = path.lstrip("/")
        # If path starts with owner/repo, use it directly
        if path.startswith(f"{self.owner}/{self.repo}/"):
            return path
        # Otherwise, prepend the repo prefix
        return f"{self._path_prefix}/{path}"

    def _from_storage_path(self, storage_path: str) -> str:
        """Convert storage path back to virtual path."""
        return f"/{storage_path}"

    def ls_info(self, path: str) -> list["FileInfo"]:
        """List files in a directory. Limited implementation for memory use case."""
        # For now, just return empty - we mainly need read/write/edit for memory
        return []

    async def als_info(self, path: str) -> list["FileInfo"]:
        """Async version of ls_info."""
        return self.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with line numbers."""
        storage_path = self._to_storage_path(file_path)
        content = self.storage.read(storage_path)

        if content is None:
            return f"Error: File '{file_path}' not found"

        lines = content.split("\n")

        # Apply offset and limit
        selected_lines = lines[offset:offset + limit]

        # Format with line numbers (cat -n style)
        result_lines = []
        for i, line in enumerate(selected_lines, start=offset + 1):
            result_lines.append(f"{i:6}\t{line}")

        return "\n".join(result_lines)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async version of read."""
        return self.read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> "WriteResult":
        """Create a new file with content."""
        from deepagents.backends.protocol import WriteResult

        storage_path = self._to_storage_path(file_path)

        # Check if file exists
        if self.storage.exists(storage_path):
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
            )

        self.storage.write(storage_path, content)
        return WriteResult(path=file_path, files_update=None)

    async def awrite(self, file_path: str, content: str) -> "WriteResult":
        """Async version of write."""
        return self.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> "EditResult":
        """Edit a file by replacing string occurrences."""
        from deepagents.backends.protocol import EditResult

        storage_path = self._to_storage_path(file_path)
        content = self.storage.read(storage_path)

        if content is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        if old_string == new_string:
            return EditResult(error="old_string and new_string must be different")

        occurrences = content.count(old_string)

        if occurrences == 0:
            return EditResult(error=f"old_string not found in {file_path}")

        if not replace_all and occurrences > 1:
            return EditResult(
                error=f"old_string appears {occurrences} times in {file_path}. "
                "Use replace_all=True to replace all occurrences, or provide more context to make it unique."
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        self.storage.write(storage_path, new_content)

        return EditResult(
            path=file_path,
            files_update=None,
            occurrences=occurrences if replace_all else 1,
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> "EditResult":
        """Async version of edit."""
        return self.edit(file_path, old_string, new_string, replace_all)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """Search for text pattern. Limited implementation."""
        return []

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list["GrepMatch"] | str:
        """Async version of grep_raw."""
        return self.grep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Find files matching glob pattern. Limited implementation."""
        return []

    async def aglob_info(self, pattern: str, path: str = "/") -> list["FileInfo"]:
        """Async version of glob_info."""
        return self.glob_info(pattern, path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list["FileUploadResponse"]:
        """Upload multiple files."""
        from deepagents.backends.protocol import FileUploadResponse

        responses = []
        for path, content in files:
            storage_path = self._to_storage_path(path)
            try:
                self.storage.write(storage_path, content.decode("utf-8"))
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
        return responses

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list["FileUploadResponse"]:
        """Async version of upload_files."""
        return self.upload_files(files)

    def download_files(self, paths: list[str]) -> list["FileDownloadResponse"]:
        """Download multiple files."""
        from deepagents.backends.protocol import FileDownloadResponse

        responses = []
        for path in paths:
            storage_path = self._to_storage_path(path)
            content = self.storage.read(storage_path)

            if content is None:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            else:
                responses.append(FileDownloadResponse(path=path, content=content.encode("utf-8"), error=None))

        return responses

    async def adownload_files(self, paths: list[str]) -> list["FileDownloadResponse"]:
        """Async version of download_files."""
        return self.download_files(paths)
