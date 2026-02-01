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
