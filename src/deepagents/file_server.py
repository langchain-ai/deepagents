"""
Standalone file server for Deep Agents UI.

This module provides a FastAPI-based HTTP server that handles file reading
and downloads independently from the LangGraph server. It serves files from
thread-specific workspaces with security validation to prevent directory
traversal attacks.

The server exposes endpoints for:
- Health checks
- Reading file content with metadata
- Streaming file downloads

All file operations are restricted to thread-specific workspaces within
the configured FILE_SYSTEM_BASE directory.
"""

import os
import mimetypes
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="Deep Agents File Server",
    description="HTTP server for serving Deep Agents workspace files",
    version="1.0.0"
)

# Configure CORS middleware to allow requests from the UI
# In production, replace with specific UI domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # UI development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration: Base directory for file system paths
# This environment variable maps thread IDs to their workspace directories
FILE_SYSTEM_BASE = os.environ.get("FILE_SYSTEM_BASE", "/tmp/deepagents")

logger.info(f"File server initialized with FILE_SYSTEM_BASE: {FILE_SYSTEM_BASE}")


def get_thread_workspace(thread_id: str) -> Path:
    """
    Get the workspace directory for a given thread.
    
    Each thread has its own isolated workspace directory within FILE_SYSTEM_BASE.
    This function validates that the workspace exists and returns its Path object.
    
    Args:
        thread_id: Unique identifier for the thread/session
        
    Returns:
        Path object pointing to the thread's workspace directory
        
    Raises:
        HTTPException: 404 if the workspace directory does not exist
        
    Example:
        >>> workspace = get_thread_workspace("thread-123")
        >>> print(workspace)
        /tmp/deepagents/thread-123
    """
    workspace = Path(FILE_SYSTEM_BASE) / thread_id
    
    if not workspace.exists():
        logger.warning(f"Workspace not found for thread: {thread_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Thread workspace not found: {thread_id}"
        )
    
    logger.debug(f"Workspace found for thread {thread_id}: {workspace}")
    return workspace


def validate_file_path(workspace: Path, file_path: str) -> Path:
    """
    Validate that a file path is within workspace boundaries.
    
    This function prevents directory traversal attacks by ensuring that
    the resolved absolute path stays within the workspace directory.
    It resolves symlinks and relative path components (../) before checking.
    
    Args:
        workspace: The workspace directory Path object
        file_path: Relative file path within the workspace
        
    Returns:
        Resolved absolute Path object if validation succeeds
        
    Raises:
        HTTPException: 403 if path is outside workspace boundary
        HTTPException: 400 if path is invalid
        
    Security:
        - Prevents directory traversal attacks (../)
        - Resolves symbolic links
        - Validates against workspace boundary
        
    Example:
        >>> workspace = Path("/tmp/deepagents/thread-123")
        >>> validated = validate_file_path(workspace, "data/test.txt")
        >>> # Returns: /tmp/deepagents/thread-123/data/test.txt
        
        >>> validate_file_path(workspace, "../../etc/passwd")
        >>> # Raises: HTTPException 403 (outside boundary)
    """
    try:
        # Resolve absolute path from workspace root
        full_path = (workspace / file_path).resolve()
        
        # Ensure the resolved path is within workspace (prevent directory traversal)
        workspace_resolved = workspace.resolve()
        if not str(full_path).startswith(str(workspace_resolved)):
            logger.warning(
                f"Path traversal attempt detected: {file_path} "
                f"(resolved to {full_path}, workspace: {workspace_resolved})"
            )
            raise HTTPException(
                status_code=403,
                detail="Access denied: Path is outside workspace boundary"
            )
        
        logger.debug(f"Path validated: {file_path} -> {full_path}")
        return full_path
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Invalid path error for {file_path}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid path: {str(e)}"
        )


def detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding (UTF-8 or Latin-1).
    
    Attempts to read the file with UTF-8 encoding first (most common).
    Falls back to Latin-1 if UTF-8 decoding fails. Latin-1 is chosen
    as fallback because it can decode any byte sequence without errors.
    
    Args:
        file_path: Path object pointing to the file
        
    Returns:
        String indicating the detected encoding: 'utf-8' or 'latin-1'
        
    Note:
        Latin-1 (ISO-8859-1) is used as fallback because it maps
        byte values 0-255 directly to Unicode code points 0-255,
        ensuring any byte sequence can be decoded.
        
    Example:
        >>> path = Path("/tmp/file.txt")
        >>> encoding = detect_encoding(path)
        >>> print(encoding)
        'utf-8'
    """
    try:
        # Try UTF-8 first (most common encoding)
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        logger.debug(f"File {file_path.name} detected as UTF-8")
        return 'utf-8'
    except UnicodeDecodeError:
        # Fall back to Latin-1 if UTF-8 fails
        logger.debug(f"File {file_path.name} detected as Latin-1 (UTF-8 failed)")
        return 'latin-1'


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.
    
    Returns basic service status information. Used by monitoring tools
    and Docker health checks to verify the service is running.
    
    Returns:
        Dictionary with status and service name
        
    Example:
        >>> curl http://localhost:8001/health
        {"status": "healthy", "service": "deep-agents-file-server"}
    """
    return {
        "status": "healthy",
        "service": "deep-agents-file-server"
    }


@app.get("/files/{thread_id}/{file_path:path}")
async def read_file_content(
    thread_id: str,
    file_path: str,
    authorization: Optional[str] = Header(None)
) -> JSONResponse:
    """
    Read file content for viewing/copying in the UI.
    
    Retrieves file content along with metadata (size, modified time, encoding).
    The content is returned as JSON, making it suitable for display in the UI
    with syntax highlighting and other text processing features.
    
    Args:
        thread_id: Unique identifier for the thread/session
        file_path: Relative path to the file within the thread workspace
        authorization: Optional authorization header (for future use)
        
    Returns:
        JSONResponse containing:
        - path: Original file path
        - content: File content as string
        - size: File size in bytes
        - modified: Last modified timestamp
        - encoding: Detected file encoding
        
    Raises:
        HTTPException: 404 if workspace or file not found
        HTTPException: 403 if path is outside workspace boundary
        HTTPException: 400 if path is not a file (e.g., directory)
        HTTPException: 500 if file reading fails
        
    Example:
        >>> curl http://localhost:8001/files/thread-123/config.yaml
        {
            "path": "config.yaml",
            "content": "key: value\\n",
            "size": 11,
            "modified": 1234567890.123,
            "encoding": "utf-8"
        }
    """
    try:
        # Get thread workspace and validate path
        workspace = get_thread_workspace(thread_id)
        full_path = validate_file_path(workspace, file_path)
        
        # Check if file exists
        if not full_path.exists():
            logger.warning(f"File not found: {file_path} in thread {thread_id}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Ensure it's a file, not a directory
        if not full_path.is_file():
            logger.warning(f"Path is not a file: {file_path} in thread {thread_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a file: {file_path}"
            )
        
        # Get file metadata
        file_stat = full_path.stat()
        file_size = file_stat.st_size
        modified_time = file_stat.st_mtime
        
        # Detect encoding and read content
        encoding = detect_encoding(full_path)
        
        try:
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read file: {str(e)}"
            )
        
        logger.info(
            f"File read successful: {file_path} "
            f"(thread: {thread_id}, size: {file_size}, encoding: {encoding})"
        )
        
        return JSONResponse({
            "path": file_path,
            "content": content,
            "size": file_size,
            "modified": modified_time,
            "encoding": encoding
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{thread_id}/{file_path:path}/download")
async def download_file(
    thread_id: str,
    file_path: str,
    authorization: Optional[str] = Header(None)
) -> FileResponse:
    """
    Stream file directly for download.
    
    Returns the file as a streaming response with proper Content-Disposition
    headers for browser download. This endpoint is optimized for large files
    as it streams the content without loading it entirely into memory.
    
    The MIME type is automatically detected based on the file extension.
    If detection fails, falls back to 'application/octet-stream'.
    
    Args:
        thread_id: Unique identifier for the thread/session
        file_path: Relative path to the file within the thread workspace
        authorization: Optional authorization header (for future use)
        
    Returns:
        FileResponse that streams the file content
        
    Raises:
        HTTPException: 404 if workspace or file not found
        HTTPException: 403 if path is outside workspace boundary
        HTTPException: 400 if path is not a file
        HTTPException: 500 if download preparation fails
        
    Headers:
        Content-Disposition: attachment; filename="<filename>"
        Content-Type: Detected MIME type or application/octet-stream
        
    Example:
        >>> curl -O http://localhost:8001/files/thread-123/data.csv/download
        # Downloads data.csv with proper filename and MIME type
    """
    try:
        # Get thread workspace and validate path
        workspace = get_thread_workspace(thread_id)
        full_path = validate_file_path(workspace, file_path)
        
        # Check if file exists
        if not full_path.exists():
            logger.warning(f"File not found for download: {file_path} in thread {thread_id}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}"
            )
        
        # Ensure it's a file, not a directory
        if not full_path.is_file():
            logger.warning(f"Path is not a file for download: {file_path} in thread {thread_id}")
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a file: {file_path}"
            )
        
        # Determine MIME type from file extension
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        logger.info(
            f"File download initiated: {file_path} "
            f"(thread: {thread_id}, mime: {mime_type})"
        )
        
        # Return file for download with streaming
        return FileResponse(
            path=str(full_path),
            media_type=mime_type,
            filename=full_path.name,
            headers={
                "Content-Disposition": f'attachment; filename="{full_path.name}"'
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point for running the server directly or as a module
if __name__ == "__main__":
    import uvicorn
    
    # Run the server on all interfaces, port 8001
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

