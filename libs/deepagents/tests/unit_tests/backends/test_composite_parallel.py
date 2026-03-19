import asyncio
import time

import pytest

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
)


class SlowBackend:
    """A backend that sleeps to simulate network/IO delay."""

    def __init__(self, delay: float = 0.1) -> None:
        self.delay = delay

    async def aglob(self, pattern, path="/"):
        await asyncio.sleep(self.delay)
        return GlobResult(matches=[{"path": "/file.txt", "is_dir": False, "size": 10, "modified_at": "now"}])

    async def agrep(self, pattern, path=None, glob=None):
        await asyncio.sleep(self.delay)
        return GrepResult(matches=[{"path": "/file.txt", "line": 1, "text": "match"}])

    async def aupload_files(self, files):
        await asyncio.sleep(self.delay)
        return [FileUploadResponse(path=f[0], error=None) for f in files]

    async def adownload_files(self, paths):
        await asyncio.sleep(self.delay)
        return [FileDownloadResponse(path=p, content=b"content", error=None) for p in paths]

    async def als(self, path):
        return []


@pytest.mark.asyncio
async def test_aglob_parallel():
    # Use two slow backends, each with 0.2s delay
    backend1 = SlowBackend(delay=0.2)
    backend2 = SlowBackend(delay=0.2)
    # Configure composite with one route
    # Note: aglob at "/" searches default + all routes
    composite = CompositeBackend(default=backend1, routes={"/r/": backend2})

    start_time = time.perf_counter()
    results = await composite.aglob("*.txt", "/")
    end_time = time.perf_counter()

    duration = end_time - start_time
    # If sequential, it would be >= 0.4s. If parallel, it should be ~0.2s.
    assert duration < 0.35, f"duration {duration} suggests sequential execution"
    assert len(results.matches) == 2


@pytest.mark.asyncio
async def test_agrep_parallel():
    backend1 = SlowBackend(delay=0.2)
    backend2 = SlowBackend(delay=0.2)
    composite = CompositeBackend(default=backend1, routes={"/r/": backend2})

    start_time = time.perf_counter()
    results = await composite.agrep("test", path="/")
    end_time = time.perf_counter()

    duration = end_time - start_time
    assert duration < 0.35, f"duration {duration} suggests sequential execution"
    assert len(results.matches) == 2


@pytest.mark.asyncio
async def test_aupload_files_parallel():
    backend1 = SlowBackend(delay=0.2)
    backend2 = SlowBackend(delay=0.2)
    # Default is backend1, /r2/ is backend2
    composite = CompositeBackend(default=backend1, routes={"/r2/": backend2})

    # Upload to different backends
    files = [("/f1.txt", b"c1"), ("/r2/f2.txt", b"c2")]

    start_time = time.perf_counter()
    results = await composite.aupload_files(files)
    end_time = time.perf_counter()

    duration = end_time - start_time
    assert duration < 0.35, f"duration {duration} suggests sequential execution"
    assert len(results) == 2


@pytest.mark.asyncio
async def test_adownload_files_parallel():
    backend1 = SlowBackend(delay=0.2)
    backend2 = SlowBackend(delay=0.2)
    composite = CompositeBackend(default=backend1, routes={"/r2/": backend2})

    paths = ["/f1.txt", "/r2/f2.txt"]

    start_time = time.perf_counter()
    results = await composite.adownload_files(paths)
    end_time = time.perf_counter()

    duration = end_time - start_time
    assert duration < 0.35, f"duration {duration} suggests sequential execution"
    assert len(results) == 2
