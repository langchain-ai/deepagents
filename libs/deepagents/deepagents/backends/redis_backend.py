"""Redis-backed persistent file storage for DeepAgents.

Implements :class:`~deepagents.backends.protocol.BackendProtocol` using a
Redis Hash as the storage engine.  All files for a given *namespace* are
stored under a single Hash key ``{namespace}:files`` where each field is an
absolute file path and the value is the UTF-8-encoded file content.

Advantages over the default :class:`~deepagents.backends.state.StateBackend`:

* **Persistence** -- files survive process restarts and container re-deploys.
* **Multi-session sharing** -- multiple agent threads can share the same
  namespace, enabling collaborative or long-running workflows.
* **TTL support** -- the entire namespace can expire automatically via Redis
  ``EXPIRE``, simplifying session lifecycle management.

Example::

    from deepagents.backends import RedisBackend
    from deepagents import create_deep_agent

    backend = RedisBackend(
        host="localhost",
        port=6379,
        namespace="my-agent-session",
        ttl_seconds=3600,
    )
    agent = create_deep_agent(backend=backend)
"""

from __future__ import annotations

import fnmatch
import re

import redis

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileInfo,
    GrepMatch,
    WriteResult,
)


class RedisBackend(BackendProtocol):
    """Persistent file-system backend backed by Redis Hash.

    Files are stored in a single Redis Hash:

    * **Key** ``{namespace}:files``
    * **Field** absolute file path, e.g. ``/workspace/notes.txt``
    * **Value** UTF-8-encoded file content

    Args:
        host: Redis server hostname.  Defaults to ``"localhost"``.
        port: Redis server port.  Defaults to ``6379``.
        password: Optional Redis AUTH password.
        namespace: Logical namespace for this backend instance.
            Multiple agents can share files by using the same namespace.
            Defaults to ``"deepagents"``.
        db: Redis database index.  Defaults to ``0``.
        ttl_seconds: If set, the Hash key is given a Redis ``EXPIRE`` TTL
            after every write.  Useful for automatic session cleanup.

    Example::

        backend = RedisBackend(namespace="project-x", ttl_seconds=86400)
        result = backend.write("/workspace/plan.md", "# Plan")
        print(backend.read("/workspace/plan.md"))
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        namespace: str = "deepagents",
        db: int = 0,
        ttl_seconds: int | None = None,
    ) -> None:
        """Initialize the backend and establish a Redis connection.

        Args:
            host: Redis server hostname.
            port: Redis server port.
            password: Optional AUTH password.
            namespace: Logical namespace; controls the Redis Hash key.
            db: Redis database index.
            ttl_seconds: Optional TTL applied to the Hash after each write.
        """
        self._client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=False,
        )
        self._hash_key = f"{namespace}:files"
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, path: str) -> str:
        """Normalize a path to an absolute, slash-terminated-free form.

        Args:
            path: Raw path string, possibly without a leading slash.

        Returns:
            Normalized path that starts with ``/`` and has no trailing slash
            (unless it is the root ``/`` itself).
        """
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"

    def _add_line_numbers(self, content: str, offset: int, limit: int) -> str:
        """Slice *content* and prefix each line with a 1-based line number.

        Args:
            content: Full file content as a single string.
            offset: Zero-based index of the first line to include.
            limit: Maximum number of lines to return.

        Returns:
            Newline-joined string where each entry is the 1-based line number,
            a tab character, then the line content.
        """
        lines = content.splitlines()
        sliced = lines[offset : offset + limit]
        return "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(sliced))

    # ------------------------------------------------------------------
    # BackendProtocol implementation
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
        """List direct children of *path* (non-recursive).

        Args:
            path: Absolute path to the directory to list.

        Returns:
            List of :class:`~deepagents.backends.protocol.FileInfo` dicts.
            Immediate sub-files are listed individually; deeper paths are
            collapsed to a single directory entry with a trailing ``/``.
        """
        path = self._normalize(path)
        prefix = path if path == "/" else path + "/"
        all_keys: list[bytes] = self._client.hkeys(self._hash_key)
        results: list[FileInfo] = []
        seen_dirs: set[str] = set()

        for raw_key in all_keys:
            key = raw_key.decode("utf-8")
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            parts = remainder.split("/")
            if len(parts) == 1:
                results.append(FileInfo(path=key, is_dir=False))
            else:
                subdir = prefix + parts[0] + "/"
                if subdir not in seen_dirs:
                    seen_dirs.add(subdir)
                    results.append(FileInfo(path=subdir, is_dir=True))

        return results

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read a file and return its content with line numbers.

        Args:
            file_path: Absolute path to the file.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            Content where each line is prefixed with its 1-based line number
            and a tab character, or an error string prefixed with ``"Error:"``
            when the file is missing.
        """
        file_path = self._normalize(file_path)
        raw = self._client.hget(self._hash_key, file_path)
        if raw is None:
            return f"Error: file not found: {file_path}"
        content = raw.decode("utf-8")
        return self._add_line_numbers(content, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write *content* to a new file, failing if the file already exists.

        Args:
            file_path: Absolute path where the file should be created.
            content: UTF-8 text content to store.

        Returns:
            :class:`~deepagents.backends.protocol.WriteResult` with
            ``files_update=None`` (content is persisted externally in Redis,
            not in LangGraph state).
        """
        file_path = self._normalize(file_path)
        if self._client.hexists(self._hash_key, file_path):
            return WriteResult(
                error=f"File already exists: {file_path}. Use edit() to modify.",
            )
        self._client.hset(self._hash_key, file_path, content.encode("utf-8"))
        if self._ttl:
            self._client.expire(self._hash_key, self._ttl)
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace occurrences of *old_string* with *new_string* in a file.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: Exact string to search for.
            new_string: Replacement string.
            replace_all: When ``True`` replace every occurrence; when
                ``False`` (default) the edit fails if more than one occurrence
                is found.

        Returns:
            :class:`~deepagents.backends.protocol.EditResult` with the number
            of replacements made, or an error description on failure.
        """
        file_path = self._normalize(file_path)
        raw = self._client.hget(self._hash_key, file_path)
        if raw is None:
            return EditResult(error=f"File not found: {file_path}")
        content = raw.decode("utf-8")
        count = content.count(old_string)
        if count == 0:
            return EditResult(error=f"String not found in {file_path}")
        if count > 1 and not replace_all:
            return EditResult(
                error=f"Found {count} occurrences. Pass replace_all=True to replace all.",
            )
        new_content = content.replace(old_string, new_string)
        self._client.hset(self._hash_key, file_path, new_content.encode("utf-8"))
        return EditResult(path=file_path, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,  # noqa: ARG002
    ) -> list[GrepMatch] | str:
        """Search files for lines matching a regex *pattern*.

        Args:
            pattern: Regular-expression pattern to search for.
            path: Optional directory prefix to restrict the search.
                When ``None`` all files in the namespace are searched.
            glob: Reserved for future glob-based file filtering; currently
                ignored.

        Returns:
            A list of :class:`~deepagents.backends.protocol.GrepMatch` dicts
            on success, or an error string if *pattern* is invalid.
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        prefix = self._normalize(path) if path else "/"
        all_items = self._client.hgetall(self._hash_key)
        matches: list[GrepMatch] = []

        for raw_key, raw_val in all_items.items():
            key = raw_key.decode("utf-8")
            if prefix != "/" and not key.startswith(prefix):
                continue
            content = raw_val.decode("utf-8")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(GrepMatch(path=key, line=line_no, text=line))
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:  # noqa: ARG002
        """Return files whose paths match *pattern*.

        Uses :mod:`fnmatch` glob syntax (``*``, ``**``, ``?``, ``[seq]``).

        Args:
            pattern: Glob pattern matched against absolute file paths.
            path: Unused; present for protocol compatibility.  Pass
                directory-scoped patterns (e.g. ``/workspace/*.py``) to
                restrict results.

        Returns:
            List of :class:`~deepagents.backends.protocol.FileInfo` dicts for
            matching files.
        """
        all_keys: list[bytes] = self._client.hkeys(self._hash_key)
        return [
            FileInfo(path=key, is_dir=False)
            for raw_key in all_keys
            if fnmatch.fnmatch(key := raw_key.decode("utf-8"), pattern)
        ]

    # ------------------------------------------------------------------
    # Async variants (delegate to sync; swap for redis.asyncio in prod)
    # ------------------------------------------------------------------

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async variant of :meth:`ls_info`."""
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Async variant of :meth:`read`."""
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async variant of :meth:`write`."""
        return self.write(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Async variant of :meth:`edit`."""
        return self.edit(file_path, old_string, new_string, replace_all=replace_all)

    async def agrep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        """Async variant of :meth:`grep_raw`."""
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async variant of :meth:`glob_info`."""
        return self.glob_info(pattern, path)
