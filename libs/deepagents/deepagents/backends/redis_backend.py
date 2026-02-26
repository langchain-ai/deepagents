from __future__ import annotations

import re
from typing import Optional

import redis

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileInfo,
    GrepMatch,
    WriteResult,
)


class RedisBackend(BackendProtocol):
    """
    BackendProtocol 实现：将虚拟文件系统持久化到 Redis Hash。

    Redis 数据结构：
      - Hash key:  "{namespace}:files"
        field: 文件绝对路径 (e.g. "/workspace/notes.txt")
        value: 文件内容 (UTF-8 字节)

    优点：
      - Agent 重启后文件内容不丢失
      - 支持多会话共享同一命名空间
      - 原生支持 TTL（通过 EXPIRE）
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        namespace: str = "deepagents",
        db: int = 0,
        ttl_seconds: Optional[int] = None,
    ) -> None:
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
    # 内部工具
    # ------------------------------------------------------------------

    def _normalize(self, path: str) -> str:
        """确保路径以 / 开头，去除末尾斜杠"""
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip("/") or "/"

    def _add_line_numbers(self, content: str, offset: int, limit: int) -> str:
        lines = content.splitlines()
        sliced = lines[offset : offset + limit]
        return "\n".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(sliced))

    # ------------------------------------------------------------------
    # BackendProtocol 必须实现的方法
    # ------------------------------------------------------------------

    def ls_info(self, path: str) -> list[FileInfo]:
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
                # 直接子文件
                results.append(FileInfo(path=key, is_dir=False))
            else:
                # 直接子目录
                subdir = prefix + parts[0] + "/"
                if subdir not in seen_dirs:
                    seen_dirs.add(subdir)
                    results.append(FileInfo(path=subdir, is_dir=True))

        return results

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        file_path = self._normalize(file_path)
        raw = self._client.hget(self._hash_key, file_path)
        if raw is None:
            return f"Error: file not found: {file_path}"
        content = raw.decode("utf-8")
        return self._add_line_numbers(content, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        file_path = self._normalize(file_path)
        if self._client.hexists(self._hash_key, file_path):
            return WriteResult(error=f"File already exists: {file_path}. Use edit() to modify.")
        self._client.hset(self._hash_key, file_path, content.encode("utf-8"))
        if self._ttl:
            self._client.expire(self._hash_key, self._ttl)
        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
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
                error=f"Found {count} occurrences. Pass replace_all=True to replace all."
            )
        new_content = content.replace(old_string, new_string)
        self._client.hset(self._hash_key, file_path, new_content.encode("utf-8"))
        return EditResult(path=file_path, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
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
                    matches.append(
                        GrepMatch(path=key, line=line_no, text=line)
                    )
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        import fnmatch
        all_keys: list[bytes] = self._client.hkeys(self._hash_key)
        results: list[FileInfo] = []
        for raw_key in all_keys:
            key = raw_key.decode("utf-8")
            if fnmatch.fnmatch(key, pattern):
                results.append(FileInfo(path=key, is_dir=False))
        return results

    # ------------------------------------------------------------------
    # 异步变体（委托同步方法，生产环境建议换 redis.asyncio.Redis）
    # ------------------------------------------------------------------

    async def als_info(self, path: str) -> list[FileInfo]:
        return self.ls_info(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def agrep_raw(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None) -> list[GrepMatch] | str:
        return self.grep_raw(pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)
