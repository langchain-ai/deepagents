"""Strict, backend-preserving executor for native patch operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Never

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import _permission_error

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from deepagents.middleware.filesystem import FilesystemPermission


def _fail(message: str) -> Never:
    raise ValueError(message)


def _read_hunk(patch: list[str], index: int) -> tuple[list[str], list[str], bool, int]:
    before: list[str] = []
    after: list[str] = []
    at_end = False
    while index < len(patch) and patch[index] != "@@" and not patch[index].startswith("@@ "):
        line = patch[index]
        index += 1
        if line == "*** End of File":
            at_end = True
            break
        if not line or line[0] not in " +-":
            _fail("Invalid patch line prefix")
        if line[0] != "+":
            before.append(line[1:])
        if line[0] != "-":
            after.append(line[1:])
    if not before and not after:
        _fail("Empty patch hunk")
    return before, after, at_end, index


def _locate_hunk(lines: list[str], before: list[str], cursor: int, *, at_end: bool) -> int:
    if not before:
        if lines:
            _fail("An insertion into a nonempty file requires context")
        return cursor
    candidates = [pos for pos in range(cursor, len(lines) - len(before) + 1) if lines[pos : pos + len(before)] == before]
    if at_end:
        candidates = [pos for pos in candidates if pos + len(before) == len(lines)]
    if len(candidates) != 1:
        _fail("Patch context is missing or ambiguous")
    return candidates[0]


def _apply_hunks(source: str, diff: str) -> str:
    lines = source.splitlines()
    trailing = source.endswith("\n")
    result: list[str] = []
    cursor = 0
    patch = diff.splitlines()
    index = 0
    if not patch:
        _fail("An update requires at least one hunk")
    while index < len(patch):
        header = patch[index]
        if header != "@@" and not header.startswith("@@ "):
            _fail("Expected a patch hunk beginning with @@")
        if header.startswith("@@ "):
            matches = [pos for pos in range(cursor, len(lines)) if lines[pos] == header[3:]]
            if len(matches) != 1:
                _fail("Patch anchor is missing or ambiguous")
            result.extend(lines[cursor : matches[0] + 1])
            cursor = matches[0] + 1
        before, after, at_end, index = _read_hunk(patch, index + 1)
        position = _locate_hunk(lines, before, cursor, at_end=at_end)
        result.extend(lines[cursor:position])
        result.extend(after)
        cursor = position + len(before)
    result.extend(lines[cursor:])
    return "\n".join(result) + ("\n" if result and (trailing or not source) else "")


def _created_content(diff: str) -> str:
    lines = diff.splitlines()
    if any(not line.startswith("+") for line in lines):
        _fail("A create_file diff must contain only added lines")
    return "\n".join(line[1:] for line in lines) + ("\n" if lines else "")


def _validate_operation(operation: object) -> tuple[str, str, str | None]:
    if not isinstance(operation, Mapping):
        _fail("operation must be an object")
    kind = operation.get("type")
    path = operation.get("path")
    diff = operation.get("diff")
    if not isinstance(kind, str) or kind not in {"create_file", "update_file", "delete_file"} or not isinstance(path, str) or not path.strip():
        _fail("Expected create_file, update_file, or delete_file with a nonempty path")
    if diff is not None and not isinstance(diff, str):
        _fail("diff must be a string")
    if kind != "delete_file" and diff is None:
        _fail("Create and update operations require a diff string")
    if set(operation) - {"type", "path", "diff"}:
        _fail("Unexpected patch operation fields")
    validate_path(path)
    return kind, path, diff


def _normalize_patch_path(backend: BackendProtocol | None, path: str) -> str:
    normalized = validate_path(path)
    target = backend
    if isinstance(backend, CompositeBackend):
        if any(normalized == route.rstrip("/") or normalized.startswith(route) for route in backend.routes):
            return normalized
        target = backend.default
    if isinstance(target, FilesystemBackend) and not target.virtual_mode and not path.startswith("/"):
        return validate_path(str(target.cwd / normalized.lstrip("/")))
    return normalized


def execute_patch(backend: BackendProtocol, operation: object, permissions: list[FilesystemPermission]) -> str:
    """Validate an entire operation before mutating the backend."""
    kind, path, diff = _validate_operation(operation)
    path = _normalize_patch_path(backend, path)
    for access in ("read", "write"):
        access: Literal["read", "write"]
        if error := _permission_error(permissions, access, backend, path):
            _fail(error)
    downloaded = backend.download_files([path])[0]
    if kind == "create_file":
        if downloaded.error != "file_not_found":
            _fail("Cannot create an existing or unreadable file")
        content = _created_content(diff or "")
    else:
        if downloaded.error or downloaded.content is None:
            _fail(f"Cannot access patch target: {downloaded.error}")
        source = downloaded.content.decode("utf-8")
        content = _apply_hunks(source, diff or "") if kind == "update_file" else ""
    response = backend.delete(path) if kind == "delete_file" else backend.write(path, content)
    if response.error:
        _fail(response.error)
    return f"{kind}: {path}"
