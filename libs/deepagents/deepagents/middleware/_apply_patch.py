"""Opt-in middleware that exposes an `apply_patch` tool using V4A diffs.

V4A is the patch format popularized by OpenAI's `apply_patch` tool: a
single structured payload describes `Add File`, `Update File`, and
`Delete File` operations with context-anchored hunks. Some model
distributions (notably the Codex family) are trained to produce V4A
and underperform when forced into our native `edit_file` flow.

This middleware is deliberately packaged outside `FilesystemMiddleware`:

* `apply_patch` is a model-fit detail, not a core filesystem primitive,
  so core `FilesystemMiddleware` callers shouldn't pay the surface-area
  cost (extra constructor flag, extra tool).
* Hosting it as a standalone middleware lets a `HarnessProfile` opt in
  via `extra_middleware` without touching the filesystem stack itself.

Usage — attach the middleware to a harness profile that targets a
V4A-preferring model, threading the agent's backend through so
`apply_patch` sees the same files as `FilesystemMiddleware`:

```python
from deepagents import HarnessProfile, register_harness_profile
from deepagents.middleware._apply_patch import _ApplyPatchMiddleware


def _apply_patch_factory(backend):
    return [_ApplyPatchMiddleware(backend=backend)]


register_harness_profile(
    "openai:gpt-5.1-codex",
    HarnessProfile(extra_middleware=_apply_patch_factory),
)
```

The class is private (underscore-prefixed) while the opt-in story is
still internal; promote to a public symbol once the integration
surface stabilizes.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Annotated, Any

from langchain.agents.middleware.types import AgentMiddleware, ContextT, ResponseT

# `ToolRuntime` must stay at runtime scope — `StructuredTool.from_function`
# resolves sync/async tool annotations via `typing.get_type_hints`, which
# evaluates string-form annotations against the module globals.
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.tools import StructuredTool

from deepagents.backends import StateBackend
from deepagents.backends.utils import validate_path
from deepagents.middleware.filesystem import FilesystemState
from deepagents.utils._apply_patch import (
    PatchError,
    apply_patch,
    list_referenced_files,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.tools import BaseTool

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

APPLY_PATCH_TOOL_DESCRIPTION = """\
Apply a patch to create, update, delete, or rename files using V4A diff format.

Use for targeted edits to one or more files in a single coherent change. A \
single patch may contain multiple Add/Update/Delete/Move sections. For \
auto-generated changes or bulk search-and-replace, use edit_file or shell \
commands.

The patch must follow V4A format:
- *** Begin Patch / *** End Patch — wrap the entire patch
- *** Add File: <path> — create a new file (lines prefixed with +)
- *** Delete File: <path> — delete an existing file
- *** Update File: <path> — modify a file using context lines and +/- changes
- *** Move to: <path> — optional directive directly under *** Update File to rename the file while applying the patch
- @@ — context anchor (optional text after @@ jumps to that line)
- Lines prefixed with ' ' (space) are context, '+' are additions, '-' are deletions"""

# Upper bound passed as ``limit=`` when the apply_patch helpers need the
# entire file. Backends treat ``limit`` as a line-count cap; ~1M is large
# enough that any realistic source file fits in a single request while
# still guarding against pathological inputs.
_APPLY_PATCH_READ_LIMIT = 999_999

# Fuzz scoring thresholds (see ``PatchResult.fuzz``). A single hunk matched
# via the whitespace-insensitive ``strip()`` fallback contributes 100, so
# anything at or above this level means at least one hunk required that
# aggressive fallback and the result warrants a stronger verification hint
# in the tool output.
_FUZZ_STRIP_FALLBACK_THRESHOLD = 100


def _format_fuzz_note(fuzz: int) -> str:
    """Build a trailing note describing non-zero context-match fuzz.

    Returns an empty string for exact matches so the happy path output
    stays clean. For fuzz >= `_FUZZ_STRIP_FALLBACK_THRESHOLD`, the note
    flags that at least one hunk needed whitespace-insensitive matching
    so the model can verify the result.
    """
    if fuzz <= 0:
        return ""
    if fuzz >= _FUZZ_STRIP_FALLBACK_THRESHOLD:
        return f"\nNote: applied with fuzz={fuzz}; at least one hunk matched via whitespace-insensitive fallback. Verify the patched file is correct."
    return f"\nNote: applied with fuzz={fuzz} (minor whitespace drift)."


_DELETE_UNSUPPORTED_HINT = "the configured backend does not support file deletion"


def _resolve_move_dest(
    raw_dest: str,
    safe_src: str,
) -> tuple[str | None, str | None]:
    """Validate and canonicalize a `*** Move to:` destination.

    Returns a `(safe_dest, error)` pair. A destination that resolves to
    the same path as the source means "no-op rename" and is returned as
    `(None, None)` so callers fall through to an in-place update.
    """
    try:
        safe_dest = validate_path(raw_dest)
    except ValueError as e:
        return None, f"invalid move destination '{raw_dest}': {e}"
    if safe_dest == safe_src:
        return None, None
    return safe_dest, None


def _delete_file(backend: BackendProtocol, safe: str) -> str:
    """Remove a file synchronously and return a model-readable result line."""
    try:
        result = backend.delete(safe)
    except NotImplementedError:
        return f"Error deleting '{safe}': {_DELETE_UNSUPPORTED_HINT}."
    if result.error:
        return f"Error deleting '{safe}': {result.error}"
    return f"Deleted '{safe}'"


async def _adelete_file(backend: BackendProtocol, safe: str) -> str:
    """Async analogue of `_delete_file`."""
    try:
        result = await backend.adelete(safe)
    except NotImplementedError:
        return f"Error deleting '{safe}': {_DELETE_UNSUPPORTED_HINT}."
    if result.error:
        return f"Error deleting '{safe}': {result.error}"
    return f"Deleted '{safe}'"


def _move_file(
    backend: BackendProtocol,
    safe_src: str,
    safe_dest: str,
    content: str,
) -> str:
    """Rename `safe_src` to `safe_dest`, writing `content` at the destination.

    Refuses to overwrite an existing destination — users who intend to
    clobber must emit a separate `*** Delete File:` in the same patch.
    If the destination write succeeds but the source delete fails, the
    message reports both so the caller can reconcile by hand rather than
    assume the rename was atomic.
    """
    dest_probe = backend.read(safe_dest, offset=0, limit=1)
    if not dest_probe.error:
        return f"Error moving '{safe_src}' -> '{safe_dest}': destination already exists."
    write_res = backend.write(safe_dest, content)
    if write_res.error:
        return f"Error writing '{safe_dest}': {write_res.error}"
    try:
        delete_res = backend.delete(safe_src)
    except NotImplementedError:
        return f"Wrote '{safe_dest}' but could not remove '{safe_src}': {_DELETE_UNSUPPORTED_HINT}."
    if delete_res.error:
        return f"Wrote '{safe_dest}' but could not remove '{safe_src}': {delete_res.error}"
    return f"Moved '{safe_src}' -> '{safe_dest}'"


async def _amove_file(
    backend: BackendProtocol,
    safe_src: str,
    safe_dest: str,
    content: str,
) -> str:
    """Async analogue of `_move_file`."""
    dest_probe = await backend.aread(safe_dest, offset=0, limit=1)
    if not dest_probe.error:
        return f"Error moving '{safe_src}' -> '{safe_dest}': destination already exists."
    write_res = await backend.awrite(safe_dest, content)
    if write_res.error:
        return f"Error writing '{safe_dest}': {write_res.error}"
    try:
        delete_res = await backend.adelete(safe_src)
    except NotImplementedError:
        return f"Wrote '{safe_dest}' but could not remove '{safe_src}': {_DELETE_UNSUPPORTED_HINT}."
    if delete_res.error:
        return f"Wrote '{safe_dest}' but could not remove '{safe_src}': {delete_res.error}"
    return f"Moved '{safe_src}' -> '{safe_dest}'"


def _update_or_create(
    backend: BackendProtocol,
    safe: str,
    content: str,
    *,
    existing: str | None = None,
) -> str:
    """Create `safe` with `content`, or edit it in place if it exists.

    When `existing` is provided (the parser already read the file to
    compute the patched content), it is used as the `old_string`
    argument to `backend.edit` directly — this skips the probe +
    full re-read the probe-based branch performs, eliminating two
    redundant backend round-trips per `*** Update File:` operation
    and shrinking the TOCTOU window between the parser's read and
    the edit.

    When `existing` is `None`, the function falls back to the
    original probe-based detection so callers (tests, ad-hoc
    invocations) that do not thread a pre-read cache through still
    work unchanged. That branch is taken for `*** Add File:`
    operations — the parser never reads Add File targets, so the
    applier has to decide create-vs-overwrite against a live read.
    """
    if existing is not None:
        result = backend.edit(safe, existing, content)
        if result.error:
            return f"Error updating '{safe}': {result.error}"
        return f"Updated '{safe}'"
    probe = backend.read(safe, offset=0, limit=1)
    if not probe.error:
        raw = backend.read(safe, offset=0, limit=_APPLY_PATCH_READ_LIMIT)
        old = raw.file_data["content"] if raw.file_data else ""
        result = backend.edit(safe, old, content)
        if result.error:
            return f"Error updating '{safe}': {result.error}"
        return f"Updated '{safe}'"
    result = backend.write(safe, content)
    if result.error:
        return f"Error creating '{safe}': {result.error}"
    return f"Created '{safe}'"


async def _aupdate_or_create(
    backend: BackendProtocol,
    safe: str,
    content: str,
    *,
    existing: str | None = None,
) -> str:
    """Async analogue of `_update_or_create`.

    See `_update_or_create` for the `existing`-shortcut contract.
    """
    if existing is not None:
        result = await backend.aedit(safe, existing, content)
        if result.error:
            return f"Error updating '{safe}': {result.error}"
        return f"Updated '{safe}'"
    probe = await backend.aread(safe, offset=0, limit=1)
    if not probe.error:
        raw = await backend.aread(safe, offset=0, limit=_APPLY_PATCH_READ_LIMIT)
        old = raw.file_data["content"] if raw.file_data else ""
        result = await backend.aedit(safe, old, content)
        if result.error:
            return f"Error updating '{safe}': {result.error}"
        return f"Updated '{safe}'"
    result = await backend.awrite(safe, content)
    if result.error:
        return f"Error creating '{safe}': {result.error}"
    return f"Created '{safe}'"


def _classify_and_validate(
    raw_path: str,
    content: str | None,
    moves: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Validate `raw_path` (and any move destination) up front.

    Returns `(safe_src, safe_dest, error_line)`. `safe_dest` is `None`
    when the entry is not a move (or the move target equals the source).
    `error_line` is populated when validation fails, in which case the
    first two elements are `None` and the caller must skip the entry.
    """
    try:
        safe_src = validate_path(raw_path)
    except ValueError as e:
        return None, None, f"Error: {e}"

    if content is None:
        return safe_src, None, None

    raw_dest = moves.get(raw_path)
    if not raw_dest:
        return safe_src, None, None

    safe_dest, dest_err = _resolve_move_dest(raw_dest, safe_src)
    if dest_err:
        return None, None, f"Error: {dest_err}"
    return safe_src, safe_dest, None


def _apply_file_changes(
    backend: BackendProtocol,
    changes: Mapping[str, str | None],
    moves: Mapping[str, str] | None = None,
    *,
    pre_existing: Mapping[str, str | None] | None = None,
) -> str:
    """Apply parsed V4A patch results to the backend synchronously.

    Every path flows through `validate_path` before any backend I/O.
    This is a second line of defense on top of per-read validation in
    the tool factory: the parser emits entries for `Add File` paths
    that are never fed through a `file_reader`, so validation has to
    happen here independently to keep all backend writes safe.

    Invalid paths are reported per-file without aborting the batch —
    one malformed entry must not prevent the rest of the patch from
    applying. Callers rely on the full error list to diagnose stale
    patches or path typos.

    Args:
        backend: Backend to apply changes against.
        changes: Mapping of raw paths (as written in the patch) to new
            content. `None` values indicate the file should be deleted.
        moves: Optional rename map (source path → destination path) for
            `*** Update File` + `*** Move to:` operations. When a source
            is present, the patched content is written to the destination
            and the source is removed.
        pre_existing: Optional cache mapping raw patch path → content that
            was already read from the backend (typically by the parser's
            `file_reader`). When a `*** Update File:` source appears in
            this cache, its content is passed straight to `backend.edit`
            instead of being re-read. `Add File:` paths are expected to
            be absent from the cache, and fall through to the probe-based
            path. Omitting this argument preserves the legacy probe-based
            behavior for callers that do not thread a cache through.

    Returns:
        One line per entry describing the outcome (created/updated/deleted/
        moved or error), joined by newlines.
    """
    moves = moves or {}
    msgs: list[str] = []
    for raw_path, content in changes.items():
        safe_src, safe_dest, err = _classify_and_validate(raw_path, content, moves)
        if err is not None:
            msgs.append(err)
            continue
        assert safe_src is not None  # noqa: S101  # enforced by _classify_and_validate contract
        if content is None:
            msgs.append(_delete_file(backend, safe_src))
        elif safe_dest is not None:
            msgs.append(_move_file(backend, safe_src, safe_dest, content))
        else:
            cached = pre_existing.get(raw_path) if pre_existing is not None else None
            msgs.append(_update_or_create(backend, safe_src, content, existing=cached))
    return "\n".join(msgs)


async def _aapply_file_changes(
    backend: BackendProtocol,
    changes: Mapping[str, str | None],
    moves: Mapping[str, str] | None = None,
    *,
    pre_existing: Mapping[str, str | None] | None = None,
) -> str:
    """Apply parsed V4A patch results to the backend asynchronously.

    Async analogue of `_apply_file_changes`. Shares the same security
    invariant — every path is validated before any backend I/O — and
    the same per-file error tolerance. `pre_existing` has the same
    shape and semantics as in the sync variant.
    """
    moves = moves or {}
    msgs: list[str] = []
    for raw_path, content in changes.items():
        safe_src, safe_dest, err = _classify_and_validate(raw_path, content, moves)
        if err is not None:
            msgs.append(err)
            continue
        assert safe_src is not None  # noqa: S101  # enforced by _classify_and_validate contract
        if content is None:
            msgs.append(await _adelete_file(backend, safe_src))
        elif safe_dest is not None:
            msgs.append(await _amove_file(backend, safe_src, safe_dest, content))
        else:
            cached = pre_existing.get(raw_path) if pre_existing is not None else None
            msgs.append(await _aupdate_or_create(backend, safe_src, content, existing=cached))
    return "\n".join(msgs)


class _ApplyPatchMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """Expose a V4A `apply_patch` tool backed by a shared filesystem backend.

    Shares `FilesystemState` with `FilesystemMiddleware` so the two
    cooperate cleanly: files written by `apply_patch` are visible to
    `read_file`/`ls`/etc. and vice versa.

    Parameters:
        backend: Filesystem backend to apply patches against. Defaults
            to `StateBackend()` — matches the `FilesystemMiddleware`
            default so the two see the same state out of the box. For
            a non-state backend (local FS, sandbox), pass the same
            instance you wired into `FilesystemMiddleware`.
        custom_description: Override the tool's description. Leave
            `None` to use `APPLY_PATCH_TOOL_DESCRIPTION`.
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        custom_description: str | None = None,
    ) -> None:
        """Initialize the middleware and register its sole `apply_patch` tool.

        Args:
            backend: Filesystem backend, or a factory callable (the
                factory form is deprecated; see `_get_backend`).
                Defaults to `StateBackend()`.
            custom_description: Override for the tool's description
                string. If `None`, `APPLY_PATCH_TOOL_DESCRIPTION` is
                used unchanged.
        """
        super().__init__()
        self.backend: BACKEND_TYPES = backend if backend is not None else StateBackend()
        self._custom_description = custom_description
        self.tools = [self._create_apply_patch_tool()]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Resolve the backend instance from a `ToolRuntime`.

        Mirrors `FilesystemMiddleware._get_backend`: a plain backend
        instance is returned as-is; a backend factory is invoked with
        the runtime. Factory-as-backend is deprecated and slated for
        removal in v0.7.
        """
        if callable(self.backend):
            warnings.warn(
                "Passing a callable (factory) as `backend` is deprecated and "
                "will be removed in v0.7. Pass a `BackendProtocol` instance "
                "directly instead (e.g. `StateBackend()`).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.backend(runtime)  # ty: ignore[call-top-callable]
        return self.backend

    def _create_apply_patch_tool(self) -> BaseTool:  # noqa: C901 # single factory wiring sync+async variants around the shared V4A parser; splitting duplicates the closure over `self`/`backend`
        """Create the `apply_patch` tool (V4A diff format).

        Wires the sync and async variants around the shared V4A parser
        in `deepagents.utils._apply_patch`. The parser itself is
        synchronous and backend-agnostic; the async variant prefetches
        every file the patch references (via `list_referenced_files`)
        concurrently and hands the parser a cache-backed sync reader so
        the parser never has to `await` mid-parse.

        Path-validation happens at every I/O boundary:

        1. Inside `_read_raw` / `_aread_raw` before a backend read.
        2. Inside `_apply_file_changes` / `_aapply_file_changes` before
           a backend write, edit, or delete.

        Both are necessary — `Add File` paths flow only through the
        second layer, so removing either leaves a hole.
        """
        tool_description = self._custom_description or APPLY_PATCH_TOOL_DESCRIPTION

        def _read_raw(backend: BackendProtocol, path: str) -> str | None:
            """Read raw file content (no line numbers) for the parser.

            Returns `None` only when the file is absent or the backend
            errored — an existing but empty file returns `""` so the
            parser can distinguish "missing" from "empty". Collapsing
            both into `None` (the pre-fix behavior) made the parser
            reject legal `Delete File:` / `Update File:` operations on
            empty targets with a bogus "Missing File" error.
            """
            validated = validate_path(path)
            result = backend.read(validated, offset=0, limit=_APPLY_PATCH_READ_LIMIT)
            if result.error or result.file_data is None:
                return None
            return result.file_data["content"]

        async def _aread_raw(backend: BackendProtocol, path: str) -> str | None:
            """Async variant of `_read_raw`.

            Same missing-vs-empty distinction as the sync helper: an
            existing empty file returns `""`, and `None` is reserved
            for backend errors or truly absent files.
            """
            validated = validate_path(path)
            result = await backend.aread(validated, offset=0, limit=_APPLY_PATCH_READ_LIMIT)
            if result.error or result.file_data is None:
                return None
            return result.file_data["content"]

        def sync_apply_patch(
            patch: Annotated[str, "The V4A format patch string to apply."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Apply a V4A patch synchronously.

            Wraps `_read_raw` in a per-call cache so the content the
            parser reads for each `*** Update File:` / `*** Delete File:`
            target can be reused at apply time. Without the cache the
            sync path would re-read every updated file twice more (a
            probe + a full read inside `_update_or_create`) before
            issuing `backend.edit`.
            """
            backend = self._get_backend(runtime)
            read_cache: dict[str, str | None] = {}

            def _caching_reader(p: str) -> str | None:
                if p not in read_cache:
                    read_cache[p] = _read_raw(backend, p)
                return read_cache[p]

            try:
                result = apply_patch(patch, file_reader=_caching_reader)
            except (PatchError, ValueError) as e:
                return f"Error applying patch: {e}"
            applied = _apply_file_changes(
                backend,
                result.changes,
                result.moves,
                pre_existing=read_cache,
            )
            return applied + _format_fuzz_note(result.fuzz)

        async def async_apply_patch(
            patch: Annotated[str, "The V4A format patch string to apply."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Apply a V4A patch asynchronously.

            Prefetches every referenced file concurrently with
            `asyncio.gather` so a multi-file patch doesn't pay the
            latency of N sequential round-trips to the backend.
            """
            backend = self._get_backend(runtime)

            # Prescan paths, validate up front so a traversal attempt
            # short-circuits before any I/O fires.
            needed = list_referenced_files(patch)
            for p in needed:
                try:
                    validate_path(p)
                except ValueError as e:
                    return f"Error applying patch: {e}"

            # Fan-out reads concurrently. Each entry resolves to
            # `str | None` where `None` signals "not found" — matching
            # the parser's `file_reader` contract.
            contents = await asyncio.gather(*(_aread_raw(backend, p) for p in needed))
            file_cache: dict[str, str | None] = dict(zip(needed, contents, strict=True))

            try:
                result = apply_patch(patch, file_reader=file_cache.get)
            except (PatchError, ValueError) as e:
                return f"Error applying patch: {e}"

            applied = await _aapply_file_changes(
                backend,
                result.changes,
                result.moves,
                pre_existing=file_cache,
            )
            return applied + _format_fuzz_note(result.fuzz)

        return StructuredTool.from_function(
            name="apply_patch",
            description=tool_description,
            func=sync_apply_patch,
            coroutine=async_apply_patch,
        )
