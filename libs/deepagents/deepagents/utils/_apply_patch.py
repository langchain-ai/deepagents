"""V4A diff parser and applier for the ``apply_patch`` tool.

Ported from OpenAI's reference implementation
(``openai-agents-python/src/agents/apply_diff.py``) and adapted for
Deep Agents conventions.

The module is intentionally backend-agnostic: callers pass a
``file_reader`` callback so the parser never touches the filesystem
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MIN_PATCH_LINES = 2


class PatchError(ValueError):
    """Raised when a V4A patch cannot be parsed or applied."""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    """A contiguous block of deletions and insertions at a known position."""

    orig_index: int
    del_lines: list[str] = field(default_factory=list)
    ins_lines: list[str] = field(default_factory=list)


@dataclass
class _ParserState:
    """Mutable cursor over patch lines."""

    lines: list[str]
    index: int = 0
    fuzz: int = 0


@dataclass
class _SectionResult:
    """Output of `_read_section`: context lines, chunks, and continuation info."""

    context: list[str]
    chunks: list[_Chunk]
    end_index: int
    eof: bool


@dataclass
class _ContextMatch:
    """Result of fuzzy context search."""

    index: int
    fuzz: int


@dataclass(frozen=True)
class PatchResult:
    """Outcome of applying a V4A patch.

    Attributes:
        changes: File changes keyed by source path. ``None`` values indicate
            the file should be deleted. For ``*** Update File`` sections with
            ``*** Move to:``, the key is the source path and the value is the
            post-patch content that should ultimately land at the destination.
        fuzz: Total accumulated fuzz across every hunk. ``0`` means
            every context line matched exactly. Higher values signal
            progressively weaker matches (``1`` per hunk that required
            trailing-whitespace tolerance, ``100`` per hunk that needed
            full whitespace-insensitive matching, plus ``1`` per anchor
            that required strip fallback). A high aggregate fuzz means
            the patch context barely resembles the current file — a
            useful signal that the patch may be stale.
        moves: Rename map for ``*** Update File`` sections that include a
            ``*** Move to:`` directive, mapping source path → destination
            path. Appliers are expected to write ``changes[source]`` to the
            destination and remove the source when a mapping is present.
    """

    changes: dict[str, str | None]
    fuzz: int = 0
    moves: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BEGIN_PATCH = "*** Begin Patch"
_END_PATCH = "*** End Patch"
_END_FILE = "*** End of File"

# File-operation prefixes (with trailing space; path follows).
_ADD_FILE_PREFIX = "*** Add File: "
_DELETE_FILE_PREFIX = "*** Delete File: "
_UPDATE_FILE_PREFIX = "*** Update File: "
_MOVE_TO_PREFIX = "*** Move to: "

# Section start markers: the file-op prefixes without their trailing space,
# used to detect when a new section begins while scanning hunk bodies.
_SECTION_TERMINATORS = (
    _END_PATCH,
    _UPDATE_FILE_PREFIX.rstrip(),
    _DELETE_FILE_PREFIX.rstrip(),
    _ADD_FILE_PREFIX.rstrip(),
)
_END_SECTION_MARKERS = (*_SECTION_TERMINATORS, _END_FILE)

# Prefixes whose path argument must be read from the backend before the
# parser can process the corresponding section.
_FILE_READ_PREFIXES = (_UPDATE_FILE_PREFIX, _DELETE_FILE_PREFIX)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _normalize_line_endings(text: str) -> str:
    r"""Normalize CRLF and lone CR line endings to LF.

    Models and clients often emit patch text (or file content) with
    Windows-style CRLF or, rarely, classic-Mac-style CR terminators.
    The parser splits on ``\n`` exclusively, so any stray ``\r``
    would leak into extracted paths, section markers, and context
    comparisons. Normalizing at the single entry point keeps the
    internal invariants simple: every line is clean.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def apply_patch(  # noqa: C901  # single-dispatch over 4 patch operations; splitting obscures the ported V4A state machine
    patch_text: str,
    *,
    file_reader: Callable[[str], str | None],
) -> PatchResult:
    r"""Parse and apply a V4A patch, returning the resulting file changes.

    Args:
        patch_text: Full V4A patch string (``*** Begin Patch`` ...
            ``*** End Patch``). Any ``\r\n`` or lone ``\r`` line
            endings are normalized to ``\n`` before parsing.
        file_reader: Callback that returns file content for a given
            absolute path, or ``None`` if the file does not exist.
            Called only for ``*** Update File`` and ``*** Delete File``
            operations. The callback receives the path string exactly
            as it appears in the patch; callers are responsible for
            validating the path before touching real storage.

    Returns:
        A :class:`PatchResult` whose ``changes`` map file paths to new
        content (``None`` values mean delete) and whose ``fuzz`` field
        aggregates how far context matching drifted from exactness —
        see :class:`PatchResult` for the scale.

    Raises:
        PatchError: If the patch is malformed or context cannot be matched.
    """
    lines = _normalize_line_endings(patch_text).strip().split("\n")
    if len(lines) < MIN_PATCH_LINES or not lines[0].startswith(_BEGIN_PATCH):
        msg = f"Invalid patch: missing '{_BEGIN_PATCH}' header"
        raise PatchError(msg)
    if lines[-1] != _END_PATCH:
        msg = f"Invalid patch: missing '{_END_PATCH}' footer"
        raise PatchError(msg)

    state = _ParserState(lines=lines, index=1)
    results: dict[str, str | None] = {}
    moves: dict[str, str] = {}

    while not _is_done(state, (_END_PATCH,)):
        # *** Add File
        path = _read_prefix(state, _ADD_FILE_PREFIX)
        if path:
            if path in results:
                msg = f"Add File Error: Duplicate Path: {path}"
                raise PatchError(msg)
            results[path] = _parse_add_file(state)
            continue

        # *** Delete File
        path = _read_prefix(state, _DELETE_FILE_PREFIX)
        if path:
            if path in results:
                msg = f"Delete File Error: Duplicate Path: {path}"
                raise PatchError(msg)
            content = file_reader(path)
            if content is None:
                msg = f"Delete File Error: Missing File: {path}"
                raise PatchError(msg)
            results[path] = None
            continue

        # *** Update File
        path = _read_prefix(state, _UPDATE_FILE_PREFIX)
        if path:
            if path in results:
                msg = f"Update File Error: Duplicate Path: {path}"
                raise PatchError(msg)
            content = file_reader(path)
            if content is None:
                msg = f"Update File Error: Missing File: {path}"
                raise PatchError(msg)
            # Optional *** Move to: <dest> — record the rename intent; the
            # applier is responsible for writing the patched content to the
            # destination and removing the source.
            move_to = _read_prefix(state, _MOVE_TO_PREFIX)
            if move_to:
                moves[path] = move_to
            normalized = _normalize_line_endings(content)
            chunks = _parse_update_file(state, normalized)
            results[path] = _apply_chunks(normalized, chunks)
            continue

        current = state.lines[state.index] if state.index < len(state.lines) else "<EOF>"
        msg = f"Unknown Line: {current}"
        raise PatchError(msg)

    return PatchResult(changes=results, fuzz=state.fuzz, moves=moves)


def list_referenced_files(patch_text: str) -> list[str]:
    """Return paths the patch must read from the backend before applying.

    Performs a lightweight prescan identifying every ``*** Update File:``
    and ``*** Delete File:`` operation so async callers can pre-fetch
    file contents (the main ``apply_patch`` entry point consumes a
    synchronous ``file_reader`` callback and cannot ``await`` mid-parse).

    This intentionally shares the same prefix constants as the main
    parser so the two stay in lock-step if the V4A format evolves. It
    does **not** fully validate the patch structure; callers should
    still invoke :func:`apply_patch` to surface parse errors.

    Args:
        patch_text: Full V4A patch string.

    Returns:
        Paths in the order they appear in the patch. Paths may repeat
        if the patch is malformed; :func:`apply_patch` will reject such
        duplicates when it runs.
    """
    paths: list[str] = []
    for line in _normalize_line_endings(patch_text).split("\n"):
        for prefix in _FILE_READ_PREFIXES:
            if line.startswith(prefix):
                paths.append(line[len(prefix) :])
                break
    return paths


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def _is_done(state: _ParserState, prefixes: Sequence[str]) -> bool:
    if state.index >= len(state.lines):
        return True
    return any(state.lines[state.index].startswith(p) for p in prefixes)


def _read_prefix(state: _ParserState, prefix: str) -> str:
    """If the current line starts with *prefix*, consume it and return the remainder."""
    if state.index >= len(state.lines):
        return ""
    line = state.lines[state.index]
    if line.startswith(prefix):
        state.index += 1
        return line[len(prefix) :]
    return ""


def _parse_add_file(state: _ParserState) -> str:
    """Parse ``*** Add File`` content: all lines must start with ``+``."""
    output: list[str] = []
    while not _is_done(state, _SECTION_TERMINATORS):
        line = state.lines[state.index]
        state.index += 1
        if not line.startswith("+"):
            msg = f"Invalid Add File Line: {line}"
            raise PatchError(msg)
        output.append(line[1:])
    return "\n".join(output)


def _parse_update_file(state: _ParserState, text: str) -> list[_Chunk]:
    """Parse ``*** Update File`` hunks and return positioned chunks."""
    file_lines = text.split("\n")
    chunks: list[_Chunk] = []
    cursor = 0

    while not _is_done(state, _END_SECTION_MARKERS):
        # Handle @@ anchors
        anchor = _read_prefix(state, "@@ ")
        bare_anchor = False
        if not anchor and state.index < len(state.lines) and state.lines[state.index] == "@@":
            bare_anchor = True
            state.index += 1

        if not (anchor or bare_anchor or cursor == 0):
            current = state.lines[state.index] if state.index < len(state.lines) else ""
            msg = f"Invalid Line:\n{current}"
            raise PatchError(msg)

        if anchor.strip():
            cursor = _advance_cursor_to_anchor(anchor, file_lines, cursor, state)

        section = _read_section(state.lines, state.index)
        match = _find_context(file_lines, section.context, cursor, eof=section.eof)
        if match.index == -1:
            ctx = "\n".join(section.context)
            kind = "EOF Context" if section.eof else "Context"
            msg = f"Invalid {kind} {cursor}:\n{ctx}"
            raise PatchError(msg)

        state.fuzz += match.fuzz
        state.index = section.end_index
        cursor = match.index + len(section.context)

        chunks.extend(
            _Chunk(
                orig_index=ch.orig_index + match.index,
                del_lines=list(ch.del_lines),
                ins_lines=list(ch.ins_lines),
            )
            for ch in section.chunks
        )

    # Consume optional *** End of File
    if state.index < len(state.lines) and state.lines[state.index] == _END_FILE:
        state.index += 1

    return chunks


def _advance_cursor_to_anchor(
    anchor: str,
    file_lines: list[str],
    cursor: int,
    state: _ParserState,
) -> int:
    """Jump the cursor forward to the region around *anchor*.

    Positions the cursor so that the subsequent context search
    can find the anchor line and its surroundings.  Returns the
    index of the anchor line itself (not past it), since the
    section's context typically includes the anchor as its first
    context line.
    """
    # Exact match — skip ahead only if not already seen before cursor
    if not any(line == anchor for line in file_lines[:cursor]):
        for i in range(cursor, len(file_lines)):
            if file_lines[i] == anchor:
                return i

    # Fuzzy: strip both sides
    if not any(line.strip() == anchor.strip() for line in file_lines[:cursor]):
        for i in range(cursor, len(file_lines)):
            if file_lines[i].strip() == anchor.strip():
                state.fuzz += 1
                return i

    return cursor


def _read_section(lines: list[str], start: int) -> _SectionResult:  # noqa: C901, PLR0912  # linearly consumes the ported V4A section grammar; splitting by branch obscures the state transitions
    """Read one context+change section until the next terminator."""
    context: list[str] = []
    del_lines: list[str] = []
    ins_lines: list[str] = []
    chunks: list[_Chunk] = []
    mode: Literal["keep", "add", "delete"] = "keep"
    index = start

    while index < len(lines):
        raw = lines[index]
        if raw.startswith(("@@", *_END_SECTION_MARKERS)):
            break
        if raw == "***":
            break
        if raw.startswith("***"):
            msg = f"Invalid Line: {raw}"
            raise PatchError(msg)

        index += 1
        last_mode = mode
        line = raw or " "
        prefix = line[0]

        if prefix == "+":
            mode = "add"
        elif prefix == "-":
            mode = "delete"
        elif prefix == " ":
            mode = "keep"
        else:
            msg = f"Invalid Line: {line}"
            raise PatchError(msg)

        content = line[1:]

        if mode == "keep" and last_mode != mode and (del_lines or ins_lines):
            chunks.append(
                _Chunk(
                    orig_index=len(context) - len(del_lines),
                    del_lines=list(del_lines),
                    ins_lines=list(ins_lines),
                )
            )
            del_lines = []
            ins_lines = []

        if mode == "delete":
            del_lines.append(content)
            context.append(content)
        elif mode == "add":
            ins_lines.append(content)
        else:
            context.append(content)

    if del_lines or ins_lines:
        chunks.append(
            _Chunk(
                orig_index=len(context) - len(del_lines),
                del_lines=list(del_lines),
                ins_lines=list(ins_lines),
            )
        )

    if index == start:
        next_line = lines[index] if index < len(lines) else ""
        msg = f"Nothing in this section - index={index} {next_line}"
        raise PatchError(msg)

    if index < len(lines) and lines[index] == _END_FILE:
        return _SectionResult(context, chunks, index + 1, eof=True)

    return _SectionResult(context, chunks, index, eof=False)


# ---------------------------------------------------------------------------
# Context matching (fuzzy)
# ---------------------------------------------------------------------------


def _find_context(
    lines: list[str],
    context: list[str],
    start: int,
    *,
    eof: bool,
) -> _ContextMatch:
    """Find where *context* lines appear in *lines*, starting at *start*.

    When *eof* is ``True`` the patch claimed ``*** End of File`` and we
    first try to match against the tail of the file. If that fails but
    the same context matches earlier in the file, the patch is almost
    certainly stale — the model thought it was editing the last lines
    but the file now has content after them. Rather than silently
    accept that mismatch (the previous behavior, flagged by a dead
    ``+10000`` fuzz sentinel), we raise so the caller sees it and can
    re-read the file.
    """
    if eof:
        end_start = max(0, len(lines) - len(context))
        match = _find_context_core(lines, context, end_start)
        if match.index != -1:
            return match
        fallback = _find_context_core(lines, context, start)
        if fallback.index == -1:
            return fallback
        msg = (
            "EOF context did not match at end of file, but matched earlier. "
            "The '*** End of File' marker suggests the patch is stale; "
            "re-read the file and regenerate the patch."
        )
        raise PatchError(msg)
    return _find_context_core(lines, context, start)


def _find_context_core(
    lines: list[str],
    context: list[str],
    start: int,
) -> _ContextMatch:
    """Core context search: exact -> rstrip -> strip."""
    if not context:
        return _ContextMatch(index=start, fuzz=0)

    for i in range(start, len(lines)):
        if _slice_matches(lines, context, i, lambda v: v):
            return _ContextMatch(index=i, fuzz=0)

    for i in range(start, len(lines)):
        if _slice_matches(lines, context, i, lambda v: v.rstrip()):
            return _ContextMatch(index=i, fuzz=1)

    for i in range(start, len(lines)):
        if _slice_matches(lines, context, i, lambda v: v.strip()):
            return _ContextMatch(index=i, fuzz=100)

    return _ContextMatch(index=-1, fuzz=0)


def _slice_matches(
    source: list[str],
    target: list[str],
    start: int,
    transform: Callable[[str], str],
) -> bool:
    """Check if *source[start:start+len(target)]* matches *target* after *transform*."""
    if start + len(target) > len(source):
        return False
    return all(transform(source[start + i]) == transform(target[i]) for i in range(len(target)))


# ---------------------------------------------------------------------------
# Chunk application
# ---------------------------------------------------------------------------


def _apply_chunks(text: str, chunks: list[_Chunk]) -> str:
    """Apply positioned chunks to produce the updated file content."""
    orig_lines = text.split("\n")
    dest: list[str] = []
    cursor = 0

    for chunk in chunks:
        if chunk.orig_index > len(orig_lines):
            msg = f"Chunk index {chunk.orig_index} exceeds file length {len(orig_lines)}"
            raise PatchError(msg)
        if cursor > chunk.orig_index:
            msg = f"Overlapping chunk at {chunk.orig_index} (cursor at {cursor})"
            raise PatchError(msg)

        dest.extend(orig_lines[cursor : chunk.orig_index])
        cursor = chunk.orig_index

        if chunk.ins_lines:
            dest.extend(chunk.ins_lines)

        cursor += len(chunk.del_lines)

    dest.extend(orig_lines[cursor:])
    return "\n".join(dest)
