"""Base sandbox implementation.

[`BaseSandbox`][deepagents.backends.sandbox.BaseSandbox] implements
[`SandboxBackendProtocol`][deepagents.backends.protocol.SandboxBackendProtocol].

File listing, grep, glob, and read use shell commands via `execute()`. Write
delegates content transfer to `upload_files()`. Edit uses server-side `execute()`
for payloads under `_EDIT_INLINE_MAX_BYTES` and falls back to uploading old/new
strings as temp files with a server-side replace script for larger ones.

Concrete subclasses implement `execute()` and `upload_files()`; all other
operations are derived from those.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
from abc import ABC, abstractmethod
from typing import Final

from deepagents.backends.protocol import (
    ASYNC_GREP_TIMEOUT,
    DeleteResult,
    EditResult,
    ExecuteOffloadResult,
    ExecuteResponse,
    FileData,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    execute_accepts_timeout,
)
from deepagents.backends.utils import _get_backend_read_file_type

logger = logging.getLogger(__name__)

_GLOB_COMMAND_TEMPLATE = """python3 -c "
import glob
import os
import json
import base64

# Decode base64-encoded parameters
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

try:
    real_root = os.path.realpath(path)
    os.chdir(path)
    rel_pattern = pattern.lstrip('/')
    if any(seg == '..' for seg in rel_pattern.replace(chr(92), '/').split('/')):
        print(json.dumps({{'error': 'invalid_pattern'}}))
    else:
        matches = sorted(glob.glob(rel_pattern, recursive=True))
        for m in matches:
            candidate = os.path.realpath(m)
            if candidate != real_root and not candidate.startswith(real_root + os.sep):
                continue
            try:
                st = os.stat(candidate)
            except OSError:
                continue
            print(json.dumps({{
                'path': m,
                'size': st.st_size,
                'mtime': st.st_mtime,
                'is_dir': os.path.isdir(candidate),
            }}))
except FileNotFoundError:
    print(json.dumps({{'error': 'path_not_found'}}))
except NotADirectoryError:
    print(json.dumps({{'error': 'not_a_directory'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>&1"""
"""Find files matching a pattern with metadata.

Uses base64-encoded parameters to avoid shell escaping issues.
"""


_GREP_PATH_GLOB_TEMPLATE = """python3 -c "
import glob, os, base64, sys

search_path = base64.b64decode('{path_b64}').decode('utf-8')
glob_pat = base64.b64decode('{glob_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')
max_count = {max_count}
match_count = 0

# When the search path is a directory, chdir to it so glob patterns
# resolve relative to it. When it is a single file, search it directly
# (glob filtering is irrelevant for a single-file search).
if os.path.isdir(search_path):
    os.chdir(search_path)
    # A leading `/` would make `glob.glob` treat the pattern as an
    # absolute filesystem path, searching outside the search root (e.g.
    # `/*.py` after `chdir('/workspace')` would match `/top.py` on
    # the host, not `/workspace/top.py`). Strip it so anchored globs
    # stay relative to the search root, matching the `FilesystemBackend`
    # semantics where `/` anchors to the root, not the filesystem.
    rel_glob = glob_pat.lstrip('/')
    if any(seg == '..' for seg in rel_glob.replace(chr(92), '/').split('/')):
        sys.stderr.write('glob contains path traversal\\n')
        sys.exit(2)
    real_root = os.path.realpath(search_path)
    rel_files = sorted(glob.glob(rel_glob, recursive=True))
    # Open the glob-relative path (cwd is the search root) but report the
    # path prefixed with the search root, so GrepResult.path matches the
    # `<root>/<match>` form that `grep -r` emits on the --include route.
    targets = []
    for rel in rel_files:
        real_open = os.path.realpath(rel)
        if real_open != real_root and not real_open.startswith(real_root + os.sep):
            continue
        display_path = os.path.join(search_path, os.path.relpath(real_open, real_root))
        targets.append((real_open, display_path))
else:
    targets = [(search_path, search_path)]

for open_path, display_path in targets:
    try:
        with open(open_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh, 1):
                if pattern in line:
                    # GNU grep -HnFZ always terminates each record with a
                    # newline, even when the matched line has none. Strip
                    # the line's own trailing newline and add an explicit
                    # one so records never concatenate when a file's last
                    # line lacks a final newline.
                    sys.stdout.write(display_path + chr(0) + str(i) + ':' + line.rstrip(chr(10)) + chr(10))
                    match_count += 1
                    if max_count is not None and match_count > max_count:
                        sys.exit(0)
    except OSError:
        pass
" 2>/dev/null"""
"""Search file contents for a literal string, filtered by a path-relative glob.

Used when the glob pattern contains a `/` (e.g. `src/**/*.py`), because
GNU `grep --include` only matches basenames and would silently return zero
results for such patterns. All three parameters are base64-encoded to avoid
shell escaping issues.

Emits the same `path\0line_num:text` record structure that `grep -HnFZ`
produces — each match path is prefixed with the search root to mirror
grep's output — so `_parse_grep_output` consumes it unchanged. Unlike the
`grep -r` route, results are sorted, hidden files and directories are
skipped (Python `glob` semantics), and file contents are decoded as UTF-8
with `errors='ignore'` rather than matched byte-for-byte.

`stderr` is discarded, but `|| true` is deliberately omitted: the script
exits 0 on a legitimate no-match, so a non-zero exit signals a genuine
failure (bad base64, an inaccessible search root) that `_parse_grep_output`
surfaces as an error instead of a silent empty result.
"""


_WRITE_CHECK_TEMPLATE = """python3 -c "
import os, base64

path = base64.b64decode('{path_b64}').decode('utf-8')
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
" 2>&1"""
"""Preflight for write operations: create parent directories for the target path if it doesn't exist.

Only the (small) base64-encoded path is interpolated — file content is
transferred separately via `upload_files()`.
"""

MAX_BINARY_BYTES: Final = 500 * 1024
"""Maximum size of a binary file returned by `read()` as base64.

Files exceeding this size return a `Binary file exceeds maximum preview size`
error rather than being base64-encoded in full. Backends overriding `read()`
should import and reuse this constant to stay in sync with the base
implementation. Kept in lockstep with the `MAX_BINARY_BYTES` literal in
`_READ_COMMAND_TEMPLATE` (asserted by `test_read_constants_match_template`).
"""

MAX_OUTPUT_BYTES: Final = 500 * 1024
"""Maximum size of rendered text content returned by `read()`.

Pages exceeding this cap are truncated and `TRUNCATION_MSG` is appended.
Mirrors the `MAX_OUTPUT_BYTES` literal in `_READ_COMMAND_TEMPLATE`.
"""

TRUNCATION_MSG: Final = (
    "\n\n[Output was truncated due to size limits. "
    "This paginated read result exceeded the sandbox stdout limit. "
    "Continue reading with a larger offset or smaller limit to inspect the rest of the file.]"
)
"""Sentinel appended to `read()` content when `MAX_OUTPUT_BYTES` is hit."""

_EDIT_COMMAND_TEMPLATE = """python3 -c "
import sys, os, stat as _stat, base64, json

payload = json.loads(base64.b64decode(sys.stdin.read().strip()).decode('utf-8'))
path, old, new = payload['path'], payload['old'], payload['new']
replace_all = payload.get('replace_all', False)

try:
    st = os.stat(path)
    if not _stat.S_ISREG(st.st_mode):
        print(json.dumps({{'error': 'not_a_file'}}))
        sys.exit(0)

    with open(path, 'rb') as f:
        raw = f.read()

    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        print(json.dumps({{'error': 'not_a_text_file'}}))
        sys.exit(0)

    # Match-driven CRLF handling (issue #2880): the read template normalizes
    # CRLF to LF for the LLM, so old_string arrives LF-only even when the
    # file on disk is CRLF. Try old as sent, then a CRLF variant, then an LF
    # variant. The first match reveals the file line-ending style in that
    # region; apply the same transform to new so the file style is preserved.
    old_crlf = old.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')
    old_lf = old.replace('\\r\\n', '\\n')
    new_crlf = new.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')
    new_lf = new.replace('\\r\\n', '\\n')
    count = 0
    matched_old, matched_new = old, new
    for cand_old, cand_new in ((old, new), (old_crlf, new_crlf), (old_lf, new_lf)):
        c = text.count(cand_old)
        if c >= 1:
            matched_old, matched_new, count = cand_old, cand_new, c
            break

    if count == 0:
        print(json.dumps({{'error': 'string_not_found'}}))
        sys.exit(0)
    if count > 1 and not replace_all:
        print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
        sys.exit(0)

    result = text.replace(matched_old, matched_new) if replace_all else text.replace(matched_old, matched_new, 1)
    with open(path, 'wb') as f:
        f.write(result.encode('utf-8'))

    print(json.dumps({{'count': count}}))
except FileNotFoundError:
    print(json.dumps({{'error': 'file_not_found'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>&1 <<'__DEEPAGENTS_EDIT_EOF__'
{payload_b64}
__DEEPAGENTS_EDIT_EOF__
"""
# Make sure to maintain a new line at the end of DEEPAGENTS_EDIT_EOF to denote end of
# feed. This may not matter for some integrations.

"""Server-side file edit via `execute()`.

Reads the file, performs string replacement, and writes back — all on the
sandbox. The payload (path, old/new strings, `replace_all` flag) is passed as
base64-encoded JSON via heredoc stdin to avoid shell escaping issues.

Output: single-line JSON with `{{"count": N}}` on success or `{{"error": ...}}`
on failure.

Used for payloads under `_EDIT_INLINE_MAX_BYTES`; larger payloads fall back
to `_edit_via_upload()` which transfers old/new strings as temp files.

Keeps a trailing newline after `__DEEPAGENTS_EDIT_EOF__` so integrations that
detect end-of-input on a newline-delimited heredoc feed can observe completion.
"""

_EDIT_INLINE_MAX_BYTES: Final = 50_000
"""Maximum combined byte size of `old_string` + `new_string` for inline server-side edit.

Payloads above this use _edit_via_upload (temp file upload + server-side replace)
to avoid size limits on the execute() request body imposed by some sandbox providers.
"""

_EDIT_TMPFILE_TEMPLATE = """python3 -c "
import os, stat as _stat, sys, json, base64

old_path = base64.b64decode('{old_path_b64}').decode('utf-8')
new_path = base64.b64decode('{new_path_b64}').decode('utf-8')
target = base64.b64decode('{target_b64}').decode('utf-8')
replace_all = {replace_all}

try:
    old = open(old_path, 'rb').read().decode('utf-8')
    new = open(new_path, 'rb').read().decode('utf-8')
except Exception as e:
    print(json.dumps({{'error': 'temp_read_failed', 'detail': str(e)}}))
    sys.exit(0)
finally:
    for p in (old_path, new_path):
        try: os.remove(p)
        except OSError: pass

try:
    st = os.stat(target)
    if not _stat.S_ISREG(st.st_mode):
        print(json.dumps({{'error': 'not_a_file'}}))
        sys.exit(0)

    with open(target, 'rb') as f:
        raw = f.read()

    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        print(json.dumps({{'error': 'not_a_text_file'}}))
        sys.exit(0)

    # Match-driven CRLF handling -- see _EDIT_COMMAND_TEMPLATE and issue #2880.
    old_crlf = old.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')
    old_lf = old.replace('\\r\\n', '\\n')
    new_crlf = new.replace('\\r\\n', '\\n').replace('\\n', '\\r\\n')
    new_lf = new.replace('\\r\\n', '\\n')
    count = 0
    matched_old, matched_new = old, new
    for cand_old, cand_new in ((old, new), (old_crlf, new_crlf), (old_lf, new_lf)):
        c = text.count(cand_old)
        if c >= 1:
            matched_old, matched_new, count = cand_old, cand_new, c
            break

    if count == 0:
        print(json.dumps({{'error': 'string_not_found'}}))
        sys.exit(0)
    if count > 1 and not replace_all:
        print(json.dumps({{'error': 'multiple_occurrences', 'count': count}}))
        sys.exit(0)

    result = text.replace(matched_old, matched_new) if replace_all else text.replace(matched_old, matched_new, 1)
    with open(target, 'wb') as f:
        f.write(result.encode('utf-8'))

    print(json.dumps({{'count': count}}))
except FileNotFoundError:
    print(json.dumps({{'error': 'file_not_found'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>&1"""
"""Server-side file edit via temp-file upload for large payloads.

Old/new strings are uploaded as temporary files via `upload_files()`, then this
script reads them, performs the replacement on the source file (which never
leaves the sandbox), and cleans up the temp files.

Output: single-line JSON with `{{"count": N}}` on success or
`{{"error": ...}}` on failure.  Same success contract as
`_EDIT_COMMAND_TEMPLATE`; additionally produces
`{{"error": "temp_read_failed", "detail": ...}}` when the uploaded temp
files cannot be read.
"""

_READ_COMMAND_TEMPLATE = """python3 -c "
import codecs, os, stat as _stat, sys, base64, json

MAX_OUTPUT_BYTES = 500 * 1024
MAX_BINARY_BYTES = 500 * 1024
TRUNCATION_MSG = '\\n\\n' + (
    '[Output was truncated due to size limits. '
    'This paginated read result exceeded the sandbox stdout limit. '
    'Continue reading with a larger offset or smaller limit to inspect the rest of the file.]'
)

path = base64.b64decode('{path_b64}').decode('utf-8')

try:
    st = os.stat(path)
    if not _stat.S_ISREG(st.st_mode):
        print(json.dumps({{'error': 'not_a_file'}}))
        sys.exit(0)

    if st.st_size == 0:
        print(json.dumps({{'encoding': 'utf-8', 'content': 'System reminder: File exists but has empty contents'}}))
        sys.exit(0)

    file_type = '{file_type}'
    if file_type != 'text':
        if st.st_size > MAX_BINARY_BYTES:
            print(json.dumps({{'error': 'Binary file exceeds maximum preview size of ' + str(MAX_BINARY_BYTES) + ' bytes'}}))
            sys.exit(0)
        with open(path, 'rb') as f:
            raw = f.read()
        print(json.dumps({{'encoding': 'base64', 'content': base64.b64encode(raw).decode('ascii')}}))
        sys.exit(0)

    with open(path, 'rb') as f:
        raw_prefix = f.read(8192)

    # The 8192-byte prefix can slice a multi-byte UTF-8 char (CJK is 3 bytes,
    # emoji is 4); the incremental decoder buffers a trailing partial sequence
    # instead of raising, so legitimate text isn't misclassified as binary.
    is_binary = False
    try:
        codecs.getincrementaldecoder('utf-8')().decode(raw_prefix, final=False)
    except UnicodeDecodeError:
        is_binary = True

    if is_binary:
        with open(path, 'rb') as f:
            raw = f.read()
        print(json.dumps({{'encoding': 'base64', 'content': base64.b64encode(raw).decode('ascii')}}))
        sys.exit(0)

    offset = {offset}
    limit = {limit}
    line_count = 0
    returned_lines = 0
    truncated = False
    parts = []
    current_bytes = 0
    msg_bytes = len(TRUNCATION_MSG.encode('utf-8'))
    effective_limit = MAX_OUTPUT_BYTES - msg_bytes

    with open(path, 'r', encoding='utf-8', newline=None) as f:
        for raw_line in f:
            line_count += 1
            if line_count <= offset:
                continue
            if returned_lines >= limit:
                break

            line = raw_line.rstrip('\\n').rstrip('\\r')
            piece = line if returned_lines == 0 else '\\n' + line
            piece_bytes = len(piece.encode('utf-8'))
            if current_bytes + piece_bytes > effective_limit:
                truncated = True
                remaining_bytes = effective_limit - current_bytes
                if remaining_bytes > 0:
                    prefix = piece.encode('utf-8')[:remaining_bytes].decode('utf-8', errors='ignore')
                    if prefix:
                        parts.append(prefix)
                        current_bytes += len(prefix.encode('utf-8'))
                break

            parts.append(piece)
            current_bytes += piece_bytes
            returned_lines += 1

    if returned_lines == 0 and not truncated:
        print(json.dumps({{'error': 'Line offset ' + str(offset) + ' exceeds file length (' + str(line_count) + ' lines)'}}))
        sys.exit(0)

    text = ''.join(parts)
    if truncated:
        text += TRUNCATION_MSG

    print(json.dumps({{'encoding': 'utf-8', 'content': text}}))
except FileNotFoundError:
    print(json.dumps({{'error': 'file_not_found'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>&1"""
"""Read file content with server-side pagination.

Runs on the sandbox via `execute()`. Only the requested page is returned,
avoiding full-file transfer for paginated text reads. The path is
base64-encoded; `file_type`, `offset`, and `limit` are interpolated directly
(safe because they come from internal code, not user input).

Output: single-line JSON with either `{{"encoding": ..., "content": ...}}` on
success or `{{"error": ...}}` on failure.
"""


def _build_ls_cmd(path: str) -> str:
    path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")
    return f"""python3 -c "
import os
import json
import base64

path = base64.b64decode('{path_b64}').decode('utf-8')

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': os.path.join(path, entry.name),
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    print(json.dumps({{'error': 'path_not_found'}}))
except NotADirectoryError:
    print(json.dumps({{'error': 'not_a_directory'}}))
except PermissionError:
    print(json.dumps({{'error': 'permission_denied'}}))
" 2>/dev/null"""


def _parse_ls_output(output: str, path: str) -> LsResult:
    file_infos: list[FileInfo] = []
    error: str | None = None
    for line in output.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "error" in data:
            error = data["error"]
            continue
        file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
    if error is not None:
        return LsResult(entries=None, error=f"Path '{path}': {error}")
    return LsResult(entries=file_infos)


def _build_read_cmd(file_path: str, offset: int, limit: int) -> str:
    file_type = _get_backend_read_file_type(file_path)
    path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
    # Defensive int coercion in case callers bypass type checking.
    return _READ_COMMAND_TEMPLATE.format(
        path_b64=path_b64,
        file_type=file_type,
        offset=int(offset),
        limit=int(limit),
    )


def _parse_read_output(output: str, file_path: str) -> ReadResult:
    output = output.rstrip()
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        detail = output[:200] if output else "(empty)"
        return ReadResult(error=f"File '{file_path}': unexpected server response: {detail}")
    if not isinstance(data, dict):
        detail = output[:200] if output else "(empty)"
        return ReadResult(error=f"File '{file_path}': unexpected server response: {detail}")
    if "error" in data:
        return ReadResult(error=f"File '{file_path}': {data['error']}")
    return ReadResult(
        file_data=FileData(
            content=data["content"],
            encoding=data.get("encoding", "utf-8"),
        )
    )


def _build_write_preflight_cmd(file_path: str) -> str:
    path_b64 = base64.b64encode(file_path.encode("utf-8")).decode("ascii")
    return _WRITE_CHECK_TEMPLATE.format(path_b64=path_b64)


def _check_preflight_result(result: ExecuteResponse, file_path: str) -> WriteResult | None:
    if result.exit_code != 0 or "Error:" in result.output:
        error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
        return WriteResult(error=error_msg)
    return None


def _build_grep_cmd(pattern: str, path: str | None, glob: str | None, max_count: int | None = None) -> str:
    search_path = shlex.quote(path or ".")
    # `-Z` separates the filename from line data with NUL, so filenames may
    # contain `:` without making the output ambiguous.
    grep_opts = "-rHnFZ"
    pattern_escaped = shlex.quote(pattern)

    # GNU `grep --include` only matches basenames, so a slash-containing glob
    # like `src/**/*.py` would silently match zero files. Route those to the
    # in-process Python template that resolves the glob relative to the search
    # root. Basename-only globs (no `/`) work correctly with `--include` and
    # are faster to run through GNU grep.
    if glob and "/" in glob:
        path_b64 = base64.b64encode((path or ".").encode("utf-8")).decode("ascii")
        glob_b64 = base64.b64encode(glob.encode("utf-8")).decode("ascii")
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        return _GREP_PATH_GLOB_TEMPLATE.format(
            path_b64=path_b64,
            glob_b64=glob_b64,
            pattern_b64=pattern_b64,
            max_count=None if max_count is None else int(max_count),
        )

    glob_pattern = f"--include={shlex.quote(glob)}" if glob else ""
    base = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null"
    if max_count is not None:
        # Read one record beyond the cap so the parser can distinguish "exactly
        # at the cap" (complete) from "capped early" (truncated). `head` closing
        # the pipe delivers SIGPIPE to grep, stopping it early rather than
        # letting it keep scanning a huge tree after the cap is met.
        return f"{base} | head -n {int(max_count) + 1} || true"
    return f"{base} || true"


def _parse_grep_output(result: ExecuteResponse, path: str | None, max_count: int | None = None) -> GrepResult:
    output = result.output.rstrip("\n")
    if result.exit_code is not None and result.exit_code != 0:
        detail = output.strip() if output else f"exit code {result.exit_code}"
        return GrepResult(error=f"Path '{path or '.'}': {detail}")
    if not output:
        return GrepResult(matches=[])
    matches: list[GrepMatch] = []
    parse_error: str | None = None
    for line in output.split("\n"):
        # Format is: path\0line_number:text
        try:
            file_path, rest = line.split("\0", 1)
            line_num_str, text = rest.split(":", 1)
            matches.append({"path": file_path, "line": int(line_num_str), "text": text})
        except ValueError:
            parse_error = line
    if parse_error is not None and not matches:
        return GrepResult(error=f"Path '{path or '.'}': {parse_error}")
    if max_count is not None and len(matches) > max_count:
        # More matches existed than the caller asked for; return the cap and
        # flag the result as incomplete.
        return GrepResult(matches=matches[:max_count], truncated=True)
    return GrepResult(matches=matches)


def _build_glob_cmd(pattern: str, search_path: str) -> str:
    pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
    path_b64 = base64.b64encode(search_path.encode("utf-8")).decode("ascii")
    return _GLOB_COMMAND_TEMPLATE.format(path_b64=path_b64, pattern_b64=pattern_b64)


def _parse_glob_output(output: str, search_path: str) -> GlobResult:
    output = output.strip()
    if not output:
        return GlobResult(matches=[])
    file_infos: list[FileInfo] = []
    error: str | None = None
    for line in output.split("\n"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "error" in data:
            error = data["error"]
            continue
        file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
    if error is not None:
        return GlobResult(matches=None, error=f"Path '{search_path}': {error}")
    return GlobResult(matches=file_infos)


def _build_edit_inline_cmd(file_path: str, old_string: str, new_string: str, *, replace_all: bool) -> str:
    payload = json.dumps({"path": file_path, "old": old_string, "new": new_string, "replace_all": replace_all})
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return _EDIT_COMMAND_TEMPLATE.format(payload_b64=payload_b64)


def _map_edit_error(error: str, file_path: str, old_string: str) -> EditResult:
    """Map server-side error codes to `EditResult` objects."""
    messages: dict[str, str] = {
        "file_not_found": f"Error: File '{file_path}' not found",
        "permission_denied": f"Error: Permission denied editing file '{file_path}'",
        "not_a_file": f"Error: '{file_path}' is not a regular file",
        "not_a_text_file": f"Error: File '{file_path}' is not a text file",
        "string_not_found": f"Error: String not found in file: '{old_string}'",
        "multiple_occurrences": (f"Error: String '{old_string}' appears multiple times. Use replace_all=True to replace all occurrences."),
    }
    return EditResult(error=messages.get(error, f"Error editing file '{file_path}': {error}"))


def _parse_edit_output(output: str, file_path: str, old_string: str) -> EditResult:
    output = output.rstrip()
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        detail = output[:200] if output else "(empty)"
        return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")
    if not isinstance(data, dict):
        detail = output[:200] if output else "(empty)"
        return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")
    if "error" in data:
        return _map_edit_error(data["error"], file_path, old_string)
    return EditResult(path=file_path, occurrences=data.get("count", 1))


def _build_edit_tmpfile_cmd(file_path: str, old_tmp: str, new_tmp: str, *, replace_all: bool) -> str:
    return _EDIT_TMPFILE_TEMPLATE.format(
        old_path_b64=base64.b64encode(old_tmp.encode("utf-8")).decode("ascii"),
        new_path_b64=base64.b64encode(new_tmp.encode("utf-8")).decode("ascii"),
        target_b64=base64.b64encode(file_path.encode("utf-8")).decode("ascii"),
        replace_all=replace_all,
    )


_EXECUTE_CAPTURE_SENTINEL: Final = "__DEEPAGENTS_EXEC_META__"
"""First-line marker identifying capture-wrapper output: `<sentinel> <exit_code> <offloaded> <capped>`."""

_EXECUTE_CAPTURE_HEAD_LINES: Final = 5
_EXECUTE_CAPTURE_TAIL_LINES: Final = 5
_EXECUTE_CAPTURE_HEAD_BYTES: Final = 2000
_EXECUTE_CAPTURE_TAIL_BYTES: Final = 2000

_EXECUTE_CAPTURE_MAX_BYTES: Final = 10 * 1024 * 1024
"""Hard cap on captured stdout/stderr persisted to the sandbox.

Bounds sandbox disk use for runaway output: the captured stream is piped through
`head -c`, so when the cap is hit the writer receives `SIGPIPE` and nothing
further reaches disk even if the command ignores the signal. Set well above the
inline budget so legitimately large output is still preserved in full; output
beyond the cap is truncated and flagged.
"""

# The captured stream is piped into `head -c` (caps the on-disk file) followed by
# `cat > /dev/null` (drains the rest), so the file can never exceed the cap yet the
# command still reaches EOF and exits normally -- closing the pipe early would
# SIGPIPE-kill it and corrupt its exit code. Because the command is in a pipeline,
# its real exit code is recovered from a sidecar file rather than `$?` (which would
# be the pipeline's). The command runs in a subshell so a command `exit` cannot
# abort the wrapper, and `eval` preserves the backend's own shell/env. The command
# is embedded via a quoted heredoc with a random delimiter to avoid shell-quoting
# issues; the (internal, sanitized) path is shell-quoted.
_EXECUTE_CAPTURE_CMD_TEMPLATE = """# ===== deepagents capture-at-source offload (auto-generated wrapper) =====
# Runs the requested command below, capturing its combined output to a file in
# the sandbox: returned inline when small, or as a head/tail preview when large
# (the full result stays at the path for read_file). Disable this wrapping with
# BaseSandbox.enable_capture_offload = False.
__da_f=__PATH_Q__
__da_ecf="$__da_f.ec"
mkdir -p "$(dirname "$__da_f")" 2>/dev/null
# ----- requested command (verbatim, between the heredoc markers) -----
__da_cmd=$(cat <<'__DELIM__'
__COMMAND__
__DELIM__
)
# ----- end requested command; everything below is offload machinery -----
{ ( eval "$__da_cmd" ); echo "$?" > "$__da_ecf"; } 2>&1 | { head -c __MAXBYTES__ > "$__da_f"; cat > /dev/null; }
__da_ec=$(cat "$__da_ecf" 2>/dev/null)
: "${__da_ec:=1}"
rm -f "$__da_ecf"
__da_bytes=$(wc -c < "$__da_f" 2>/dev/null | tr -d ' ')
: "${__da_bytes:=0}"
__da_capped=0
[ "$__da_bytes" -ge __MAXBYTES__ ] && __da_capped=1
if [ "$__da_bytes" -le __BUDGET__ ]; then
  printf '%s %s %s %s\\n' '__SENTINEL__' "$__da_ec" 0 0
  cat "$__da_f"
  rm -f "$__da_f"
else
  __da_lines=$(wc -l < "$__da_f" 2>/dev/null | tr -d ' ')
  : "${__da_lines:=0}"
  __da_omitted=$((__da_lines - __HEADLINES__ - __TAILLINES__))
  printf '%s %s %s %s\\n' '__SENTINEL__' "$__da_ec" 1 "$__da_capped"
  if [ "$__da_omitted" -gt 0 ]; then
    head -c __HEAD__ "$__da_f" | head -n __HEADLINES__
    printf '... [%s lines truncated] ...\\n' "$__da_omitted"
    tail -c __TAIL__ "$__da_f" | tail -n __TAILLINES__
  else
    head -c $((__HEAD__ + __TAIL__)) "$__da_f"
  fi
fi
"""
# Pure POSIX sh wrapper for capture-at-source `execute`; see the comment above the template.


def _new_heredoc_delim() -> str:
    """Return a random heredoc delimiter, e.g. `__DEEPAGENTS_CMD_<80 random bits>__`."""
    return "__DEEPAGENTS_CMD_" + base64.b32encode(os.urandom(10)).decode("ascii").rstrip("=") + "__"


def _build_capture_execute_cmd(command: str, capture_path: str, *, inline_budget: int, max_capture_bytes: int | None = None) -> str:
    """Build the capture-at-source wrapper command for `execute`.

    `inline_budget` is the byte threshold at or below which output is returned
    inline; above it the output is left at `capture_path` and only a head/tail
    preview is returned. Captured output is hard-capped at `max_capture_bytes`
    (defaulting to `_EXECUTE_CAPTURE_MAX_BYTES`, resolved here so it stays
    overridable/patchable); beyond that it is truncated and flagged.
    """
    cap = max_capture_bytes if max_capture_bytes is not None else _EXECUTE_CAPTURE_MAX_BYTES
    # The command is embedded in a quoted heredoc; guarantee the delimiter cannot
    # appear in it so the command can never terminate the heredoc early. The
    # delimiter is 80 random bits, so this regenerates only astronomically rarely.
    delim = _new_heredoc_delim()
    while delim in command:
        delim = _new_heredoc_delim()
    # __COMMAND__ is substituted last so command content can never collide with a
    # remaining placeholder token.
    return (
        _EXECUTE_CAPTURE_CMD_TEMPLATE.replace("__PATH_Q__", shlex.quote(capture_path))
        .replace("__DELIM__", delim)
        .replace("__MAXBYTES__", str(cap))
        .replace("__BUDGET__", str(inline_budget))
        .replace("__SENTINEL__", _EXECUTE_CAPTURE_SENTINEL)
        .replace("__HEADLINES__", str(_EXECUTE_CAPTURE_HEAD_LINES))
        .replace("__TAILLINES__", str(_EXECUTE_CAPTURE_TAIL_LINES))
        .replace("__HEAD__", str(_EXECUTE_CAPTURE_HEAD_BYTES))
        .replace("__TAIL__", str(_EXECUTE_CAPTURE_TAIL_BYTES))
        .replace("__COMMAND__", command)
    )


def _parse_capture_execute_output(output: str, *, backend_truncated: bool = False) -> ExecuteOffloadResult:
    r"""Parse capture-wrapper stdout into an `ExecuteOffloadResult`.

    The wrapper emits a meta line followed by the body:

        <sentinel> <exit_code> <offloaded> <capped>\n<inline output or preview>

    i.e. four space-separated fields on the first line — the sentinel, the
    command's exit code, `1`/`0` for whether output was offloaded to the capture
    file, and `1`/`0` for whether it hit the size cap — then everything after the
    first newline is the body (full output when inline, head/tail preview when
    offloaded).

    Falls back to `offloaded=False` with the raw output when the meta line is
    absent or malformed — e.g. if the backend truncated transport; the caller
    must not re-run the command in that case. `response.truncated` is set when the
    captured output hit the size cap (the saved file is incomplete) or
    `backend_truncated` is passed through from the underlying `execute`.
    """
    first, _, body = output.partition("\n")
    parts = first.split(" ")
    # Expect exactly the four meta fields described above; anything else is not
    # our wrapper's output, so fall back to returning it verbatim.
    if len(parts) != 4 or parts[0] != _EXECUTE_CAPTURE_SENTINEL:  # noqa: PLR2004
        return ExecuteOffloadResult(offloaded=False, response=ExecuteResponse(output=output, truncated=backend_truncated))
    try:
        exit_code = int(parts[1])
    except ValueError:
        return ExecuteOffloadResult(offloaded=False, response=ExecuteResponse(output=output, truncated=backend_truncated))
    return ExecuteOffloadResult(
        offloaded=parts[2] == "1",
        response=ExecuteResponse(output=body, exit_code=exit_code, truncated=parts[3] == "1" or backend_truncated),
    )


class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with `execute()` as the core abstract method.

    This class provides default implementations for all protocol methods.
    File listing, grep, and glob use shell commands via `execute()`. Read uses
    a server-side Python script via `execute()` for paginated access. Write
    delegates content transfer to `upload_files()`. Edit uses a server-side
    script for small payloads and uploads old/new strings as temp files with
    a server-side replace for large ones.

    !!! note

        `BaseSandbox` does not reduce or partition the trust boundary of
        `execute()`. Its helper methods are convenience wrappers built on top of
        the subclass-provided command-execution primitive and assume callers who
        can use `BaseSandbox` already have whatever shell-execution capability
        that backend exposes.

    Subclasses must implement `execute()`, `upload_files()`, `download_files()`,
    and the `id` property.
    """

    enable_capture_offload: bool = False
    """Whether `FilesystemMiddleware` may use capture-at-source offload for `execute`.

    When `True`, large `execute` output is captured to a file in the sandbox and
    only a preview is returned, avoiding a round-trip back through the agent
    process. Defaults to `False` (opt-in) because the capture wrapper's shell and
    coreutils assumptions are not guaranteed on every sandbox image; subclasses
    known to be compatible set it to `True`. When `False`, `execute_with_offload`
    runs the command unwrapped and the middleware falls back to inline execution
    plus generic eviction.
    """

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a command in the sandbox and return `ExecuteResponse`.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for the command to complete.

                If `None`, uses the backend's default timeout.

        Returns:
            `ExecuteResponse` with combined output, exit code, and truncation flag.
        """

    def execute_with_offload(
        self,
        command: str,
        capture_path: str,
        *,
        max_inline_bytes: int,
        max_capture_bytes: int | None = None,
        timeout: int | None = None,
    ) -> ExecuteOffloadResult:
        """Run `command`, offloading large output to a file in the sandbox.

        Captures the command's combined output: returned inline when it is at or
        below `max_inline_bytes`, otherwise left at `capture_path` (so the caller
        can surface a `read_file` pointer) with only a head/tail preview returned.
        Captured output is hard-capped at `max_capture_bytes` (default
        `_EXECUTE_CAPTURE_MAX_BYTES`) without killing the command, so the exit
        code is preserved. When `enable_capture_offload` is `False`, the command
        runs unwrapped and the full output is returned (`offloaded=False`), so
        callers can fall back to their own handling (e.g. generic eviction).

        Returns:
            An `ExecuteOffloadResult`. `offloaded=True` when the result was left
            at `capture_path` and `response.output` holds only the preview;
            `offloaded=False` when `response.output` is the complete output.
        """
        use_timeout = timeout is not None and execute_accepts_timeout(type(self))
        if not self.enable_capture_offload:
            result = self.execute(command, timeout=timeout) if use_timeout else self.execute(command)
            return ExecuteOffloadResult(offloaded=False, response=result)
        wrapper = _build_capture_execute_cmd(command, capture_path, inline_budget=max_inline_bytes, max_capture_bytes=max_capture_bytes)
        result = self.execute(wrapper, timeout=timeout) if use_timeout else self.execute(wrapper)
        return _parse_capture_execute_output(result.output, backend_truncated=result.truncated)

    async def aexecute_with_offload(
        self,
        command: str,
        capture_path: str,
        *,
        max_inline_bytes: int,
        max_capture_bytes: int | None = None,
        timeout: int | None = None,  # noqa: ASYNC109  # forwarded to the backend, not an asyncio timeout
    ) -> ExecuteOffloadResult:
        """Async version of `execute_with_offload`, delegating to `aexecute`."""
        use_timeout = timeout is not None and execute_accepts_timeout(type(self))
        if not self.enable_capture_offload:
            result = await self.aexecute(command, timeout=timeout) if use_timeout else await self.aexecute(command)
            return ExecuteOffloadResult(offloaded=False, response=result)
        wrapper = _build_capture_execute_cmd(command, capture_path, inline_budget=max_inline_bytes, max_capture_bytes=max_capture_bytes)
        result = await self.aexecute(wrapper, timeout=timeout) if use_timeout else await self.aexecute(wrapper)
        return _parse_capture_execute_output(result.output, backend_truncated=result.truncated)

    def ls(self, path: str) -> LsResult:
        """Structured listing with file metadata using os.scandir."""
        result = self.execute(_build_ls_cmd(path))
        return _parse_ls_output(result.output, path)

    async def als(self, path: str) -> LsResult:
        """Async version of `ls`, delegating to `aexecute`."""
        result = await self.aexecute(_build_ls_cmd(path))
        return _parse_ls_output(result.output, path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content with server-side line-based pagination.

        Runs a Python script on the sandbox via `execute()` that reads the
        file, detects encoding, and applies offset/limit pagination for text
        files. Only the requested page is returned over the wire, and text
        output is capped to about 500 KiB to avoid backend stdout/log transport
        failures. When that cap is exceeded, the returned content is truncated
        with guidance to continue pagination using a different `offset` or
        smaller `limit`.

        Binary files (non-UTF-8) are returned base64-encoded without
        pagination.

        Args:
            file_path: Absolute path to the file to read.
            offset: Starting line number (0-indexed).

                Only applied to text files.
            limit: Maximum number of lines to return.

                Only applied to text files.

        Returns:
            `ReadResult` with `file_data` on success or `error` on failure.
        """
        result = self.execute(_build_read_cmd(file_path, offset, limit))
        return _parse_read_output(result.output, file_path)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Async version of `read`, delegating to `aexecute`."""
        result = await self.aexecute(_build_read_cmd(file_path, offset, limit))
        return _parse_read_output(result.output, file_path)

    def _write_preflight(self, file_path: str) -> WriteResult | None:
        """Create parent directories for `write()`.

        Subclasses overriding `write()` (e.g., to use a native SDK transport)
        should call this first so they preserve the parent-mkdir semantics of
        `BaseSandbox.write()`. There is a TOCTOU window between this and the
        actual write — an inherent limitation of splitting the operation across
        two backend calls.

        Args:
            file_path: Absolute path for the file about to be written.

        Returns:
            `None` if the preflight passes (parents created); a populated
                `WriteResult` with `error` set if the preflight fails.
        """
        result = self.execute(_build_write_preflight_cmd(file_path))
        return _check_preflight_result(result, file_path)

    async def _awrite_preflight(self, file_path: str) -> WriteResult | None:
        """Async version of `_write_preflight`, delegating to `aexecute`."""
        result = await self.aexecute(_build_write_preflight_cmd(file_path))
        return _check_preflight_result(result, file_path)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a file, creating or overwriting it if it already exists.

        Args:
            file_path: Absolute path for the file.
            content: UTF-8 text content to write.

        Returns:
            `WriteResult` with `path` on success or `error` on failure.
        """
        preflight_error = self._write_preflight(file_path)
        if preflight_error is not None:
            return preflight_error

        responses = self.upload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            # An unreachable condition was reached
            msg = f"Responses was expected to return 1 result, but it returned {len(responses)} with type {type(responses)}"
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(error=f"Failed to write file '{file_path}': {response.error}")

        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async version of `write`, delegating to `aexecute` and `aupload_files`."""
        preflight_error = await self._awrite_preflight(file_path)
        if preflight_error is not None:
            return preflight_error
        responses = await self.aupload_files([(file_path, content.encode("utf-8"))])
        if not responses:
            msg = f"Responses was expected to return 1 result, but it returned {len(responses)} with type {type(responses)}"
            raise AssertionError(msg)
        response = responses[0]
        if response.error:
            return WriteResult(error=f"Failed to write file '{file_path}': {response.error}")
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Edit a file by replacing exact string occurrences.

        For small payloads (combined old/new under `_EDIT_INLINE_MAX_BYTES`),
        runs a server-side Python script via `execute()` — single round-trip,
        no file transfer.  For larger payloads, uploads old/new strings as
        temp files and runs a server-side replace script — the source file
        never leaves the sandbox.

        `read()` normalizes CRLF to LF for the LLM, so `old_string` is
        typically LF-only. The server-side script tries `old_string` as-is
        first, then CRLF- and LF-normalized variants, and applies the same
        transform to `new_string` so the file's line-ending style is
        preserved on write. On mixed-ending files, `replace_all=True` only
        touches occurrences in the first matching style — subsequent edits
        can replace the rest.

        Args:
            file_path: Absolute path to the file to edit.
            old_string: The exact substring to find.
            new_string: The replacement string.
            replace_all: If `True`, replace every occurrence.

                If `False` (default), error when more than one
                occurrence exists.

        Returns:
            `EditResult` with `path` and `occurrences` on success, or `error`
                on failure.
        """
        payload_size = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))

        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            return self._edit_inline(file_path, old_string, new_string, replace_all)

        return self._edit_via_upload(file_path, old_string, new_string, replace_all)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Async version of `edit`, delegating to `aexecute` and `aupload_files`."""
        payload_size = len(old_string.encode("utf-8")) + len(new_string.encode("utf-8"))
        if payload_size <= _EDIT_INLINE_MAX_BYTES:
            return await self._aedit_inline(file_path, old_string, new_string, replace_all)
        return await self._aedit_via_upload(file_path, old_string, new_string, replace_all)

    def _edit_inline(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Server-side replace via `execute()` (single round-trip)."""
        result = self.execute(_build_edit_inline_cmd(file_path, old_string, new_string, replace_all=replace_all))
        return _parse_edit_output(result.output, file_path, old_string)

    async def _aedit_inline(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Async version of `_edit_inline`, delegating to `aexecute`."""
        result = await self.aexecute(_build_edit_inline_cmd(file_path, old_string, new_string, replace_all=replace_all))
        return _parse_edit_output(result.output, file_path, old_string)

    def _edit_via_upload(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Upload old/new as temp files, replace server-side.

        The source file never leaves the sandbox. Only the old/new strings are
        transferred via `upload_files()`, and a server-side script reads them,
        performs the replacement, and cleans up the temp files.
        """
        uid = base64.b32encode(os.urandom(10)).decode("ascii").lower()
        old_tmp = f"/tmp/.deepagents_edit_{uid}_old"  # noqa: S108  # sandbox-internal temp file with 80-bit random uid
        new_tmp = f"/tmp/.deepagents_edit_{uid}_new"  # noqa: S108

        resps = self.upload_files(
            [
                (old_tmp, old_string.encode("utf-8")),
                (new_tmp, new_string.encode("utf-8")),
            ]
        )
        if len(resps) < 2:  # noqa: PLR2004  # expecting exactly 2 responses
            return EditResult(error=f"Error editing file '{file_path}': upload returned no response")
        for r in resps:
            if r.error:
                return EditResult(error=f"Error editing file '{file_path}': {r.error}")

        cmd = _build_edit_tmpfile_cmd(file_path, old_tmp, new_tmp, replace_all=replace_all)
        result = self.execute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            # Script may not have started or its finally block may not have
            # run — best-effort cleanup of temp files.
            cleanup = self.execute(f"rm -f {shlex.quote(old_tmp)} {shlex.quote(new_tmp)}")
            if cleanup.exit_code != 0:
                logger.warning(
                    "Failed to clean up temp files for edit %s: %s",
                    file_path,
                    cleanup.output[:200],
                )
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if "error" in data:
            return _map_edit_error(data["error"], file_path, old_string)

        return EditResult(
            path=file_path,
            occurrences=data.get("count", 1),
        )

    async def _aedit_via_upload(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,  # noqa: FBT001
    ) -> EditResult:
        """Async version of `_edit_via_upload`, delegating to `aexecute` and `aupload_files`."""
        uid = base64.b32encode(os.urandom(10)).decode("ascii").lower()
        old_tmp = f"/tmp/.deepagents_edit_{uid}_old"  # noqa: S108
        new_tmp = f"/tmp/.deepagents_edit_{uid}_new"  # noqa: S108

        resps = await self.aupload_files(
            [
                (old_tmp, old_string.encode("utf-8")),
                (new_tmp, new_string.encode("utf-8")),
            ]
        )
        if len(resps) < 2:  # noqa: PLR2004
            return EditResult(error=f"Error editing file '{file_path}': upload returned no response")
        for r in resps:
            if r.error:
                return EditResult(error=f"Error editing file '{file_path}': {r.error}")

        cmd = _build_edit_tmpfile_cmd(file_path, old_tmp, new_tmp, replace_all=replace_all)
        result = await self.aexecute(cmd)
        output = result.output.rstrip()

        try:
            data = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            cleanup = await self.aexecute(f"rm -f {shlex.quote(old_tmp)} {shlex.quote(new_tmp)}")
            if cleanup.exit_code != 0:
                logger.warning(
                    "Failed to clean up temp files for edit %s: %s",
                    file_path,
                    cleanup.output[:200],
                )
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if not isinstance(data, dict):
            detail = output[:200] if output else "(empty)"
            return EditResult(error=f"Error editing file '{file_path}': unexpected server response: {detail}")

        if "error" in data:
            return _map_edit_error(data["error"], file_path, old_string)

        return EditResult(path=file_path, occurrences=data.get("count", 1))

    def delete(self, file_path: str) -> DeleteResult:
        """Delete a file or directory from the sandbox via a server-side `rm`.

        Runs `test -e || test -L` first: a path that does not exist (and is not
        a broken symlink) returns a not-found error, matching the contract of
        `FilesystemBackend` and `StateBackend`. Because a shell `test` has no
        error channel, a non-zero probe conflates "absent" with "unstattable"
        (e.g. an unsearchable parent directory); an unknown exit code is not
        treated as absent and falls through to the delete.

        Uses `rm -rf`, so directories are removed recursively along with their
        contents. A recursive delete may remove some entries before failing
        partway; a non-zero `rm` exit (e.g. a permission error) is reported as
        a failure.

        Args:
            file_path: Absolute path to the file or directory to delete.

        Returns:
            `DeleteResult` with the deleted path on success, or an error if the
                path does not exist or the deletion command fails.
        """
        # `shlex.quote` only neutralizes shell metacharacters so the path is
        # passed to `rm` as a single literal argument. It is NOT a security
        # boundary: it does not confine the deletion to any sandbox root or
        # block traversal. Whatever the sandbox shell can reach, this can delete.
        quoted = shlex.quote(file_path)
        exists = self.execute(f"test -e {quoted} || test -L {quoted}")
        # `exit_code` may be None when the backend cannot determine a status;
        # only a definite non-zero means the path is absent. Treating None as
        # not-found would fabricate a diagnosis and skip the delete, so fall
        # through to `rm` on an unknown probe result (matches the `rm` check
        # below and `_parse_grep_output`, which both guard `is not None`).
        if exists.exit_code is not None and exists.exit_code != 0:
            return DeleteResult(error=f"Error: '{file_path}' not found")
        result = self.execute(f"rm -rf {quoted}")

        if result.exit_code == 0:
            return DeleteResult(path=file_path)

        return DeleteResult(error=f"Error deleting file '{file_path}': {result.output.strip() or 'unknown error'}")

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents for a literal string using `grep -F`.

        Args:
            pattern: Literal string to search for (not a regex).
            path: Directory or file to search in.

                Defaults to `"."`.
            glob: Optional glob to restrict the search. Patterns without a
                `/` (e.g. `'*.py'`) match basenames at any depth via
                `grep --include`; patterns containing a `/` (e.g.
                `'src/**/*.py'`) match the search-root-relative path via an
                in-process Python glob.
            max_count: Optional total cap on returned matches across all files.
                `None` returns every match; an int stops the search once the cap
                is reached and flags the result with `truncated=True`.

        Returns:
            `GrepResult` with a list of `GrepMatch` dicts, or `error` on failure.
        """
        result = self.execute(_build_grep_cmd(pattern, path, glob, max_count))
        return _parse_grep_output(result, path, max_count)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Async version of `grep`, delegating to `aexecute` with timeout guard."""
        try:
            result = await asyncio.wait_for(
                self.aexecute(_build_grep_cmd(pattern, path, glob, max_count)),
                timeout=ASYNC_GREP_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "agrep timed out after %ds (pattern=%r, path=%r, glob=%r)",
                ASYNC_GREP_TIMEOUT,
                pattern,
                path,
                glob,
            )
            return GrepResult(
                error=f"Error: grep timed out after {ASYNC_GREP_TIMEOUT}s. Try a more specific pattern or a narrower path.",
            )
        return _parse_grep_output(result, path, max_count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Structured glob matching returning `GlobResult`."""
        search_path = path or "/"
        result = self.execute(_build_glob_cmd(pattern, search_path))
        return _parse_glob_output(result.output, search_path)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Async version of `glob`, delegating to `aexecute`."""
        search_path = path or "/"
        result = await self.aexecute(_build_glob_cmd(pattern, search_path))
        return _parse_glob_output(result.output, search_path)

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""

    @abstractmethod
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in `FileUploadResponse` objects rather than raising.

        Upload files is responsible for ensuring that the parent path exists
        (if user permissions allow the user to write to the given directory)
        """

    @abstractmethod
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Implementations must support partial success - catch exceptions per-file
        and return errors in `FileDownloadResponse` objects rather than raising.
        """
