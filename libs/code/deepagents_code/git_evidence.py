"""Bounded, read-only Git evidence collected in the tool execution environment."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shlex
import subprocess  # noqa: S404  # Bounded read-only Git subprocesses.
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from deepagents_code._git import (
    RepositoryMetadata,
    find_git_root,
    parse_repository_metadata,
    resolve_git_branch,
    resolve_git_commit_sha,
    resolve_git_remote_url,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deepagents.backends.protocol import BackendProtocol

_LIMIT = 256_000
_MAX_REPOSITORIES = 16
_MAX_PATHS = 4096
_MAX_DIRECTORIES = 64
_ROW_FIELDS = 5


@dataclass(frozen=True)
class GitSnapshot:
    """Repository identity and opaque non-content evidence of checkout work."""

    root: str
    repository: RepositoryMetadata | None
    branch: str
    commit: str
    fingerprint: str


def _git(path: str, *args: str) -> bytes:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        path,
        *args,
    ]
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(  # noqa: S603  # Literal Git arguments; never a shell.
            command,
            stdout=output,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=2,
            check=False,
        )
        output.seek(0)
        data = output.read(_LIMIT + 1)
    if result.returncode or len(data) > _LIMIT:
        msg = "Git evidence is unavailable or exceeds the inspection limit."
        raise ValueError(msg)
    return data


def _fingerprint(root: str) -> str:
    status = _git(root, "status", "--porcelain=v2", "-z", "--untracked-files=all")
    paths = _git(root, "ls-files", "-m", "-o", "--exclude-standard", "-z").split(b"\0")
    digest = hashlib.sha256(status)
    if len(paths) > _MAX_PATHS:
        msg = "Git evidence exceeds the dirty-file limit."
        raise ValueError(msg)
    for raw in paths:
        if raw:
            path = Path(root) / os.fsdecode(raw)
            try:
                stat = path.lstat()
                digest.update(
                    repr(
                        (
                            raw,
                            stat.st_mode,
                            stat.st_size,
                            stat.st_mtime_ns,
                            stat.st_ctime_ns,
                        )
                    ).encode()
                )
            except FileNotFoundError:
                digest.update(raw + b"\0missing")
    return digest.hexdigest()


def _discover(paths: list[str]) -> list[str]:
    candidates = list(paths)
    pending = [(Path(paths[0]), 0)] if paths else []
    visited = 0
    while pending and visited < _MAX_DIRECTORIES:
        directory, depth = pending.pop(0)
        visited += 1
        try:
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= _MAX_DIRECTORIES:
                        break
                    if entry.name.startswith(".") or entry.name in {
                        "node_modules",
                        "vendor",
                        "__pycache__",
                    }:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        candidates.append(entry.path)
                        if depth < 1:
                            pending.append((Path(entry.path), depth + 1))
        except OSError:
            continue
    roots: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        while not path.is_dir() and path != path.parent:
            path = path.parent
        for parent in (path, *path.parents):
            if (parent / ".git").exists():
                root = str(parent.resolve())
                if root not in roots:
                    roots.append(root)
                break
        if len(roots) >= _MAX_REPOSITORIES:
            break
    return roots


def _remote_rows(paths: list[str]) -> list[list[str]]:
    rows = []
    for root in _discover(paths):
        try:
            branch = _git(root, "symbolic-ref", "--short", "HEAD").decode().strip()
        except ValueError:
            branch = "HEAD"
        try:
            commit = _git(root, "rev-parse", "--verify", "HEAD").decode().strip()
        except ValueError:
            commit = ""
        try:
            remote = _git(root, "config", "--get", "remote.origin.url").decode().strip()
        except ValueError:
            remote = ""
        if "://" in remote:
            parsed = urlparse(remote)
            remote = f"https://{parsed.hostname or ''}{parsed.path}"
        else:
            remote = remote.rsplit("@", 1)[-1]
        rows.append([root, remote, branch, commit, _fingerprint(root)])
    return rows


def _candidate_paths(cwd: str, args: Mapping[str, object] | None) -> list[str]:
    args = args or {}
    values = [cwd]
    values.extend(
        value
        for key in ("cwd", "file_path", "path")
        if isinstance(value := args.get(key), str)
    )
    command = args.get("command")
    if isinstance(command, str):
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = []
        for index, token in enumerate(tokens):
            if token.startswith("/") or (index and tokens[index - 1] in {"cd", "-C"}):
                values.append(token)
    paths = []
    for value in values[:32]:
        if (
            not value
            or len(value) > _MAX_PATHS
            or any(char in value for char in ("\0", "\n", "\r", "$", "`", "~"))
        ):
            continue
        if value.split("/")[1:2] in (["memories"], ["conversation_history"]):
            continue
        path = os.path.normpath(
            value if value.startswith("/") else str(Path(cwd) / value)
        )
        if path not in paths:
            paths.append(path)
    return paths


def _sandbox_command(paths: list[str]) -> str:
    source = (
        "import hashlib,json,os,subprocess,tempfile\nfrom pathlib import Path\n"
        "from urllib.parse import urlparse\n"
    )
    source += f"_LIMIT={_LIMIT}\n_MAX_REPOSITORIES={_MAX_REPOSITORIES}\n"
    source += f"_MAX_PATHS={_MAX_PATHS}\n_MAX_DIRECTORIES={_MAX_DIRECTORIES}\n"
    source += "\n".join(
        inspect.getsource(function)
        for function in (_git, _fingerprint, _discover, _remote_rows)
    )
    source += f"\nprint(json.dumps(_remote_rows({paths!r})))\n"
    return "python3 -c " + shlex.quote(source)


def _snapshots(
    output: str, *, exit_code: int | None, truncated: bool
) -> list[GitSnapshot]:
    if exit_code != 0 or truncated or len(output) > _LIMIT:
        msg = "Git evidence inspection failed or was truncated."
        raise ValueError(msg)
    rows = json.loads(output)
    if not isinstance(rows, list) or len(rows) > _MAX_REPOSITORIES:
        msg = "Malformed Git evidence."
        raise ValueError(msg)
    result = []
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != _ROW_FIELDS
            or not all(isinstance(value, str) for value in row)
        ):
            msg = "Malformed Git evidence row."
            raise ValueError(msg)
        result.append(
            GitSnapshot(
                row[0], parse_repository_metadata(row[1]), row[2], row[3], row[4]
            )
        )
    return result


def _local_probe(
    backend: BackendProtocol, cwd: str, args: Mapping[str, object] | None
) -> list[GitSnapshot]:
    from deepagents.backends.filesystem import FilesystemBackend

    if not isinstance(backend, FilesystemBackend):
        msg = "Backend does not support local Git inspection."
        raise NotImplementedError(msg)
    paths = _candidate_paths(cwd, args)
    paths = [str(backend._resolve_path(path)) for path in paths]
    result = []
    for root in _discover(paths):
        if backend.virtual_mode and not Path(root).is_relative_to(backend.cwd):
            continue
        if find_git_root(root) is None:
            continue
        result.append(
            GitSnapshot(
                root,
                parse_repository_metadata(resolve_git_remote_url(root)),
                resolve_git_branch(root),
                resolve_git_commit_sha(root),
                _fingerprint(root),
            )
        )
    return result


def probe_git(
    backend: BackendProtocol, cwd: str, tool_args: Mapping[str, object] | None = None
) -> list[GitSnapshot]:
    """Inspect candidate checkouts on the backend; optional failures may propagate.

    Returns:
        Snapshots in discovery order, without file contents or remote credentials.
    """
    from deepagents.backends.protocol import (
        SandboxBackendProtocol,
        execute_accepts_timeout,
    )

    if not isinstance(backend, SandboxBackendProtocol):
        return _local_probe(backend, cwd, tool_args)
    command = _sandbox_command(_candidate_paths(cwd, tool_args))
    if not execute_accepts_timeout(type(backend)):
        msg = "Git evidence requires bounded backend execution."
        raise NotImplementedError(msg)
    result = backend.execute(command, timeout=10)
    return _snapshots(
        result.output, exit_code=result.exit_code, truncated=result.truncated
    )


async def aprobe_git(
    backend: BackendProtocol, cwd: str, tool_args: Mapping[str, object] | None = None
) -> list[GitSnapshot]:
    """Inspect backend checkouts without blocking the event loop.

    Returns:
        Snapshots in discovery order from the actual execution environment.
    """
    from deepagents.backends.protocol import (
        SandboxBackendProtocol,
        execute_accepts_timeout,
    )

    if not isinstance(backend, SandboxBackendProtocol):
        return await asyncio.to_thread(_local_probe, backend, cwd, tool_args)
    command = _sandbox_command(_candidate_paths(cwd, tool_args))
    if not execute_accepts_timeout(type(backend)):
        msg = "Git evidence requires bounded backend execution."
        raise NotImplementedError(msg)
    result = await backend.aexecute(command, timeout=10)
    return _snapshots(
        result.output, exit_code=result.exit_code, truncated=result.truncated
    )


def changed_snapshots(
    before: Sequence[GitSnapshot], after: Sequence[GitSnapshot]
) -> list[GitSnapshot]:
    """Compare successful observations, including newly created checkouts.

    Returns:
        Changed snapshots in discovery order, excluding unchanged preexisting dirt.
    """
    previous = {snapshot.root: snapshot for snapshot in before}
    return [
        snapshot
        for snapshot in after
        if (old := previous.get(snapshot.root)) is None
        or (old.branch, old.commit, old.fingerprint)
        != (snapshot.branch, snapshot.commit, snapshot.fingerprint)
    ]
