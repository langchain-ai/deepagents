# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import base64
import logging
import posixpath
import re
import shlex
import threading
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional, Dict, List, Union, Tuple, Any, cast

from k8s_agent_sandbox import SandboxClient
from k8s_agent_sandbox.sandbox import Sandbox

try:
    from deepagents.backends.protocol import (
        EditResult,
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
    )
    from deepagents.backends.utils import create_file_data
except (ImportError, ModuleNotFoundError) as exc:
    raise ImportError(
        "deepagents>=0.5.0 is required to use langchain-agent-sandbox"
    ) from exc

try:
    from langgraph._internal._constants import CONFIG_KEY_SEND
    from langgraph.config import get_config
except (ImportError, ModuleNotFoundError):
    CONFIG_KEY_SEND = None  # type: ignore[assignment]
    get_config = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# `requests` is a transitive dependency of `k8s_agent_sandbox`, so the
# import is always satisfiable when this module is importable, but we
# guard it defensively so a future SDK refactor that drops requests
# doesn't break our own import.
try:
    from requests.exceptions import Timeout as _RequestsTimeout
except ImportError:  # pragma: no cover - defensive
    _RequestsTimeout = None  # type: ignore[assignment]

# `httpx` is used by the SDK's async transport. Imported defensively
# so the timeout walker also catches `httpx.TimeoutException` if a
# future code path surfaces an async-transport exception synchronously.
try:
    from httpx import TimeoutException as _HttpxTimeout
except ImportError:  # pragma: no cover - defensive
    _HttpxTimeout = None  # type: ignore[assignment]


def _is_timeout_exception(exc: BaseException) -> bool:
    """Return True if ``exc`` (or any chained cause/context) is a timeout.

    Walks ``__cause__``/``__context__`` so wrapped exceptions are
    detected even when the backend doesn't import the wrapper class
    directly. Three layers of detection, in order:

    1. The Python builtin ``TimeoutError``.
    2. Known transport-level timeout types (``requests.exceptions.Timeout``,
       ``httpx.TimeoutException``) if installed.
    3. A duck-typed fallback on the exception class name (``"timeout"``
       substring) — catches future SDK exception classes that don't
       chain via ``raise ... from e`` and that we haven't seen yet.
       This is a soft signal but strictly better than the hard miss
       that bare ``except TimeoutError`` produced before.

    The duck-typed fallback can produce false positives if a non-timeout
    exception class happens to contain "timeout" in its name. The
    tradeoff is acceptable: ``execute()`` already classifies any
    unhandled exception as a generic failure with ``exit_code=-1``,
    so escalating it to ``exit_code=-2`` (timeout) for an exception
    named ``"FooTimeoutPolicyError"`` is a strictly better signal than
    flattening it to a generic error.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, TimeoutError):
            return True
        if _RequestsTimeout is not None and isinstance(cur, _RequestsTimeout):
            return True
        if _HttpxTimeout is not None and isinstance(cur, _HttpxTimeout):
            return True
        # Duck-typed fallback for exceptions that don't chain via
        # `from e` (background workers, asyncio.to_thread boundaries,
        # explicit `raise X() from None`, future SDK exception types).
        if "timeout" in type(cur).__name__.lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _compile_glob(pattern: str) -> Callable[[str], bool]:
    """Compile a glob pattern into a matcher that correctly handles ``**``.

    ``pathlib.PurePath.match`` in Python 3.11 treats ``**`` as two literal
    ``*`` wildcards (equivalent to a single ``*``), which means a pattern
    like ``**/langchain_e2e.txt`` fails to match ``langchain_e2e.txt`` at
    the root of the search. Agent-driven glob calls routinely write
    ``**/...`` patterns expecting recursive semantics, so we translate
    the pattern to a regex ourselves.

    Supported syntax:
        ``**``  -- zero or more path components (greedy, slashes allowed)
        ``*``   -- any characters except ``/``
        ``?``   -- any single character except ``/``
        ``[..]``-- character class (passed through to the regex engine)

    ``**/X`` matches ``X`` at any depth including the root (equivalent
    to ``X`` OR ``*/X`` OR ``*/*/X`` ...). Patterns that contain no
    ``/`` are treated as a basename match -- ``*.py`` finds every
    ``.py`` file in the search tree, mirroring ``pathlib.PurePath.match``
    semantics that existing callers rely on.
    """
    # A pattern without any `/` matches against the basename only. This
    # preserves the existing behavior of calls like `glob("*.py")` which
    # walkers interpret as "any .py file in the search tree", and it
    # matches what `pathlib.PurePath.match` does for non-path patterns.
    if "/" not in pattern:
        segment_regex = _translate_glob_segment(pattern)
        compiled_basename = re.compile("^" + segment_regex + "$")
        return lambda path: compiled_basename.fullmatch(posixpath.basename(path)) is not None

    # Split the pattern on `/` into segments. Collapse consecutive `**`
    # segments since `**/**` is semantically identical to a single `**`.
    segments = pattern.split("/")
    collapsed: List[str] = []
    for seg in segments:
        if seg == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(seg)
    segments = collapsed

    # Pattern is just `**` (after collapsing) → matches anything.
    if len(segments) == 1 and segments[0] == "**":
        return lambda path: True

    # Build the regex piece by piece. The key invariant: `**` always
    # consumes its surrounding slashes, never adds a spurious one.
    # That means a middle `**` produces `/(?:[^/]+/)*` so `a/**/b`
    # requires at least `a/b` (zero intermediate components) and rejects
    # `ab`. Leading `**` produces `(?:[^/]+/)*` (zero or more full
    # components followed by a slash, then the next literal segment
    # follows without an extra slash). Trailing `**` produces `/.*` so
    # `a/**` matches `a/`, `a/b`, and `a/b/c`, but NOT bare `a` —
    # mirroring the gitignore semantic that "x/**" means "things inside
    # x/".
    regex_parts: List[str] = []
    for i, seg in enumerate(segments):
        is_first = i == 0
        is_last = i == len(segments) - 1

        if seg == "**":
            if is_first:
                # Leading: emit zero-or-more components with trailing /.
                # The next literal segment will NOT get a prepended `/`.
                regex_parts.append("(?:[^/]+/)*")
            elif is_last:
                # Trailing: require a slash plus anything (possibly
                # empty). `a/**` matches `a/`, `a/b`, `a/b/c` but not
                # bare `a` itself, mirroring the gitignore semantic
                # that "x/**" means "things inside x/".
                regex_parts.append("/.*")
            else:
                # Middle: require the literal slash that was part of
                # the pattern, plus zero or more components with their
                # own trailing slashes. The next literal segment will
                # NOT get a prepended `/` because this block ends in
                # one already.
                regex_parts.append("/(?:[^/]+/)*")
            continue

        # Literal segment (possibly with `*`, `?`, `[...]` wildcards).
        # Emit a `/` separator unless this is the first segment or the
        # previous segment was a `**` token that already produced a
        # trailing slash.
        if not is_first:
            prev = segments[i - 1]
            if prev != "**":
                regex_parts.append("/")
            # else: previous `**` token already ends with `/`, don't
            # emit a second one.
        regex_parts.append(_translate_glob_segment(seg))

    regex_str = "^" + "".join(regex_parts) + "$"
    compiled = re.compile(regex_str)
    return lambda path: compiled.fullmatch(path) is not None


def _translate_glob_segment(segment: str) -> str:
    """Translate one path segment (no `/`) into a regex fragment."""
    out: List[str] = []
    i = 0
    while i < len(segment):
        c = segment[i]
        if c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = segment.find("]", i + 1)
            if end == -1:
                out.append(re.escape(c))
                i += 1
            else:
                out.append(segment[i : end + 1])
                i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


class AgentSandboxBackend(SandboxBackendProtocol):
    """DeepAgents backend adapter for agent-sandbox runtimes.

    Implements the DeepAgents SandboxBackendProtocol by wrapping a Sandbox handle.
    All file operations are virtualized under `root_dir` (default: /app).

    Requirements:
        - Sandbox image must have GNU coreutils: sh, grep, find (with -printf), mkdir, test
        - Sandbox handle must be connected before use

    Note:
        The backend can optionally manage the sandbox lifecycle when created via from_template().
        When manage_lifecycle=True, the backend must be used as a context manager.
    """

    SESSION_LABEL_KEY = "agent-sandbox.sigs.k8s.io/session-id"
    """Kubernetes label key used for session-based sandbox reattach."""

    def __init__(
        self,
        sandbox: Sandbox,
        root_dir: str = "/app",
        manage_lifecycle: bool = False,
        allow_absolute_paths: bool = False,
        sdk_client: Optional[SandboxClient] = None,
        runtime_root: str = "/app",
        _template: Optional[str] = None,
        _namespace: str = "default",
        _sandbox_ready_timeout: int = 180,
        _labels: Optional[Dict[str, str]] = None,
        _shutdown_after_seconds: Optional[int] = None,
        _session_id: Optional[str] = None,
        _default_timeout_seconds: Optional[int] = None,
    ) -> None:
        if not root_dir.startswith("/"):
            raise ValueError(f"root_dir must be an absolute path, got: {root_dir}")
        if not runtime_root.startswith("/"):
            raise ValueError(f"runtime_root must be an absolute path, got: {runtime_root}")
        self._sandbox: Optional[Sandbox] = sandbox
        self._root_dir = posixpath.normpath(root_dir)
        # Path the sandbox runtime treats as its filesystem base — the
        # SDK file APIs (``files.read`` / ``files.list`` / ``files.exists``
        # / ``files.write``) interpret their path argument as relative to
        # this directory. Default ``/app`` matches the example runtime.
        self._runtime_root = posixpath.normpath(runtime_root)
        self._manage_lifecycle = manage_lifecycle
        self._allow_absolute_paths = allow_absolute_paths
        self._sdk_client = sdk_client
        self._template = _template
        self._namespace = _namespace
        self._sandbox_ready_timeout = _sandbox_ready_timeout
        self._labels = _labels
        self._shutdown_after_seconds = _shutdown_after_seconds
        self._session_id = _session_id
        self._default_timeout_seconds = _default_timeout_seconds
        self._reattached = False
        self._draining = False
        self._inflight = 0
        self._inflight_cv = threading.Condition(threading.Lock())

    def _send_files_update(self, update: dict[str, Any]) -> None:
        """Queue a ``files`` channel write via LangGraph's CONFIG_KEY_SEND.

        Follows the same contract as ``StateBackend._send_files_update`` so
        that ``state["files"]`` stays in sync with the sandbox filesystem.
        Silently no-ops when called outside a LangGraph graph context (e.g.
        unit tests, standalone scripts).
        """
        if get_config is None or CONFIG_KEY_SEND is None:
            return
        try:
            config = get_config()
        except RuntimeError:
            return
        send = config.get("configurable", {}).get(CONFIG_KEY_SEND)
        if send is None:
            return
        send([("files", update)])

    def _try_reattach(self) -> bool:
        """Try to reattach to an existing sandbox by session_id label.

        Returns True if a sandbox was found and reattached, False otherwise.

        Safety invariants:

        - Multi-match is refused, not silently resolved. In a shared
          namespace another tenant could plant a claim with a matching
          ``session-id`` label; picking ``claims[0]`` would hand them
          this session's subsequent reads/writes/commands. Similarly,
          stale claims from a prior run can mask a newer one if we pick
          arbitrarily. When ``len(claims) > 1`` we log loudly and treat
          the session as unresolvable so the caller creates a fresh
          sandbox instead of attaching to the wrong one.
        - Template is verified via ``get_sandbox(template_name=...)``
          (see J1). A ``ValueError`` from that check is a signed-off
          refusal — re-raise it instead of swallowing to ``False``,
          otherwise a mismatched claim silently falls through to a
          fresh create and the agent loses continuity without any
          operator-visible signal.
        """
        assert self._sdk_client is not None
        if self._session_id is None:
            return False
        selector = f"{self.SESSION_LABEL_KEY}={self._session_id}"
        try:
            claims = self._sdk_client.list_all_sandboxes(
                namespace=self._namespace, label_selector=selector,
            )
        except Exception as e:
            logger.warning(
                "Session lookup failed for session_id=%s: %s",
                self._session_id, e,
            )
            return False
        if not claims:
            return False
        if len(claims) > 1:
            logger.error(
                "Refusing to reattach: %d claims match session_id=%s (%s). "
                "Resolve by deleting the stale claim(s) before retrying.",
                len(claims), self._session_id, claims,
            )
            return False
        claim_name = claims[0]
        try:
            self._sandbox = self._sdk_client.get_sandbox(
                claim_name=claim_name,
                namespace=self._namespace,
                template_name=self._template,
            )
            self._reattached = True
            logger.info(
                "Reattached to existing sandbox '%s' for session '%s'",
                claim_name, self._session_id,
            )
            return True
        except ValueError:
            # Template mismatch — a claim with our session-id label
            # exists but was created from a different template. Do NOT
            # swallow: surface to the caller so they see the security-
            # relevant refusal instead of a silent fresh-create.
            raise
        except Exception as e:
            logger.warning(
                "Reattach to '%s' failed for session '%s': %s",
                claim_name, self._session_id, e,
            )
            return False

    @contextmanager
    def _track_op(self):
        """Track an in-flight operation for drain-on-close.

        ``__exit__`` sets ``_draining`` and waits for ``_inflight`` to
        reach zero before deleting the sandbox. This prevents claim
        deletion while an ``execute()`` or file operation is mid-flight
        — particularly relevant for the factory pattern where
        ``weakref.finalize`` can fire from GC at any time.
        """
        with self._inflight_cv:
            if self._draining:
                raise RuntimeError("Backend is shutting down")
            self._inflight += 1
        try:
            yield
        finally:
            with self._inflight_cv:
                self._inflight -= 1
                if self._inflight == 0:
                    self._inflight_cv.notify_all()

    def __enter__(self) -> "AgentSandboxBackend":
        if self._manage_lifecycle:
            if self._sdk_client is None:
                raise RuntimeError(
                    "Cannot manage lifecycle without an sdk_client. "
                    "Use from_template() to create a managed backend."
                )
            self._reattached = False
            self._draining = False
            if not self._try_reattach():
                create_kwargs: Dict[str, Any] = {
                    "template": self._template,
                    "namespace": self._namespace,
                    "sandbox_ready_timeout": self._sandbox_ready_timeout,
                }
                # Merge session_id into labels so the sandbox can be
                # found again by future invocations with the same session_id.
                labels = dict(self._labels) if self._labels else {}
                if self._session_id is not None:
                    labels[self.SESSION_LABEL_KEY] = self._session_id
                if labels:
                    create_kwargs["labels"] = labels
                if self._shutdown_after_seconds is not None:
                    create_kwargs["shutdown_after_seconds"] = self._shutdown_after_seconds
                self._sandbox = self._sdk_client.create_sandbox(**create_kwargs)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._manage_lifecycle and self._sandbox is not None:
            # Signal draining and wait for in-flight ops to finish
            # before deleting the claim. Mirrors Go SDK's Close()
            # which sets draining + waits on inflightOps.
            with self._inflight_cv:
                self._draining = True
                while self._inflight > 0:
                    self._inflight_cv.wait()
            if self._reattached:
                self._sandbox = None
                return
            claim: str = self._sandbox.claim_name
            ns: str = self._sandbox.namespace
            cleanup_error: Optional[BaseException] = None
            assert self._sdk_client is not None
            try:
                self._sdk_client.delete_sandbox(claim_name=claim, namespace=ns)
            except Exception as e:
                cleanup_error = e
                logger.error(
                    "Failed to delete sandbox (claim=%s, namespace=%s): %s",
                    claim, ns, e,
                )
            finally:
                self._sandbox = None
            if cleanup_error is not None:
                if exc_type is None:
                    raise cleanup_error
                if exc is not None:
                    raise BaseExceptionGroup(
                        "sandbox cleanup failed during exception unwind",
                        [exc, cleanup_error],
                    ) from None
                import warnings
                warnings.warn(
                    f"sandbox cleanup failed during exception unwind: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                    ResourceWarning,
                    stacklevel=2,
                )

    async def __aenter__(self) -> "AgentSandboxBackend":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.__exit__(exc_type, exc, tb)

    @classmethod
    def from_existing(
        cls,
        sandbox: Sandbox,
        root_dir: str = "/app",
        allow_absolute_paths: bool = False,
    ) -> "AgentSandboxBackend":
        """Wrap an existing, already-connected Sandbox handle.

        Use this when the caller manages the sandbox lifecycle externally
        (e.g. via ``SandboxClient.create_sandbox()`` or
        ``SandboxClient.get_sandbox()``).  The backend will **not**
        create or delete the sandbox on enter/exit.

        Args:
            sandbox: A connected Sandbox instance.
            root_dir: Virtual root for file operations.
            allow_absolute_paths: If True, write/upload operations may
                target absolute paths outside root_dir.

        Returns:
            AgentSandboxBackend in unmanaged mode.
        """
        return cls(
            sandbox=sandbox,
            root_dir=root_dir,
            manage_lifecycle=False,
            allow_absolute_paths=allow_absolute_paths,
        )

    @classmethod
    def from_template(
        cls,
        client: SandboxClient,
        template_name: str,
        namespace: str = "default",
        root_dir: str = "/app",
        allow_absolute_paths: bool = False,
        sandbox_ready_timeout: int = 180,
        labels: Optional[Dict[str, str]] = None,
        shutdown_after_seconds: Optional[int] = None,
        session_id: Optional[str] = None,
        default_timeout_seconds: Optional[int] = 120,
    ) -> "AgentSandboxBackend":
        """Create a backend with lifecycle management via SandboxClient.

        The sandbox will be automatically created on ``__enter__`` and
        deleted on ``__exit__`` when the backend is used as a context manager.

        When ``session_id`` is provided, ``__enter__`` will first search
        for an existing SandboxClaim labelled with
        ``agent-sandbox.sigs.k8s.io/session-id=<session_id>`` in the
        given namespace.  If found, the backend reattaches to that
        sandbox (via ``SandboxClient.get_sandbox``).  If not found, a
        new sandbox is created with that label.  This enables
        multi-turn agents to resume work in the same persistent sandbox
        across invocations — leveraging agent-sandbox's stable identity
        and persistent storage.

        Args:
            client: Pre-configured ``SandboxClient``. Connection mode
                (gateway, tunnel, direct) and tracing are configured on
                the client — the adapter does not expose transport details.
            template_name: Name of the SandboxTemplate to claim.
            namespace: Kubernetes namespace for the sandbox.
            root_dir: Virtual root for file operations.
            allow_absolute_paths: If True, write/upload operations may
                target absolute paths outside root_dir.
            sandbox_ready_timeout: Timeout in seconds waiting for sandbox
                readiness.
            labels: Kubernetes labels to attach to the SandboxClaim.
            shutdown_after_seconds: Time-to-live in seconds. After this
                duration the controller auto-deletes the claim.
            session_id: Stable session identifier for sandbox reuse.
                When set, ``__enter__`` looks up an existing sandbox
                by label before creating a new one.
            default_timeout_seconds: Default timeout for ``execute()``
                when the caller doesn't pass an explicit ``timeout``.
                Set to ``None`` to use the SDK's own default. The
                default of 120 s is conservative — the sandbox-router
                proxy times out at 180 s in Gateway/Tunnel mode.

        Returns:
            AgentSandboxBackend with manage_lifecycle=True.
        """
        return cls(
            sandbox=None,  # type: ignore[arg-type]
            root_dir=root_dir,
            manage_lifecycle=True,
            allow_absolute_paths=allow_absolute_paths,
            sdk_client=client,
            _template=template_name,
            _namespace=namespace,
            _sandbox_ready_timeout=sandbox_ready_timeout,
            _labels=labels,
            _shutdown_after_seconds=shutdown_after_seconds,
            _session_id=session_id,
            _default_timeout_seconds=default_timeout_seconds,
        )

    def _assert_sandbox(self) -> Sandbox:
        """Return the sandbox handle or raise if not initialized.

        Guards every public method that talks to the sandbox so that a
        forgotten ``__enter__`` on a ``from_template()`` backend produces
        a clear ``RuntimeError`` instead of an opaque
        ``AttributeError: 'NoneType' object has no attribute 'commands'``
        buried inside a generic error result.
        """
        if self._sandbox is None:
            raise RuntimeError(
                "Sandbox is not initialized. Use 'with backend:' "
                "(context manager) or call '__enter__()' first."
            )
        return self._sandbox

    def execute(
        self,
        command: str,
        *,
        timeout: Optional[int] = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox.

        The command runs with its working directory set to ``root_dir``
        so that relative paths in shell commands resolve consistently
        with the paths exposed by ``ls`` / ``read`` / ``write``.

        Args:
            command: Shell command to execute.
            timeout: Maximum time in seconds to wait for the command to complete.
                If None, no explicit timeout is passed and the underlying
                sandbox client's default applies.

        Returns:
            ExecuteResponse with combined stdout/stderr output and exit code.
            On timeout, exit_code=-2 and output is prefixed with "Timed out".
            On other failures, exit_code=-1.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            wrapped = f"sh -c {shlex.quote(f'cd {self._root_dir} && {command}')}"
            effective_timeout = timeout if timeout is not None else self._default_timeout_seconds
            try:
                if effective_timeout is not None:
                    result = sandbox.commands.run(wrapped, timeout=effective_timeout)
                else:
                    result = sandbox.commands.run(wrapped)
            except Exception as e:
                if _is_timeout_exception(e):
                    logger.warning("execute timed out for command: %s", e)
                    return ExecuteResponse(
                        output=f"Timed out: {e}",
                        exit_code=-2,
                        truncated=False,
                    )
                logger.error("execute failed: %s", e)
                return ExecuteResponse(
                    output=f"Error: {e}",
                    exit_code=-1,
                    truncated=False,
                )
            combined = result.stdout
            if result.stderr:
                combined = f"{combined}\n{result.stderr}" if combined else result.stderr
            return ExecuteResponse(
                output=combined,
                exit_code=result.exit_code,
                truncated=False,
            )

    def ls(self, path: str) -> LsResult:
        """List directory contents.

        Uses the sandbox runtime's native ``/list/{path}`` endpoint which
        returns structured JSON (name, size, type, mod_time). This
        populates the ``size`` and ``modified_at`` optional fields on
        ``FileInfo`` that the ``SandboxBackendProtocol`` supports.

        Args:
            path: Directory path to list.

        Returns:
            LsResult with entries on success, or ``error`` populated on
            failure. Entries are always an empty list when ``error`` is
            set, so callers iterating over ``entries`` without checking
            ``error`` still work.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._to_internal(path)
                file_entries = sandbox.files.list(self._to_runtime_relative(internal_path))
            except Exception as e:
                logger.warning("ls failed for path '%s': %s", path, e)
                return LsResult(entries=[], error=f"Cannot list '{path}': {e}")

            entries: List[FileInfo] = []
            public_dir = self._normalize_public_dir(path)
            for fe in file_entries:
                name = fe.name
                if name in (".", ".."):
                    continue
                is_dir = fe.type == "directory"
                public_path = posixpath.join(public_dir, name)
                info = FileInfo(path=public_path, is_dir=is_dir)
                if fe.size is not None:
                    info["size"] = fe.size
                if fe.mod_time is not None:
                    info["modified_at"] = datetime.fromtimestamp(
                        fe.mod_time, tz=timezone.utc,
                    ).isoformat()
                entries.append(info)

            entries.sort(key=lambda item: item["path"])
            return LsResult(entries=entries)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        """Read file content for the requested line range.

        Returns raw (unformatted) content — the middleware handles
        line-number formatting and long-line chunking. Content is decoded
        strictly as UTF-8; non-UTF-8 files are reported as an error so the
        caller isn't handed lossy data mislabeled as utf-8.

        Args:
            file_path: Path to the file to read.
            offset: 0-based line index to start reading from.
            limit: Maximum number of lines to return.

        Returns:
            ReadResult with file_data on success, or `error` populated on
            read failure, decode failure, or an out-of-range offset on a
            non-empty file. Empty files always succeed with empty content
            regardless of offset.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._to_internal(file_path)
                content = self._read_bytes_at(internal_path)
            except Exception as e:
                logger.warning("Failed to read file '%s': %s", file_path, e)
                return ReadResult(error=f"Failed to read '{file_path}': {e}")

            try:
                decoded = content.decode("utf-8")
            except UnicodeDecodeError as e:
                logger.warning("Failed to decode '%s' as UTF-8: %s", file_path, e)
                return ReadResult(
                    error=f"Cannot decode '{file_path}' as UTF-8: {e}"
                )

            lines = decoded.splitlines()
            if not lines:
                return ReadResult(file_data=FileData(content="", encoding="utf-8"))

            start = max(0, offset)
            if start >= len(lines):
                return ReadResult(
                    error=f"Line offset {offset} exceeds file length ({len(lines)} lines)"
                )
            end = min(len(lines), start + limit)
            selected = "\n".join(lines[start:end])
            return ReadResult(file_data=FileData(content=selected, encoding="utf-8"))

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        """Search for a literal text pattern in files.

        Args:
            pattern: Fixed string pattern to search for.
            path: Base path to search in (default: /).
            glob: Optional glob pattern to filter files (e.g., "*.py").

        Returns:
            GrepResult with matches on success or error message on failure.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            base_path = path or "/"
            internal_path = self._to_internal(base_path)

            grep_opts = "-rHnFZ"
            parts = ["grep", grep_opts]
            if glob:
                parts.append(f"--include={shlex.quote(glob)}")
            parts.extend(["-e", shlex.quote(pattern), shlex.quote(internal_path)])
            command = " ".join(parts)
            try:
                run_kwargs: Dict[str, Any] = {}
                if self._default_timeout_seconds is not None:
                    run_kwargs["timeout"] = self._default_timeout_seconds
                result = sandbox.commands.run(command, **run_kwargs)
            except Exception as e:
                logger.warning("grep failed in '%s': %s", base_path, e)
                return GrepResult(matches=[], error=f"grep failed in '{base_path}': {e}")
            if result.exit_code >= 2:
                detail = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"exit code {result.exit_code}"
                )
                return GrepResult(
                    matches=[],
                    error=f"grep failed in '{base_path}': {detail}",
                )
            if not result.stdout.strip():
                return GrepResult(matches=[])

            matches: List[GrepMatch] = []
            for line in result.stdout.splitlines():
                nul_pos = line.find("\0")
                if nul_pos < 0:
                    parts = line.split(":", 2)
                    if len(parts) != 3:
                        logger.debug("Skipping malformed grep output line: %s", line)
                        continue
                    raw_path, line_no, text = parts
                else:
                    raw_path = line[:nul_pos]
                    remainder = line[nul_pos + 1:]
                    parts = remainder.split(":", 1)
                    if len(parts) != 2:
                        logger.debug("Skipping malformed grep output line: %s", line)
                        continue
                    line_no, text = parts
                public_path = self._to_public(raw_path)
                try:
                    line_int = int(line_no)
                except ValueError:
                    logger.debug("Skipping grep line with invalid line number: %s", line)
                    continue
                matches.append(GrepMatch(path=public_path, line=line_int, text=text))

            return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern to match (e.g., "*.py", "**/*.txt").
            path: Base path to search in.

        Returns:
            GlobResult with matching entries on success, or `error` populated
            on total failure. On partial failures (e.g. permission denied on
            a subdirectory), available results are still returned without an
            error so callers can proceed with what was accessible.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            internal_path = self._to_internal(path)

            normalized_pattern = pattern.lstrip("/")
            try:
                matcher = _compile_glob(normalized_pattern)
            except re.error as e:
                return GlobResult(
                    matches=[],
                    error=f"invalid glob pattern {pattern!r}: {e}",
                )
            except Exception as e:
                logger.error(
                    "glob matcher compilation failed for pattern %r: %s: %s",
                    pattern, type(e).__name__, e,
                )
                return GlobResult(
                    matches=[],
                    error=(
                        f"internal error compiling glob pattern {pattern!r}: "
                        f"{type(e).__name__}: {e}"
                    ),
                )

            # NUL-delimited fields guard against whitespace or newlines in
            # pathnames — splitting on spaces breaks if any component
            # contains one, silently dropping or mis-attributing matches.
            # Record format: "<type>\\t<size>\\t<mtime>\\t<path>\\0"
            command = (
                f"find -L {shlex.quote(internal_path)} -mindepth 1"
                f" \\( -type d -printf 'd\\t%s\\t%T@\\t%p\\0' \\)"
                f" -o \\( -printf 'f\\t%s\\t%T@\\t%p\\0' \\)"
            )
            try:
                run_kwargs: Dict[str, Any] = {}
                if self._default_timeout_seconds is not None:
                    run_kwargs["timeout"] = self._default_timeout_seconds
                result = sandbox.commands.run(command, **run_kwargs)
            except Exception as e:
                logger.warning("glob failed for path '%s': %s", path, e)
                return GlobResult(matches=[], error=f"glob failed in '{path}': {e}")
            if result.exit_code != 0:
                detail = result.stderr.strip() or f"exit code {result.exit_code}"
                logger.warning(
                    "glob partial failure for path '%s': exit_code=%d, stderr=%s",
                    path, result.exit_code, result.stderr
                )
                if not result.stdout:
                    return GlobResult(matches=[], error=f"glob failed in '{path}': {detail}")

            entries: List[FileInfo] = []
            for record in result.stdout.split("\x00"):
                if not record:
                    continue
                parts = record.split("\t", 3)
                if len(parts) != 4:
                    logger.debug("glob: malformed record %r", record)
                    continue
                type_char, size_str, mod_str, raw = parts
                if type_char not in ("d", "f"):
                    continue
                rel_path = posixpath.relpath(raw, internal_path)
                if matcher(rel_path):
                    public_path = self._to_public(raw)
                    info = FileInfo(path=public_path, is_dir=(type_char == "d"))
                    try:
                        info["size"] = int(size_str)
                    except ValueError:
                        logger.debug("glob: unparseable size %r for %s", size_str, raw)
                    try:
                        info["modified_at"] = datetime.fromtimestamp(
                            float(mod_str), tz=timezone.utc,
                        ).isoformat()
                    except (ValueError, OSError):
                        logger.debug("glob: unparseable mod_time %r for %s", mod_str, raw)
                    entries.append(info)

            entries.sort(key=lambda item: item["path"])
            return GlobResult(matches=entries)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file with the given content.

        Args:
            file_path: Path for the new file.
            content: Content to write.

        Returns:
            WriteResult with error=None on success, or error message on failure.
            Fails if file already exists.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._resolve_write_path(file_path)
            except ValueError as e:
                return WriteResult(
                    error=f"Error: Invalid path '{file_path}': {e}",
                    path=file_path,
                )

            try:
                if self._path_exists_at(internal_path):
                    return WriteResult(
                        error=f"File '{file_path}' already exists",
                        path=file_path,
                    )
                self._ensure_parent_dir(internal_path)
                payload = content.encode("utf-8") if isinstance(content, str) else content
                self._write_bytes_at(internal_path, payload)
            except Exception as e:
                logger.error("write failed for '%s': %s", file_path, e)
                return WriteResult(
                    error=f"Error writing '{file_path}': {e}",
                    path=file_path,
                )
            self._send_files_update({file_path: create_file_data(content)})
            return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace a string in a file.

        Args:
            file_path: Path to the file to edit.
            old_string: String to find and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences. If False, requires
                         exactly one occurrence.

        Returns:
            EditResult with error=None on success, or error message on failure.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            try:
                internal_path = self._resolve_write_path(file_path)
            except ValueError as e:
                return EditResult(
                    error=f"Error: Invalid path '{file_path}': {e}",
                    path=file_path,
                    occurrences=0,
                )
            try:
                raw = self._read_bytes_at(internal_path)
            except Exception as e:
                logger.warning("Failed to read file '%s' for edit: %s", file_path, e)
                return EditResult(
                    error=f"Error reading '{file_path}': {e}",
                    path=file_path,
                    occurrences=0,
                )

            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as e:
                logger.warning("Failed to decode '%s' as UTF-8 for edit: %s", file_path, e)
                return EditResult(
                    error=f"Error: Cannot edit '{file_path}': not valid UTF-8 ({e})",
                    path=file_path,
                    occurrences=0,
                )
            occurrences = content.count(old_string)
            if occurrences == 0:
                return EditResult(
                    error=f"Error: String not found in file: '{old_string}'",
                    path=file_path,
                    occurrences=0,
                )
            if not replace_all and occurrences > 1:
                return EditResult(
                    error=(
                        f"Error: String '{old_string}' appears multiple times. "
                        "Use replace_all=True to replace all occurrences."
                    ),
                    path=file_path,
                    occurrences=occurrences,
                )

            if replace_all:
                updated = content.replace(old_string, new_string)
            else:
                updated = content.replace(old_string, new_string, 1)
            try:
                self._write_bytes_at(internal_path, updated.encode("utf-8"))
            except Exception as e:
                logger.error("edit upload failed for '%s': %s", file_path, e)
                return EditResult(
                    error=f"Error writing '{file_path}': {e}",
                    path=file_path,
                    occurrences=0,
                )
            self._send_files_update({file_path: create_file_data(updated)})
            return EditResult(
                path=file_path,
                occurrences=occurrences if replace_all else 1,
            )

    def upload_files(
        self, files: Union[Dict[str, bytes], Iterable[Tuple[str, bytes]]]
    ) -> List[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Args:
            files: Dict or iterable of (path, content) pairs.

        Returns:
            List of FileUploadResponse for each file.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            pairs = cast(
                List[Tuple[str, bytes]],
                list(files.items()) if isinstance(files, dict) else list(files),
            )
            responses: List[FileUploadResponse] = []
            for path, payload in pairs:
                try:
                    internal_path = self._resolve_write_path(path)
                except ValueError as e:
                    logger.warning("Invalid path '%s': %s", path, e)
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                    continue

                state = self._file_state(internal_path)
                if state == "error":
                    responses.append(FileUploadResponse(path=path, error=cast(Any, "upload_failed")))
                    continue
                if state == "dir":
                    responses.append(FileUploadResponse(path=path, error="is_directory"))
                    continue
                if state == "denied":
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                    continue

                parent_dir = posixpath.dirname(internal_path)
                parent_state = self._dir_state(parent_dir)
                if parent_state == "error":
                    responses.append(FileUploadResponse(path=path, error=cast(Any, "upload_failed")))
                    continue
                if parent_state == "not_dir":
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                    continue
                if parent_state == "denied":
                    responses.append(FileUploadResponse(path=path, error="permission_denied"))
                    continue
                if parent_state == "missing":
                    try:
                        self._ensure_parent_dir(internal_path)
                    except Exception as e:
                        logger.error(
                            "Failed to create parent dir for upload '%s': %s: %s",
                            path, type(e).__name__, e,
                        )
                        responses.append(FileUploadResponse(path=path, error=cast(Any, "upload_failed")))
                        continue

                try:
                    self._write_bytes_at(internal_path, payload)
                    responses.append(FileUploadResponse(path=path, error=None))
                except Exception as e:
                    logger.error("Upload failed for '%s': %s: %s", path, type(e).__name__, e)
                    responses.append(FileUploadResponse(path=path, error=cast(Any, "upload_failed")))

            # Batch-send state updates for all successfully uploaded files.
            files_update: dict[str, Any] = {}
            for resp, (orig_path, payload) in zip(responses, pairs):
                if resp.error is None:
                    try:
                        text = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        import base64 as _b64
                        text = _b64.b64encode(payload).decode("ascii")
                    files_update[orig_path] = create_file_data(text)
            if files_update:
                self._send_files_update(files_update)

            return responses

    def download_files(self, paths: Iterable[str]) -> List[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Args:
            paths: Iterable of file paths to download.

        Returns:
            List of FileDownloadResponse for each file.
        """
        sandbox = self._assert_sandbox()
        with self._track_op():
            responses: List[FileDownloadResponse] = []
            for path in paths:
                try:
                    internal_path = self._to_internal(path)
                except ValueError as e:
                    logger.warning("Invalid path '%s': %s", path, e)
                    responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))
                    continue
                state = self._file_state(internal_path)
                if state == "error":
                    responses.append(FileDownloadResponse(path=path, content=None, error=cast(Any, "download_failed")))
                    continue
                if state == "missing":
                    responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
                    continue
                if state == "dir":
                    responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
                    continue
                if state == "denied":
                    responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
                    continue
                try:
                    content = self._read_bytes_at(internal_path)
                    responses.append(FileDownloadResponse(path=path, content=content, error=None))
                except Exception as e:
                    logger.error("Download failed for '%s': %s: %s", path, type(e).__name__, e)
                    responses.append(FileDownloadResponse(path=path, content=None, error=cast(Any, "download_failed")))
            return responses

    def _ensure_parent_dir(self, internal_path: str) -> None:
        assert self._sandbox is not None
        parent = posixpath.dirname(internal_path)
        command = shlex.join(["mkdir", "-p", parent])
        result = self._sandbox.commands.run(command)
        if result.exit_code != 0:
            error_msg = result.stderr.strip() if result.stderr else f"mkdir failed with exit code {result.exit_code}"
            raise RuntimeError(f"Cannot create parent directory '{parent}': {error_msg}")

    def _resolve_write_path(self, path: str) -> str:
        """Resolve a write path while allowing absolute paths outside root_dir."""
        candidate = path.strip()
        if not candidate:
            raise ValueError("empty path")
        normalized = posixpath.normpath(candidate)
        if (
            self._allow_absolute_paths
            and normalized.startswith("/")
            and normalized != self._root_dir
            and not normalized.startswith(self._root_dir + "/")
        ):
            return normalized
        return self._to_internal(candidate)

    def _to_internal(self, path: str) -> str:
        """Convert a public virtual path to an internal filesystem path.

        All paths are treated as relative to root_dir after normalization:
        - Leading slashes are stripped: '/file.txt' -> '{root_dir}/file.txt'
        - Paths already containing root_dir prefix are normalized
        - Paths that escape root_dir (.. or ../) raise ValueError

        Args:
            path: Public virtual path.

        Returns:
            Internal filesystem path under root_dir.

        Raises:
            ValueError: If the path escapes root_dir or contains control chars.
        """
        stripped = path.strip() or "/"
        # Reject NUL bytes and ASCII control characters. A NUL byte in a
        # path passes Python's normpath check but gets truncated at the
        # C/syscall layer when the sandbox runtime processes it — so
        # "/app/safe\x00../../etc/passwd" would pass our relpath defense
        # but resolve to "/app/safe" on the filesystem.
        if any(ord(c) < 0x20 for c in stripped):
            raise ValueError(
                f"Path contains ASCII control characters: {path!r}"
            )
        normalized = stripped
        if normalized == self._root_dir or normalized.startswith(self._root_dir + "/"):
            normalized = normalized[len(self._root_dir):]
        normalized = normalized.lstrip("/")
        internal_path = posixpath.normpath(posixpath.join(self._root_dir, normalized))
        rel = posixpath.relpath(internal_path, self._root_dir)
        if rel == ".." or rel.startswith("../"):
            raise ValueError(f"Path '{path}' escapes root_dir '{self._root_dir}'")
        return internal_path

    def _to_relative(self, path: str) -> str:
        internal_path = self._to_internal(path)
        rel = posixpath.relpath(internal_path, self._root_dir)
        return "." if rel == "." else rel

    def _to_runtime_relative(self, internal_path: str) -> str:
        """Convert an internal absolute path to a runtime-root-relative path.

        The sandbox runtime's file APIs (``files.read`` / ``files.list`` /
        ``files.exists`` / ``files.write``) treat the path argument as
        relative to ``runtime_root`` (default ``/app``). This helper
        bridges the gap between our internal absolute paths (rooted under
        ``root_dir``, which may differ from ``runtime_root``) and what the
        runtime expects.

        Raises ValueError if ``internal_path`` escapes ``runtime_root``.
        """
        rel = posixpath.relpath(internal_path, self._runtime_root)
        if rel == ".." or rel.startswith("../"):
            raise ValueError(
                f"Internal path '{internal_path}' is outside runtime_root "
                f"'{self._runtime_root}'"
            )
        return "." if rel == "." else rel

    # ---- Dual-mode file IO helpers --------------------------------------
    #
    # The runtime file API (``sandbox.files.*``) is anchored at
    # ``runtime_root`` and rejects paths outside it (as of the C4 SDK
    # sanitizer). When ``allow_absolute_paths=True`` callers can target
    # paths outside ``runtime_root`` (e.g. /tmp/...); for those we fall
    # back to running shell commands to honor the same operations
    # safely, so the feature keeps working without re-introducing the
    # upload-path-traversal vector.

    def _write_bytes_at(self, internal_path: str, payload: bytes) -> None:
        """Write ``payload`` to ``internal_path`` via file API or shell fallback."""
        assert self._sandbox is not None
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            self._sandbox.files.write(runtime_rel, payload)
            return
        encoded = base64.b64encode(payload).decode("ascii")
        pipeline = (
            f"printf %s {shlex.quote(encoded)} "
            f"| base64 -d > {shlex.quote(internal_path)}"
        )
        command = f"sh -c {shlex.quote(pipeline)}"
        result = self._sandbox.commands.run(command)
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(f"Write to '{internal_path}' failed: {detail}")

    def _read_bytes_at(self, internal_path: str) -> bytes:
        """Read bytes from ``internal_path`` via file API or shell fallback."""
        assert self._sandbox is not None
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            return self._sandbox.files.read(runtime_rel)
        command = f"sh -c {shlex.quote(f'base64 < {shlex.quote(internal_path)}')}"
        result = self._sandbox.commands.run(command)
        if result.exit_code != 0:
            detail = result.stderr.strip() or f"exit code {result.exit_code}"
            raise RuntimeError(f"Read from '{internal_path}' failed: {detail}")
        try:
            return base64.b64decode(result.stdout)
        except Exception as e:
            raise RuntimeError(f"Decode of '{internal_path}' output failed: {e}") from e

    def _path_exists_at(self, internal_path: str) -> bool:
        """Check whether ``internal_path`` exists via file API or shell fallback."""
        assert self._sandbox is not None
        try:
            runtime_rel = self._to_runtime_relative(internal_path)
        except ValueError:
            runtime_rel = None
        if runtime_rel is not None:
            return self._sandbox.files.exists(runtime_rel)
        command = f"sh -c {shlex.quote(f'test -e {shlex.quote(internal_path)}')}"
        result = self._sandbox.commands.run(command)
        return result.exit_code == 0

    def _to_public(self, internal_path: str) -> str:
        rel = posixpath.relpath(internal_path, self._root_dir)
        if rel == ".":
            return "/"
        return "/" + rel

    def _normalize_public_dir(self, path: str) -> str:
        if not path or path == "/":
            return "/"
        if path == self._root_dir or path.startswith(self._root_dir + "/"):
            path = path[len(self._root_dir):]
        return "/" + path.strip("/")

    def _probe_state(
        self, check_script: str, valid: frozenset, label: str,
    ) -> str:
        """Run a shell probe and validate the single-word stdout result."""
        assert self._sandbox is not None
        result = self._sandbox.commands.run(f"sh -c {shlex.quote(check_script)}")
        output = result.stdout.strip()
        if not output and result.exit_code != 0:
            logger.warning(
                "_%s command failed: exit_code=%d, stderr=%s",
                label, result.exit_code, result.stderr,
            )
            return "error"
        state = output or "missing"
        if state not in valid:
            logger.warning("_%s returned unexpected value %r", label, state)
            return "error"
        return state

    _FILE_STATE_VALID = frozenset({"missing", "dir", "file", "denied"})

    def _file_state(self, internal_path: str) -> str:
        """Returns: 'missing', 'dir', 'file', 'denied', or 'error'."""
        check = (
            f"if [ ! -e {shlex.quote(internal_path)} ]; then echo missing; exit 0; fi; "
            f"if [ -d {shlex.quote(internal_path)} ]; then echo dir; exit 0; fi; "
            f"if [ -r {shlex.quote(internal_path)} ]; then echo file; else echo denied; fi"
        )
        return self._probe_state(check, self._FILE_STATE_VALID, "file_state")

    _DIR_STATE_VALID = frozenset({"missing", "writable", "denied", "not_dir"})

    def _dir_state(self, internal_path: str) -> str:
        """Returns: 'missing', 'writable', 'denied', 'not_dir', or 'error'."""
        check = (
            f"if [ ! -e {shlex.quote(internal_path)} ]; then echo missing; exit 0; fi; "
            f"if [ -d {shlex.quote(internal_path)} ]; then "
            f"if [ -w {shlex.quote(internal_path)} ]; then echo writable; else echo denied; fi; "
            f"exit 0; fi; "
            f"echo not_dir"
        )
        return self._probe_state(check, self._DIR_STATE_VALID, "dir_state")

    @staticmethod
    def delete_all(
        client: SandboxClient,
        namespace: str = "default",
        best_effort: bool = True,
        label_selector: Optional[str] = None,
    ) -> int:
        """Delete SandboxClaims in the given namespace.

        When ``label_selector`` is provided, only claims matching the
        selector are deleted — safe for use in shared namespaces.
        Without a selector, **all** claims in the namespace are deleted.

        Args:
            client: Pre-configured ``SandboxClient``.
            namespace: Kubernetes namespace to clean up.
            best_effort: If True (default), log delete failures and
                continue. If False, raise on the first failure.
            label_selector: Optional Kubernetes label selector string
                (e.g. ``"app=myapp"``). When set, only claims matching
                the selector are deleted.

        Returns:
            Number of claims successfully deleted.

        Raises:
            Exception: On delete failure when ``best_effort=False``.
        """
        claims = client.list_all_sandboxes(
            namespace=namespace, label_selector=label_selector,
        )
        deleted = 0
        for claim in claims:
            try:
                client.delete_sandbox(claim_name=claim, namespace=namespace)
                deleted += 1
            except Exception as e:
                if best_effort:
                    logger.warning(
                        "delete_all: failed to delete '%s' in namespace '%s': %s",
                        claim, namespace, e,
                    )
                else:
                    raise
        return deleted

    @property
    def id(self) -> str:
        """Return a namespace-qualified sandbox identifier.

        Format: ``{namespace}/{claim_name}``. Namespace-qualification
        prevents collisions when multiple namespaces are in use.
        Falls back to ``"agent-sandbox"`` before ``__enter__`` is called.
        """
        if self._sandbox is not None:
            ns = getattr(self._sandbox, "namespace", None) or "default"
            claim = getattr(self._sandbox, "claim_name", None)
            if claim:
                return f"{ns}/{claim}"
        return "agent-sandbox"


def create_sandbox_backend_factory(
    template_name: str,
    namespace: str = "default",
    **kwargs: Any,
) -> Callable[[Any], AgentSandboxBackend]:
    """Create a BackendFactory for use with create_deep_agent().

    This factory function returns a callable that creates an AgentSandboxBackend
    when invoked with a ToolRuntime. The ToolRuntime provides state and store,
    but our backend doesn't need them since execution happens in the sandbox.

    The factory eagerly provisions the sandbox before returning, so the
    returned backend is immediately usable for `execute`/`ls`/`read`/etc.
    A finalizer is registered to delete the sandbox when the backend is
    garbage-collected or at interpreter shutdown -- the deepagents
    ``create_deep_agent(backend=...)`` contract uses the factory result
    directly without a ``with`` block, so the lifecycle has to be
    managed transparently rather than via a context manager.

    If you DO want context-manager semantics (eager teardown on exit
    rather than GC-time), use ``AgentSandboxBackend.from_template(...)``
    directly inside a ``with`` block.

    Usage:
        from deepagents import create_deep_agent
        from langchain_agent_sandbox import create_sandbox_backend_factory

        agent = create_deep_agent(
            backend=create_sandbox_backend_factory("my-template")
        )

    Args:
        template_name: Name of the SandboxTemplate to claim.
        namespace: Kubernetes namespace for the sandbox.
        **kwargs: Additional arguments passed to AgentSandboxBackend.from_template().

    Returns:
        A factory callable that accepts a ToolRuntime and returns a fully
        initialized AgentSandboxBackend.
    """
    def factory(_runtime: Any) -> AgentSandboxBackend:
        backend = AgentSandboxBackend.from_template(
            template_name=template_name,
            namespace=namespace,
            **kwargs,
        )
        # Eagerly enter so the sandbox is provisioned and `_sandbox` is
        # populated before the deepagents agent loop starts calling
        # methods on it. Without this, the first execute()/ls()/etc.
        # raises `AttributeError: 'NoneType' object has no attribute
        # 'commands'` because from_template() defers sandbox creation
        # to __enter__.
        backend.__enter__()
        # Register a finalizer that tears the sandbox down when the
        # backend is garbage-collected OR at interpreter shutdown
        # (whichever comes first). weakref.finalize captures strong
        # references to the SDK client and sandbox handle in its
        # closure, so the cleanup can fire even after the backend
        # itself is collected. If the user explicitly calls __exit__
        # on the backend, the finalizer's redundant cleanup attempt
        # 404s and is silently swallowed by `_factory_atexit_cleanup`.
        #
        # The finalize handle is also stored on the backend so it can
        # be invoked deterministically from tests (calling
        # `backend._finalizer()` triggers the cleanup synchronously
        # without relying on `del + gc.collect()` GC ordering).
        backend._finalizer = weakref.finalize(  # type: ignore[attr-defined]
            backend,
            _factory_atexit_cleanup,
            backend._sdk_client,
            backend._sandbox,
        )
        return backend

    return factory


def _factory_atexit_cleanup(sdk_client: Any, sandbox: Any) -> None:
    """Best-effort sandbox teardown for factory-created backends.

    Called by ``weakref.finalize`` when the backend is garbage-collected
    OR at interpreter shutdown. Two failure modes are silenced:

    1. A 404 from the SDK — the user already called ``__exit__``
       explicitly and the claim is already gone. Not actionable.
    2. Any exception while logging — at interpreter shutdown the
       logging subsystem may already be torn down, and a finalizer
       that raises during shutdown produces a confusing traceback.

    Every other failure (auth expiry, network partition, RBAC
    revocation, API outage, malformed claim, transient 5xx) is logged
    at ERROR so operators can see leaked sandboxes. Logging-during-GC
    is safe; the shutdown-only concern is handled by the inner guard.
    """
    if sdk_client is None or sandbox is None:
        return
    claim = getattr(sandbox, "claim_name", None)
    ns = getattr(sandbox, "namespace", None)
    if claim is None:
        return
    try:
        sdk_client.delete_sandbox(claim_name=claim, namespace=ns)
    except Exception as e:
        # 404 means the user's explicit __exit__ already cleaned up;
        # this finalizer is the redundant best-effort path. Status
        # comes from either the kubernetes ApiException shape or the
        # requests Response shape, depending on which transport raised.
        status = getattr(e, "status", None) or getattr(
            getattr(e, "response", None), "status_code", None
        )
        if status == 404:
            return
        # Log the leak so it's at least visible. Wrap in its own try
        # because at interpreter shutdown the logging machinery can
        # be partially torn down — a logging failure inside the
        # finalizer must not propagate.
        try:
            logger.error(
                "Finalizer failed to delete sandbox (claim=%s, namespace=%s): "
                "%s: %s",
                claim, ns, type(e).__name__, e,
            )
        except Exception:
            pass


class SandboxPolicyWrapper(SandboxBackendProtocol):
    """Wraps AgentSandboxBackend with policy enforcement.

    Provides best-effort restrictions on sandbox operations as an
    application-layer guardrail. **This is not a security boundary**
    -- the underlying sandbox controller and kernel-level isolation
    (gVisor, Kata Containers, etc.) are the real security boundary.

    Features:
    - deny_prefixes: Block writes/edits under certain paths
      (e.g., /etc, /sys). Paths are canonicalized before matching so
      traversal-style bypasses like `/app/../etc` are caught.
    - deny_commands: Substring match against execute() commands
      (e.g., "rm -rf"). This is trivially bypassable via aliases or
      shell variations -- treat it as a speed bump, not a barrier.
    - audit_log: Optional callable invoked for every write / edit /
      execute / upload operation. By default, callback failures are
      logged at WARNING and the operation proceeds (fail-open). Set
      ``strict_audit=True`` to refuse operations whose audit log
      cannot be written -- use this when audit completeness matters
      more than availability (e.g. compliance environments).

    Audit log callback signature:
        def audit_log(operation: str, target: str, metadata: dict) -> None:
            # operation: "write", "edit", "execute", "upload"
            # target: file path or command string
            # metadata: {"size": int} for write/upload, {"replace_all": bool} for edit

    Usage:
        backend = AgentSandboxBackend.from_template("my-template")
        wrapped = SandboxPolicyWrapper(
            backend,
            deny_prefixes=["/etc", "/sys", "/proc"],
            deny_commands=["rm -rf", "shutdown", "reboot"],
            audit_log=lambda op, target, meta: print(f"{op}: {target}"),
            strict_audit=False,  # default: fail-open audit
        )
    """

    def __init__(
        self,
        backend: AgentSandboxBackend,
        deny_prefixes: Optional[List[str]] = None,
        deny_commands: Optional[List[str]] = None,
        audit_log: Optional[Callable[[str, str, dict], None]] = None,
        *,
        strict_audit: bool = False,
    ) -> None:
        self._backend = backend
        self._deny_prefixes_write: List[str] = []
        self._deny_prefixes_edit: List[str] = []
        self._deny_prefixes_canonical: List[str] = []
        for prefix in (deny_prefixes or []):
            self._deny_prefixes_write.append(
                self._normalize_prefix(self._resolve_prefix(prefix, for_write=True))
            )
            self._deny_prefixes_edit.append(
                self._normalize_prefix(self._resolve_prefix(prefix, for_write=False))
            )
            self._deny_prefixes_canonical.append(
                self._normalize_prefix(self._canonicalize_path(prefix))
            )
        self._deny_commands = deny_commands or []
        self._audit_log = audit_log
        self._strict_audit = strict_audit

    def __enter__(self) -> "SandboxPolicyWrapper":
        self._backend.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._backend.__exit__(exc_type, exc, tb)

    async def __aenter__(self) -> "SandboxPolicyWrapper":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.__exit__(exc_type, exc, tb)

    def _emit_audit(self, operation: str, target: str, metadata: dict) -> Optional[str]:
        """Invoke the audit log callback if configured.

        Returns ``None`` on success (or when no audit log is set), and
        a deny-reason string when the operation should be refused. In
        ``strict_audit=False`` mode (the default), audit failures are
        logged at WARNING with full operation context (including
        ``exc_info``) and the operation proceeds. In ``strict_audit=True``
        mode, the same failures cause this method to return a refusal
        string that callers convert into a per-operation denied result.
        """
        if self._audit_log is None:
            return None
        try:
            self._audit_log(operation, target, metadata)
            return None
        except Exception as e:
            if self._strict_audit:
                logger.error(
                    "Audit log callback failed for %s on %s "
                    "(strict mode: refusing operation); metadata=%s",
                    operation, target, metadata,
                    exc_info=True,
                )
                return f"Audit log unavailable; refusing {operation}: {e}"
            # Fail-open: include operation/target/metadata so SREs can
            # diagnose audit-pipeline failures without grepping through
            # the request log to find which call wasn't recorded.
            # `exc_info=True` preserves the traceback for non-trivial
            # callback failures (e.g. transient ConnectionError vs.
            # configuration bug).
            logger.warning(
                "Audit log callback failed for %s on %s "
                "(fail-open: operation will proceed); metadata=%s",
                operation, target, metadata,
                exc_info=True,
            )
            return None

    @staticmethod
    def _canonicalize_path(path: str) -> str:
        normalized = posixpath.normpath(path.strip() or "/")
        return "/" + normalized.lstrip("/")

    @staticmethod
    def _normalize_prefix(path: str) -> str:
        canonical_path = SandboxPolicyWrapper._canonicalize_path(path)
        return canonical_path.rstrip("/") + "/" if canonical_path != "/" else "/"

    def _resolve_prefix(self, path: str, for_write: bool) -> str:
        stripped = path.strip() or "/"
        if stripped == "/":
            return "/"
        try:
            if for_write:
                return self._backend._resolve_write_path(stripped)
            return self._backend._to_internal(stripped)
        except ValueError:
            return self._canonicalize_path(stripped)

    def _resolve_candidate_path(self, path: str, for_write: bool) -> str:
        try:
            if for_write:
                return self._backend._resolve_write_path(path)
            return self._backend._to_internal(path)
        except ValueError:
            return self._canonicalize_path(path)

    def _is_denied_path(self, path: str, for_write: bool) -> bool:
        normalized = self._normalize_prefix(self._resolve_candidate_path(path, for_write))
        deny_prefixes = self._deny_prefixes_write if for_write else self._deny_prefixes_edit
        if any(normalized.startswith(prefix) for prefix in deny_prefixes):
            return True

        # Catch traversal-style bypasses (e.g. /app/../etc) using canonical path checks.
        parts = [part for part in path.split("/") if part]
        if ".." in parts:
            canonical = self._normalize_prefix(self._canonicalize_path(path))
            return any(canonical.startswith(prefix) for prefix in self._deny_prefixes_canonical)

        return False

    def _is_denied_command(self, cmd: str) -> bool:
        return any(pattern in cmd for pattern in self._deny_commands)

    def ls(self, path: str) -> LsResult:
        return self._backend.ls(path)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        return self._backend.read(file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        return self._backend.grep(pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return self._backend.glob(pattern, path)

    def download_files(self, paths: Iterable[str]) -> List[FileDownloadResponse]:
        return self._backend.download_files(paths)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write file with policy check."""
        if self._is_denied_path(file_path, for_write=True):
            return WriteResult(
                error=f"Policy denied: writes not allowed under '{file_path}'",
                path=file_path,
            )
        deny = self._emit_audit("write", file_path, {"size": len(content)})
        if deny is not None:
            return WriteResult(error=deny, path=file_path)
        return self._backend.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit file with policy check."""
        if self._is_denied_path(file_path, for_write=False):
            return EditResult(
                error=f"Policy denied: edits not allowed under '{file_path}'",
                path=file_path,
                occurrences=0,
            )
        deny = self._emit_audit("edit", file_path, {"replace_all": replace_all})
        if deny is not None:
            return EditResult(error=deny, path=file_path, occurrences=0)
        return self._backend.edit(file_path, old_string, new_string, replace_all)

    def execute(
        self,
        command: str,
        *,
        timeout: Optional[int] = None,
    ) -> ExecuteResponse:
        """Execute command with policy check."""
        if self._is_denied_command(command):
            return ExecuteResponse(
                output="Policy denied: command blocked by policy",
                exit_code=1,
                truncated=False,
            )
        deny = self._emit_audit("execute", command, {})
        if deny is not None:
            return ExecuteResponse(output=deny, exit_code=1, truncated=False)
        return self._backend.execute(command, timeout=timeout)

    def upload_files(
        self, files: Union[Dict[str, bytes], Iterable[Tuple[str, bytes]]]
    ) -> List[FileUploadResponse]:
        """Upload files with policy check. Preserves input order in responses."""
        pairs = cast(
            List[Tuple[str, bytes]],
            list(files.items()) if isinstance(files, dict) else list(files),
        )
        responses: List[Optional[FileUploadResponse]] = [None] * len(pairs)
        allowed: List[Tuple[int, str, bytes]] = []

        for idx, (path, payload) in enumerate(pairs):
            if self._is_denied_path(path, for_write=True):
                responses[idx] = FileUploadResponse(path=path, error=cast(Any, "policy_denied"))
                continue
            deny = self._emit_audit("upload", path, {"size": len(payload)})
            if deny is not None:
                responses[idx] = FileUploadResponse(path=path, error=cast(Any, deny))
                continue
            allowed.append((idx, path, payload))

        if allowed:
            backend_items = [(path, payload) for _, path, payload in allowed]
            backend_responses = self._backend.upload_files(backend_items)
            for (idx, _, _), resp in zip(allowed, backend_responses):
                responses[idx] = resp

        return responses  # type: ignore[return-value]

    @property
    def id(self) -> str:
        return self._backend.id


