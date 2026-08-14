"""Read-only inspection tools for the `/goal` rubric grader.

This module deliberately omits `from __future__ import annotations`. The
grader's tools reuse the SDK filesystem tools' `args_schema`, so LangChain
resolves each wrapper's injected `ToolRuntime` from the function's annotations
rather than from that schema. Under postponed evaluation those annotations are
plain strings, the runtime is not recognized as injected, and it is stripped
during input validation - leaving the tool call to fail with a missing
`runtime` argument. Keeping annotations concrete here confines the constraint to
this module instead of `agent.py`, which relies on postponed evaluation for its
`TYPE_CHECKING`-only imports.
"""

import inspect
import logging
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from typing import Any, cast

from deepagents import FsToolName
from deepagents.backends import CompositeBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware import FilesystemMiddleware
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.tools import StructuredTool, tool

from deepagents_code._repository_bounds import (
    REPOSITORY_GREP_MATCH_LIMIT,
    REPOSITORY_TOOL_CALL_LIMIT,
    REPOSITORY_TOOL_NAMES,
    RepositoryBounds,
)

logger = logging.getLogger(__name__)


def _rubric_grader_read_file_prefix(backend: CompositeBackend) -> str:
    """Return the offloaded-results directory the rubric grader is allowed to read.

    Mirrors how `FilesystemMiddleware` derives its large-tool-results prefix from
    the backend's `artifacts_root`, so the grader's read allow-list tracks wherever
    offloaded results actually land (a real per-session `/tmp` dir in local mode,
    or `/large_tool_results/` when `artifacts_root` is the default `/`).

    Args:
        backend: The composite backend the agent uses.

    Returns:
        The large-tool-results prefix, always ending with a trailing slash.
    """
    root = backend.artifacts_root.rstrip("/")
    return f"{root}/large_tool_results/"


def _validate_rubric_grader_read_path(
    file_path: str, read_file_prefix: str
) -> str | None:
    normalized = file_path.replace("\\", "/")
    if not normalized.startswith(read_file_prefix):
        return f"Rubric grader can only read files under {read_file_prefix}."
    parts = PurePosixPath(normalized).parts
    if ".." in parts or "~" in parts:
        return "Invalid path."
    return None


_RUBRIC_GRADER_BUDGET_MESSAGE = (
    "Rubric grader repository inspection limit reached. Decide each remaining "
    "criterion from the evidence already gathered."
)
_RUBRIC_GRADER_NON_TEXT_MESSAGE = (
    "Non-text repository content omitted; the rubric grader supports text results only."
)
_RUBRIC_GRADER_REPOSITORY_TOOL_NAMES: tuple[FsToolName, ...] = (
    "ls",
    "read_file",
    "glob",
    "grep",
)


def _rubric_grader_repository_tool_names(
    fs_tools: Sequence[FsToolName] | None,
) -> list[FsToolName]:
    """Return repository tools allowed for rubric grading.

    Args:
        fs_tools: Parent agent filesystem allowlist, or `None` for all tools.

    Returns:
        The read-only repository tools retained by the parent allowlist.
    """
    if fs_tools is None:
        return list(_RUBRIC_GRADER_REPOSITORY_TOOL_NAMES)
    allowed = frozenset(fs_tools)
    return [name for name in _RUBRIC_GRADER_REPOSITORY_TOOL_NAMES if name in allowed]


def _rubric_grader_repo_call_count(
    runtime: ToolRuntime[None, Any], read_file_prefix: str
) -> int:
    """Count prior working-directory tool results in the current grading run.

    The grader sub-agent is invoked with a fresh message list per grading run,
    so counting repository `ToolMessage`s already present in state naturally
    scopes the budget to the current run without any external counter.

    The grader's `read_file` tool serves both offloaded tool results and
    working-directory files. Only working-directory reads are charged to this
    budget: a `read_file` result is skipped when its originating call targeted
    a path under `read_file_prefix` (an offloaded-result read), so reading many
    offloaded artifacts cannot exhaust the working-directory inspection budget.
    `ls`, `glob`, and `grep` are always working-directory operations. A
    `read_file` result whose originating call cannot be located is counted, so
    the budget fails toward the limit rather than treating an unclassifiable
    read as free.

    Returns:
        The number of working-directory tool results emitted so far this run.
    """
    from langchain_core.messages import (
        AIMessage as LCAIMessage,
        ToolMessage as LCToolMessage,
    )

    state = getattr(runtime, "state", None)
    if isinstance(state, dict):
        messages = state.get("messages") or []
    else:
        messages = getattr(state, "messages", None) or []

    # Map each `read_file` tool-call id to the path it requested so offloaded
    # reads can be told apart from working-directory reads after the fact.
    read_file_paths: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, LCAIMessage):
            continue
        for call in message.tool_calls:
            if call.get("name") != "read_file":
                continue
            call_id = call.get("id")
            file_path = (call.get("args") or {}).get("file_path")
            if isinstance(call_id, str) and isinstance(file_path, str):
                read_file_paths[call_id] = file_path

    count = 0
    for message in messages:
        if not isinstance(message, LCToolMessage):
            continue
        name = getattr(message, "name", None)
        if name not in REPOSITORY_TOOL_NAMES:
            continue
        if name == "read_file":
            requested = read_file_paths.get(getattr(message, "tool_call_id", None))
            if requested is not None and requested.replace("\\", "/").startswith(
                read_file_prefix
            ):
                continue
        count += 1
    return count


def _normalize_rubric_grader_context_tools(
    tools: Sequence[BaseTool | Callable[..., Any]],
) -> list[BaseTool]:
    """Normalize synchronous and asynchronous grader context tools.

    Returns:
        Structured tools that preserve each callable's supported invocation mode.
    """
    normalized: list[BaseTool] = []
    for candidate in tools:
        if isinstance(candidate, BaseTool):
            normalized.append(candidate)
        elif inspect.iscoroutinefunction(candidate):
            normalized.append(StructuredTool.from_function(coroutine=candidate))
        else:
            normalized.append(StructuredTool.from_function(func=candidate))
    return normalized


def _create_rubric_grader_tools(
    backend: CompositeBackend,
    *,
    repository_backend: BackendProtocol | None = None,
    repository_root: str | None = None,
    context_tools: Sequence[BaseTool | Callable[..., Any]] = (),
    fs_tools: Sequence[FsToolName] | None = None,
) -> list[BaseTool]:
    """Build the rubric grader's read-only inspection tools.

    The grader always gets a `read_file` tool for offloaded tool results. When a
    working-directory backend and root are supplied, it also gets `ls`,
    `read_file`, `glob`, and `grep` scoped to that root, bounded identically to
    the goal-criteria agent's repository tools so a single evaluation cannot
    escape the working directory or blow the grader's context budget.

    Args:
        backend: Composite backend used to read offloaded tool results.
        repository_backend: Working-directory backend for repository inspection,
            or `None` to expose only offloaded-result reads.
        repository_root: Absolute root that bounds repository reads.
        context_tools: External read-only tools for checking MCP-backed or web
            resources referenced by the rubric.
        fs_tools: Parent agent filesystem allowlist, or `None` for all tools.
            The grader's working-directory tools are narrowed to this subset so
            `--allow-fs-tools` cannot be bypassed via the rubric grader.

    Returns:
        The grader tool list, with `read_file` first.
    """
    from langchain_core.messages import ToolMessage as LCToolMessage

    repository_tool_names = _rubric_grader_repository_tool_names(fs_tools)

    read_file_prefix = _rubric_grader_read_file_prefix(backend)
    artifact_filesystem = FilesystemMiddleware(
        backend=backend,
        tools=["read_file"],
        tool_token_limit_before_evict=None,
    )
    artifact_tools = {
        candidate.name: candidate for candidate in artifact_filesystem.tools
    }

    def _fs_func(tools_by_name: dict[str, BaseTool], name: str) -> Callable[..., Any]:
        candidate = cast("StructuredTool | None", tools_by_name.get(name))
        if candidate is None or candidate.func is None:
            msg = f"SDK {name} tool is unavailable."
            raise RuntimeError(msg)
        return candidate.func

    artifact_read_file = cast("StructuredTool", artifact_tools["read_file"])
    artifact_read_file_func = _fs_func(artifact_tools, "read_file")

    bounds: RepositoryBounds | None = None
    repository_tools: dict[str, BaseTool] = {}
    if (
        repository_backend is not None
        and repository_root is not None
        and repository_tool_names
    ):
        try:
            bounds = RepositoryBounds(repository_backend, root=repository_root)
        except ValueError:
            logger.warning(
                "Invalid rubric grader repository root %r; disabling "
                "working-directory inspection",
                repository_root,
            )
        if bounds is not None:
            # `FilesystemMiddleware` always requires `read_file`, so include it
            # even when the parent allowlist excludes it; the working-directory
            # `read_file` tool is only *exposed* to the grader (below) when the
            # allowlist actually permits it.
            filesystem_tool_names = list(repository_tool_names)
            if "read_file" not in filesystem_tool_names:
                filesystem_tool_names.append("read_file")
            repository_filesystem = FilesystemMiddleware(
                backend=repository_backend,
                tools=filesystem_tool_names,
                grep_max_count=REPOSITORY_GREP_MATCH_LIMIT,
                tool_token_limit_before_evict=None,
            )
            repository_tools = {
                candidate.name: candidate for candidate in repository_filesystem.tools
            }
    repository_read_file_func = (
        _fs_func(repository_tools, "read_file")
        if bounds is not None and "read_file" in repository_tool_names
        else None
    )

    def _bound(active: RepositoryBounds, name: str, result: object) -> object:
        if isinstance(result, LCToolMessage):
            if isinstance(result.content, str):
                return result.model_copy(
                    update={"content": active.bound_text(name, result.content)}
                )
            return _RUBRIC_GRADER_NON_TEXT_MESSAGE
        if isinstance(result, str):
            return active.bound_text(name, result)
        return _RUBRIC_GRADER_NON_TEXT_MESSAGE

    @tool(
        description=artifact_read_file.description,
        args_schema=artifact_read_file.args_schema,
    )
    def read_file(
        file_path: str,
        runtime: ToolRuntime[None, Any],
        offset: int = 0,
        limit: int = 100,
    ) -> object:
        """Read an offloaded tool result or a working-directory file.

        Returns:
            The tool result, or an error message when the path is outside the
            grader's allowed directories or the inspection budget is exhausted.
        """
        normalized = file_path.replace("\\", "/")
        if normalized.startswith(read_file_prefix):
            if error := _validate_rubric_grader_read_path(file_path, read_file_prefix):
                return error
            return artifact_read_file_func(
                file_path=file_path,
                runtime=runtime,
                offset=offset,
                limit=limit,
            )
        if bounds is None or repository_read_file_func is None:
            return f"Rubric grader can only read files under {read_file_prefix}."
        if (
            _rubric_grader_repo_call_count(runtime, read_file_prefix)
            >= REPOSITORY_TOOL_CALL_LIMIT
        ):
            return _RUBRIC_GRADER_BUDGET_MESSAGE
        args: dict[str, Any] = {"file_path": file_path, "limit": limit}
        if error := bounds.preflight("read_file", args):
            return error
        clamped = bounds.clamp_args("read_file", args)
        return _bound(
            bounds,
            "read_file",
            repository_read_file_func(
                file_path=file_path,
                runtime=runtime,
                offset=offset,
                limit=clamped["limit"],
            ),
        )

    normalized_context_tools = _normalize_rubric_grader_context_tools(context_tools)

    def _with_context_tools(grader_tools: list[BaseTool]) -> list[BaseTool]:
        reserved_names = {"GraderResponse", *(tool.name for tool in grader_tools)}
        conflicts: list[str] = []
        for context_tool in normalized_context_tools:
            if context_tool.name in reserved_names:
                conflicts.append(context_tool.name)
            reserved_names.add(context_tool.name)
        if conflicts:
            names = ", ".join(sorted(set(conflicts)))
            msg = f"Context tool names conflict with rubric-grader tools: {names}."
            raise ValueError(msg)
        return [*grader_tools, *normalized_context_tools]

    grader_tools: list[BaseTool] = [read_file]
    if bounds is None:
        return _with_context_tools(grader_tools)

    # `bounds` is available: expose whichever working-directory search tools the
    # parent allowlist permits. `read_file`'s working-directory branch is gated
    # separately (above) on the allowlist including `read_file`, so `ls`,
    # `glob`, and `grep` remain available even when `read_file` is excluded.
    active_bounds = bounds

    repository_wrapper_tools: list[BaseTool] = []

    if "ls" in repository_tools:
        fs_ls = cast("StructuredTool", repository_tools["ls"])
        fs_ls_func = _fs_func(repository_tools, "ls")

        @tool(
            description=fs_ls.description,
            args_schema=fs_ls.args_schema,
        )
        def ls(path: str, runtime: ToolRuntime[None, Any]) -> object:
            """List a working-directory path to verify criteria against files.

            Returns:
                The bounded listing, or an error message when the path is
                disallowed or the inspection budget is exhausted.
            """
            if (
                _rubric_grader_repo_call_count(runtime, read_file_prefix)
                >= REPOSITORY_TOOL_CALL_LIMIT
            ):
                return _RUBRIC_GRADER_BUDGET_MESSAGE
            args: dict[str, Any] = {"path": path}
            if error := active_bounds.preflight("ls", args):
                return error
            return _bound(active_bounds, "ls", fs_ls_func(path=path, runtime=runtime))

        ls.name = "ls"
        repository_wrapper_tools.append(ls)

    if "glob" in repository_tools:
        fs_glob = cast("StructuredTool", repository_tools["glob"])
        fs_glob_func = _fs_func(repository_tools, "glob")

        @tool(
            description=fs_glob.description,
            args_schema=fs_glob.args_schema,
        )
        def glob(
            pattern: str,
            runtime: ToolRuntime[None, Any],
            path: str | None = None,
        ) -> object:
            """Find working-directory files matching a glob pattern.

            Returns:
                The bounded matches, or an error message when the path/pattern
                is disallowed or the inspection budget is exhausted.
            """
            if (
                _rubric_grader_repo_call_count(runtime, read_file_prefix)
                >= REPOSITORY_TOOL_CALL_LIMIT
            ):
                return _RUBRIC_GRADER_BUDGET_MESSAGE
            args: dict[str, Any] = {"pattern": pattern}
            if path is not None:
                args["path"] = path
            if error := active_bounds.preflight("glob", args):
                return error
            clamped = active_bounds.clamp_args("glob", args)
            return _bound(
                active_bounds,
                "glob",
                fs_glob_func(
                    pattern=pattern, runtime=runtime, path=clamped.get("path")
                ),
            )

        glob.name = "glob"
        repository_wrapper_tools.append(glob)

    if "grep" in repository_tools:
        fs_grep = cast("StructuredTool", repository_tools["grep"])
        fs_grep_func = _fs_func(repository_tools, "grep")

        @tool(
            description=fs_grep.description,
            args_schema=fs_grep.args_schema,
        )
        def grep(
            pattern: str,
            runtime: ToolRuntime[None, Any],
            path: str | None = None,
            glob: str | None = None,
            output_mode: str = "files_with_matches",
            max_count: int | None = None,
        ) -> object:
            """Search working-directory file contents to verify criteria.

            Returns:
                The bounded search output, or an error message when the
                path/pattern is disallowed or the inspection budget is
                exhausted.
            """
            if (
                _rubric_grader_repo_call_count(runtime, read_file_prefix)
                >= REPOSITORY_TOOL_CALL_LIMIT
            ):
                return _RUBRIC_GRADER_BUDGET_MESSAGE
            args: dict[str, Any] = {"pattern": pattern}
            if path is not None:
                args["path"] = path
            if glob is not None:
                args["glob"] = glob
            if max_count is not None:
                args["max_count"] = max_count
            if error := active_bounds.preflight("grep", args):
                return error
            clamped = active_bounds.clamp_args("grep", args)
            return _bound(
                active_bounds,
                "grep",
                fs_grep_func(
                    pattern=pattern,
                    runtime=runtime,
                    path=clamped.get("path"),
                    glob=glob,
                    output_mode=output_mode,
                    max_count=clamped.get("max_count"),
                ),
            )

        grep.name = "grep"
        repository_wrapper_tools.append(grep)

    grader_tools.extend(repository_wrapper_tools)
    return _with_context_tools(grader_tools)
