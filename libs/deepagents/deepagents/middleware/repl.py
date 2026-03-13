"""Middleware for providing a Monty-backed repl tool to an agent."""

from __future__ import annotations

import contextlib
import inspect
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any

import pydantic_monty
from langchain.agents.middleware.types import AgentMiddleware, ContextT, ModelRequest, ModelResponse, ResponseT
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.tools import BaseTool, StructuredTool
from pydantic_monty import AbstractOS, ResourceLimits, StatResult

from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol  # noqa: TC001
from deepagents.middleware._utils import append_to_system_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _MontyOS(AbstractOS):
    def __init__(self, backend: BackendProtocol) -> None:
        """Init with abckend."""
        self._backend = backend

    def path_exists(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return True
        infos = self._backend.ls_info(p)
        if infos:
            return True
        parent = str(path.parent)
        if parent == p:
            return False
        parent_infos = self._backend.ls_info(parent)
        return any(info.get("path") == p for info in parent_infos)

    def path_is_file(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return False
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                return not bool(info.get("is_dir", False))
        res = self._backend.download_files([p])[0]
        return res.error is None and res.content is not None

    def path_is_dir(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return True
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                return bool(info.get("is_dir", False))
        return bool(self._backend.ls_info(p))

    def path_is_symlink(self, path: PurePosixPath) -> bool:  # noqa: ARG002
        return False

    def path_read_text(self, path: PurePosixPath) -> str:
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content.decode("utf-8")

    def path_read_bytes(self, path: PurePosixPath) -> bytes:
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content

    def path_write_text(self, path: PurePosixPath, data: str) -> None:
        self._backend.write(str(path), data)

    def path_write_bytes(self, path: PurePosixPath, data: bytes) -> None:
        self._backend.upload_files([(str(path), data)])

    def path_mkdir(
        self,
        path: PurePosixPath,  # noqa: ARG002
        parents: bool,  # noqa: FBT001, ARG002
        exist_ok: bool,  # noqa: FBT001
    ) -> None:
        if not exist_ok:
            msg = "mkdir with exist_ok=False is not supported"
            raise NotImplementedError(msg)

    def path_unlink(self, path: PurePosixPath) -> None:
        msg = "unlink is not supported"
        raise NotImplementedError(msg)

    def path_rmdir(self, path: PurePosixPath) -> None:
        msg = "rmdir is not supported"
        raise NotImplementedError(msg)

    def path_iterdir(self, path: PurePosixPath) -> list[PurePosixPath]:
        infos = self._backend.ls_info(str(path))
        return [PurePosixPath(info["path"]) for info in infos]

    def path_stat(self, path: PurePosixPath) -> StatResult:
        p = str(path)
        if p == "/":
            return StatResult.dir_stat(0o755, 0.0)
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                if info.get("is_dir", False):
                    return StatResult.dir_stat(0o755, 0.0)
                size = int(info.get("size", 0))
                return StatResult.file_stat(size, 0o644, 0.0)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return StatResult.file_stat(len(res.content), 0o644, 0.0)

    def path_rename(self, path: PurePosixPath, target: PurePosixPath) -> None:
        msg = "rename is not supported"
        raise NotImplementedError(msg)

    def path_resolve(self, path: PurePosixPath) -> str:
        return str(path)

    def path_absolute(self, path: PurePosixPath) -> str:
        p = str(path)
        if p.startswith("/"):
            return p
        return "/" + p

    def getenv(self, key: str, default: str | None = None) -> str | None:  # noqa: ARG002
        return default

    def get_environ(self) -> dict[str, str]:
        return {}


REPL_TOOL_DESCRIPTION = """Evaluates code using a Monty-backed Python-like REPL.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, imports, or helper functions from prior `repl` calls are available.

Capabilities and limitations:
- Very limited subset of Python syntax (basic expressions, for-loops, if/else).
- No Python standard library (e.g., `math.sin` is not available). Use an equivalent foreign function if provided.
- For file access, use `pathlib` (do not use `open`), and do not use context managers (they are not supported).

Available foreign functions:
{external_functions}
"""

REPL_SYSTEM_PROMPT = """## REPL tool

You have access to a `repl` tool.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, imports, or helper functions from prior `repl` calls are available.

- The REPL supports a very limited subset of Python (roughly Python syntax), but does NOT provide the Python standard library.
  For example, you cannot use `math.sin`.
  If you need functionality that would normally come from the standard library, use an equivalent foreign function if one has been provided.
- If you want file access, use `pathlib` (do not use `open`), and do not use context managers (they are not supported).
- Use it for small computations, control flow (for-loops, if/else), and calling externally registered foreign functions.

Available foreign functions:
{external_functions}
"""


class MontyMiddleware(AgentMiddleware[dict[str, Any], ContextT, ResponseT]):
    """Provide a Monty-backed `repl` tool to an agent."""

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        script_name: str = "repl.py",
        inputs: list[str] | None = None,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any]] | None = None,
        auto_include: bool = False,
        type_check: bool = False,
        type_check_stubs: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            backend: Backend to use for filesystem operations from within Monty.
            script_name: The script name Monty should report in tracebacks.
            inputs: Optional stdin lines available to the script.
            external_functions: Names of external functions allowed by Monty.
            external_function_implementations: Implementations for allowed external functions.
            auto_include: Whether to automatically include function signatures and docstrings
                for foreign functions in the system prompt.
            type_check: Whether to enable Monty's type checking.
            type_check_stubs: Optional stubs to use when type checking.
        """
        self.backend = backend
        self._script_name = script_name
        self._inputs = inputs
        self._external_functions = external_functions
        self._external_function_implementations = external_function_implementations
        self._auto_include = auto_include
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self.tools = [self._create_repl_tool()]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Get a backend."""
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _format_foreign_function_docs(self, name: str) -> str | None:
        if not self._auto_include:
            return None
        if not self._external_function_implementations:
            return None
        func = self._external_function_implementations.get(name)
        if func is None:
            return None

        signature = "(…)"
        with contextlib.suppress(TypeError, ValueError):
            signature = str(inspect.signature(func))

        doc = inspect.getdoc(func)
        if not doc:
            return f"`{name}{signature}`"

        doc_lines = doc.splitlines()
        max_doc_lines = 10
        if len(doc_lines) > max_doc_lines:
            doc_lines = [*doc_lines[:max_doc_lines], "..."]
        truncated_doc = "\n".join(doc_lines)
        return f"`{name}{signature}`\n{truncated_doc}"

    def _format_repl_system_prompt(self) -> str:
        external_functions = self._external_functions or []
        if not external_functions:
            formatted_functions = "- (none)"
        else:
            formatted_lines: list[str] = []
            for name in external_functions:
                docs = self._format_foreign_function_docs(name)
                if docs is None:
                    formatted_lines.append(f"- {name}")
                else:
                    indented_docs = docs.replace("\n", "\n  ")
                    formatted_lines.append(f"- {indented_docs}")
            formatted_functions = "\n".join(formatted_lines)

        return REPL_SYSTEM_PROMPT.format(external_functions=formatted_functions)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject REPL usage instructions into the system message."""
        repl_prompt = self._format_repl_system_prompt()
        new_system_message = append_to_system_message(request.system_message, repl_prompt)
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model call to inject REPL instructions into system prompt."""
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async wrap model call to inject REPL instructions into system prompt."""
        modified_request = self.modify_request(request)
        return await handler(modified_request)

    def _create_repl_tool(self) -> BaseTool:
        def _run_monty(
            code: str,
            *,
            timeout: int | None,
            runtime: ToolRuntime[None, dict[str, Any]],
        ) -> str:
            limits = ResourceLimits()
            if timeout is not None:
                if timeout <= 0:
                    return f"Error: timeout must be positive, got {timeout}."
                limits["max_duration_secs"] = timeout

            resolved_backend = self._get_backend(runtime)

            try:
                m = pydantic_monty.Monty(
                    code,
                    inputs=self._inputs or [],
                    external_functions=self._external_functions or [],
                    script_name=self._script_name,
                    type_check=self._type_check,
                    type_check_stubs=self._type_check_stubs,
                )
            except pydantic_monty.MontyError as e:
                return str(e)

            result = []

            def print_callback(stream, value):
                """"""
                result.append(str(value))

            try:
                m.run(
                    os=_MontyOS(resolved_backend),
                    limits=limits,
                    external_functions=self._external_function_implementations,
                    print_callback=print_callback,
                )
            except pydantic_monty.MontyError as e:
                return str(e)
            return str("\n".join(result))

        async def _arun_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, dict[str, Any]],
            timeout: Annotated[int | None, "Optional timeout in seconds for this evaluation."] = None,  # noqa: ASYNC109
        ) -> str:
            return _run_monty(code, timeout=timeout, runtime=runtime)

        def _sync_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, dict[str, Any]],
            timeout: Annotated[int | None, "Optional timeout in seconds for this evaluation."] = None,
        ) -> str:
            return _run_monty(code, timeout=timeout, runtime=runtime)

        formatted_functions = self._format_repl_system_prompt().split("Available foreign functions:\n", 1)[1].rstrip()
        tool_description = REPL_TOOL_DESCRIPTION.format(external_functions=formatted_functions)

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_monty,
            coroutine=_arun_monty,
        )
