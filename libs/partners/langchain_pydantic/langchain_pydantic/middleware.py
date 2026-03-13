"""Middleware for providing a Monty-backed repl tool to an agent.

AbstractOS implementation adapted from Monty code.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any

import pydantic_monty
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime  # noqa: TC002
from langchain_core.tools import BaseTool, StructuredTool
from pydantic_monty import AbstractOS, ResourceLimits, StatResult

from langchain_pydantic._foreign_function_docs import (
    format_foreign_function_docs,
    render_foreign_function_section,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import BACKEND_TYPES


class _MontyOS(AbstractOS):
    """Adapt a Deep Agents backend to the filesystem interface expected by Monty."""

    def __init__(self, backend: BackendProtocol) -> None:
        """Store the backend used for filesystem lookups and reads."""
        self._backend = backend

    def path_exists(self, path: PurePosixPath) -> bool:
        """Return whether the path exists according to the backend."""
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
        """Return whether the path resolves to a file."""
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
        """Return whether the path resolves to a directory."""
        p = str(path)
        if p == "/":
            return True
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                return bool(info.get("is_dir", False))
        return bool(self._backend.ls_info(p))

    def path_is_symlink(self, path: PurePosixPath) -> bool:  # noqa: ARG002
        """Report that symlinks are unsupported by this backend adapter."""
        return False

    def path_read_text(self, path: PurePosixPath) -> str:
        """Read a UTF-8 text file through the backend."""
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content.decode("utf-8")

    def path_read_bytes(self, path: PurePosixPath) -> bytes:
        """Read a binary file through the backend."""
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content

    def path_write_text(self, path: PurePosixPath, data: str) -> int:
        """Write UTF-8 text through the backend."""
        self._backend.write(str(path), data)
        return len(data)

    def path_write_bytes(self, path: PurePosixPath, data: bytes) -> int:
        """Write binary content through the backend."""
        self._backend.upload_files([(str(path), data)])
        return len(data)

    def path_mkdir(
        self,
        path: PurePosixPath,  # noqa: ARG002
        parents: bool,  # noqa: FBT001, ARG002
        exist_ok: bool,  # noqa: FBT001
    ) -> None:
        """Validate the requested mkdir behavior supported by this adapter."""
        if not exist_ok:
            msg = "mkdir with exist_ok=False is not supported"
            raise NotImplementedError(msg)

    def path_unlink(self, path: PurePosixPath) -> None:
        """Reject unlink operations because the backend adapter is read/write only."""
        msg = "unlink is not supported"
        raise NotImplementedError(msg)

    def path_rmdir(self, path: PurePosixPath) -> None:
        """Reject directory removal because the backend adapter does not support it."""
        msg = "rmdir is not supported"
        raise NotImplementedError(msg)

    def path_iterdir(self, path: PurePosixPath) -> list[PurePosixPath]:
        """List directory entries using backend metadata."""
        infos = self._backend.ls_info(str(path))
        return [PurePosixPath(info["path"]) for info in infos]

    def path_stat(self, path: PurePosixPath) -> StatResult:
        """Build a synthetic stat result from backend metadata."""
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
        """Reject rename operations because the backend adapter does not support them.

        Args:
            path: Source path requested by Monty.
            target: Destination path requested by Monty.
        """
        msg = "rename is not supported"
        raise NotImplementedError(msg)

    def path_resolve(self, path: PurePosixPath) -> str:
        """Return the backend path unchanged because no symlink resolution is needed."""
        return str(path)

    def path_absolute(self, path: PurePosixPath) -> str:
        """Normalize relative paths into absolute POSIX paths."""
        p = str(path)
        if p.startswith("/"):
            return p
        return "/" + p

    def getenv(self, key: str, default: str | None = None) -> str | None:  # noqa: ARG002
        """Return the provided default because the adapter exposes no environment."""
        return default

    def get_environ(self) -> dict[str, str]:
        """Return an empty environment mapping for Monty execution."""
        return {}


REPL_TOOL_DESCRIPTION = """Evaluates code using a Monty-backed Python-like REPL.

CRITICAL RULES:
- The REPL does NOT retain state between calls. Each `repl` invocation starts from scratch.
- Prefer solving the task in a single `repl` call whenever possible.
- Values are NOT surfaced automatically. If you need to see any result, you must `print(...)` it.
- A bare final expression is not enough. Use `print(value)`.

Syntax and limitations:
- Supports a limited subset of Python syntax (basic expressions, variables, for-loops, if/else).
- No Python standard library (for example, `math.sin` is not available) unless equivalent foreign functions are provided.
- For file access, use `pathlib` (do not use `open`) and do not use context managers.
- If a function is shown as `def ...`, call it normally. If it is shown as `async def ...`, use `await`.
- If a function returns structured data like `list[...]` or typed records, inspect that data directly in the same program.
- You can use lists, indexing, loops, and conditionals inside one program to inspect result sizes, choose items, and branch on the number of results.
- Do not split work across multiple `repl` calls just to confirm intermediate IDs, names, or other simple values. Compute them and keep using them in the same program.
- Usually you should `print(...)` only the final answer, not intermediate IDs.

Examples:
```python
user_id = get_id()
location_id = get_location(user_id)
print(location_id)
```

```python
items = search("alice")
first = items[0]
print(first["id"])
```

```python
result = await fetch_value("weather")
print(result)
```
{external_functions_section}
"""  # noqa: E501  # preserve prompt text formatting exactly for the model

REPL_SYSTEM_PROMPT = """## REPL tool

You have access to a `repl` tool.

Use it when the task can be solved by writing a small self-contained program.

CRITICAL RULES:
- The REPL does NOT retain state between calls. Each `repl` invocation starts from scratch.
- Prefer solving the task in a single `repl` call whenever possible.
- Do as much useful work as possible in one program when it is practical and clear to do so.
- Values are NOT surfaced automatically. If you need to observe any result, you must `print(...)` it.
- A bare final expression like `city` is not enough; use `print(city)`.

Guidance:
- Store intermediate results in variables and continue the computation in the same program.
- If one function's output can be used directly by later code, do that in the same `repl` call instead of making another `repl` call.
- When several independent lookups are needed, group them into the same program and complete as much of the reasoning as possible before returning.
- If a function returns structured data like `list[...]` or typed records, inspect that data directly in the same program.
- You can use lists, indexing, loops, and conditionals inside one program to inspect result sizes, pick items, and change logic based on the number of results.
- Do not make a separate `repl` call just to confirm an intermediate ID, location, name, list item, or other simple lookup result.
- If you can compute an intermediate value and immediately use it, you should usually keep that work in the same `repl` call.
- Usually you should `print(...)` only the final answer, not intermediate IDs.
- The REPL supports a limited subset of Python syntax (basic expressions, variables, for-loops, if/else).
- The Python standard library is not available unless equivalent foreign functions are provided.
- If a function is shown as `def ...`, call it normally. If it is shown as `async def ...`, use `await`.
- For file access, use `pathlib` (do not use `open`) and do not use context managers.

Bad pattern:
```python
location_id = get_location(user_id)
print(location_id)
```
Then making another `repl` call to use `location_id`.

Better pattern:
```python
location_id = get_location(user_id)
city = get_city(location_id)
time = get_time(location_id)
weather = get_weather(location_id)
print(city, time, weather)
```

Examples:
```python
user_id = get_id()
location_id = get_location(user_id)
city = get_city(location_id)
print(city)
```

```python
items = search("alice")
first = items[0]
print(first["id"])
```

```python
result = await fetch_value("weather")
print(result)
```
{external_functions_section}
"""  # noqa: E501  # preserve prompt text formatting exactly for the model


class MontyMiddleware(AgentMiddleware[AgentState[Any], ContextT, ResponseT]):
    """Provide a Monty-backed `repl` tool to an agent."""

    def __init__(  # noqa: PLR0913  # middleware config needs several independent options
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        inputs: list[str] | None = None,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any] | BaseTool]
        | None = None,
        auto_include: bool = False,
        type_check: bool = False,
        type_check_stubs: str | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            backend: Backend to use for filesystem operations from within Monty.
                If omitted, the middleware resolves a backend from the tool runtime.
            inputs: Optional stdin lines available to the script.
            external_functions: Names of external functions allowed by Monty.
            external_function_implementations: Implementations for allowed
                external functions. Values may be plain callables or LangChain
                tools.
            auto_include: Whether to automatically include function
                signatures and docstrings for foreign functions in the
                system prompt.
            type_check: Whether to enable Monty's type checking.
            type_check_stubs: Optional stubs to use when type checking.
        """
        self.backend = backend
        self._script_name: str = "repl.py"
        self._inputs = inputs
        self._external_functions = external_functions
        self._external_function_implementations = external_function_implementations
        self._auto_include = auto_include
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self.tools = [self._create_repl_tool()]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol | None:
        """Resolve the backend from configuration or the current tool runtime.

        Args:
            runtime: Tool runtime for the current repl invocation.

        Returns:
            The resolved backend, or `None` when no backend is available.
        """
        if callable(self.backend):
            return self.backend(runtime)
        if self.backend is not None:
            return self.backend

        state = runtime.state
        if isinstance(state, dict):
            backend = state.get("backend")
            if isinstance(backend, BackendProtocol):
                return backend
        return None

    def _format_foreign_function_docs(self, name: str) -> str | None:
        """Render a compact signature and docstring block for a foreign function."""
        if not self._auto_include:
            return None
        if not self._external_function_implementations:
            return None
        implementation = self._external_function_implementations.get(name)
        if implementation is None:
            return None
        return format_foreign_function_docs(name, implementation)

    def _format_repl_system_prompt(self) -> str:
        """Build the system prompt fragment describing the repl tool."""
        external_functions_section = self._format_external_functions_section()
        return REPL_SYSTEM_PROMPT.format(
            external_functions_section=external_functions_section
        )

    def _format_external_functions_section(self) -> str:
        """Build the optional prompt section describing foreign functions."""
        external_functions = self._external_functions or []
        if not external_functions:
            return ""

        if not self._auto_include or not self._external_function_implementations:
            formatted_functions = "\n".join(f"- {name}" for name in external_functions)
            return f"\n\nAvailable foreign functions:\n{formatted_functions}"

        implementations = {
            name: implementation
            for name in external_functions
            if (implementation := self._external_function_implementations.get(name))
            is not None
        }
        if not implementations:
            formatted_functions = "\n".join(f"- {name}" for name in external_functions)
            return f"\n\nAvailable foreign functions:\n{formatted_functions}"
        return f"\n\n{render_foreign_function_section(implementations)}"

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject REPL usage instructions into the system message."""
        repl_prompt = self._format_repl_system_prompt()
        new_system_message = append_to_system_message(
            request.system_message, repl_prompt
        )
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
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Async wrap model call to inject REPL instructions into system prompt."""
        modified_request = self.modify_request(request)
        return await handler(modified_request)

    def _build_tool_payload(
        self, tool: BaseTool, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str | dict[str, Any]:
        """Convert Monty call arguments into a payload suitable for a LangChain tool."""
        if kwargs:
            return kwargs
        if len(args) == 1 and isinstance(args[0], str | dict):
            return args[0]

        input_schema = tool.get_input_schema()
        fields = list(getattr(input_schema, "__annotations__", {}))
        if len(args) == 1 and len(fields) == 1:
            return {fields[0]: args[0]}
        if len(args) == len(fields) and fields:
            return dict(zip(fields, args, strict=False))
        return {"args": list(args)}

    def _build_external_functions(self) -> dict[str, Callable[..., Any]] | None:
        """Normalize foreign implementations into plain sync callables for Monty.

        Returns:
            A mapping of function names to plain callables Monty can invoke,
            or `None` when no foreign functions were configured.
        """
        if not self._external_function_implementations:
            return None

        external_functions: dict[str, Callable[..., Any]] = {}
        for name, implementation in self._external_function_implementations.items():
            if isinstance(implementation, BaseTool):

                def tool_wrapper(
                    *args: Any, _tool: BaseTool = implementation, **kwargs: Any
                ) -> Any:
                    payload = self._build_tool_payload(_tool, args, kwargs)
                    return _tool.invoke(payload)

                external_functions[name] = tool_wrapper
            else:
                external_functions[name] = implementation
        return external_functions

    def _build_external_functions_async(
        self,
    ) -> dict[str, Callable[..., Awaitable[Any]] | Callable[..., Any]] | None:
        """Normalize foreign implementations for async Monty execution.

        Returns:
            A mapping of function names to async-compatible callables Monty can
            invoke, or `None` when no foreign functions were configured.
        """
        if not self._external_function_implementations:
            return None

        external_functions: dict[
            str, Callable[..., Awaitable[Any]] | Callable[..., Any]
        ] = {}
        for name, implementation in self._external_function_implementations.items():
            if isinstance(implementation, BaseTool):

                async def tool_wrapper(
                    *args: Any, _tool: BaseTool = implementation, **kwargs: Any
                ) -> Any:
                    payload = self._build_tool_payload(_tool, args, kwargs)
                    return await _tool.ainvoke(payload)

                external_functions[name] = tool_wrapper
            else:
                external_functions[name] = implementation
        return external_functions

    def _create_repl_tool(self) -> BaseTool:  # noqa: C901  # tool adapter needs sync/async setup in one place
        """Create the LangChain tool wrapper around Monty execution."""

        def _run_monty(
            code: str,
            *,
            timeout: int | None,
            runtime: ToolRuntime[None, dict[str, Any]],
        ) -> str:
            """Execute a single Monty program and return captured stdout."""
            limits = ResourceLimits()
            if timeout is not None:
                if timeout <= 0:
                    return f"Error: timeout must be positive, got {timeout}."
                limits["max_duration_secs"] = timeout

            resolved_backend = self._get_backend(runtime)
            os_access = (
                _MontyOS(resolved_backend) if resolved_backend is not None else None
            )

            try:
                m = pydantic_monty.Monty(
                    code,
                    inputs=self._inputs or [],
                    script_name=self._script_name,
                    type_check=self._type_check,
                    type_check_stubs=self._type_check_stubs,
                )
            except pydantic_monty.MontyError as e:
                return str(e)

            result: list[str] = []

            def print_callback(stream: Any, value: Any) -> None:  # noqa: ARG001
                """Collect printed output emitted by the Monty program."""
                result.append(str(value))

            try:
                m.run(
                    os=os_access,
                    limits=limits,
                    external_functions=self._build_external_functions(),
                    print_callback=print_callback,
                )
            except pydantic_monty.MontyError as e:
                return str(e)
            return "\n".join(result).rstrip()

        async def _arun_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, dict[str, Any]],
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            """Execute a single Monty program using Monty's async Python API."""
            limits: ResourceLimits = ResourceLimits()
            if timeout is not None:
                if timeout <= 0:
                    return f"Error: timeout must be positive, got {timeout}."
                limits["max_duration_secs"] = timeout

            resolved_backend = self._get_backend(runtime)
            os_access = (
                _MontyOS(resolved_backend) if resolved_backend is not None else None
            )

            try:
                m = pydantic_monty.Monty(
                    code,
                    inputs=self._inputs or [],
                    script_name=self._script_name,
                    type_check=self._type_check,
                    type_check_stubs=self._type_check_stubs,
                )
            except pydantic_monty.MontyError as e:
                return str(e)

            result: list[str] = []

            def print_callback(stream: Any, value: Any) -> None:  # noqa: ARG001
                """Collect printed output emitted by the Monty program."""
                result.append(str(value))

            try:
                await pydantic_monty.run_monty_async(
                    m,
                    os=os_access,
                    limits=limits,
                    external_functions=self._build_external_functions_async(),
                    print_callback=print_callback,
                )
            except pydantic_monty.MontyError as e:
                return str(e)

            return "\n".join(result).rstrip()

        def _sync_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, dict[str, Any]],
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            """Sync wrapper around the shared Monty execution helper."""
            return _run_monty(code, timeout=timeout, runtime=runtime)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_monty,
            coroutine=_arun_monty,
        )
