"""Middleware for providing a Monty-backed repl tool to an agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import pydantic_monty
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
from langchain_core.tools.base import (
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)
from pydantic_monty import ResourceLimits

from langchain_monty._foreign_function_docs import render_foreign_function_section

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


REPL_TOOL_DESCRIPTION = """Evaluates code using a Monty-backed Python-like REPL.

CRITICAL RULES:
- The REPL does NOT retain state between calls. Each `repl` invocation starts from scratch.
- Prefer solving the task in a single `repl` call whenever possible.
- Values are NOT surfaced automatically. If you need to see any result, you must `print(...)` it.
- A bare final expression is not enough. Use `print(value)`.

Syntax and limitations:
- Supports a limited subset of Python syntax (basic expressions, variables, for-loops, if/else).
- No Python standard library (for example, `math.sin` is not available) unless equivalent foreign functions are provided.
- If a function is shown as `def ...`, call it normally. If it is shown as `async def ...`, use `await`.
- When several awaited calls are independent, prefer running them in parallel when the REPL supports it to reduce latency.
- If a function returns structured data like `list[...]` or typed records, inspect that data directly in the same program.
- You can use lists, indexing, loops, and conditionals inside one program to inspect result sizes, choose items, and branch on the number of results.
- If the task needs multiple foreign function calls, prefer writing one complete Python program instead of splitting the work across multiple `repl` invocations.
- If one foreign function returns an ID or other value that can be passed directly into the next foreign function, trust it and chain the calls instead of stopping to double-check it.
- If you want to inspect an intermediate value, print it inside the same REPL program; otherwise, try to fetch as much information as possible in one program.
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
- Do NOT assume variables, functions, imports, or helper objects from prior `repl` calls are available.
- Prefer solving the task in a single `repl` call whenever possible.
- Do as much useful work as possible in one program when it is practical and clear to do so.
- Values are NOT surfaced automatically. If you need to observe any result, you must `print(...)` it.
- A bare final expression like `city` is not enough; use `print(city)`.

Guidance:
- Store intermediate results in variables and continue the computation in the same program.
- If one function's output can be used directly by later code, do that in the same `repl` call instead of making another `repl` call.
- When several independent lookups are needed, group them into the same program and complete as much of the reasoning as possible before returning.
- When several awaited calls are independent, prefer running them in parallel when the REPL supports it to reduce latency.
- If a function returns structured data like `list[...]` or typed records, inspect that data directly in the same program.
- You can use lists, indexing, loops, and conditionals inside one program to inspect result sizes, pick items, and change logic based on the number of results.
- If the task needs multiple foreign function calls, prefer writing one complete Python program instead of splitting the work across multiple `repl` invocations.
- If one foreign function returns an ID or other value that can be passed directly into the next foreign function, trust it and chain the calls instead of stopping to double-check it.
- If you want to inspect an intermediate value, print it inside the same REPL program; otherwise, try to fetch as much information as possible in one program.
- Do not make a separate `repl` call just to confirm an intermediate ID, location, name, list item, or other simple lookup result.
- If you can compute an intermediate value and immediately use it, you should usually keep that work in the same `repl` call.
- Usually you should `print(...)` only the final answer, not intermediate IDs.
- The REPL supports a limited subset of Python syntax (basic expressions, variables, for-loops, if/else).
- The Python standard library is not available unless equivalent foreign functions are provided.
- If a function is shown as `def ...`, call it normally. If it is shown as `async def ...`, use `await`.

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

    def __init__(
        self,
        *,
        ptc: list[Callable[..., Any] | BaseTool] | None = None,
        add_ptc_docs: bool = False,
    ) -> None:
        """Initialize the middleware.

        Args:
            ptc: Optional plain callables or LangChain tools exposed as foreign
                functions in the REPL.
            add_ptc_docs: Whether to include rendered signatures and docstrings
                for `ptc` functions in the system prompt.
        """
        self._ptc = ptc or []
        self._add_ptc_docs = add_ptc_docs
        self.tools = [self._create_repl_tool()]

    def _format_repl_system_prompt(self, *, async_context: bool = False) -> str:
        """Build the system prompt fragment describing the repl tool."""
        external_functions_section = self._format_external_functions_section(
            async_context=async_context
        )
        return REPL_SYSTEM_PROMPT.format(
            external_functions_section=external_functions_section
        )

    def _get_ptc_name(self, item: Callable[..., Any] | BaseTool) -> str:
        """Return the exported name for a foreign function or tool."""
        if isinstance(item, BaseTool):
            return item.name
        return getattr(item, "__name__", item.__class__.__name__)

    def _format_external_functions_section(self, *, async_context: bool = False) -> str:
        """Build the optional prompt section describing foreign functions."""
        if not self._ptc:
            return ""

        implementations = {self._get_ptc_name(item): item for item in self._ptc}
        if not self._add_ptc_docs:
            formatted_functions = "\n".join(f"- {name}" for name in implementations)
            return f"\n\nAvailable foreign functions:\n{formatted_functions}"
        return f"\n\n{render_foreign_function_section(implementations, async_context=async_context)}"

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
        repl_prompt = self._format_repl_system_prompt(async_context=True)
        new_system_message = append_to_system_message(
            request.system_message, repl_prompt
        )
        modified_request = request.override(system_message=new_system_message)
        return await handler(modified_request)

    def _get_injected_arg_names(self, tool: BaseTool) -> set[str]:
        return {
            name
            for name, type_ in get_all_basemodel_annotations(
                tool.get_input_schema()
            ).items()
            if _is_injected_arg_type(type_)
        }

    def _get_runtime_arg_name(self, tool: BaseTool) -> str | None:
        if "runtime" in self._get_injected_arg_names(tool):
            return "runtime"
        return None

    def _filter_injected_kwargs(
        self, tool: BaseTool, payload: dict[str, Any]
    ) -> dict[str, Any]:
        injected_arg_names = self._get_injected_arg_names(tool)
        return {
            name: value
            for name, value in payload.items()
            if name not in injected_arg_names
        }

    def _build_tool_payload(
        self,
        tool: BaseTool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        runtime: ToolRuntime | None = None,
    ) -> str | dict[str, Any]:
        """Convert Monty call arguments into a payload suitable for a LangChain tool."""
        input_schema = tool.get_input_schema()
        schema_annotations = getattr(input_schema, "__annotations__", {})
        fields = [
            name
            for name, type_ in schema_annotations.items()
            if not _is_injected_arg_type(type_)
        ]
        runtime_arg_name = self._get_runtime_arg_name(tool)

        if kwargs:
            payload: str | dict[str, Any] = self._filter_injected_kwargs(tool, kwargs)
        elif len(args) == 1 and isinstance(args[0], dict):
            payload = self._filter_injected_kwargs(tool, args[0])
        elif len(args) == 1 and isinstance(args[0], str) and runtime_arg_name is None:
            payload = args[0]
        elif len(args) == 1 and len(fields) == 1:
            payload = {fields[0]: args[0]}
        elif len(args) == len(fields) and fields:
            payload = dict(zip(fields, args, strict=False))
        else:
            payload = {"args": list(args)}

        if (
            runtime is not None
            and runtime_arg_name is not None
            and isinstance(payload, dict)
        ):
            return {**payload, runtime_arg_name: runtime}
        return payload

    def _wrap_tool_for_sync_monty(
        self, tool: BaseTool, *, runtime: ToolRuntime | None = None
    ) -> Callable[..., Any]:
        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs, runtime=runtime)
            return tool.invoke(payload)

        return tool_wrapper

    def _wrap_tool_for_async_monty(
        self, tool: BaseTool, *, runtime: ToolRuntime | None = None
    ) -> Callable[..., Awaitable[Any]]:
        async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs, runtime=runtime)
            return await tool.ainvoke(payload)

        return tool_wrapper

    def _build_external_functions(
        self, *, runtime: ToolRuntime | None = None
    ) -> dict[str, Callable[..., Any]] | None:
        """Normalize foreign implementations into plain sync callables for Monty."""
        if not self._ptc:
            return None

        external_functions: dict[str, Callable[..., Any]] = {}
        for item in self._ptc:
            name = self._get_ptc_name(item)
            implementation = item
            if isinstance(implementation, BaseTool):
                external_functions[name] = self._wrap_tool_for_sync_monty(
                    implementation,
                    runtime=runtime,
                )
            else:
                external_functions[name] = implementation
        return external_functions

    def _build_external_functions_async(
        self, *, runtime: ToolRuntime | None = None
    ) -> dict[str, Callable[..., Awaitable[Any]] | Callable[..., Any]] | None:
        """Normalize foreign implementations for async Monty execution."""
        if not self._ptc:
            return None

        external_functions: dict[
            str, Callable[..., Awaitable[Any]] | Callable[..., Any]
        ] = {}
        for item in self._ptc:
            name = self._get_ptc_name(item)
            implementation = item
            if isinstance(implementation, BaseTool):
                external_functions[name] = self._wrap_tool_for_async_monty(
                    implementation,
                    runtime=runtime,
                )
            else:
                external_functions[name] = implementation
        return external_functions

    def _format_monty_result(self, result: list[str]) -> str:
        return "\n".join(result).rstrip()

    def _run_monty(
        self,
        code: str,
        *,
        timeout: int | None,
        runtime: ToolRuntime | None = None,
    ) -> str:
        """Execute a single Monty program and return captured stdout."""
        limits = ResourceLimits()
        if timeout is not None:
            if timeout <= 0:
                return f"Error: timeout must be positive, got {timeout}."
            limits["max_duration_secs"] = timeout

        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=[],
                script_name="repl.py",
                type_check=False,
                type_check_stubs=None,
            )
        except pydantic_monty.MontyError as exc:
            return str(exc)

        result: list[str] = []

        def print_callback(stream: Any, value: Any) -> None:  # noqa: ARG001
            """Collect printed output emitted by the Monty program."""
            result.append(str(value))

        try:
            monty.run(
                limits=limits,
                external_functions=self._build_external_functions(runtime=runtime),
                print_callback=print_callback,
            )
        except pydantic_monty.MontyError as exc:
            return str(exc)
        return self._format_monty_result(result)

    async def _arun_monty(
        self,
        code: str,
        *,
        timeout: int | None,
        runtime: ToolRuntime | None = None,
    ) -> str:
        """Execute a single Monty program using Monty's async Python API."""
        limits = ResourceLimits()
        if timeout is not None:
            if timeout <= 0:
                return f"Error: timeout must be positive, got {timeout}."
            limits["max_duration_secs"] = timeout

        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=[],
                script_name="repl.py",
                type_check=False,
                type_check_stubs=None,
            )
        except pydantic_monty.MontyError as exc:
            return str(exc)

        result: list[str] = []

        def print_callback(stream: Any, value: Any) -> None:  # noqa: ARG001
            """Collect printed output emitted by the Monty program."""
            result.append(str(value))

        try:
            await pydantic_monty.run_monty_async(
                monty,
                limits=limits,
                external_functions=self._build_external_functions_async(
                    runtime=runtime
                ),
                print_callback=print_callback,
            )
        except pydantic_monty.MontyError as exc:
            return str(exc)

        return self._format_monty_result(result)

    def _create_repl_tool(self) -> BaseTool:
        """Create the LangChain tool wrapper around Monty execution."""

        def _sync_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime,
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            return self._run_monty(code, timeout=timeout, runtime=runtime)

        async def _async_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime,
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            return await self._arun_monty(code, timeout=timeout, runtime=runtime)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_monty,
            coroutine=_async_monty,
        )
