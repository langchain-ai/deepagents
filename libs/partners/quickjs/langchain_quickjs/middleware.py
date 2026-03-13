"""Middleware for providing a QuickJS-backed repl tool to an agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import quickjs
from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain_core.tools import BaseTool, StructuredTool

from langchain_quickjs._foreign_function_docs import (
    format_foreign_function_docs,
    render_foreign_function_section,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


REPL_TOOL_DESCRIPTION = """Evaluates code using a QuickJS-backed JavaScript REPL.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, imports, or helper objects from prior `repl` calls are available.

Capabilities and limitations:
- Executes JavaScript with QuickJS.
- Use `print(...)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- There is no filesystem or network access unless you expose Python callables as foreign functions.
{external_functions_section}
"""  # noqa: E501  # preserve prompt text formatting exactly for the model

REPL_SYSTEM_PROMPT = """## REPL tool

You have access to a `repl` tool.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, imports, or helper objects from prior `repl` calls are available.

- The REPL executes JavaScript with QuickJS.
- Use `print(...)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- There is no filesystem or network access unless equivalent foreign functions have been provided.
- Use it for small computations, control flow, JSON manipulation, and calling externally registered foreign functions.
{external_functions_section}
"""  # noqa: E501  # preserve prompt text formatting exactly for the model


class QuickJSMiddleware(AgentMiddleware[AgentState[Any], ContextT, ResponseT]):
    """Provide a QuickJS-backed `repl` tool to an agent."""

    def __init__(
        self,
        *,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any] | BaseTool]
        | None = None,
        auto_include: bool = False,
        timeout: int | None = None,
        memory_limit: int | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            external_functions: Names of external functions available to the REPL.
            external_function_implementations: Implementations for allowed external
                functions. Values may be plain callables or LangChain tools.
            auto_include: Whether to automatically include function signatures and
                docstrings for foreign functions in the system prompt.
            timeout: Optional timeout in seconds for each evaluation.
            memory_limit: Optional memory limit in bytes for each evaluation.
        """
        self._external_functions = external_functions
        self._external_function_implementations = external_function_implementations
        self._auto_include = auto_include
        self._timeout = timeout
        self._memory_limit = memory_limit
        self.tools = [self._create_repl_tool()]

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
        """Convert QuickJS call arguments into a LangChain tool payload."""
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

    def _wrap_tool_for_js(self, tool: BaseTool) -> Callable[..., Any]:
        """Adapt a LangChain tool invocation into a plain callable for QuickJS."""

        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs)
            return tool.invoke(payload)

        return tool_wrapper

    def _build_external_functions(self) -> dict[str, Callable[..., Any]]:
        """Normalize foreign implementations into plain sync callables for QuickJS."""
        external_functions: dict[str, Callable[..., Any]] = {}
        for name, implementation in (
            self._external_function_implementations or {}
        ).items():
            if isinstance(implementation, BaseTool):
                external_functions[name] = self._wrap_tool_for_js(implementation)
            else:
                external_functions[name] = implementation
        return external_functions

    def _create_context(
        self, timeout: int | None, printed_lines: list[str]
    ) -> quickjs.Context:
        """Create a configured QuickJS context for a single evaluation."""
        context = quickjs.Context()
        effective_timeout = timeout if timeout is not None else self._timeout
        if effective_timeout is not None:
            if effective_timeout <= 0:
                msg = f"timeout must be positive, got {effective_timeout}."
                raise ValueError(msg)
            context.set_time_limit(effective_timeout)
        if self._memory_limit is not None:
            if self._memory_limit <= 0:
                msg = f"memory_limit must be positive, got {self._memory_limit}."
                raise ValueError(msg)
            context.set_memory_limit(self._memory_limit)

        def _print(*args: Any) -> None:
            printed_lines.append(" ".join(map(str, args)))

        context.add_callable("print", _print)
        for name, implementation in self._build_external_functions().items():
            context.add_callable(name, implementation)
        return context

    def _evaluate(self, code: str, *, timeout: int | None) -> str:
        """Execute JavaScript and return printed output or final value."""
        printed_lines: list[str] = []
        try:
            context = self._create_context(timeout, printed_lines)
        except ValueError as exc:
            return f"Error: {exc}"

        try:
            value = context.eval(code)
        except quickjs.JSException as exc:
            return str(exc)

        if printed_lines:
            return "\n".join(printed_lines).rstrip()
        if value is None:
            return ""
        return str(value)

    def _create_repl_tool(self) -> BaseTool:
        """Create the LangChain tool wrapper around QuickJS execution."""

        def _sync_quickjs(
            code: Annotated[str, "Code string to evaluate in QuickJS."],
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            """Execute a single QuickJS program and return captured stdout."""
            return self._evaluate(code, timeout=timeout)

        async def _async_quickjs(
            code: Annotated[str, "Code string to evaluate in QuickJS."],
            execution_timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            """Execute a single QuickJS program in the async tool path."""
            return self._evaluate(code, timeout=execution_timeout)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_quickjs,
            coroutine=_async_quickjs,
        )
