"""Middleware for providing a Lisp-backed repl tool to an agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

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

from langchain_lisp._foreign_function_docs import (
    format_foreign_function_docs,
    render_foreign_function_section,
)
from langchain_lisp.interpreter import LispInterpreter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


REPL_TOOL_DESCRIPTION = """Evaluates code using a small Lisp-like REPL.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, or helper values from prior `repl` calls are available.

Capabilities and limitations:
- The language uses prefix forms like `(+ 1 2)` and `(if cond then else)`.
- Use `(print value)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- Builtins include arithmetic, comparisons, boolean helpers, `get`, `length`, `list`, `dict`, `if`, `let`, and `parallel`.
- `parallel` evaluates independent expressions concurrently using isolated snapshots of the current bindings.
- There is no filesystem or network access unless you expose Python callables as foreign functions.
{external_functions_section}
"""  # noqa: E501

REPL_SYSTEM_PROMPT = """## REPL tool

You have access to a `repl` tool.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, or helper values from prior `repl` calls are available.

- The REPL executes a small Lisp-like language with prefix forms.
- Write function calls like `(+ 1 2)`, `(length items)`, `(get user \"name\")`, and `(if cond then else)`.
- Use `(let name expr)` to bind a variable within the current repl program.
- Use `(print value)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- Builtins include arithmetic, comparisons, boolean helpers, `get`, `length`, `list`, `dict`, `if`, `let`, and `parallel`.
- Use `parallel` only for independent expressions that can run concurrently.
- There is no filesystem or network access unless equivalent foreign functions have been provided.
- Use the repl for small computations, collection manipulation, branching, and calling externally registered foreign functions.
{external_functions_section}
"""  # noqa: E501


class LispMiddleware(AgentMiddleware[AgentState[Any], ContextT, ResponseT]):
    """Provide a Lisp-backed `repl` tool to an agent."""

    def __init__(
        self,
        *,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any] | BaseTool]
        | None = None,
        auto_include: bool = False,
        max_workers: int | None = None,
    ) -> None:
        self._external_functions = external_functions
        self._external_function_implementations = external_function_implementations
        self._auto_include = auto_include
        self._max_workers = max_workers
        self.tools = [self._create_repl_tool()]

    def _format_foreign_function_docs(self, name: str) -> str | None:
        if not self._auto_include:
            return None
        if not self._external_function_implementations:
            return None
        implementation = self._external_function_implementations.get(name)
        if implementation is None:
            return None
        return format_foreign_function_docs(name, implementation)

    def _format_repl_system_prompt(self) -> str:
        external_functions_section = self._format_external_functions_section()
        return REPL_SYSTEM_PROMPT.format(
            external_functions_section=external_functions_section
        )

    def _format_external_functions_section(self) -> str:
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
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        modified_request = self.modify_request(request)
        return await handler(modified_request)

    def _build_tool_payload(
        self, tool: BaseTool, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> str | dict[str, Any]:
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

    def _wrap_tool_for_lisp(self, tool: BaseTool) -> Callable[..., Any]:
        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs)
            return tool.invoke(payload)

        return tool_wrapper

    def _build_external_functions(self) -> dict[str, Callable[..., Any]]:
        external_functions: dict[str, Callable[..., Any]] = {}
        for name, implementation in (
            self._external_function_implementations or {}
        ).items():
            if isinstance(implementation, BaseTool):
                external_functions[name] = self._wrap_tool_for_lisp(implementation)
            else:
                external_functions[name] = implementation
        return external_functions

    def _run_interpreter(self, code: str) -> str:
        interpreter = LispInterpreter(
            functions=self._build_external_functions(),
            max_workers=self._max_workers,
        )
        try:
            value = interpreter.evaluate(code)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"
        if interpreter.printed_lines:
            return "\n".join(interpreter.printed_lines).rstrip()
        if value is None:
            return ""
        return str(value)

    def _create_repl_tool(self) -> BaseTool:
        def _sync_lisp(
            code: Annotated[str, "Code string to evaluate in Lisp."],
        ) -> str:
            return self._run_interpreter(code)

        async def _async_lisp(
            code: Annotated[str, "Code string to evaluate in Lisp."],
        ) -> str:
            return self._run_interpreter(code)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_lisp,
            coroutine=_async_lisp,
        )
