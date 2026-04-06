"""Middleware for providing a REPL-backed repl tool to an agent."""

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

from langchain_repl._foreign_function_docs import render_foreign_function_section
from langchain_repl.interpreter import Interpreter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


REPL_TOOL_DESCRIPTION = """Evaluates code using a small imperative REPL.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, or helper values from prior `repl` calls are available.

Capabilities and limitations:
- The language supports assignment, `if ... then ... else ... end`, `for ... in ... do ... end`, indexing, function calls, `parallel(...)`, and `try(...)`.
- Use `print(value)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- Values include strings, `None`, `True`, `False`, integers, floats, lists, and dicts.
- `parallel(...)` evaluates independent expressions concurrently using isolated snapshots of the current bindings.
- There is no filesystem or network access unless you expose Python callables as foreign functions.
{external_functions_section}
"""  # noqa: E501

REPL_SYSTEM_PROMPT = """## REPL tool

You have access to a `repl` tool.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, or helper values from prior `repl` calls are available.

- The REPL executes a small imperative language.
- Write assignments like `user = lookup_fn("value")`.
- Use indexing like `items[0]` and `user["id"]`.
- Use `if cond then ... else ... end` for branching.
- Use `for item in items do ... end` for loops.
- Use `print(value)` to emit output. The tool returns printed lines joined with newlines.
- The final expression value is returned only if nothing was printed.
- Use `parallel(expr1, expr2)` only for independent expressions that can run concurrently.
- Use `try(expr, fallback)` when a failed lookup or function call should fall back to another value.
- The REPL can only use the language features above and the foreign functions listed below.
- If the task needs multiple foreign function calls, prefer writing one complete REPL program instead of splitting the work across multiple `repl` invocations.
- If one foreign function returns an ID or other value that can be passed directly into the next foreign function, trust it and chain the calls instead of stopping to double-check it.
- If you want to inspect an intermediate value, print it inside the same REPL program; otherwise, try to fetch as much information as possible in one program.
- Example syntax only - this shows the language shape, not specific available foreign functions:
  `items = lookup_fn("value")`
  `first_item = items[0]`
  `item_id = first_item["id"]`
  `print(parallel(detail_fn(item_id), status_fn(item_id)))`
- Use the repl for small computations, collection manipulation, branching, loops, and calling externally registered foreign functions.
{external_functions_section}
"""  # noqa: E501


class ReplMiddleware(AgentMiddleware[AgentState[Any], ContextT, ResponseT]):
    """Provide a REPL-backed `repl` tool to an agent."""

    def __init__(
        self,
        *,
        ptc: list[Callable[..., Any] | BaseTool] | None = None,
        add_ptc_docs: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Initialize the middleware and register the `repl` tool."""
        self._ptc = ptc or []
        self._add_ptc_docs = add_ptc_docs
        self._max_workers = max_workers
        self.tools = [self._create_repl_tool()]

    def _get_ptc_implementations(self) -> dict[str, Callable[..., Any] | BaseTool]:
        implementations: dict[str, Callable[..., Any] | BaseTool] = {}
        for implementation in self._ptc:
            if isinstance(implementation, BaseTool):
                name = str(implementation.name)
            elif hasattr(implementation, "__name__"):
                name = str(implementation.__name__)
            else:
                msg = (
                    f"Implementation type: {type(implementation)} is "
                    "not a BaseTool or callable"
                )
                raise TypeError(msg)
            implementations[name] = implementation
        return implementations

    def _format_repl_system_prompt(self) -> str:
        external_functions_section = self._format_external_functions_section()
        return REPL_SYSTEM_PROMPT.format(
            external_functions_section=external_functions_section
        )

    def _format_external_functions_section(self) -> str:
        implementations = self._get_ptc_implementations()
        if not implementations:
            return ""

        if not self._add_ptc_docs:
            formatted_functions = "\n".join(f"- {name}" for name in implementations)
            return f"\n\nAvailable foreign functions:\n{formatted_functions}"

        return f"\n\n{render_foreign_function_section(implementations)}"

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Append the REPL prompt guidance to the request system message."""
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
        """Apply request modifications before a synchronous model call."""
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Apply request modifications before an asynchronous model call."""
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

    def _wrap_tool_for_repl(self, tool: BaseTool) -> Callable[..., Any]:
        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs)
            return tool.invoke(payload)

        return tool_wrapper

    def _build_external_functions(self) -> dict[str, Callable[..., Any]]:
        external_functions: dict[str, Callable[..., Any]] = {}
        for name, implementation in self._get_ptc_implementations().items():
            if isinstance(implementation, BaseTool):
                external_functions[name] = self._wrap_tool_for_repl(implementation)
            else:
                external_functions[name] = implementation
        return external_functions

    def _run_interpreter(self, code: str) -> str:
        interpreter = Interpreter(
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
        def _sync_repl(
            code: Annotated[str, "Code string to evaluate in the REPL."],
        ) -> str:
            return self._run_interpreter(code)

        async def _async_repl(
            code: Annotated[str, "Code string to evaluate in the REPL."],
        ) -> str:
            return self._run_interpreter(code)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_repl,
            coroutine=_async_repl,
        )
