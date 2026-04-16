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
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # StructuredTool evaluates this annotation at runtime
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.tools.base import (
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)

from langchain_repl._foreign_function_docs import render_foreign_function_section
from langchain_repl.interpreter import Interpreter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


REPL_TOOL_DESCRIPTION = """Evaluates code using a small imperative REPL.

CRITICAL: The REPL does NOT retain state between calls. Each `repl` invocation is evaluated from scratch.
Do NOT assume variables, functions, or helper values from prior `repl` calls are available.

Capabilities and limitations:
- The language supports assignment, `if ... then ... else ... end`, `for ... in ... do ... end`, indexing, function calls, and `parallel(...)`.
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
        max_concurrency: int | None = None,
    ) -> None:
        """Initialize the middleware and register the `repl` tool."""
        self._ptc = ptc or []
        self._add_ptc_docs = add_ptc_docs
        self._max_concurrency = max_concurrency
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

    def _wrap_tool_for_repl(
        self, tool: BaseTool, *, runtime: ToolRuntime | None = None
    ) -> Callable[..., Any]:
        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            payload = self._build_tool_payload(tool, args, kwargs, runtime=runtime)
            return tool.invoke(payload)

        return tool_wrapper

    def _build_external_functions(
        self, *, runtime: ToolRuntime | None = None
    ) -> dict[str, Callable[..., Any] | BaseTool]:
        external_functions: dict[str, Callable[..., Any] | BaseTool] = {}
        for name, implementation in self._get_ptc_implementations().items():
            if isinstance(implementation, BaseTool):
                if self._get_runtime_arg_name(implementation) is not None:
                    external_functions[name] = self._wrap_tool_for_repl(
                        implementation,
                        runtime=runtime,
                    )
                else:
                    external_functions[name] = implementation
            else:
                external_functions[name] = implementation
        return external_functions

    def _format_interpreter_result(self, interpreter: Interpreter, value: Any) -> str:
        if interpreter.printed_lines:
            return "\n".join(interpreter.printed_lines).rstrip()
        if value is None:
            return ""
        return str(value)

    def _run_interpreter(self, code: str, *, runtime: ToolRuntime | None = None) -> str:
        interpreter = Interpreter(
            functions=self._build_external_functions(runtime=runtime),
            max_concurrency=self._max_concurrency,
            runtime=runtime,
        )
        try:
            value = interpreter.evaluate(code)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"
        return self._format_interpreter_result(interpreter, value)

    async def _arun_interpreter(
        self, code: str, *, runtime: ToolRuntime | None = None
    ) -> str:
        interpreter = Interpreter(
            functions=self._build_external_functions(runtime=runtime),
            max_concurrency=self._max_concurrency,
            runtime=runtime,
        )
        try:
            value = await interpreter.aevaluate(code)
        except Exception as exc:  # noqa: BLE001
            return f"Error: {exc}"
        return self._format_interpreter_result(interpreter, value)

    def _create_repl_tool(self) -> BaseTool:
        def _sync_repl(
            code: Annotated[str, "Code string to evaluate in the REPL."],
            runtime: ToolRuntime,
        ) -> str:
            return self._run_interpreter(code, runtime=runtime)

        async def _async_repl(
            code: Annotated[str, "Code string to evaluate in the REPL."],
            runtime: ToolRuntime,
        ) -> str:
            return await self._arun_interpreter(code, runtime=runtime)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=self._format_external_functions_section()
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_repl,
            coroutine=_async_repl,
        )
