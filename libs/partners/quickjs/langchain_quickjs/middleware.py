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

from langchain_quickjs._foreign_function_bridge import install_external_functions
from langchain_quickjs._foreign_function_docs import (
    get_ptc_implementations,
    render_external_functions_section,
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
        ptc: list[Callable[..., Any] | BaseTool] | None = None,
        add_ptc_docs: bool = False,
        timeout: int | None = None,
        memory_limit: int | None = None,
    ) -> None:
        """Initialize the middleware.

        Args:
            ptc: Functions or LangChain tools to expose inside the REPL.
            add_ptc_docs: Whether to add signatures and docstrings for exposed PTC
                functions to the system prompt.
            timeout: Optional timeout in seconds for each evaluation.
            memory_limit: Optional memory limit in bytes for each evaluation.
        """
        self._ptc = ptc or []
        self._add_ptc_docs = add_ptc_docs
        self._timeout = timeout
        self._memory_limit = memory_limit
        self.tools = [self._create_repl_tool()]

    def _format_repl_system_prompt(self) -> str:
        """Build the system prompt fragment describing the repl tool."""
        external_functions_section = render_external_functions_section(
            get_ptc_implementations(self._ptc),
            add_docs=self._add_ptc_docs,
        )
        return REPL_SYSTEM_PROMPT.format(
            external_functions_section=external_functions_section
        )

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

    def _create_context(
        self,
        timeout: int | None,
        printed_lines: list[str],
        *,
        prefer_async: bool = False,
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
        install_external_functions(
            context,
            get_ptc_implementations(self._ptc),
            execution_mode="async" if prefer_async else "sync",
        )
        return context

    def _evaluate(
        self,
        code: str,
        *,
        timeout: int | None,
        prefer_async: bool = False,
    ) -> str:
        """Execute JavaScript and return printed output or final value."""
        printed_lines: list[str] = []
        try:
            context = self._create_context(
                timeout,
                printed_lines,
                prefer_async=prefer_async,
            )
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
            return self._evaluate(code, timeout=timeout, prefer_async=False)

        async def _async_quickjs(
            code: Annotated[str, "Code string to evaluate in QuickJS."],
            timeout: Annotated[
                int | None, "Optional timeout in seconds for this evaluation."
            ] = None,
        ) -> str:
            """Execute a single QuickJS program in the async tool path."""
            return self._evaluate(code, timeout=timeout, prefer_async=True)

        tool_description = REPL_TOOL_DESCRIPTION.format(
            external_functions_section=render_external_functions_section(
                get_ptc_implementations(self._ptc),
                add_docs=self._add_ptc_docs,
            )
        )

        return StructuredTool.from_function(
            name="repl",
            description=tool_description,
            func=_sync_quickjs,
            coroutine=_async_quickjs,
        )
