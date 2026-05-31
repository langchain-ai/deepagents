"""Stagehand-style browser automation middleware."""
# ruff: noqa: ANN401, E501

from __future__ import annotations

import asyncio
import base64
import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message

StagehandMode = Literal["dom", "hybrid"]
VariablePrimitive = str | int | float | bool
VariableValue = VariablePrimitive | Mapping[str, VariablePrimitive | str]
BrowserResolver = Any | Callable[[ToolRuntime[Any, Any]], Any]
ToolResultCallback = Callable[[dict[str, Any]], None]

_DEFAULT_TOOL_TIMEOUT = 45.0

_DOM_TOOL_NAMES = {
    "act",
    "ariaTree",
    "extract",
    "fillForm",
    "goto",
    "keys",
    "navback",
    "screenshot",
    "scroll",
    "think",
    "wait",
}
_HYBRID_TOOL_NAMES = {
    "act",
    "ariaTree",
    "click",
    "clickAndHold",
    "dragAndDrop",
    "extract",
    "fillFormVision",
    "goto",
    "keys",
    "navback",
    "screenshot",
    "scroll",
    "think",
    "type",
    "wait",
}
_MUTATING_TOOLS = {
    "act",
    "click",
    "clickAndHold",
    "dragAndDrop",
    "fillForm",
    "fillFormVision",
    "goto",
    "keys",
    "navback",
    "scroll",
    "type",
    "wait",
}


class StagehandToolEvent(TypedDict):
    """Event emitted after a Stagehand browser tool runs."""

    tool: str
    args: dict[str, Any]
    ok: bool
    result: NotRequired[Any]
    error: NotRequired[str]
    page_url: NotRequired[str]


class ActSchema(BaseModel):
    """Schema for the act tool."""

    action: str = Field(description="A specific atomic browser action to perform, such as clicking or typing.")


class AriaTreeSchema(BaseModel):
    """Schema for the ariaTree tool."""

    instruction: str | None = Field(default=None, description="Optional focus for the accessibility tree snapshot.")


class ClickSchema(BaseModel):
    """Schema for coordinate click tools."""

    x: int = Field(description="X coordinate in the viewport.")
    y: int = Field(description="Y coordinate in the viewport.")


class ClickAndHoldSchema(ClickSchema):
    """Schema for clickAndHold."""

    duration_ms: int = Field(default=1000, description="How long to hold the click, in milliseconds.")


class DragAndDropSchema(BaseModel):
    """Schema for dragAndDrop."""

    from_x: int = Field(description="Starting X coordinate in the viewport.")
    from_y: int = Field(description="Starting Y coordinate in the viewport.")
    to_x: int = Field(description="Ending X coordinate in the viewport.")
    to_y: int = Field(description="Ending Y coordinate in the viewport.")


class ExtractSchema(BaseModel):
    """Schema for extract."""

    instruction: str = Field(description="Description of the structured data to extract from the current page.")


class FillFormSchema(BaseModel):
    """Schema for fillForm."""

    instruction: str = Field(description="Instruction describing the form fields to fill and values to use.")


class GotoSchema(BaseModel):
    """Schema for goto."""

    url: str = Field(description="URL to navigate to.")


class KeysSchema(BaseModel):
    """Schema for keys."""

    keys: str = Field(description="Keyboard key or key sequence to press.")


class ScrollSchema(BaseModel):
    """Schema for scroll."""

    direction: Literal["up", "down", "left", "right"] | None = Field(default=None, description="Semantic scroll direction.")
    pixels: int = Field(default=700, description="Number of pixels to scroll.")


class ThinkSchema(BaseModel):
    """Schema for think."""

    reasoning: str = Field(description="Reasoning or plan before taking browser actions.")


class TypeSchema(BaseModel):
    """Schema for type."""

    text: str = Field(description="Text to type.")
    x: int | None = Field(default=None, description="Optional X coordinate of the target input.")
    y: int | None = Field(default=None, description="Optional Y coordinate of the target input.")


class WaitSchema(BaseModel):
    """Schema for wait."""

    time_ms: int = Field(default=1000, description="Time to wait in milliseconds.")


class NoArgsSchema(BaseModel):
    """Empty schema for no-argument tools."""


class StagehandBrowserToolsMiddleware(AgentMiddleware[Any, Any, Any]):
    """Bundle Stagehand-like browser automation tools and instructions into an agent."""

    def __init__(
        self,
        *,
        browser: BrowserResolver | None = None,
        mode: StagehandMode = "dom",
        system_prompt: str | None = None,
        exclude_tools: list[str] | None = None,
        variables: Mapping[str, VariableValue] | None = None,
        tool_timeout: float | None = _DEFAULT_TOOL_TIMEOUT,
        use_search: bool = False,
        search: Callable[[str], Any] | Callable[[str], Awaitable[Any]] | None = None,
        on_tool_result: ToolResultCallback | None = None,
    ) -> None:
        """Initialize the middleware."""
        if mode not in ("dom", "hybrid"):
            msg = "StagehandBrowserToolsMiddleware only supports mode='dom' or mode='hybrid'. Use a separate CUA harness for mode='cua'."
            raise ValueError(msg)
        self.browser = browser
        self.mode = mode
        self.system_prompt = system_prompt
        self.exclude_tools = frozenset(exclude_tools or [])
        self.variables = dict(variables or {})
        self.tool_timeout = tool_timeout
        self.use_search = use_search
        self.search = search
        self.on_tool_result = on_tool_result
        self.tools = self._build_tools()

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject Stagehand prompt guidance and filter tools by mode."""
        request = self._prepare_request(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async variant of wrap_model_call."""
        request = self._prepare_request(request)
        return await handler(request)

    def _prepare_request(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        current_url = self._get_current_url_sync(request.runtime) or "unknown"
        system_prompt = self._build_system_prompt(current_url)
        request = request.override(system_message=append_to_system_message(request.system_message, system_prompt))
        active_tool_names = self._active_tool_names()
        filtered_tools = [tool for tool in request.tools if self._tool_name(tool) not in self.exclude_tools]
        filtered_tools = [tool for tool in filtered_tools if self._tool_name(tool) not in self._all_stagehand_tool_names() or self._tool_name(tool) in active_tool_names]
        return request.override(tools=filtered_tools)

    def _build_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = [
            self._tool("act", "Perform a specific atomic action on the page, such as click or type.", ActSchema, self._act, self._aact),
            self._tool("ariaTree", "Get an accessibility tree snapshot for full page context.", AriaTreeSchema, self._aria_tree, self._aaria_tree),
            self._tool("extract", "Extract structured data from the current page.", ExtractSchema, self._extract, self._aextract),
            self._tool("goto", "Navigate to a URL.", GotoSchema, self._goto, self._agoto),
            self._tool("keys", "Press a keyboard key or key sequence.", KeysSchema, self._keys, self._akeys),
            self._tool("navback", "Navigate back in browser history.", NoArgsSchema, self._navback, self._anavback),
            self._tool("screenshot", "Take a screenshot for visual context.", NoArgsSchema, self._screenshot, self._ascreenshot),
            self._tool("scroll", "Scroll the page.", ScrollSchema, self._scroll, self._ascroll),
            self._tool("think", "Reason about the next browser action without changing the page.", ThinkSchema, self._think, self._athink),
            self._tool("wait", "Wait for a specified time.", WaitSchema, self._wait, self._await),
        ]
        if self.mode == "dom":
            tools.append(self._tool("fillForm", "Fill out a form using DOM semantics.", FillFormSchema, self._fill_form, self._afill_form))
        else:
            tools.extend(
                [
                    self._tool("click", "Click on visible coordinates in the viewport.", ClickSchema, self._click, self._aclick),
                    self._tool("clickAndHold", "Click and hold on visible coordinates.", ClickAndHoldSchema, self._click_and_hold, self._aclick_and_hold),
                    self._tool("dragAndDrop", "Drag from one viewport coordinate to another.", DragAndDropSchema, self._drag_and_drop, self._adrag_and_drop),
                    self._tool("fillFormVision", "Fill out a form using visual coordinates.", FillFormSchema, self._fill_form_vision, self._afill_form_vision),
                    self._tool("type", "Type text, optionally at visible coordinates.", TypeSchema, self._type, self._atype),
                ]
            )
        if self.use_search:
            tools.append(self._tool("search", "Perform a web search and return results.", ExtractSchema, self._search, self._asearch))
        return [tool for tool in tools if tool.name not in self.exclude_tools]

    @staticmethod
    def _tool(
        name: str,
        description: str,
        args_schema: type[BaseModel],
        func: Callable[..., ToolMessage],
        coroutine: Callable[..., Awaitable[ToolMessage]],
    ) -> BaseTool:
        return StructuredTool.from_function(
            name=name,
            description=description,
            func=func,
            coroutine=coroutine,
            args_schema=args_schema,
            infer_schema=False,
        )

    def _active_tool_names(self) -> set[str]:
        names = set(_HYBRID_TOOL_NAMES if self.mode == "hybrid" else _DOM_TOOL_NAMES)
        if self.use_search:
            names.add("search")
        return names - set(self.exclude_tools)

    @staticmethod
    def _all_stagehand_tool_names() -> set[str]:
        return _DOM_TOOL_NAMES | _HYBRID_TOOL_NAMES | {"search"}

    @staticmethod
    def _tool_name(tool: BaseTool | dict[str, Any]) -> str | None:
        if isinstance(tool, dict):
            name = tool.get("name")
            return name if isinstance(name, str) else None
        name = getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _build_system_prompt(self, current_url: str) -> str:
        now = datetime.now(UTC)
        custom = f"\n  <customInstructions>{self.system_prompt}</customInstructions>" if self.system_prompt else ""
        search_rule = "\n    <rule>If you are not confident in the URL, use the search tool to find it.</rule>" if self.use_search else ""
        variables = self._variables_prompt()
        tools = self._tools_prompt()
        page_protocol = self._page_understanding_prompt()
        strategy = self._strategy_prompt()
        return f"""<system>
  <identity>You are a web automation assistant using browser automation tools to accomplish the user's goal.</identity>{custom}
  <task>
    <date display="local" iso="{now.isoformat()}">{now.date().isoformat()}</date>
    <note>You may think the date is different due to knowledge cutoff, but this is the actual date.</note>
  </task>
  <page>
    <startingUrl>you are starting your task on this url: {current_url}</startingUrl>
  </page>
  <mindset>
    <note>Be very intentional about your actions. Slight variations of the user's goal can lead to failures.</note>
    <note>When the task is complete, do not seek more information; you have completed the task.</note>
  </mindset>
  <guidelines>
    <item>Always start by understanding the current page state.</item>
    <item>Use appropriate tools for each action.</item>
    <item>Keep actions atomic and verify outcomes before proceeding.</item>
  </guidelines>
  {page_protocol}
  <navigation>
    <rule>If you are confident in the URL, navigate directly to it.</rule>{search_rule}
  </navigation>
  {tools}
  <strategy>
    {strategy}
    <item>Use extract ONLY when the task explicitly requires structured data output. For reading page content or understanding elements, use ariaTree or screenshot instead.</item>
    <item>For each action, provide clear reasoning about why you're taking that step.</item>
    <item>When entering text that could be typed through multiple separate inputs, prefer the keys tool to type the entire sequence when appropriate.</item>
  </strategy>
  {variables}
  <completion>
    <note>When you complete the task, explain any information that was found that was relevant to the original task.</note>
  </completion>
</system>"""

    def _tools_prompt(self) -> str:
        descriptions = {
            "act": "Perform a specific atomic action (click, type, etc.).",
            "ariaTree": "Get an accessibility tree for full page context.",
            "click": "Click on an element using coordinates.",
            "clickAndHold": "Click and hold on an element using coordinates.",
            "dragAndDrop": "Drag and drop an element using coordinates.",
            "extract": "Extract structured data.",
            "fillForm": "Fill out a form.",
            "fillFormVision": "Fill out a form using coordinates.",
            "goto": "Navigate to a URL.",
            "keys": "Press a keyboard key.",
            "navback": "Navigate back in browser history.",
            "screenshot": "Take a screenshot for visual context.",
            "scroll": "Scroll the page.",
            "search": "Perform a web search and return results. Prefer this over navigating to a search engine.",
            "think": "Think about the task.",
            "type": "Type text into an element using coordinates when provided.",
            "wait": "Wait for a specified time.",
        }
        lines = [f'    <tool name="{name}">{descriptions[name]}</tool>' for name in sorted(self._active_tool_names())]
        return "<tools>\n" + "\n".join(lines) + "\n  </tools>"

    def _strategy_prompt(self) -> str:
        if self.mode == "hybrid":
            return """<item>Tool selection priority: use specific tools (click, type) when elements are visible in viewport.</item>
    <item>Use screenshot to ground coordinates before clicking or typing.</item>
    <item>Use ariaTree as a secondary tool when elements are not visible in screenshot or to get full page context.</item>
    <item>Only use act when the element is in ariaTree but not visible in screenshot.</item>"""
        return """<item>Tool selection priority: use act for clicking and typing on a page.</item>
    <item>Always check ariaTree first to understand full page content without scrolling.</item>
    <item>If an element is present in ariaTree, use act to interact with it directly.</item>
    <item>Use screenshot for visual confirmation when needed, but rely primarily on ariaTree for element detection.</item>"""

    def _page_understanding_prompt(self) -> str:
        if self.mode == "hybrid":
            return """<page_understanding_protocol>
    <step_1><title>UNDERSTAND THE PAGE</title><primary_tool>screenshot</primary_tool><secondary_tool>ariaTree</secondary_tool></step_1>
  </page_understanding_protocol>"""
        return """<page_understanding_protocol>
    <step_1><title>UNDERSTAND THE PAGE</title><primary_tool>ariaTree</primary_tool><secondary_tool>screenshot</secondary_tool></step_1>
  </page_understanding_protocol>"""

    def _variables_prompt(self) -> str:
        if not self.variables:
            return ""
        entries: list[str] = []
        for name, value in self.variables.items():
            description = value.get("description") if isinstance(value, Mapping) else None
            entries.append(f'    <variable name="{name}">{description}</variable>' if description else f'    <variable name="{name}" />')
        usage = "Use %variableName% syntax in tool fields. The tool runtime substitutes the actual value before interacting with the browser."
        return "<variables>\n    <note>You have access to the following variables. Use %variableName% syntax for sensitive data.</note>\n" f"    <usage>{usage}</usage>\n" + "\n".join(entries) + "\n  </variables>"

    def _resolve_browser(self, runtime: ToolRuntime[Any, Any]) -> Any:
        if self.browser is None:
            msg = "StagehandBrowserToolsMiddleware requires a browser object or resolver before browser tools can run."
            raise RuntimeError(msg)
        return self.browser(runtime) if callable(self.browser) else self.browser

    def _get_current_url_sync(self, runtime: ToolRuntime[Any, Any]) -> str | None:
        if self.browser is None:
            return None
        try:
            browser = self._resolve_browser(runtime)
            url = self._read_attr_or_call_sync(browser, ("url", "current_url", "page_url"))
            return str(url) if url else None
        except Exception:  # noqa: BLE001
            return None

    async def _get_current_url_async(self, runtime: ToolRuntime[Any, Any]) -> str | None:
        if self.browser is None:
            return None
        try:
            browser = self._resolve_browser(runtime)
            url = await self._read_attr_or_call_async(browser, ("url", "current_url", "page_url"))
            return str(url) if url else None
        except Exception:  # noqa: BLE001
            return None

    def _success(self, runtime: ToolRuntime[Any, Any], tool: str, args: Mapping[str, Any], result: Any) -> ToolMessage:
        event: StagehandToolEvent = {"tool": tool, "args": dict(args), "ok": True, "result": result}
        page_url = self._get_current_url_sync(runtime)
        if page_url:
            event["page_url"] = page_url
        self._emit_event(event, runtime)
        return ToolMessage(content=self._format_tool_result(result), name=tool, tool_call_id=runtime.tool_call_id, status="success")

    async def _asuccess(self, runtime: ToolRuntime[Any, Any], tool: str, args: Mapping[str, Any], result: Any) -> ToolMessage:
        event: StagehandToolEvent = {"tool": tool, "args": dict(args), "ok": True, "result": result}
        page_url = await self._get_current_url_async(runtime)
        if page_url:
            event["page_url"] = page_url
        self._emit_event(event, runtime)
        return ToolMessage(content=self._format_tool_result(result), name=tool, tool_call_id=runtime.tool_call_id, status="success")

    def _error(self, runtime: ToolRuntime[Any, Any], tool: str, args: Mapping[str, Any], error: Exception) -> ToolMessage:
        event: StagehandToolEvent = {"tool": tool, "args": dict(args), "ok": False, "error": f"{type(error).__name__}: {error}"}
        self._emit_event(event, runtime)
        return ToolMessage(content=f"Error: {event['error']}", name=tool, tool_call_id=runtime.tool_call_id, status="error")

    def _emit_event(self, event: StagehandToolEvent, runtime: ToolRuntime[Any, Any]) -> None:
        if self.on_tool_result is not None:
            self.on_tool_result(cast("dict[str, Any]", event))
        if event["tool"] in _MUTATING_TOOLS:
            runtime.stream_writer({"type": "stagehand_tool_result", **event})

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        if isinstance(result, bytes):
            return base64.b64encode(result).decode("utf-8")
        if isinstance(result, str):
            return result
        return repr(result)

    def _substitute_variables(self, value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for name, variable in self.variables.items():
                raw_value = variable.get("value") if isinstance(variable, Mapping) else variable
                result = result.replace(f"%{name}%", str(raw_value))
            return result
        if isinstance(value, list):
            return [self._substitute_variables(item) for item in value]
        if isinstance(value, dict):
            return {key: self._substitute_variables(item) for key, item in value.items()}
        return value

    def _run_sync(
        self,
        runtime: ToolRuntime[Any, Any],
        tool: str,
        args: Mapping[str, Any],
        call: Callable[[Any, dict[str, Any]], Any],
        *,
        browser_required: bool = True,
    ) -> ToolMessage:
        resolved_args = cast("dict[str, Any]", self._substitute_variables(dict(args)))
        try:
            browser = self._resolve_browser(runtime) if browser_required else None
            result = self._with_timeout_sync(lambda: call(browser, resolved_args))
        except Exception as exc:  # noqa: BLE001
            return self._error(runtime, tool, resolved_args, exc)
        return self._success(runtime, tool, resolved_args, result)

    async def _run_async(
        self,
        runtime: ToolRuntime[Any, Any],
        tool: str,
        args: Mapping[str, Any],
        call: Callable[[Any, dict[str, Any]], Awaitable[Any]],
        *,
        browser_required: bool = True,
    ) -> ToolMessage:
        resolved_args = cast("dict[str, Any]", self._substitute_variables(dict(args)))
        try:
            browser = self._resolve_browser(runtime) if browser_required else None
            result = await self._with_timeout_async(lambda: call(browser, resolved_args))
        except Exception as exc:  # noqa: BLE001
            return self._error(runtime, tool, resolved_args, exc)
        return await self._asuccess(runtime, tool, resolved_args, result)

    def _with_timeout_sync(self, func: Callable[[], Any]) -> Any:
        if self.tool_timeout is None:
            return func()
        with asyncio.Runner() as runner:
            return runner.run(asyncio.wait_for(self._maybe_await(func()), timeout=self.tool_timeout))

    async def _with_timeout_async(self, func: Callable[[], Awaitable[Any]]) -> Any:
        if self.tool_timeout is None:
            return await func()
        return await asyncio.wait_for(func(), timeout=self.tool_timeout)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _method(browser: Any, names: tuple[str, ...]) -> Callable[..., Any]:
        for name in names:
            method = getattr(browser, name, None)
            if callable(method):
                return method
        msg = f"Browser object does not implement any of: {', '.join(names)}"
        raise AttributeError(msg)

    @staticmethod
    def _call_method_sync(browser: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        return StagehandBrowserToolsMiddleware._method(browser, names)(*args, **kwargs)

    @staticmethod
    async def _call_method_async(browser: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        return await StagehandBrowserToolsMiddleware._maybe_await(StagehandBrowserToolsMiddleware._method(browser, names)(*args, **kwargs))

    @staticmethod
    def _read_attr_or_call_sync(browser: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            value = getattr(browser, name, None)
            if value is not None:
                return value() if callable(value) else value
        return None

    @staticmethod
    async def _read_attr_or_call_async(browser: Any, names: tuple[str, ...]) -> Any:
        return await StagehandBrowserToolsMiddleware._maybe_await(StagehandBrowserToolsMiddleware._read_attr_or_call_sync(browser, names))

    def _act(self, runtime: ToolRuntime[Any, Any], action: Annotated[str, "Atomic browser action to perform."]) -> ToolMessage:
        return self._run_sync(runtime, "act", {"action": action}, lambda browser, args: self._call_method_sync(browser, ("act",), args["action"]))

    async def _aact(self, runtime: ToolRuntime[Any, Any], action: Annotated[str, "Atomic browser action to perform."]) -> ToolMessage:
        return await self._run_async(runtime, "act", {"action": action}, lambda browser, args: self._call_method_async(browser, ("act",), args["action"]))

    def _aria_tree(self, runtime: ToolRuntime[Any, Any], instruction: str | None = None) -> ToolMessage:
        return self._run_sync(runtime, "ariaTree", {"instruction": instruction}, lambda browser, args: self._call_method_sync(browser, ("aria_tree", "ariaTree", "accessibility_snapshot"), args.get("instruction")))

    async def _aaria_tree(self, runtime: ToolRuntime[Any, Any], instruction: str | None = None) -> ToolMessage:
        return await self._run_async(runtime, "ariaTree", {"instruction": instruction}, lambda browser, args: self._call_method_async(browser, ("aria_tree", "ariaTree", "accessibility_snapshot"), args.get("instruction")))

    def _click(self, runtime: ToolRuntime[Any, Any], x: int, y: int) -> ToolMessage:
        return self._run_sync(runtime, "click", {"x": x, "y": y}, lambda browser, args: self._call_method_sync(browser, ("click",), args["x"], args["y"]))

    async def _aclick(self, runtime: ToolRuntime[Any, Any], x: int, y: int) -> ToolMessage:
        return await self._run_async(runtime, "click", {"x": x, "y": y}, lambda browser, args: self._call_method_async(browser, ("click",), args["x"], args["y"]))

    def _click_and_hold(self, runtime: ToolRuntime[Any, Any], x: int, y: int, duration_ms: int = 1000) -> ToolMessage:
        return self._run_sync(runtime, "clickAndHold", {"x": x, "y": y, "duration_ms": duration_ms}, lambda browser, args: self._call_method_sync(browser, ("click_and_hold", "clickAndHold"), args["x"], args["y"], args["duration_ms"]))

    async def _aclick_and_hold(self, runtime: ToolRuntime[Any, Any], x: int, y: int, duration_ms: int = 1000) -> ToolMessage:
        return await self._run_async(runtime, "clickAndHold", {"x": x, "y": y, "duration_ms": duration_ms}, lambda browser, args: self._call_method_async(browser, ("click_and_hold", "clickAndHold"), args["x"], args["y"], args["duration_ms"]))

    def _drag_and_drop(self, runtime: ToolRuntime[Any, Any], from_x: int, from_y: int, to_x: int, to_y: int) -> ToolMessage:
        args = {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}
        return self._run_sync(runtime, "dragAndDrop", args, lambda browser, a: self._call_method_sync(browser, ("drag_and_drop", "dragAndDrop"), a["from_x"], a["from_y"], a["to_x"], a["to_y"]))

    async def _adrag_and_drop(self, runtime: ToolRuntime[Any, Any], from_x: int, from_y: int, to_x: int, to_y: int) -> ToolMessage:
        args = {"from_x": from_x, "from_y": from_y, "to_x": to_x, "to_y": to_y}
        return await self._run_async(runtime, "dragAndDrop", args, lambda browser, a: self._call_method_async(browser, ("drag_and_drop", "dragAndDrop"), a["from_x"], a["from_y"], a["to_x"], a["to_y"]))

    def _extract(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return self._run_sync(runtime, "extract", {"instruction": instruction}, lambda browser, args: self._call_method_sync(browser, ("extract",), args["instruction"]))

    async def _aextract(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return await self._run_async(runtime, "extract", {"instruction": instruction}, lambda browser, args: self._call_method_async(browser, ("extract",), args["instruction"]))

    def _fill_form(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return self._run_sync(runtime, "fillForm", {"instruction": instruction}, lambda browser, args: self._call_method_sync(browser, ("fill_form", "fillForm"), args["instruction"]))

    async def _afill_form(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return await self._run_async(runtime, "fillForm", {"instruction": instruction}, lambda browser, args: self._call_method_async(browser, ("fill_form", "fillForm"), args["instruction"]))

    def _fill_form_vision(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return self._run_sync(runtime, "fillFormVision", {"instruction": instruction}, lambda browser, args: self._call_method_sync(browser, ("fill_form_vision", "fillFormVision"), args["instruction"]))

    async def _afill_form_vision(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        return await self._run_async(runtime, "fillFormVision", {"instruction": instruction}, lambda browser, args: self._call_method_async(browser, ("fill_form_vision", "fillFormVision"), args["instruction"]))

    def _goto(self, runtime: ToolRuntime[Any, Any], url: str) -> ToolMessage:
        return self._run_sync(runtime, "goto", {"url": url}, lambda browser, args: self._call_method_sync(browser, ("goto", "navigate"), args["url"]))

    async def _agoto(self, runtime: ToolRuntime[Any, Any], url: str) -> ToolMessage:
        return await self._run_async(runtime, "goto", {"url": url}, lambda browser, args: self._call_method_async(browser, ("goto", "navigate"), args["url"]))

    def _keys(self, runtime: ToolRuntime[Any, Any], keys: str) -> ToolMessage:
        return self._run_sync(runtime, "keys", {"keys": keys}, lambda browser, args: self._call_method_sync(browser, ("key_press", "press", "keys"), args["keys"]))

    async def _akeys(self, runtime: ToolRuntime[Any, Any], keys: str) -> ToolMessage:
        return await self._run_async(runtime, "keys", {"keys": keys}, lambda browser, args: self._call_method_async(browser, ("key_press", "press", "keys"), args["keys"]))

    def _navback(self, runtime: ToolRuntime[Any, Any]) -> ToolMessage:
        return self._run_sync(runtime, "navback", {}, lambda browser, _args: self._call_method_sync(browser, ("go_back", "goBack", "back")))

    async def _anavback(self, runtime: ToolRuntime[Any, Any]) -> ToolMessage:
        return await self._run_async(runtime, "navback", {}, lambda browser, _args: self._call_method_async(browser, ("go_back", "goBack", "back")))

    def _screenshot(self, runtime: ToolRuntime[Any, Any]) -> ToolMessage:
        return self._run_sync(runtime, "screenshot", {}, lambda browser, _args: self._call_method_sync(browser, ("screenshot",)))

    async def _ascreenshot(self, runtime: ToolRuntime[Any, Any]) -> ToolMessage:
        return await self._run_async(runtime, "screenshot", {}, lambda browser, _args: self._call_method_async(browser, ("screenshot",)))

    def _scroll(self, runtime: ToolRuntime[Any, Any], direction: Literal["up", "down", "left", "right"] | None = None, pixels: int = 700) -> ToolMessage:
        return self._run_sync(runtime, "scroll", {"direction": direction, "pixels": pixels}, lambda browser, args: self._call_method_sync(browser, ("scroll",), args.get("direction"), args["pixels"]))

    async def _ascroll(self, runtime: ToolRuntime[Any, Any], direction: Literal["up", "down", "left", "right"] | None = None, pixels: int = 700) -> ToolMessage:
        return await self._run_async(runtime, "scroll", {"direction": direction, "pixels": pixels}, lambda browser, args: self._call_method_async(browser, ("scroll",), args.get("direction"), args["pixels"]))

    def _think(self, runtime: ToolRuntime[Any, Any], reasoning: str) -> ToolMessage:
        return self._success(runtime, "think", {"reasoning": reasoning}, reasoning)

    async def _athink(self, runtime: ToolRuntime[Any, Any], reasoning: str) -> ToolMessage:
        return await self._asuccess(runtime, "think", {"reasoning": reasoning}, reasoning)

    def _type(self, runtime: ToolRuntime[Any, Any], text: str, x: int | None = None, y: int | None = None) -> ToolMessage:
        return self._run_sync(runtime, "type", {"text": text, "x": x, "y": y}, lambda browser, args: self._call_method_sync(browser, ("type", "type_text"), args["text"], args.get("x"), args.get("y")))

    async def _atype(self, runtime: ToolRuntime[Any, Any], text: str, x: int | None = None, y: int | None = None) -> ToolMessage:
        return await self._run_async(runtime, "type", {"text": text, "x": x, "y": y}, lambda browser, args: self._call_method_async(browser, ("type", "type_text"), args["text"], args.get("x"), args.get("y")))

    def _wait(self, runtime: ToolRuntime[Any, Any], time_ms: int = 1000) -> ToolMessage:
        async def wait() -> dict[str, int]:
            await asyncio.sleep(max(time_ms, 0) / 1000)
            return {"waited": time_ms}

        return self._run_sync(runtime, "wait", {"time_ms": time_ms}, lambda _browser, _args: wait(), browser_required=False)

    async def _await(self, runtime: ToolRuntime[Any, Any], time_ms: int = 1000) -> ToolMessage:
        async def wait() -> dict[str, int]:
            await asyncio.sleep(max(time_ms, 0) / 1000)
            return {"waited": time_ms}

        return await self._run_async(runtime, "wait", {"time_ms": time_ms}, lambda _browser, _args: wait(), browser_required=False)

    def _search(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        if self.search is None:
            return self._error(runtime, "search", {"instruction": instruction}, RuntimeError("No search callable was provided."))
        return self._run_sync(
            runtime,
            "search",
            {"instruction": instruction},
            lambda _browser, args: self.search(args["instruction"]),
            browser_required=False,
        )

    async def _asearch(self, runtime: ToolRuntime[Any, Any], instruction: str) -> ToolMessage:
        if self.search is None:
            return self._error(runtime, "search", {"instruction": instruction}, RuntimeError("No search callable was provided."))
        return await self._run_async(
            runtime,
            "search",
            {"instruction": instruction},
            lambda _browser, args: self._maybe_await(self.search(args["instruction"])),
            browser_required=False,
        )
