"""`subagent(...)` baseline extension."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_quickjs._baseline.shared import (
    CapacityGuard,
    error,
    invoke_runtime_tool,
    is_unknown_subagent_type_message,
    parse_schema,
    parse_structured_output,
    reject_unknown_keys,
    require_dict,
    require_str,
    with_timeout,
)

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolRuntime

    from langchain_quickjs._extensions import ExtensionContext

SUBAGENT_SYSTEM_PROMPT = """Baseline capability available:

- `await subagent({ description, subagentType, responseSchema? })`
- Use this to delegate complex work to a named subagent.
- Input keys are camelCase only (`subagentType`, `responseSchema`).
- Returns a string by default.
- Returns a native structured value when `responseSchema` is provided.
"""


@dataclass(frozen=True)
class SubagentExtension:
    """Expose `subagent(...)` in the interpreter."""

    max_in_flight: int = 10
    timeout_s: float | None = None
    system_prompt: str | None = SUBAGENT_SYSTEM_PROMPT
    exported_globals: tuple[str, ...] = ("subagent",)

    def on_setup(self, ctx: ExtensionContext) -> None:
        ctx.eval(
            "globalThis.subagent = async function(options) {"
            " return __ciSubagent(options ?? {});"
            "};"
            "undefined"
        )

    def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime | None) -> None:
        guard = CapacityGuard("subagent", self.max_in_flight)

        @ctx.function(name="__ciSubagent")
        async def _subagent(payload: Any = None) -> Any:
            args = require_dict(payload, "subagent")
            reject_unknown_keys(
                args,
                allowed={"description", "subagentType", "responseSchema"},
                call="subagent",
            )
            description = require_str(args, "description", "subagent")
            subagent_type = require_str(args, "subagentType", "subagent")
            response_schema = parse_schema(args.get("responseSchema"), call="subagent")
            if runtime is None:
                error(
                    "ERR_NOT_SUPPORTED",
                    "subagent is unavailable without a runtime context",
                )

            async def _call() -> Any:
                result = await invoke_runtime_tool(
                    runtime,
                    tool_name="task",
                    payload={
                        "description": description,
                        "subagent_type": subagent_type,
                    },
                )
                result_text = result if isinstance(result, str) else str(result)
                if is_unknown_subagent_type_message(result_text):
                    error("ERR_SUBAGENT_TYPE_UNKNOWN", result_text)
                if response_schema is None:
                    return result_text
                return parse_structured_output(result, surface="subagent")

            return await guard.run(
                lambda: with_timeout(
                    _call(),
                    timeout_s=self.timeout_s,
                    surface="subagent",
                )
            )
