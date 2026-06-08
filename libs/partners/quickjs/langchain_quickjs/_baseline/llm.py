"""`llm(...)` baseline extension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deepagents._models import resolve_model
from langchain_core.messages import AIMessage, HumanMessage

from langchain_quickjs._baseline.shared import (
    CapacityGuard,
    error,
    infer_model_from_runtime,
    parse_schema,
    parse_structured_output,
    reject_unknown_keys,
    require_dict,
    require_str,
    with_timeout,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.prebuilt import ToolRuntime

    from langchain_quickjs._extensions import ExtensionContext

LLM_SYSTEM_PROMPT = """Baseline capability available:

- `await llm({ prompt, responseSchema? })`
- Use this for one-shot model calls without a tool loop.
- Returns a string by default.
- Returns a native structured value when `responseSchema` is provided.
"""


async def _invoke_one_shot(
    *,
    model: str | BaseChatModel,
    prompt: str,
    response_schema: dict[str, Any] | None,
) -> str:
    resolved = resolve_model(model) if isinstance(model, str) else model
    messages = [HumanMessage(content=prompt)]

    if response_schema is not None:
        schema = (
            response_schema
            if "title" in response_schema
            else {**response_schema, "title": "structured_output"}
        )
        structured_model = resolved.with_structured_output(schema)
        result = await structured_model.ainvoke(messages)
        return json.dumps(result)

    result = await resolved.ainvoke(messages)
    if isinstance(result, str):
        return result
    if isinstance(result, AIMessage):
        return str(result.text)
    return json.dumps(result)


@dataclass(frozen=True)
class LlmExtension:
    """Expose `llm(...)` in the interpreter."""

    max_in_flight: int = 10
    timeout_s: float | None = None
    default_model: str | BaseChatModel | None = None
    system_prompt: str | None = LLM_SYSTEM_PROMPT
    exported_globals: tuple[str, ...] = ("llm",)

    def on_setup(self, ctx: ExtensionContext) -> None:
        ctx.eval(
            "globalThis.llm = async function(options) {"
            " return __ciLlm(options ?? {});"
            "};"
            "undefined"
        )

    def on_eval(self, ctx: ExtensionContext, runtime: ToolRuntime | None) -> None:
        guard = CapacityGuard("llm", self.max_in_flight)

        @ctx.function(name="__ciLlm")
        async def _llm(payload: Any = None) -> Any:
            args = require_dict(payload, "llm")
            reject_unknown_keys(args, allowed={"prompt", "responseSchema"}, call="llm")
            prompt = require_str(args, "prompt", "llm")
            response_schema = parse_schema(args.get("responseSchema"), call="llm")
            model = self.default_model or infer_model_from_runtime(runtime)
            if model is None:
                error(
                    "ERR_MODEL_UNAVAILABLE",
                    "llm model is unavailable in runtime context",
                )

            async def _call() -> Any:
                result = await _invoke_one_shot(
                    model=model,
                    prompt=prompt,
                    response_schema=response_schema,
                )
                if response_schema is None:
                    return result
                return parse_structured_output(result, surface="llm")

            return await guard.run(
                lambda: with_timeout(
                    _call(),
                    timeout_s=self.timeout_s,
                    surface="llm",
                )
            )
