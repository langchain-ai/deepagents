"""Let Claude drop preserved thinking blocks whose prompt prefix changed.

Claude 5+ models bind each thinking block to the prompt prefix that produced it
(the system prompt, the tools, and every earlier message). When a later request
replays a block after that prefix changed, accounts under the history-editing
check get a 400 on every following request. Deep agents change the prefix on
their own: summarization replaces older history, and memory and skills rewrite
the system prompt. Setting `thinking.block_binding.prefix_mismatch_behavior` to
`"drop_block"` makes the API drop the stale blocks instead.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard, cast

from langchain.agents.middleware.types import AgentMiddleware, TracePolicy, omit_payload

from deepagents.middleware._utils import is_anthropic_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
    from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)

THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
"""Beta flag that allows `thinking.block_binding` in a request.

Drop it once the controls are generally available.
"""

_MIN_CLAUDE_MAJOR = 5
"""First Claude major version with preserved thinking."""

_CLAUDE_MAJOR_PATTERN = re.compile(r"^claude-[a-z]+-(\d+)")
"""Read the major version from family-first ids such as `claude-opus-5-5`.

Legacy `claude-<major>-<minor>-<family>` ids (`claude-3-5-haiku-*`) don't
match, so they stay excluded.
"""

_DIRECT_API_LLM_TYPE = "anthropic-chat"
"""`_llm_type` of a `ChatAnthropic` that calls the Claude API directly.

Subclasses that route through another platform (such as Amazon Bedrock)
override `_llm_type`, and are skipped.
"""


def _is_claude_5_or_later(model_name: str) -> bool:
    match = _CLAUDE_MAJOR_PATTERN.match(model_name.lower())
    return match is not None and int(match.group(1)) >= _MIN_CLAUDE_MAJOR


def _forces_tool_call(request: ModelRequest) -> bool:
    """Return whether the request forces a tool call or asks for structured output.

    A forced tool call can't be combined with thinking, so the request's
    thinking setting is left as the caller configured it.
    """
    if request.response_format is not None:
        return True
    choice = request.tool_choice
    if isinstance(choice, Mapping):
        return choice.get("type") in {"any", "tool"}
    return isinstance(choice, str) and choice not in {"auto", "none"}


def _sets_reasoning_effort(model: ChatAnthropic, settings: Mapping[str, Any]) -> bool:
    return bool(model.reasoning_effort or settings.get("effort") or settings.get("reasoning_effort"))


def _default_thinking(model: ChatAnthropic, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the thinking config a Claude 5+ model runs when the request omits `thinking`.

    Claude 5+ models run adaptive thinking by default, so naming it explicitly
    doesn't turn thinking on. `ChatAnthropic` asks for summarized display when a
    reasoning effort is set and `thinking` is omitted; an explicit `thinking`
    disables that default, so it's carried over here.
    """
    thinking: dict[str, Any] = {"type": "adaptive"}
    profile = model.profile if isinstance(model.profile, Mapping) else {}
    levels = profile.get("reasoning_effort_levels")
    if _sets_reasoning_effort(model, settings) and isinstance(levels, (list, tuple)) and "xhigh" in levels:
        thinking["display"] = "summarized"
    return thinking


def _merged_betas(model: ChatAnthropic, settings: Mapping[str, Any]) -> list[Any] | None:
    """Return the request's betas with the binding beta added, or `None` if they're malformed.

    Call-time `betas` replace `model_kwargs["betas"]`, which replace
    `ChatAnthropic.betas`, so the first one set is the list the request sends.
    """
    if "betas" in settings:
        betas = settings["betas"]
    elif "betas" in model.model_kwargs:
        betas = model.model_kwargs["betas"]
    else:
        betas = model.betas
    if betas is None:
        betas = []
    if not isinstance(betas, (list, tuple)):
        logger.warning("Skipping the Anthropic thinking binding: `betas` is a %s, not a list.", type(betas).__name__)
        return None
    return list(dict.fromkeys([*betas, THINKING_BINDING_BETA]))


def _bind_thinking(thinking: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return `thinking` with `drop_block` as the default mismatch behavior.

    Returns `None` when the binding doesn't apply: thinking isn't adaptive (the
    API accepts `block_binding` only with adaptive thinking), or the existing
    `block_binding` isn't a mapping.
    """
    if thinking.get("type") != "adaptive":
        return None
    block_binding = thinking.get("block_binding")
    if block_binding is not None and not isinstance(block_binding, Mapping):
        logger.warning(
            "Skipping the Anthropic thinking binding: `thinking.block_binding` is a %s, not a mapping.",
            type(block_binding).__name__,
        )
        return None
    binding = dict(block_binding or {})
    binding.setdefault("prefix_mismatch_behavior", "drop_block")
    return {**thinking, "block_binding": binding}


def _is_direct_claude_5(model: object) -> TypeGuard[ChatAnthropic]:
    """Return whether `model` is a `ChatAnthropic` calling a Claude 5+ model on the Claude API.

    Subclasses that route through another platform (such as Amazon Bedrock)
    report a different `_llm_type`; they're skipped because the binding beta
    isn't confirmed there.
    """
    return (
        is_anthropic_chat_model(model)
        and model._llm_type == _DIRECT_API_LLM_TYPE  # how langchain-anthropic tells platform subclasses apart
        and _is_claude_5_or_later(model.model)
    )


def _request_thinking(request: ModelRequest, model: ChatAnthropic) -> object:
    """Return the thinking config the request will send, or `None` to leave the request alone."""
    # `ChatAnthropic.thinking` beats a call-time `thinking`, which beats `model_kwargs`.
    thinking = model.thinking
    if thinking is None:
        thinking = request.model_settings.get("thinking", model.model_kwargs.get("thinking"))
    if thinking is None and not _forces_tool_call(request):
        thinking = _default_thinking(model, request.model_settings)
    return thinking


def apply_thinking_binding(request: ModelRequest) -> ModelRequest:
    """Return `request` set up to drop stale thinking blocks, or unchanged if it doesn't apply.

    Applies only to `ChatAnthropic` calling the Claude API directly with a
    Claude 5+ model and adaptive thinking (set explicitly or the model's
    default). Never turns thinking on, and keeps a `prefix_mismatch_behavior`
    the caller already set.

    Args:
        request: The model request about to be sent.

    Returns:
        The request with `thinking.block_binding` and the binding beta set, or
            `request` itself.
    """
    model = request.model
    if not _is_direct_claude_5(model):
        return request
    thinking = _request_thinking(request, model)
    if not isinstance(thinking, Mapping):
        if thinking is not None:
            logger.warning("Skipping the Anthropic thinking binding: `thinking` is a %s, not a mapping.", type(thinking).__name__)
        return request
    bound = _bind_thinking(cast("Mapping[str, Any]", thinking))
    betas = _merged_betas(model, request.model_settings)
    if bound is None or betas is None:
        return request
    settings = {**request.model_settings, "betas": betas}
    if model.thinking is not None:
        # A call-time `thinking` can't override the model field, so swap in a copy.
        return request.override(model=model.model_copy(update={"thinking": bound}), model_settings=settings)
    return request.override(model_settings={**settings, "thinking": bound})


class _AnthropicThinkingBindingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Have Claude drop thinking blocks invalidated by a prompt prefix change, instead of failing.

    `create_deep_agent` adds it next to prompt caching on every stack, after
    caller middleware, so it sees the model actually called. Exclude it with
    `excluded_middleware={"AnthropicThinkingBindingMiddleware"}` on a harness
    profile.
    """

    trace_policy = TracePolicy(process_inputs=omit_payload)

    serialized_name: ClassVar[str] = "AnthropicThinkingBindingMiddleware"
    """Public name used for string-form exclusion."""

    @property
    def name(self) -> str:
        """Use the public alias, so `excluded_middleware` can name it."""
        return self.serialized_name

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Add the thinking binding to Claude 5+ requests."""
        return handler(apply_thinking_binding(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Add the thinking binding to Claude 5+ requests (async version)."""
        return await handler(apply_thinking_binding(request))
