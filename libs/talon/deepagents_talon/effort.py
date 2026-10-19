"""Model-specific reasoning controls for `/effort`."""

from __future__ import annotations

import contextvars
from typing import Any

from deepagents_code.reasoning_effort import supported_efforts_for_model

ACTIVE_EFFORT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "talon_active_effort", default=None
)

# Qwen's OpenAI-compatible API accepts a token budget rather than effort labels.
# https://www.alibabacloud.com/help/en/model-studio/deep-thinking
QWEN_BUDGETS = {"low": 1024, "medium": 4096, "high": 8192, "xhigh": 16384, "max": None}

# Provider profiles can omit effort metadata even for reasoning-capable models.
# https://developers.openai.com/api/docs/models/gpt-5.6
# https://developers.openai.com/api/docs/models/gpt-6.1-sol
# https://docs.baseten.co/development/model-apis/reasoning
_FULL_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
_DOCUMENTED_EFFORTS = {
    **dict.fromkeys(
        (
            "openai:gpt-5.6",
            "openai:gpt-5.6-luna",
            "openai:gpt-5.6-sol",
            "openai:gpt-5.6-terra",
            "openai:gpt-6-luna",
            "openai:gpt-6-sol",
        ),
        _FULL_EFFORTS,
    ),
    **dict.fromkeys(("openai:gpt-6-astra", "openai:gpt-6.1-sol"), _FULL_EFFORTS[1:]),
    **dict.fromkeys(
        (
            "baseten:zai-org/glm-5.2",
            "baseten:zai-org/glm-5.2-fast",
            "baseten:zai-org/glm-5.3",
            "baseten:zai-org/glm-5.3-fast",
            "baseten:zai-org/glm-5.3-flash",
        ),
        ("none", "low", "high", "max"),
    ),
}


def _family(spec: str) -> str:
    provider, _, name = spec.lower().partition(":")
    name = name.rsplit("/", 1)[-1]
    if provider == "openai":
        if name.startswith(("qwen3", "qwen-plus", "qwen-flash")):
            if any(part in name for part in ("instruct", "coder", "omni")):
                return ""
            return "qwen"
        if name.startswith(("glm-4.5", "glm-4.6", "glm-4.7", "glm-5")):
            return "glm"
    return ""


def effort_levels(spec: str) -> tuple[str, ...]:
    """Return only reasoning controls supported by the selected model.

    Args:
        spec: Selected `provider:model` identifier.

    Returns:
        Native effort labels, Qwen budget presets, or GLM thinking modes.
    """
    family = _family(spec)
    if family == "qwen":
        return tuple(QWEN_BUDGETS)
    if family == "glm":
        # GLM-5.3 uses forced thinking; earlier GLM models allow toggling it.
        # https://docs.z.ai/guides/capabilities/thinking-mode
        return ("on",) if "glm-5.3" in spec.lower() else ("off", "on")
    provider, _, name = spec.partition(":")
    if provider == "openai" and "/" in name:
        namespace, name = name.rsplit("/", 1)
        if namespace in {"openai", "anthropic"}:
            spec = f"{namespace}:{name}"
        else:
            return ()
    return supported_efforts_for_model(spec) or _DOCUMENTED_EFFORTS.get(spec.lower(), ())


def effort_settings(spec: str, effort: str) -> dict[str, Any]:
    """Translate an allowlisted control into model invocation parameters.

    Args:
        spec: Selected `provider:model` identifier.
        effort: A supported effort label.

    Returns:
        Provider-specific invocation settings.

    Raises:
        ValueError: When the model does not support the requested control.
    """
    if effort not in effort_levels(spec):
        msg = "Unsupported reasoning effort for this model"
        raise ValueError(msg)
    family = _family(spec)
    if family == "qwen":
        body: dict[str, object] = {}
        if budget := QWEN_BUDGETS[effort]:
            body["thinking_budget"] = budget
        # Thinking-only checkpoints reject a switch to hybrid thinking.
        if "thinking" not in spec.lower():
            body["enable_thinking"] = True
        return {"extra_body": body}
    if family == "glm":
        return {"extra_body": {"thinking": {"type": "enabled" if effort == "on" else "disabled"}}}
    return {"reasoning_effort": effort}


def normalize_effort(argument: str) -> str:
    """Normalize an effort label, including the spelled-out extra-high alias.

    Args:
        argument: Command argument supplied by the operator.

    Returns:
        Canonical lowercase effort label.
    """
    value = " ".join(argument.lower().replace("-", " ").replace("_", " ").split())
    return "xhigh" if value in {"extra high", "x high"} else value
