"""Check known model capabilities against the request without network calls."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.language_models import BaseChatModel


def request_tokens(request: ModelRequest) -> int:
    """Estimate the prompt including the system message, tools, and output schema.

    Args:
        request: The request as prepared by upstream middleware.

    Returns:
        A local token estimate; no provider token-counting call is made.
    """
    messages = [*([request.system_message] if request.system_message else []), *request.messages]
    schemas = [convert_to_openai_tool(tool) for tool in request.tools]
    response_format = request.response_format
    if isinstance(response_format, ProviderStrategy):
        schemas.append(response_format.schema_spec.json_schema)
    elif isinstance(response_format, (AutoStrategy, ToolStrategy)):
        schemas.extend(
            spec.json_schema for spec in ToolStrategy(response_format.schema).schema_specs
        )
    return count_tokens_approximately(messages) + (len(json.dumps(schemas)) + 3) // 4


def _required_features(request: ModelRequest) -> set[str]:
    features = {"text_inputs"}
    if request.tools:
        features.add("tool_calling")
    choice = (
        request.tool_choice.get("type")
        if isinstance(request.tool_choice, dict)
        else request.tool_choice
    )
    if choice not in (None, "auto", "none"):
        features.add("tool_choice")
    if isinstance(request.response_format, ProviderStrategy):
        features.add("structured_output")
    elif isinstance(request.response_format, ToolStrategy):
        features.update(("tool_calling", "tool_choice"))
    for message in request.messages:
        if isinstance(message, (HumanMessage, ToolMessage)):
            for block in message.content_blocks:
                kind = block["type"]
                if kind in {"image", "audio", "video"}:
                    features.add(f"{kind}_inputs")
                if kind == "image" and "url" in block:
                    features.add("image_url_inputs")
                elif kind == "file" and block.get("mime_type") == "application/pdf":
                    features.add("pdf_inputs")
                    if isinstance(message, ToolMessage):
                        features.add("pdf_tool_message")
                if isinstance(message, ToolMessage) and kind == "image":
                    features.add("image_tool_message")
    return features


def _output_budget(model: BaseChatModel, request: ModelRequest, estimate: int) -> int:
    budgets = []
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = request.model_settings.get(key, getattr(model, key, None))
        if isinstance(value, int) and not isinstance(value, bool):
            budgets.append(value)
    return max(budgets, default=estimate)


def ineligible_reasons(
    model: BaseChatModel, request: ModelRequest, *, input_tokens: int, output_tokens: int
) -> list[str]:
    """Explain known incompatibilities, preserving support for incomplete profiles.

    Args:
        model: Candidate with capabilities in its LangChain model profile.
        request: Request to check before selecting a candidate.
        input_tokens: Estimated complete prompt size.
        output_tokens: Output reserve when no model/request limit is configured.

    Returns:
        Unsupported features and exceeded token limits. Unknown profile fields are
        not treated as an explicit rejection, matching Deep Agents' content handling.
    """
    profile = model.profile or {}
    reasons = sorted(
        feature for feature in _required_features(request) if profile.get(feature) is False
    )
    if (
        isinstance(request.response_format, AutoStrategy)
        and profile.get("structured_output") is False
        and (profile.get("tool_calling") is False or profile.get("tool_choice") is False)
    ):
        reasons.append("structured_output")
    output = _output_budget(model, request, output_tokens)
    if (limit := profile.get("max_input_tokens")) is not None and input_tokens + output > limit:
        reasons.append("context_window")
    if (limit := profile.get("max_output_tokens")) is not None and output > limit:
        reasons.append("output_budget")
    return reasons
