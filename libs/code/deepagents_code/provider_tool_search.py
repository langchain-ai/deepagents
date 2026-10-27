"""Provider-native discovery for MCP tools on supported models."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from langchain.agents.middleware import ProviderToolSearchMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    TracePolicy,
    omit_payload,
)
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)


def _supported_endpoint(endpoint: object, provider: str) -> bool:
    """Return whether the endpoint exposes the provider's native request format."""
    if not endpoint:
        return True
    from deepagents_code.model_config import is_langsmith_gateway_host

    parsed = urlparse(str(endpoint))
    if parsed.hostname == f"api.{provider}.com":
        return True
    return (
        parsed.scheme == "https"
        and is_langsmith_gateway_host(parsed.hostname)
        and parsed.path.rstrip("/") in {"/v1", f"/{provider}/v1"}
    )


def _supported_provider(request: ModelRequest) -> str | None:
    """Return a supported provider on direct or LangSmith native-format routes."""
    model = request.model
    profile = model.profile or {}
    capability = profile.get("provider_tool_search")
    if capability is False:
        return None
    name = getattr(model, "model_name", None) or getattr(model, "model", "")
    if not isinstance(name, str):
        return None
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    if type(model) is ChatOpenAI:
        endpoint = getattr(model, "openai_api_base", None)
        if capability is not True and not _supported_endpoint(endpoint, "openai"):
            return None
        if (
            model.use_responses_api is False
            or request.model_settings.get("use_responses_api") is False
        ):
            return None
        if name.startswith("openai/"):
            name = name.removeprefix("openai/")
        if capability is True or name in {
            "gpt-5.4",
            "gpt-5.5",
            "gpt-6-astra",
            "gpt-6-sol",
            "gpt-6-luna",
            "gpt-6.1-sol",
        }:
            return "openai"
    if type(model) is ChatAnthropic:
        endpoint = getattr(model, "anthropic_api_url", None)
        if capability is not True and not _supported_endpoint(endpoint, "anthropic"):
            return None
        if name.startswith("anthropic/"):
            name = name.removeprefix("anthropic/")
        if capability is True or re.fullmatch(
            r"claude-(?:sonnet-4[.-][56]|opus-4[.-][5-8]|haiku-4[.-]5|"
            r"(?:sonnet|opus|haiku)-5(?:[.-]5)?|(?:fable|mythos)-5(?:[.-]1)?)"
            r"(?:-\d{8})?",
            name,
        ):
            return "anthropic"
    return None


def _is_search_block(block: dict, provider: str | None) -> bool:
    kind = block.get("type")
    name = str(block.get("name", ""))
    if kind == "non_standard" and isinstance(block.get("value"), dict):
        return _is_search_block(block["value"], provider)
    if provider != "openai" and (
        kind in {"tool_search_call", "tool_search_output"}
        or (
            kind in {"server_tool_call", "server_tool_result"} and name == "tool_search"
        )
    ):
        return True
    return provider != "anthropic" and (
        kind == "tool_search_tool_result"
        or (
            kind in {"server_tool_use", "server_tool_call", "server_tool_result"}
            and name.startswith("tool_search_tool_")
        )
    )


def _portable_history(
    messages: list[AnyMessage], provider: str | None
) -> list[AnyMessage]:
    """Return history with only compatible native search blocks."""
    result = []
    for message in messages:
        if not isinstance(message, AIMessage) or not isinstance(message.content, list):
            result.append(message)
            continue
        content = [
            block
            for block in message.content
            if not isinstance(block, dict) or not _is_search_block(block, provider)
        ]
        extras = dict(message.additional_kwargs)
        outputs = extras.get("tool_outputs")
        if isinstance(outputs, list):
            extras["tool_outputs"] = [
                block
                for block in outputs
                if not isinstance(block, dict) or not _is_search_block(block, provider)
            ]
        result.append(
            message.model_copy(update={"content": content, "additional_kwargs": extras})
            if content != message.content or extras != message.additional_kwargs
            else message
        )
    return result


class MCPProviderToolSearchMiddleware(AgentMiddleware):
    """Defer only currently exposed MCP tools on verified native endpoints."""

    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(self, tools: Sequence[BaseTool]) -> None:
        """Retain eligible identities without modifying executor registrations."""
        self._tools = {id(tool) for tool in tools}
        self._warned: set[tuple[str, str]] = set()

    def _prepare(self, request: ModelRequest) -> ModelRequest:
        provider = _supported_provider(request)
        eligible = [
            tool
            for tool in request.tools
            if isinstance(tool, BaseTool) and id(tool) in self._tools
        ]
        request = request.override(
            messages=_portable_history(request.messages, provider)
        )
        if not eligible:
            return request
        if provider is None:
            key = (
                type(request.model).__name__,
                str(getattr(request.model, "model_name", "")),
            )
            if key not in self._warned:
                logger.warning(
                    "Native MCP tool search is unavailable for %s; using eager tools. "
                    "Use supported models with OpenAI Responses or Anthropic Messages "
                    "on a direct endpoint or LangSmith gateway.",
                    key,
                )
                self._warned.add(key)
            return request
        native: ProviderToolSearchMiddleware = ProviderToolSearchMiddleware(
            searchable_tools=list(eligible)
        )
        return native._prepare_request(request)

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Return the response without changing executor registrations."""
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Return the async response after runtime model selection."""
        return await handler(self._prepare(request))
