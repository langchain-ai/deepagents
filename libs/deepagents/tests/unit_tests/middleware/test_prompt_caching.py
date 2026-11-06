"""Tests for lazy, request-time provider caching dispatch."""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage

from deepagents import create_deep_agent
from deepagents._excluded_middleware import _apply_excluded_middleware
from deepagents.graph import _apply_custom_middleware
from deepagents.middleware._prompt_caching import _CachingMiddleware
from deepagents.profiles import HarnessProfile
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@pytest.mark.parametrize("concurrent", [False, True])
def test_graph_construction_never_probes_optional_providers(*, concurrent: bool) -> None:
    code = f"""
import importlib.abc
import sys
from concurrent.futures import ThreadPoolExecutor
from deepagents import create_deep_agent
from langchain_core.language_models import GenericFakeChatModel
class RejectProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'langchain_aws', 'langchain_fireworks', 'langchain_anthropic'}}:
            raise AssertionError('Unexpected provider import: ' + fullname)
sys.meta_path.insert(0, RejectProviders())
def build(_):
    create_deep_agent(model=GenericFakeChatModel(messages=iter([])), subagents=[
        {{'name': 'worker', 'description': 'Worker', 'system_prompt': 'Help.'}}
    ])
if {concurrent!r}:
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(build, range(8)))
else:
    for i in range(3):
        build(i)
"""
    result = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr


def test_one_dispatcher_per_agent_stack() -> None:
    agent = MagicMock()
    with (
        patch("deepagents.graph.create_agent", return_value=agent) as create,
        patch("deepagents.middleware.subagents.create_agent", return_value=agent) as create_subagent,
    ):
        create_deep_agent(
            model=GenericFakeChatModel(messages=iter([])),
            subagents=[{"name": "worker", "description": "Worker", "system_prompt": "Help."}],
        )
    calls = [*create.call_args_list, *create_subagent.call_args_list]
    assert {call.kwargs.get("name") for call in calls} == {None, "general-purpose", "worker"}
    for call in calls:
        stack = call.kwargs["middleware"]
        assert sum(isinstance(m, _CachingMiddleware) for m in stack) == 1
        assert not any(m.name in _CachingMiddleware.provider_names for m in stack)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_model_switching_applies_real_provider_caching(*, asynchronous: bool) -> None:
    caching = _CachingMiddleware()
    models = [
        GenericFakeChatModel(messages=iter([])),
        ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake"),
        ChatBedrockConverse.model_construct(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-east-1", client=MagicMock()),
    ]
    seen: list[ModelRequest] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request)
        return AIMessage(content="ok")

    async def ahandler(request: ModelRequest) -> AIMessage:
        return handler(request)

    for model in [*models, models[0], models[1]]:
        request = ModelRequest(model=model, messages=[], state={"messages": []})
        if asynchronous:
            await caching.awrap_model_call(request, ahandler)
        else:
            caching.wrap_model_call(request, handler)
    assert "cache_control" not in seen[0].model_settings
    assert seen[1].model_settings["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert seen[2].model_settings["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert "cache_control" not in seen[3].model_settings
    assert seen[4].model_settings == seen[1].model_settings


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_fireworks_delegate_is_lazy_and_reused(*, asynchronous: bool) -> None:
    module = ModuleType("langchain_fireworks")
    module.ChatFireworks = GenericFakeChatModel
    delegate = MagicMock(spec=AgentMiddleware)
    delegate.wrap_model_call.side_effect = lambda request, handler: handler(request)

    async def awrap(request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(request)

    delegate.awrap_model_call.side_effect = awrap
    factory = MagicMock(return_value=delegate)
    imported = ModuleType("langchain_fireworks.middleware.prompt_caching")
    imported.FireworksPromptCachingMiddleware = factory
    caching = _CachingMiddleware()
    request = ModelRequest(model=GenericFakeChatModel(messages=iter([])), messages=[], state={"messages": []})
    response = AIMessage(content="ok")

    async def ahandler(_request: ModelRequest) -> AIMessage:
        return response

    with (
        patch.dict(sys.modules, langchain_fireworks=module),
        patch("deepagents.middleware._prompt_caching.import_module", return_value=imported) as load,
    ):
        for _ in range(2):
            result = await caching.awrap_model_call(request, ahandler) if asynchronous else caching.wrap_model_call(request, lambda _: response)
            assert result is response
    load.assert_called_once_with("langchain_fireworks.middleware.prompt_caching")
    factory.assert_called_once_with(unsupported_model_behavior="ignore")
    assert (delegate.awrap_model_call if asynchronous else delegate.wrap_model_call).call_count == 2


def test_missing_middleware_is_not_retried_but_transitive_errors_propagate() -> None:
    request = ModelRequest(model=ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake"), messages=[], state={"messages": []})
    response = AIMessage(content="ok")
    caching = _CachingMiddleware()
    with patch(
        "deepagents.middleware._prompt_caching.import_module", side_effect=ModuleNotFoundError(name="langchain_anthropic.middleware.prompt_caching")
    ) as load:
        for _ in range(2):
            assert caching.wrap_model_call(request, lambda _: response) is response
    load.assert_called_once()
    with (
        patch("deepagents.middleware._prompt_caching.import_module", side_effect=ModuleNotFoundError(name="broken_dependency")),
        pytest.raises(ModuleNotFoundError) as error,
    ):
        _CachingMiddleware().wrap_model_call(request, lambda _: response)
    assert error.value.name == "broken_dependency"


@pytest.mark.parametrize("entry", ["AnthropicPromptCachingMiddleware", AnthropicPromptCachingMiddleware, "CachingMiddleware"])
def test_exclusions_disable_caching_without_imports(entry: type[AgentMiddleware] | str) -> None:
    caching = _CachingMiddleware()
    matched_classes = set()
    matched_names = set()
    stack = _apply_excluded_middleware(
        [caching], HarnessProfile(excluded_middleware=frozenset({entry})), matched_classes=matched_classes, matched_names=matched_names
    )
    assert entry in matched_classes | matched_names
    request = ModelRequest(model=ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake"), messages=[], state={"messages": []})
    with patch("deepagents.middleware._prompt_caching.import_module", side_effect=AssertionError("Unexpected import")):
        for middleware in stack:
            middleware.wrap_model_call(request, lambda _: AIMessage(content="ok"))


def test_custom_provider_middleware_disables_automatic_delegate() -> None:
    caching = _CachingMiddleware()
    custom = AnthropicPromptCachingMiddleware(ttl="1h", unsupported_model_behavior="ignore")
    stack = _apply_custom_middleware([caching], [custom])
    request = ModelRequest(model=ChatAnthropic(model_name="claude-sonnet-4-6", anthropic_api_key="fake"), messages=[], state={"messages": []})
    seen: list[ModelRequest] = []

    def handler(request: ModelRequest) -> AIMessage:
        seen.append(request)
        return AIMessage(content="ok")

    with patch("deepagents.middleware._prompt_caching.import_module", side_effect=AssertionError("Unexpected import")):
        stack[0].wrap_model_call(request, lambda request: stack[1].wrap_model_call(request, handler))
    assert seen[0].model_settings["cache_control"]["ttl"] == "1h"
