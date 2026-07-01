"""Live integration coverage for provider-hosted web search."""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from langchain_core.messages import BaseMessage


_MODEL_ENV = "DEEPAGENTS_CODE_WEB_SEARCH_TEST_MODEL"


@contextmanager
def _ignore_oauth_model_warning() -> Iterator[None]:
    """Suppress the dependency's expected experimental class warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`_ChatOpenAICodex` is experimental",
            category=UserWarning,
        )
        yield


def _has_env(name: str) -> bool:
    """Return whether an env var has a non-empty value."""
    return bool(os.environ.get(name))


def _resolve_live_model_spec() -> str:
    """Choose a live model spec for the hosted web search integration test."""
    if model_spec := os.environ.get(_MODEL_ENV):
        return model_spec

    if _has_env("OPENAI_API_KEY") or _has_env("DEEPAGENTS_CODE_OPENAI_API_KEY"):
        return "openai:gpt-5.5"

    from deepagents_code.integrations import openai_codex
    from deepagents_code.model_config import CODEX_PROVIDER

    if openai_codex.is_logged_in():
        return f"{CODEX_PROVIDER}:gpt-5.5"

    pytest.skip(
        f"set {_MODEL_ENV}, OPENAI_API_KEY, or sign in locally to run this test"
    )


def _server_tool_blocks(messages: Sequence[BaseMessage]) -> list[Mapping[str, Any]]:
    """Extract hosted tool content blocks from provider responses."""
    from langchain_core.messages import AIMessage

    blocks: list[Mapping[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        blocks.extend(
            block
            for block in message.content_blocks
            if isinstance(block, dict)
            and str(block.get("type", "")).startswith("server_tool_")
        )
    return blocks


@pytest.mark.timeout(90)
async def test_provider_web_search_calls_provider_tool() -> None:
    """A real provider-backed run returns hosted web-search content blocks."""
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.config import create_model

    model_spec = _resolve_live_model_spec()

    with _ignore_oauth_model_warning():
        model_result = create_model(model_spec)

    agent, _backend = create_cli_agent(
        model=model_result.model,
        assistant_id="itest-model-provider-web-search",
        tools=[{"type": "web_search"}],
        auto_approve=True,
        enable_ask_user=False,
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        enable_interpreter=False,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Use web search before answering. Search for current "
                        "fun things to do in New York City, then answer in one "
                        "short sentence."
                    ),
                }
            ]
        },
        config={"recursion_limit": 12},
    )

    messages = result["messages"]
    blocks = _server_tool_blocks(messages)

    assert any(
        block.get("type") == "server_tool_call" and block.get("name") == "web_search"
        for block in blocks
    )
    assert any(block.get("type") == "server_tool_result" for block in blocks)
