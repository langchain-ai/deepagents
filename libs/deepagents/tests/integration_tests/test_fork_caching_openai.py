"""End-to-end verification that fork mode preserves OpenAI prompt caching."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from tests.integration_tests.fork_cache_utils import (
    UsageCapture,
    assert_fork_reuses_inherited_message_cache,
    build_fork_agent,
    invoke_twice_with_large_message,
)

OPENAI_MODEL = "gpt-5.4"


def _openai_cache_read_tokens(msg: object) -> int:
    """Return OpenAI cached prompt tokens from the raw provider usage shape."""
    response_metadata = getattr(msg, "response_metadata", None) or {}
    usage = response_metadata["token_usage"]
    prompt_details = usage["prompt_tokens_details"]
    return int(prompt_details["cached_tokens"])


class TestForkPromptCachingOpenAI:
    """Live verification of fork prompt-cache behavior on OpenAI."""

    def test_fork_caches_inherited_messages_not_just_system_prompt(self) -> None:
        """Fork cache reads should include inherited messages, not just the system prompt."""
        capture = UsageCapture(_openai_cache_read_tokens)
        agent = build_fork_agent(ChatOpenAI(model_name=OPENAI_MODEL, temperature=0))

        invoke_twice_with_large_message(agent, capture)

        assert_fork_reuses_inherited_message_cache(capture, provider="OpenAI")
