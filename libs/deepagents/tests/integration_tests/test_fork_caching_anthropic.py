"""End-to-end verification that fork mode preserves Anthropic prompt caching."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from tests.integration_tests.fork_cache_utils import (
    UsageCapture,
    assert_fork_reuses_inherited_message_cache,
    build_fork_agent,
    invoke_twice_with_large_message,
)

HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _anthropic_cache_read_tokens(msg: object) -> int:
    """Return Anthropic cache-read tokens from raw provider usage metadata."""
    # Read raw Anthropic fields directly rather than the normalized
    # ``usage_metadata.input_token_details`` path: LangChain's normalizer
    # zeroes ``cache_creation`` whenever the ephemeral TTL breakdown is
    # present, making that surface unusable for verifying cache writes. See
    # https://github.com/langchain-ai/langchain/issues/36991.
    usage = (getattr(msg, "response_metadata", None) or {}).get("usage", {})
    return int(usage.get("cache_read_input_tokens", 0))


class TestForkPromptCachingAnthropic:
    """Live verification of fork prompt-cache behavior on Anthropic."""

    def test_fork_caches_inherited_messages_not_just_system_prompt(self) -> None:
        """Fork cache reads should include inherited messages, not just the system prompt."""
        capture = UsageCapture(_anthropic_cache_read_tokens)
        agent = build_fork_agent(ChatAnthropic(model_name=HAIKU_MODEL, temperature=0))

        invoke_twice_with_large_message(agent, capture)

        assert_fork_reuses_inherited_message_cache(capture, provider="Anthropic")
