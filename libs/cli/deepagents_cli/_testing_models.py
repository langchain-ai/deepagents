"""Internal chat models used by local integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool


class DeterministicIntegrationChatModel(GenericFakeChatModel):
    """Deterministic chat model for CLI integration tests.

    This subclasses LangChain's `GenericFakeChatModel` so the implementation
    stays aligned with the core fake-chat-model test surface, while overriding
    generation to remain prompt-driven and restart-safe for real CLI server
    integration tests.

    Why the existing `langchain_core` fakes cannot be reused here:

    1. Every core fake (`GenericFakeChatModel`, `FakeListChatModel`,
        `FakeMessagesListChatModel`) pops from an iterator or cycles an index —
        the actual prompt is ignored. CLI integration tests start and stop the
        server process, which resets in-memory state. An iterator-based model
        either raises `StopIteration` or replays from the beginning after a
        restart, producing wrong or missing responses. This model derives output
        solely from the prompt text, so identical input always produces
        identical output regardless of process lifecycle.

    2. The agent runtime calls `model.bind_tools(schemas)` during
        initialization. None of the core fakes implement `bind_tools`, so they
        raise `AttributeError` in any agent-loop context. This model provides a
        no-op passthrough.

    3. The CLI server reads `model.profile` for capability negotiation (e.g.
        `tool_calling`, `max_input_tokens`). Core fakes have no such attribute,
        causing `AttributeError` or silent misconfiguration at runtime.

    Additionally, the compact middleware issues summarization prompts mid-
    conversation. A list-based model cannot distinguish these from normal user
    turns without pre-knowledge of exact call ordering, whereas this model
    detects summary requests by inspecting the prompt content.
    """

    model: str = "fake"
    # Required by `GenericFakeChatModel`, but our override does not consume it.
    messages: object = Field(default_factory=lambda: iter(()))
    profile: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "tool_calling": True,
            "max_input_tokens": 8000,
        }
    )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return self so the agent can bind tool schemas during tests."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Produce a deterministic reply derived from the prompt text.

        Returns:
            A single-message `ChatResult` with deterministic content.
        """
        prompt = "\n".join(
            text
            for message in messages
            if (text := self._stringify_message(message)).strip()
        )
        if self._looks_like_summary_request(prompt):
            content = "integration summary"
        else:
            excerpt = " ".join(prompt.split()[-18:])
            if excerpt:
                content = f"integration reply: {excerpt}"
            else:
                content = "integration reply"

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        """Return the LangChain model type identifier."""
        return "deterministic-integration"

    @staticmethod
    def _stringify_message(message: BaseMessage) -> str:
        """Flatten message content into plain text for deterministic responses.

        Returns:
            Plain-text content extracted from the message.
        """
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _looks_like_summary_request(prompt: str) -> bool:
        """Detect the middleware's summary-generation prompt.

        Returns:
            `True` when the prompt appears to be a summarization request.
        """
        lowered = prompt.lower()
        return (
            "messages to summarize" in lowered
            or "condense the following conversation" in lowered
            or "<summary>" in lowered
        )
