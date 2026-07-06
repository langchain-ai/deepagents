"""Internal fake chat models.

Shared by local integration tests and the `dcode tools list` tool-enumeration
path (`tool_catalog._CatalogModel`), which both need a model that binds tools
during agent compilation but is never invoked.
"""

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


# Prompt markers that drive `ToolCallingIntegrationChatModel`. Each marker is the
# full token (including the trailing `=`); the file path follows on the same line,
# e.g. `DCA_TEST_WRITE_FILE=/tmp/out.txt`. These are the single source of truth
# shared with the integration tests, so the model and tests cannot drift apart.
DCA_TEST_WRITE_FILE_MARKER = "DCA_TEST_WRITE_FILE="
DCA_TEST_DELEGATE_WRITE_MARKER = "DCA_TEST_DELEGATE_WRITE="
DCA_SUBAGENT_WRITE_FILE_MARKER = "DCA_SUBAGENT_WRITE_FILE="

# Distinct file contents per write path, so a test asserting on file content can
# prove which branch executed — in particular that subagent mode delegated through
# the `task` tool rather than writing directly.
TOP_LEVEL_WRITE_CONTENT = "auto-approved"
SUBAGENT_WRITE_CONTENT = "auto-approved-subagent"


_TOOL_BINDING_MODEL_PROFILE: dict[str, Any] = {
    "tool_calling": True,
    "max_input_tokens": 8000,
}
"""Minimal capability profile the agent runtime reads while compiling a model.

Only `tool_calling` is load-bearing — the agent negotiates tool support at
setup. `max_input_tokens` is required by the profile surface but inert here:
these models are compiled to bind tools and are never invoked, so no token
budget ever applies. Defined once so both the fakes below and
`tool_catalog._CatalogModel` share a single source of truth.
"""


class _ToolBindingFakeModel(GenericFakeChatModel):
    """Base for fake chat models that must bind tools but are never invoked.

    The agent runtime calls `model.bind_tools(schemas)` and reads
    `model.profile` while compiling the graph; `GenericFakeChatModel` provides
    neither, so a bare fake raises `AttributeError` in any agent-loop context.
    This base supplies a no-op `bind_tools` passthrough and the minimal
    `profile`, leaving subclasses to add generation behavior (tests) or nothing
    at all (tool enumeration).
    """

    # Required by `GenericFakeChatModel`, but subclasses never consume it.
    messages: object = Field(default_factory=lambda: iter(()))
    profile: dict[str, Any] | None = Field(
        default_factory=lambda: dict(_TOOL_BINDING_MODEL_PROFILE)
    )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return self so the agent can bind tool schemas without a real model."""
        return self


class DeterministicIntegrationChatModel(_ToolBindingFakeModel):
    """Deterministic chat model for integration tests.

    This subclasses LangChain's `GenericFakeChatModel` so the implementation
    stays aligned with the core fake-chat-model test surface, while overriding
    generation to remain prompt-driven and restart-safe for real CLI server
    integration tests.

    Why the existing `langchain_core` fakes cannot be reused here:

    1. Every core fake (`GenericFakeChatModel`, `FakeListChatModel`,
        `FakeMessagesListChatModel`) pops from an iterator or cycles an index —
        the actual prompt is ignored. App integration tests start and stop the
        server process, which resets in-memory state. An iterator-based model
        either raises `StopIteration` or replays from the beginning after a
        restart, producing wrong or missing responses. This model derives output
        solely from the prompt text, so identical input always produces
        identical output regardless of process lifecycle.

    2. The agent runtime calls `model.bind_tools(schemas)` during
        initialization. None of the core fakes implement `bind_tools`, so they
        raise `AttributeError` in any agent-loop context. This model provides a
        no-op passthrough.

    3. The app server reads `model.profile` for capability negotiation (e.g.
        `tool_calling`, `max_input_tokens`). Core fakes have no such attribute,
        causing `AttributeError` or silent misconfiguration at runtime.

    Additionally, the compact middleware issues summarization prompts mid-
    conversation. A list-based model cannot distinguish these from normal user
    turns without pre-knowledge of exact call ordering, whereas this model
    detects summary requests by inspecting the prompt content.
    """

    model: str = "fake"
    # `messages`, `profile`, and the `bind_tools` passthrough are inherited from
    # `_ToolBindingFakeModel`; this model adds only prompt-driven generation.

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
        """LangChain model type identifier."""
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


def _extract_marker_path(prompt: str, marker: str) -> str:
    """Extract the file path that follows a prompt marker on the same line.

    Args:
        prompt: The flattened prompt text.
        marker: The marker token (including its trailing `=`) to locate.

    Returns:
        The stripped file path immediately following the marker.

    Raises:
        ValueError: If the marker is present but not followed by a path, so a
            malformed test prompt fails loudly here instead of silently
            degrading to a `"done"` reply or raising an opaque `IndexError`.
    """
    _, _, tail = prompt.partition(marker)
    lines = tail.splitlines()
    file_path = lines[0].strip() if lines else ""
    if not file_path:
        msg = (
            f"Test model saw marker {marker!r} but found no file path after it; "
            f"check the integration-test prompt construction."
        )
        raise ValueError(msg)
    return file_path


def _tool_call_result(name: str, args: dict[str, Any], call_id: str) -> ChatResult:
    """Build a single-tool-call `ChatResult`.

    Returns:
        A `ChatResult` wrapping an `AIMessage` with exactly one tool call.
    """
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": args,
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            )
        ]
    )


class ToolCallingIntegrationChatModel(DeterministicIntegrationChatModel):
    """Deterministic tool-calling model for auto-approve integration tests.

    Generation is driven entirely by prompt markers (the module-level `DCA_*`
    constants), so output is restart-safe and independent of call ordering — the
    same rationale as the parent `DeterministicIntegrationChatModel`:

    - `DCA_TEST_WRITE_FILE=<path>` emits a top-level `write_file` call.
    - `DCA_TEST_DELEGATE_WRITE=<path>` emits a `task` call delegating to the
      `general-purpose` subagent, whose prompt then carries
      `DCA_SUBAGENT_WRITE_FILE=<path>` to trigger the subagent's `write_file`.
    - `DCA_SUBAGENT_WRITE_FILE=<path>` emits the subagent's `write_file` call.

    Each marker fires only on the agent's first turn (no prior `ToolMessage`),
    so once the tool result returns the model replies with a plain `"done"` and
    the agent loop terminates instead of re-issuing the tool call.
    """

    # Only `_generate` is overridden; the inherited `_stream` would bypass this
    # marker dispatch entirely, so streaming must be disabled.
    disable_streaming: bool = True

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """Emit a deterministic tool call (or terminal reply) from prompt markers.

        The `has_tool_result` guard ensures each marker fires only on the
        agent's first turn; once a `ToolMessage` is present the model returns
        `"done"` so the agent loop terminates.

        A recognized marker with no file path raises `ValueError` (via
        `_extract_marker_path`) rather than degrading to `"done"`.

        Returns:
            A single-message `ChatResult`: an `AIMessage` carrying a `task` or
            `write_file` tool call when a marker matches and no tool result is
            present yet, otherwise a plain `"done"` reply.
        """
        prompt = "\n".join(
            text
            for message in messages
            if (text := self._stringify_message(message)).strip()
        )
        has_tool_result = any(message.type == "tool" for message in messages)
        if not has_tool_result:
            for marker, build_tool_call in self._marker_dispatch():
                if marker in prompt:
                    name, args, call_id = build_tool_call(
                        _extract_marker_path(prompt, marker)
                    )
                    return _tool_call_result(name, args, call_id)

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="done"))]
        )

    @staticmethod
    def _marker_dispatch() -> tuple[
        tuple[str, Callable[[str], tuple[str, dict[str, Any], str]]], ...
    ]:
        """Return ordered `(marker, tool-call builder)` pairs.

        The delegate marker is checked before the plain write markers because
        its emitted `task` description embeds `DCA_SUBAGENT_WRITE_FILE=`; the
        first marker found in the prompt wins.

        Returns:
            Marker-to-builder pairs in precedence order. Each builder maps an
            extracted file path to a `(tool_name, args, call_id)` triple.
        """
        return (
            (
                DCA_TEST_DELEGATE_WRITE_MARKER,
                lambda path: (
                    "task",
                    {
                        "description": f"{DCA_SUBAGENT_WRITE_FILE_MARKER}{path}",
                        "subagent_type": "general-purpose",
                    },
                    "call_task",
                ),
            ),
            (
                DCA_SUBAGENT_WRITE_FILE_MARKER,
                lambda path: (
                    "write_file",
                    {"file_path": path, "content": SUBAGENT_WRITE_CONTENT},
                    "call_write_file",
                ),
            ),
            (
                DCA_TEST_WRITE_FILE_MARKER,
                lambda path: (
                    "write_file",
                    {"file_path": path, "content": TOP_LEVEL_WRITE_CONTENT},
                    "call_write_file",
                ),
            ),
        )
