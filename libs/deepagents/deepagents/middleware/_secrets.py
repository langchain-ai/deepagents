"""Detect and redact user-supplied API keys from message history and tool I/O.

When a user pastes a plaintext credential into chat (or an `ask_user` answer
that becomes a `HumanMessage`), the model will happily echo it back into
subsequent shell commands -- e.g. as `Authorization: Bearer pylon_api_...`
inside an `execute` tool call. That persists the secret in every recorded
tool-call args blob and tool-result payload for the rest of the run.

`SecretsRedactionMiddleware` mitigates this on three layers:

1. `before_model` rewrites any `HumanMessage` containing a known
    secret pattern, replacing matches with named placeholders
    (`<<PYLON_API_KEY>>`, `<<OPENAI_API_KEY>>`, ...) before the message
    is re-emitted into state. The scrub runs on every model turn so it
    also catches messages appended during HITL resumption.
2. `wrap_tool_call` mirrors the same regex set against the `command` arg
    of `execute` (and the resulting `ToolMessage` content) so any
    literal credential the model still managed to assemble never reaches
    the trace store.
3. A one-shot system note is appended to the conversation when a
    redaction fires, instructing the agent to reference the matching
    environment variable instead of asking the user to re-paste the
    secret.

Patterns are intentionally conservative -- they match well-known
prefixes plus a length floor so generic alphanumerics don't trip the
detector.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.messages import ToolCall
    from langgraph.runtime import Runtime
    from langgraph.types import Command


_SECRET_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("PYLON_API_KEY", "<<PYLON_API_KEY>>", re.compile(r"pylon_api_[a-f0-9]{32,}", re.IGNORECASE)),
    ("LANGSMITH_API_KEY", "<<LANGSMITH_API_KEY>>", re.compile(r"lsv2_(?:pt|sk)_[A-Za-z0-9_-]{20,}")),
    ("ANTHROPIC_API_KEY", "<<ANTHROPIC_API_KEY>>", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OPENAI_PROJECT_KEY", "<<OPENAI_PROJECT_KEY>>", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}")),
    ("OPENAI_API_KEY", "<<OPENAI_API_KEY>>", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    (
        "REDACTED_SECRET",
        "<<REDACTED_SECRET>>",
        re.compile(r"[A-Z][A-Z0-9_]*(?:API_KEY|SECRET|TOKEN)[A-Z0-9_]*\s*[=:]\s*[A-Za-z0-9_-]{20,}"),
    ),
)


def _scrub_text(text: str) -> tuple[str, set[str]]:
    """Replace known credential patterns in `text`, returning the new text and pattern names that matched."""
    matched: set[str] = set()
    scrubbed = text
    for name, placeholder, pattern in _SECRET_PATTERNS:
        if pattern.search(scrubbed):
            matched.add(name)
            scrubbed = pattern.sub(placeholder, scrubbed)
    return scrubbed, matched


def _build_redaction_note(matched: set[str]) -> str:
    """Build the system-level guidance attached to the conversation after a redaction."""
    names = sorted(matched)
    placeholders = ", ".join(f"<<{n}>>" for n in names)
    return (
        f"A secret matching {', '.join(names)} was detected in user input and "
        f"replaced with the placeholder {placeholders}. Do not ask the user to "
        f"re-paste the secret. Reference it via the matching environment "
        f"variable in shell commands "
        f"(e.g. `Authorization: Bearer ${names[0]}`)."
    )


def _scrub_message_content(content: Any) -> tuple[Any, set[str]]:  # noqa: ANN401  # mirrors langchain Message.content typing
    """Scrub secrets out of a message's `content` payload, preserving its shape."""
    if isinstance(content, str):
        scrubbed, matched = _scrub_text(content)
        return scrubbed, matched
    if isinstance(content, list):
        matched: set[str] = set()
        new_blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                new_text, block_matched = _scrub_text(block["text"])
                matched |= block_matched
                if block_matched:
                    new_blocks.append({**block, "text": new_text})
                else:
                    new_blocks.append(block)
            else:
                new_blocks.append(block)
        return new_blocks, matched
    return content, set()


class SecretsRedactionMiddleware(AgentMiddleware):
    """Scrub plaintext API keys from human messages and `execute` tool I/O."""

    def _scrub_messages(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        any_redacted = False
        all_matched: set[str] = set()
        new_messages: list[Any] = []
        note_already_present = any(
            isinstance(m, SystemMessage) and isinstance(m.content, str) and m.content.startswith("A secret matching ") for m in messages
        )

        for msg in messages:
            if isinstance(msg, HumanMessage):
                new_content, matched = _scrub_message_content(msg.content)
                if matched:
                    any_redacted = True
                    all_matched |= matched
                    new_messages.append(msg.model_copy(update={"content": new_content}))
                    continue
            new_messages.append(msg)

        if not any_redacted:
            return None

        if not note_already_present:
            new_messages.append(SystemMessage(content=_build_redaction_note(all_matched)))

        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}

    def before_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Scrub secrets from human messages before each model call."""
        return self._scrub_messages(state)

    async def abefore_model(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Scrub secrets from human messages before each model call (async)."""
        return self._scrub_messages(state)

    def _scrub_tool_request(self, request: ToolCallRequest) -> ToolCallRequest:
        if request.tool_call.get("name") != "execute":
            return request
        args = request.tool_call.get("args") or {}
        command = args.get("command")
        if not isinstance(command, str):
            return request
        scrubbed, matched = _scrub_text(command)
        if not matched:
            return request
        new_tool_call = cast("ToolCall", {**request.tool_call, "args": {**args, "command": scrubbed}})
        return request.override(tool_call=new_tool_call)

    def _scrub_tool_result(self, result: ToolMessage | Command) -> ToolMessage | Command:
        if not isinstance(result, ToolMessage) or result.name != "execute":
            return result
        new_content, matched = _scrub_message_content(result.content)
        if not matched:
            return result
        return result.model_copy(update={"content": new_content})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Scrub secrets from `execute` tool args before invocation and from its result."""
        scrubbed_request = self._scrub_tool_request(request)
        result = handler(scrubbed_request)
        return self._scrub_tool_result(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Scrub secrets from `execute` tool args and result (async)."""
        scrubbed_request = self._scrub_tool_request(request)
        result = await handler(scrubbed_request)
        return self._scrub_tool_result(result)
