"""Convert between LangChain message content and ACP content."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
)

from deepagents_acp.utils import (
    convert_audio_block_to_content_blocks,
    convert_embedded_resource_block_to_content_blocks,
    convert_image_block_to_content_blocks,
    convert_resource_block_to_content_blocks,
    convert_text_block_to_content_blocks,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from acp.schema import StopReason
    from langchain_core.messages import BaseMessage

_STOP_REASONS: dict[str, StopReason] = {
    "max_tokens": "max_tokens",  # Anthropic
    "length": "max_tokens",  # OpenAI Chat Completions
    "max_output_tokens": "max_tokens",  # OpenAI Responses `incomplete_details.reason`
    "refusal": "refusal",  # Anthropic
    "content_filter": "refusal",  # OpenAI
    "safety": "refusal",  # Gemini
    "prohibited_content": "refusal",  # Gemini
    "blocklist": "refusal",  # Gemini
    "spii": "refusal",  # Gemini
}
"""Provider stop and finish reasons that ACP reports as something other than `end_turn`."""


def _content_block(
    block: Mapping[str, Any],
) -> TextContentBlock | ImageContentBlock | AudioContentBlock | None:
    """Convert one normalized LangChain content block into an ACP block.

    Returns `None` when the block has no ACP equivalent, so callers skip it.
    Empty text is dropped because it renders nothing.
    """
    block_type = block.get("type")
    if block_type in {"text", "refusal"}:
        # OpenAI Responses streams a refusal as its own block; show it like text.
        text = block.get(block_type)
        return TextContentBlock(type="text", text=text) if isinstance(text, str) and text else None
    data = block.get("base64")
    mime_type = block.get("mime_type")
    if not (isinstance(data, str) and isinstance(mime_type, str)):
        return None
    if block_type == "image":
        return ImageContentBlock(type="image", data=data, mime_type=mime_type, uri=block.get("url"))
    if block_type == "audio":
        return AudioContentBlock(type="audio", data=data, mime_type=mime_type)
    return None


def _content_blocks(
    message: Any,
) -> list[TextContentBlock | ImageContentBlock | AudioContentBlock]:
    """Convert normalized user-message content into ACP blocks.

    Reasoning blocks are not expected on user messages. They have no ACP
    equivalent here and are dropped.
    """
    return [block for item in message.content_blocks if (block := _content_block(item))]


def _visible_reasoning(block: Mapping[str, Any]) -> str | None:
    """Return reasoning text a provider exposed in plaintext, or `None`.

    Redacted or encrypted reasoning arrives as a `non_standard` block with no
    `reasoning` string, so it never returns here. Whitespace is kept: reasoning
    streams one delta at a time, and a delta that holds only a space or a line
    break carries the word and paragraph breaks of the finished thought.
    """
    reasoning = block.get("reasoning")
    if block.get("type") != "reasoning" or not isinstance(reasoning, str):
        return None
    return reasoning or None


def _content_updates(
    message: Any,
    message_id: str | None = None,
) -> list[AgentMessageChunk | AgentThoughtChunk]:
    """Convert normalized assistant content into ordered ACP updates.

    Both the live stream and session replay project through this function, so
    block ordering and block-type support cannot drift between them. Live
    chunks carry no message ID; replay passes the persisted one.
    """
    if message.additional_kwargs.get("lc_source") == "local_context":
        return []
    updates: list[AgentMessageChunk | AgentThoughtChunk] = []
    for block in message.content_blocks:
        if reasoning := _visible_reasoning(block):
            updates.append(
                AgentThoughtChunk(
                    session_update="agent_thought_chunk",
                    content=TextContentBlock(type="text", text=reasoning),
                    message_id=message_id,
                )
            )
        elif content := _content_block(block):
            updates.append(
                AgentMessageChunk(
                    session_update="agent_message_chunk",
                    content=content,
                    message_id=message_id,
                )
            )
    return updates


def _stop_reason(message: BaseMessage) -> StopReason | None:
    """Map a final model message's provider stop reason to an ACP stop reason."""
    if message.additional_kwargs.get("refusal") or any(
        block.get("type") == "refusal" for block in message.content_blocks
    ):
        return "refusal"  # OpenAI reports refusals outside the stop reason
    metadata = message.response_metadata or {}
    incomplete = metadata.get("incomplete_details")
    reasons = (
        metadata.get("stop_reason"),
        metadata.get("finish_reason"),
        incomplete.get("reason") if isinstance(incomplete, dict) else None,
    )
    for reason in reasons:
        if reason and (mapped := _STOP_REASONS.get(str(reason).lower())):
            return mapped
    return None


def _prompt_content(prompt: Sequence[Any], cwd: str) -> list[dict[str, Any]]:
    """Convert ACP content blocks to LangChain multimodal content."""
    content: list[dict[str, Any]] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            content.extend(convert_text_block_to_content_blocks(block))
        elif isinstance(block, ImageContentBlock):
            content.extend(convert_image_block_to_content_blocks(block))
        elif isinstance(block, AudioContentBlock):
            content.extend(convert_audio_block_to_content_blocks(block))
        elif isinstance(block, ResourceContentBlock):
            content.extend(convert_resource_block_to_content_blocks(block, root_dir=cwd))
        elif isinstance(block, EmbeddedResourceContentBlock):
            content.extend(convert_embedded_resource_block_to_content_blocks(block))
    return content
