"""Codex Responses API -> LangChain message translation."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _parse_tool_calls_from_items(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract tool calls from response output items."""
    result = []
    for item in items:
        if item.get("type") == "function_call":
            raw_args = item.get("arguments", "")
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {"__raw__": raw_args}
            else:
                parsed_args = raw_args
            result.append(
                {
                    "id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "args": parsed_args,
                    "type": "tool_call",
                }
            )
    return result


def _extract_usage(data: dict[str, Any]) -> dict[str, int] | None:
    """Extract usage metadata from a response.completed event."""
    response = data.get("response", {})
    usage = response.get("usage")
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def collect_response_events(
    events: list[dict[str, Any]],
) -> ChatResult:
    """Collect all SSE events into a ChatResult.

    Processes Responses API events: output_text deltas, function_calls,
    and response.completed for usage metadata.

    Args:
        events: List of parsed SSE event dicts.

    Returns:
        LangChain ChatResult.
    """
    text_parts: list[str] = []
    tool_call_items: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    model = ""

    for event in events:
        event_type = event.get("type", "")

        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            text_parts.append(delta)

        elif event_type == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                tool_call_items.append(item)

        elif event_type == "response.completed":
            usage = _extract_usage(event)
            model = event.get("response", {}).get("model", "")

    content = "".join(text_parts)
    kwargs: dict[str, Any] = {}
    if tool_call_items:
        kwargs["tool_calls"] = _parse_tool_calls_from_items(
            tool_call_items,
        )

    message = AIMessage(content=content, **kwargs)
    message.response_metadata = {"model": model}
    if usage:
        message.usage_metadata = usage  # type: ignore[assignment]

    return ChatResult(generations=[ChatGeneration(message=message)])


def parse_stream_event(
    event: dict[str, Any],
) -> ChatGenerationChunk | None:
    """Parse a single SSE event into a ChatGenerationChunk.

    Only produces chunks for text deltas and completed function calls.
    Returns None for events that don't produce output.

    Args:
        event: Parsed SSE event dict.

    Returns:
        ChatGenerationChunk or None.
    """
    event_type = event.get("type", "")

    if event_type == "response.output_text.delta":
        delta = event.get("delta", "")
        chunk = AIMessageChunk(content=delta)
        return ChatGenerationChunk(message=chunk)

    if event_type == "response.output_item.done":
        item = event.get("item", {})
        if item.get("type") == "function_call":
            tool_calls = _parse_tool_calls_from_items([item])
            # Use tool_call_chunks for streaming so downstream consumers
            # can start processing tool calls incrementally.
            tool_call_chunks = [
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": (
                        tc["args"]
                        if isinstance(tc["args"], str)
                        else json.dumps(tc["args"])
                    ),
                    "index": i,
                    "type": "tool_call_chunk",
                }
                for i, tc in enumerate(tool_calls)
            ]
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=tool_call_chunks,
            )
            return ChatGenerationChunk(message=chunk)

    if event_type == "response.completed":
        usage = _extract_usage(event)
        model = event.get("response", {}).get("model", "")
        chunk = AIMessageChunk(content="")
        chunk.response_metadata = {"model": model}
        if usage:
            chunk.usage_metadata = usage  # type: ignore[assignment]
        return ChatGenerationChunk(message=chunk)

    return None
