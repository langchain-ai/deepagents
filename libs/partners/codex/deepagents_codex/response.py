"""Codex response -> LangChain message translation."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


def _parse_tool_calls(tool_calls_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls format to LangChain format."""
    result = []
    for tc in tool_calls_data:
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "")
        # LangChain expects args as a dict, not a JSON string
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                parsed_args = {"__raw__": raw_args}
        else:
            parsed_args = raw_args
        result.append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "args": parsed_args,
            "type": "tool_call",
        })
    return result


def _extract_usage(data: dict[str, Any]) -> dict[str, int] | None:
    """Extract usage metadata from API response."""
    usage = data.get("usage")
    if not usage:
        return None
    return {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def parse_chat_response(data: dict[str, Any]) -> ChatResult:
    """Parse a non-streaming chat completion response into a ChatResult.

    Args:
        data: Raw API response dict.

    Returns:
        LangChain ChatResult.
    """
    choices = data.get("choices", [])
    generations = []
    for choice in choices:
        msg_data = choice.get("message", {})
        content = msg_data.get("content") or ""
        tool_calls_data = msg_data.get("tool_calls")

        kwargs: dict[str, Any] = {}
        if tool_calls_data:
            kwargs["tool_calls"] = _parse_tool_calls(tool_calls_data)

        message = AIMessage(content=content, **kwargs)

        # Attach response metadata
        message.response_metadata = {
            "model": data.get("model", ""),
            "finish_reason": choice.get("finish_reason"),
        }

        usage = _extract_usage(data)
        if usage:
            message.usage_metadata = usage  # type: ignore[assignment]

        generations.append(ChatGeneration(message=message))

    return ChatResult(generations=generations)


def parse_stream_chunk(data: dict[str, Any]) -> ChatGenerationChunk:
    """Parse a single streaming chunk into a ChatGenerationChunk.

    Args:
        data: Raw SSE event dict from the streaming response.

    Returns:
        LangChain ChatGenerationChunk.
    """
    choices = data.get("choices", [])
    if not choices:
        return ChatGenerationChunk(
            message=AIMessageChunk(content=""),
        )

    delta = choices[0].get("delta", {})
    content = delta.get("content") or ""

    kwargs: dict[str, Any] = {}
    tool_calls_data = delta.get("tool_calls")
    if tool_calls_data:
        # Streaming tool calls come as incremental updates
        tool_call_chunks = []
        for tc in tool_calls_data:
            fn = tc.get("function", {})
            tool_call_chunks.append({
                "index": tc.get("index", 0),
                "id": tc.get("id"),
                "name": fn.get("name"),
                "args": fn.get("arguments", ""),
            })
        kwargs["tool_call_chunks"] = tool_call_chunks

    chunk = AIMessageChunk(content=content, **kwargs)
    chunk.response_metadata = {
        "model": data.get("model", ""),
        "finish_reason": choices[0].get("finish_reason"),
    }

    usage = _extract_usage(data)
    if usage:
        chunk.usage_metadata = usage  # type: ignore[assignment]

    return ChatGenerationChunk(message=chunk)
