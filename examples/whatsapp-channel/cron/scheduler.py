"""Cron tick loop and single-job runner for the WhatsApp channel example."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

SILENT_MARKER = "[SILENT]"

_CRON_HINT = (
    "[SYSTEM: You are running as a scheduled cron job. "
    "DELIVERY: Your final response will be automatically delivered to the "
    "user — do not call tools that send messages yourself. "
    "SILENT: If there is genuinely nothing new to report, respond with "
    f"exactly \"{SILENT_MARKER}\" (nothing else) to suppress delivery.]\n\n"
)


def _build_prompt(user_prompt: str) -> str:
    """Prepend the cron-execution hint to *user_prompt*."""
    return _CRON_HINT + (user_prompt or "")


def _extract_final_text(agent_output: Any) -> str:
    """Extract the final ``AIMessage`` text from an agent output dict.

    Mirrors the extraction logic in ``main.py`` so cron-run results look like
    live-chat responses.
    """
    if not agent_output:
        return ""
    messages = agent_output.get("messages", []) if isinstance(agent_output, dict) else []
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if not content:
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for block in reversed(content):
                if isinstance(block, str):
                    return block
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
    return ""
