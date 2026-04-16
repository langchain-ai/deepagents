"""WhatsApp adapter for Deep Agents.

Standalone adapter (no base class) that communicates with a Node.js
whatsapp-web.js bridge over HTTP.  Inspired by the Hermes Agent
WhatsApp adapter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import signal
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"

# Text-readable document extensions whose content is injected into the
# message body so the agent can read them inline (capped at 100 KB).
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".log", ".py", ".js", ".ts", ".html", ".css",
})

_MAX_TEXT_INJECT_BYTES = 100 * 1024


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class MessageType(Enum):
    """Inbound message content type."""
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    VOICE = "voice"
    DOCUMENT = "document"


@dataclass
class MessageEvent:
    """An inbound message from WhatsApp."""
    text: str
    message_type: MessageType
    chat_id: str
    chat_name: str
    chat_type: str  # "dm" | "group"
    user_id: str
    user_name: str
    message_id: str | None = None
    media_urls: list[str] = field(default_factory=list)
    media_types: list[str] = field(default_factory=list)
    raw_message: dict[str, Any] = field(default_factory=dict)


@dataclass
class SendResult:
    """Result of sending a message via the bridge."""
    success: bool
    message_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kill_port_process(port: int) -> None:
    """Kill any process listening on *port*."""
    try:
        if _IS_WINDOWS:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    if parts[1].endswith(f":{port}"):
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", parts[4], "/F"],
                                capture_output=True, timeout=5,
                            )
                        except subprocess.SubprocessError:
                            pass
        else:
            result = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["fuser", "-k", f"{port}/tcp"],
                    capture_output=True, timeout=5,
                )
    except Exception:
        pass


def check_whatsapp_requirements() -> bool:
    """Return True if Node.js is available."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_message(content: str) -> str:
    """Convert standard markdown to WhatsApp-compatible formatting.

    WhatsApp supports: *bold*, _italic_, ~strikethrough~, ```code```.
    Standard markdown uses different delimiters for bold/italic/strike,
    so we convert here.  Code blocks and inline code are protected from
    conversion via placeholder substitution.
    """
    if not content:
        return content

    # 1. Protect fenced code blocks
    _FENCE_PH = "\x00FENCE"
    fences: list[str] = []

    def _save_fence(m: re.Match) -> str:
        fences.append(m.group(0))
        return f"{_FENCE_PH}{len(fences) - 1}\x00"

    result = re.sub(r"```[\s\S]*?```", _save_fence, content)

    # 2. Protect inline code
    _CODE_PH = "\x00CODE"
    codes: list[str] = []

    def _save_code(m: re.Match) -> str:
        codes.append(m.group(0))
        return f"{_CODE_PH}{len(codes) - 1}\x00"

    result = re.sub(r"`[^`\n]+`", _save_code, result)

    # 3. Markdown -> WhatsApp
    result = re.sub(r"\*\*(.+?)\*\*", r"*\1*", result)
    result = re.sub(r"__(.+?)__", r"*\1*", result)
    result = re.sub(r"~~(.+?)~~", r"~\1~", result)

    # 4. Headers -> bold
    result = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", result, flags=re.MULTILINE)

    # 5. Links: [text](url) -> text (url)
    result = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", result)

    # 6. Restore protected sections
    for i, fence in enumerate(fences):
        result = result.replace(f"{_FENCE_PH}{i}\x00", fence)
    for i, code in enumerate(codes):
        result = result.replace(f"{_CODE_PH}{i}\x00", code)

    return result


def truncate_message(text: str, max_length: int = 4096) -> list[str]:
    """Split *text* into chunks of at most *max_length* characters.

    Tries to split on newline boundaries.  Falls back to hard cut.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Try to split at the last newline within the limit
        cut = text.rfind("\n", 0, max_length)
        if cut <= 0:
            cut = max_length
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ---------------------------------------------------------------------------
# Group filtering
# ---------------------------------------------------------------------------

def _normalize_whatsapp_id(value: str | None) -> str:
    if not value:
        return ""
    normalized = str(value).strip()
    if ":" in normalized and "@" in normalized:
        normalized = normalized.replace(":", "@", 1)
    return normalized


def _compile_mention_patterns(raw: str | list | None) -> list[re.Pattern]:
    """Parse mention patterns from config into compiled regexes."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except Exception:
            raw = [p.strip() for p in raw.split(",") if p.strip()]
    if not isinstance(raw, list):
        return []
    compiled = []
    for pattern in raw:
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            logger.warning("Invalid mention pattern %r: %s", pattern, exc)
    return compiled


def _bot_ids_from_message(data: dict[str, Any]) -> set[str]:
    return {
        nid for candidate in (data.get("botIds") or [])
        if (nid := _normalize_whatsapp_id(candidate))
    }


def _message_is_reply_to_bot(data: dict[str, Any]) -> bool:
    quoted = _normalize_whatsapp_id(data.get("quotedParticipant"))
    return bool(quoted) and quoted in _bot_ids_from_message(data)


def _message_mentions_bot(data: dict[str, Any]) -> bool:
    bot_ids = _bot_ids_from_message(data)
    if not bot_ids:
        return False
    mentioned = {
        nid for candidate in (data.get("mentionedIds") or [])
        if (nid := _normalize_whatsapp_id(candidate))
    }
    if mentioned & bot_ids:
        return True
    body = str(data.get("body") or "").lower()
    for bot_id in bot_ids:
        bare = bot_id.split("@", 1)[0].lower()
        if bare and (f"@{bare}" in body or bare in body):
            return True
    return False


def _message_matches_patterns(data: dict[str, Any], patterns: list[re.Pattern]) -> bool:
    if not patterns:
        return False
    body = str(data.get("body") or "")
    return any(p.search(body) for p in patterns)


def _clean_bot_mention_text(text: str, data: dict[str, Any]) -> str:
    if not text:
        return text
    bot_ids = _bot_ids_from_message(data)
    cleaned = text
    for bot_id in bot_ids:
        bare = bot_id.split("@", 1)[0]
        if bare:
            cleaned = re.sub(rf"@{re.escape(bare)}\b[,:\-]*\s*", "", cleaned)
    return cleaned.strip() or text


def should_process_message(
    data: dict[str, Any],
    *,
    require_mention: bool,
    free_response_chats: set[str],
    mention_patterns: list[re.Pattern],
) -> bool:
    """Decide whether an inbound message should be processed."""
    if not data.get("isGroup"):
        return True
    chat_id = str(data.get("chatId") or "")
    if chat_id in free_response_chats:
        return True
    if not require_mention:
        return True
    body = str(data.get("body") or "").strip()
    if body.startswith("/"):
        return True
    if _message_is_reply_to_bot(data):
        return True
    if _message_mentions_bot(data):
        return True
    return _message_matches_patterns(data, mention_patterns)
