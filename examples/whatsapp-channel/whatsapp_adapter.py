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


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Callback type: async function receiving a MessageEvent
MessageCallback = Callable[[MessageEvent], Awaitable[None]]


class WhatsAppAdapter:
    """WhatsApp adapter using a Node.js whatsapp-web.js bridge.

    Communicates with the bridge over HTTP on localhost.
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config: dict[str, Any]) -> None:
        self._bridge_port: int = int(config.get("bridge_port", 3000))
        self._bridge_script: str = config.get(
            "bridge_script",
            str(Path(__file__).resolve().parent / "bridge" / "bridge.js"),
        )
        self._session_path: Path = Path(config.get("session_path", "./session"))
        self._reply_prefix: str | None = config.get("reply_prefix")
        self._require_mention: bool = str(
            config.get("require_mention", "false")
        ).lower() in ("true", "1", "yes", "on")
        self._free_response_chats: set[str] = {
            c.strip()
            for c in str(config.get("free_response_chats", "")).split(",")
            if c.strip()
        }
        self._mention_patterns = _compile_mention_patterns(
            config.get("mention_patterns")
        )
        self._on_message: MessageCallback | None = None
        self._bridge_process: subprocess.Popen | None = None
        self._http_session: Any = None  # aiohttp.ClientSession
        self._poll_task: asyncio.Task | None = None
        self._running = False

    def on_message(self, callback: MessageCallback) -> None:
        """Register the callback invoked for each inbound message."""
        self._on_message = callback

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> bool:
        """Start the bridge and begin polling for messages."""
        import aiohttp

        if not check_whatsapp_requirements():
            logger.error("Node.js not found. WhatsApp bridge requires Node.js.")
            return False

        bridge_path = Path(self._bridge_script)
        if not bridge_path.exists():
            logger.error("Bridge script not found: %s", bridge_path)
            return False

        # Auto-install npm deps
        bridge_dir = bridge_path.parent
        if not (bridge_dir / "node_modules").exists():
            print(f"[whatsapp] Installing bridge dependencies...")
            try:
                install_env = os.environ.copy()
                install_env["PUPPETEER_SKIP_DOWNLOAD"] = "true"
                install = subprocess.run(
                    ["sfw", "npm", "install"],
                    cwd=str(bridge_dir),
                    capture_output=True, text=True, timeout=120,
                    env=install_env,
                )
                if install.returncode != 0:
                    print(f"[whatsapp] npm install failed: {install.stderr}")
                    return False
                print("[whatsapp] Dependencies installed")
            except Exception as e:
                print(f"[whatsapp] Failed to install dependencies: {e}")
                return False

        self._session_path.mkdir(parents=True, exist_ok=True)

        # Check for already-running bridge
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{self._bridge_port}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "connected":
                            print(f"[whatsapp] Using existing bridge (connected)")
                            self._http_session = aiohttp.ClientSession()
                            self._running = True
                            self._poll_task = asyncio.create_task(self._poll_messages())
                            return True
        except Exception:
            pass

        # Kill orphaned process on the port
        _kill_port_process(self._bridge_port)
        await asyncio.sleep(1)

        # Build env for bridge subprocess
        bridge_env = os.environ.copy()
        if self._reply_prefix is not None:
            bridge_env["WHATSAPP_REPLY_PREFIX"] = self._reply_prefix

        # Start bridge
        self._bridge_process = subprocess.Popen(
            [
                "node", str(bridge_path),
                "--port", str(self._bridge_port),
                "--session", str(self._session_path),
                "--media-dir", str(self._session_path.parent / "media"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=None if _IS_WINDOWS else os.setsid,
            env=bridge_env,
        )

        # Phase 1: wait for HTTP server (up to 15s)
        http_ready = False
        health_data: dict = {}
        for _ in range(15):
            await asyncio.sleep(1)
            if self._bridge_process.poll() is not None:
                print(f"[whatsapp] Bridge died (exit {self._bridge_process.returncode})")
                return False
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{self._bridge_port}/health",
                        timeout=aiohttp.ClientTimeout(total=2),
                    ) as resp:
                        if resp.status == 200:
                            http_ready = True
                            health_data = await resp.json()
                            if health_data.get("status") == "connected":
                                break
            except Exception:
                continue

        if not http_ready:
            print("[whatsapp] Bridge HTTP server did not start in 15s")
            return False

        # Phase 2: wait for WhatsApp connection (up to 15s more)
        if health_data.get("status") != "connected":
            print("[whatsapp] Bridge HTTP ready, waiting for WhatsApp connection...")
            for _ in range(15):
                await asyncio.sleep(1)
                if self._bridge_process.poll() is not None:
                    print("[whatsapp] Bridge died during connection")
                    return False
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"http://127.0.0.1:{self._bridge_port}/health",
                            timeout=aiohttp.ClientTimeout(total=2),
                        ) as resp:
                            if resp.status == 200:
                                health_data = await resp.json()
                                if health_data.get("status") == "connected":
                                    break
                except Exception:
                    continue
            else:
                print("[whatsapp] WhatsApp not connected after 30s (may need QR scan)")

        self._http_session = aiohttp.ClientSession()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_messages())
        print(f"[whatsapp] Bridge started on port {self._bridge_port}")
        return True

    async def disconnect(self) -> None:
        """Stop polling and shut down the bridge process."""
        self._running = False

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None

        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None

        if self._bridge_process:
            try:
                if _IS_WINDOWS:
                    self._bridge_process.terminate()
                else:
                    os.killpg(os.getpgid(self._bridge_process.pid), signal.SIGTERM)
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    if _IS_WINDOWS:
                        self._bridge_process.kill()
                    else:
                        os.killpg(os.getpgid(self._bridge_process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._bridge_process = None

        print("[whatsapp] Disconnected")

    # -- send ----------------------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
    ) -> SendResult:
        """Send a text message, formatting and chunking as needed."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")

        if not content or not content.strip():
            return SendResult(success=True)

        import aiohttp

        formatted = format_message(content)
        chunks = truncate_message(formatted, self.MAX_MESSAGE_LENGTH)

        last_id = None
        for i, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"chatId": chat_id, "message": chunk}
            if reply_to and i == 0:
                payload["replyTo"] = reply_to
            try:
                async with self._http_session.post(
                    f"http://127.0.0.1:{self._bridge_port}/send",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        last_id = data.get("messageId")
                    else:
                        error = await resp.text()
                        return SendResult(success=False, error=error)
            except Exception as e:
                return SendResult(success=False, error=str(e))

            if len(chunks) > 1:
                await asyncio.sleep(0.3)

        return SendResult(success=True, message_id=last_id)

    async def send_media(
        self,
        chat_id: str,
        file_path: str,
        media_type: str,
        caption: str | None = None,
        file_name: str | None = None,
    ) -> SendResult:
        """Send a media file via the bridge."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")

        import aiohttp

        payload: dict[str, Any] = {
            "chatId": chat_id,
            "filePath": file_path,
            "mediaType": media_type,
        }
        if caption:
            payload["caption"] = caption
        if file_name:
            payload["fileName"] = file_name

        try:
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/send-media",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(success=True, message_id=data.get("messageId"))
                else:
                    error = await resp.text()
                    return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator (fire-and-forget)."""
        if not self._running or not self._http_session:
            return
        try:
            import aiohttp
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/typing",
                json={"chatId": chat_id},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                pass  # Fire-and-forget; response consumed to avoid leaks
        except Exception:
            pass

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
    ) -> SendResult:
        """Edit a previously sent message."""
        if not self._running or not self._http_session:
            return SendResult(success=False, error="Not connected")
        try:
            import aiohttp
            async with self._http_session.post(
                f"http://127.0.0.1:{self._bridge_port}/edit",
                json={
                    "chatId": chat_id,
                    "messageId": message_id,
                    "message": content,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return SendResult(success=True, message_id=message_id)
                else:
                    error = await resp.text()
                    return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        """Get information about a chat."""
        if not self._running or not self._http_session:
            return {"name": chat_id, "type": "dm"}
        try:
            import aiohttp
            async with self._http_session.get(
                f"http://127.0.0.1:{self._bridge_port}/chat/{chat_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "name": data.get("name", chat_id),
                        "type": "group" if data.get("isGroup") else "dm",
                        "participants": data.get("participants", []),
                    }
        except Exception:
            pass
        return {"name": chat_id, "type": "dm"}

    # -- receive -------------------------------------------------------------

    async def _poll_messages(self) -> None:
        """Poll the bridge for inbound messages and dispatch them."""
        import aiohttp

        while self._running:
            if not self._http_session:
                break
            try:
                async with self._http_session.get(
                    f"http://127.0.0.1:{self._bridge_port}/messages",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg_data in messages:
                            event = self._build_message_event(msg_data)
                            if event and self._on_message:
                                try:
                                    await self._on_message(event)
                                except Exception:
                                    logger.exception("Error in message callback")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Poll error: %s", e)
                await asyncio.sleep(5)

            await asyncio.sleep(1)

    def _build_message_event(self, data: dict[str, Any]) -> MessageEvent | None:
        """Build a MessageEvent from raw bridge data."""
        if not should_process_message(
            data,
            require_mention=self._require_mention,
            free_response_chats=self._free_response_chats,
            mention_patterns=self._mention_patterns,
        ):
            return None

        # Determine message type
        msg_type = MessageType.TEXT
        if data.get("hasMedia"):
            media_type = data.get("mediaType", "")
            if "image" in media_type:
                msg_type = MessageType.PHOTO
            elif "video" in media_type:
                msg_type = MessageType.VIDEO
            elif "audio" in media_type or "ptt" in media_type:
                msg_type = MessageType.VOICE
            else:
                msg_type = MessageType.DOCUMENT

        is_group = data.get("isGroup", False)

        media_urls: list[str] = list(data.get("mediaUrls") or [])
        media_types: list[str] = []
        for url in media_urls:
            if msg_type == MessageType.PHOTO:
                media_types.append("image/jpeg")
            elif msg_type == MessageType.VOICE:
                media_types.append("audio/ogg")
            elif msg_type == MessageType.VIDEO:
                media_types.append("video/mp4")
            elif msg_type == MessageType.DOCUMENT:
                media_types.append("application/octet-stream")
            else:
                media_types.append("unknown")

        # Build body, cleaning bot mentions in groups
        body = data.get("body", "")
        if is_group:
            body = _clean_bot_mention_text(body, data)

        # Inject text content from readable documents
        if msg_type == MessageType.DOCUMENT and media_urls:
            for doc_path in media_urls:
                ext = Path(doc_path).suffix.lower()
                if ext in _TEXT_EXTENSIONS:
                    try:
                        size = Path(doc_path).stat().st_size
                        if size > _MAX_TEXT_INJECT_BYTES:
                            continue
                        content = Path(doc_path).read_text(errors="replace")
                        display_name = Path(doc_path).name
                        injection = f"[Content of {display_name}]:\n{content}"
                        body = f"{injection}\n\n{body}" if body else injection
                    except Exception:
                        pass

        return MessageEvent(
            text=body,
            message_type=msg_type,
            chat_id=data.get("chatId", ""),
            chat_name=data.get("chatName", ""),
            chat_type="group" if is_group else "dm",
            user_id=data.get("senderId", ""),
            user_name=data.get("senderName", ""),
            message_id=data.get("messageId"),
            media_urls=media_urls,
            media_types=media_types,
            raw_message=data,
        )
