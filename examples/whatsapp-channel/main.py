"""Entry point: connects a Deep Agent to WhatsApp via the bridge adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from tools import fetch_url, http_request, web_search
from config import build_adapter_config
from cron import build_cron_tools, origin_ctx, start_ticker
from mcp_tools import MCPSessionManager, get_mcp_tools
from whatsapp_adapter import (
    MessageEvent,
    MessageType,
    WhatsAppAdapter,
    _build_inbound_content,
    extract_markdown_media,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Status-message helpers
# ---------------------------------------------------------------------------

_MEDIA_ATTACH_INSTRUCTIONS = (
    "You are running as a WhatsApp assistant. "
    "CRITICAL: Do NOT send any text response until your task is fully complete. "
    "Any text you output is immediately delivered to the user as a final message — "
    "there is no way to continue working after sending text. "
    "Use tools silently until the work is done, then respond once with your complete answer. "
    "Do NOT send progress updates or interim messages mid-task. "
    "\n\n"
    "To attach an image or video in your reply, include "
    "`![short description](/absolute/path/to/file)` in your final message. "
    "The path must be a local file you have already saved (for example, "
    "downloaded via http_request or generated via the shell). "
    "Supported formats: images (PNG, JPEG, GIF, WebP) and videos "
    "(MP4, MOV, WebM, 3GP). MP4 with H.264/AAC is most reliable for video. "
    "Size limit: 16 MB per file."
)

_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".webm", ".3gp", ".m4v"})

# Minimum seconds between successive message edits to avoid rate-limits.
_EDIT_THROTTLE_SECS = 2.0

# Base dir for CLI-shared state. Kept in sync with the deepagents-cli layout
# so users can drop skills/memory/MCP config under ~/.deepagents on the host
# and share them with this example (mounted at /root/.deepagents in Docker).
_DEEPAGENTS_HOME = Path("~/.deepagents").expanduser()


def _describe_action(tool_name: str, tool_input: object) -> str:
    """Return a short, human-readable label for a tool invocation."""
    name = tool_name.replace("_", " ")

    detail = ""
    if isinstance(tool_input, dict):
        for key in ("command", "cmd", "query", "url", "path", "file_path", "filename"):
            if key in tool_input:
                val = str(tool_input[key]).strip()
                if len(val) > 80:
                    val = val[:77] + "..."
                detail = val
                break
    elif isinstance(tool_input, str):
        detail = tool_input[:80]

    return f"{name}: {detail}" if detail else name


def _build_status_text(actions: list[dict], *, done: bool = False) -> str:
    """Build the body of the in-progress / completed status message."""
    header = "Done" if done else "Working..."

    if not actions:
        return header

    lines = [header, ""]
    for action in actions:
        mark = ">" if action["status"] == "done" else "..."
        lines.append(f"{mark} {action['description']}")

    return "\n".join(lines)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_recursion_limit() -> int:
    """Parse AGENT_RECURSION_LIMIT, falling back to 150 on missing/invalid values."""
    raw = os.getenv("AGENT_RECURSION_LIMIT", "").strip()
    if not raw:
        return 150
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "AGENT_RECURSION_LIMIT=%r is not an int; using default 150", raw,
        )
        return 150
    if value <= 0:
        logger.warning(
            "AGENT_RECURSION_LIMIT=%d must be positive; using default 150", value,
        )
        return 150
    return value


def _assistant_id() -> str:
    """Return the agent folder name under ~/.deepagents/ (default: whatsapp)."""
    return os.getenv("AGENT_ASSISTANT_ID", "whatsapp").strip() or "whatsapp"


def _agent_dir() -> Path:
    """Return ~/.deepagents/<assistant_id>/ — mirrors the CLI's layout."""
    return _DEEPAGENTS_HOME / _assistant_id()


def _parse_skill_sources() -> list[str]:
    """Return skill source dirs, always including ~/.deepagents/<agent>/skills/.

    Additional dirs from ``SKILLS_DIRS`` (``;``- or ``:``-separated) are
    appended after the default so explicit user entries take precedence.
    """
    default = _agent_dir() / "skills"
    try:
        default.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Could not create skills dir %s", default, exc_info=True)

    sources: list[str] = [str(default)] if default.is_dir() else []

    raw = os.getenv("SKILLS_DIRS", "").strip()
    if raw:
        sep = ";" if ";" in raw else os.pathsep
        for part in raw.split(sep):
            if not part.strip():
                continue
            p = str(Path(part).expanduser().resolve())
            if p not in sources:
                sources.append(p)
    return sources


def _parse_memory_sources() -> list[str]:
    """Return absolute paths to AGENTS.md files for MemoryMiddleware.

    When ``AGENT_MEMORY_PATHS`` is unset, defaults to
    ``~/.deepagents/<assistant_id>/AGENTS.md`` — the same layout as the CLI.
    Missing files are created (empty) so the agent has a writable destination
    and the middleware's system-prompt guidance is active on first run.
    """
    raw = os.getenv("AGENT_MEMORY_PATHS", "").strip()
    if raw:
        sep = ";" if ";" in raw else os.pathsep
        paths = [
            Path(part).expanduser().resolve()
            for part in raw.split(sep)
            if part.strip()
        ]
    else:
        paths = [_agent_dir() / "AGENTS.md"]

    resolved: list[str] = []
    for p in paths:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.touch()
            resolved.append(str(p))
        except OSError:
            logger.warning("Could not prepare memory file %s; skipping", p, exc_info=True)
    return resolved


def _resolve_mcp_config_path() -> str:
    """Return MCP config path, defaulting to ~/.deepagents/.mcp.json if present.

    Mirrors the CLI's `discover_mcp_configs()` user-level fallback: users who
    already have a global MCP config from the CLI get it for free here too.
    Explicit ``MCP_CONFIG`` env values always win.
    """
    explicit = os.getenv("MCP_CONFIG", "").strip()
    if explicit:
        return explicit
    default = _DEEPAGENTS_HOME / ".mcp.json"
    return str(default) if default.is_file() else ""


async def main() -> None:
    # --- Model setup ---
    model_name = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
    print(f"[main] Initializing model: {model_name}")
    model = init_chat_model(model_name)

    recursion_limit = _parse_recursion_limit()
    print(f"[main] Agent recursion_limit: {recursion_limit}")

    print(f"[main] Assistant ID: {_assistant_id()} (dir: {_agent_dir()})")

    # --- Cron jobs path ---
    # Default lives under the CLI-aligned agent dir so `jobs.json` is
    # persistent (via the ~/.deepagents bind mount in docker-compose) and
    # user-editable on the host.
    cron_default = _agent_dir() / "cron" / "jobs.json"
    jobs_path = Path(
        os.getenv("WHATSAPP_CRON_PATH", str(cron_default)),
    ).expanduser().resolve()

    # --- MCP tools (optional; follows the CLI's loader pattern) ---
    mcp_session_manager: MCPSessionManager | None = None
    mcp_extra_tools: list = []
    mcp_config_path = _resolve_mcp_config_path()
    if mcp_config_path:
        print(f"[main] Loading MCP tools from {mcp_config_path}")
        mcp_extra_tools, mcp_session_manager = await get_mcp_tools(mcp_config_path)
        print(f"[main] Loaded {len(mcp_extra_tools)} MCP tool(s)")

    # --- Skills sources (optional) ---
    skill_sources = _parse_skill_sources()
    if skill_sources:
        print(f"[main] Skills sources: {skill_sources}")

    # --- Memory sources (optional; AGENTS.md files loaded into system prompt) ---
    memory_sources = _parse_memory_sources()
    if memory_sources:
        print(f"[main] Memory sources: {memory_sources}")

    # --- Agent setup ---
    # Patch model.profile with context size so summarization middleware can
    # compute fraction-based trigger thresholds. AGENT_CONTEXT_SIZE overrides
    # whatever LangChain's built-in registry reports (needed for models like
    # local/OpenRouter endpoints whose profile is empty or wrong).
    raw_ctx = os.getenv("AGENT_CONTEXT_SIZE", "").strip()
    if raw_ctx:
        try:
            ctx_size = int(raw_ctx)
            model.profile = {**model.profile, "max_input_tokens": ctx_size}
            print(f"[main] Context size override: {ctx_size} tokens")
        except ValueError:
            logger.warning("AGENT_CONTEXT_SIZE=%r is not an int; ignoring", raw_ctx)
    elif not model.profile.get("max_input_tokens"):
        logger.warning(
            "Model profile has no max_input_tokens; summarization will use fixed "
            "token fallbacks (170k trigger). Set AGENT_CONTEXT_SIZE to override."
        )

    backend = LocalShellBackend(virtual_mode=False)
    agent = create_deep_agent(
        model=model,
        backend=backend,
        tools=[
            http_request,
            web_search,
            fetch_url,
            *build_cron_tools(jobs_path),
            *mcp_extra_tools,
        ],
        skills=skill_sources or None,
        memory=memory_sources,
        system_prompt=_MEDIA_ATTACH_INSTRUCTIONS,
        middleware=[create_summarization_tool_middleware(model, backend)],
    )

    # --- Per-chat conversation history (in-memory) ---
    conversations: dict[str, list] = {}
    chat_locks: dict[str, asyncio.Lock] = {}
    # Active agent tasks, keyed by chat_id. A chat may briefly have more than
    # one task in flight (an in-progress run and a queued follow-up waiting on
    # the chat lock), so we store a set and cancel all of them on /stop.
    active_agent_tasks: dict[str, set[asyncio.Task]] = {}

    # --- Adapter setup ---
    config = build_adapter_config()
    adapter = WhatsAppAdapter(config)

    async def handle_message(event: MessageEvent) -> None:
        """Callback invoked for each inbound WhatsApp message.

        /stop is handled inline (without the chat lock) so it can interrupt
        a running agent. Every other message is dispatched to
        ``_process_message`` as a cancellable task.
        """
        chat_id = event.chat_id
        text = event.text.strip()

        # /stop must NOT acquire the chat lock — a stuck agent is holding it.
        if text.lower() == "/stop":
            tasks = active_agent_tasks.get(chat_id)
            if tasks:
                for task in list(tasks):
                    task.cancel()
                await adapter.send(chat_id, "Stopping current task…")
            else:
                await adapter.send(chat_id, "No active task to stop.")
            return

        task = asyncio.create_task(_process_message(event))
        active_agent_tasks.setdefault(chat_id, set()).add(task)

        def _cleanup(t: asyncio.Task) -> None:
            active_agent_tasks.get(chat_id, set()).discard(t)
            # Surface unexpected exceptions (CancelledError is expected on /stop).
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "Unhandled exception in _process_message for %s: %s",
                    chat_id, exc,
                )

        task.add_done_callback(_cleanup)

    _speech_enabled: bool = os.getenv("SPEECH_ENABLED", "").lower() in ("true", "1", "yes")

    async def _process_message(event: MessageEvent) -> None:
        """Run the agent for one inbound message, serialized per chat."""
        chat_id = event.chat_id

        # Serialize message processing per chat to avoid race conditions
        if chat_id not in chat_locks:
            chat_locks[chat_id] = asyncio.Lock()

        async with chat_locks[chat_id]:
            # Handle slash commands
            text = event.text.strip()
            if text.lower() in ("/new", "/clear", "/reset"):
                conversations.pop(chat_id, None)
                await adapter.send(chat_id, "Conversation cleared.")
                return

            # --- Voice message handling ---
            if event.message_type == MessageType.VOICE:
                if not _speech_enabled:
                    await adapter.send(
                        chat_id,
                        "_(Voice messages are not enabled. "
                        "Set SPEECH_ENABLED=true to transcribe them.)_",
                        reply_to=event.message_id,
                    )
                    return
                if not event.media_urls:
                    await adapter.send(
                        chat_id,
                        "_(Received a voice message but no audio file was attached.)_",
                        reply_to=event.message_id,
                    )
                    return
                await adapter.send_typing(chat_id)
                import speech as _speech  # noqa: PLC0415  # optional heavy dep
                logger.info("[speech] Transcribing voice message from chat %s", chat_id)
                transcription = await asyncio.to_thread(
                    _speech.transcribe, event.media_urls[0],
                )
                if transcription:
                    event.text = f"[Voice message]: {transcription}"
                    logger.info(
                        "[speech] Transcription for %s: %s",
                        chat_id,
                        transcription[:120],
                    )
                else:
                    await adapter.send(
                        chat_id,
                        "_(Could not transcribe voice message.)_",
                        reply_to=event.message_id,
                    )
                    return
            # --- End voice message handling ---

            # Send typing indicator
            await adapter.send_typing(chat_id)

            # Get or create conversation history
            if chat_id not in conversations:
                conversations[chat_id] = []
            history = conversations[chat_id]

            # Append user message (multimodal content if inbound photo)
            history.append(HumanMessage(content=_build_inbound_content(event)))

            origin_ctx.set({
                "chat_id": chat_id,
                "message_id": event.message_id,
            })

            status_msg_id: str | None = None
            try:
                # Send an initial status message (will be edited with progress)
                status_result = await adapter.send(
                    chat_id,
                    _build_status_text([], done=False),
                    reply_to=event.message_id,
                )
                status_msg_id = (
                    status_result.message_id if status_result.success else None
                )

                # Stream agent execution so we can surface tool actions.
                # Retry up to 3 times on model parse errors (e.g. malformed
                # tool call responses from OpenAI-compat endpoints).
                _MAX_RETRIES = 3
                actions: list[dict] = []
                last_edit_time = 0.0
                root_run_id: str | None = None
                final_output: dict | None = None

                for _attempt in range(_MAX_RETRIES):
                  try:
                    actions = []
                    root_run_id = None
                    final_output = None

                    async for ev in agent.astream_events(
                        {"messages": history},
                        version="v2",
                        config={"recursion_limit": recursion_limit},
                    ):
                        kind = ev["event"]
                        run_id = ev.get("run_id")

                        # Capture the root run so we can grab its output later
                        if root_run_id is None and kind == "on_chain_start":
                            root_run_id = run_id

                        elif kind == "on_tool_start":
                            tool_input = ev.get("data", {}).get("input", {})
                            desc = _describe_action(
                                ev.get("name", "tool"), tool_input,
                            )
                            actions.append({
                                "run_id": run_id,
                                "description": desc,
                                "status": "running",
                            })

                            # Throttled status edit
                            now = time.monotonic()
                            if status_msg_id and now - last_edit_time >= _EDIT_THROTTLE_SECS:
                                await adapter.edit_message(
                                    chat_id, status_msg_id,
                                    _build_status_text(actions),
                                )
                                last_edit_time = now

                        elif kind == "on_tool_end":
                            # Mark the matching action as done (by run_id)
                            tool_run_id = ev.get("run_id")
                            for a in actions:
                                if a.get("run_id") == tool_run_id and a["status"] == "running":
                                    a["status"] = "done"
                                    break

                            now = time.monotonic()
                            if status_msg_id and now - last_edit_time >= _EDIT_THROTTLE_SECS:
                                await adapter.edit_message(
                                    chat_id, status_msg_id,
                                    _build_status_text(actions),
                                )
                                last_edit_time = now

                        elif kind == "on_chain_end" and run_id == root_run_id:
                            final_output = ev.get("data", {}).get("output", {})

                    break  # success — exit retry loop

                  except asyncio.CancelledError:
                    raise
                  except Exception as _retry_exc:
                    _err = str(_retry_exc)
                    _status = getattr(_retry_exc, "status_code", None)
                    if _status in (413, 400) and ("exceed_context_size_error" in _err or "exceeds the available context size" in _err) or _status == 413:
                        if _attempt + 1 < _MAX_RETRIES and len(history) > 2:
                            logger.warning(
                                "[whatsapp] Context overflow (attempt %d/%d), trimming history (%d → %d messages)",
                                _attempt + 1, _MAX_RETRIES, len(history), max(2, len(history) // 2),
                            )
                            keep = max(2, len(history) // 2)
                            del history[1:-keep]  # keep first (oldest) human msg + last `keep` messages
                            continue
                    if "Failed to parse" in _err or "tool_call" in _err.lower():
                        logger.warning(
                            "[whatsapp] Model parse error (attempt %d/%d), retrying: %s",
                            _attempt + 1, _MAX_RETRIES, _retry_exc,
                        )
                        if _attempt + 1 < _MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                    raise

                # Mark every remaining action as done
                for a in actions:
                    a["status"] = "done"

                # ---- Extract the agent's final response text ----
                response_text = ""
                if final_output:
                    response_messages = final_output.get("messages", [])
                    for msg in reversed(response_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            if isinstance(msg.content, str):
                                response_text = msg.content
                            elif isinstance(msg.content, list):
                                for block in reversed(msg.content):
                                    if isinstance(block, str):
                                        response_text = block
                                        break
                                    elif (
                                        isinstance(block, dict)
                                        and block.get("type") == "text"
                                    ):
                                        response_text = block["text"]
                                        break
                            if response_text:
                                break

                # ---- Deliver the response ----
                if response_text:
                    # Extract any markdown media refs; validate each path
                    cleaned_text, media_refs = extract_markdown_media(
                        response_text,
                    )
                    # Each entry: (alt, path, media_type) where media_type is
                    # "image" or "video" — classified by file extension.
                    valid_media: list[tuple[str, str, str]] = []
                    missing_alts: list[str] = []
                    for alt, path in media_refs:
                        try:
                            p = Path(path)
                            ok = (
                                p.is_file()
                                and p.stat().st_size <= 16 * 1024 * 1024
                            )
                        except OSError:
                            ok = False
                        if ok:
                            mtype = (
                                "video"
                                if p.suffix.lower() in _VIDEO_EXTENSIONS
                                else "image"
                            )
                            valid_media.append((alt, str(p), mtype))
                        else:
                            missing_alts.append(alt or "attachment")

                    text_to_send = cleaned_text
                    if missing_alts:
                        seen: set[str] = set()
                        for name in missing_alts:
                            if name in seen:
                                continue
                            seen.add(name)
                            text_to_send = (
                                f"{text_to_send.rstrip()}\n"
                                f"_(couldn't attach: {name})_"
                            )

                    if valid_media:
                        # Combine: first attachment carries the response text
                        # as its caption (single chat bubble). Extra files
                        # follow up with their alt as caption. Status msg is
                        # finalized to a brief "Done" since we can't edit a
                        # text bubble into a media attachment.
                        if status_msg_id:
                            await adapter.edit_message(
                                chat_id, status_msg_id,
                                _build_status_text(actions, done=True),
                            )

                        first_alt, first_path, first_type = valid_media[0]
                        first_caption = text_to_send or first_alt or None
                        first_result = await adapter.send_media(
                            chat_id, first_path, first_type,
                            caption=first_caption,
                        )
                        if not first_result.success:
                            logger.warning(
                                "send_media failed for %s: %s",
                                first_path, first_result.error,
                            )

                        for alt, m_path, m_type in valid_media[1:]:
                            media_result = await adapter.send_media(
                                chat_id, m_path, m_type,
                                caption=alt or None,
                            )
                            if not media_result.success:
                                logger.warning(
                                    "send_media failed for %s: %s",
                                    m_path, media_result.error,
                                )
                    elif actions:
                        # Tools were used: update status to "Done", then send
                        # the response as a separate message.
                        if status_msg_id:
                            await adapter.edit_message(
                                chat_id, status_msg_id,
                                _build_status_text(actions, done=True),
                            )
                        send_result = await adapter.send(chat_id, text_to_send)
                        if not send_result.success:
                            logger.error("Failed to send: %s", send_result.error)
                    else:
                        # No tools, no images — edit the status message into
                        # the response so only one message is visible.
                        sent = False
                        if status_msg_id:
                            edit_result = await adapter.edit_message(
                                chat_id, status_msg_id, text_to_send,
                            )
                            sent = edit_result.success
                        if not sent:
                            await adapter.send(
                                chat_id, text_to_send,
                                reply_to=event.message_id,
                            )

                    # History stores the original response_text (refs intact)
                    # so the agent sees its own markdown on the next turn.
                    history.append(AIMessage(content=response_text))
                else:
                    logger.warning(
                        "Agent returned no text response for chat %s", chat_id,
                    )
                    if status_msg_id:
                        await adapter.edit_message(
                            chat_id, status_msg_id,
                            _build_status_text(actions, done=True),
                        )

                # Sync history with full agent output
                if final_output:
                    conversations[chat_id] = final_output.get("messages", history)

            except asyncio.CancelledError:
                logger.info("Task for chat %s was cancelled by /stop", chat_id)
                if status_msg_id:
                    try:
                        await adapter.edit_message(
                            chat_id, status_msg_id, "Task stopped.",
                        )
                    except Exception:
                        pass
                raise
            except Exception:
                logger.exception("Error processing message from %s", chat_id)
                if status_msg_id:
                    try:
                        await adapter.edit_message(
                            chat_id, status_msg_id,
                            "Something went wrong processing your message.",
                        )
                    except Exception:
                        pass
                else:
                    await adapter.send(
                        chat_id,
                        "Sorry, something went wrong processing your message.",
                    )

    adapter.on_message(handle_message)

    # --- Connect ---
    print("[main] Connecting to WhatsApp...")
    connected = await adapter.connect()
    if not connected:
        print("[main] Failed to connect. Exiting.")
        sys.exit(1)

    tick_interval = float(os.getenv("WHATSAPP_CRON_TICK_SECONDS", "60"))
    ticker_task = start_ticker(
        jobs_path, adapter, agent, chat_locks,
        tick_interval=tick_interval,
        recursion_limit=recursion_limit,
    )

    print("[main] WhatsApp channel running. Press Ctrl+C to stop.")

    # --- Graceful shutdown ---
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        print("\n[main] Shutting down...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()
    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass
    # Cancel any in-flight per-message tasks so disconnect can proceed cleanly.
    pending_tasks = [t for ts in active_agent_tasks.values() for t in ts if not t.done()]
    for t in pending_tasks:
        t.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    await adapter.disconnect()
    if mcp_session_manager is not None:
        try:
            await mcp_session_manager.cleanup()
        except Exception:
            logger.warning("MCP session cleanup failed", exc_info=True)
    print("[main] Done.")


if __name__ == "__main__":
    asyncio.run(main())
