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
from tools import fetch_url, http_request, web_search
from cron import build_cron_tools, origin_ctx, start_ticker
from whatsapp_adapter import (
    MessageEvent,
    WhatsAppAdapter,
    _build_inbound_content,
    extract_markdown_images,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Status-message helpers
# ---------------------------------------------------------------------------

# Minimum seconds between successive message edits to avoid rate-limits.
_EDIT_THROTTLE_SECS = 2.0


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


def _build_config() -> dict:
    """Build adapter config from environment variables."""
    return {
        "bridge_port": os.getenv("WHATSAPP_BRIDGE_PORT", "3000"),
        "session_path": os.getenv("WHATSAPP_SESSION_PATH", "./session"),
        "require_mention": os.getenv("WHATSAPP_REQUIRE_MENTION", "false"),
        "mention_patterns": os.getenv("WHATSAPP_MENTION_PATTERNS"),
        "free_response_chats": os.getenv("WHATSAPP_FREE_RESPONSE_CHATS", ""),
        "reply_prefix": os.getenv("WHATSAPP_REPLY_PREFIX"),
        "self_only": os.getenv("WHATSAPP_SELF_ONLY", "true"),
    }


async def main() -> None:
    # --- Model setup ---
    model_name = os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
    print(f"[main] Initializing model: {model_name}")
    model = init_chat_model(model_name)

    # --- Cron jobs path ---
    jobs_path = Path(
        os.getenv("WHATSAPP_CRON_PATH", "./cron/jobs.json"),
    ).expanduser().resolve()

    # --- Agent setup ---
    agent = create_deep_agent(
        model=model,
        backend=LocalShellBackend(virtual_mode=False),
        tools=[http_request, web_search, fetch_url, *build_cron_tools(jobs_path)],
    )

    # --- Per-chat conversation history (in-memory) ---
    conversations: dict[str, list] = {}
    chat_locks: dict[str, asyncio.Lock] = {}

    # --- Adapter setup ---
    config = _build_config()
    adapter = WhatsAppAdapter(config)

    async def handle_message(event: MessageEvent) -> None:
        """Callback invoked for each inbound WhatsApp message."""
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

                # Stream agent execution so we can surface tool actions
                actions: list[dict] = []
                last_edit_time = 0.0
                root_run_id: str | None = None
                final_output: dict | None = None

                async for ev in agent.astream_events(
                    {"messages": history}, version="v2",
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
                    # Extract any markdown image refs; validate each path
                    cleaned_text, image_refs = extract_markdown_images(
                        response_text,
                    )
                    valid_images: list[tuple[str, str]] = []
                    missing_alts: list[str] = []
                    for alt, path in image_refs:
                        try:
                            p = Path(path)
                            ok = (
                                p.is_file()
                                and p.stat().st_size <= 16 * 1024 * 1024
                            )
                        except OSError:
                            ok = False
                        if ok:
                            valid_images.append((alt, str(p)))
                        else:
                            missing_alts.append(alt or "image")

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

                    if actions:
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
                        # No tools — edit the status message into the response
                        # so only one message is visible.
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

                    # Send each valid image as a follow-up media message.
                    # History stores the original response_text (refs intact)
                    # so the agent sees its own markdown on the next turn.
                    for alt, img_path in valid_images:
                        media_result = await adapter.send_media(
                            chat_id, img_path, "image",
                            caption=alt or None,
                        )
                        if not media_result.success:
                            logger.warning(
                                "send_media failed for %s: %s",
                                img_path, media_result.error,
                            )

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
        jobs_path, adapter, agent, chat_locks, tick_interval=tick_interval,
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
    await adapter.disconnect()
    print("[main] Done.")


if __name__ == "__main__":
    asyncio.run(main())
