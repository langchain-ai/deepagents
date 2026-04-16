"""Entry point: connects a Deep Agent to WhatsApp via the bridge adapter."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from deepagents import create_deep_agent
from whatsapp_adapter import MessageEvent, WhatsAppAdapter

load_dotenv()

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

    # --- Agent setup ---
    agent = create_deep_agent(model=model)

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

            # Append user message
            history.append(HumanMessage(content=text))

            try:
                # Invoke the agent
                result = await agent.ainvoke({"messages": history})

                # Extract the agent's final response text
                response_messages = result.get("messages", [])
                response_text = ""
                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        if isinstance(msg.content, str):
                            response_text = msg.content
                        elif isinstance(msg.content, list):
                            # Content blocks: find the last text block
                            for block in reversed(msg.content):
                                if isinstance(block, str):
                                    response_text = block
                                    break
                                elif isinstance(block, dict) and block.get("type") == "text":
                                    response_text = block["text"]
                                    break
                        if response_text:
                            break

                if response_text:
                    # Send response
                    send_result = await adapter.send(
                        chat_id, response_text, reply_to=event.message_id,
                    )
                    if not send_result.success:
                        logger.error("Failed to send: %s", send_result.error)

                    # Append to history
                    history.append(AIMessage(content=response_text))
                else:
                    logger.warning("Agent returned no text response for chat %s", chat_id)

                # Sync history with full agent output
                conversations[chat_id] = result.get("messages", history)

            except Exception:
                logger.exception("Error processing message from %s", chat_id)
                await adapter.send(chat_id, "Sorry, something went wrong processing your message.")

    adapter.on_message(handle_message)

    # --- Connect ---
    print("[main] Connecting to WhatsApp...")
    connected = await adapter.connect()
    if not connected:
        print("[main] Failed to connect. Exiting.")
        sys.exit(1)

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
    await adapter.disconnect()
    print("[main] Done.")


if __name__ == "__main__":
    asyncio.run(main())
