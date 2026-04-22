"""Env-var -> adapter config mapping for the WhatsApp channel example.

Kept in its own module (with no heavy imports) so the defaults can be unit
tested without pulling in the agent's dependency tree.
"""

from __future__ import annotations

import os
from typing import Any


def build_adapter_config() -> dict[str, Any]:
    """Build adapter config from environment variables.

    ``WHATSAPP_SELF_ONLY`` defaults to ``"true"`` so the example is safe
    out of the box — the agent only replies to messages you send to
    yourself until you explicitly opt in to replying to others.
    """
    return {
        "bridge_port": os.getenv("WHATSAPP_BRIDGE_PORT", "3000"),
        "session_path": os.getenv("WHATSAPP_SESSION_PATH", "./session"),
        "require_mention": os.getenv("WHATSAPP_REQUIRE_MENTION", "false"),
        "mention_patterns": os.getenv("WHATSAPP_MENTION_PATTERNS"),
        "free_response_chats": os.getenv("WHATSAPP_FREE_RESPONSE_CHATS", ""),
        "self_only": os.getenv("WHATSAPP_SELF_ONLY", "true"),
    }
