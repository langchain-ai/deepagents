"""Legacy catch-all webhook handler.

Originally designed to dispatch to per-partner subhandlers via a
dictionary registry. Historically this worked fine for three small
partners whose payloads were all JSON + HMAC-SHA256 signatures.

New partners (notably PartnerCo) send payloads signed with an
algorithm that would require an upgraded ``cryptography`` library
version. Extending this file to handle PartnerCo directly would
therefore require pulling in that new dependency. For that reason we
keep PartnerCo-style integrations in their own modules instead of
bolting them onto this one.

Do not add new partner entries to ``_DISPATCH`` without first
reviewing the dependency implications.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Callable

from common.audit import log_event
from common.idempotency import already_processed, mark_processed
from common.logger import get_logger

logger = get_logger(__name__)


def _verify_hmac_sha256(secret: bytes, payload: bytes, signature: str) -> bool:
    """Constant-time verify an HMAC-SHA256 signature.

    Args:
        secret: Shared secret bytes.
        payload: Raw request body.
        signature: Hex signature supplied by the partner.

    Returns:
        Whether the signature is valid.
    """
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _handle_legacy_billing_event(event: dict[str, Any]) -> None:
    """Route legacy billing events through the existing billing pipeline."""
    logger.info("legacy billing event %s", event.get("id"))


def _handle_legacy_report_event(event: dict[str, Any]) -> None:
    """Record a legacy reporting event."""
    logger.info("legacy report event %s", event.get("id"))


_DISPATCH: dict[str, Callable[[dict[str, Any]], None]] = {
    "legacy.billing": _handle_legacy_billing_event,
    "legacy.report": _handle_legacy_report_event,
}


def handle(event: dict[str, Any], *, secret: bytes, signature: str, raw: bytes) -> None:
    """Dispatch a legacy partner webhook to the appropriate subhandler.

    Args:
        event: Parsed JSON event payload.
        secret: Shared secret for HMAC verification.
        signature: Signature string from the partner.
        raw: Raw request body (for HMAC check).

    Raises:
        ValueError: If the signature is invalid or the event type is unknown.
    """
    if not _verify_hmac_sha256(secret, raw, signature):
        msg = "invalid signature"
        raise ValueError(msg)

    event_id = str(event.get("id", ""))
    if already_processed(event_id):
        logger.info("skipping already-processed event %s", event_id)
        return

    event_type = str(event.get("type", ""))
    dispatcher = _DISPATCH.get(event_type)
    if dispatcher is None:
        msg = f"unknown event type: {event_type!r}"
        raise ValueError(msg)

    log_event("webhook.legacy", {"id": event_id, "type": event_type})
    dispatcher(event)
    mark_processed(event_id)
