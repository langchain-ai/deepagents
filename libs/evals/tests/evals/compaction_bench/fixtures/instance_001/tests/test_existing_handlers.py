"""Sanity tests for the fixture's baseline code.

These pass against the unmodified fixture. Feature work is expected to
add additional tests but should not break these; a failing existing
test counts against the overall-correctness grader.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from common import idempotency
from webhooks import generic_handler


@pytest.fixture(autouse=True)
def _reset_idempotency():
    """Clear the in-memory idempotency store between tests."""
    idempotency.reset()
    yield
    idempotency.reset()


def _sign(secret: bytes, payload: bytes) -> str:
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def test_legacy_billing_event_dispatches():
    """The generic handler accepts a legacy billing event with a valid signature."""
    secret = b"shared-secret"
    event = {"id": "evt_1", "type": "legacy.billing", "amount_cents": 2500}
    raw = json.dumps(event).encode("utf-8")
    signature = _sign(secret, raw)

    generic_handler.handle(event, secret=secret, signature=signature, raw=raw)


def test_legacy_event_is_deduped():
    """Re-delivering the same event id is a no-op (idempotency)."""
    secret = b"shared-secret"
    event = {"id": "evt_2", "type": "legacy.report"}
    raw = json.dumps(event).encode("utf-8")
    signature = _sign(secret, raw)

    generic_handler.handle(event, secret=secret, signature=signature, raw=raw)
    generic_handler.handle(event, secret=secret, signature=signature, raw=raw)


def test_invalid_signature_rejected():
    """A bad signature raises ``ValueError``."""
    event = {"id": "evt_3", "type": "legacy.billing"}
    raw = json.dumps(event).encode("utf-8")

    with pytest.raises(ValueError, match="invalid signature"):
        generic_handler.handle(event, secret=b"real", signature="deadbeef", raw=raw)


def test_unknown_event_type_rejected():
    """Unknown ``type`` values raise ``ValueError``."""
    secret = b"shared-secret"
    event = {"id": "evt_4", "type": "legacy.nope"}
    raw = json.dumps(event).encode("utf-8")
    signature = _sign(secret, raw)

    with pytest.raises(ValueError, match="unknown event type"):
        generic_handler.handle(event, secret=secret, signature=signature, raw=raw)
