"""Structured audit-log writer.

Every handler for an external event must emit a single audit record
via ``log_event(kind, fields)``. The audit sink is intentionally
append-only and downstream-consumed; callers should not read it back.
"""

from __future__ import annotations

import json
from typing import Any

from common.logger import get_logger

logger = get_logger("audit")

_audit_records: list[dict[str, Any]] = []


def log_event(kind: str, fields: dict[str, Any]) -> None:
    """Record a single audit event.

    Args:
        kind: Short dotted category, e.g. ``"webhook.partnerco"``.
        fields: JSON-serializable metadata. Must include an ``id`` key
            when the event corresponds to a partner-supplied identifier.
    """
    record = {"kind": kind, **fields}
    _audit_records.append(record)
    logger.info("audit %s", json.dumps(record, sort_keys=True))


def drain() -> list[dict[str, Any]]:
    """Return and clear the in-memory audit buffer (used by tests).

    Returns:
        The audit records recorded since the last drain.
    """
    out = list(_audit_records)
    _audit_records.clear()
    return out
