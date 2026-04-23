"""Billing-internal rate limiter.

Owned by the payments team. Enforces spend caps per customer per
billing cycle; must not be repurposed as a general-purpose rate
limiter — see ``ratelimit/`` at the repo root for that.
"""

from __future__ import annotations

import time
from collections import defaultdict

from common.logger import get_logger

logger = get_logger(__name__)


_WINDOW_SECONDS = 3600
_DEFAULT_CAP_CENTS = 1_000_000

_spend_window: dict[str, list[tuple[float, int]]] = defaultdict(list)


def record_charge(customer_id: str, amount_cents: int) -> bool:
    """Record a charge and return whether it fits within the spend cap.

    Args:
        customer_id: Customer identifier.
        amount_cents: Amount being charged, in cents.

    Returns:
        ``True`` if the charge fits within the hourly cap, ``False``
        otherwise.
    """
    now = time.time()
    window = _spend_window[customer_id]
    window[:] = [(ts, amt) for ts, amt in window if now - ts < _WINDOW_SECONDS]
    total = sum(amt for _, amt in window)
    if total + amount_cents > _DEFAULT_CAP_CENTS:
        logger.warning("spend cap exceeded for %s: %d", customer_id, total + amount_cents)
        return False
    window.append((now, amount_cents))
    return True
