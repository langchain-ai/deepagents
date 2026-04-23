"""Event-id based idempotency helpers.

External partner events must be deduplicated by their stable
``event.id`` field. Handlers should wrap their effectful work in an
``already_processed`` / ``mark_processed`` pair:

    if already_processed(event_id):
        return
    ... do the work ...
    mark_processed(event_id)

The in-memory implementation here is fine for the service's test
fixture; production wires up a Redis-backed store via the same
interface.
"""

from __future__ import annotations


_processed: set[str] = set()


def already_processed(event_id: str) -> bool:
    """Return whether ``event_id`` has already been marked processed.

    Args:
        event_id: Stable per-event identifier.

    Returns:
        ``True`` iff ``mark_processed`` was previously called with
        this id.
    """
    return event_id in _processed


def mark_processed(event_id: str) -> None:
    """Record that ``event_id`` has been handled.

    Args:
        event_id: Stable per-event identifier.
    """
    _processed.add(event_id)


def reset() -> None:
    """Clear the in-memory dedupe set (used by tests)."""
    _processed.clear()
