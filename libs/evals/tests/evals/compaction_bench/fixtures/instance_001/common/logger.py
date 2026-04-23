"""Configured logger factory.

All feature code should obtain loggers via ``get_logger(__name__)``
rather than calling ``logging.getLogger`` directly. This keeps the
global formatter/handlers consistent with the rest of the service.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def _configure_root() -> None:
    """Attach a single stderr handler to the root logger if not already present."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the service's standard formatting.

    Args:
        name: Dotted module name (usually ``__name__``).

    Returns:
        A ``logging.Logger`` instance safe for use at module scope.
    """
    _configure_root()
    return logging.getLogger(name)
