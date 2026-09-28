"""Cron error types.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations


class CronJobError(ValueError):
    """Raised when a cron job request is invalid."""
