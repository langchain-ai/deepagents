"""Goal ledger error types.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations


class GoalLedgerError(ValueError):
    """Raised when a goal ledger request is invalid, stale, or unauthorized."""


class GoalLedgerCorruptError(RuntimeError):
    """Raised when the ledger file cannot be read; it is never overwritten."""
