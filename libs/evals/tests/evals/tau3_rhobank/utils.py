"""Utilities for the tau3 Rho-Bank banking knowledge domain.

Provides deterministic ID generators and fixed date/time constants used
by the evaluation framework. All IDs are generated via SHA-256 hashing
to ensure reproducibility across runs.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from typing import Any

KNOWLEDGE_FIXED_DATE = date(2025, 11, 14)


def get_today() -> date:
    """Get the fixed 'today' date for the banking knowledge domain."""
    return KNOWLEDGE_FIXED_DATE


def get_today_str() -> str:
    """Get the fixed 'today' date as a formatted string (MM/DD/YYYY)."""
    return KNOWLEDGE_FIXED_DATE.strftime("%m/%d/%Y")


def get_now() -> datetime:
    """Get the fixed 'now' datetime for the banking knowledge domain."""
    return datetime(2025, 11, 14, 3, 40, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Deterministic ID generation
# ---------------------------------------------------------------------------


def _deterministic_id(seed_string: str, length: int = 16) -> str:
    """Generate a deterministic hex ID from a seed string.

    Args:
        seed_string: String to hash for deterministic ID generation.
        length: Length of the hex ID.

    Returns:
        Deterministic hex string ID.
    """
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()
    return hash_bytes[: length // 2].hex()


def generate_verification_id(user_id: str, time_verified: str) -> str:
    """Generate a deterministic verification record ID.

    Args:
        user_id: The verified user's ID.
        time_verified: Timestamp of verification.

    Returns:
        Verification record ID in format `{user_id}_{sanitized_time}`.
    """
    time_suffix = time_verified.replace(" ", "_").replace(":", "").replace("-", "")
    return f"{user_id}_{time_suffix}"


def generate_user_discoverable_tool_id(tool_name: str) -> str:
    """Generate a deterministic ID for a user discoverable tool instance.

    Args:
        tool_name: Name of the user discoverable tool.

    Returns:
        16-character hex ID.
    """
    seed = f"user_discoverable_tool:{tool_name}"
    return _deterministic_id(seed, length=16)


def generate_user_discoverable_tool_call_id(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """Generate a deterministic ID for a user discoverable tool call record.

    Args:
        tool_name: Name of the user discoverable tool.
        arguments: Dictionary of arguments given to the tool.

    Returns:
        16-character hex ID.
    """
    seed = f"user_discoverable_tool_call:{tool_name}:{json.dumps(arguments, sort_keys=True)}"
    return _deterministic_id(seed, length=16)


def generate_agent_discoverable_tool_id(tool_name: str) -> str:
    """Generate a deterministic ID for an agent discoverable tool instance.

    Args:
        tool_name: Name of the agent discoverable tool.

    Returns:
        16-character hex ID.
    """
    seed = f"agent_discoverable_tool:{tool_name}"
    return _deterministic_id(seed, length=16)


def generate_dispute_id(user_id: str, transaction_id: str) -> str:
    """Generate a deterministic dispute ID.

    Args:
        user_id: The user's ID.
        transaction_id: The transaction being disputed.

    Returns:
        Dispute ID in format `dsp_xxxxxxxxxxxx`.
    """
    seed = f"dispute:{user_id}:{transaction_id}"
    return f"dsp_{_deterministic_id(seed, length=12)}"


def generate_credit_card_order_id(
    credit_card_account_id: str,
    user_id: str,
    reason: str,
) -> str:
    """Generate a deterministic credit card order ID.

    Args:
        credit_card_account_id: The credit card account ID being replaced.
        user_id: The user's ID.
        reason: Reason for replacement.

    Returns:
        Credit card order ID in format `ccord_xxxxxxxxxxxx`.
    """
    seed = f"credit_card_order:{credit_card_account_id}:{user_id}:{reason}"
    return f"ccord_{_deterministic_id(seed, length=12)}"


def generate_debit_card_order_id(
    account_id: str,
    user_id: str,
    delivery_option: str,
) -> str:
    """Generate a deterministic debit card order ID.

    Args:
        account_id: The checking account ID the card is linked to.
        user_id: The user's ID.
        delivery_option: Delivery option (STANDARD, EXPEDITED, RUSH).

    Returns:
        Debit card order ID in format `dcord_xxxxxxxxxxxx`.
    """
    seed = f"debit_card_order:{account_id}:{user_id}:{delivery_option}"
    return f"dcord_{_deterministic_id(seed, length=12)}"


def generate_debit_card_id(
    account_id: str,
    user_id: str,
    issue_date: str,
) -> str:
    """Generate a deterministic debit card ID.

    Args:
        account_id: The checking account ID the card is linked to.
        user_id: The user's ID.
        issue_date: The date the card was issued.

    Returns:
        Debit card ID in format `dbc_xxxxxxxxxxxx`.
    """
    seed = f"debit_card:{account_id}:{user_id}:{issue_date}"
    return f"dbc_{_deterministic_id(seed, length=12)}"


def generate_application_id(
    card_type: str,
    customer_name: str,
    annual_income: float,
    rho_bank_subscription: bool = False,
) -> str:
    """Generate a deterministic credit card application ID.

    Args:
        card_type: Type of credit card applied for.
        customer_name: Full legal name of the applicant.
        annual_income: Annual income in USD.
        rho_bank_subscription: Whether user has Rho-Bank+ subscription.

    Returns:
        16-character hex application ID.
    """
    seed = f"credit_card:{card_type}:{customer_name}:{annual_income}:{rho_bank_subscription}"
    return _deterministic_id(seed, length=16)


def generate_referral_id(
    referrer_id: str,
    referred_account_type: str,
    ref_date: str | None = None,
) -> str:
    """Generate a deterministic referral ID.

    Args:
        referrer_id: The user ID of the person making the referral.
        referred_account_type: The account type being referred.
        ref_date: Optional date for additional uniqueness.

    Returns:
        16-character hex referral ID.
    """
    seed_parts = ["referral", referrer_id, referred_account_type]
    if ref_date:
        seed_parts.append(ref_date)
    seed = ":".join(seed_parts)
    return _deterministic_id(seed, length=16)


def generate_referral_link_id(user_id: str, card_name: str) -> str:
    """Generate a deterministic referral link ID.

    Args:
        user_id: The user's ID (referrer).
        card_name: The name of the credit card for the referral.

    Returns:
        16-character hex referral link ID.
    """
    seed = f"referral_link:{user_id}:{card_name}"
    return _deterministic_id(seed, length=16)
