"""Rho-Bank domain models, data loaders, and LangChain tool wrappers.

Reimplements the tau3 banking_knowledge domain as self-contained code
so evals run without a cross-repo dependency. Includes TransactionalDB,
KnowledgeBase, and all agent/user tools needed for tasks 017, 041, 080, 081.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tests.evals.tau3_rhobank.db_query import (
    add_to_db,
    query_database_tool,
)
from tests.evals.tau3_rhobank.utils import (
    _deterministic_id,
    generate_agent_discoverable_tool_id,
    generate_application_id,
    generate_credit_card_order_id,
    generate_debit_card_id,
    generate_debit_card_order_id,
    generate_dispute_id,
    generate_user_discoverable_tool_call_id,
    generate_user_discoverable_tool_id,
    generate_verification_id,
    get_now,
    get_today_str,
)

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Transfer reason literal (must match upstream exactly for validation)
# ---------------------------------------------------------------------------

TransferReasonLiteral = Literal[
    "fraud_or_security_concern",
    "account_closure_request",
    "deceased_account_holder",
    "legal_or_regulatory_matter",
    "account_ownership_dispute",
    "complex_billing_dispute",
    "abusive_customer_behavior",
    "third_party_inquiry",
    "technical_system_error",
    "unconfirmed_external_communication",
    "customer_demands_after_unavailable_offer_refusal",
    "kb_search_unsuccessful_customer_requests_transfer",
    "specialized_department_required",
    "accessibility_or_special_needs",
    "customer_frustrated_demands_human",
    "supervisor_request_service_complaint",
    "customer_requests_human_no_specific_reason",
    "request_completed_customer_wants_human_followup",
    "other",
]

# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class DatabaseTable(BaseModel):
    """A database table with data and optional notes."""

    data: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


class TransactionalDB(BaseModel):
    """Transactional database for the banking knowledge domain.

    Contains all mutable state and is hashed for DB state comparison
    during evaluation. Field order must match db.json for hash compatibility.
    """

    users: DatabaseTable = Field(default_factory=DatabaseTable)
    accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    debit_cards: DatabaseTable = Field(default_factory=DatabaseTable)
    referrals: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_applications: DatabaseTable = Field(default_factory=DatabaseTable)
    user_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    user_discoverable_tool_calls: DatabaseTable = Field(default_factory=DatabaseTable)
    verification_history: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_transaction_history: DatabaseTable = Field(default_factory=DatabaseTable)
    cash_back_disputes: DatabaseTable = Field(default_factory=DatabaseTable)
    bank_account_transaction_history: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_accounts: DatabaseTable = Field(default_factory=DatabaseTable)
    agent_discoverable_tools: DatabaseTable = Field(default_factory=DatabaseTable)
    task_config: DatabaseTable = Field(default_factory=DatabaseTable)
    human_transfer_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    transaction_disputes: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_orders: DatabaseTable = Field(default_factory=DatabaseTable)
    debit_card_orders: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_closure_reasons: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_card_account_flags: DatabaseTable = Field(default_factory=DatabaseTable)
    credit_limit_increase_requests: DatabaseTable = Field(default_factory=DatabaseTable)
    payment_history: DatabaseTable = Field(default_factory=DatabaseTable)
    debit_card_disputes: DatabaseTable = Field(default_factory=DatabaseTable)


class Document(BaseModel):
    """A document in the knowledge base."""

    id: str
    title: str
    content: str


class KnowledgeBase(BaseModel):
    """Knowledge base containing documents for semantic search."""

    documents: dict[str, Document] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_db() -> TransactionalDB:
    """Load a fresh TransactionalDB from the bundled db.json."""
    with (_DATA_DIR / "db.json").open() as fp:
        return TransactionalDB.model_validate_json(fp.read())


def load_task(task_id: str) -> dict[str, Any]:
    """Load a single task by ID.

    Args:
        task_id: The task ID string (e.g. "017").

    Returns:
        The raw task dict.

    Raises:
        FileNotFoundError: If the task file doesn't exist.
    """
    path = _DATA_DIR / "tasks" / f"task_{task_id}.json"
    with path.open() as fp:
        return json.load(fp)


def get_documents_dir() -> Path:
    """Return the absolute path to the vendored knowledge-base documents folder.

    The folder contains 698 JSON documents (one per policy/procedure) that the
    agent can search at runtime using its built-in tools.
    """
    return _DATA_DIR / "documents"


# ---------------------------------------------------------------------------
# Tool call logging
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEntry:
    """Record of a single tool invocation."""

    name: str
    args: dict[str, Any]
    result: str
    requestor: str = "assistant"
    error: bool = False


# ---------------------------------------------------------------------------
# Helper functions for discoverable tools
# ---------------------------------------------------------------------------


def _parse_balance(val: float | str) -> float:
    """Parse a balance value that may be a number or a string like '$2,850.00'."""
    if isinstance(val, int | float):
        return float(val)
    if isinstance(val, str):
        return float(val.replace("$", "").replace(",", ""))
    return 0.0


def parse_discoverable_tool_docstring(func: object) -> dict[str, Any]:
    """Parse a discoverable tool's docstring into description and parameters.

    Args:
        func: The function to parse.

    Returns:
        Dictionary with `name`, `description`, `parameters`, `success_message`.
    """
    docstring = inspect.getdoc(func) or ""
    parts = re.split(r"\n\s*Args:\s*\n", docstring, maxsplit=1)
    description = parts[0].strip()

    parameters: dict[str, dict[str, Any]] = {}
    success_message = "Action completed successfully."

    if len(parts) > 1:
        remaining = parts[1]
        returns_split = re.split(r"\n\s*Returns:\s*\n", remaining, maxsplit=1)
        args_section = returns_split[0]

        if len(returns_split) > 1:
            success_message = returns_split[1].strip().split("\n")[0].strip()

        arg_pattern = re.compile(
            r"^\s*(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.+?)$",
            re.MULTILINE,
        )
        for match in arg_pattern.finditer(args_section):
            param_name = match.group(1)
            type_info = match.group(2) or "string"
            param_desc = match.group(3).strip()
            is_optional = "optional" in type_info.lower()
            param_type = re.sub(r",?\s*optional", "", type_info, flags=re.IGNORECASE).strip()
            if not param_type:
                param_type = "string"
            parameters[param_name] = {
                "type": param_type,
                "description": param_desc,
                "required": not is_optional,
            }

    return {
        "name": func.__name__,
        "description": description,
        "parameters": parameters,
        "success_message": success_message,
    }


def format_discoverable_tool_for_agent(tool_info: dict[str, Any]) -> str:
    """Format a discoverable tool's info for display to the agent.

    Args:
        tool_info: Dictionary from `parse_discoverable_tool_docstring`.

    Returns:
        Formatted string for agent display.
    """
    param_strs: list[str] = []
    for param_name, param_def in tool_info.get("parameters", {}).items():
        required = param_def.get("required", True)
        param_type = param_def.get("type", "string")
        desc = param_def.get("description", "")
        req_str = " (required)" if required else " (optional)"
        param_strs.append(f"  - {param_name}: {param_type}{req_str} - {desc}")

    params_section = "\n".join(param_strs) if param_strs else "  (no parameters)"
    return (
        f"Tool: {tool_info['name']}\n"
        f"Description: {tool_info['description']}\n"
        f"Parameters:\n{params_section}"
    )


def _validate_pin(pin: str) -> str | None:
    """Validate a PIN meets security requirements. Returns None if valid."""
    if not pin or not pin.isdigit() or len(pin) != 4:
        return "PIN must be exactly 4 digits."

    sequential_pins = [
        "0123",
        "1234",
        "2345",
        "3456",
        "4567",
        "5678",
        "6789",
        "9876",
        "8765",
        "7654",
        "6543",
        "5432",
        "4321",
        "3210",
    ]
    if pin in sequential_pins:
        return "PIN cannot be sequential (e.g., 1234). Please choose a more secure PIN."

    if len(set(pin)) == 1:
        return "PIN cannot be all the same digit (e.g., 1111). Please choose a more secure PIN."

    return None


def _validate_activation_common(
    args: dict[str, Any],
    db: TransactionalDB,
    allowed_issue_reasons: list[str],
    tool_name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Common validation for debit card activation tools."""
    card_id = args.get("card_id")
    last_4_digits = args.get("last_4_digits")
    expiration_date = args.get("expiration_date")
    cvv = args.get("cvv")
    pin = args.get("pin")

    if not all([card_id, last_4_digits, expiration_date, cvv, pin]):
        return (
            "Error: Missing required parameters. Required: card_id, last_4_digits, expiration_date, cvv, pin.",
            None,
        )

    pin_error = _validate_pin(pin)
    if pin_error:
        return f"Error: {pin_error}", None

    if not cvv.isdigit() or len(cvv) != 3:
        return "Error: CVV must be exactly 3 digits.", None

    if not last_4_digits.isdigit() or len(last_4_digits) != 4:
        return "Error: Last 4 digits must be exactly 4 digits.", None

    if card_id not in db.debit_cards.data:
        return f"Error: Debit card '{card_id}' not found.", None

    card = db.debit_cards.data[card_id]

    issue_reason = card.get("issue_reason", "new_account")
    if issue_reason not in allowed_issue_reasons:
        reason_map = {
            "new_account": "activate_debit_card_8291",
            "first_card": "activate_debit_card_8291",
            "lost": "activate_debit_card_8292",
            "stolen": "activate_debit_card_8292",
            "fraud": "activate_debit_card_8292",
            "expired": "activate_debit_card_8293",
            "damaged": "activate_debit_card_8293",
            "upgrade": "activate_debit_card_8293",
            "bank_reissue": "activate_debit_card_8293",
        }
        correct_tool = reason_map.get(issue_reason, "unknown")
        return (
            f"Error: Wrong activation tool. This card has issue_reason='{issue_reason}'. "
            f"Please use {correct_tool} instead of {tool_name}.",
            None,
        )

    if card.get("status") == "ACTIVE":
        return f"Error: Debit card '{card_id}' is already active.", None

    if card.get("status") != "PENDING":
        return (
            f"Error: Debit card '{card_id}' cannot be activated. Current status: "
            f"{card.get('status')}. Only PENDING cards can be activated.",
            None,
        )

    if card.get("last_4_digits") != last_4_digits:
        return (
            "Error: Card verification failed. The last 4 digits do not match our records.",
            None,
        )

    account_id = card.get("account_id")
    if account_id and account_id in db.accounts.data:
        account = db.accounts.data[account_id]
        if account.get("status") != "OPEN":
            return (
                f"Error: The linked checking account '{account_id}' is no longer open. "
                f"Card cannot be activated.",
                None,
            )

    return None, card


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


class TransferToHumanAgentsInput(BaseModel):
    """Input schema for the transfer_to_human_agents tool."""

    summary: str = Field(description="Summary of the user's issue.")
    reason: TransferReasonLiteral = Field(
        default="other",
        description="Reason for transfer. Must be one of the allowed specific reasons.",
    )


def create_rhobank_tools(
    db: TransactionalDB,
) -> tuple[list[StructuredTool], list[ToolCallEntry], list[StructuredTool], list[ToolCallEntry]]:
    """Create LangChain tools backed by the given TransactionalDB instance.

    Returns agent tools, agent tool log, user tools, and user tool log.
    Each tool mutates the shared `db` and logs its invocation.

    Args:
        db: The mutable TransactionalDB instance.

    Returns:
        A tuple of (agent_tools, agent_log, user_tools, user_log).
    """
    agent_log: list[ToolCallEntry] = []
    user_log: list[ToolCallEntry] = []

    # State for discoverable tools protocol
    user_discoverable_tools_state: dict[str, dict[str, Any]] = {}
    agent_discoverable_tools_state: dict[str, dict[str, Any]] = {}

    def _log_agent(name: str, args: dict[str, Any], result: str) -> str:
        agent_log.append(ToolCallEntry(name=name, args=args, result=result))
        return result

    def _log_agent_error(name: str, args: dict[str, Any], error: str) -> str:
        agent_log.append(ToolCallEntry(name=name, args=args, result=error, error=True))
        return error

    def _log_user(name: str, args: dict[str, Any], result: str) -> str:
        user_log.append(ToolCallEntry(name=name, args=args, result=result, requestor="user"))
        return result

    # ===================================================================
    # Agent discoverable tool METHOD registry
    # (methods the agent can unlock and call via call_discoverable_agent_tool)
    # ===================================================================

    def _get_all_user_accounts_by_user_id_3847(user_id: str) -> str:
        """Get all bank and credit card accounts for a user.

        Args:
            user_id (str): The unique identifier of the user.

        Returns:
            Formatted string listing all found accounts.
        """
        if not user_id:
            return "Error: Missing required parameter: user_id"
        accounts_result = query_database_tool("accounts", f'{{"user_id": "{user_id}"}}', db=db)
        cc_result = query_database_tool(
            "credit_card_accounts", f'{{"user_id": "{user_id}"}}', db=db
        )
        result_parts = [
            "User accounts retrieved successfully.",
            "",
            "Executed: get_all_user_accounts_by_user_id_3847",
            f"Accounts for user {user_id}:",
            "",
            "Bank Accounts:",
        ]
        if "No records found" not in accounts_result and "No results found" not in accounts_result:
            result_parts.append(accounts_result)
        else:
            result_parts.append("  No bank accounts found.")
        result_parts.append("\nCredit Card Accounts:")
        if "No records found" not in cc_result and "No results found" not in cc_result:
            result_parts.append(cc_result)
        else:
            result_parts.append("  No credit card accounts found.")
        return "\n".join(result_parts)

    def _get_debit_cards_by_account_id_7823(account_id: str) -> str:
        """Get all debit cards associated with a checking account.

        Args:
            account_id (str): The checking account ID to lookup debit cards for.

        Returns:
            JSON array string of debit cards associated with the account.
        """
        if not account_id:
            return "Error: Missing required parameter 'account_id'."
        if account_id not in db.accounts.data:
            return f"Error: Account '{account_id}' not found."
        account = db.accounts.data[account_id]
        account_class = account.get("class", "").lower()
        if account_class not in ["checking", "business_checking"]:
            return f"Error: Account '{account_id}' is not a checking account."
        account_cards: list[dict[str, Any]] = []
        for card_id, card in db.debit_cards.data.items():
            if card.get("account_id") == account_id:
                card_info: dict[str, Any] = {"card_id": card_id}
                card_info.update(card)
                if "last_4_digits" in card_info and "card_number_last_4" not in card_info:
                    card_info["card_number_last_4"] = card_info.pop("last_4_digits")
                if "issue_date" in card_info and "date_issued" not in card_info:
                    card_info["date_issued"] = card_info.pop("issue_date")
                elif "created_date" in card_info and "date_issued" not in card_info:
                    card_info["date_issued"] = card_info.pop("created_date")
                account_cards.append(card_info)
        if not account_cards:
            return f"No debit cards found for account '{account_id}'."
        account_cards.sort(key=lambda x: x.get("date_issued") or "", reverse=True)
        return json.dumps(account_cards, indent=2)

    def _freeze_debit_card_3892(card_id: str) -> str:
        """Freeze a debit card to temporarily prevent new transactions.

        Args:
            card_id (str): The ID of the debit card to freeze.

        Returns:
            Success message with freezing details.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id."
        if card_id not in db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."
        card = db.debit_cards.data[card_id]
        if card.get("status") == "FROZEN":
            return f"Error: Debit card '{card_id}' is already frozen."
        if card.get("status") != "ACTIVE":
            return f"Error: Debit card '{card_id}' cannot be frozen. Current status: {card.get('status')}. Only ACTIVE cards can be frozen."
        card["status"] = "FROZEN"
        card["frozen_date"] = get_today_str()
        return "\n".join(
            [
                "Debit Card Frozen Successfully",
                f"Card ID: {card_id}",
                "Status: FROZEN",
                f"Frozen Date: {get_today_str()}",
                "",
                "While frozen:",
                "- All new purchase transactions will be declined",
                "- Recurring payments and subscriptions will be declined",
                "- Pending transactions already authorized may still process",
                "",
                "To unfreeze, the customer can call customer service or use the mobile app.",
                "If the card is confirmed lost or stolen, recommend closing the card permanently instead.",
            ]
        )

    def _unfreeze_debit_card_3893(card_id: str) -> str:
        """Unfreeze a previously frozen debit card.

        Args:
            card_id (str): The ID of the frozen debit card to unfreeze.

        Returns:
            Success message confirming the card is active again.
        """
        if not card_id:
            return "Error: Missing required parameter: card_id."
        if card_id not in db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."
        card = db.debit_cards.data[card_id]
        if card.get("status") == "ACTIVE":
            return f"Error: Debit card '{card_id}' is already active."
        if card.get("status") != "FROZEN":
            return f"Error: Debit card '{card_id}' cannot be unfrozen. Current status: {card.get('status')}. Only FROZEN cards can be unfrozen."
        account_id = card.get("account_id")
        if account_id and account_id in db.accounts.data:
            account = db.accounts.data[account_id]
            if account.get("status") != "OPEN":
                return f"Error: The linked checking account '{account_id}' is no longer open. Card cannot be unfrozen."
        card["status"] = "ACTIVE"
        card["unfrozen_date"] = get_today_str()
        return "\n".join(
            [
                "Debit Card Unfrozen Successfully",
                f"Card ID: {card_id}",
                "Status: ACTIVE",
                f"Unfrozen Date: {get_today_str()}",
                "",
                "The card is now active and ready to use immediately.",
                "All transactions will process normally.",
            ]
        )

    def _close_debit_card_4721(card_id: str, reason: str) -> str:
        """Permanently close a debit card.

        Args:
            card_id (str): The ID of the debit card to close.
            reason (str): The reason for closure (e.g. 'lost', 'stolen', 'fraud_suspected', 'damaged', 'no_longer_needed', 'account_closing').

        Returns:
            Success message with closure details.
        """
        if not card_id or not reason:
            return "Error: Missing required parameters. Required: card_id, reason."
        valid_reasons = [
            "lost",
            "stolen",
            "fraud_suspected",
            "damaged",
            "no_longer_needed",
            "account_closing",
        ]
        if reason.lower() not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"
        reason = reason.lower()
        if card_id not in db.debit_cards.data:
            return f"Error: Debit card '{card_id}' not found."
        card = db.debit_cards.data[card_id]
        if card.get("status") not in ["ACTIVE", "PENDING"]:
            return f"Error: Debit card '{card_id}' cannot be closed. Current status: {card.get('status')}. Only ACTIVE or PENDING cards can be closed."
        previous_status = card.get("status")
        card["status"] = "CLOSED"
        card["closed_date"] = get_today_str()
        card["closure_reason"] = reason
        result_parts = [
            "Debit Card Closed Successfully",
            f"Card ID: {card_id}",
            f"Previous Status: {previous_status}",
            "New Status: CLOSED",
            f"Closure Reason: {reason.replace('_', ' ').title()}",
            f"Closure Date: {get_today_str()}",
            "",
        ]
        if reason in ["lost", "stolen", "fraud_suspected"]:
            result_parts.append(
                "IMPORTANT: This card has been immediately deactivated for security."
            )
            result_parts.append("Any pending transactions may still be processed.")
            if reason == "fraud_suspected":
                result_parts.append(
                    "Please advise the customer to review recent transactions and file disputes for any unauthorized charges."
                )
                result_parts.append("Also recommend changing their online banking password.")
        result_parts.append("")
        result_parts.append(
            "Note: This card cannot be reactivated. If the customer needs a new card, they can order one through the standard ordering process."
        )
        result_parts.append(
            "Any recurring payments linked to this card will need to be updated with new payment information."
        )
        return "\n".join(result_parts)

    def _get_bank_account_transactions_9173(account_id: str) -> str:
        """Get all transactions for a specific bank account.

        Args:
            account_id (str): The bank account ID to retrieve transactions for.

        Returns:
            Formatted string containing transaction history.
        """
        if not account_id:
            return "Error: Missing required parameter: account_id"
        if account_id not in db.accounts.data:
            return f"Error: Account '{account_id}' not found."
        txn_result = query_database_tool(
            "bank_account_transaction_history", f'{{"account_id": "{account_id}"}}', db=db
        )
        result_parts = [
            "Bank account transactions retrieved successfully.",
            "",
            "Executed: get_bank_account_transactions_9173",
            f"Transactions for account {account_id}:",
        ]
        if "No records found" not in txn_result and "No results found" not in txn_result:
            result_parts.append(txn_result)
        else:
            result_parts.append("\nNo transactions found for this account.")
        return "\n".join(result_parts)

    def _order_debit_card_5739(
        account_id: str,
        user_id: str,
        delivery_option: str,
        delivery_fee: float,
        card_design: str,
        design_fee: float,
        shipping_address: str,
        excess_replacement_fee: float | None = None,
    ) -> str:
        """Order a new or replacement debit card.

        Args:
            account_id (str): The checking account ID to link the card to.
            user_id (str): The user ID of the account owner.
            delivery_option (str): Shipping method ('STANDARD', 'EXPEDITED', 'RUSH').
            delivery_fee (float): Fee amount for the selected delivery option.
            card_design (str): The selected card design ('CLASSIC', 'PREMIUM', 'CUSTOM').
            design_fee (float): Fee amount for the selected card design.
            shipping_address (str): The address to ship the card to.
            excess_replacement_fee (float, optional): Additional fee if the user has exceeded their free replacement limit.

        Returns:
            Success message with order details and charged fees.
        """
        excess_replacement_fee = excess_replacement_fee or 0
        try:
            delivery_fee = float(delivery_fee) if delivery_fee is not None else None
        except (TypeError, ValueError):
            return "Error: delivery_fee must be a number."
        try:
            design_fee = float(design_fee) if design_fee is not None else None
        except (TypeError, ValueError):
            return "Error: design_fee must be a number."
        try:
            excess_replacement_fee = float(excess_replacement_fee)
        except (TypeError, ValueError):
            excess_replacement_fee = 0

        if not all(
            [
                account_id,
                user_id,
                delivery_option,
                delivery_fee is not None,
                card_design,
                design_fee is not None,
                shipping_address,
            ]
        ):
            return "Error: Missing required parameters. Required: account_id, user_id, delivery_option, delivery_fee, card_design, design_fee, shipping_address."

        valid_delivery = ["STANDARD", "EXPEDITED", "RUSH"]
        if delivery_option.upper() not in valid_delivery:
            return f"Error: Invalid delivery_option. Must be one of: {valid_delivery}"
        delivery_option = delivery_option.upper()

        valid_design = ["CLASSIC", "PREMIUM", "CUSTOM"]
        if card_design.upper() not in valid_design:
            return f"Error: Invalid card_design. Must be one of: {valid_design}"
        card_design = card_design.upper()

        if account_id not in db.accounts.data:
            return f"Error: Account '{account_id}' not found."
        account = db.accounts.data[account_id]
        if account.get("class") != "checking":
            return f"Error: Debit cards can only be ordered for checking accounts. Account '{account_id}' is a {account.get('class')} account."
        if account.get("status") != "OPEN":
            return f"Error: Account must be OPEN. Account '{account_id}' has status: {account.get('status')}"
        if account.get("user_id") != user_id:
            return f"Error: Account '{account_id}' does not belong to user '{user_id}'."

        try:
            current_holdings = float(str(account.get("current_holdings", "0")).replace(",", ""))
        except ValueError:
            current_holdings = 0.0
        if current_holdings < 25.0:
            return f"Error: Account must have a minimum balance of $25. Current balance: ${current_holdings:.2f}"

        for order in db.debit_card_orders.data.values():
            if order.get("account_id") == account_id and order.get("status") == "PENDING":
                return f"Error: There is already a pending debit card order for account '{account_id}'."

        active_cards_count = sum(
            1
            for card in db.debit_cards.data.values()
            if card.get("account_id") == account_id and card.get("status") == "ACTIVE"
        )
        if active_cards_count >= 1:
            return f"Error: Account '{account_id}' already has an active debit card. Maximum 1 active card per checking account."

        total_fee = delivery_fee + design_fee + excess_replacement_fee
        if total_fee > 0 and current_holdings < total_fee:
            return f"Error: Insufficient funds for fees. Total fees: ${total_fee:.2f}. Current balance: ${current_holdings:.2f}"

        delivery_times = {
            "STANDARD": "7-10 business days",
            "EXPEDITED": "3-5 business days",
            "RUSH": "1-2 business days",
        }
        expected_delivery = delivery_times[delivery_option]

        order_date = get_today_str()
        order_id = generate_debit_card_order_id(account_id, user_id, delivery_option)

        order_record = {
            "order_id": order_id,
            "account_id": account_id,
            "user_id": user_id,
            "delivery_option": delivery_option,
            "card_design": card_design,
            "shipping_address": shipping_address,
            "delivery_fee": delivery_fee,
            "design_fee": design_fee,
            "excess_replacement_fee": excess_replacement_fee,
            "total_fee": total_fee,
            "order_date": order_date,
            "expected_delivery": expected_delivery,
            "status": "PENDING",
        }
        success = add_to_db("debit_card_orders", order_id, order_record, db=db)
        if not success:
            return "Error: Failed to create debit card order. Order may already exist."

        if total_fee > 0:
            new_balance = current_holdings - total_fee
            db.accounts.data[account_id]["current_holdings"] = f"{new_balance:.2f}"
            fee_description_parts = []
            if delivery_fee > 0:
                fee_description_parts.append(f"Delivery ${delivery_fee}")
            if design_fee > 0:
                fee_description_parts.append(f"Design ${design_fee}")
            if excess_replacement_fee > 0:
                fee_description_parts.append(f"Excess Replacement ${excess_replacement_fee:.0f}")
            fee_description = f"DEBIT CARD ORDER FEE - {', '.join(fee_description_parts)}"
            fee_txn_id = f"btxn_dcfee_{order_id[-8:]}"
            fee_transaction = {
                "transaction_id": fee_txn_id,
                "account_id": account_id,
                "date": order_date,
                "description": fee_description,
                "amount": -total_fee,
                "type": "debit_card_fee",
                "status": "posted",
            }
            add_to_db("bank_account_transaction_history", fee_txn_id, fee_transaction, db=db)

        card_id = generate_debit_card_id(account_id, user_id, order_date)
        card_details_seed = f"card_details:{card_id}"
        last_4_hash = _deterministic_id(card_details_seed + ":last4", length=8)
        cvv_hash = _deterministic_id(card_details_seed + ":cvv", length=6)
        last_4_digits = "".join(c for c in last_4_hash if c.isdigit())[:4].zfill(4)
        cvv = "".join(c for c in cvv_hash if c.isdigit())[:3].zfill(3)

        cardholder_name = "CARDHOLDER"
        if user_id in db.users.data:
            user = db.users.data[user_id]
            cardholder_name = user.get("name", "CARDHOLDER").upper()

        try:
            parts = order_date.split("/")
            exp_month = parts[0]
            exp_year = str(int(parts[2]) + 4)
            if exp_month in ["01", "03", "05", "07", "08", "10", "12"]:
                exp_day = "31"
            elif exp_month == "02":
                exp_day = "28"
            else:
                exp_day = "30"
            expiration_date = f"{exp_month}/{exp_day}/{exp_year}"
        except (ValueError, IndexError):
            expiration_date = "12/31/2029"

        existing_cards = [
            c for c in db.debit_cards.data.values() if c.get("account_id") == account_id
        ]
        closed_card = next((c for c in existing_cards if c.get("status") == "CLOSED"), None)
        if closed_card:
            closure_reason = closed_card.get("closure_reason", "first_card")
            if closure_reason in ["lost", "stolen", "fraud", "fraud_suspected"]:
                issue_reason = closure_reason if closure_reason != "fraud_suspected" else "fraud"
            else:
                issue_reason = "first_card"
        elif existing_cards:
            issue_reason = "first_card"
        else:
            issue_reason = "new_account"

        card_record = {
            "card_id": card_id,
            "account_id": account_id,
            "user_id": user_id,
            "cardholder_name": cardholder_name,
            "last_4_digits": last_4_digits,
            "cvv": cvv,
            "status": "PENDING",
            "issue_date": order_date,
            "expiration_date": expiration_date,
            "card_design": card_design,
            "issue_reason": issue_reason,
        }
        add_to_db("debit_cards", card_id, card_record, db=db)

        result_parts = [
            "Debit Card Order Confirmed",
            f"Order ID: {order_id}",
            f"Card ID: {card_id}",
            f"Linked Account: {account_id}",
            f"Delivery Option: {delivery_option}",
            f"Card Design: {card_design}",
            f"Shipping Address: {shipping_address}",
            f"Expected Delivery: {expected_delivery}",
            "",
            "Note: Card will arrive with status PENDING. Customer must call to activate after receiving the card.",
        ]
        if total_fee > 0:
            fee_details = []
            if delivery_fee > 0:
                fee_details.append(f"Delivery: ${delivery_fee}")
            if design_fee > 0:
                fee_details.append(f"Design: ${design_fee}")
            if excess_replacement_fee > 0:
                fee_details.append(f"Excess Replacement: ${excess_replacement_fee:.0f}")
            result_parts.append(
                f"Total Fees: ${total_fee:.2f} ({', '.join(fee_details)}) - CHARGED to account {account_id}"
            )
            result_parts.append(f"New Account Balance: ${current_holdings - total_fee:.2f}")
        else:
            result_parts.append("Total Fees: $0 (No additional charges)")
        return "\n".join(result_parts)

    def _order_replacement_credit_card_7291(
        credit_card_account_id: str,
        user_id: str,
        shipping_address: str,
        reason: str,
        expedited_shipping: bool = False,
    ) -> str:
        """Order a replacement for an existing credit card.

        Args:
            credit_card_account_id (str): The ID of the credit card account.
            user_id (str): The user ID of the account owner.
            shipping_address (str): The address to ship the replacement card to.
            reason (str): The reason for replacement (e.g. 'fraud_suspected', 'lost', 'stolen', 'damaged', 'expired', 'other').
            expedited_shipping (bool, optional): Whether to use expedited shipping (default False).

        Returns:
            Success message confirming the replacement order.
        """
        if not credit_card_account_id or not user_id or not shipping_address or not reason:
            return "Error: Missing required parameters (credit_card_account_id, user_id, shipping_address, reason)."
        valid_reasons = ["fraud_suspected", "lost", "stolen", "damaged", "expired", "other"]
        if reason not in valid_reasons:
            return f"Error: Invalid reason. Must be one of: {valid_reasons}"
        result = query_database_tool(
            "credit_card_accounts", f'{{"account_id": "{credit_card_account_id}"}}', db=db
        )
        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."
        order_id = generate_credit_card_order_id(credit_card_account_id, user_id, reason)
        today = get_today_str()
        order_record = {
            "order_id": order_id,
            "credit_card_account_id": credit_card_account_id,
            "user_id": user_id,
            "shipping_address": shipping_address,
            "reason": reason,
            "expedited_shipping": expedited_shipping,
            "order_date": today,
            "status": "ORDERED",
            "old_card_cancelled": True,
        }
        success = add_to_db("credit_card_orders", order_id, order_record, db=db)
        if not success:
            return "Error: Order may have already been placed for this card replacement."
        if credit_card_account_id in db.credit_card_accounts.data:
            db.credit_card_accounts.data[credit_card_account_id]["status"] = "CLOSED"
            db.credit_card_accounts.data[credit_card_account_id]["closed_date"] = get_today_str()
        shipping_method = "Expedited" if expedited_shipping else "Standard"
        expected_delivery = "2-3 business days" if expedited_shipping else "7-10 business days"
        return "\n".join(
            [
                f"Order ID: {order_id}",
                f"Card Account: {credit_card_account_id}",
                f"Reason: {reason.replace('_', ' ').title()}",
                f"Shipping Address: {shipping_address}",
                f"Shipping Method: {shipping_method}",
                f"Expected Delivery: {expected_delivery}",
                "",
                "The old card has been cancelled for security. The new card will have the same account number but a new card number and CVV.",
            ]
        )

    def _get_user_dispute_history_7291(user_id: str) -> str:
        """Retrieve the transaction dispute history for a user.

        Args:
            user_id (str): The ID of the user.

        Returns:
            Formatted string with the user's dispute history.
        """
        if not user_id:
            return "Error: Missing required parameter: user_id"
        transaction_disputes_result = query_database_tool(
            "transaction_disputes", f'{{"user_id": "{user_id}"}}', db=db
        )
        result_parts = [
            "User transaction dispute history retrieved successfully.",
            "",
            "Executed: get_user_dispute_history_7291",
            f"Transaction dispute history for user {user_id}:",
        ]
        has_disputes = (
            "No records found" not in transaction_disputes_result
            and "No results found" not in transaction_disputes_result
        )
        if has_disputes:
            result_parts.append(transaction_disputes_result)
        else:
            result_parts.append("\nNo transaction disputes found for this user.")
        return "\n".join(result_parts)

    def _file_credit_card_transaction_dispute_4829(
        transaction_id: str,
        card_action: str,
        card_last_4_digits: str,
        full_name: str,
        user_id: str,
        phone: str,
        email: str,
        address: str,
        contacted_merchant: bool,
        purchase_date: str,
        issue_noticed_date: str,
        dispute_reason: str,
        resolution_requested: str,
        eligible_for_provisional_credit: bool,
        partial_refund_amount: float | None = None,
    ) -> str:
        """File a formal dispute for a credit card transaction.

        Args:
            transaction_id (str): The ID of the disputed transaction.
            card_action (str): What to do with the card ('keep_active' or 'cancel_and_reissue').
            card_last_4_digits (str): The last 4 digits of the card.
            full_name (str): The customer's full name.
            user_id (str): The customer's user ID.
            phone (str): The customer's phone number.
            email (str): The customer's email address.
            address (str): The customer's billing address.
            contacted_merchant (bool): True if the customer has already contacted the merchant.
            purchase_date (str): The date the purchase was made.
            issue_noticed_date (str): The date the customer noticed the issue.
            dispute_reason (str): The reason for the dispute (e.g. 'unauthorized_fraudulent_charge', 'duplicate_charge', etc.).
            resolution_requested (str): The resolution sought ('full_refund' or 'partial_refund').
            eligible_for_provisional_credit (bool): True if the dispute is eligible for provisional credit.
            partial_refund_amount (float, optional): The amount requested if partial_refund is chosen.

        Returns:
            Success message with dispute details and provisional credit info.
        """
        if not transaction_id or not user_id:
            return "Error: Missing required parameters."
        valid_card_actions = ["keep_active", "cancel_and_reissue"]
        if card_action not in valid_card_actions:
            return f"Error: Invalid card_action. Must be one of: {valid_card_actions}"
        valid_reasons = [
            "unauthorized_fraudulent_charge",
            "duplicate_charge",
            "incorrect_amount",
            "goods_services_not_received",
            "goods_services_not_as_described",
            "canceled_subscription_still_charging",
            "refund_never_processed",
        ]
        if dispute_reason not in valid_reasons:
            return f"Error: Invalid dispute_reason. Must be one of: {valid_reasons}"
        valid_resolutions = ["full_refund", "partial_refund"]
        if resolution_requested not in valid_resolutions:
            return f"Error: Invalid resolution_requested. Must be one of: {valid_resolutions}"
        if resolution_requested == "partial_refund" and partial_refund_amount is None:
            return "Error: partial_refund_amount is required when resolution_requested is 'partial_refund'."

        dispute_id = generate_dispute_id(user_id, transaction_id)
        dispute_record = {
            "dispute_id": dispute_id,
            "transaction_id": transaction_id,
            "user_id": user_id,
            "card_action": card_action,
            "card_last_4_digits": card_last_4_digits,
            "full_name": full_name,
            "phone": phone,
            "email": email,
            "address": address,
            "contacted_merchant": contacted_merchant,
            "purchase_date": purchase_date,
            "issue_noticed_date": issue_noticed_date,
            "dispute_reason": dispute_reason,
            "resolution_requested": resolution_requested,
            "partial_refund_amount": partial_refund_amount,
            "eligible_for_provisional_credit": eligible_for_provisional_credit,
            "provisional_credit_given": eligible_for_provisional_credit,
            "submitted_at": get_today_str(),
            "status": "SUBMITTED",
        }
        success = add_to_db("transaction_disputes", dispute_id, dispute_record, db=db)
        if not success:
            return "Error: Dispute may have already been filed for this transaction."

        result_parts = [
            "Credit card transaction dispute filed successfully. A case has been opened and will be reviewed within 10 business days.",
            "",
            "Executed: file_credit_card_transaction_dispute_4829",
            f"Dispute ID: {dispute_id}",
            f"Transaction: {transaction_id}",
            f"Reason: {dispute_reason.replace('_', ' ').title()}",
            f"Resolution Requested: {resolution_requested.replace('_', ' ').title()}",
        ]
        if partial_refund_amount:
            result_parts.append(f"Partial Refund Amount: ${partial_refund_amount:.2f}")
        if eligible_for_provisional_credit:
            result_parts.append(
                "Provisional Credit: ELIGIBLE - Credit will be applied within 2 business days."
            )
        else:
            result_parts.append("Provisional Credit: Not eligible at this time.")
        return "\n".join(result_parts)

    def _activate_debit_card_8292(
        card_id: str,
        last_4_digits: str,
        expiration_date: str,
        cvv: str,
        pin: str,
    ) -> str:
        """Activate a replacement debit card (for lost/stolen/fraud scenarios).

        Args:
            card_id (str): The ID of the card to activate.
            last_4_digits (str): The last 4 digits of the card.
            expiration_date (str): The expiration date of the card.
            cvv (str): The 3-digit CVV code on the back of the card.
            pin (str): The 4-digit PIN the customer wishes to set.

        Returns:
            Success message confirming the card is activated.
        """
        args = {
            "card_id": card_id,
            "last_4_digits": last_4_digits,
            "expiration_date": expiration_date,
            "cvv": cvv,
            "pin": pin,
        }
        error, card = _validate_activation_common(
            args, db, ["lost", "stolen", "fraud"], "activate_debit_card_8292"
        )
        if error:
            return error
        account_id = card.get("account_id")
        issue_reason = card.get("issue_reason", "lost")
        card["status"] = "ACTIVE"
        card["activated_date"] = get_today_str()
        deactivated_cards: list[str] = []
        for other_card_id, other_card in db.debit_cards.data.items():
            if (
                other_card_id != card_id
                and other_card.get("account_id") == account_id
                and other_card.get("status") == "ACTIVE"
            ):
                other_card["status"] = "DEACTIVATED"
                other_card["deactivated_date"] = get_today_str()
                other_card["deactivation_reason"] = f"Replacement card activated ({issue_reason})"
                deactivated_cards.append(other_card_id)
        result_parts = [
            "Replacement Debit Card Activation Successful",
            f"Card ID: {card_id}",
            f"Replacement Reason: {issue_reason.replace('_', ' ').title()}",
            "Status: ACTIVE",
            f"Activation Date: {get_today_str()}",
            "",
            "Your replacement card is now ready to use.",
            "",
            "IMPORTANT SECURITY REMINDERS:",
            "- Please review your recent transactions for any unauthorized charges",
            "- Report any suspicious activity immediately",
        ]
        if issue_reason == "fraud":
            result_parts.append(
                "- Since fraud was suspected, we recommend changing your online banking password"
            )
        if deactivated_cards:
            result_parts.append(
                f"\nPrevious card(s) have been deactivated for security: {', '.join(deactivated_cards)}"
            )
        return "\n".join(result_parts)

    agent_discoverable_registry: dict[str, Any] = {
        "get_all_user_accounts_by_user_id_3847": _get_all_user_accounts_by_user_id_3847,
        "get_debit_cards_by_account_id_7823": _get_debit_cards_by_account_id_7823,
        "freeze_debit_card_3892": _freeze_debit_card_3892,
        "unfreeze_debit_card_3893": _unfreeze_debit_card_3893,
        "close_debit_card_4721": _close_debit_card_4721,
        "get_bank_account_transactions_9173": _get_bank_account_transactions_9173,
        "order_debit_card_5739": _order_debit_card_5739,
        "order_replacement_credit_card_7291": _order_replacement_credit_card_7291,
        "get_user_dispute_history_7291": _get_user_dispute_history_7291,
        "file_credit_card_transaction_dispute_4829": _file_credit_card_transaction_dispute_4829,
        "activate_debit_card_8292": _activate_debit_card_8292,
    }

    # ===================================================================
    # User discoverable tool METHOD registry
    # ===================================================================

    def _submit_cash_back_dispute_0589(user_id: str, transaction_id: str) -> str:
        """Submit a cash back dispute for a specific transaction.

        Args:
            user_id (str): The customer's user ID.
            transaction_id (str): The ID of the transaction to dispute.

        Returns:
            Status of the cash back dispute submission.
        """
        check_result = query_database_tool(
            "user_discoverable_tools", '{"tool_name": "submit_cash_back_dispute_0589"}', db=db
        )
        if "No records found" in check_result:
            return "Error: Tool 'submit_cash_back_dispute_0589' has not been given to you by the agent."
        call_args = {"user_id": user_id, "transaction_id": transaction_id}
        call_record = {
            "tool_name": "submit_cash_back_dispute_0589",
            "arguments": call_args,
            "called_at": get_today_str(),
            "status": "CALLED",
        }
        call_id = generate_user_discoverable_tool_call_id(
            "submit_cash_back_dispute_0589", call_args
        )
        add_to_db("user_discoverable_tool_calls", call_id, call_record, db=db)

        dispute_id = generate_dispute_id(user_id, transaction_id)
        auto_resolve = False
        if db.task_config.data:
            config = db.task_config.data.get("dispute_settings", {})
            auto_resolve = config.get("auto_resolve_disputes", False)
        if auto_resolve:
            dispute_record = {
                "dispute_id": dispute_id,
                "user_id": user_id,
                "transaction_id": transaction_id,
                "submitted_at": get_today_str(),
                "status": "RESOLVED",
                "resolution": "APPROVED",
            }
            status_msg = "Status: RESOLVED - The dispute has been reviewed and approved. The transaction rewards need to be updated."
        else:
            dispute_record = {
                "dispute_id": dispute_id,
                "user_id": user_id,
                "transaction_id": transaction_id,
                "submitted_at": get_today_str(),
                "status": "SUBMITTED",
            }
            status_msg = "Status: SUBMITTED - Your dispute has been queued for review."
        success = add_to_db("cash_back_disputes", dispute_id, dispute_record, db=db)
        result = f"Cash back dispute submitted successfully. Your case has been queued for review.\n\nExecuted: submit_cash_back_dispute_0589\nArguments: {json.dumps(call_args, indent=2)}\n"
        if success:
            result += f"Dispute ID: {dispute_id}\n{status_msg}"
        else:
            result += "Note: Dispute may have already been submitted for this transaction."
        return result

    def _get_card_last_4_digits(credit_card_account_id: str) -> str:
        """Retrieve the last 4 digits of a credit card.

        Args:
            credit_card_account_id (str): The credit card account ID.

        Returns:
            The last 4 digits of the requested credit card.
        """
        check_result = query_database_tool(
            "user_discoverable_tools", '{"tool_name": "get_card_last_4_digits"}', db=db
        )
        if "No records found" in check_result:
            return "Error: Tool 'get_card_last_4_digits' has not been given to you by the agent."
        call_args = {"credit_card_account_id": credit_card_account_id}
        call_record = {
            "tool_name": "get_card_last_4_digits",
            "arguments": call_args,
            "called_at": get_today_str(),
            "status": "CALLED",
        }
        call_id = generate_user_discoverable_tool_call_id("get_card_last_4_digits", call_args)
        add_to_db("user_discoverable_tool_calls", call_id, call_record, db=db)

        result = query_database_tool(
            "credit_card_accounts", f'{{"account_id": "{credit_card_account_id}"}}', db=db
        )
        if "No results found" in result or "No records found" in result:
            return f"Error: Credit card account '{credit_card_account_id}' not found."
        hash_input = f"card_last4:{credit_card_account_id}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()
        last_4 = ""
        for char in hash_digest:
            if char.isdigit():
                last_4 += char
                if len(last_4) == 4:
                    break
        last_4 = last_4.ljust(4, "0")
        return f"Card information retrieved successfully.\n\nExecuted: get_card_last_4_digits\nArguments: {json.dumps(call_args, indent=2)}\nLast 4 digits of card: {last_4}"

    user_discoverable_registry: dict[str, Any] = {
        "submit_cash_back_dispute_0589": _submit_cash_back_dispute_0589,
        "get_card_last_4_digits": _get_card_last_4_digits,
    }

    # ===================================================================
    # Standard AGENT tools (always visible in the agent's tool list)
    # ===================================================================

    def get_current_time() -> str:
        return _log_agent("get_current_time", {}, "The current time is 2025-11-14 03:40:00 EST.")

    def get_user_information_by_id(user_id: str) -> str:
        args = {"user_id": user_id}
        return _log_agent(
            "get_user_information_by_id",
            args,
            query_database_tool("users", f'{{"user_id": "{user_id}"}}', db=db),
        )

    def get_user_information_by_name(customer_name: str) -> str:
        args = {"customer_name": customer_name}
        return _log_agent(
            "get_user_information_by_name",
            args,
            query_database_tool("users", f'{{"name": "{customer_name}"}}', db=db),
        )

    def get_user_information_by_email(email: str) -> str:
        args = {"email": email}
        return _log_agent(
            "get_user_information_by_email",
            args,
            query_database_tool("users", f'{{"email": "{email}"}}', db=db),
        )

    def get_credit_card_transactions_by_user(user_id: str) -> str:
        args = {"user_id": user_id}
        return _log_agent(
            "get_credit_card_transactions_by_user",
            args,
            query_database_tool(
                "credit_card_transaction_history", f'{{"user_id": "{user_id}"}}', db=db
            ),
        )

    def get_credit_card_accounts_by_user(user_id: str) -> str:
        args = {"user_id": user_id}
        return _log_agent(
            "get_credit_card_accounts_by_user",
            args,
            query_database_tool("credit_card_accounts", f'{{"user_id": "{user_id}"}}', db=db),
        )

    def log_verification(
        name: str,
        user_id: str,
        address: str,
        email: str,
        phone_number: str,
        date_of_birth: str,
        time_verified: str,
    ) -> str:
        args = {
            "name": name,
            "user_id": user_id,
            "address": address,
            "email": email,
            "phone_number": phone_number,
            "date_of_birth": date_of_birth,
            "time_verified": time_verified,
        }
        record_id = generate_verification_id(user_id, time_verified)
        record = {
            "name": name,
            "user_id": user_id,
            "address": address,
            "email": email,
            "phone_number": phone_number,
            "date_of_birth": date_of_birth,
            "time_verified": time_verified,
        }
        success = add_to_db("verification_history", record_id, record, db=db)
        if not success:
            return _log_agent_error(
                "log_verification", args, "Failed to log verification: Record may already exist."
            )
        return _log_agent(
            "log_verification",
            args,
            f"Verification logged successfully.\n  - User: {name} (ID: {user_id})\n  - Verified at: {time_verified}",
        )

    def give_discoverable_user_tool(discoverable_tool_name: str, arguments: str = "{}") -> str:
        args = {"discoverable_tool_name": discoverable_tool_name, "arguments": arguments}
        if discoverable_tool_name not in user_discoverable_registry:
            return _log_agent_error(
                "give_discoverable_user_tool",
                args,
                f"Error: Unknown discoverable tool '{discoverable_tool_name}'.",
            )
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return _log_agent_error(
                "give_discoverable_user_tool", args, f"Error: Invalid JSON in arguments: {e}"
            )
        user_discoverable_tools_state[discoverable_tool_name] = {
            "arguments": args_dict,
            "given_at": get_now().isoformat(),
        }
        discoverable_tool_record = {"tool_name": discoverable_tool_name, "status": "GIVEN"}
        record_id = generate_user_discoverable_tool_id(discoverable_tool_name)
        add_to_db("user_discoverable_tools", record_id, discoverable_tool_record, db=db)
        args_str = json.dumps(args_dict, indent=2) if args_dict else "(no arguments)"
        return _log_agent(
            "give_discoverable_user_tool",
            args,
            (
                f"Tool given to user: {discoverable_tool_name}\nArguments: {args_str}\n\n"
                f"The user can now execute this action by calling `call_discoverable_user_tool` "
                f"with discoverable_tool_name='{discoverable_tool_name}' and the same arguments."
            ),
        )

    def unlock_discoverable_agent_tool(agent_tool_name: str) -> str:
        args = {"agent_tool_name": agent_tool_name}
        if agent_tool_name not in agent_discoverable_registry:
            return _log_agent_error(
                "unlock_discoverable_agent_tool",
                args,
                f"Error: Unknown agent tool '{agent_tool_name}'. This tool is not available.",
            )
        method = agent_discoverable_registry[agent_tool_name]
        tool_info = parse_discoverable_tool_docstring(method)
        agent_discoverable_tools_state[agent_tool_name] = {
            "unlocked_at": get_now().isoformat(),
            "tool_info": tool_info,
        }
        formatted_tool = format_discoverable_tool_for_agent(tool_info)
        return _log_agent(
            "unlock_discoverable_agent_tool",
            args,
            (
                f"Tool unlocked: {agent_tool_name}\n"
                f"Description: {tool_info['description']}\n\n{formatted_tool}\n\n"
                f"You can now use this tool by calling `call_discoverable_agent_tool` with "
                f"agent_tool_name='{agent_tool_name}' and the required arguments."
            ),
        )

    def call_discoverable_agent_tool(agent_tool_name: str, arguments: str = "{}") -> str:
        args = {"agent_tool_name": agent_tool_name, "arguments": arguments}
        if agent_tool_name not in agent_discoverable_registry:
            return _log_agent_error(
                "call_discoverable_agent_tool",
                args,
                f"Error: Unknown agent tool '{agent_tool_name}'.",
            )
        if agent_tool_name not in agent_discoverable_tools_state:
            return _log_agent_error(
                "call_discoverable_agent_tool",
                args,
                f"Error: Tool '{agent_tool_name}' has not been unlocked.",
            )
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return _log_agent_error(
                "call_discoverable_agent_tool", args, f"Error: Invalid JSON in arguments: {e}"
            )
        method = agent_discoverable_registry[agent_tool_name]
        try:
            result = method(**args_dict)
        except TypeError as e:
            return _log_agent_error(
                "call_discoverable_agent_tool", args, f"Error: Invalid arguments: {e}"
            )
        agent_tool_record = {"tool_name": agent_tool_name, "status": "CALLED"}
        record_id = generate_agent_discoverable_tool_id(agent_tool_name)
        add_to_db("agent_discoverable_tools", record_id, agent_tool_record, db=db)
        return _log_agent("call_discoverable_agent_tool", args, result)

    def list_discoverable_agent_tools() -> str:
        result = query_database_tool("agent_discoverable_tools", "{}", db=db)
        if "No results found" in result:
            return _log_agent(
                "list_discoverable_agent_tools", {}, "No agent tools have been called yet."
            )
        return _log_agent(
            "list_discoverable_agent_tools", {}, f"Your called agent tools:\n{result}"
        )

    def transfer_to_human_agents(summary: str, reason: str = "other") -> str:
        args = {"summary": summary, "reason": reason}
        valid_reasons = list(get_args(TransferReasonLiteral))
        if reason not in valid_reasons:
            return _log_agent_error(
                "transfer_to_human_agents", args, f"Error: Invalid transfer reason '{reason}'."
            )
        return _log_agent(
            "transfer_to_human_agents",
            args,
            f"Transfer successful (reason: {reason}). A human agent will assist you shortly.",
        )

    # Build agent StructuredTools
    agent_tools = [
        StructuredTool.from_function(
            func=get_current_time,
            name="get_current_time",
            description="Get the current time for logging verification records.",
        ),
        StructuredTool.from_function(
            func=get_user_information_by_id,
            name="get_user_information_by_id",
            description="Get user information by their user ID.",
        ),
        StructuredTool.from_function(
            func=get_user_information_by_name,
            name="get_user_information_by_name",
            description="Get user information by their name. Case Sensitive.",
        ),
        StructuredTool.from_function(
            func=get_user_information_by_email,
            name="get_user_information_by_email",
            description="Get user information by their email.",
        ),
        StructuredTool.from_function(
            func=get_credit_card_transactions_by_user,
            name="get_credit_card_transactions_by_user",
            description="Get all credit card transactions for a user.",
        ),
        StructuredTool.from_function(
            func=get_credit_card_accounts_by_user,
            name="get_credit_card_accounts_by_user",
            description="Get all credit card accounts for a user.",
        ),
        StructuredTool.from_function(
            func=log_verification,
            name="log_verification",
            description="Log a verification record after verifying a user's identity.",
        ),
        StructuredTool.from_function(
            func=give_discoverable_user_tool,
            name="give_discoverable_user_tool",
            description="Pass a tool to the user so they can execute it themselves.",
        ),
        StructuredTool.from_function(
            func=unlock_discoverable_agent_tool,
            name="unlock_discoverable_agent_tool",
            description="Unlock an agent discoverable tool found in the knowledge base.",
        ),
        StructuredTool.from_function(
            func=call_discoverable_agent_tool,
            name="call_discoverable_agent_tool",
            description="Call an agent discoverable tool that was previously unlocked.",
        ),
        StructuredTool.from_function(
            func=list_discoverable_agent_tools,
            name="list_discoverable_agent_tools",
            description="List all agent discoverable tools that have been called.",
        ),
        StructuredTool.from_function(
            func=transfer_to_human_agents,
            name="transfer_to_human_agents",
            description="Transfer the user to a human agent with a summary and reason.",
            args_schema=TransferToHumanAgentsInput,
        ),
    ]

    # ===================================================================
    # Standard USER tools (bound to user sim model)
    # ===================================================================

    def call_discoverable_user_tool(discoverable_tool_name: str, arguments: str = "{}") -> str:
        uargs = {"discoverable_tool_name": discoverable_tool_name, "arguments": arguments}
        if discoverable_tool_name not in user_discoverable_registry:
            return _log_user(
                "call_discoverable_user_tool",
                uargs,
                f"Error: Unknown discoverable tool '{discoverable_tool_name}'.",
            )
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            return _log_user(
                "call_discoverable_user_tool", uargs, f"Error: Invalid JSON in arguments: {e}"
            )
        method = user_discoverable_registry[discoverable_tool_name]
        try:
            result = method(**args_dict)
        except TypeError as e:
            return _log_user("call_discoverable_user_tool", uargs, f"Error: Invalid arguments: {e}")
        return _log_user("call_discoverable_user_tool", uargs, result)

    def request_human_agent_transfer() -> str:
        today = get_today_str()
        existing_requests = query_database_tool("human_transfer_requests", "{}", db=db)
        if "No records found" in existing_requests or "No results found" in existing_requests:
            request_count = 1
        else:
            request_count = existing_requests.count("request_id") + 1
        request_id = f"transfer_request_{request_count}"
        record = {
            "request_id": request_id,
            "request_number": request_count,
            "requested_at": today,
            "status": "PENDING",
        }
        add_to_db("human_transfer_requests", request_id, record, db=db)
        return _log_user(
            "request_human_agent_transfer",
            {},
            f"Transfer request #{request_count} submitted.\nThe agent will process your request.",
        )

    def apply_for_credit_card(
        card_type: str,
        customer_name: str,
        annual_income: float,
        rho_bank_subscription: bool = False,
    ) -> str:
        uargs = {
            "card_type": card_type,
            "customer_name": customer_name,
            "annual_income": annual_income,
            "rho_bank_subscription": rho_bank_subscription,
        }
        application_id = generate_application_id(
            card_type, customer_name, annual_income, rho_bank_subscription
        )
        today = get_today_str()
        record = {
            "application_id": application_id,
            "card_type": card_type,
            "customer_name": customer_name,
            "annual_income": annual_income,
            "rho_bank_subscription": rho_bank_subscription,
            "status": "PENDING",
            "date": today,
        }
        success = add_to_db("credit_card_applications", application_id, record, db=db)
        if not success:
            return _log_user(
                "apply_for_credit_card",
                uargs,
                f"Failed to submit application: Record ID '{application_id}' may already exist.",
            )
        return _log_user(
            "apply_for_credit_card",
            uargs,
            "Credit card application submitted:\nYour application has been successfully submitted. You will receive a decision within 5-7 business days via email.",
        )

    user_tools = [
        StructuredTool.from_function(
            func=call_discoverable_user_tool,
            name="call_discoverable_user_tool",
            description="Call a tool that was given to you by the agent.",
        ),
        StructuredTool.from_function(
            func=request_human_agent_transfer,
            name="request_human_agent_transfer",
            description="Request to be transferred to a human agent.",
        ),
        StructuredTool.from_function(
            func=apply_for_credit_card,
            name="apply_for_credit_card",
            description="Apply for a credit card.",
        ),
    ]

    return agent_tools, agent_log, user_tools, user_log
