"""BFCL v3 stateful-API tools for the external-benchmark eval suite.

The API simulator classes live in `bfcl_apis`; this module exposes the class
registry, the agent system prompt, and helpers to wrap their public methods as
tools.

The pytest suite instantiates only each case's involved classes (seeded with
per-case `initial_config`). The registry/dispatcher path uses
`make_bfcl_tools` to bind the full default tool suite — the per-case
involved-class subset and initial config are runtime data, not agent setup.
"""

from __future__ import annotations

import copy
import inspect
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from deepagents_evals.mock_tools.bfcl_apis.message_api import MessageAPI
from deepagents_evals.mock_tools.bfcl_apis.ticket_api import TicketAPI
from deepagents_evals.mock_tools.bfcl_apis.trading_bot import TradingBot
from deepagents_evals.mock_tools.bfcl_apis.travel_booking import TravelAPI
from deepagents_evals.mock_tools.bfcl_apis.vehicle_control import VehicleControlAPI

if TYPE_CHECKING:
    from collections.abc import Iterable

BFCL_CLASS_REGISTRY: dict[str, type] = {
    "VehicleControlAPI": VehicleControlAPI,
    "MessageAPI": MessageAPI,
    "TradingBot": TradingBot,
    "TravelAPI": TravelAPI,
    "TicketAPI": TicketAPI,
}
"""Map BFCL `involved_classes` names to their simulator class."""

BFCL_SYSTEM_PROMPT = (
    "You are an assistant with access to domain-specific API tools. "
    "Use these tools to accomplish the user's requests. "
    "Do not use the task tool or any subagent delegation. "
    "Do not use file tools (ls, read_file, write_file, etc.)."
)


def wrap_bfcl_methods_as_tools(instances: Iterable[Any]) -> list[StructuredTool]:
    """Wrap the public methods of BFCL API instances as StructuredTools."""
    tools: list[StructuredTool] = []
    for instance in instances:
        for method_name, method in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            tools.append(
                StructuredTool.from_function(
                    func=method,
                    name=method_name,
                    description=(method.__doc__ or "").strip(),
                )
            )
    return tools


def make_bfcl_tools(
    *,
    involved_classes: Iterable[str] | None = None,
    initial_config: dict[str, Any] | None = None,
) -> list[StructuredTool]:
    """Build BFCL tools, optionally scoped and seeded to a case.

    Mirrors the pytest BFCL setup: instantiate each class in `involved_classes`
    (defaulting to every registered class) and load its `initial_config` entry
    (defaulting to an empty scenario) before wrapping its methods as tools. The
    case's `involved_classes` / `initial_config` are runtime data the dispatcher
    forwards via `configurable["eval_config"]`.
    """
    names = list(involved_classes) if involved_classes else list(BFCL_CLASS_REGISTRY)
    initial_config = initial_config or {}
    instances = []
    for name in names:
        instance = BFCL_CLASS_REGISTRY[name]()
        instance._load_scenario(copy.deepcopy(initial_config.get(name, {})), long_context=False)  # noqa: SLF001
        instances.append(instance)
    return wrap_bfcl_methods_as_tools(instances)
