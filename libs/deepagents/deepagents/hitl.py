"""Types for human-in-the-loop configuration."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict, cast

from langchain.agents.middleware import InterruptOnConfig, ToolCallRequest

HITLDecision = Literal["approve", "edit", "reject", "respond"]
"""Decision types available for human-in-the-loop review."""


class DeepAgentInterruptOnConfig(TypedDict, total=False):
    """Deep Agents interrupt configuration for a tool."""

    enabled: bool
    """Whether interrupts are enabled for the tool."""

    allowed_decisions: list[HITLDecision]
    """The decisions that are allowed for the tool."""

    description: str | Callable[..., str]
    """Description or factory used for the human review request."""

    args_schema: dict[str, Any]
    """JSON schema for editable tool arguments."""

    when: Callable[[ToolCallRequest], bool]
    """Predicate controlling whether an individual tool call interrupts."""


InterruptOnConfigValue = bool | InterruptOnConfig | DeepAgentInterruptOnConfig
"""A Deep Agents `interrupt_on` value for one tool."""

_DEFAULT_INTERRUPT_ALLOWED_DECISIONS: tuple[HITLDecision, ...] = ("approve", "edit", "reject", "respond")


def _normalize_interrupt_on_config(tool_config: InterruptOnConfigValue) -> bool | InterruptOnConfig:
    """Normalize Deep Agents HITL config extensions before delegating to LangChain."""
    if isinstance(tool_config, bool):
        return tool_config

    config = dict(tool_config)
    if "enabled" not in config:
        return cast("InterruptOnConfig", config)

    enabled = config.pop("enabled")
    if not isinstance(enabled, bool):
        msg = "`enabled` in `interrupt_on` config must be a bool."
        raise TypeError(msg)
    if not enabled:
        return False
    if "allowed_decisions" not in config:
        config["allowed_decisions"] = list(_DEFAULT_INTERRUPT_ALLOWED_DECISIONS)
    return cast("InterruptOnConfig", config)


def _normalize_interrupt_on(
    interrupt_on: Mapping[str, InterruptOnConfigValue],
) -> dict[str, bool | InterruptOnConfig]:
    """Normalize all tool interrupt configs."""
    return {tool_name: _normalize_interrupt_on_config(tool_config) for tool_name, tool_config in interrupt_on.items()}


__all__ = ["DeepAgentInterruptOnConfig", "HITLDecision", "InterruptOnConfigValue"]
