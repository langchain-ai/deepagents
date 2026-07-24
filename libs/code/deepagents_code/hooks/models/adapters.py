"""Cached runtime validators for hook model boundaries."""

from pydantic import TypeAdapter

from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import (
    HookDecision,
    HookDomainEvent,
    HookInvocation,
)
from deepagents_code.hooks.models.transport import (
    HookInvocationRequest,
    HookInvocationResponse,
)
from deepagents_code.hooks.models.wire import HookWireInput, HookWireOutput

HOOK_DOMAIN_EVENT_ADAPTER = TypeAdapter(HookDomainEvent)
HOOK_INVOCATION_ADAPTER = TypeAdapter(HookInvocation)
HOOK_DECISION_ADAPTER = TypeAdapter(HookDecision)
HOOK_WIRE_INPUT_ADAPTER = TypeAdapter(HookWireInput)
HOOK_WIRE_OUTPUT_ADAPTER = TypeAdapter(HookWireOutput)
HOOKS_CONFIG_ADAPTER = TypeAdapter(HooksConfig)
HOOK_INVOCATION_REQUEST_ADAPTER = TypeAdapter(HookInvocationRequest)
HOOK_INVOCATION_RESPONSE_ADAPTER = TypeAdapter(HookInvocationResponse)
