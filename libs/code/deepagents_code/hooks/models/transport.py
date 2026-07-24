"""Versioned hook invocation transport models."""

from __future__ import annotations

from datetime import (  # noqa: TC003 - Pydantic resolves model annotations at runtime.
    datetime,
)
from typing import Literal
from uuid import UUID  # noqa: TC003 - Pydantic resolves model annotations at runtime.

from pydantic import BaseModel, ConfigDict

from deepagents_code.hooks.models.domain import (  # noqa: TC001 - Pydantic runtime annotation.
    HookDecision,
    HookInvocation,
)


class _TransportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HookInvocationRequest(_TransportModel):
    """Request sent for a server-owned hook invocation."""

    protocol_version: Literal[1]
    invocation_id: UUID
    snapshot_id: str
    run_id: str
    invocation: HookInvocation
    deadline: datetime


class HookInvocationResponse(_TransportModel):
    """Response returned for a server-owned hook invocation."""

    protocol_version: Literal[1]
    invocation_id: UUID
    snapshot_id: str
    decision: HookDecision
