"""Validated hook configuration models."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from deepagents_code.hooks.models.domain import (  # noqa: TC001 - Pydantic runtime annotation.
    HookEvent,
)


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CommandHandlerSpec(_ConfigModel):
    """Configuration for a synchronous command hook."""

    type: Literal["command"]
    command: str
    timeout: float | None = None
    status_message: str | None = Field(default=None, alias="statusMessage")


HandlerSpec: TypeAlias = CommandHandlerSpec


class MatcherGroup(_ConfigModel):
    """A matcher and its ordered hook handlers."""

    matcher: str | None = None
    hooks: list[HandlerSpec]


class HooksConfig(_ConfigModel):
    """Top-level configuration grouped by hook event."""

    hooks: dict[HookEvent, list[MatcherGroup]]
