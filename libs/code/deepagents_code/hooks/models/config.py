"""Validated hook configuration models."""

from __future__ import annotations

from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from deepagents_code.hooks.models.domain import (  # ruff:ignore[typing-only-first-party-import] - Pydantic runtime annotation.
    HookEvent,
)


class _ConfigModel(BaseModel):
    # Ignore unknown keys so newer external handler fields do not fail config load.
    # Known-but-unsupported fields such as `async` are modeled explicitly and rejected.
    model_config = ConfigDict(extra="ignore")


class CommandHandlerSpec(_ConfigModel):
    """Configuration for a synchronous command hook.

    Currently only `type: "command"` is supported. Additional handler types
    remain a discriminated-union extension point and are rejected until
    implemented.
    """

    type: Literal["command"]
    command: str
    timeout: float | None = None
    status_message: str | None = Field(default=None, alias="statusMessage")
    async_: bool | None = Field(default=None, alias="async")

    @model_validator(mode="after")
    def _reject_async_commands(self) -> Self:
        if self.async_:
            msg = "async command hooks are not supported in MVP"
            raise ValueError(msg)
        # Normalize explicit `async: false` to omitted so equivalent configs
        # share a snapshot hash.
        if self.async_ is False:
            return self.model_copy(update={"async_": None})
        return self


HandlerSpec: TypeAlias = CommandHandlerSpec


class MatcherGroup(_ConfigModel):
    """A matcher and its ordered hook handlers."""

    matcher: str | None = None
    hooks: list[HandlerSpec]


class HooksConfig(_ConfigModel):
    """Top-level configuration grouped by hook event."""

    hooks: dict[HookEvent, list[MatcherGroup]]
