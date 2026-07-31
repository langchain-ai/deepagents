"""Validated hook configuration models."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deepagents_code.hooks.models.domain import (  # ruff:ignore[typing-only-first-party-import] - Pydantic runtime annotation.
    HookEvent,
)


class _ConfigModel(BaseModel):
    # Ignore unknown keys so newer external handler fields do not fail config load.
    # Known-but-unsupported fields such as `async` are modeled explicitly and rejected.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class CommandHandlerSpec(_ConfigModel):
    """Configuration for a synchronous command hook.

    Currently only `type: "command"` is supported. Additional handler types
    remain a discriminated-union extension point and are rejected until
    implemented.

    When `argv` is set, the runner launches via `create_subprocess_exec` and
    ignores shell metacharacters in `command`.

    `argv` is a temporary legacy-migration compatibility field. Remove it with
    `hooks.legacy` and `hooks.migration` after September 1, 2026.
    """

    type: Literal["command"]
    command: str
    argv: list[str] | None = None
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    status_message: str | None = Field(default=None, alias="statusMessage")
    async_: bool | None = Field(default=None, alias="async")

    @field_validator("argv", mode="after")
    @classmethod
    def _normalize_argv(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value or not all(isinstance(part, str) for part in value):
            msg = "argv must be a non-empty list of strings when provided."
            raise ValueError(msg)
        if not value[0].strip():
            msg = "argv[0] must be a non-empty executable path."
            raise ValueError(msg)
        return value

    @field_validator("async_", mode="after")
    @classmethod
    def _normalize_async(cls, value: bool | None) -> None:
        if value:
            msg = "async command hooks are not yet supported."
            raise ValueError(msg)


HandlerSpec: TypeAlias = CommandHandlerSpec


class MatcherGroup(_ConfigModel):
    """A matcher and its ordered hook handlers."""

    matcher: str | None = None
    hooks: list[HandlerSpec]


class HooksConfig(_ConfigModel):
    """Top-level configuration grouped by hook event."""

    hooks: dict[HookEvent, list[MatcherGroup]]
