"""Pydantic request/response schemas for the control API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from obs_runner.state_machine import Event, Phase


class TransitionRequest(BaseModel):
    """Single entry point for state transitions.

    The API intentionally mirrors the state machine's shape: one event,
    one payload. Payload schema is per-event, validated downstream.
    """

    event: Event
    payload: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Current machine snapshot returned by `GET /state` and transitions."""

    phase: Phase
    round_num: int | None = None
    prompt: str | None = None
    contestants: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """OBS + FSM connectivity reported by `GET /healthz`."""

    obs_connected: bool
    phase: Phase
