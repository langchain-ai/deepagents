"""Pydantic request/response schemas for the OBS runner API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SceneRequest(BaseModel):
    """Direct OBS scene switch request."""

    name: str = Field(min_length=1)


class TextRequest(BaseModel):
    """Direct OBS text-source update."""

    source: str = Field(min_length=1)
    value: str = ""


class HealthResponse(BaseModel):
    """OBS connectivity reported by `GET /healthz`."""

    obs_connected: bool
