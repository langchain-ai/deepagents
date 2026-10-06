"""Provider-independent model metadata exchanged with the server."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModelPurpose = Literal["main", "auxiliary"]


@dataclass(frozen=True)
class ModelMetadata:
    """Resolved model properties needed by the interactive client."""

    model_name: str
    provider: str
    context_limit: int | None = None
    unsupported_modalities: frozenset[str] = frozenset()
    structured_output: bool | None = field(default=None, kw_only=True)

    @classmethod
    def from_payload(cls, payload: object) -> ModelMetadata:
        """Validate server metadata before it can replace the active model.

        Returns:
            Provider-independent model properties.

        Raises:
            TypeError: If the server response is malformed.
        """
        if not isinstance(payload, dict):
            msg = "Model metadata must be an object."
            raise TypeError(msg)
        name = payload.get("model_name")
        provider = payload.get("provider")
        limit = payload.get("context_limit")
        modalities = payload.get("unsupported_modalities")
        structured_output = payload.get("structured_output")
        if not isinstance(name, str) or not name or not isinstance(provider, str):
            msg = "Model metadata has an invalid model name or provider."
            raise TypeError(msg)
        if limit is not None and (type(limit) is not int or limit <= 0):
            msg = "Model metadata has an invalid context limit."
            raise TypeError(msg)
        if not isinstance(modalities, list):
            msg = "Model metadata has invalid unsupported modalities."
            raise TypeError(msg)
        if structured_output is not None and not isinstance(structured_output, bool):
            msg = "Model metadata has invalid structured output support."
            raise TypeError(msg)
        validated: set[str] = set()
        for item in modalities:
            if not isinstance(item, str):
                msg = "Model metadata has invalid unsupported modalities."
                raise TypeError(msg)
            validated.add(item)
        return cls(
            name,
            provider,
            limit,
            frozenset(validated),
            structured_output=structured_output,
        )

    def to_payload(self) -> dict[str, object]:
        """Return only JSON-safe properties, never a model or its credentials."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "context_limit": self.context_limit,
            "unsupported_modalities": sorted(self.unsupported_modalities),
            "structured_output": self.structured_output,
        }

    def apply_to_runtime_state(self) -> None:
        """Commit validated metadata to the UI's runtime state."""
        from deepagents_code.config import runtime_state

        runtime_state.model_name = self.model_name
        runtime_state.model_provider = self.provider
        runtime_state.model_context_limit = self.context_limit
        runtime_state.model_unsupported_modalities = self.unsupported_modalities
