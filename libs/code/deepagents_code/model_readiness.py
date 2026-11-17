"""Decide whether dcode can start a session without asking the user anything.

Ready means two things: a saved default or recent model resolves under
`models.allowed`, and that model's provider has credentials. Nothing here picks
a model from the available keys; a missing choice is reported, not guessed.

Shared by `dcode auth login`, which exits `0` only when ready, and by ACP
session setup, which turns a problem into an error that names what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ready:
    """A permitted model resolves and its provider has credentials."""

    model_spec: str
    """Canonical `provider:model` spec that a new session would use."""

    provider: str
    """Provider of `model_spec`."""


@dataclass(frozen=True, slots=True)
class NoModelSelected:
    """Neither a permitted default nor a permitted recent model is saved."""

    @property
    def message(self) -> str:
        """Short, user-facing description of what is missing."""
        return "no model selected"


@dataclass(frozen=True, slots=True)
class MissingProviderKey:
    """A permitted model resolves, but its provider has no credentials."""

    model_spec: str
    """Canonical `provider:model` spec that resolved."""

    provider: str
    """Provider that needs a key."""

    @property
    def message(self) -> str:
        """Short, user-facing description of what is missing."""
        return f"no key for {self.provider}"


@dataclass(frozen=True, slots=True)
class InvalidModelConfig:
    """Configuration blocks every model, or the saved model names no provider.

    Signing in cannot fix this, so callers should report it as a configuration
    error rather than an authentication requirement.
    """

    message: str
    """The configuration error, suitable to show the user verbatim."""


@dataclass(frozen=True, slots=True)
class UnverifiedProviderAuth:
    """A permitted model resolves, but its provider's auth cannot be checked.

    Distinct from `MissingProviderKey`: some custom endpoints authenticate in
    ways only the provider SDK can confirm, so a caller may let them proceed.
    """

    model_spec: str
    """Canonical `provider:model` spec that resolved."""

    provider: str
    """Provider whose credentials are unknown."""

    @property
    def message(self) -> str:
        """Short, user-facing description of what is missing."""
        return f"cannot verify credentials for {self.provider}"


NotReady = (
    NoModelSelected | MissingProviderKey | UnverifiedProviderAuth | InvalidModelConfig
)
"""Why a new session cannot start yet."""

Readiness = Ready | NotReady
"""Outcome of `check_readiness`."""


def check_readiness(*, model_override: str | None = None) -> Readiness:
    """Resolve the session model and check its policy and credentials.

    Args:
        model_override: Launch-time `--model`, which outranks saved models.

    Returns:
        `Ready` with the resolved model, or the first problem found.
    """
    from deepagents_code.model_config import ModelConfig, ModelSpec

    spec = _resolve_spec(model_override)
    if not isinstance(spec, str):
        return spec
    config = ModelConfig.load()
    canonical = config.canonical_model_spec(spec)
    parsed = ModelSpec.try_parse(canonical) if canonical else None
    if parsed is None:
        return InvalidModelConfig(f"Cannot infer a provider for model {spec!r}.")
    blocked = config.policy_error(str(parsed))
    if blocked is not None:
        return InvalidModelConfig(str(blocked))
    return _check_credentials(str(parsed), parsed.provider)


def _resolve_spec(model_override: str | None) -> str | NotReady:
    """Return the override, else the saved default or recent model.

    Returns:
        The model spec to check, or why none resolves.
    """
    from deepagents_code.config import _get_default_model_spec
    from deepagents_code.model_config import (
        ModelNotAllowedError,
        NoModelConfiguredError,
    )

    if model_override and model_override.strip():
        return model_override.strip()
    try:
        return _get_default_model_spec()
    except NoModelConfiguredError:
        return NoModelSelected()
    except ModelNotAllowedError as exc:
        return InvalidModelConfig(str(exc))


def _check_credentials(model_spec: str, provider: str) -> Readiness:
    """Map the provider's auth status to readiness.

    Returns:
        `Ready` only for an affirmative status; otherwise the problem.
    """
    from deepagents_code.model_config import (
        ProviderAuthState,
        get_provider_auth_status,
    )

    state = get_provider_auth_status(provider).state
    if state is ProviderAuthState.MISSING:
        return MissingProviderKey(model_spec=model_spec, provider=provider)
    if state in {
        ProviderAuthState.CONFIGURED,
        ProviderAuthState.IMPLICIT,
        ProviderAuthState.MANAGED,
        ProviderAuthState.NOT_REQUIRED,
    }:
        return Ready(model_spec=model_spec, provider=provider)
    return UnverifiedProviderAuth(model_spec=model_spec, provider=provider)


def readiness_problem(*, model_override: str | None = None) -> NotReady | None:
    """Return why a new session cannot start, or `None` when it can.

    Args:
        model_override: Launch-time `--model`, which outranks saved models.

    Returns:
        The problem to report, or `None` when ready.
    """
    result = check_readiness(model_override=model_override)
    return None if isinstance(result, Ready) else result
