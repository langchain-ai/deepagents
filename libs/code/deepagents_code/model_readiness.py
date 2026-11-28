"""Decide whether dcode can start a session without asking the user anything.

A session can start when a saved default or recent model (or the launch
`--model`) resolves under `models.allowed` and its provider's credentials are
not known to be missing. Nothing here picks a model from the available keys; a
missing choice is reported, not guessed.

Shared by `dcode auth login` and by ACP session setup, which turns a problem
into an error that names what is missing. Login also confirms the model
constructs (`construction_problem`) before it reports success.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents_code.model_config import ModelConfig, ModelSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Ready:
    """A permitted model resolves and its provider has credentials."""

    model_spec: str
    """Canonical `provider:model` spec that a new session would use."""

    provider: str
    """Provider of `model_spec`."""


@dataclass(frozen=True, slots=True)
class UnverifiedProviderAuth:
    """A permitted model resolves, but its provider's auth cannot be checked.

    Some custom endpoints authenticate in ways only the provider SDK can
    confirm, so a session may start; the SDK reports a real failure.
    """

    model_spec: str
    """Canonical `provider:model` spec that resolved."""

    provider: str
    """Provider whose credentials are unknown."""


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
    """The model cannot start for a reason signing in cannot fix.

    Policy blocks it, no provider can be inferred, or constructing it fails.
    """

    message: str
    """The configuration error, suitable to show the user verbatim."""


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Launch flags that change how a session constructs its model."""

    model_params: dict[str, object] | None = None
    """`--model-params`: extra constructor kwargs."""

    profile_overrides: dict[str, object] | None = None
    """`--profile-override`: extra model profile fields."""

    max_retries: int | None = None
    """`--max-retries`."""


CanStart = Ready | UnverifiedProviderAuth
"""A new session would start."""

NotReady = NoModelSelected | MissingProviderKey | InvalidModelConfig
"""Why a new session cannot start yet."""

Readiness = CanStart | NotReady
"""Outcome of `check_readiness`."""


def check_readiness(*, model_override: str | None = None) -> Readiness:
    """Resolve the session model and check its policy and credentials.

    Args:
        model_override: Launch-time `--model`, which outranks saved models.

    Returns:
        Whether a session can start with the resolved model, or the first
            problem found.
    """
    from deepagents_code.model_config import ModelConfig

    spec = _resolve_spec(model_override)
    if not isinstance(spec, str):
        return spec
    parsed = resolve_permitted_spec(ModelConfig.load(), spec)
    if isinstance(parsed, str):
        return InvalidModelConfig(parsed)
    return _check_credentials(str(parsed), parsed.provider)


def readiness_problem(*, model_override: str | None = None) -> NotReady | None:
    """Return why a new session cannot start, or `None` when it can.

    Args:
        model_override: Launch-time `--model`, which outranks saved models.

    Returns:
        The problem to report, or `None` when a session can start.
    """
    result = check_readiness(model_override=model_override)
    return None if isinstance(result, CanStart) else result


def resolve_permitted_spec(config: ModelConfig, spec: str) -> ModelSpec | str:
    """Canonicalize `spec` and check it against `models.allowed`.

    Args:
        config: Loaded model config.
        spec: A `provider:model` spec or a bare model name.

    Returns:
        The canonical spec, or a user-facing reason it is not permitted.
    """
    from deepagents_code.model_config import ModelSpec

    canonical = config.canonical_model_spec(spec)
    parsed = ModelSpec.try_parse(canonical) if canonical else None
    if parsed is None:
        return f"Cannot infer a provider for model {spec!r}."
    blocked = config.policy_error(str(parsed))
    return str(blocked) if blocked is not None else parsed


def construction_problem(
    model_spec: str, provider: str, options: LaunchOptions
) -> MissingProviderKey | InvalidModelConfig | None:
    """Construct the model as a session would, without running inference.

    The only check that agrees with session start for every provider route:
    built-in, custom `class_path`, missing integration packages, and
    provider/model pairings `create_model` rejects. It runs against a snapshot
    of the environment, as the server's model check does, so stored keys are
    never copied into this process's environment. It imports the provider
    package, so call it off the event loop.

    Returns:
        `None` when the model constructs; `MissingProviderKey` when its
            credentials fail (a missing key, an expired ChatGPT session);
            otherwise `InvalidModelConfig`.
    """
    from deepagents_code.config import create_model, use_environment
    from deepagents_code.model_config import MissingCredentialsError, ModelConfigError

    try:
        with use_environment(dict(os.environ)):
            create_model(
                model_spec,
                extra_kwargs=dict(options.model_params or {}),
                profile_overrides=dict(options.profile_overrides or {}),
                cli_max_retries=options.max_retries,
            )
    except MissingCredentialsError:
        return MissingProviderKey(model_spec=model_spec, provider=provider)
    except ModelConfigError as exc:
        if _is_dcode_message(exc):
            return InvalidModelConfig(str(exc))
        return _opaque_failure(model_spec, exc.__cause__ or exc)
    except Exception as exc:  # noqa: BLE001  # provider constructors raise anything
        return _opaque_failure(model_spec, exc)
    return None


def _is_dcode_message(exc: Exception) -> bool:
    """Return whether `exc`'s text is dcode's own, with no provider error in it.

    `create_model` wraps provider exceptions as `ModelConfigError(f"...: {e}")`,
    and a provider's text can echo constructor input, keys included.
    """
    from deepagents_code.model_config import (
        MissingProviderPackageError,
        ModelConfigError,
        UnknownProviderError,
    )

    return (
        exc.__cause__ is None
        or isinstance(exc.__cause__, ModelConfigError)
        or isinstance(exc, MissingProviderPackageError | UnknownProviderError)
    )


def _opaque_failure(model_spec: str, exc: BaseException) -> InvalidModelConfig:
    """Report a provider failure by its type only; its text may hold a key.

    Returns:
        The config error to show.
    """
    logger.warning("Constructing %s failed: %s", model_spec, type(exc).__name__)
    return InvalidModelConfig(
        f"{model_spec} failed to construct ({type(exc).__name__}). "
        "Check its provider settings in config.toml."
    )


def provider_problem(provider: str) -> str | None:
    """Return why `provider` cannot run a session, short of its credentials.

    Returns:
        `None` when it is enabled, installed (or uses a custom `class_path`),
            and `models.allowed` permits some of its models; else the reason.
    """
    from deepagents_code.config_manifest import (
        is_provider_package_installed,
        provider_install_extra,
    )
    from deepagents_code.model_config import (
        CODEX_PROVIDER,
        PROVIDER_API_KEY_ENV,
        ModelConfig,
        ModelSpec,
    )

    config = ModelConfig.load()
    if provider not in {*PROVIDER_API_KEY_ENV, *config.providers, CODEX_PROVIDER}:
        return f"dcode has no sign-in for {provider}."
    if not config.is_provider_enabled(provider):
        return f"{provider} is disabled in config.toml."
    custom = bool(config.providers.get(provider, {}).get("class_path"))
    # ChatGPT runs on OpenAI's integration but has its own enabled setting.
    package_owner = "openai" if provider == CODEX_PROVIDER else provider
    if not custom and not is_provider_package_installed(package_owner):
        extra = provider_install_extra(package_owner) or provider
        return (
            f"The {provider} integration is not installed. "
            f"Install it with `dcode install {extra}`."
        )
    allowed = config.allowed_models
    if allowed is not None and not any(
        (parsed := ModelSpec.try_parse(entry)) and parsed.provider == provider
        for entry in allowed
    ):
        return f"models.allowed permits no {provider} models."
    return None


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
    """Map the provider's credential status to readiness.

    Returns:
        `Ready` when credentials are confirmed, `MissingProviderKey` when they
            are known to be missing, else `UnverifiedProviderAuth`.
    """
    from deepagents_code.model_config import has_provider_credentials

    has_key = has_provider_credentials(provider)
    if has_key is None:
        return UnverifiedProviderAuth(model_spec=model_spec, provider=provider)
    if has_key:
        return Ready(model_spec=model_spec, provider=provider)
    return MissingProviderKey(model_spec=model_spec, provider=provider)


def keyed_providers() -> frozenset[str]:
    """Return usable model providers with a key, env var, or ChatGPT session.

    Ambient or provider-managed auth does not count: those providers would
    always qualify, so they cannot tell that the user signed in.
    """
    from deepagents_code.model_config import (
        CODEX_PROVIDER,
        PROVIDER_API_KEY_ENV,
        ModelConfig,
        ProviderAuthState,
        get_provider_auth_status,
    )

    candidates = {*PROVIDER_API_KEY_ENV, *ModelConfig.load().providers, CODEX_PROVIDER}
    return frozenset(
        provider
        for provider in candidates
        if get_provider_auth_status(provider).state is ProviderAuthState.CONFIGURED
        and provider_problem(provider) is None
    )
