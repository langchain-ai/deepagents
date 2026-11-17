"""Standalone sign-in app behind `dcode auth login`.

ACP clients (T3 Code, Zed) sign a user in by running the agent command with
`auth login <provider>` appended in a terminal, then wait for exit `0`. This
app runs that flow without the full dcode TUI or its server:

1. Key step: the `/auth` manager, listing installed model providers only. The
   first saved credential moves on to the next step.
2. Model step: when no permitted default or recent model is saved, the model
   selector, limited to the provider just signed in and to `models.allowed`.

The app exits with a `LoginOutcome`; the caller maps it to an exit code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from textual.app import App
from textual.content import Content
from textual.theme import Theme

from deepagents_code import theme
from deepagents_code.model_readiness import (
    NoModelSelected,
    Readiness,
    Ready,
    check_readiness,
    has_credentials,
)
from deepagents_code.tui.widgets.auth import (
    AuthManagerScreen,
    provider_display_name,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from textual.screen import ModalScreen

    from deepagents_code.model_catalog import ModelCatalog


@dataclass(frozen=True, slots=True)
class LoginCancelled:
    """The user closed a step without completing it."""

    step: Literal["key", "model"]
    """Step the user left."""

    @property
    def message(self) -> str:
        """Short, user-facing description of the cancellation."""
        if self.step == "key":
            return "sign-in cancelled before a key was saved"
        return "model selection cancelled"


@dataclass(frozen=True, slots=True)
class ModelSaveFailed:
    """The chosen model could not be saved."""

    message: str
    """Why the save failed, suitable to show the user verbatim."""


LoginOutcome = Readiness | LoginCancelled | ModelSaveFailed
"""Final state of a login run: readiness after both steps, or why it stopped."""


@dataclass(frozen=True, slots=True)
class ModelStep:
    """Go straight to the model step for a provider that already has a key."""

    provider: str


def skip_key_step(
    readiness: Readiness,
    provider: str | None,
    provider_has_credentials: Callable[[str], bool],
) -> Ready | ModelStep | None:
    """Decide whether a sign-in can bypass the key step.

    Credentials can already exist (an env var, a stored key, a ChatGPT
    session) while only the model is missing, which is the common ACP case.

    Args:
        readiness: Current readiness.
        provider: Provider the client asked to sign in to, if any.
        provider_has_credentials: Whether a provider's auth is affirmative.

    Returns:
        `Ready` to finish, `ModelStep` to pick a model, or `None` when a key
            is still needed.
    """
    if isinstance(readiness, Ready) and (
        provider is None or provider_has_credentials(provider)
    ):
        return readiness
    if (
        isinstance(readiness, NoModelSelected)
        and provider is not None
        and provider_has_credentials(provider)
    ):
        return ModelStep(provider)
    return None


class AuthLoginApp(App[LoginOutcome]):
    """Two-step sign-in: save a provider key, then pick a model if needed."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        readiness: Callable[[], Readiness] = check_readiness,
        provider_has_credentials: Callable[[str], bool] = has_credentials,
    ) -> None:
        """Initialize the app.

        Args:
            provider: Provider to sign in to; highlighted first in the key step.
            readiness: Readiness check used to skip steps and to compute the
                final outcome.
            provider_has_credentials: Whether a provider's auth is affirmative,
                used to skip the key step when a key already exists.
        """
        super().__init__()
        self._provider = provider
        self._readiness = readiness
        self._has_credentials = provider_has_credentials
        self._manager = AuthManagerScreen(initial_provider=provider, sign_in_only=True)
        self._signed_in: str | None = None
        _register_default_theme(self)

    @property
    def signed_in(self) -> str | None:
        """Provider whose credential the key step saved, if any."""
        return self._signed_in

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Add dcode's app-specific CSS variables to Textual's defaults.

        Returns:
            Dict of CSS variable names to color values.
        """
        defaults = super().get_theme_variable_defaults()
        colors = theme.get_theme_colors(self)
        return {**defaults, **theme.get_css_variable_defaults(colors=colors)}

    @staticmethod
    async def _reload_settings_from_environment_serialized() -> list[str]:
        """Reload `.env` and environment settings for the key prompt's Ctrl+R.

        The main app serializes this against other environment mutations; this
        app has none, so a plain off-thread reload is enough.

        Returns:
            Human-readable descriptions of settings changes.
        """
        from deepagents_code.config import credentials

        return await asyncio.to_thread(credentials.reload_from_environment)

    def on_mount(self) -> None:
        """Finish, open the model step, or open the key step, as needed."""
        step = skip_key_step(self._readiness(), self._provider, self._has_credentials)
        if step is None:
            self.push_screen(self._manager, self._on_key_step_closed)
            return
        self._continue(step)

    def _continue(self, step: Ready | ModelStep) -> None:
        """Exit when ready, or open the model step for `step.provider`."""
        if isinstance(step, Ready):
            self.exit(step)
            return
        self.push_screen(_build_model_selector(step.provider), self._on_model_chosen)

    def on_auth_manager_screen_credential_saved(
        self, event: AuthManagerScreen.CredentialSaved
    ) -> None:
        """Close the key step after the first saved credential."""
        event.stop()
        if self._signed_in is not None:
            return
        self._signed_in = event.provider
        self._manager.dismiss(None)

    def _on_key_step_closed(self, _result: None) -> None:
        """Continue after a save; without one, continue only if nothing is missing."""
        if self._signed_in is not None:
            self.call_after_refresh(self._start_model_step, self._signed_in)
            return
        # Closing without a save is not a cancel when a key already existed.
        readiness = self._readiness()
        step = (
            readiness
            if isinstance(readiness, Ready)
            else skip_key_step(readiness, self._provider, self._has_credentials)
        )
        if step is None:
            self.exit(LoginCancelled("key"))
            return
        self.call_after_refresh(self._continue, step)

    def _start_model_step(self, provider: str) -> None:
        """Ask for a model only when no permitted one is saved yet."""
        readiness = self._readiness()
        if not isinstance(readiness, NoModelSelected):
            self.exit(readiness)
            return
        self.push_screen(_build_model_selector(provider), self._on_model_chosen)

    def _on_model_chosen(self, result: tuple[str, str] | None) -> None:
        """Save the chosen model, then exit with the final readiness."""
        if result is None:
            self.exit(LoginCancelled("model"))
            return
        failure = save_model_choice(result[0])
        self.exit(ModelSaveFailed(failure) if failure else self._readiness())


def _build_model_selector(provider: str) -> ModalScreen[tuple[str, str] | None]:
    """Return the model selector limited to `provider`'s permitted models."""
    from deepagents_code.tui.widgets.model_selector import ModelSelectorScreen

    description = Content.from_markup(
        "Signed in to $name. New sessions use this model; change it with /model.",
        name=provider_display_name(provider),
    )
    return ModelSelectorScreen(
        title="Choose a model",
        description=description,
        include_recent_models=False,
        default_scope=None,
        catalog_loader=partial(_load_provider_catalog, provider),
        validate_selection=partial(selection_problem, provider),
    )


async def _load_provider_catalog(
    provider: str, recommended: Sequence[str], current_spec: str | None
) -> ModelCatalog:
    """Load the model catalog off-thread and limit it to `provider`.

    Returns:
        The catalog with only `provider`'s models allowed by `models.allowed`.
    """
    from deepagents_code.model_catalog import load_model_catalog

    catalog = await asyncio.to_thread(
        load_model_catalog,
        recommended_models=recommended,
        current_spec=current_spec,
    )
    return limit_catalog(catalog, provider)


def limit_catalog(catalog: ModelCatalog, provider: str) -> ModelCatalog:
    """Keep only `provider`'s models that `models.allowed` permits.

    Dropping every other provider entry also stops the selector from offering
    to install a package, which a running `uvx` install would refuse anyway.

    Returns:
        A copy of `catalog` limited to `provider`.
    """
    policy = catalog.presentation_config()
    models = [
        spec
        for spec in catalog.models
        if spec.partition(":")[0] == provider and policy.is_model_allowed(spec)
    ]
    return catalog.model_copy(
        update={
            "models": models,
            "profiles": {s: p for s, p in catalog.profiles.items() if s in models},
            "providers": {
                name: info
                for name, info in catalog.providers.items()
                if name == provider
            },
        }
    )


def selection_problem(provider: str, model_spec: str) -> str | None:
    """Reject a typed spec outside the signed-in provider or `models.allowed`.

    The selector accepts free-form `provider:model` text, which bypasses the
    provider-limited catalog, so the login flow checks the final choice.

    Returns:
        `None` when the spec may be saved, else a footer message.
    """
    from deepagents_code.model_config import ModelConfig, ModelSpec

    config = ModelConfig.load()
    canonical = config.canonical_model_spec(model_spec)
    parsed = ModelSpec.try_parse(canonical) if canonical else None
    if parsed is None or parsed.provider != provider:
        return f"Choose a model from {provider_display_name(provider)}."
    blocked = config.policy_error(str(parsed))
    return str(blocked) if blocked is not None else None


def save_model_choice(model_spec: str) -> str | None:
    """Persist the model chosen in the model step.

    Writes `[models].default` unless an explicit default is already set (even
    one `models.allowed` now blocks); then it writes `[models].recent`, which
    default resolution also reads, so a sign-in never replaces a default.

    Returns:
        `None` on success, or a user-facing reason the save failed.
    """
    from deepagents_code.model_config import (
        ModelConfig,
        ModelNotAllowedError,
        save_default_model,
        save_recent_model,
    )

    save = save_recent_model if ModelConfig.load().default_model else save_default_model
    try:
        saved = save(model_spec)
    except ModelNotAllowedError as exc:
        return str(exc)
    return None if saved else f"could not save {model_spec} to config.toml"


def _register_default_theme(app: App[LoginOutcome]) -> None:
    """Register and select dcode's default theme so modals match the main TUI."""
    entry = theme.get_registry()[theme.DEFAULT_THEME]
    c = entry.colors
    app.register_theme(
        Theme(
            name=theme.DEFAULT_THEME,
            primary=c.primary,
            secondary=c.secondary,
            accent=c.accent,
            foreground=c.foreground,
            background=c.background,
            surface=c.surface,
            panel=c.panel,
            warning=c.warning,
            error=c.error,
            success=c.success,
            dark=entry.dark,
        )
    )
    app.theme = theme.DEFAULT_THEME


def run_login(
    provider: str | None, *, model_override: str | None = None
) -> LoginOutcome:
    """Run the sign-in app in the current terminal.

    Args:
        provider: Provider row to highlight first in the key step.
        model_override: Launch-time `--model`; when set, readiness checks it
            instead of saved models and the model step never runs.

    Returns:
        The final outcome; an app quit by other means counts as a cancel.
    """
    readiness = partial(check_readiness, model_override=model_override)
    app = AuthLoginApp(provider=provider, readiness=readiness)
    result = app.run()
    if result is not None:
        return result
    return LoginCancelled("model" if app.signed_in else "key")
