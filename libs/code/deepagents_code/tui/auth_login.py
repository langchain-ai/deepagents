"""Standalone sign-in app behind `dcode auth login`.

ACP clients (T3 Code, Zed) sign a user in by running the agent command with
`auth login <provider>` appended in a terminal, then wait for exit `0`. This
app runs that flow without the full dcode TUI or its server.

Every transition comes from `plan_step`, which reads readiness and the target
provider:

1. Key step: the `/auth` manager, listing installed model providers only, with
   the target provider highlighted. It stays open until the target has a key.
2. Model step: the model selector, limited to the target provider (or, with no
   target, to usable providers that already have a key) and to
   `models.allowed`. A choice is constructed before it is saved as the
   default: picking a provider in the client is an explicit choice to use it.

A launch `--model` fixes the session model, so the flow only ever asks for
that model's key. The run ends successfully only once the session model
constructs, as `session/new` would.
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
    CanStart,
    InvalidModelConfig,
    LaunchOptions,
    MissingProviderKey,
    Readiness,
    check_readiness,
    construction_problem,
    keyed_providers,
    provider_problem,
    resolve_permitted_spec,
)
from deepagents_code.tui.widgets.auth import (
    AuthManagerScreen,
    provider_display_name,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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


LoginOutcome = Readiness | LoginCancelled
"""Final state of a login run: readiness after the last step, or where it stopped."""


@dataclass(frozen=True, slots=True)
class KeyStep:
    """Ask for a key, highlighting `provider` when known."""

    provider: str | None

    after_failure: bool = False
    """The provider's credentials failed when the model was built, so closing
    without a save cancels rather than retrying the same credentials."""


@dataclass(frozen=True, slots=True)
class ModelStep:
    """Pick a model from `providers`, all of which have credentials."""

    providers: frozenset[str]


Step = KeyStep | ModelStep | Readiness
"""What the app does next; a readiness value ends the run with it."""


def plan_step(
    readiness: Readiness,
    *,
    provider: str | None,
    fixed_model: bool,
    has_key: Callable[[str], bool],
    unavailable: Callable[[str], str | None],
    keyed: Callable[[], frozenset[str]],
) -> Step:
    """Decide the next step from readiness and the provider being signed in to.

    Args:
        readiness: Current readiness.
        provider: Provider the client asked for or, without one, whose key
            was just saved.
        fixed_model: Whether a launch `--model` fixes the session model.
        has_key: Whether a provider's credentials are confirmed.
        unavailable: Why a provider cannot run a session, short of its key.
        keyed: Usable providers that already have a key.

    Returns:
        The next step, or the readiness to finish with.
    """
    if isinstance(readiness, InvalidModelConfig):
        return readiness
    if isinstance(readiness, CanStart) and (
        fixed_model or provider in {None, readiness.provider}
    ):
        return readiness
    missing = readiness.provider if isinstance(readiness, MissingProviderKey) else None
    if fixed_model:
        return KeyStep(missing)
    if provider is not None:
        problem = unavailable(provider)
        if problem is not None:
            return InvalidModelConfig(problem)
        return (
            ModelStep(frozenset({provider})) if has_key(provider) else KeyStep(provider)
        )
    found = keyed()
    return ModelStep(found) if found else KeyStep(missing)


class AuthLoginApp(App[LoginOutcome]):
    """Sign-in: save a provider key and pick a model, as readiness requires."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        readiness: Callable[[], Readiness] = check_readiness,
        fixed_model: bool = False,
        options: LaunchOptions | None = None,
    ) -> None:
        """Initialize the app.

        Args:
            provider: Provider the client asked to sign in to.
            readiness: Readiness check used to plan each step and to compute
                the final outcome.
            fixed_model: Whether a launch `--model` fixes the session model.
            options: Launch flags the session model is constructed with.
        """
        super().__init__()
        self._provider = provider
        self._readiness = readiness
        self._fixed_model = fixed_model
        self._options = options or LaunchOptions()
        self._manager: AuthManagerScreen | None = None
        self._picker: ModalScreen[tuple[str, str] | None] | None = None
        self._next: Step | None = None
        # Read by `run_login`, so a quit by other means reports the step on screen.
        self.active_step: Literal["key", "model"] = "key"
        _register_default_theme(self)

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

    def _fresh_readiness(self) -> Readiness:
        """Check readiness without cached config or credentials.

        Returns:
            Current readiness.
        """
        from deepagents_code.model_config import clear_caches

        clear_caches()
        return self._readiness()

    def _plan(self, provider: str | None) -> Step:
        """Plan the next step for `provider` from fresh readiness.

        Returns:
            The next step, or the readiness to finish with.
        """
        from deepagents_code.model_config import has_provider_credentials

        return plan_step(
            self._fresh_readiness(),
            provider=provider,
            fixed_model=self._fixed_model,
            has_key=lambda name: has_provider_credentials(name) is True,
            unavailable=provider_problem,
            keyed=keyed_providers,
        )

    def on_mount(self) -> None:
        """Run the first step."""
        self._run(self._plan(self._provider))

    def _run(self, step: Step) -> None:
        """Open the key or model step, or exit with readiness."""
        if isinstance(step, KeyStep):
            self.active_step = "key"
            self._manager = AuthManagerScreen(
                initial_provider=step.provider, sign_in_only=True
            )
            self.push_screen(self._manager, partial(self._on_key_step_closed, step))
        elif isinstance(step, ModelStep):
            self.active_step = "model"
            self._picker = _build_model_selector(
                step.providers, partial(self._accept, step.providers)
            )
            self.push_screen(self._picker, self._on_model_chosen)
        else:
            self.run_worker(self._finish(step), exclusive=True)

    async def _finish(self, readiness: Readiness, *, built: str | None = None) -> None:
        """Exit with readiness, once a startable model also constructs.

        Args:
            readiness: Readiness to finish with.
            built: A spec the model step already constructed, not rebuilt.
        """
        problem = None
        if isinstance(readiness, CanStart) and readiness.model_spec != built:
            problem = await asyncio.to_thread(
                construction_problem,
                readiness.model_spec,
                readiness.provider,
                self._options,
            )
        if isinstance(problem, MissingProviderKey):
            # The key or ChatGPT session failed in use; sign in to it again.
            self._run(KeyStep(problem.provider, after_failure=True))
            return
        self.exit(problem or readiness)

    async def _accept(self, providers: frozenset[str], model_spec: str) -> str | None:
        """Accept a picked model; on a credential failure, go sign in again.

        Returns:
            `None` once saved, else a footer message.
        """
        result = await accept_model_choice(providers, self._options, model_spec)
        if not isinstance(result, MissingProviderKey):
            return result
        self._next = KeyStep(result.provider, after_failure=True)
        if self._picker is not None:
            self.call_later(self._picker.dismiss, None)
        return f"Sign in to {provider_display_name(result.provider)} again."

    def on_auth_manager_screen_credential_saved(
        self, event: AuthManagerScreen.CredentialSaved
    ) -> None:
        """Leave the key step once a save makes progress; else keep it open."""
        event.stop()
        # An explicit target stays the target; another provider's key is not it.
        step = self._plan(self._provider or event.provider)
        if isinstance(step, KeyStep) or self._manager is None:
            return
        self._next = step
        self._manager.dismiss(None)

    def _on_key_step_closed(self, key_step: KeyStep, _result: None) -> None:
        """Continue after a save, or after a close that left dcode able to start."""
        step, self._next = self._next, None
        if step is None and key_step.after_failure:
            step = LoginCancelled("key")
        if step is None:
            readiness = self._fresh_readiness()
            step = readiness if isinstance(readiness, CanStart) else None
        if step is None:
            step = self._plan(self._provider)
            if isinstance(step, KeyStep):
                step = LoginCancelled("key")
        self.call_after_refresh(self._finish_or_run, step)

    def _finish_or_run(self, step: Step | LoginCancelled) -> None:
        """Exit with a cancellation, or run the step."""
        if isinstance(step, LoginCancelled):
            self.exit(step)
            return
        self._run(step)

    def _on_model_chosen(self, result: tuple[str, str] | None) -> None:
        """Exit with the final readiness once the selector saved a choice."""
        from deepagents_code.model_config import ModelConfig

        if result is None:
            step, self._next = self._next, None
            if step is None:
                self.exit(LoginCancelled("model"))
            else:
                self.call_after_refresh(self._run, step)
            return
        built = resolve_permitted_spec(ModelConfig.load(), result[0])
        self.run_worker(
            self._finish(self._fresh_readiness(), built=str(built)), exclusive=True
        )


def _build_model_selector(
    providers: frozenset[str], accept: Callable[[str], Awaitable[str | None]]
) -> ModalScreen[tuple[str, str] | None]:
    """Return the model selector limited to `providers`' permitted models."""
    from deepagents_code.tui.widgets.model_selector import ModelSelectorScreen

    if len(providers) == 1:
        provider = next(iter(providers))
        # Discovery can list nothing a permitted ID would match, so always say
        # that typing works rather than guessing whether the list is empty.
        description = Content.from_markup(
            "Signed in to $name. Pick a model, or type one as $provider:<model>. "
            "New sessions use it; change it with /model.",
            name=provider_display_name(provider),
            provider=provider,
        )
    else:
        description = Content(
            "Choose a model from a provider you have a key for. "
            "New sessions use it; change it with /model."
        )
    return ModelSelectorScreen(
        title="Choose a model",
        description=description,
        include_recent_models=False,
        default_scope=None,
        catalog_loader=partial(_load_catalog_for, providers),
        accept_selection=accept,
    )


async def _load_catalog_for(
    providers: frozenset[str], recommended: Sequence[str], current_spec: str | None
) -> ModelCatalog:
    """Load the model catalog off-thread and limit it to `providers`.

    Returns:
        The catalog with only `providers`' models allowed by `models.allowed`.
    """
    from deepagents_code.model_catalog import load_model_catalog

    catalog = await asyncio.to_thread(
        load_model_catalog,
        recommended_models=recommended,
        current_spec=current_spec,
    )
    return limit_catalog(catalog, providers)


def limit_catalog(catalog: ModelCatalog, providers: frozenset[str]) -> ModelCatalog:
    """Keep only `providers`' models that `models.allowed` permits.

    Dropping every other provider entry also stops the selector from offering
    to install a package, which a running `uvx` install would refuse anyway.

    Returns:
        A copy of `catalog` limited to `providers`.
    """
    policy = catalog.presentation_config()
    models = [
        spec
        for spec in catalog.models
        if spec.partition(":")[0] in providers and policy.is_model_allowed(spec)
    ]
    return catalog.model_copy(
        update={
            "models": models,
            "profiles": {s: p for s, p in catalog.profiles.items() if s in models},
            "providers": {
                name: info
                for name, info in catalog.providers.items()
                if name in providers
            },
        }
    )


async def accept_model_choice(
    providers: frozenset[str], options: LaunchOptions, model_spec: str
) -> str | MissingProviderKey | None:
    """Check a chosen spec, construct it, then save it as the default model.

    The selector accepts free-form `provider:model` text, which bypasses the
    provider-limited catalog, so the choice is checked here against current
    config and credentials. Nothing is saved unless the canonical spec
    constructs, so a working default is never replaced by a broken one.

    Args:
        providers: Providers the step offers.
        options: Launch flags the session model is constructed with.
        model_spec: The spec the user chose or typed.

    Returns:
        `None` once saved; `MissingProviderKey` when its credentials failed;
            else a footer message. Nothing is saved unless it returns `None`.
    """
    from deepagents_code.model_config import (
        ModelConfig,
        ModelNotAllowedError,
        clear_caches,
        save_default_model,
    )

    clear_caches()
    parsed = resolve_permitted_spec(ModelConfig.load(), model_spec)
    if isinstance(parsed, str):
        return parsed
    if parsed.provider not in providers:
        names = ", ".join(sorted(provider_display_name(p) for p in providers))
        return f"Choose a model from {names}."
    unusable = provider_problem(parsed.provider)
    if unusable is not None:
        return unusable
    problem = await asyncio.to_thread(
        construction_problem, str(parsed), parsed.provider, options
    )
    if isinstance(problem, MissingProviderKey):
        return problem
    if problem is not None:
        return problem.message
    try:
        saved = save_default_model(str(parsed))
    except ModelNotAllowedError as exc:
        return str(exc)
    return None if saved else f"Could not save {parsed} to config.toml."


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
    provider: str | None,
    *,
    model_override: str | None = None,
    options: LaunchOptions | None = None,
) -> LoginOutcome:
    """Run the sign-in app in the current terminal.

    Args:
        provider: Provider the client asked to sign in to.
        model_override: Launch-time `--model`; when set, readiness checks it
            instead of saved models and the model step never runs.
        options: Launch flags the session model is constructed with.

    Returns:
        The final outcome; an app quit by other means counts as a cancel of
            the step on screen.
    """
    fixed_model = bool(model_override and model_override.strip())
    readiness = partial(check_readiness, model_override=model_override)
    app = AuthLoginApp(
        provider=provider,
        readiness=readiness,
        fixed_model=fixed_model,
        options=options,
    )
    result = app.run()
    return result if result is not None else LoginCancelled(app.active_step)
