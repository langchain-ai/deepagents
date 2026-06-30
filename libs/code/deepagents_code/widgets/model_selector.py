"""Interactive model selector screen for `/model` command."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.events import (
    Click,  # noqa: TC002 - needed at runtime for Textual event dispatch
)
from textual.fuzzy import Matcher
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from textual.app import ComposeResult

from deepagents_code import theme
from deepagents_code.auth_display import format_auth_indicator
from deepagents_code.config import Glyphs, get_glyphs, is_ascii_mode
from deepagents_code.model_config import (
    CODEX_PROVIDER,
    ModelConfig,
    ModelProfileEntry,
    ProviderAuthState,
    ProviderAuthStatus,
    clear_default_model,
    get_available_models,
    get_credential_env_var,
    get_model_profiles,
    get_provider_auth_status,
    load_recent_models,
    save_default_model,
)

logger = logging.getLogger(__name__)

_MODEL_LIST_MAX_HEIGHT = 16
"""Upper bound (in cells) for the model selector list.

Keep in sync with the `max-height: 16` in the `.model-list` CSS below; Textual
CSS cannot reference Python constants, so the static cap and the runtime
`_fit_model_list` clamp must agree.
"""

_MODEL_LIST_MIN_HEIGHT = 1
"""Floor (in cells) so the model selector list never collapses to zero."""

_RECENT_SECTION_LABEL = "Recent"
"""Header label for the MRU pseudo-provider section pinned at the top of `/model`.

Recent picks are surfaced regardless of the recommended-only toggle —
they're a personal signal that outweighs curation — and de-duplicated
from the per-provider sections below.
"""


_RECOMMENDED_MODELS: frozenset[str] = frozenset(
    {
        "anthropic:claude-opus-4-6",
        "anthropic:claude-opus-4-7",
        "anthropic:claude-opus-4-8",
        "anthropic:claude-sonnet-5",
        "baseten:deepseek-ai/DeepSeek-V4-Pro",
        "baseten:moonshotai/Kimi-K2.7-Code",
        "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
        "baseten:zai-org/GLM-5.2",
        "fireworks:accounts/fireworks/models/deepseek-v4-pro",
        "fireworks:accounts/fireworks/models/glm-5p1",
        "fireworks:accounts/fireworks/models/glm-5p2",
        "fireworks:accounts/fireworks/models/kimi-k2p6",
        "fireworks:accounts/fireworks/models/kimi-k2p7-code",
        "fireworks:accounts/fireworks/models/minimax-m2p7",
        "fireworks:accounts/fireworks/models/minimax-m3",
        "fireworks:accounts/fireworks/models/qwen3p6-plus",
        "fireworks:accounts/fireworks/models/qwen3p7-plus",
        "google_genai:gemini-3.5-flash",
        "google_genai:gemini-3.1-pro-preview",
        "ollama:deepseek-v4-flash:cloud",
        "ollama:deepseek-v4-pro:cloud",
        "ollama:glm-5.2:cloud",
        "ollama:kimi-k2.7-code:cloud",
        "ollama:minimax-m3:cloud",
        "openai:gpt-5.4",
        "openai:gpt-5.4-mini",
        "openai:gpt-5.4-pro",
        "openai:gpt-5.5",
        "openai:gpt-5.5-pro",
        "openai_codex:gpt-5.2",
        "openai_codex:gpt-5.3-codex",
        "openai_codex:gpt-5.4",
        "openai_codex:gpt-5.4-mini",
        "openai_codex:gpt-5.5",
        "openrouter:anthropic/claude-opus-4.6",
        "openrouter:anthropic/claude-opus-4.7",
        "openrouter:anthropic/claude-opus-4.7-fast",
        "openrouter:anthropic/claude-opus-4.8",
        "openrouter:anthropic/claude-sonnet-5",
        "openrouter:deepseek/deepseek-v4-flash",
        "openrouter:deepseek/deepseek-v4-flash:free",
        "openrouter:deepseek/deepseek-v4-pro",
        "openrouter:google/gemini-3.5-flash",
        "openrouter:google/gemini-3.1-pro-preview",
        "openrouter:minimax/minimax-m2.7",
        "openrouter:moonshotai/kimi-k2.7-code",
        "openrouter:openai/gpt-5.4",
        "openrouter:openai/gpt-5.4-mini",
        "openrouter:openai/gpt-5.4-pro",
        "openrouter:openai/gpt-5.5",
        "openrouter:openai/gpt-5.5-pro",
        "openrouter:openrouter/fusion",
        "openrouter:qwen/qwen3.7-plus",
        "openrouter:z-ai/glm-5.2",
    }
)
"""Hand-curated frontier-tier models promoted across the UI.

Used by the onboarding picker (`curated=True`) and by the in-`/model`
"Recommended only" toggle (Ctrl+R). Same model IDs may appear under multiple
providers (e.g. Kimi-K2.7-Code via `baseten`, `fireworks`, `ollama`, and
`openrouter`) and are listed under each provider intentionally so the user
can pick whichever provider they have credentials for.
"""


class _ModelData(NamedTuple):
    """Model discovery data returned by `ModelSelectorScreen._load_model_data`.

    Attributes:
        all_models: `(provider:model spec, provider)` pairs for every model to
            surface, including install-required recommended models.
        default_spec: The configured default model spec, or `None`.
        profiles: Spec string to profile entry mapping.
        recent_specs: Most-recent-first `provider:model` specs read from
            `~/.deepagents/.state/recent_models.json`.
        install_extras: Each surfaced-but-uninstalled provider mapped to the
            extra that installs it.
    """

    all_models: list[tuple[str, str]]
    default_spec: str | None
    profiles: Mapping[str, ModelProfileEntry]
    recent_specs: list[str]
    install_extras: dict[str, str]


class ModelOption(Static):
    """A clickable model option in the selector."""

    def __init__(
        self,
        label: str | Content,
        model_spec: str,
        provider: str,
        index: int,
        *,
        auth_status: ProviderAuthStatus | None = None,
        classes: str = "",
    ) -> None:
        """Initialize a model option.

        Args:
            label: Display content — a `Content` object (preferred) or a
                plain string that `Static` will parse as markup.
            model_spec: The model specification (provider:model format).
            provider: The provider name.
            index: The index of this option in the filtered list.
            auth_status: Provider auth/readiness status.
            classes: CSS classes for styling.
        """
        super().__init__(label, classes=classes)
        self.model_spec = model_spec
        self.index = index
        self.auth_status = auth_status or ProviderAuthStatus(
            state=ProviderAuthState.UNKNOWN,
            provider=provider,
            detail="credentials unknown",
        )

    @property
    def provider(self) -> str:
        """Provider name, derived from the embedded auth status."""
        return self.auth_status.provider

    class Clicked(Message):
        """Message sent when a model option is clicked."""

        def __init__(self, model_spec: str, provider: str, index: int) -> None:
            """Initialize the Clicked message.

            Args:
                model_spec: The model specification.
                provider: The provider name.
                index: The index of the clicked option.
            """
            super().__init__()
            self.model_spec = model_spec
            self.provider = provider
            self.index = index

    def on_click(self, event: Click) -> None:
        """Handle click on this option.

        Args:
            event: The click event.
        """
        event.stop()
        self.post_message(self.Clicked(self.model_spec, self.provider, self.index))


class ModelSelectorScreen(ModalScreen[tuple[str, str] | None]):
    """Full-screen modal for model selection.

    Displays available models grouped by provider with keyboard navigation
    and search filtering. Current model is highlighted.

    Returns (model_spec, provider) tuple on selection, or None on cancel.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("tab", "tab_complete", "Tab complete", show=False, priority=True),
        Binding("pageup", "page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "page_down", "Page down", show=False, priority=True),
        Binding("enter", "select", "Select", show=False, priority=True),
        Binding("ctrl+s", "set_default", "Set default", show=False, priority=True),
        Binding(
            "ctrl+r",
            "toggle_recommended",
            "Recommended only",
            show=False,
            priority=True,
        ),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]
    """Key bindings for model navigation, selection, defaulting, and cancel.

    Arrows move the cursor, Page Up/Down jump by a visual page, Tab copies
    the highlighted spec into the filter input, Enter selects, Ctrl+S
    toggles the default model, Ctrl+R toggles between showing all installed
    models and the hand-curated "recommended" subset, and Esc dismisses. All
    bindings use `priority=True` so they take precedence over the embedded
    `Input`; vim-style `j`/`k` bindings are deliberately omitted because
    they would prevent typing those letters into the always-focused filter
    input.
    """

    CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    ModelSelectorScreen > Vertical {
        width: 76;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    ModelSelectorScreen .model-selector-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    ModelSelectorScreen .model-selector-description {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    ModelSelectorScreen .model-selector-info {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    ModelSelectorScreen #model-filter {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    ModelSelectorScreen #model-filter:focus {
        border: solid $primary;
    }

    ModelSelectorScreen .model-list {
        height: auto;
        min-height: 1;
        max-height: 16;  /* keep in sync with `_MODEL_LIST_MAX_HEIGHT` */
        scrollbar-gutter: stable;
        background: $background;
    }

    ModelSelectorScreen #model-options {
        height: auto;
    }

    ModelSelectorScreen .model-provider-header {
        color: $primary;
        margin-top: 1;
    }

    ModelSelectorScreen #model-options > .model-provider-header:first-child {
        margin-top: 0;
    }

    ModelSelectorScreen .model-option {
        height: 1;
        padding: 0 1;
    }

    ModelSelectorScreen .model-option:hover {
        background: $surface-lighten-1;
    }

    ModelSelectorScreen .model-option-selected {
        background: $primary;
        color: $background;
        text-style: bold;
    }

    ModelSelectorScreen .model-option-selected:hover {
        background: $primary-lighten-1;
    }

    ModelSelectorScreen .model-option-current {
        text-style: italic;
    }

    ModelSelectorScreen .model-selector-help {
        height: auto;  /* keep auto so the standard footer wraps; see _help_text */
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
        text-align: center;
    }

    ModelSelectorScreen .model-detail-footer {
        height: 4;
        padding: 0 2;
        margin-top: 1;
    }
    """
    """Styling for the modal shell, filter input, provider-grouped list, detail
    footer, and help text."""

    def __init__(
        self,
        current_model: str | None = None,
        current_provider: str | None = None,
        cli_profile_override: dict[str, Any] | None = None,
        *,
        curated: bool = False,
        title: str | None = None,
        description: str | Content | None = None,
        result_callback: Callable[[tuple[str, str] | None], None] | None = None,
    ) -> None:
        """Initialize the ModelSelectorScreen.

        Data loading (model discovery, profiles) is deferred to `on_mount`
        so the screen pushes instantly and populates asynchronously.

        Args:
            current_model: The currently active model name (to highlight).
            current_provider: The provider of the current model.
            cli_profile_override: Extra profile fields from `--profile-override`.

                Merged on top of upstream + config.toml profiles so that app
                overrides appear with `*` markers in the detail footer.
            curated: Whether to show a short, profile-ranked model subset.
            title: Optional title override for the selector.
            description: Optional description shown below the title.
            result_callback: Optional callback for selector results when the
                screen is displayed without a `push_screen` result callback.
        """
        super().__init__()
        self._current_model = current_model
        self._current_provider = current_provider
        self._cli_profile_override = cli_profile_override
        self._curated = curated
        self._title = title
        self._description = description
        self._result_callback = result_callback
        # Standard /model defaults to the curated recommended subset so users
        # face less decision fatigue; onboarding (`curated=True`) already
        # constrains the list via `_curated`, so leaving this False there
        # avoids double-flagging in `_apply_subset`.
        self._recommended_only = not curated

        self._unfiltered_models: list[tuple[str, str]] = []
        self._recent_specs: list[str] = []
        # Providers surfaced in the list whose integration package is not
        # installed, mapped to the extra that installs them. Selecting one
        # routes through the install-confirm modal instead of an auth prompt.
        self._install_extras: dict[str, str] = {}
        # Set when the user confirms installing a provider's extra; the app
        # reads this off the screen after dismissal to install then switch.
        self.pending_install_extra: str | None = None

        self._all_models: list[tuple[str, str]] = []
        self._filtered_models: list[tuple[str, str]] = []
        self._selected_index = 0
        self._options_container: Container | None = None
        self._option_widgets: list[ModelOption] = []
        self._filter_text = ""
        self._current_spec: str | None = None
        if current_model and current_provider:
            self._current_spec = f"{current_provider}:{current_model}"
        self._default_spec: str | None = None
        self._profiles: Mapping[str, ModelProfileEntry] = {}
        self._loaded = False

    def _info_line_content(self) -> Content:
        """Build the info line shown above the filter input.

        Reflects whether the screen is filtered to the recommended subset.

        Returns:
            Styled `Content` for the info line.
        """
        if self._filter_text.strip():
            return Content.styled("Searching all models from installed providers")
        if self._recommended_only:
            return Content.styled(
                "Showing recommended models — Ctrl+R for all",
            )
        return Content.styled(
            "Showing all models from installed providers — Ctrl+R for recommended",
        )

    def _update_info_line(self) -> None:
        """Refresh the standard selector info line."""
        if self._curated:
            return
        info = self.query_one("#model-selector-info", Static)
        info.update(self._info_line_content())

    def _help_text(self) -> str:
        """Build the footer help text.

        Curated/onboarding mode omits the Ctrl+S and Ctrl+R hints. Escape stays
        bound but is left off the hint line — modal dismissal via Escape is
        conventional, and advertising it would only lengthen an already-wrapping
        line. In standard mode the full line exceeds the modal width, so the
        help `Static` is sized to grow (auto height) and wraps to two rows
        rather than clipping the trailing hints.

        Returns:
            The bullet-separated help line.
        """
        glyphs = get_glyphs()
        parts = [
            f"{glyphs.arrow_up}/{glyphs.arrow_down} navigate",
            "Tab autocomplete",
            "Enter select",
        ]
        if not self._curated:
            parts.extend(("Ctrl+S set default", "Ctrl+R recommended"))
        sep = f" {glyphs.bullet} "
        return sep.join(parts)

    def _find_current_model_index(self) -> int:
        """Find the index of the current model in the filtered list.

        Returns:
            Index of the current model, or 0 if not found.
        """
        if not self._current_model or not self._current_provider:
            return 0

        current_spec = f"{self._current_provider}:{self._current_model}"
        for i, (model_spec, _) in enumerate(self._filtered_models):
            if model_spec == current_spec:
                return i
        return 0

    def _initial_selected_index(self) -> int:
        """Return the default highlighted row for the current selector mode."""
        if self._curated:
            return 0
        return self._find_current_model_index()

    def compose(self) -> ComposeResult:
        """Compose the screen layout.

        Yields:
            Widgets for the model selector UI.
        """
        with Vertical():
            # Title with current model in provider:model format
            if self._title:
                title = self._title
            elif self._current_model and self._current_provider:
                current_spec = f"{self._current_provider}:{self._current_model}"
                title = f"Select Model (current: {current_spec})"
            elif self._current_model:
                title = f"Select Model (current: {self._current_model})"
            else:
                title = "Select Model"
            yield Static(title, classes="model-selector-title")
            if self._description:
                yield Static(
                    self._description,
                    classes="model-selector-description",
                )

            if not self._curated:
                yield Static(
                    self._info_line_content(),
                    classes="model-selector-info",
                    id="model-selector-info",
                )

            # Search input
            yield Input(
                placeholder="Type to filter or enter provider:model...",
                id="model-filter",
            )

            # Scrollable model list
            with VerticalScroll(classes="model-list"):
                self._options_container = Container(id="model-options")
                yield self._options_container

            # Model detail footer
            yield Static("", classes="model-detail-footer", id="model-detail-footer")

            yield Static(self._help_text(), classes="model-selector-help")

    @staticmethod
    def _load_model_data(
        cli_override: dict[str, Any] | None,
        *,
        include_uninstalled: bool = True,
        include_recent: bool = True,
    ) -> _ModelData:
        """Gather model discovery data synchronously.

        Intended to be called via `asyncio.to_thread` so filesystem I/O in
        `get_available_models` does not block the event loop.

        Args:
            cli_override: Extra profile fields from `--profile-override`.
            include_uninstalled: When `True`, append recommended models that
                aren't already surfaced, in two cases: (1) the provider
                integration isn't installed, added as greyed-out
                install-required rows; (2) the provider is installed but its
                upstream profiles omit the model, added as normal selectable
                rows.
            include_recent: When `True`, load the recent-models MRU so the
                pinned "Recent" section can render. Onboarding sets this
                `False`: first-run users have never picked a model, and the
                startup default-fallback resolution writes its auto-detected
                pick into the MRU, which would otherwise surface as a bogus
                "Recent" entry the user never chose.

        Returns:
            A `_ModelData` bundle of the discovered models, default spec,
                profiles, recent specs, and install-required provider extras.
        """
        available = get_available_models()
        config = ModelConfig.load()
        all_models: list[tuple[str, str]] = [
            (f"{provider}:{model}", provider)
            for provider, models in available.items()
            for model in models
        ]

        install_extras: dict[str, str] = {}
        if include_uninstalled:
            from deepagents_code.config_manifest import (
                is_provider_package_installed,
                provider_install_extra,
            )

            # Seeded from the discovered models; a recommended spec already
            # surfaced here is skipped below. Recommended specs are unique (a
            # frozenset iterated once), so this entry guard is the only dedup
            # needed and the set never has to grow inside the loop.
            existing_specs = {spec for spec, _ in all_models}
            installed_recommended: list[tuple[str, str]] = []
            uninstalled_recommended: list[tuple[str, str]] = []
            for spec in sorted(_RECOMMENDED_MODELS):
                if spec in existing_specs:
                    continue
                provider = spec.split(":", 1)[0]
                try:
                    if not config.is_provider_enabled(provider):
                        continue
                    extra = provider_install_extra(provider)
                    provider_installed = is_provider_package_installed(provider)
                except Exception:
                    # Isolate per-provider probe failures so one bad recommended
                    # provider can't take down the entire model list (the caller
                    # degrades any raise here to an empty selector). The append
                    # bookkeeping below stays outside this guard so genuine logic
                    # bugs surface instead of being silently swallowed.
                    logger.warning(
                        "Skipping recommended model %r while merging "
                        "recommendations into the model list",
                        spec,
                        exc_info=True,
                    )
                    continue
                if provider in available and provider_installed:
                    # Provider is installed and discoverable, but its upstream
                    # profiles don't surface this curated model (missing entry
                    # or filtered out). Add it as a normal selectable row so the
                    # hardcoded recommendation isn't silently dropped when the
                    # profile list lags.
                    installed_recommended.append((spec, provider))
                    continue
                if extra is None or provider_installed:
                    continue
                install_extras[provider] = extra
                uninstalled_recommended.append((spec, provider))
            all_models.extend(installed_recommended)
            all_models.extend(uninstalled_recommended)

        profiles = get_model_profiles(cli_override=cli_override)
        recent_specs = load_recent_models() if include_recent else []
        return _ModelData(
            all_models,
            config.default_model,
            profiles,
            recent_specs,
            install_extras,
        )

    def _apply_subset(
        self,
        all_models: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Apply the active subset filter (onboarding or recommended-only).

        Recently-used specs are unioned in even when the recommended-only
        toggle is on, so personal usage always wins over curation. Onboarding
        intentionally keeps a tight curated subset and skips this union.

        Args:
            all_models: Full list of `(provider:model, provider)` pairs.

        Returns:
            The list reduced to the recommended subset when either
                `_curated` (onboarding) or `_recommended_only` (Ctrl+R) is
                active. Falls back to the full list when no recommended
                models are installed so the screen is never empty.
        """
        if self._curated:
            return self._curate_models(all_models)
        if self._recommended_only:
            curated = self._curate_models(all_models)
            curated_specs = {spec for spec, _ in curated}
            # Order follows all_models (insertion), not MRU; _update_display
            # rebuilds visual order by iterating self._recent_specs directly.
            recent_extra = [
                (spec, provider)
                for spec, provider in all_models
                if spec in self._recent_specs and spec not in curated_specs
            ]
            return [*recent_extra, *curated]
        return list(all_models)

    @staticmethod
    def _curate_models(
        all_models: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Return the curated onboarding list in the model switcher's order.

        Returns the eval-backed frontier subset when any of those models are
        available. When none are, returns the full switcher list so onboarding
        still surfaces every installed provider rather than a truncated slice.

        Args:
            all_models: Full list of `(provider:model, provider)` pairs.

        Returns:
            Curated model list for onboarding setup.
        """
        frontier = [
            (spec, provider)
            for spec, provider in all_models
            if spec in _RECOMMENDED_MODELS
        ]
        return frontier or all_models

    async def on_mount(self) -> None:
        """Set up the screen on mount.

        Loads model data in a background thread so the screen frame renders
        immediately, then populates the model list.
        """
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            container = self.query_one(Vertical)
            container.styles.border = ("ascii", colors.success)
        self.call_after_refresh(self._fit_model_list)

        # Focus the filter input immediately so the user can start typing
        # while model data loads.
        filter_input = self.query_one("#model-filter", Input)
        filter_input.focus()

        # Offload to thread because get_available_models does filesystem I/O
        try:
            data = await asyncio.to_thread(
                self._load_model_data,
                self._cli_profile_override,
                include_uninstalled=True,
                include_recent=not self._curated,
            )
        except Exception:
            logger.exception("Failed to load model data for /model selector")
            self._loaded = True
            if self.is_running:
                self.notify(
                    "Could not load model list. "
                    "Check provider packages and config.toml.",
                    severity="error",
                    timeout=10,
                    markup=False,
                )
                await self._update_display()
                self._update_footer()
            return

        # Screen may have been dismissed while the thread was running
        if not self.is_running:
            return

        self._unfiltered_models = data.all_models
        self._default_spec = data.default_spec
        self._profiles = data.profiles
        self._recent_specs = data.recent_specs
        self._install_extras = data.install_extras
        self._all_models = self._apply_subset(self._unfiltered_models)
        self._filtered_models = list(self._all_models)
        self._selected_index = self._initial_selected_index()
        self._loaded = True

        # Re-apply any filter text the user typed while data was loading
        if self._filter_text:
            self._update_filtered_list()

        await self._update_display()
        self._update_footer()

    def on_resize(self) -> None:
        """Refit the model list when terminal dimensions change."""
        self.call_after_refresh(self._fit_model_list)

    def _fit_model_list(self) -> None:
        """Cap the model list so modal controls stay visible."""
        try:
            container = self.query_one(Vertical)
        except NoMatches:
            # This runs deferred via `call_after_refresh`/`on_resize`; the
            # screen may have been popped before it fires (e.g. a resize racing
            # dismissal). Sizing is cosmetic, so skip quietly but leave a
            # breadcrumb rather than letting it surface in the event loop.
            logger.debug(
                "Skipping model-list refit; screen not mounted",
                exc_info=True,
            )
            return
        # The screen is still mounted, so `.model-list` (always composed) must
        # exist; a missing body here is a structural regression, not the
        # teardown race, so let `NoMatches` surface rather than silently
        # rendering an uncapped list.
        body = self.query_one(".model-list", VerticalScroll)
        non_body_height = max(0, container.region.height - body.region.height)
        available_height = self.size.height - non_body_height
        max_height = max(
            _MODEL_LIST_MIN_HEIGHT,
            min(_MODEL_LIST_MAX_HEIGHT, available_height),
        )
        current = body.styles.max_height
        if current is not None and current.cells == max_height:
            return
        body.styles.max_height = max_height

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter models as user types.

        Args:
            event: The input changed event.
        """
        self._filter_text = event.value
        self._update_info_line()
        if not self._loaded:
            return  # on_mount will re-apply filter after data loads
        self._update_filtered_list()
        self.call_after_refresh(self._update_display)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key when filter input is focused.

        Args:
            event: The input submitted event.
        """
        event.stop()
        self.action_select()

    def on_model_option_clicked(self, event: ModelOption.Clicked) -> None:
        """Handle click on a model option.

        Args:
            event: The click event with model info.
        """
        self._selected_index = event.index
        self._select_with_auth_check(event.model_spec, event.provider)

    def _update_filtered_list(self) -> None:
        """Update the filtered models based on search text using fuzzy matching.

        Results are sorted by match score (best first), with installed
        providers ranked above not-yet-installed ones so the common case of
        picking an available model is never displaced by an install-required
        suggestion. In standard `/model` mode, non-empty searches span the
        full installed model list even when the default view is currently
        constrained to recommended models.
        """
        query = self._filter_text.strip()
        if not query:
            self._filtered_models = list(self._all_models)
            self._selected_index = self._initial_selected_index()
            return

        tokens = query.split()
        search_models = self._all_models if self._curated else self._unfiltered_models

        try:
            matchers = [Matcher(token, case_sensitive=False) for token in tokens]
            scored: list[tuple[float, str, str]] = []
            for spec, provider in search_models:
                scores = [m.match(spec) for m in matchers]
                if all(s > 0 for s in scores):
                    scored.append((min(scores), spec, provider))
        except Exception:
            # graceful fallback if Matcher fails on edge-case input
            logger.warning(
                "Fuzzy matcher failed for query %r, falling back to full list",
                query,
                exc_info=True,
            )
            self._filtered_models = list(search_models)
            self._selected_index = self._initial_selected_index()
            return

        self._filtered_models = [
            (spec, provider)
            for _installed, _score, spec, provider in sorted(
                (
                    (provider not in self._install_extras, score, spec, provider)
                    for score, spec, provider in scored
                ),
                reverse=True,
            )
        ]
        self._selected_index = 0

    @staticmethod
    def _unique_model_entries(
        models: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Return unique model/provider pairs while preserving first-seen order."""
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str]] = []
        for entry in models:
            if entry in seen:
                continue
            seen.add(entry)
            unique.append(entry)
        return unique

    @staticmethod
    def _find_same_occurrence_index(
        models: list[tuple[str, str]],
        entry: tuple[str, str],
        occurrence: int,
    ) -> int:
        """Find the `occurrence`th matching model/provider tuple in `models`.

        Args:
            models: Render-ordered model/provider pairs to search.
            entry: Exact `(model_spec, provider)` tuple to match.
            occurrence: One-based occurrence count to find.

        Returns:
            Matching index, or `0` if no matching occurrence exists.
        """
        matches = 0
        first_match: int | None = None
        for i, candidate in enumerate(models):
            if candidate != entry:
                continue
            if first_match is None:
                first_match = i
            matches += 1
            if matches == occurrence:
                return i
        return first_match or 0

    async def _update_display(self) -> None:
        """Render the model list grouped by provider.

        Performs a full DOM rebuild (removes all children, re-mounts).
        Arrow-key navigation uses `_move_selection` instead to avoid
        the cost of a full rebuild.
        """
        if not self._options_container:
            return

        await self._options_container.remove_children()
        self._option_widgets = []

        if not self._filtered_models:
            if not self._loaded:
                empty_content: Content = Content.styled("Loading models…", "dim")
            else:
                typed = self._filter_text.strip()
                if typed and ":" in typed:
                    empty_content = Content.assemble(
                        ("No matching models — press ", "dim"),
                        ("Enter", "bold"),
                        (" to use ", "dim"),
                        (typed, "bold"),
                        (" as a custom provider:model spec", "dim"),
                    )
                elif typed:
                    empty_content = Content.assemble(
                        ("No matching models — press ", "dim"),
                        ("Enter", "bold"),
                        (" to use ", "dim"),
                        (typed, "bold"),
                        (" as a custom model spec (no provider prefix)", "dim"),
                    )
                else:
                    empty_content = Content.styled("No matching models", "dim")
            await self._options_container.mount(Static(empty_content))
            self._update_footer()
            self.call_after_refresh(self._fit_model_list)
            return

        has_filter = bool(self._filter_text.strip())
        source = self._filtered_models if has_filter else self._all_models
        source_models = self._unique_model_entries(source)

        # Resolve which recent specs are present in the current filtered set.
        # Recent rendering only happens at the top of an unfiltered view; once
        # the user starts fuzzy-filtering, recents are surfaced through the
        # match logic like any other model so the search remains predictable.
        if has_filter:
            recent_entries: list[tuple[str, str]] = []
        else:
            spec_to_provider = dict(source_models)
            recent_entries = [
                (spec, spec_to_provider[spec])
                for spec in self._recent_specs
                if spec in spec_to_provider
            ]
        # Group models by provider, preserving insertion order so models
        # from the same provider cluster together in the visual list. Specs
        # also in the Recent section are intentionally kept here so a user
        # who opens `/model` always finds their model at its provider's
        # familiar position in addition to the MRU shortcut at the top.
        by_provider: dict[str, list[tuple[str, str]]] = {}
        for model_spec, provider in source_models:
            by_provider.setdefault(provider, []).append((model_spec, provider))

        # Rebuild _filtered_models to match the rendered order (recents first,
        # then provider-grouped). Without this, _filtered_models stays in
        # score-sorted order while _option_widgets follow rendered order,
        # causing _update_footer to look up the wrong model for the
        # highlighted index.
        grouped_order: list[tuple[str, str]] = list(recent_entries)
        for entries in by_provider.values():
            grouped_order.extend(entries)

        # Remap selected_index so the same visual occurrence stays highlighted.
        old_entry = self._filtered_models[self._selected_index]
        old_occurrence = self._filtered_models[: self._selected_index + 1].count(
            old_entry
        )
        self._filtered_models = grouped_order
        self._selected_index = self._find_same_occurrence_index(
            grouped_order,
            old_entry,
            old_occurrence,
        )

        glyphs = get_glyphs()
        flat_index = 0
        selected_widget: ModelOption | None = None

        # Build current model spec for comparison
        current_spec = None
        if self._current_model and self._current_provider:
            current_spec = f"{self._current_provider}:{self._current_model}"

        # Resolve provider auth upfront so the widget-building loop
        # stays focused on layout
        auth_providers = {provider for _, provider in self._filtered_models}
        auth_statuses = {p: get_provider_auth_status(p) for p in auth_providers}

        # Collect all widgets first, then batch-mount once to avoid
        # individual DOM mutations per widget
        all_widgets: list[Static] = []

        # Pinned "Recent" section — pseudo-provider header with no auth badge
        # because each entry already carries its real provider's auth state
        # via `ModelOption.auth_status`.
        if recent_entries:
            all_widgets.append(
                Static(
                    Content.from_markup(
                        "[bold]$label[/bold]", label=_RECENT_SECTION_LABEL
                    ),
                    classes="model-provider-header",
                )
            )
            for model_spec, real_provider in recent_entries:
                auth_status = auth_statuses[real_provider]
                is_current = model_spec == current_spec
                is_selected = flat_index == self._selected_index

                classes = "model-option"
                if is_selected:
                    classes += " model-option-selected"
                if is_current:
                    classes += " model-option-current"

                label = self._build_option_label(
                    model_spec, real_provider, auth_status, selected=is_selected
                )
                widget = ModelOption(
                    label=label,
                    model_spec=model_spec,
                    provider=real_provider,
                    index=flat_index,
                    auth_status=auth_status,
                    classes=classes,
                )
                all_widgets.append(widget)
                self._option_widgets.append(widget)
                if is_selected:
                    selected_widget = widget
                flat_index += 1

        for provider, model_entries in by_provider.items():
            # Provider header; auth/readiness indicator appended only when non-empty.
            auth_status = auth_statuses[provider]
            if provider in self._install_extras:
                auth_indicator = self._install_indicator()
            else:
                auth_indicator = self._format_auth_indicator(auth_status, glyphs)
            if auth_indicator:
                header_content = Content.from_markup(
                    "[bold]$provider[/bold] [dim]$auth[/dim]",
                    provider=provider,
                    auth=auth_indicator,
                )
            else:
                header_content = Content.from_markup(
                    "[bold]$provider[/bold]",
                    provider=provider,
                )
            all_widgets.append(Static(header_content, classes="model-provider-header"))

            for model_spec, _prov in model_entries:
                is_current = model_spec == current_spec
                is_selected = flat_index == self._selected_index

                classes = "model-option"
                if is_selected:
                    classes += " model-option-selected"
                if is_current:
                    classes += " model-option-current"

                label = self._build_option_label(
                    model_spec, provider, auth_status, selected=is_selected
                )
                widget = ModelOption(
                    label=label,
                    model_spec=model_spec,
                    provider=provider,
                    index=flat_index,
                    auth_status=auth_status,
                    classes=classes,
                )
                all_widgets.append(widget)
                self._option_widgets.append(widget)

                if is_selected:
                    selected_widget = widget

                flat_index += 1

        await self._options_container.mount(*all_widgets)

        # Scroll the selected item into view without animation so the list
        # appears already scrolled to the current model on first paint.
        if selected_widget:
            if self._selected_index == 0:
                # First item: scroll to top so header is visible
                scroll_container = self.query_one(".model-list", VerticalScroll)
                scroll_container.scroll_home(animate=False)
            else:
                selected_widget.scroll_visible(animate=False)

        self._update_footer()
        self.call_after_refresh(self._fit_model_list)

    @staticmethod
    def _format_auth_indicator(
        auth_status: ProviderAuthStatus,
        glyphs: Glyphs,
    ) -> str:
        """Build the provider header auth indicator.

        Args:
            auth_status: Provider auth/readiness status.
            glyphs: Glyph table for the active terminal mode.

        Returns:
            Text shown next to the provider name, or an empty string when no
                indicator should be rendered (e.g., `CONFIGURED`).
        """
        return format_auth_indicator(auth_status, glyphs)

    @staticmethod
    def _install_indicator() -> str:
        """Return the provider-header text for an uninstalled provider."""
        return "not installed"

    def _build_option_label(
        self,
        model_spec: str,
        provider: str,
        auth_status: ProviderAuthStatus,
        *,
        selected: bool,
    ) -> Content:
        """Build a model-option label from the current screen state.

        Centralizes the per-row flag derivation (current/default/status/
        `install_required`) shared by the full rebuild in `_update_display`
        and the incremental relabel in `_move_selection`, so the two paths
        cannot drift. The original `/model` dim-persistence bug came from
        exactly such drift: `_move_selection` omitted `install_required`,
        so uninstalled rows stopped rendering dimmed after navigation.

        Args:
            model_spec: The `provider:model` string for the row.
            provider: The row's provider key, tested against the
                install-required set.
            auth_status: Provider auth/readiness status for the row.
            selected: Whether this row is the highlighted one.

        Returns:
            Styled `Content` label.
        """
        return self._format_option_label(
            model_spec,
            selected=selected,
            current=model_spec == self._current_spec,
            auth_status=auth_status,
            is_default=model_spec == self._default_spec,
            status=self._get_model_status(model_spec),
            install_required=provider in self._install_extras,
        )

    @staticmethod
    def _format_option_label(
        model_spec: str,
        *,
        selected: bool,
        current: bool,
        auth_status: ProviderAuthStatus,
        is_default: bool = False,
        status: str | None = None,
        install_required: bool = False,
    ) -> Content:
        """Build the display label for a model option.

        Args:
            model_spec: The `provider:model` string.
            selected: Whether this option is currently highlighted.
            current: Whether this is the active model.
            auth_status: Provider auth/readiness status.
            is_default: Whether this is the configured default model.
            status: Model status from profile (e.g., `'deprecated'`,
                `'beta'`, `'alpha'`). `'deprecated'` renders in red;
                other non-None values render in yellow.
            install_required: Whether the provider's integration package is not
                installed; renders the spec dimmed since selecting it prompts
                an install rather than switching immediately.

        Returns:
            Styled Content label.
        """
        colors = theme.get_theme_colors()
        glyphs = get_glyphs()
        cursor = f"{glyphs.cursor} " if selected else "  "
        # When selected, skip the inline primary color — CSS already flips the
        # row to ($primary bg, $background fg). Keep `bold` so the default
        # emphasis survives both states.
        if install_required and not selected:
            spec = Content.styled(model_spec, "dim")
        elif auth_status.blocks_start:
            spec = Content.styled(model_spec, colors.warning)
        elif is_default and selected:
            spec = Content.styled(model_spec, "bold")
        elif is_default:
            spec = Content.styled(model_spec, f"bold {colors.primary}")
        else:
            spec = Content(model_spec)
        suffix = Content.styled(" (current)", "dim") if current else Content("")
        if is_default and selected:
            default_suffix = Content.styled(" (default)", "bold")
        elif is_default:
            default_suffix = Content.styled(" (default)", f"bold {colors.primary}")
        else:
            default_suffix = Content("")
        if status == "deprecated":
            status_suffix = Content.styled(" (deprecated)", colors.error)
        elif status:
            status_suffix = Content.styled(f" ({status})", colors.warning)
        else:
            status_suffix = Content("")
        return Content.assemble(cursor, spec, suffix, default_suffix, status_suffix)

    @staticmethod
    def _format_footer(
        profile_entry: ModelProfileEntry | None,
        glyphs: Glyphs,
    ) -> Content:
        """Build the detail footer text for the highlighted model.

        Args:
            profile_entry: Profile data with override tracking, or None.
            glyphs: Glyph set for display characters.

        Returns:
            Styled `Content` for the 4-line footer.
        """
        from deepagents_code.textual_adapter import format_token_count

        if profile_entry is None or not profile_entry["profile"]:
            return Content.styled("Model profile not available :(\n\n\n", "dim")

        profile = profile_entry["profile"]
        overridden = profile_entry["overridden_keys"]

        colors = theme.get_theme_colors()

        def _mark(key: str, text: str) -> Content:
            if key in overridden:
                return Content.styled(f"*{text}", colors.warning)
            return Content(text)

        def _format_token(key: str, suffix: str) -> Content | None:
            """Format a token-count profile key, falling back to the raw value.

            Returns:
                Styled `Content` with override marker, or None if key absent.
            """
            val = profile.get(key)
            if val is None:
                return None
            try:
                text = f"{format_token_count(int(val))} {suffix}"
            except (ValueError, TypeError, OverflowError):
                text = f"{val} {suffix}"
            return _mark(key, text)

        def _format_flags(keys: list[tuple[str, str]]) -> list[Content]:
            """Render boolean profile keys as green (on) or dim (off) labels.

            Returns:
                List of styled `Content` objects for present keys.
            """
            parts: list[Content] = []
            for key, label in keys:
                if key in profile:
                    base = (
                        Content.styled(label, colors.success)
                        if profile[key]
                        else Content.styled(label, "dim")
                    )
                    if key in overridden:
                        base = Content.assemble(
                            Content.styled("*", colors.warning), base
                        )
                    parts.append(base)
            return parts

        # Line 1: Context window
        token_keys = [("max_input_tokens", "in"), ("max_output_tokens", "out")]
        ctx_parts = [p for k, s in token_keys if (p := _format_token(k, s)) is not None]
        bullet_sep = Content(f" {glyphs.bullet} ")
        line1 = (
            Content.assemble("Context: ", bullet_sep.join(ctx_parts))
            if ctx_parts
            else Content("")
        )

        # Line 2: Input modalities
        modality_keys = [
            ("text_inputs", "text"),
            ("image_inputs", "image"),
            ("audio_inputs", "audio"),
            ("pdf_inputs", "pdf"),
            ("video_inputs", "video"),
        ]
        modality_parts = _format_flags(modality_keys)
        space = Content(" ")
        line2 = (
            Content.assemble("Input: ", space.join(modality_parts))
            if modality_parts
            else Content("")
        )

        # Line 3: Capabilities
        capability_keys = [
            ("reasoning_output", "reasoning"),
            ("tool_calling", "tool calling"),
            ("structured_output", "structured output"),
        ]
        cap_parts = _format_flags(capability_keys)
        line3 = (
            Content.assemble("Capabilities: ", space.join(cap_parts))
            if cap_parts
            else Content("")
        )

        # Line 4: Override notice
        displayed_keys = {k for k, _ in token_keys + modality_keys + capability_keys}
        has_visible_override = bool(overridden & displayed_keys)
        line4 = (
            Content.from_markup("[dim][yellow]*[/yellow] = override[/dim]")
            if has_visible_override
            else Content("")
        )

        return Content.assemble(line1, "\n", line2, "\n", line3, "\n", line4)

    def _get_model_status(self, model_spec: str) -> str | None:
        """Look up the status field for a model from its profile.

        Args:
            model_spec: The `provider:model` string.

        Returns:
            Status string (e.g., `'deprecated'`) if the model has a profile
            with a `status` key, otherwise None.
        """
        entry = self._profiles.get(model_spec)
        if entry is None:
            return None
        profile = entry.get("profile")
        if not profile:
            return None
        return profile.get("status")

    def _update_footer(self) -> None:
        """Update the detail footer for the currently highlighted model."""
        footer = self.query_one("#model-detail-footer", Static)
        if not self._filtered_models:
            footer.update(Content.styled("No model selected", "dim"))
            return
        index = min(self._selected_index, len(self._filtered_models) - 1)
        spec, _ = self._filtered_models[index]
        entry = self._profiles.get(spec)
        try:
            text = self._format_footer(entry, get_glyphs())
        except (KeyError, ValueError, TypeError):  # Resilient footer rendering
            logger.warning("Failed to format footer for %s", spec, exc_info=True)
            text = Content.styled("Could not load profile details\n\n\n", "dim")
        footer.update(text)

    def _move_selection(self, delta: int) -> None:
        """Move selection by delta, updating only the affected widgets.

        Args:
            delta: Number of positions to move (-1 for up, +1 for down).
        """
        if not self._filtered_models or not self._option_widgets:
            return

        count = len(self._filtered_models)
        old_index = self._selected_index
        new_index = (old_index + delta) % count
        self._selected_index = new_index

        # Update the previously selected widget
        old_widget = self._option_widgets[old_index]
        old_widget.remove_class("model-option-selected")
        old_widget.update(
            self._build_option_label(
                old_widget.model_spec,
                old_widget.provider,
                old_widget.auth_status,
                selected=False,
            )
        )

        # Update the newly selected widget
        new_widget = self._option_widgets[new_index]
        new_widget.add_class("model-option-selected")
        new_widget.update(
            self._build_option_label(
                new_widget.model_spec,
                new_widget.provider,
                new_widget.auth_status,
                selected=True,
            )
        )

        # Scroll the selected item into view
        if new_index == 0:
            scroll_container = self.query_one(".model-list", VerticalScroll)
            scroll_container.scroll_home(animate=False)
        else:
            new_widget.scroll_visible()

        self._update_footer()

    def action_move_up(self) -> None:
        """Move selection up."""
        self._move_selection(-1)

    def action_move_down(self) -> None:
        """Move selection down."""
        self._move_selection(1)

    def action_tab_complete(self) -> None:
        """Replace search text with the currently selected model spec."""
        if not self._filtered_models:
            return
        model_spec, _ = self._filtered_models[self._selected_index]
        filter_input = self.query_one("#model-filter", Input)
        filter_input.value = model_spec
        filter_input.cursor_position = len(model_spec)

    def _visible_page_size(self) -> int:
        """Return the number of model options that fit in one visual page.

        Returns:
            Number of model options per page, at least 1.
        """
        default_page_size = 10
        try:
            scroll = self.query_one(".model-list", VerticalScroll)
            height = scroll.size.height
        except Exception:  # noqa: BLE001  # Fallback to default page size on any widget query error
            return default_page_size
        if height <= 0:
            return default_page_size

        total_models = len(self._filtered_models)
        if total_models == 0:
            return default_page_size

        # Each provider header = 1 row + margin-top: 1 (first has margin 0)
        num_headers = len(self.query(".model-provider-header"))
        header_rows = max(0, num_headers * 2 - 1) if num_headers else 0
        total_rows = total_models + header_rows
        return max(1, int(height * total_models / total_rows))

    def action_page_up(self) -> None:
        """Move selection up by one visible page."""
        if not self._filtered_models:
            return
        page = self._visible_page_size()
        target = max(0, self._selected_index - page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_page_down(self) -> None:
        """Move selection down by one visible page."""
        if not self._filtered_models:
            return
        count = len(self._filtered_models)
        page = self._visible_page_size()
        target = min(count - 1, self._selected_index + page)
        delta = target - self._selected_index
        if delta != 0:
            self._move_selection(delta)

    def action_select(self) -> None:
        """Select the current model."""
        # If there are filtered results, always select the highlighted model
        if self._filtered_models:
            model_spec, provider = self._filtered_models[self._selected_index]
            self._select_with_auth_check(model_spec, provider)
            return

        # No matches - check if user typed a custom provider:model spec
        filter_input = self.query_one("#model-filter", Input)
        custom_input = filter_input.value.strip()

        if custom_input and ":" in custom_input:
            provider = custom_input.split(":", 1)[0]
            self._select_with_auth_check(custom_input, provider)
        elif custom_input:
            self._dismiss_with_result((custom_input, ""))

    def _select_with_auth_check(self, model_spec: str, provider: str) -> None:
        """Either dismiss with the selection, or prompt for credentials first.

        When the highlighted provider has `blocks_start` auth (typically a
        missing API key), open the in-TUI auth prompt instead of dismissing.
        On save, dismiss with the originally-selected model. On cancel, stay
        on the selector and refresh the credential indicator so the user can
        try again or pick a different provider.
        """
        if not provider:
            self._dismiss_with_result((model_spec, provider))
            return

        from deepagents_code.config_manifest import (
            is_provider_package_installed,
            provider_install_extra,
        )

        extra = provider_install_extra(provider)
        if extra is not None and not is_provider_package_installed(provider):
            if self._curated:
                # Onboarding installs first, then prompts for credentials from the
                # launch flow, matching the dependency screen's auto-install copy.
                self._dismiss_with_result((model_spec, provider))
                return
            self._prompt_install_provider(model_spec, provider, extra)
            return

        status = get_provider_auth_status(provider)
        if not status.blocks_start:
            self._dismiss_with_result((model_spec, provider))
            return

        if provider == CODEX_PROVIDER:
            # ChatGPT auth is an OAuth browser flow, not an API key, so the
            # generic key/base-url prompt doesn't apply. Route to the
            # dedicated sign-in modal (the same one the auth manager uses).
            self._prompt_codex_sign_in(model_spec, provider)
            return

        env_var = status.env_var or get_credential_env_var(provider)

        from deepagents_code.widgets.auth import AuthPromptScreen, AuthResult

        def _on_auth_done(result: AuthResult | None) -> None:
            if result is AuthResult.SAVED:
                self._dismiss_with_result((model_spec, provider))
                return
            # On DELETED or CANCELLED the user explicitly chose not to
            # provide a key; refresh the credential indicator and stay on
            # the selector so they can pick a different provider.
            self.call_after_refresh(self._update_display)

        self.app.push_screen(
            AuthPromptScreen(
                provider,
                env_var,
                reason=f"Required to use {model_spec}",
            ),
            _on_auth_done,
        )

    def _prompt_install_provider(
        self, model_spec: str, provider: str, extra: str
    ) -> None:
        """Confirm installing a provider's extra before selecting its model.

        On confirm, record the extra on `pending_install_extra` and dismiss
        with the selected model so the app can install the extra and then
        switch. On cancel, refresh the credential indicator and stay on the
        selector so the user can pick a different provider.
        """
        from deepagents_code.widgets.install_confirm import (
            InstallProviderConfirmScreen,
        )

        def _on_confirm(proceed: bool | None) -> None:
            if proceed:
                self.pending_install_extra = extra
                self._dismiss_with_result((model_spec, provider))
                return
            self.call_after_refresh(self._update_display)

        self.app.push_screen(
            InstallProviderConfirmScreen(provider, extra, model_spec),
            _on_confirm,
        )

    def _prompt_codex_sign_in(self, model_spec: str, provider: str) -> None:
        """Confirm, then run the ChatGPT OAuth flow for the selected model.

        Signing in launches a browser and a multi-minute loopback wait, so a
        confirmation modal is shown first. If the user declines, refresh the
        credential indicator and stay on the selector so they can retry or
        pick a different provider.
        """
        from deepagents_code.widgets.auth import AuthConfirmScreen

        def _on_confirm(proceed: bool | None) -> None:
            if proceed:
                self._run_codex_oauth(model_spec, provider)
                return
            self.call_after_refresh(self._update_display)

        confirm = AuthConfirmScreen(
            title="No ChatGPT sign-in detected",
            body=Content.from_markup(
                "[bold]$model[/bold] authenticates with ChatGPT, but no "
                "sign-in was detected. Sign in now, or return to the model "
                "list.",
                model=model_spec,
            ),
            help_text="Enter to sign in, Esc to return to model list",
        )
        self.app.push_screen(confirm, _on_confirm)

    def _run_codex_oauth(self, model_spec: str, provider: str) -> None:
        """Run the ChatGPT OAuth sign-in flow for the selected codex model.

        On a successful sign-in, dismiss with the originally-selected model.
        On cancel or error, refresh the credential indicator and stay on the
        selector so the user can retry or pick a different provider.
        """
        from deepagents_code.model_config import clear_caches
        from deepagents_code.widgets.codex_auth import CodexAuthScreen

        def _on_codex_done(signed_in: bool | None) -> None:
            clear_caches()
            if signed_in:
                self._dismiss_with_result((model_spec, provider))
                return
            self.call_after_refresh(self._update_display)

        self.app.push_screen(CodexAuthScreen(), _on_codex_done)

    async def action_set_default(self) -> None:
        """Toggle the highlighted model as the default.

        If the highlighted model is already the default, clears it.
        Otherwise sets it as the new default.
        """
        if not self._filtered_models or not self._option_widgets:
            return

        model_spec, _provider = self._filtered_models[self._selected_index]
        help_widget = self.query_one(".model-selector-help", Static)

        if model_spec == self._default_spec:
            # Already default — clear it
            if await asyncio.to_thread(clear_default_model):
                self._default_spec = None
                self.call_after_refresh(self._update_display)
                help_widget.update(Content.styled("Default cleared", "bold"))
                self.set_timer(3.0, self._restore_help_text)
            else:
                help_widget.update(
                    Content.styled(
                        "Failed to clear default",
                        f"bold {theme.get_theme_colors(self).error}",
                    )
                )
                self.set_timer(3.0, self._restore_help_text)
        elif await asyncio.to_thread(save_default_model, model_spec):
            self._default_spec = model_spec
            self.call_after_refresh(self._update_display)
            help_widget.update(
                Content.from_markup(
                    "[bold]Default set to $spec[/bold]", spec=model_spec
                )
            )
            self.set_timer(3.0, self._restore_help_text)
        else:
            help_widget.update(
                Content.styled(
                    "Failed to save default",
                    f"bold {theme.get_theme_colors(self).error}",
                )
            )
            self.set_timer(3.0, self._restore_help_text)

    def _restore_help_text(self) -> None:
        """Restore the default help text after a temporary message."""
        help_widget = self.query_one(".model-selector-help", Static)
        help_widget.update(self._help_text())

    async def action_toggle_recommended(self) -> None:
        """Toggle between the full model list and the recommended subset.

        Disabled while in `_curated` (onboarding) mode — that screen is
        already constrained to the recommended subset and the user should
        finish or skip onboarding rather than browse the full list.
        Preserves the highlighted model when it survives the toggle and
        falls back to the current/default/first model otherwise.
        """
        if self._curated or not self._loaded:
            return

        prev_spec: str | None = None
        if self._filtered_models and 0 <= self._selected_index < len(
            self._filtered_models
        ):
            prev_spec = self._filtered_models[self._selected_index][0]

        self._recommended_only = not self._recommended_only
        self._all_models = self._apply_subset(self._unfiltered_models)

        if self._filter_text.strip():
            self._update_filtered_list()
        else:
            self._filtered_models = list(self._all_models)
            self._selected_index = self._find_current_model_index()

        if prev_spec is not None:
            for i, (spec, _) in enumerate(self._filtered_models):
                if spec == prev_spec:
                    self._selected_index = i
                    break

        info = self.query_one("#model-selector-info", Static)
        info.update(self._info_line_content())

        await self._update_display()

    def action_cancel(self) -> None:
        """Cancel the selection."""
        self._dismiss_with_result(None)

    def _dismiss_with_result(self, result: tuple[str, str] | None) -> None:
        """Dismiss the selector and notify an optional direct result callback."""
        if self._result_callback is not None:
            self._result_callback(result)
        self.dismiss(result)
