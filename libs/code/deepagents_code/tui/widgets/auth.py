"""TUI screens for managing stored model-provider credentials.

`AuthPromptScreen` accepts an API key for a single provider, persists it via
`auth_store`, and is the sole place that deletes existing credentials (after
a `DeleteCredentialConfirmScreen` confirmation). `AuthManagerScreen` lists
known providers and routes the user into the prompt; it does not delete
directly. Both are reachable via the `/auth` slash command.

Security notes:

- Inputs are rendered with `password=True` so the key is never echoed to
    the terminal.
- This module never logs the key value, never includes it in `notify()`
    payloads, and never round-trips it through Rich markup. Callers that
    introduce new logging here must do the same.
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, NamedTuple
from urllib.parse import urlsplit

from textual.binding import Binding, BindingType
from textual.color import Color as TColor
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Input, OptionList, RadioButton, RadioSet, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click, MouseMove

    from deepagents_code.tui.widgets.codex_auth import CodexSignedInAction

from deepagents_code import auth_store, theme
from deepagents_code.auth_display import format_auth_badge
from deepagents_code.config import (
    LANGSMITH_EU_ENDPOINT,
    LANGSMITH_US_ENDPOINT,
    apply_stored_langsmith_auth,
    get_glyphs,
    is_ascii_mode,
    is_http_url,
    normalize_langsmith_endpoint,
)
from deepagents_code.model_config import (
    CODEX_PROVIDER,
    PROVIDER_API_KEY_ENV,
    PROVIDERS_DOCS_URL as _PROVIDERS_DOCS_URL,
    SERVICE_API_KEY_ENV,
    ModelConfig,
    ProviderAuthSource,
    ProviderAuthState,
    ProviderAuthStatus,
    clear_caches,
    get_available_models,
    get_base_url_env_var,
    get_base_url_env_vars,
    get_credential_env_var,
    get_default_base_url_env,
    get_provider_auth_status,
    get_service_auth_status,
    is_langsmith,
    is_service,
    resolved_env_var_name,
)
from deepagents_code.tui.widgets._links import open_style_link

logger = logging.getLogger(__name__)


CONFIGURATION_DOCS_URL = (
    "https://docs.langchain.com/oss/python/deepagents/code/configuration"
)


class Region(StrEnum):
    """The closed set of LangSmith region selections in the `/auth` prompt."""

    US = "us"
    EU = "eu"
    CUSTOM = "custom"


class _RegionSpec(NamedTuple):
    """One region row: its enum, radio widget id, SaaS endpoint, and label.

    The single source of truth for the region <-> radio-id <-> endpoint mapping,
    so `_REGION_BY_RADIO_ID`, `compose`'s radio buttons, `_region_for_endpoint`,
    and `_resolve_langsmith_endpoint` can't drift apart. `endpoint` is the
    canonical URL stored for a fixed-SaaS region (`""` for the US default, which
    clears the stored endpoint); `Region.CUSTOM` carries `""` here because the
    user supplies its endpoint at prompt time, so it is handled explicitly.
    """

    region: Region
    radio_id: str
    endpoint: str
    label: str


_REGION_SPECS: tuple[_RegionSpec, ...] = (
    _RegionSpec(Region.US, "auth-region-us", "", "United States (default)"),
    _RegionSpec(Region.EU, "auth-region-eu", LANGSMITH_EU_ENDPOINT, "Europe"),
    _RegionSpec(
        Region.CUSTOM, "auth-region-custom", "", "Custom (self-hosted / proxy)"
    ),
)

_REGION_BY_RADIO_ID: dict[str, Region] = {
    spec.radio_id: spec.region for spec in _REGION_SPECS
}
"""Map each region radio's widget id to its region, avoiding string-munging."""

_ENDPOINT_BY_REGION: dict[Region, str] = {
    spec.region: spec.endpoint for spec in _REGION_SPECS
}
"""Canonical endpoint stored for each fixed-SaaS region (Custom is dynamic)."""


class ResolvedEndpoint(NamedTuple):
    """Outcome of resolving the region selector to an endpoint to persist.

    `endpoint` is the canonical URL to store (empty for the US SaaS default);
    `error` is a user-facing message when a Custom URL is missing or malformed,
    in which case `endpoint` is empty and nothing should be saved.
    """

    endpoint: str
    error: str | None


def _region_for_endpoint(base_url: str) -> Region:
    """Map a stored LangSmith endpoint to its `/auth` region selection.

    Args:
        base_url: The stored endpoint, or an empty string for the SaaS default.

    Returns:
        `Region.US` (blank, or the canonical US SaaS URL that the CLI
            `--base-url us` shorthand stores), `Region.EU` (the EU SaaS URL), or
            `Region.CUSTOM` (any other endpoint, e.g. self-hosted).
    """
    if not base_url or base_url == LANGSMITH_US_ENDPOINT:
        return Region.US
    for spec in _REGION_SPECS:
        if spec.endpoint and base_url == spec.endpoint:
            return spec.region
    return Region.CUSTOM


PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "azure_openai": "Azure OpenAI",
    "baseten": "Baseten",
    "cohere": "Cohere",
    "deepseek": "DeepSeek",
    "fireworks": "Fireworks",
    "google_genai": "Google Gemini",
    "google_vertexai": "Google Vertex AI",
    "groq": "Groq",
    "huggingface": "Hugging Face",
    "ibm": "IBM watsonx",
    "langsmith": "LangSmith (tracing)",
    "litellm": "LiteLLM",
    "mistralai": "Mistral AI",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "openai_codex": "OpenAI Codex (ChatGPT login)",
    "openrouter": "OpenRouter",
    "perplexity": "Perplexity",
    "together": "Together AI",
    "xai": "xAI",
}


PROVIDER_SHORT_NAMES: dict[str, str] = {
    # Only providers whose `PROVIDER_DISPLAY_NAMES` label is too verbose for a
    # compact tag need an entry here; everything else falls back to the display
    # name, which is already short.
    "openai_codex": "OpenAI Codex",
}
"""Compact brand labels for space-constrained UI (e.g. the `/model` Recent tag).

Sparse companion to `PROVIDER_DISPLAY_NAMES`: an entry exists only when the full
display name carries a parenthetical qualifier that reads badly inside a tag
(e.g. `"OpenAI Codex (ChatGPT login)"`). Resolved via `provider_short_name`.
"""


PROVIDER_API_KEY_URLS: dict[str, str] = {
    "anthropic": "https://platform.claude.com/login?returnTo=%2Fsettings%2Fkeys",
    "baseten": "https://docs.baseten.co/organization/api-keys",
    "cohere": "https://dashboard.cohere.com/welcome/login?redirect_uri=%2Fapi-keys",
    "deepseek": "https://platform.deepseek.com/api_keys",
    "fireworks": "https://app.fireworks.ai/settings/users/api-keys",
    "google_genai": "https://aistudio.google.com/api-keys",
    "groq": "https://console.groq.com/keys",
    "huggingface": "https://huggingface.co/login?next=%2Fsettings%2Ftokens",
    "ibm": "https://cloud.ibm.com/iam/apikeys",
    "langsmith": "https://smith.langchain.com/settings",
    "litellm": "https://docs.litellm.ai/docs/proxy/virtual_keys",
    "mistralai": "https://console.mistral.ai/api-keys",
    "nvidia": "https://build.nvidia.com/settings/api-keys",
    "openai": "https://platform.openai.com/api-keys",
    "openrouter": "https://openrouter.ai/workspaces/default/keys",
    "perplexity": "https://www.perplexity.ai/settings/api",
    "tavily": "https://app.tavily.com",
    "together": "https://api.together.ai/settings/api-keys",
    "xai": "https://console.x.ai/team/default/api-keys",
}


def _is_safe_acquisition_url(url: str) -> bool:
    """Return whether `url` is safe to render as a clickable link.

    Built-in links are trusted, but `api_key_url` can come from user-owned
    config; restricting to `http`/`https` keeps a malformed or
    `javascript:`-scheme value from becoming a live hyperlink.

    Args:
        url: Candidate link target.

    Returns:
        `True` if the URL uses an `http` or `https` scheme.
    """
    return urlsplit(url).scheme in {"http", "https"}


def provider_display_name(provider: str, config: ModelConfig | None = None) -> str:
    """Return a human-readable provider label.

    Shared by the auth UI and the model selector so a provider is labeled
    identically in both. (The install prompt reuses the underlying
    `PROVIDER_DISPLAY_NAMES` map directly rather than this function, to avoid an
    event-loop config read, so a user-configured `display_name` won't surface
    there.)

    Resolution order: a configured `display_name`, then the built-in
    `PROVIDER_DISPLAY_NAMES` map, then a title-cased form of the provider key.

    Args:
        provider: Provider config key.
        config: Parsed model config, if already loaded by the caller.

    Returns:
        Configured display name, built-in display name, or title-cased provider key.
    """
    model_config = config or ModelConfig.load()
    return model_config.get_provider_display_name(
        provider
    ) or PROVIDER_DISPLAY_NAMES.get(provider, provider.replace("_", " ").title())


def provider_short_name(provider: str, config: ModelConfig | None = None) -> str:
    """Return a compact brand label for a provider.

    For space-constrained UI (e.g. the `/model` Recent tag). Resolution order:
    a configured `short_name`, then the built-in `PROVIDER_SHORT_NAMES` map,
    then the full `provider_display_name` (which is already short for providers
    without a parenthetical qualifier).

    Args:
        provider: Provider config key.
        config: Parsed model config, if already loaded by the caller.

    Returns:
        Compact brand label, falling back to the display name when none is set.
    """
    model_config = config or ModelConfig.load()
    return (
        model_config.get_provider_short_name(provider)
        or PROVIDER_SHORT_NAMES.get(provider)
        or provider_display_name(provider, model_config)
    )


def _auth_status_for(provider: str) -> ProviderAuthStatus:
    """Resolve the credential readiness of a provider or non-model service.

    Routes services (e.g. Tavily) and model providers to their respective
    status helpers. Each call reads the credential file, so callers that need
    the status more than once should resolve it here a single time and reuse
    the result.

    Args:
        provider: Provider or service config key.

    Returns:
        The auth status used for both ordering and badge rendering.
    """
    if is_service(provider):
        return get_service_auth_status(provider)
    return get_provider_auth_status(provider)


class AuthResult(StrEnum):
    """Outcome of an `AuthPromptScreen` interaction.

    The three outcomes need to stay distinguishable because callers in the
    recovery path retry the original failing operation only on `SAVED` —
    retrying after `DELETED` would loop into the same missing-credentials
    error indefinitely.
    """

    SAVED = "saved"
    """User pasted a key and it was persisted."""

    DELETED = "deleted"
    """User cleared the existing stored key. No retry should follow."""

    CANCELLED = "cancelled"
    """User dismissed the prompt without saving."""


class AuthConfirmScreen(ModalScreen[bool]):
    """Confirm before launching an authentication flow for a model.

    A provider-agnostic gate shown when a selected model needs credentials
    that aren't detected, and starting the auth flow is disruptive enough
    that the user should opt in first (e.g. an OAuth flow that launches a
    browser and a multi-minute loopback wait). The caller supplies all copy
    so the screen carries no provider assumptions; currently only the
    `openai_codex` model-switcher path uses it.

    Dismissal values:

    - `True`: proceed to the auth flow.
    - `False`: go back without authenticating (also the outcome of Esc).
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Continue", show=False, priority=True),
        Binding("escape", "cancel", "Back", show=False, priority=True),
    ]

    CSS = """
    AuthConfirmScreen {
        align: center middle;
    }

    AuthConfirmScreen > Vertical {
        width: 64;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AuthConfirmScreen .auth-confirm-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AuthConfirmScreen .auth-confirm-body {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    AuthConfirmScreen .auth-confirm-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        *,
        title: str,
        body: str | Content,
        help_text: str = "Enter to continue, Esc to go back",
    ) -> None:
        """Initialize the prompt.

        Args:
            title: Heading shown at the top of the dialog.
            body: Explanatory copy. Pass a `Content` for inline styling, or a
                plain string for unstyled text.
            help_text: Key-hint line shown at the bottom.
        """
        super().__init__()
        self._title = title
        self._body = body
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Title, body, and key-hint widgets parented inside a `Vertical`.
        """
        with Vertical():
            yield Static(self._title, classes="auth-confirm-title", markup=False)
            yield Static(self._body, classes="auth-confirm-body", markup=False)
            yield Static(self._help_text, classes="auth-confirm-help", markup=False)

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def action_confirm(self) -> None:
        """Dismiss with `True` to proceed to the auth flow."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Dismiss with `False` to go back without authenticating.

        The method name must stay `cancel`: the app owns a priority `escape`
        binding that, for an active `ModalScreen`, dispatches to
        `action_cancel` if present and otherwise falls through to
        `dismiss(None)`. Renaming this would silently regress Esc to a
        `None` dismiss instead of an explicit "go back".
        """
        self.dismiss(False)


class DeleteCredentialConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay shown before clearing a stored credential.

    Patterned on `DeleteThreadConfirmScreen` so the destructive prompt feels
    consistent across the app. Always dismisses with `True` on confirm or
    `False` on cancel; the caller does the actual delete.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "confirm", "Confirm", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
    ]

    CSS = """
    DeleteCredentialConfirmScreen {
        align: center middle;
    }

    DeleteCredentialConfirmScreen > Vertical {
        width: 56;
        height: auto;
        background: $surface;
        border: solid red;
        padding: 1 2;
    }

    DeleteCredentialConfirmScreen .auth-confirm-text {
        text-align: center;
        margin-bottom: 1;
    }

    DeleteCredentialConfirmScreen .auth-confirm-help {
        text-align: center;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, provider: str) -> None:
        """Initialize the confirmation modal.

        Args:
            provider: Provider whose stored credential is about to be cleared.
        """
        super().__init__()
        self._provider = provider

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog.

        Yields:
            Widgets for the delete confirmation prompt.
        """
        with Vertical():
            yield Static(
                Content.from_markup(
                    "Delete stored API key for [bold]$provider[/bold]?",
                    provider=self._provider,
                ),
                classes="auth-confirm-text",
            )
            yield Static(
                "Enter to confirm, Esc to cancel",
                classes="auth-confirm-help",
            )

    def action_confirm(self) -> None:
        """Confirm deletion."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel deletion."""
        self.dismiss(False)


class AuthPromptScreen(ModalScreen[AuthResult]):
    """Modal that captures and persists an API key for one provider.

    Dismissal values are members of `AuthResult` so callers in the recovery
    path can distinguish "user just saved a key — retry the failed
    operation" from "user just cleared their key — don't retry, that would
    loop into the same error" from "user cancelled — leave state alone".
    """

    AUTO_FOCUS = "#auth-prompt-input"

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("f2", "toggle_advanced", "Advanced", show=False, priority=True),
        Binding("ctrl+d", "delete_stored", "Delete stored", show=False, priority=True),
    ]

    CSS = """
    AuthPromptScreen {
        align: center middle;
    }

    AuthPromptScreen > Vertical {
        width: 72;
        max-width: 90%;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AuthPromptScreen .auth-prompt-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-copy {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-status {
        height: auto;
        color: $text;
        background: $background;
        padding: 0 1;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-instructions {
        height: auto;
        color: $text;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-advanced-toggle {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    AuthPromptScreen #auth-prompt-base-url-label {
        text-align: center;
    }

    AuthPromptScreen .auth-prompt-meta {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    AuthPromptScreen #auth-prompt-project-hint {
        margin-top: 1;
    }

    AuthPromptScreen #auth-prompt-region {
        height: auto;
        width: 100%;
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    AuthPromptScreen #auth-prompt-region:focus-within {
        border: solid $primary;
    }

    AuthPromptScreen #auth-prompt-input,
    AuthPromptScreen #auth-prompt-base-url {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    AuthPromptScreen #auth-prompt-input:focus,
    AuthPromptScreen #auth-prompt-base-url:focus {
        border: solid $primary;
    }

    AuthPromptScreen .auth-prompt-error {
        height: auto;
        color: $error;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
    }
    """

    def __init__(
        self,
        provider: str,
        env_var: str | None,
        *,
        reason: str | None = None,
        allow_empty_submit: bool = False,
        input_placeholder: str | None = None,
        submit_label: str | None = None,
    ) -> None:
        """Initialize the prompt for `provider`.

        Args:
            provider: Provider name (e.g., `"anthropic"`).
            env_var: Canonical env var the SDK reads, shown as helper text.
                May be `None` for providers that don't use one of the
                hardcoded env-var bindings (rare; the prompt still works).
            reason: Optional context, e.g.,
                `"Required to use anthropic:claude-opus-4-8"`.
            allow_empty_submit: Whether pressing Enter on an empty key dismisses
                with `AuthResult.CANCELLED` instead of showing a validation error.
            input_placeholder: Optional placeholder override for the key input.
            submit_label: Optional help-label override for the Enter action.
        """
        super().__init__()
        self._provider = provider
        self._env_var = env_var
        self._reason = reason
        self._allow_empty_submit = allow_empty_submit
        self._input_placeholder = input_placeholder
        self._submit_label = submit_label
        # LangSmith is configured as a tracing service: it carries an optional
        # project name and an endpoint chosen from a region selector (US/EU SaaS
        # or a custom self-hosted URL), and saving a key turns tracing on.
        self._is_langsmith = is_langsmith(provider)
        # Resolve the current credential source and probe the store, but never
        # let a corrupt `auth.json`/config crash the screen at construction
        # time — Textual would propagate the exception before the modal mounts.
        # Treat unreadable as "no env source" / "no existing key" and surface a
        # one-line warning at compose time. The status is consumed only
        # cosmetically (title prefix, env note), so a MISSING fallback is safe.
        # The config is loaded once here so compose-time helpers can read the
        # cached instance instead of reloading outside this crash-safety guard.
        try:
            self._config = ModelConfig.load()
            self._auth_status = _auth_status_for(provider)
            self._has_existing = auth_store.get_stored_key(provider) is not None
            self._existing_base_url = auth_store.get_stored_base_url(provider) or ""
            self._existing_project = auth_store.get_stored_project(provider) or ""
            self._advanced_visible = bool(
                self._existing_base_url or self._existing_project
            )
            self._region: Region = _region_for_endpoint(self._existing_base_url)
            self._store_warning: str | None = None
        except RuntimeError as exc:
            logger.warning(
                "Could not read stored credentials for %s: %s", provider, exc
            )
            self._config = ModelConfig()
            self._auth_status = ProviderAuthStatus(
                state=ProviderAuthState.MISSING, provider=provider
            )
            self._has_existing = False
            self._existing_base_url = ""
            self._existing_project = ""
            self._advanced_visible = False
            self._region = Region.US
            self._store_warning = (
                f"Credential file is unreadable ({exc}). Saving here will overwrite it."
            )
        # Surface when an environment endpoint is set: at startup an existing
        # `LANGSMITH_ENDPOINT`/`LANGCHAIN_ENDPOINT` takes precedence over the
        # stored region, so without this note the radio could show one region
        # while traces route somewhere else. (The note fires on presence, not on
        # divergence — it may show even when the env value matches the stored
        # region.) Saving here applies the selection (the save path replaces the
        # env value), so word the note around that.
        self._endpoint_env_notice: str | None = None
        if self._is_langsmith and any(
            os.environ.get(var) for var in ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
        ):
            self._endpoint_env_notice = (
                "An endpoint is set in your environment "
                "(LANGSMITH_ENDPOINT/LANGCHAIN_ENDPOINT) and takes precedence at "
                "startup; saving a region here applies your selection instead."
            )

    def compose(self) -> ComposeResult:
        """Compose the prompt.

        Yields:
            Widgets that make up the auth prompt modal.
        """
        glyphs = get_glyphs()
        provider_label = provider_display_name(self._provider, self._config)
        with Vertical():
            # Tag the title with `(stored)` so the user knows a replacement
            # (or the `Ctrl+D delete` affordance shown in the help line) is
            # what's about to happen — both are gated on `_has_existing`.
            resolved_from_env = self._auth_status.source is ProviderAuthSource.ENV
            scoped_env_var = None
            if self._auth_status.env_var:
                candidate = resolved_env_var_name(self._auth_status.env_var)
                if candidate.startswith("DEEPAGENTS_CODE_") and os.environ.get(
                    candidate
                ):
                    scoped_env_var = candidate
            active_env_var = scoped_env_var or (
                resolved_env_var_name(self._auth_status.env_var)
                if resolved_from_env and self._auth_status.env_var
                else None
            )
            if active_env_var and active_env_var.startswith("DEEPAGENTS_CODE_"):
                title_prefix = f"{glyphs.warning} "
            else:
                title_prefix = f"{glyphs.checkmark} " if resolved_from_env else ""
            if self._has_existing:
                title = Content.assemble(
                    title_prefix,
                    Content.from_markup(
                        "Replace key for [bold]$provider[/bold] [dim](stored)[/dim]",
                        provider=provider_label,
                    ),
                )
            elif resolved_from_env:
                title = Content.assemble(
                    title_prefix,
                    Content.from_markup(
                        "Replace key for [bold]$provider[/bold]",
                        provider=provider_label,
                    ),
                )
            else:
                title = Content.assemble(
                    title_prefix,
                    Content.from_markup(
                        "API key for [bold]$provider[/bold]",
                        provider=provider_label,
                    ),
                )
            yield Static(title, classes="auth-prompt-title")
            if active_env_var:
                env_var = active_env_var
                is_scoped_env = env_var.startswith("DEEPAGENTS_CODE_")
                env_status_style = "$warning" if is_scoped_env else "$success"
                env_note = (
                    "This scoped env var takes priority. A saved key will be used "
                    f"only when {env_var} is unset."
                    if is_scoped_env
                    else ("Paste a key below to use a different key for dcode.")
                )
                yield Static(
                    Content.assemble(
                        (
                            "Current key is set from environment variable ",
                            env_status_style,
                        ),
                        (env_var, f"bold {env_status_style}"),
                        (".", env_status_style),
                        "\n",
                        (env_note, "italic $text-muted"),
                    ),
                    classes="auth-prompt-status",
                    id="auth-prompt-env-status",
                )
            if self._reason:
                yield Static(
                    Content.from_markup("$reason", reason=self._reason),
                    classes="auth-prompt-copy",
                )
            yield Static(
                self._build_key_instructions(),
                classes="auth-prompt-instructions",
                id="auth-prompt-key-instructions",
            )
            if self._store_warning:
                yield Static(
                    Content.from_markup("$msg", msg=self._store_warning),
                    classes="auth-prompt-error",
                )
            yield Input(
                placeholder=self._input_placeholder
                or (
                    "Paste a new key to replace the stored one"
                    if self._has_existing
                    else "Paste your API key"
                ),
                password=True,
                id="auth-prompt-input",
            )
            storage_note: Content | None
            if self._is_langsmith:
                storage_note = Content.from_markup(
                    "dcode stores the above key locally and turns on "
                    "LangSmith tracing. To pause tracing without removing the key, "
                    "set [bold]DEEPAGENTS_CODE_LANGSMITH_TRACING=false[/bold]."
                )
            elif is_service(self._provider):
                # Services (e.g. Tavily) skip the storage note: the title and
                # reason copy already say what the key is for, so it only adds
                # redundant copy here.
                storage_note = None
            else:
                storage_note = Content.from_markup(
                    "dcode stores the above key locally and uses it "
                    "when you select [bold]$provider[/bold] models.",
                    provider=provider_label,
                )
            if storage_note is not None:
                yield Static(
                    storage_note,
                    classes="auth-prompt-meta",
                    id="auth-prompt-storage-note",
                )
            yield Static(
                self._build_advanced_toggle_label(),
                classes="auth-prompt-advanced-toggle",
                id="auth-prompt-advanced-toggle",
            )
            if self._env_var:
                key_meta = Static(
                    Content.assemble(
                        "Alternatively, environment variables can be used in place "
                        "of the key stored above. Set ",
                        (f"DEEPAGENTS_CODE_{self._env_var}", TStyle(bold=True)),
                        " for a dcode-only key; it has the highest priority. Set ",
                        (self._env_var, TStyle(bold=True)),
                        " to share a key with other provider SDK tools; it is used "
                        "only when no scoped or stored key exists. ",
                        (
                            "Configuration docs",
                            self._link_style(CONFIGURATION_DOCS_URL),
                        ),
                        ".",
                    ),
                    classes="auth-prompt-meta",
                    id="auth-prompt-key-meta",
                )
                key_meta.display = self._advanced_visible
                yield key_meta
            if self._is_langsmith:
                if self._endpoint_env_notice:
                    yield Static(
                        Content.from_markup("$msg", msg=self._endpoint_env_notice),
                        classes="auth-prompt-meta",
                        id="auth-prompt-endpoint-env-notice",
                    )
                region_label = Static(
                    Content.from_markup("[bold]LangSmith region[/bold]"),
                    classes="auth-prompt-meta",
                    id="auth-prompt-region-label",
                )
                region_label.display = self._advanced_visible
                yield region_label
                region_set = RadioSet(
                    *(
                        RadioButton(
                            spec.label,
                            value=self._region == spec.region,
                            id=spec.radio_id,
                        )
                        for spec in _REGION_SPECS
                    ),
                    id="auth-prompt-region",
                )
                region_set.display = self._advanced_visible
                yield region_set
                custom_visible = (
                    self._advanced_visible and self._region == Region.CUSTOM
                )
                base_url_input = Input(
                    value=(
                        self._existing_base_url if self._region == Region.CUSTOM else ""
                    ),
                    placeholder="https://my-langsmith.example.com",
                    id="auth-prompt-base-url",
                )
                base_url_input.display = custom_visible
                yield base_url_input
                base_url_hint_widget = Static(
                    Content.from_markup(
                        "Point tracing at a self-hosted or proxied LangSmith. "
                        "Sets [bold]LANGSMITH_ENDPOINT[/bold]; must be an "
                        "http(s) URL."
                    ),
                    classes="auth-prompt-meta",
                    id="auth-prompt-base-url-hint",
                )
                base_url_hint_widget.display = custom_visible
                yield base_url_hint_widget
                project_label = Static(
                    Content.from_markup("[bold]Project name[/bold]"),
                    classes="auth-prompt-meta",
                    id="auth-prompt-project-label",
                )
                project_label.display = self._advanced_visible
                yield project_label
                project_input = Input(
                    value=self._existing_project,
                    placeholder="LANGSMITH_PROJECT (default: deepagents-code)",
                    id="auth-prompt-project",
                )
                project_input.display = self._advanced_visible
                yield project_input
                project_hint_widget = Static(
                    Content.from_markup(
                        "Route agent traces to this LangSmith project. "
                        "Leave blank to use the default [bold]deepagents-code[/bold]."
                    ),
                    classes="auth-prompt-meta",
                    id="auth-prompt-project-hint",
                )
                project_hint_widget.display = self._advanced_visible
                yield project_hint_widget
            else:
                base_url_label = Static(
                    Content.from_markup("[bold]Base URL override[/bold]"),
                    classes="auth-prompt-meta",
                    id="auth-prompt-base-url-label",
                )
                base_url_label.display = self._advanced_visible
                yield base_url_label
                base_url_input = Input(
                    value=self._existing_base_url,
                    placeholder="Base URL",
                    id="auth-prompt-base-url",
                )
                base_url_input.display = self._advanced_visible
                yield base_url_input
                base_url_hint_widget = Static(
                    self._build_base_url_hint(),
                    classes="auth-prompt-meta",
                    id="auth-prompt-base-url-hint",
                )
                base_url_hint_widget.display = self._advanced_visible
                yield base_url_hint_widget
            yield Static("", classes="auth-prompt-error", id="auth-prompt-error")
            save_label = self._submit_label or (
                "Enter replace" if self._has_existing else "Enter save"
            )
            help_parts = [f"{save_label} {glyphs.bullet} Esc cancel", "F2 advanced"]
            if self._has_existing:
                help_parts.append("Ctrl+D delete stored")
            yield Static(
                f" {glyphs.bullet} ".join(help_parts),
                classes="auth-prompt-help",
            )

    def _link_style(self, url: str) -> TStyle:
        """Return a theme-aware style for inline modal links.

        Args:
            url: Link target.

        Returns:
            Textual style that opens `url` when clicked.
        """
        colors = theme.get_theme_colors(self)
        if self.app.theme in {"ansi-dark", "ansi-light"}:
            return TStyle(bold=True, underline=True, link=url)
        return TStyle(foreground=TColor.parse(colors.primary), underline=True, link=url)

    def _build_key_instructions(self) -> Content:
        """Build provider-specific API-key acquisition guidance.

        Returns:
            Content shown before the API-key input. May append muted notices: a
                provider-specific caveat (e.g. Anthropic subscription plans are
                unsupported) and/or a warning that a user-configured `api_key_url`
                was rejected for using an unsupported URL scheme.
        """
        config = self._config
        configured_url = config.get_provider_api_key_url(self._provider)
        rejected_url = False
        if configured_url and not _is_safe_acquisition_url(configured_url):
            logger.warning(
                "Ignoring api_key_url for %s: unsupported URL scheme", self._provider
            )
            configured_url = None
            rejected_url = True
        url = configured_url or PROVIDER_API_KEY_URLS.get(
            self._provider, _PROVIDERS_DOCS_URL
        )
        provider = provider_display_name(self._provider, config)
        label = (
            f"{provider} key page"
            if configured_url or self._provider in PROVIDER_API_KEY_URLS
            else f"{provider} setup docs"
        )
        if self._provider == "azure_openai":
            instructions = Content.assemble(
                "Find your key in your Azure OpenAI resource's "
                "Keys and Endpoint page, then paste it below. ",
                (label, self._link_style(url)),
            )
        elif self._provider == "openai":
            instructions = Content.assemble(
                f"Sign in to {provider}, create or copy an API key, then "
                "paste it below. Minimum permissions needed: "
                "under Model capabilities, grant Write access to Responses "
                "(/v1/responses). For older models, you may also need "
                "Request access to Chat completions (/v1/chat/completions). ",
                (label, self._link_style(url)),
            )
        elif self._provider == "anthropic":
            instructions = Content.assemble(
                f"Sign in to {provider}, create or copy an API key, "
                "then paste it below. ",
                (label, self._link_style(url)),
                "\n",
                (
                    (
                        "Subscription plans (Claude Pro/Max, Claude Code) cannot "
                        "be used for Anthropic calls in dcode. Only a "
                        "standard API key with pay-as-you-go billing works here."
                    ),
                    "italic $text-muted",
                ),
            )
        else:
            instructions = Content.assemble(
                f"Sign in to {provider}, create or copy an API key, "
                "then paste it below. ",
                (label, self._link_style(url)),
            )
        if rejected_url:
            notice = (
                "Your configured api_key_url was ignored (unsupported URL "
                "scheme); showing the default link instead."
            )
            instructions = Content.assemble(
                instructions,
                "\n",
                (notice, "italic $text-muted"),
            )
        return instructions

    def _build_advanced_toggle_label(self) -> str:
        """Build the disclosure-row label for advanced settings.

        Returns:
            Toggle label reflecting the current expanded state.
        """
        glyphs = get_glyphs()
        marker = (
            glyphs.disclosure_expanded
            if self._advanced_visible
            else glyphs.disclosure_collapsed
        )
        return f"{marker} Advanced (F2)"

    def _build_base_url_hint(self) -> Content:
        """Build the optional base-URL hint shown inside Advanced.

        Returns:
            Content describing blank behavior and env-var precedence.
        """
        surviving_base_url_env = get_default_base_url_env(self._provider)
        endpoint_envs = get_base_url_env_vars(self._provider)
        env_order = ", then ".join(
            item for env in endpoint_envs for item in (f"DEEPAGENTS_CODE_{env}", env)
        )
        if surviving_base_url_env and env_order:
            return Content.from_markup(
                "Override the provider endpoint for this stored key. "
                "Leave blank to use [bold]$prefixed[/bold].\n"
                "Env override order: [bold]$order[/bold].",
                prefixed=surviving_base_url_env,
                order=env_order,
            )
        endpoint_env = get_base_url_env_var(self._provider)
        if endpoint_env and env_order:
            return Content.from_markup(
                "Override the provider endpoint for this stored key. "
                "Leave blank to use the provider default.\n"
                "Env override order: [bold]$order[/bold].",
                order=env_order,
            )
        return Content.from_markup(
            "Override the provider endpoint for this stored key. "
            "Leave blank to use the provider default."
        )

    def action_toggle_advanced(self) -> None:
        """Show or hide optional endpoint and env-var details.

        Restores focus to the key input when collapsing so keyboard entry
        resumes on the field the user most likely wants.
        """
        self._advanced_visible = not self._advanced_visible
        for selector in (
            "#auth-prompt-key-meta",
            "#auth-prompt-region-label",
            "#auth-prompt-region",
            "#auth-prompt-base-url-label",
            "#auth-prompt-base-url",
            "#auth-prompt-base-url-hint",
            "#auth-prompt-project-label",
            "#auth-prompt-project",
            "#auth-prompt-project-hint",
        ):
            for widget in self.query(selector):
                widget.display = self._advanced_visible
        # For LangSmith the custom URL field only shows under the Custom region,
        # so re-apply its region-gated visibility after the blanket toggle above.
        if self._is_langsmith:
            self._refresh_langsmith_custom_visibility()
        self.query_one("#auth-prompt-advanced-toggle", Static).update(
            self._build_advanced_toggle_label()
        )
        if not self._advanced_visible:
            self.query_one("#auth-prompt-input", Input).focus()

    def _refresh_langsmith_custom_visibility(self) -> None:
        """Show the custom URL field only when Advanced is open and region is Custom."""
        custom_visible = self._advanced_visible and self._region == Region.CUSTOM
        for selector in ("#auth-prompt-base-url", "#auth-prompt-base-url-hint"):
            for widget in self.query(selector):
                widget.display = custom_visible

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Track the chosen LangSmith region and reveal the custom URL field."""
        event.stop()
        region = _REGION_BY_RADIO_ID.get(event.pressed.id or "")
        if region is None:
            # An unknown id (e.g. a renamed radio) must not silently degrade to a
            # region — leave the current selection unchanged rather than guess.
            return
        self._region = region
        self._refresh_langsmith_custom_visibility()
        if self._region == Region.CUSTOM:
            self.query_one("#auth-prompt-base-url", Input).focus()

    def on_click(self, event: Click) -> None:
        """Open style-embedded hyperlinks or toggle Advanced."""
        widget = event.widget
        if (
            widget is not None
            and widget.id == "auth-prompt-advanced-toggle"
            and not event.style.link
        ):
            self.action_toggle_advanced()
            event.stop()
            return
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer over links and the clickable Advanced row."""
        widget = event.widget
        self.styles.pointer = (
            "pointer"
            if event.style.link
            or (widget is not None and widget.id == "auth-prompt-advanced-toggle")
            else "default"
        )

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the prompt."""
        self.styles.pointer = "default"

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def _resolve_langsmith_endpoint(self) -> ResolvedEndpoint:
        """Resolve the endpoint to store from the LangSmith region selector.

        Returns:
            A `ResolvedEndpoint`: the canonical endpoint to persist (empty for
                the US SaaS default), and a non-`None` error when a custom URL is
                missing or malformed (so an explicit `Custom` selection never
                silently routes the key to the US SaaS default, and the key is
                never paired with a non-http(s) endpoint).
        """
        if self._region != Region.CUSTOM:
            # US and EU resolve to their fixed SaaS endpoints from the table
            # (US -> "" clears the stored endpoint back to the SaaS default).
            return ResolvedEndpoint(_ENDPOINT_BY_REGION[self._region], None)
        raw = self.query_one("#auth-prompt-base-url", Input).value.strip()
        if not raw:
            # `Custom` is a deliberate non-default choice; a blank field must not
            # silently fall back to the US SaaS default (which would reroute the
            # key and traces to the very endpoint a self-hosted user is avoiding).
            missing_url_error = (
                "Custom endpoint URL is required (or choose United States/Europe)."
            )
            return ResolvedEndpoint("", missing_url_error)
        endpoint = normalize_langsmith_endpoint(raw)
        if not is_http_url(endpoint):
            return ResolvedEndpoint("", "Custom endpoint must be an http(s) URL.")
        return ResolvedEndpoint(endpoint, None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Validate, persist, and dismiss.

        Reads both fields regardless of which one was submitted, so pressing
        Enter in either the key or the secondary input (base URL, or the
        LangSmith project name) saves the pair.
        """
        event.stop()
        cleaned = self.query_one("#auth-prompt-input", Input).value.strip()
        if self._is_langsmith:
            project = self.query_one("#auth-prompt-project", Input).value.strip()
            resolved = self._resolve_langsmith_endpoint()
            if resolved.error:
                self._show_error(resolved.error)
                return
            base_url = resolved.endpoint
        else:
            base_url = self.query_one("#auth-prompt-base-url", Input).value.strip()
            # Match the CLI `--base-url` guard: never pair the key with a
            # non-http(s) endpoint. Validate only a *changed* value, though: the
            # field is pre-filled with the stored base URL, so rotating just the
            # key must not be blocked by a legacy non-http(s) endpoint (the CLI
            # never validated base URLs before this feature). A new or edited
            # value must still be http(s).
            if (
                base_url
                and base_url != self._existing_base_url
                and not is_http_url(base_url)
            ):
                self._show_error("Base URL must be an http(s) URL.")
                return
            project = ""
        if not cleaned:
            if self._allow_empty_submit:
                # Optional prompts (e.g. the Tavily onboarding step) treat an
                # empty submit as an intentional skip. We deliberately reuse
                # `CANCELLED` rather than add a `SKIPPED` outcome: every caller
                # that allows empty submit wants identical "did not save"
                # handling for skip and Escape, so the distinction would be
                # dead weight. Revisit if a caller ever needs to tell a
                # deliberate decline from an accidental dismissal.
                self.dismiss(AuthResult.CANCELLED)
                return
            self._show_error("API key cannot be empty.")
            return
        try:
            outcome = auth_store.set_stored_key(
                self._provider,
                cleaned,
                base_url=base_url or None,
                project=project or None,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            # `auth_store` exception messages never include the secret value,
            # but the path can include user-controlled `DEFAULT_STATE_DIR`
            # bytes — render via `Content.from_markup` so a `[` in the path
            # can't break Textual's markup pipeline.
            logger.warning(
                "Failed to persist credential for %s: %s", self._provider, exc
            )
            self._show_error("Could not save credential: $exc", exc=str(exc))
            return
        for warning in outcome.warnings:
            # chmod failures are security regressions the user must see —
            # `logger.warning` alone is invisible inside a Textual session.
            self.app.notify(warning, severity="warning", markup=False)
        if self._is_langsmith:
            apply_stored_langsmith_auth(replace_project=True)
        clear_caches()
        self.dismiss(AuthResult.SAVED)

    def action_cancel(self) -> None:
        """Dismiss without saving."""
        self.dismiss(AuthResult.CANCELLED)

    def action_delete_stored(self) -> None:
        """Open the delete-confirmation overlay, or quit when nothing is stored.

        Ctrl+D deletes a stored credential, but its `priority` binding also
        intercepts the app-level Ctrl+D=quit. When there's no credential to
        delete, fall through to quit rather than swallowing the key (mirroring
        the thread selector). `app.exit()` is used instead of `dismiss()`, which
        would just close the modal silently and re-swallow the key.
        """
        if not self._has_existing:
            self.app.exit()
            return
        self.app.push_screen(
            DeleteCredentialConfirmScreen(self._provider),
            self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirmed: bool | None) -> None:
        """Handle the result of the confirmation overlay.

        Args:
            confirmed: `True` if the user pressed Enter, `False` on Esc.
        """
        if not confirmed:
            return
        try:
            removed = auth_store.delete_stored_key(self._provider)
        except RuntimeError as exc:
            logger.warning(
                "Failed to delete credential for %s: %s", self._provider, exc
            )
            self._show_error("Could not delete credential: $exc", exc=str(exc))
            return
        if not removed:
            # The entry was gone — likely a concurrent delete from another
            # app instance. Surface that fact so "delete" UX doesn't lie when
            # nothing actually happened on disk.
            self.app.notify(
                f"No stored credential for {self._provider} — already removed.",
                severity="information",
                markup=False,
            )
        clear_caches()
        self.dismiss(AuthResult.DELETED)

    def _show_error(self, template: str, /, **substitutions: str) -> None:
        """Render `template` via markup substitution in the inline error slot.

        Args:
            template: Markup template (e.g. `"Could not save: $exc"`).
            **substitutions: `$name` substitution values; Textual escapes them.
        """
        error = self.query_one("#auth-prompt-error", Static)
        error.update(Content.from_markup(template, **substitutions))


class AuthManagerScreen(ModalScreen[None]):
    """Modal that lists configured providers and lets the user manage keys.

    Reachable via the `/auth` slash command. Always dismisses with `None`;
    state changes are persisted by `AuthPromptScreen` and reflected by
    re-rendering the option list when this screen is reopened or after a
    save/delete completes.

    Well-known providers whose integration package isn't installed yet are
    surfaced greyed-out so they stay discoverable. Selecting one routes
    through an install confirmation: on confirm the screen records the extra on
    `pending_install_extra` and dismisses so the app can install it (mirroring
    the model selector's install-on-select flow) and reopen the manager.
    """

    class CredentialSaved(Message):
        """Posted when a key prompt successfully persists credentials."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Close", show=False, priority=True),
        Binding("tab", "cursor_down", "Next", show=False, priority=True),
        Binding("shift+tab", "cursor_up", "Previous", show=False, priority=True),
    ]

    CSS = """
    AuthManagerScreen {
        align: center middle;
    }

    AuthManagerScreen > Vertical {
        width: 76;
        max-width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }

    AuthManagerScreen .auth-manager-title {
        text-style: bold;
        color: $primary;
        text-align: center;
        margin-bottom: 1;
    }

    AuthManagerScreen .auth-manager-copy {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    /* `1fr` + `min-height` keeps the option list from pushing the footer
    off-screen on short terminals: the list shrinks (and starts scrolling)
    before the footer is hidden. */
    AuthManagerScreen OptionList {
        height: 1fr;
        min-height: 3;
        background: $background;
    }

    AuthManagerScreen .auth-manager-warning {
        height: auto;
        color: $warning;
        margin-bottom: 1;
    }

    AuthManagerScreen .auth-manager-help {
        height: auto;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self, *, initial_provider: str | None = None) -> None:
        """Initialize the manager with an empty install-on-select registry.

        Args:
            initial_provider: Provider whose row should start highlighted —
                set when reopening after an install-on-select so the cursor
                lands on the just-installed provider ready for a key, rather
                than resetting to the top of the list.
        """
        super().__init__()
        # Uninstalled known providers mapped to the extra that installs them,
        # populated each time the option list is built. Selecting one routes
        # to the install confirmation instead of the key prompt.
        self._install_extras: dict[str, str] = {}
        # Set when the user confirms installing a provider's extra; the app
        # reads these off the screen after dismissal to install then reopen
        # the manager with the just-installed provider highlighted.
        self.pending_install_extra: str | None = None
        self.pending_install_provider: str | None = None
        self._initial_provider = initial_provider

    def compose(self) -> ComposeResult:
        """Compose the manager.

        Yields:
            Widgets for the manager listing.
        """
        glyphs = get_glyphs()
        options, store_warning = self._build_options_with_warning()
        with Vertical():
            yield Static("Manage API keys", classes="auth-manager-title")
            yield Static(self._build_description(), classes="auth-manager-copy")
            if store_warning:
                # Surface auth.json corruption directly — `_build_options`
                # falling back silently used to make a corrupt file look
                # identical to "no keys stored".
                yield Static(
                    Content.from_markup("$msg", msg=store_warning),
                    classes="auth-manager-warning",
                )
            yield OptionList(*options, id="auth-manager-options")
            yield Static(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab/Shift+Tab "
                f"navigate {glyphs.bullet} Enter add/replace/delete/install "
                f"{glyphs.bullet} Esc close",
                classes="auth-manager-help",
            )

    def _build_description(self) -> Content:
        """Build the description line with an inline docs hyperlink.

        Returns:
            Description content. Themes other than the ANSI palette render
            the link in the primary color so it reads as clickable; ANSI
            users get a bold-only treatment that still reaches the
            terminal's link handler via `Style(link=...)`.
        """
        colors = theme.get_theme_colors(self)
        ansi = self.app.theme in {"ansi-dark", "ansi-light"}
        link_style: str | TStyle = (
            TStyle(bold=True, link=_PROVIDERS_DOCS_URL)
            if ansi
            else TStyle(
                foreground=TColor.parse(colors.primary),
                link=_PROVIDERS_DOCS_URL,
            )
        )
        return Content.assemble(
            "Lists installed model providers, services like web search, and any "
            "providers you've configured in ~/.deepagents/config.toml. Greyed-out "
            "providers aren't installed yet — select one to install it. ",
            ("Docs", link_style),
        )

    def on_mount(self) -> None:
        """Apply ASCII border and highlight the initial provider when set."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)
        self._highlight_initial_provider()

    def _highlight_initial_provider(self) -> None:
        """Move the cursor to `initial_provider`'s row if it is listed.

        Used when the manager reopens after an install-on-select so the cursor
        lands on the just-installed provider (ready for a key) instead of
        resetting to the top of the list.
        """
        if self._initial_provider is None:
            return
        option_list = self.query_one("#auth-manager-options", OptionList)
        try:
            index = option_list.get_option_index(self._initial_provider)
        except OptionDoesNotExist:
            return
        option_list.highlighted = index
        option_list.scroll_to_highlight()

    def on_click(self, event: Click) -> None:  # noqa: PLR6301 - Textual handler
        """Open style-embedded hyperlinks (the title `Docs` link)."""
        open_style_link(event)

    def on_mouse_move(self, event: MouseMove) -> None:
        """Show a pointer over inline docs links."""
        self.styles.pointer = "pointer" if event.style.link else "default"

    def on_leave(self) -> None:
        """Reset the pointer shape when the mouse leaves the manager."""
        self.styles.pointer = "default"

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the prompt for the selected provider.

        Greyed-out (uninstalled) providers route to an install confirmation
        instead of the key prompt, since their package must be installed before
        a credential is useful.
        """
        provider = event.option.id
        if not provider:
            return
        extra = self._install_extras.get(provider)
        if extra is not None:
            self._prompt_install_provider(provider, extra)
            return
        if provider == CODEX_PROVIDER:
            # ChatGPT auth uses an OAuth browser flow, not an API key. The
            # selector dispatches to a dedicated modal that already knows
            # how to surface the authorize URL inline (so headless / SSH
            # users can still paste it manually) and run the loopback
            # callback wait on a worker.
            self._open_codex_screen()
            return
        if is_service(provider):
            # Services (e.g. Tavily web search) use a plain API key, stored the
            # same way as a model-provider key.
            self.app.push_screen(
                AuthPromptScreen(provider, SERVICE_API_KEY_ENV[provider]),
                self._on_prompt_closed,
            )
            return
        env_var = get_credential_env_var(provider)
        self.app.push_screen(
            AuthPromptScreen(provider, env_var),
            self._on_prompt_closed,
        )

    def _prompt_install_provider(self, provider: str, extra: str) -> None:
        """Confirm installing an uninstalled provider's extra, then dismiss.

        On confirm, record the extra on `pending_install_extra` and dismiss so
        the app can install it (with a server restart) and reopen the manager
        with the provider now installed. On cancel, stay on the manager.

        Args:
            provider: The uninstalled provider the user selected.
            extra: The `deepagents-code` extra that installs `provider`.
        """
        from deepagents_code.tui.widgets.install_confirm import (
            InstallProviderConfirmScreen,
        )

        def _on_confirm(proceed: bool | None) -> None:
            if proceed:
                self.pending_install_extra = extra
                self.pending_install_provider = provider
                self.dismiss(None)
            else:
                # Declined or dismissed: clear any pending request so a reused
                # screen never carries a stale install request, and stay put.
                self.pending_install_extra = None
                self.pending_install_provider = None

        self.app.push_screen(
            InstallProviderConfirmScreen(provider, extra),
            _on_confirm,
        )

    def _open_codex_screen(self) -> None:
        """Push the ChatGPT OAuth flow modal and refresh on close.

        When `openai_codex` is already signed in, give the user a chance to
        sign out before launching a fresh sign-in flow. Otherwise just run
        the sign-in worker.
        """
        from deepagents_code.integrations import openai_codex
        from deepagents_code.tui.widgets.codex_auth import (
            CodexAuthScreen,
            CodexSignedInScreen,
        )

        status = openai_codex.get_status()
        if status.logged_in and not status.is_expired:
            self.app.push_screen(
                CodexSignedInScreen(),
                self._on_codex_signed_in_closed,
            )
            return
        self.app.push_screen(CodexAuthScreen(), self._on_codex_closed)

    def _on_codex_closed(self, _result: bool | None) -> None:
        """Refresh the option list once the codex flow dismisses."""
        clear_caches()
        self._refresh_options()

    def _on_codex_signed_in_closed(self, action: CodexSignedInAction | None) -> None:
        """Handle dismissal of the "already signed in" overlay.

        Args:
            action: `SIGN_OUT` to clear the token, `REAUTH` to run the
                sign-in flow again, `None` to close cleanly.
        """
        from deepagents_code.tui.widgets.codex_auth import CodexSignedInAction

        if action is CodexSignedInAction.SIGN_OUT:
            from deepagents_code.integrations import openai_codex

            removed = openai_codex.logout()
            if removed:
                self.app.notify("Signed out of ChatGPT.", markup=False)
            clear_caches()
            self._refresh_options()
        elif action is CodexSignedInAction.REAUTH:
            from deepagents_code.tui.widgets.codex_auth import CodexAuthScreen

            self.app.push_screen(CodexAuthScreen(), self._on_codex_closed)
        else:
            self._refresh_options()

    def action_cancel(self) -> None:
        """Close the manager."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the option-list cursor down."""
        self.query_one("#auth-manager-options", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option-list cursor up."""
        self.query_one("#auth-manager-options", OptionList).action_cursor_up()

    def _on_prompt_closed(self, result: AuthResult | None) -> None:
        """Refresh the option list once the prompt dismisses."""
        self._refresh_options()
        if result is AuthResult.SAVED:
            self.post_message(self.CredentialSaved())

    def _refresh_options(self) -> None:
        """Rebuild option labels from current store state."""
        option_list = self.query_one("#auth-manager-options", OptionList)
        highlighted = option_list.highlighted
        option_list.clear_options()
        options, _ = self._build_options_with_warning()
        for option in options:
            option_list.add_option(option)
        if highlighted is not None and option_list.option_count:
            option_list.highlighted = min(highlighted, option_list.option_count - 1)

    def _build_options_with_warning(self) -> tuple[list[Option], str | None]:
        """Render the option list, returning a corruption warning if any.

        Returns:
            `(options, warning_message)`. `warning_message` is `None` when
            the credential file is readable; otherwise a one-line hint
            telling the user the file is unreadable so a corrupt store
            doesn't silently look identical to "no keys stored".
        """
        warning: str | None = None
        try:
            stored = set(auth_store.list_configured_providers())
        except RuntimeError as exc:
            logger.warning("Failed to list stored credentials: %s", exc)
            stored = set()
            warning = (
                f"Credential file is unreadable ({exc}). "
                "Saving a key here will overwrite it."
            )

        config = ModelConfig.load()
        config_providers = {
            name for name, cfg in config.providers.items() if cfg.get("api_key_env")
        }

        # Only show well-known providers whose LangChain package is actually
        # installed. `get_available_models` returns providers it could
        # successfully import profiles for, so it doubles as an install
        # gate. Stored and config-defined providers are always shown — even
        # if the package was later uninstalled — so a stale credential can
        # still be cleaned up and an explicitly-declared provider stays
        # visible.
        installed = set(get_available_models().keys())
        well_known_installed = set(PROVIDER_API_KEY_ENV) & installed
        # `openai_codex` is gated on `langchain-openai` being installed (we
        # surface it whenever `openai` was discovered) rather than on
        # `PROVIDER_API_KEY_ENV`, since it has no env var of its own.
        codex_installed = {CODEX_PROVIDER} if "openai" in installed else set()

        shown = well_known_installed | codex_installed | stored | config_providers
        # Surface well-known providers whose package isn't installed yet as
        # greyed-out, install-on-select entries so they stay
        # discoverable (mirrors the model selector). Disabled providers and
        # ones already shown above are skipped.
        self._install_extras = self._uninstalled_known_providers(config, shown)

        # Resolve each manageable entry's auth status once and reuse it for
        # both ordering and badge rendering. `_auth_status_for` reads the
        # credential file, so resolving it separately in the sort key and in
        # `_format_label` would read `auth.json` twice per row (and, on a
        # corrupt store, log the same warning twice). A single pass halves both.
        services = set(SERVICE_API_KEY_ENV) - shown - set(self._install_extras)
        status_by_key = {key: _auth_status_for(key) for key in shown | services}

        # Float entries that already have a credential configured to the top so
        # the keys a user is actively using are easiest to find; everything else
        # keeps alphabetical order (the `key` tiebreaker). Uninstalled
        # install-on-select entries are listed afterwards (alphabetically) since
        # selecting them installs a package rather than managing a key.
        def sort_key(key: str) -> tuple[int, str]:
            configured = status_by_key[key].state is ProviderAuthState.CONFIGURED
            return (0 if configured else 1, key)

        manageable = sorted(status_by_key, key=sort_key)
        extra_providers = sorted(self._install_extras)
        options = [
            Option(self._format_label(key, status=status_by_key[key]), id=key)
            for key in manageable
        ]
        options.extend(
            Option(self._format_label(provider, installed=False), id=provider)
            for provider in extra_providers
        )
        return options, warning

    @staticmethod
    def _uninstalled_known_providers(
        config: ModelConfig, shown: set[str]
    ) -> dict[str, str]:
        """Map known providers missing their package to the installing extra.

        Args:
            config: Loaded model config, used to skip disabled providers.
            shown: Providers already listed (installed/stored/config) to skip.

        Returns:
            `{provider: extra}` for each well-known, enabled provider whose
                integration package is not installed and has a curated extra.
        """
        from deepagents_code.config_manifest import (
            is_provider_package_installed,
            provider_install_extra,
        )

        uninstalled: dict[str, str] = {}
        for provider in PROVIDER_API_KEY_ENV:
            if provider in shown or not config.is_provider_enabled(provider):
                continue
            extra = provider_install_extra(provider)
            if extra is None or is_provider_package_installed(provider):
                continue
            uninstalled[provider] = extra
        return uninstalled

    @staticmethod
    def _format_label(
        provider: str,
        *,
        installed: bool = True,
        status: ProviderAuthStatus | None = None,
    ) -> Content:
        """Build a `Content` label for `provider` showing its credential source.

        Args:
            provider: Provider config key.
            installed: Whether the provider's integration package is installed.
                Uninstalled providers render dimmed with a `[not installed]`
                marker since selecting them prompts an install, not a key.
            status: Precomputed auth status to render. Pass this when the
                caller already resolved it to avoid a duplicate credential-file
                read; resolved on demand when omitted. Ignored for uninstalled
                providers, which render no badge.

        Returns:
            A composed `Content` with the provider label and a status badge.
        """
        name = provider_display_name(provider)
        if not installed:
            return Content.assemble(
                Content.styled(name, "dim"),
                "  ",
                Content.styled("[not installed]", "dim"),
            )
        if status is None:
            status = _auth_status_for(provider)
        badge = format_auth_badge(status)
        return Content.assemble(
            Content.from_markup("$provider", provider=name),
            "  ",
            badge,
        )
