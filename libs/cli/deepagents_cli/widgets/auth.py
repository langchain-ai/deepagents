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
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from textual.binding import Binding, BindingType
from textual.color import Color as TColor
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.style import Style as TStyle
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.events import Click

from deepagents_cli import auth_store, theme
from deepagents_cli.config import get_glyphs, is_ascii_mode
from deepagents_cli.model_config import (
    PROVIDER_API_KEY_ENV,
    PROVIDERS_DOCS_URL as _PROVIDERS_DOCS_URL,
    ModelConfig,
    ProviderAuthSource,
    clear_caches,
    get_available_models,
    get_credential_env_var,
    get_provider_auth_status,
    resolved_env_var_name,
)
from deepagents_cli.widgets._links import open_style_link

logger = logging.getLogger(__name__)


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


class DeleteCredentialConfirmScreen(ModalScreen[bool]):
    """Confirmation overlay shown before clearing a stored credential.

    Patterned on `DeleteThreadConfirmScreen` so the destructive prompt feels
    consistent across the CLI. Always dismisses with `True` on confirm or
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

    AuthPromptScreen .auth-prompt-meta {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
    }

    AuthPromptScreen #auth-prompt-input {
        margin-bottom: 1;
        border: solid $primary-lighten-2;
    }

    AuthPromptScreen #auth-prompt-input:focus {
        border: solid $primary;
    }

    AuthPromptScreen .auth-prompt-error {
        height: auto;
        color: $error;
        margin-bottom: 1;
    }

    AuthPromptScreen .auth-prompt-help {
        height: 1;
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
    ) -> None:
        """Initialize the prompt for `provider`.

        Args:
            provider: Provider name (e.g., `"anthropic"`).
            env_var: Canonical env var the SDK reads, shown as helper text.
                May be `None` for providers that don't use one of the
                hardcoded env-var bindings (rare; the prompt still works).
            reason: Optional one-line context, e.g.,
                `"Required to use anthropic:claude-opus-4-7"`.
        """
        super().__init__()
        self._provider = provider
        self._env_var = env_var
        self._reason = reason
        # Probe the store, but never let a corrupt `auth.json` crash the
        # screen at construction time — Textual would propagate the
        # exception before the modal mounts. Treat unreadable as
        # "no existing key" and surface a one-line warning at compose time.
        try:
            self._has_existing = auth_store.get_stored_key(provider) is not None
            self._store_warning: str | None = None
        except RuntimeError as exc:
            self._has_existing = False
            self._store_warning = (
                f"Credential file is unreadable ({exc}). Saving here will overwrite it."
            )

    def compose(self) -> ComposeResult:
        """Compose the prompt.

        Yields:
            Widgets that make up the auth prompt modal.
        """
        glyphs = get_glyphs()
        with Vertical():
            # Tag the title with `(stored)` so the user knows a replacement
            # (or the `Ctrl+D delete` affordance shown in the help line) is
            # what's about to happen — both are gated on `_has_existing`.
            if self._has_existing:
                title = Content.from_markup(
                    "API key for [bold]$provider[/bold] [dim](stored)[/dim]",
                    provider=self._provider,
                )
            else:
                title = Content.from_markup(
                    "API key for [bold]$provider[/bold]",
                    provider=self._provider,
                )
            yield Static(title, classes="auth-prompt-title")
            if self._reason:
                yield Static(
                    Content.from_markup("$reason", reason=self._reason),
                    classes="auth-prompt-copy",
                )
            if self._env_var:
                yield Static(
                    f"Equivalent to setting {self._env_var} (or "
                    f"DEEPAGENTS_CLI_{self._env_var}).",
                    classes="auth-prompt-meta",
                )
            if self._store_warning:
                yield Static(
                    Content.from_markup("$msg", msg=self._store_warning),
                    classes="auth-prompt-error",
                )
            yield Input(
                placeholder=(
                    "Paste a new key to replace the stored one"
                    if self._has_existing
                    else "Paste your API key"
                ),
                password=True,
                id="auth-prompt-input",
            )
            yield Static("", classes="auth-prompt-error", id="auth-prompt-error")
            save_label = "Enter replace" if self._has_existing else "Enter save"
            help_parts = [f"{save_label} {glyphs.bullet} Esc cancel"]
            if self._has_existing:
                help_parts.append("Ctrl+D delete stored")
            yield Static(
                f" {glyphs.bullet} ".join(help_parts),
                classes="auth-prompt-help",
            )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Validate, persist, and dismiss."""
        event.stop()
        cleaned = event.value.strip()
        if not cleaned:
            self._show_error("API key cannot be empty.")
            return
        try:
            outcome = auth_store.set_stored_key(self._provider, cleaned)
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
        clear_caches()
        self.dismiss(AuthResult.SAVED)

    def action_cancel(self) -> None:
        """Dismiss without saving."""
        self.dismiss(AuthResult.CANCELLED)

    def action_delete_stored(self) -> None:
        """Open the delete-confirmation overlay for the stored credential."""
        if not self._has_existing:
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
            # CLI window. Surface that fact so "delete" UX doesn't lie when
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
    """

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
        height: 1;
        color: $text-muted;
        text-style: italic;
        text-align: center;
        margin-top: 1;
    }
    """

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
            option_list = OptionList(*options, id="auth-manager-options")
            yield option_list
            yield Static(
                f"{glyphs.arrow_up}/{glyphs.arrow_down} or Tab/Shift+Tab "
                f"navigate {glyphs.bullet} Enter add/replace/delete "
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
            "Lists installed providers and any you've configured in "
            "~/.deepagents/config.toml. Install more via "
            "`pip install deepagents-cli[<provider>]`. ",
            ("Docs", link_style),
        )

    def on_mount(self) -> None:
        """Apply ASCII border when needed."""
        if is_ascii_mode():
            container = self.query_one(Vertical)
            colors = theme.get_theme_colors(self)
            container.styles.border = ("ascii", colors.success)

    def on_click(self, event: Click) -> None:  # noqa: PLR6301 - Textual handler
        """Open style-embedded hyperlinks (the title `Docs` link)."""
        open_style_link(event)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Open the prompt for the selected provider."""
        provider = event.option.id
        if not provider:
            return
        env_var = get_credential_env_var(provider)
        self.app.push_screen(
            AuthPromptScreen(provider, env_var),
            self._on_prompt_closed,
        )

    def action_cancel(self) -> None:
        """Close the manager."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move the option-list cursor down."""
        self.query_one("#auth-manager-options", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move the option-list cursor up."""
        self.query_one("#auth-manager-options", OptionList).action_cursor_up()

    def _on_prompt_closed(self, _result: AuthResult | None) -> None:
        """Refresh the option list once the prompt dismisses."""
        self._refresh_options()

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

        providers = sorted(well_known_installed | stored | config_providers)
        options = [
            Option(self._format_label(provider), id=provider) for provider in providers
        ]
        return options, warning

    @staticmethod
    def _format_label(provider: str) -> Content:
        """Build a `Content` label for `provider` showing its credential source.

        Returns:
            A composed `Content` with the provider name and a status badge.
        """
        status = get_provider_auth_status(provider)
        env_var = status.env_var or get_credential_env_var(provider) or ""
        if status.source is ProviderAuthSource.STORED:
            badge = Content.styled("[stored]", "bold $success")
        elif status.source is ProviderAuthSource.ENV:
            if env_var:
                badge = Content.assemble(
                    ("[env: ", "$text-muted"),
                    Content.styled(resolved_env_var_name(env_var), "$text-muted"),
                    ("]", "$text-muted"),
                )
            else:
                badge = Content.styled("[env]", "$text-muted")
        else:
            badge = Content.styled("[missing]", "bold $warning")
        return Content.assemble(
            Content.from_markup("$provider", provider=provider),
            "  ",
            badge,
        )
