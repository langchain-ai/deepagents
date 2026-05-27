"""UI-agnostic interaction interface for MCP OAuth login.

The OAuth login flow needs to ask the user a few things during the
handshake — open or display the authorize URL, accept a pasted callback
URL when the provider has no loopback redirect, show RFC 8628 device-code
instructions, and report success or failure. The CLI uses `print` and
`input`; a TUI surface needs in-app widgets instead. `OAuthInteraction` is
the small Protocol both implementations satisfy, and `CliOAuthInteraction`
is the existing CLI behavior preserved as one implementation of that
interface.

Important: implementations must never embed access or refresh tokens
in user-facing messages. The interaction surface only ever sees
authorize URLs, callback URLs, device codes, and short status strings,
so leaks come from misuse, not from this interface's shape.
"""

from __future__ import annotations

from typing import Protocol


class OAuthInteraction(Protocol):
    """User-facing OAuth interaction surface shared by CLI and TUI."""

    async def show_authorize_url(self, url: str, *, opened_in_browser: bool) -> None:
        """Tell the user about the authorize URL.

        Args:
            url: Final authorize URL with provider-specific extras applied.
            opened_in_browser: `True` when the caller already launched the
                URL via `webbrowser.open`; `False` when the caller needs the
                user to open it manually.
        """
        ...

    async def request_callback_url(self) -> str:
        """Wait for the user to paste back the full provider callback URL.

        Returns:
            The raw pasted URL (the caller parses `code`/`state`/`error`).

        Raises:
            RuntimeError: When the user interaction cannot complete (for
                example, the input surface is unavailable or was dismissed).
        """
        ...

    async def show_device_code(
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Show RFC 8628 device-code instructions to the user.

        Args:
            verification_uri: Provider URL the user visits in a browser.
            user_code: Short code the user enters on `verification_uri`.
            expires_in: Lifetime of the device code in seconds.
        """
        ...

    async def show_success(self, message: str) -> None:
        """Report a successful login step.

        Implementations must not embed token material in `message`; the
        login code only passes structural facts ("logged in", token file
        path) here.

        Args:
            message: Plain-text status line.
        """
        ...

    async def show_notice(self, message: str) -> None:
        """Report a non-fatal progress notice (e.g. fallback path taken).

        Args:
            message: Plain-text notice.
        """
        ...

    async def show_error(self, message: str) -> None:
        """Report a fatal (flow-ending) error.

        "Fatal" rather than "terminal" because this is a TUI codebase
        where "terminal" reads ambiguously.

        Args:
            message: Plain-text error description.
        """
        ...


class CliOAuthInteraction:
    """Default `OAuthInteraction` that drives the flow via stdin/stdout.

    Preserves the previous `dcode mcp login` behavior — paste-back input,
    plain-text prompts, success messages printed to stdout.
    """

    async def show_authorize_url(  # noqa: PLR6301
        self,
        url: str,
        *,
        opened_in_browser: bool,
    ) -> None:
        """Print the full authorize instruction block to stdout.

        Uses browser-opened wording when `opened_in_browser` is `True`;
        otherwise instructs the user to open the URL and paste back the callback.
        """
        if opened_in_browser:
            print(  # noqa: T201
                "\nOpened your browser to approve MCP access. "
                "If it did not open, visit this URL:\n"
                f"\n  {url}\n",
            )
        else:
            print(  # noqa: T201
                "\nOpen this URL in a browser, approve access, then paste the full "
                "callback URL back here:\n"
                f"\n  {url}\n",
            )

    async def request_callback_url(self) -> str:  # noqa: PLR6301
        """Read a trimmed callback URL from stdin via a worker thread.

        Returns:
            The trimmed callback URL string.

        Raises:
            RuntimeError: If stdin is closed before the user replies.
        """
        import asyncio

        try:
            raw = await asyncio.to_thread(input, "Callback URL: ")
        except EOFError as exc:
            msg = (
                "No callback URL received (stdin closed). "
                "Re-run `dcode mcp login <server>` and paste the URL."
            )
            raise RuntimeError(msg) from exc
        return raw.strip()

    async def show_device_code(  # noqa: PLR6301
        self,
        *,
        verification_uri: str,
        user_code: str,
        expires_in: int,
    ) -> None:
        """Print RFC 8628 device-code instructions to stdout."""
        print(  # noqa: T201
            f"\nVisit {verification_uri} and enter code: "
            f"{user_code}\n(code expires in {expires_in}s)\n",
        )

    async def prompt_slack_team_id(self) -> str | None:  # noqa: PLR6301
        """Ask for a Slack team ID via `input()` on a worker thread.

        Returns:
            The entered Slack team ID, or `None` if the prompt was blank or
                stdin was closed.
        """
        import asyncio

        try:
            raw = await asyncio.to_thread(
                input,
                "Slack team ID to install the app into "
                "(e.g. T01234567 — leave blank to pick on Slack's page): ",
            )
        except EOFError:
            return None
        return raw.strip() or None

    async def show_success(self, message: str) -> None:  # noqa: PLR6301
        """Print success message to stdout."""
        print(message)  # noqa: T201

    async def show_notice(self, message: str) -> None:  # noqa: PLR6301
        """Print progress notice to stdout."""
        print(message)  # noqa: T201

    async def show_error(self, message: str) -> None:  # noqa: PLR6301
        """Print error message to stderr."""
        import sys

        print(message, file=sys.stderr)  # noqa: T201


__all__ = [
    "CliOAuthInteraction",
    "OAuthInteraction",
]
